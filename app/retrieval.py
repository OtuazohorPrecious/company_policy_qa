"""Hybrid retrieval: Chroma (dense, cosine) + BM25 (keywords), merged with Reciprocal Rank Fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb
from rank_bm25 import BM25Okapi

from app.documents import Section

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    def embed(self, texts: list[str], timeout: float | None = None) -> list[list[float]]: ...


@dataclass(frozen=True)
class RetrievalResult:
    sections: list[Section]          # top-k after fusion, best first
    top_dense_similarity: float      # highest cosine similarity from the dense search (used by the gate)
    dense_ranking: list[tuple[str, float]]  # (chunk_id, cosine similarity), for debugging/calibration
    bm25_ranking: list[tuple[str, float]]   # (chunk_id, BM25 score)


class HybridIndex:
    def __init__(
        self,
        sections: list[Section],
        embedder: Embedder,
        chroma_path: Path,
        collection_name: str,
        candidates_per_retriever: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self.sections = sections
        self.by_id = {s.chunk_id: s for s in sections}
        self.embedder = embedder
        self.candidates = candidates_per_retriever
        self.rrf_k = rrf_k

        # Dense index: rebuilt from scratch on every startup so it can never be stale.
        self._client = chromadb.PersistentClient(path=str(chroma_path))
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass  # collection did not exist yet
        self._collection = self._client.create_collection(
            name=collection_name, configuration={"hnsw": {"space": "cosine"}}
        )
        texts = [s.index_text for s in sections]
        self._collection.add(
            ids=[s.chunk_id for s in sections],
            embeddings=embedder.embed(texts),
            documents=texts,
            metadatas=[{"doc_id": s.doc_id, "section": s.section} for s in sections],
        )

        # Keyword index.
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])
        self._bm25_ids = [s.chunk_id for s in sections]

    def dense_search(self, question: str, n: int) -> list[tuple[str, float]]:
        [query_vector] = self.embedder.embed([question])
        return self._dense_from_vector(query_vector, n)

    def _dense_from_vector(self, query_vector: list[float], n: int) -> list[tuple[str, float]]:
        result = self._collection.query(
            query_embeddings=[query_vector], n_results=min(n, len(self.sections))
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        # Chroma's cosine distance = 1 - cosine similarity.
        return [(cid, 1.0 - d) for cid, d in zip(ids, distances)]

    def bm25_search(self, question: str, n: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(question))
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, float(score)) for cid, score in ranked[:n] if score > 0]

    def _fused_ranking(self, query: str, vector: list[float]):
        """Dense + BM25 for one query, merged with Reciprocal Rank Fusion. Returns (ids, dense, bm25)."""
        dense = self._dense_from_vector(vector, self.candidates)
        keyword = self.bm25_search(query, self.candidates)
        fused: dict[str, float] = {}
        for ranking in (dense, keyword):
            for rank, (cid, _score) in enumerate(ranking, start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(fused, key=fused.get, reverse=True), dense, keyword

    def search(self, question: str, top_k: int, timeout: float | None = None) -> RetrievalResult:
        """Single-part question: hybrid retrieval, top-k after fusion (unchanged from the baseline)."""
        [vector] = self.embedder.embed([question], timeout=timeout)
        ids, dense, keyword = self._fused_ranking(question, vector)
        return RetrievalResult(
            sections=[self.by_id[cid] for cid in ids[:top_k]],
            top_dense_similarity=dense[0][1] if dense else 0.0,
            dense_ranking=dense,
            bm25_ranking=keyword,
        )

    def search_split(
        self, question: str, sub_questions: list[str], top_k: int, timeout: float | None = None
    ) -> RetrievalResult:
        """Multi-part question (change 1).

        Retrieval: each sub-question gets its own fused ranking; slots are filled by taking turns
        (#1 of each sub-question, then #2 of each, ...) until top_k unique sections are chosen, so a
        section that matters for only one part is not outvoted by the dominant topic.
        Gate score: best cosine similarity across the original question and every sub-question.
        """
        vectors = self.embedder.embed([question, *sub_questions], timeout=timeout)  # one embeddings call
        original_dense = self._dense_from_vector(vectors[0], 1)
        best = original_dense[0][1] if original_dense else 0.0
        per_part: list[list[str]] = []
        dense_all: list[tuple[str, float]] = []
        bm25_all: list[tuple[str, float]] = []
        for sub, vector in zip(sub_questions, vectors[1:]):
            ids, dense, keyword = self._fused_ranking(sub, vector)
            per_part.append(ids)
            dense_all.extend(dense)
            bm25_all.extend(keyword)
            if dense:
                best = max(best, dense[0][1])
        chosen: list[str] = []
        depth = 0
        while len(chosen) < top_k and any(depth < len(ids) for ids in per_part):
            for ids in per_part:
                if depth < len(ids) and ids[depth] not in chosen and len(chosen) < top_k:
                    chosen.append(ids[depth])
            depth += 1
        return RetrievalResult(
            sections=[self.by_id[cid] for cid in chosen],
            top_dense_similarity=best,
            dense_ranking=dense_all,
            bm25_ranking=bm25_all,
        )
