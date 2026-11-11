"""Show how retrieval ranks the policy sections for a query (diagnostic only; changes nothing).

For each query it prints, per section:
  dense rank / cosine   meaning-based search over all 48 sections
  bm25 rank / score     keyword search (sections with no shared word have no BM25 rank)
  fused rank            the real pipeline's Reciprocal Rank Fusion over the top-10 candidates
                        of each search ("-" = not in either top-10, so it can never be retrieved)

Usage (from the project root, with OPENAI_API_KEY set):
    python scripts/inspect_retrieval.py "query one" ["query two" ...] [--target 05_expense_reimbursement_policy#3] [--top 12]
    python scripts/inspect_retrieval.py "a multi-part question" --split     # also runs the splitter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import HybridIndex  # noqa: E402


def rank_table(index: HybridIndex, query: str) -> list[dict]:
    """One row per section with its dense, BM25 and fused ranks for this query."""
    [vector] = index.embedder.embed([query])
    n = len(index.sections)
    dense = index._dense_from_vector(vector, n)
    bm25 = index.bm25_search(query, n)
    fused_ids, _, _ = index._fused_ranking(query, vector)
    dense_rank = {cid: (i, score) for i, (cid, score) in enumerate(dense, start=1)}
    bm25_rank = {cid: (i, score) for i, (cid, score) in enumerate(bm25, start=1)}
    fused_rank = {cid: i for i, cid in enumerate(fused_ids, start=1)}
    rows = []
    for s in index.sections:
        d = dense_rank.get(s.chunk_id)
        b = bm25_rank.get(s.chunk_id)
        rows.append({
            "chunk_id": s.chunk_id,
            "heading": s.heading,
            "dense_rank": d[0] if d else None,
            "cosine": d[1] if d else None,
            "bm25_rank": b[0] if b else None,
            "bm25": b[1] if b else None,
            "fused_rank": fused_rank.get(s.chunk_id),
        })
    rows.sort(key=lambda r: (r["fused_rank"] or 999, r["dense_rank"] or 999))
    return rows


def _fmt(value, spec):
    return "-" if value is None else format(value, spec)


def print_table(query: str, rows: list[dict], top: int, target: str | None) -> None:
    print(f"\nQUERY: {query}")
    print(f"{'fused':>5} {'dense':>5} {'cosine':>7} {'bm25':>5} {'score':>6}  section")
    shown = rows[:top]
    if target and all(r["chunk_id"] != target for r in shown):
        shown = shown + [r for r in rows if r["chunk_id"] == target]
    for r in shown:
        mark = "  <== target" if r["chunk_id"] == target else ""
        print(f"{_fmt(r['fused_rank'], 'd'):>5} {_fmt(r['dense_rank'], 'd'):>5} {_fmt(r['cosine'], '.3f'):>7} "
              f"{_fmt(r['bm25_rank'], 'd'):>5} {_fmt(r['bm25'], '.2f'):>6}  {r['chunk_id']} ({r['heading']}){mark}")


def main() -> None:
    from app.config import load_settings
    from app.documents import load_sections
    from app.llm import OpenAIClient

    parser = argparse.ArgumentParser()
    parser.add_argument("queries", nargs="+")
    parser.add_argument("--target", help="chunk id to always show, e.g. 05_expense_reimbursement_policy#3")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--split", action="store_true", help="also inspect the splitter's sub-questions")
    args = parser.parse_args()

    settings = load_settings()
    if not settings.openai_api_key:
        sys.exit("OPENAI_API_KEY is not set.")
    client = OpenAIClient(settings.openai_api_key, settings.chat_model, settings.embed_model,
                          settings.llm_timeout_seconds)
    index = HybridIndex(load_sections(settings.policies_dir), client, settings.chroma_path,
                        settings.collection_name + "_inspect", settings.candidates_per_retriever, settings.rrf_k)

    for query in args.queries:
        print_table(query, rank_table(index, query), args.top, args.target)
        if args.split:
            parts = client.split_question(query, settings.max_sub_questions)
            print(f"\n  splitter returned {len(parts)} sub-question(s): {parts}")
            if len(parts) > 1:
                for part in parts:
                    print_table(part, rank_table(index, part), args.top, args.target)


if __name__ == "__main__":
    main()
