"""Offline test fixtures: a deterministic fake embedder and a scriptable fake answer model."""

from __future__ import annotations

import hashlib
import math

import pytest

from app.config import PROJECT_ROOT
from app.documents import load_sections
from app.retrieval import HybridIndex, tokenize

POLICIES = PROJECT_ROOT / "docs" / "policies"


class FakeEmbedder:
    """Bag-of-words hashed into 256 dims and L2-normalised. No network needed."""

    dims = 256

    def embed(self, texts: list[str], timeout=None) -> list[list[float]]:
        vectors = []
        for text in texts:
            v = [0.0] * self.dims
            for token in tokenize(text):
                v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dims] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


class FakeModel:
    """Returns pre-scripted responses in order; can also raise."""

    # Evidence that really is in 04_business_travel_policy section 4 (used by the default checker).
    DEFAULT_EVIDENCE = {"doc_id": "04_business_travel_policy", "section": "4",
                        "evidence": "Domestic hotels: maximum NGN 80,000 per night."}

    def __init__(self, responses: list, splits=None, checks=None, clock=None, seconds_per_call=0.0):
        self.clock = clock                      # optional FakeClock, advanced on every model call
        self.seconds_per_call = seconds_per_call
        self.timeouts: list[float | None] = []  # timeout passed to each model call
        self.responses = list(responses)
        self.checks = list(checks) if checks else []  # each item: list of label dicts, or a verdict string for all
        self.check_calls: list[list[str]] = []
        self.splits = splits  # None -> question unchanged; list[str] -> returned as sub-questions; Exception -> raised
        self.calls: list[dict] = []
        self.split_calls: list[str] = []

    def _tick(self, timeout):
        self.timeouts.append(timeout)
        if self.clock is not None:
            self.clock.t += self.seconds_per_call

    def check_answer(self, sentences, sections, timeout=None):
        self._tick(timeout)
        self.check_calls.append(list(sentences))
        item = self.checks.pop(0) if self.checks else "supported"
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            ev = self.DEFAULT_EVIDENCE if item == "supported" else {"doc_id": "", "section": "", "evidence": ""}
            return [{"index": i, "verdict": item, **ev} for i in range(len(sentences))]
        return item

    def split_question(self, question, max_parts, timeout=None):
        self._tick(timeout)
        self.split_calls.append(question)
        if isinstance(self.splits, Exception):
            raise self.splits
        return (self.splits or [question])[:max_parts]

    def answer(self, question, sections, feedback=None, timeout=None):
        self._tick(timeout)
        self.calls.append({"question": question, "feedback": feedback, "sections": [s.chunk_id for s in sections]})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    """Controllable replacement for time.monotonic."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture(scope="session")
def sections():
    return load_sections(POLICIES)


@pytest.fixture()
def index(sections, tmp_path):
    return HybridIndex(sections, FakeEmbedder(), tmp_path / "chroma", "test_sections")
