"""Change 3: claim checking.

The answer is split into sentences in code. A separate model call labels each sentence against the
retrieved sections only (it never sees the question). Code then checks the checker: its evidence must
appear word for word in the section it names, and a code-only number guard rejects sentences with
numbers that come from neither the sections, the question, nor a simple percentage calculation.
Checks can only lower a verdict, never raise it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.documents import Section
from app.verify import verify_citation

PASSING = {"supported"}  # partly_supported counts as a failure (decision for change 3)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(*])")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def _numbers(text: str) -> set[float]:
    values = set()
    for raw in _NUMBER.findall(text):
        try:
            values.add(float(raw.replace(",", "")))
        except ValueError:
            pass
    return values


def allowed_numbers(question: str, sections: list[Section]) -> set[float]:
    """Numbers in the sections or the question, plus question amount x section percentage."""
    section_text = "\n".join(s.raw_text for s in sections)
    allowed = _numbers(section_text) | _numbers(question)
    percents = [float(p) for p in _PERCENT.findall(section_text)]
    for amount in _numbers(question):
        for p in percents:
            allowed.add(round(amount * p / 100, 2))
    return allowed


def number_guard(sentence: str, allowed: set[float]) -> list[float]:
    """Numbers in the sentence that are not allowed (empty list = passes)."""
    return sorted(n for n in _numbers(sentence) if n not in allowed)


@dataclass
class SentenceCheck:
    index: int
    sentence: str
    verdict: str          # final verdict after code checks
    model_verdict: str    # what the checker model said
    reason: str = ""
    support: str = ""     # "doc_id#section" of the verified evidence, when the sentence passed

    @property
    def passed(self) -> bool:
        return self.verdict in PASSING


def check_sentences(
    sentences: list[str],
    labels: list[dict],
    question: str,
    sections: list[Section],
    sections_by_id: dict[str, Section],
) -> list[SentenceCheck]:
    by_index = {int(l.get("index", -1)): l for l in labels}
    allowed = allowed_numbers(question, sections)
    results = []
    for i, sentence in enumerate(sentences):
        label = by_index.get(i, {})
        model_verdict = str(label.get("verdict", "not_supported"))
        verdict, reason = model_verdict, ""
        if i not in by_index:
            verdict, reason = "not_supported", "checker returned no label"
        elif model_verdict in PASSING:
            evidence = {"doc_id": label.get("doc_id", ""), "section": label.get("section", ""),
                        "quote": label.get("evidence", "")}
            if verify_citation(evidence, sections_by_id) is None:
                verdict, reason = "not_supported", "checker evidence not found in the named section"
        bad_numbers = number_guard(sentence, allowed)
        if bad_numbers and verdict in PASSING:
            verdict, reason = "not_supported", f"number guard: {bad_numbers} not in sections or question"
        support = ""
        if verdict in PASSING:
            support = f'{str(label.get("doc_id", "")).strip().removesuffix(".md")}#{str(label.get("section", "")).strip()}'
        results.append(SentenceCheck(i, sentence, verdict, model_verdict, reason, support))
    return results
