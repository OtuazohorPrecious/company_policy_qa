"""The question-answering pipeline.

split question -> retrieve -> score gate -> answer (answer + citations + not_covered)
-> verify quotes (retry once) -> check each sentence (retry once, then drop) -> response.

Every request runs under a time budget (REQUEST_BUDGET_SECONDS, default 45 s) so it always returns
within the contract's 60 s: each model call's timeout is taken from the time left, optional steps
(retries, claim checks) are skipped when time is short, and if nothing verified exists when time runs
out the request fails with HTTP 500 rather than a fake refusal.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.checker import SentenceCheck, check_sentences, split_sentences
from app.llm import AnswerModel
from app.retrieval import HybridIndex
from app.verify import Citation, split_citations

logger = logging.getLogger("policy_qa")

REFUSAL_TEXT = "The policy documents do not contain information about this."


@dataclass
class AskResult:
    answered: bool
    answer: str
    citations: list[Citation]
    # Diagnostics (not part of the API response; logged for evaluation).
    trace: dict = field(default_factory=dict)


class BudgetExceeded(RuntimeError):
    """The request ran out of time before anything verified could be returned (-> HTTP 500)."""


class Budget:
    """Wall-clock budget for one request."""

    MIN_FOR_REQUIRED_CALL = 2.0    # seconds; below this a required call is not attempted
    MIN_FOR_OPTIONAL_STEP = 10.0   # seconds; retries/checks are skipped below this

    def __init__(self, total_seconds: float, per_call_seconds: float, clock=time.monotonic):
        self.clock = clock
        self.deadline = clock() + total_seconds
        self.per_call = per_call_seconds

    def left(self) -> float:
        return self.deadline - self.clock()

    def call_timeout(self) -> float:
        """Timeout for a required call. The SDK retries once, so two attempts must fit in the time left."""
        left = self.left()
        if left < self.MIN_FOR_REQUIRED_CALL:
            raise BudgetExceeded("time budget exhausted")
        return max(1.0, min(self.per_call, left / 2))

    def can_afford_optional(self) -> bool:
        return self.left() >= self.MIN_FOR_OPTIONAL_STEP


def _refusal(trace: dict) -> AskResult:
    return AskResult(answered=False, answer=REFUSAL_TEXT, citations=[], trace=trace)


def _feedback(invalid: list[dict]) -> str:
    listed = "; ".join(f'{c.get("doc_id")}#{c.get("section")}: "{c.get("quote")}"' for c in invalid)
    return (
        "These citation quotes were not found word for word in the cited section: "
        f"{listed}. Copy each quote exactly from the section with the matching doc_id and section "
        "number, or remove the citation and any fact that depends on it."
    )


def _says_nothing_covered(response: dict) -> bool:
    """The model addressed none of the question: empty answer and no citations."""
    return not response.get("answer", "").strip() and not response.get("citations")


def _plain(text: str) -> str:
    """The contract asks for plain language: remove Markdown bold markers from the answer text."""
    return " ".join(text.replace("**", "").split())


def _answer(response: dict, citations: list[Citation], trace: dict, text: str | None = None) -> AskResult:
    """Answer text plus, if the model listed gaps, one appended sentence naming them (change 2)."""
    text = _plain(response.get("answer", "") if text is None else text)
    gap = _plain(response.get("not_covered", "")).rstrip(".")
    if gap:
        trace["not_covered"] = gap
        text = f"{text} The policy documents do not cover: {gap}."
    return AskResult(True, text, citations, trace)


def _trim_citations(citations: list[Citation], kept: list[SentenceCheck]) -> list[Citation]:
    """After sentences are dropped, keep only citations from sections that support a kept sentence.

    The contract says to cite only text the answer relies on. If nothing matches, the original list is
    kept, because an answered response needs at least one citation.
    """
    supported = {c.support for c in kept if c.support}
    trimmed = [c for c in citations if f"{c.doc_id}#{c.section}" in supported]
    return trimmed or citations


class PolicyQA:
    def __init__(
        self,
        index: HybridIndex,
        model: AnswerModel,
        top_k: int,
        gate_threshold: float,
        top_k_split: int = 8,
        max_sub_questions: int = 3,
        budget_seconds: float = 45.0,
        call_timeout_seconds: float = 20.0,
        clock=time.monotonic,
    ):
        self.index = index
        self.model = model
        self.top_k = top_k
        self.gate_threshold = gate_threshold
        self.top_k_split = top_k_split
        self.max_sub_questions = max_sub_questions
        self.budget_seconds = budget_seconds
        self.call_timeout_seconds = call_timeout_seconds
        self.clock = clock

    def ask(self, question: str) -> AskResult:
        budget = Budget(self.budget_seconds, self.call_timeout_seconds, self.clock)

        # Change 1: multi-part questions are searched part by part.
        sub_questions = self.model.split_question(question, self.max_sub_questions, timeout=budget.call_timeout())
        is_split = len(sub_questions) > 1
        if is_split:
            retrieval = self.index.search_split(question, sub_questions, self.top_k_split,
                                                timeout=budget.call_timeout())
        else:
            retrieval = self.index.search(question, self.top_k, timeout=budget.call_timeout())
        trace: dict = {
            "top_dense_similarity": round(retrieval.top_dense_similarity, 4),
            "retrieved": [s.chunk_id for s in retrieval.sections],
        }
        if is_split:
            trace["sub_questions"] = sub_questions

        # Layer 1: score gate.
        if retrieval.top_dense_similarity < self.gate_threshold:
            trace["refused_by"] = "gate"
            return _refusal(trace)

        # Layer 2: the model answers what the sections cover and lists gaps in "not_covered" (change 2).
        # Provider errors propagate -> HTTP 500 (never a fake refusal).
        first = self.model.answer(question, retrieval.sections, timeout=budget.call_timeout())
        if _says_nothing_covered(first):
            trace["refused_by"] = "llm_nothing_covered"
            trace["not_covered"] = first.get("not_covered", "")
            return _refusal(trace)

        verified, invalid = split_citations(first.get("citations", []), self.index.by_id)
        trace["first_attempt"] = {"verified": len(verified), "invalid": len(invalid)}
        if verified and not invalid:
            return self._checked(question, retrieval.sections, first, verified, trace, budget)

        # Some or all quotes failed verification, or the answer had no citations: retry once.
        if budget.can_afford_optional():
            retry_feedback = _feedback(invalid) if invalid else "Every fact in the answer needs a citation."
            second = self.model.answer(question, retrieval.sections, feedback=retry_feedback,
                                       timeout=budget.call_timeout())
            verified2, invalid2 = split_citations(second.get("citations", []), self.index.by_id)
            trace["retry"] = {"verified": len(verified2), "invalid": len(invalid2)}
            if verified2:
                return self._checked(question, retrieval.sections, second, verified2, trace, budget)
        else:
            trace["retry"] = "skipped_time_budget"

        # Fall back to the first attempt's verified citations.
        if verified:
            trace["used"] = "first_attempt_verified_only"
            return self._checked(question, retrieval.sections, first, verified, trace, budget)

        if trace["retry"] == "skipped_time_budget":
            raise BudgetExceeded("time budget exhausted before any citation could be verified")

        # answered=false only when no citation verifies (change 2 rule).
        trace["refused_by"] = "no_verified_citation"
        return _refusal(trace)

    # ── Change 3: claim checking ─────────────────────────────────────────
    def _check(self, question, sections, response, budget: Budget) -> list[SentenceCheck]:
        sentences = split_sentences(_plain(response.get("answer", "")))
        labels = (self.model.check_answer(sentences, sections, timeout=budget.call_timeout())
                  if sentences else [])
        return check_sentences(sentences, labels, question, sections, self.index.by_id)

    def _checked(self, question, sections, response, citations, trace, budget: Budget) -> AskResult:
        """Check every sentence; if any fails, retry the answer once, then drop sentences that still fail."""
        if not budget.can_afford_optional():
            trace["claim_check"] = "skipped_time_budget"
            return _answer(response, citations, trace)

        checks = self._check(question, sections, response, budget)
        failed = [c for c in checks if not c.passed]
        trace["claim_check"] = {"sentences": len(checks), "failed": len(failed)}
        if not failed:
            return _answer(response, citations, trace)

        trace["claim_check"]["failures"] = [
            {"sentence": c.sentence, "verdict": c.verdict, "reason": c.reason} for c in failed
        ]
        used_retry = False
        if budget.can_afford_optional():
            listed = " | ".join(c.sentence for c in failed)
            feedback = (
                "These sentences of your answer are not supported by the policy sections: "
                f"{listed}. Remove them, or rewrite them so the sections support them with a citation. "
                'Put anything the sections do not address in "not_covered".'
            )
            second = self.model.answer(question, sections, feedback=feedback, timeout=budget.call_timeout())
            verified2, _ = split_citations(second.get("citations", []), self.index.by_id)
            if verified2 and budget.can_afford_optional():
                response, citations, used_retry = second, verified2, True
                checks = self._check(question, sections, response, budget)

        kept = [c for c in checks if c.passed]
        dropped = [c.sentence for c in checks if not c.passed]
        trace["claim_check_retry"] = {"used_retry": used_retry, "dropped": dropped}
        if not kept:
            trace["refused_by"] = "claim_check"
            return _refusal(trace)
        if dropped:
            citations = _trim_citations(citations, kept)
        return _answer(response, citations, trace, text=" ".join(c.sentence for c in kept))
