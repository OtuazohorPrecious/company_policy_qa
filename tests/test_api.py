import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.service import REFUSAL_TEXT, PolicyQA
from tests.conftest import FakeModel

HOTEL_Q = "I stayed in a hotel costing NGN 70,000 per night on a domestic work trip. Is that within the limit?"
GOOD = {"doc_id": "04_business_travel_policy", "section": "4",
        "quote": "Domestic hotels: maximum NGN 80,000 per night."}
BAD = {"doc_id": "04_business_travel_policy", "section": "4",
       "quote": "Hotels may cost up to NGN 80,000."}
GOOD_2 = {"doc_id": "05_expense_reimbursement_policy", "section": "3",
          "quote": "Items up to NGN 50,000: line manager approval."}


def answered(*citations, text="Yes, it is within the NGN 80,000 limit.", not_covered=""):
    return {"answer": text, "citations": list(citations), "not_covered": not_covered}


NOTHING_COVERED = {"answer": "", "citations": [], "not_covered": "The question is not addressed."}


def make_client(index, responses, gate=-1.0, splits=None, checks=None):
    model = FakeModel(responses, splits, checks)
    app = create_app(qa_factory=lambda: PolicyQA(index, model, top_k=5, gate_threshold=gate))
    return TestClient(app, raise_server_exceptions=False), model


def test_health(index):
    client, _ = make_client(index, [])
    with client:
        r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 1001}, {"question": 5}])
def test_invalid_requests_return_422_with_error_body(index, body):
    client, model = make_client(index, [])
    with client:
        r = client.post("/ask", json=body)
    assert r.status_code == 422
    assert set(r.json()) == {"error"}
    assert model.calls == [] and model.split_calls == []


def test_malformed_json_returns_422(index):
    client, _ = make_client(index, [])
    with client:
        r = client.post("/ask", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and "error" in r.json()


def test_answered_response_matches_contract(index):
    client, _ = make_client(index, [answered(GOOD)])
    with client:
        r = client.post("/ask", json={"question": HOTEL_Q})
    body = r.json()
    assert r.status_code == 200
    assert body["question"] == HOTEL_Q
    assert body["answered"] is True
    assert body["citations"] == [{"doc_id": "04_business_travel_policy", "section": "4",
                                  "quote": "Domestic hotels: maximum **NGN 80,000 per night**."}]


def test_question_echoed_exactly(index):
    q = "  How much paternity leave do I get?  "
    client, _ = make_client(index, [NOTHING_COVERED])
    with client:
        assert client.post("/ask", json={"question": q}).json()["question"] == q


def test_llm_refusal_shape(index):
    # Model writes an uncited guess: retried once, still uncited -> refused with the fixed text.
    guess = {"answer": "I think maybe...", "citations": [], "not_covered": ""}
    client, _ = make_client(index, [guess, guess])
    with client:
        body = client.post("/ask", json={"question": "Can I bring my dog to the office?"}).json()
    assert body["answered"] is False and body["citations"] == [] and body["answer"] == REFUSAL_TEXT


def test_gate_refuses_without_calling_model(index):
    client, model = make_client(index, [], gate=1.01)
    with client:
        body = client.post("/ask", json={"question": "Can I bring my dog to the office?"}).json()
    assert body["answered"] is False and body["citations"] == []
    assert model.calls == []


def test_provider_failure_returns_500_not_refusal(index):
    client, _ = make_client(index, [RuntimeError("provider down")])
    with client:
        r = client.post("/ask", json={"question": HOTEL_Q})
    assert r.status_code == 500 and set(r.json()) == {"error"}


def test_all_quotes_invalid_then_fixed_on_retry(index):
    client, model = make_client(index, [answered(BAD), answered(GOOD)])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert body["answered"] is True and len(body["citations"]) == 1
    assert len(model.calls) == 2 and "not found word for word" in model.calls[1]["feedback"]


def test_all_quotes_invalid_twice_refuses(index):
    client, model = make_client(index, [answered(BAD), answered(BAD)])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert body["answered"] is False and body["citations"] == [] and len(model.calls) == 2


def test_partial_quotes_retry_then_keep_verified(index):
    client, model = make_client(index, [answered(GOOD, BAD), answered(BAD)])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert body["answered"] is True
    assert [c["doc_id"] for c in body["citations"]] == ["04_business_travel_policy"]
    assert len(model.calls) == 2


def test_multi_document_citations_kept(index):
    client, _ = make_client(index, [answered(GOOD, GOOD_2)])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q + " Who approves my claim?"}).json()
    assert {c["doc_id"] for c in body["citations"]} == {"04_business_travel_policy", "05_expense_reimbursement_policy"}


def test_retry_error_returns_500(index):
    client, _ = make_client(index, [answered(BAD), RuntimeError("provider down")])
    with client:
        assert client.post("/ask", json={"question": HOTEL_Q}).status_code == 500



# ── Change 1: question splitting ──────────────────────────────────────────
Q11 = ("I stayed in a hotel costing NGN 70,000 per night on a domestic work trip. "
       "Is that within the limit, and who needs to approve my claim?")
Q11_PARTS = ["Is a hotel costing NGN 70,000 per night on a domestic work trip within the limit?",
             "Who needs to approve my expense claim for a NGN 70,000 per night hotel?"]


def test_single_part_question_uses_top_k_5(index):
    client, model = make_client(index, [answered(GOOD)])
    with client:
        client.post("/ask", json={"question": HOTEL_Q})
    assert model.split_calls == [HOTEL_Q]
    assert len(model.calls[0]["sections"]) == 5


def test_split_question_uses_up_to_8_sections_and_original_question(index):
    client, model = make_client(index, [answered(GOOD, GOOD_2)], splits=Q11_PARTS)
    with client:
        body = client.post("/ask", json={"question": Q11}).json()
    assert body["answered"] is True
    assert len(model.calls[0]["sections"]) == 8
    assert model.calls[0]["question"] == Q11  # the model answers the original question
    assert "05_expense_reimbursement_policy#3" in model.calls[0]["sections"]


def test_split_capped_at_three(index):
    # Number-free answer: these dummy sub-questions may not retrieve the hotel section, and the
    # number guard (change 3) would rightly flag "NGN 80,000" if it isn't in the sections given.
    client, model = make_client(index, [answered(GOOD, text="It is within the limit.")],
                                splits=["a hotel?", "b claim?", "c leave?", "d gift?"])
    with client:
        client.post("/ask", json={"question": HOTEL_Q})
    assert len(model.calls) == 1


def test_gate_uses_best_score_across_sub_questions(index):
    original = "zzz qqq xxx?"
    part = "Domestic hotels: maximum NGN 80,000 per night."
    r_orig = index.search(original, 5).top_dense_similarity
    r_all = index.search_split(original, ["unrelated words here", part], 8).top_dense_similarity
    assert r_all > r_orig
    client, model = make_client(index, [answered(GOOD)], gate=(r_orig + r_all) / 2,
                                splits=["unrelated words here", part])
    with client:
        body = client.post("/ask", json={"question": original}).json()
    assert body["answered"] is True and len(model.calls) == 1


def test_splitter_failure_returns_500(index):
    client, _ = make_client(index, [], splits=RuntimeError("provider down"))
    with client:
        assert client.post("/ask", json={"question": HOTEL_Q}).status_code == 500


# ── Change 2: answer + citations + not_covered ────────────────────────────



def test_nothing_covered_refuses_without_retry(index):
    client, model = make_client(index, [NOTHING_COVERED])
    with client:
        body = client.post("/ask", json={"question": "Does the company provide health insurance?"}).json()
    assert body["answered"] is False and body["answer"] == REFUSAL_TEXT and body["citations"] == []
    assert len(model.calls) == 1


def test_gap_is_appended_as_a_sentence(index):
    client, _ = make_client(index, [answered(GOOD, not_covered="who approves the claim.")])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert body["answered"] is True
    assert body["answer"] == ("Yes, it is within the NGN 80,000 limit. "
                              "The policy documents do not cover: who approves the claim.")


def test_no_gap_means_no_extra_sentence(index):
    client, _ = make_client(index, [answered(GOOD)])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert "do not cover" not in body["answer"]


def test_answer_with_only_unverifiable_quotes_is_refused(index):
    client, _ = make_client(index, [answered(BAD, not_covered="x"), answered(BAD)])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert body["answered"] is False and body["answer"] == REFUSAL_TEXT


# ── Change 3: claim checker ──────────────────────────────────────────────
TWO_SENTENCES = "Yes, it is within the NGN 80,000 limit. The Travel Desk approves the claim."


def label(i, verdict, evidence=None):
    ev = evidence or {"doc_id": "", "section": "", "evidence": ""}
    return {"index": i, "verdict": verdict, **ev}


def test_all_sentences_supported_answer_unchanged(index):
    client, model = make_client(index, [answered(GOOD)])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert body["answer"] == "Yes, it is within the NGN 80,000 limit."
    assert len(model.check_calls) == 1 and len(model.calls) == 1


def test_unsupported_sentence_fixed_by_retry(index):
    ok = label(0, "supported", FakeModel.DEFAULT_EVIDENCE)
    client, model = make_client(
        index, [answered(GOOD, text=TWO_SENTENCES), answered(GOOD)],
        checks=[[ok, label(1, "not_supported")], "supported"])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert body["answer"] == "Yes, it is within the NGN 80,000 limit."
    assert len(model.calls) == 2 and "The Travel Desk approves the claim." in model.calls[1]["feedback"]


def test_still_unsupported_after_retry_is_dropped(index):
    ok = label(0, "supported", FakeModel.DEFAULT_EVIDENCE)
    client, model = make_client(
        index, [answered(GOOD, text=TWO_SENTENCES), answered(GOOD, text=TWO_SENTENCES)],
        checks=[[ok, label(1, "not_supported")], [ok, label(1, "contradicted")]])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert body["answered"] is True
    assert body["answer"] == "Yes, it is within the NGN 80,000 limit."


def test_partly_supported_counts_as_failure(index):
    ok = label(0, "supported", FakeModel.DEFAULT_EVIDENCE)
    client, model = make_client(
        index, [answered(GOOD, text=TWO_SENTENCES), answered(GOOD, text=TWO_SENTENCES)],
        checks=[[ok, label(1, "partly_supported")], [ok, label(1, "partly_supported")]])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert "Travel Desk" not in body["answer"]


def test_checker_evidence_must_exist(index):
    fake_evidence = {"doc_id": "04_business_travel_policy", "section": "4", "evidence": "The Travel Desk approves claims."}
    ok = label(0, "supported", FakeModel.DEFAULT_EVIDENCE)
    client, _ = make_client(
        index, [answered(GOOD, text=TWO_SENTENCES), answered(GOOD, text=TWO_SENTENCES)],
        checks=[[ok, label(1, "supported", fake_evidence)], [ok, label(1, "supported", fake_evidence)]])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert "Travel Desk" not in body["answer"]


def test_all_sentences_dropped_means_refusal(index):
    client, _ = make_client(index, [answered(GOOD), answered(GOOD)], checks=["not_supported", "not_supported"])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert body["answered"] is False and body["answer"] == REFUSAL_TEXT and body["citations"] == []


def test_gap_sentence_appended_after_check(index):
    client, model = make_client(index, [answered(GOOD, not_covered="who approves the claim")])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert body["answer"].endswith("The policy documents do not cover: who approves the claim.")
    assert model.check_calls[0] == ["Yes, it is within the NGN 80,000 limit."]  # gap sentence not checked


def test_checker_failure_returns_500(index):
    client, _ = make_client(index, [answered(GOOD)], checks=[RuntimeError("provider down")])
    with client:
        assert client.post("/ask", json={"question": HOTEL_Q}).status_code == 500


# ── Final fixes: plain-text answers, citation trimming, time budget ─────────
from app.service import Budget, BudgetExceeded  # noqa: E402
from tests.conftest import FakeClock  # noqa: E402


def test_markdown_removed_from_answer_text(index):
    client, _ = make_client(index, [answered(GOOD, text="Yes, the limit is **NGN 80,000** per night.")])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert body["answer"] == "Yes, the limit is NGN 80,000 per night."


def test_citations_of_dropped_sentences_are_removed(index):
    ok = label(0, "supported", FakeModel.DEFAULT_EVIDENCE)  # evidence from 04#4
    text = "Yes, it is within the NGN 80,000 limit. Line manager and Finance must approve it."
    resp = answered(GOOD, GOOD_2, text=text)
    client, _ = make_client(index, [resp, resp], checks=[[ok, label(1, "not_supported")],
                                                         [ok, label(1, "not_supported")]])
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert body["answer"] == "Yes, it is within the NGN 80,000 limit."
    assert [c["doc_id"] for c in body["citations"]] == ["04_business_travel_policy"]


def make_budget_client(index, responses, clock, seconds_per_call, budget=45.0, checks=None):
    model = FakeModel(responses, checks=checks, clock=clock, seconds_per_call=seconds_per_call)
    app = create_app(qa_factory=lambda: PolicyQA(index, model, top_k=5, gate_threshold=-1.0,
                                                 budget_seconds=budget, call_timeout_seconds=20.0, clock=clock))
    return TestClient(app, raise_server_exceptions=False), model


def test_call_timeouts_never_exceed_half_the_time_left(index):
    clock = FakeClock()
    client, model = make_budget_client(index, [answered(GOOD)], clock, seconds_per_call=1.0, budget=30.0)
    with client:
        assert client.post("/ask", json={"question": HOTEL_Q}).status_code == 200
    assert model.timeouts and all(t is not None and t <= 15.0 for t in model.timeouts)


def test_no_time_for_quote_retry_and_nothing_verified_returns_500(index):
    clock = FakeClock()
    # split (1 call) + answer (1 call) at 18 s each = 36 s used: only 9 s left, below the 10 s for a retry.
    client, model = make_budget_client(index, [answered(BAD)], clock, seconds_per_call=18.0)
    with client:
        r = client.post("/ask", json={"question": HOTEL_Q})
    assert r.status_code == 500 and set(r.json()) == {"error"}
    assert len(model.calls) == 1  # the retry was skipped


def test_no_time_for_claim_check_returns_verified_answer(index):
    clock = FakeClock()
    client, model = make_budget_client(index, [answered(GOOD)], clock, seconds_per_call=18.0)
    with client:
        body = client.post("/ask", json={"question": HOTEL_Q}).json()
    assert body["answered"] is True and body["citations"]
    assert model.check_calls == []  # check skipped to stay inside the budget


def test_required_call_with_almost_no_time_returns_500(index):
    clock = FakeClock()
    client, _ = make_budget_client(index, [answered(GOOD)], clock, seconds_per_call=44.0)
    with client:
        r = client.post("/ask", json={"question": HOTEL_Q})
    assert r.status_code == 500


def test_budget_math():
    clock = FakeClock()
    b = Budget(45.0, 20.0, clock)
    assert b.call_timeout() == 20.0
    clock.t = 30.0                                  # 15 s left: optional steps still allowed
    assert b.call_timeout() == 7.5 and b.can_afford_optional()
    clock.t = 36.0                                  # 9 s left: optional steps skipped
    assert b.call_timeout() == 4.5 and not b.can_afford_optional()
    clock.t = 44.0
    try:
        b.call_timeout()
        raise AssertionError("expected BudgetExceeded")
    except BudgetExceeded:
        pass
