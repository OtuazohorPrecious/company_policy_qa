def test_hybrid_search_finds_relevant_section(index):
    result = index.search("What is the maximum domestic hotel cost per night?", top_k=5)
    assert len(result.sections) == 5
    assert "04_business_travel_policy#4" in [s.chunk_id for s in result.sections]
    assert -1.0 <= result.top_dense_similarity <= 1.0


def test_bm25_catches_exact_terms(index):
    ids = [cid for cid, _ in index.bm25_search("Spendly", 10)]
    assert ids and ids[0] == "05_expense_reimbursement_policy#2"


def test_index_rebuild_does_not_duplicate(sections, tmp_path):
    from app.retrieval import HybridIndex
    from tests.conftest import FakeEmbedder

    path = tmp_path / "chroma"
    HybridIndex(sections, FakeEmbedder(), path, "rebuild_test")
    second = HybridIndex(sections, FakeEmbedder(), path, "rebuild_test")
    assert second._collection.count() == 48

