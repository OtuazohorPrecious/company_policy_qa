import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "inspect_retrieval", Path(__file__).resolve().parent.parent / "scripts" / "inspect_retrieval.py")
inspect_retrieval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspect_retrieval)


def test_rank_table_covers_every_section(index):
    rows = inspect_retrieval.rank_table(index, "What is the maximum domestic hotel cost per night?")
    assert len(rows) == 48
    fused = [r for r in rows if r["fused_rank"] is not None]
    assert 1 <= len(fused) <= 20                     # only top-10 candidates of each search can be fused
    assert rows[0]["fused_rank"] == 1
    assert all(r["dense_rank"] is not None for r in rows)   # dense ranks every section


def test_rank_table_matches_pipeline_retrieval(index):
    q = "What is the maximum domestic hotel cost per night?"
    rows = inspect_retrieval.rank_table(index, q)
    top5 = [r["chunk_id"] for r in rows[:5]]
    assert top5 == [s.chunk_id for s in index.search(q, 5).sections]
