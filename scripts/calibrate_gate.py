"""Calibrate the refusal score gate on the dev questions (NOT the 15 test questions).

For each dev question it prints the top dense cosine similarity and the best-matching section, then
reports the gap between the answerable and unanswerable groups and suggests a threshold.

Usage (from the project root, with OPENAI_API_KEY set):
    python scripts/calibrate_gate.py
Writes eval/gate_calibration.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT, load_settings  # noqa: E402
from app.documents import load_sections  # noqa: E402
from app.llm import OpenAIClient  # noqa: E402
from app.retrieval import HybridIndex  # noqa: E402


def main() -> None:
    settings = load_settings()
    if not settings.openai_api_key:
        sys.exit("OPENAI_API_KEY is not set.")
    client = OpenAIClient(settings.openai_api_key, settings.chat_model, settings.embed_model, settings.llm_timeout_seconds)
    index = HybridIndex(
        load_sections(settings.policies_dir), client, settings.chroma_path,
        settings.collection_name + "_calibration", settings.candidates_per_retriever, settings.rrf_k,
    )

    rows = []
    with open(PROJECT_ROOT / "eval" / "dev_questions.csv", newline="", encoding="utf-8") as f:
        for q in csv.DictReader(f):
            dense = index.dense_search(q["question"], 1)
            top_id, top_sim = dense[0]
            rows.append({**q, "top_dense_similarity": round(top_sim, 4), "top_section": top_id})

    print(f"{'id':<4} {'expected':<8} {'sim':>6}  top section  |  question")
    for r in sorted(rows, key=lambda r: r["top_dense_similarity"], reverse=True):
        print(f"{r['id']:<4} {r['expected']:<8} {r['top_dense_similarity']:>6.3f}  {r['top_section']}  |  {r['question']}")

    answerable = [r["top_dense_similarity"] for r in rows if r["expected"] == "answer"]
    unanswerable = [r["top_dense_similarity"] for r in rows if r["expected"] == "refuse"]
    lowest_answerable, highest_unanswerable = min(answerable), max(unanswerable)
    print(f"\nLowest answerable similarity:    {lowest_answerable:.3f}")
    print(f"Highest unanswerable similarity: {highest_unanswerable:.3f}")
    if lowest_answerable > highest_unanswerable:
        midpoint = (lowest_answerable + highest_unanswerable) / 2
        print(f"The groups separate. Midpoint threshold: {midpoint:.3f}")
    else:
        print("The groups OVERLAP: a threshold cannot separate them. Consider a low, conservative value")
        print("that only blocks clearly off-topic questions, and let the LLM decide the rest.")

    out = PROJECT_ROOT / "eval" / "gate_calibration.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
