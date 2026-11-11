"""Run the 15 test questions against the running service and write results.csv.

The `outcome` and `notes` columns are left empty on purpose: they are filled in by hand
(the contract asks for your own judgement). Full JSON responses are saved to
eval/responses.jsonl to help with judging.

Usage (service must be running):
    python scripts/evaluate.py [--url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["id", "question", "answered", "answer", "citations", "outcome", "notes"]


def post_ask(base_url: str, question: str) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base_url}/ask",
        data=json.dumps({"question": question}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=f"http://localhost:{os.getenv('PORT', '8000')}")
    parser.add_argument("--questions", default=str(ROOT / "docs" / "test_questions.csv"))
    parser.add_argument("--out", default=str(ROOT / "results.csv"))
    args = parser.parse_args()

    with open(args.questions, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    rows, raw = [], []
    for q in questions:
        status, body = post_ask(args.url, q["question"])
        raw.append({"id": q["id"], "status": status, "response": body})
        if status != 200:
            print(f"[{q['id']}] HTTP {status}: {body}")
            rows.append({"id": q["id"], "question": q["question"], "answered": "", "answer": "",
                         "citations": "", "outcome": "", "notes": f"HTTP {status}: {body.get('error', '')}"})
            continue
        citations = ";".join(f"{c['doc_id']}#{c['section']}" for c in body["citations"])
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "answered": str(body["answered"]).lower(),
            "answer": body["answer"],
            "citations": citations,
            "outcome": "",
            "notes": "",
        })
        print(f"[{q['id']}] answered={body['answered']} citations={citations or '-'}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    responses_path = ROOT / "eval" / "responses.jsonl"
    with open(responses_path, "w", encoding="utf-8") as f:
        for item in raw:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nWrote {args.out} and {responses_path}. Now fill in `outcome` and `notes` by hand.")
    if any(r["notes"].startswith("HTTP") for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
