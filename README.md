# Company Policy Q&A Assistant (RAG with citations)

A small REST service that answers employee questions using **only** the 10 Kestrel Nova policy documents in `docs/policies/`. Every answer cites the document, section and an exact quote. If the documents do not contain the answer, the service says so instead of guessing. The API follows `docs/API_CONTRACT.md`.

## Setup and running

Requirements: **Python 3.14** (tested on 3.14.5, Windows 11) and an OpenAI API key.

```powershell
# Windows (PowerShell)
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env        # then put your key in .env as OPENAI_API_KEY=...
.\.venv\Scripts\python.exe -m app  # serves on http://localhost:8000
```

```bash
# macOS / Linux
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env               # then put your key in .env
.venv/bin/python -m app
```

- **Port:** the `PORT` environment variable (default `8000`).
- **Startup:** the index is built at startup and takes a few seconds. Wait for the `Indexed 48 sections` log line.
- **Running uvicorn directly:** `python -m uvicorn app.main:app --port 8000`. Uvicorn ignores `PORT`, so pass `--port` yourself.

**Try it:**

```powershell
# Windows PowerShell (there, `curl` is an alias for Invoke-WebRequest and takes different arguments)
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType "application/json" -Body '{"question": "How much paternity leave do I get?"}'
```

```bash
# macOS / Linux
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How much paternity leave do I get?"}'
```

**Other commands:**

| Command | What it does |
|---|---|
| `python -m pytest -q` | 57 offline tests (fake embedder and fake model; no key needed, no tokens spent) |
| `python scripts/evaluate.py [--out FILE] [--questions FILE]` | Runs a question set against the running service. Writes `results.csv` (or `--out`) and `eval/responses.jsonl` |
| `python scripts/calibrate_gate.py` | Re-calibrates the refusal score gate on `eval/dev_questions.csv` |
| `python scripts/inspect_retrieval.py "question" [--split] [--target doc#section]` | Diagnostic: dense, BM25 and fused rank of every section for a query |

Every answered question is logged as one `policy_qa: ask {...}` line. It records the retrieved sections, the gate score, any sub-questions, quote retries, claim-check results and why a question was refused (`refused_by`).

## Models used

All model calls use temperature 0 and OpenAI Structured Outputs (strict JSON schemas).

| Purpose | Model | Setting / prompt |
|---|---|---|
| Embeddings | OpenAI `text-embedding-3-small` | `EMBED_MODEL` |
| Question splitting | OpenAI `gpt-4o-mini` | `SPLIT_PROMPT` |
| Answer generation | OpenAI `gpt-4o-mini` | `CHAT_MODEL`, `SYSTEM_PROMPT` |
| Claim checking | OpenAI `gpt-4o-mini` | `CHECK_PROMPT` |

All prompts are in `app/llm.py`.

## How it works

```
question ─► split into ≤3 sub-questions (a single-part question stays unchanged)
         ─► hybrid retrieval: Chroma cosine top-10 + BM25 top-10 → Reciprocal Rank Fusion
                single-part ─► top 5 sections
                multi-part  ─► up to 8 sections, taking turns between sub-questions
         ─► score gate: best cosine < GATE_THRESHOLD (0.40) → refuse
         ─► gpt-4o-mini answers from those sections only: answer + citations + not_covered
                answer empty and no citations ─► refuse
         ─► server verifies every quote word for word in the cited section (one retry if any fail)
                no verified citation ─► refuse
         ─► claim checker labels every sentence against the sections (never sees the question)
                + code checks: checker evidence must exist; number guard
                any sentence not "supported" ─► one retry, then drop sentences that still fail
         ─► response: plain-text answer (+ "The policy documents do not cover: …" if gaps) and citations
```

The whole request runs under a 45 s time budget, which keeps it inside the contract's 60 s.

## Design choices and reasons

**Document splitting: one chunk per top-level `## N.` section (48 chunks).**
- The contract asks for the top-level section number in every citation, so each chunk maps 1:1 to a citable section.
- Sections are short (a few sentences or bullets), so splitting them further would gain little.
- The disclaimer and metadata lines at the top of each file are not indexed.
- The document title is put in front of each chunk's indexed text (e.g. "Business Travel Policy · 4. Accommodation"). Without it, a heading like "4. Accommodation" doesn't say it is about travel.

**Embedding model: `text-embedding-3-small`.** It's cheap and OpenAI's cost-efficient default. With 48 short chunks, the larger model's extra quality is unlikely to matter.

**Vector store: Chroma (`PersistentClient`, cosine space).**
- Stores vectors with `doc_id` and `section` metadata.
- The collection is deleted and rebuilt at every startup, so the index can never be stale.
- `.chroma/` is git-ignored.

**Retrieval: hybrid dense + BM25 (`rank_bm25`), merged with Reciprocal Rank Fusion (k = 60).**
- Dense search handles paraphrases. BM25 catches exact terms such as amounts or "Spendly".
- RRF merges the two rankings by position, so there are no scores to normalise or weights to tune.
- **How many chunks:** a single-part question gets the top **5** sections.
- **Multi-part questions:** a `gpt-4o-mini` call first splits the question into at most 3 standalone sub-questions. Each gets its own fused ranking, and up to **8** sections are chosen by taking turns (#1 of each, then #2 of each, …). That way a section that matters for only one part isn't outvoted by the question's dominant topic. The model still answers the original question.

**How refusal works.** A question is refused (`answered: false`, a fixed sentence, `citations: []`) in any of these cases:
1. **Score gate.** The best dense cosine similarity is below `GATE_THRESHOLD = 0.40`. The gate uses the dense score because RRF scores are rank-based and say nothing about whether the best match is actually relevant. The threshold was calibrated on 10 separate development questions, **not** the 15 test questions (see below).
2. **Model finds nothing covered.** The model is told to answer every part of the question the sections address. A rule, limit, deadline, procedure, condition or responsible person counts, even without a direct yes/no. Parts not addressed go in `not_covered`. If it addresses nothing, it returns an empty answer and no citations.
3. **No verified citation.** If, after one retry, no quote can be found word for word in its cited section, the answer is refused rather than shown unsupported.
4. **Claim check removes everything.** If no sentence of the answer survives the claim check, the answer is refused.

If the model answers part of a question, the answer ends with "The policy documents do not cover: …".

**Citations.**
- Each quote is normalised the way the contract's checker does (Markdown `**`, `#`, `-` and extra spaces ignored), and it must appear in the cited `doc_id` + `section`.
- A matching quote is returned as the exact source text.
- If any quote fails, the model gets one retry with the failures listed. Only verified citations are kept.

**Claim checker** (catches sentences that no quote supports).
- A separate model call labels each answer sentence `supported` / `partly_supported` / `not_supported` / `contradicted`. It sees only the retrieved sections, never the question, so it can't be swayed by what an answer "should" say. For `supported`, it must copy the evidence.
- Code then lowers verdicts that don't hold up:
  - the checker's evidence must exist word for word in the section it names;
  - a **number guard** rejects numbers that are in neither the sections nor the question, unless they're a question amount × a percentage from the sections (so 50% × NGN 300,000 = NGN 150,000 passes).
- Anything not `supported` triggers one retry. Sentences that still fail are dropped, along with citations that only supported dropped sentences.

**Plain text.** Markdown bold markers are removed from the answer text. Quotes keep the source's exact text.

**Errors.** Validation problems (missing, empty or over-long `question`, or invalid JSON) return `422 {"error": ...}`. Model or provider failures return `500 {"error": ...}`, never `200` with `answered: false`.

**Time budget** (`REQUEST_BUDGET_SECONDS`, default 45 s).
- Each model call's timeout is half the time left, because the SDK retries once.
- Optional steps (retries, claim checks) are skipped when less than 10 s remain.
- If time runs out before any citation is verified, the request fails with `500`.
- In the evaluation runs the slowest question took about 13 s; the median was about 3 s.

**Determinism.** Temperature 0 makes answers near-deterministic, but not fully: some answers varied between identical runs (see Evaluation).

### Gate calibration (`eval/gate_calibration.csv`)

- **Scores:** top dense cosine similarity on the 10 dev questions was 0.485–0.739 for answerable ones and 0.373–0.483 for unanswerable ones. The two groups are only 0.002 apart, so the midpoint (0.484) would have been fragile. Close-call questions that use policy vocabulary (e.g. "bereavement leave", 0.483) score like real ones.
- **Threshold used: 0.40.** This is a conservative value. It blocks only clearly off-topic questions (the dress-code question scored 0.373) and leaves close calls to the model.
- **BM25 as a gate signal: considered, not used.** Raw BM25 scores also overlap: the off-topic dress-code question scored 6.49 because "office" matches, above an answerable question at 6.25. They also depend on question length.

## Evaluation process

Each version was run on the 15 test questions **three times** (temperature 0 is not fully deterministic). The changes after the baseline were also run on the 10 dev questions, to catch missed refusals. **Changes were added and measured one at a time**, so each effect can be attributed. 

| Version | What changed | Correct / correct refusals / failures | Kept? |
|---|---|---|---|
| Baseline (run 1 + 3 reruns) | Design above, without splitting, `not_covered` or the checker | 10 / 3 / 2 in every run | – |
| "Run 2" | Four changes at once (stricter prompt, splitting, stemming, formatting) | 6 / 3 / 6 | **Reverted**: four correct answers became refusals, and the cause couldn't be isolated |
| Change 1 | Question splitting | 10 / 3 / 2 | Kept: Q11 stopped inventing a claim |
| Change 1b | Splitter worded sub-questions like policy rules | – (1 run) | **Reverted**: Q11 got worse (an uncited, false "Travel Desk approves" claim) |
| Change 2 | `answer` + `citations` + `not_covered` instead of a yes/no flag | 11/3/1, 10/3/2, 11/3/1 | Kept: Q10 answered in 2 of 3 runs. No missed refusals on the dev set |
| Change 3 | Claim checker + number guard | 11/3/1, 10/3/2, 11/3/1 | Kept: no false statements in any run |
| **Final** (submitted `results.csv`) | Time budget, plain-text answers, citation trimming | **10 / 3 / 2** | Submitted. Dev set: 10/10 handled correctly |

**What the `eval/` folder holds:**

| File | Contents |
|---|---|
| `dev_questions.csv` | The 10 dev questions |
| `gate_calibration.csv` | Gate calibration scores |
| `results_run1.csv` | The original run, judged |
| `results_baseline_1..3.csv` | Baseline reruns |
| `results_run2.csv` | The reverted four-changes-at-once attempt |
| `results_change1_*`, `results_change1b_1`, `results_change2_*`, `results_change3_*` | Each change's runs, including `_dev` |
| `results_final_dev.csv` | The final system on the dev questions |
| `responses_run1.jsonl`, `responses_run2.jsonl` | Raw responses for those runs |
| `service_log_change2_change3.txt` | Service log with every `ask` trace for changes 2 and 3 |

### Failure analysis

**Q10 (gift worth NGN 25,000): answered in 2 of 3 change runs, refused in the other one and in the final run.**
- Retrieval found the right section (09 #2) at rank 1 in every run.
- The service log shows the model's first quote failed the word-for-word check **in every run**. The refused runs also failed the retry (`refused_by: no_verified_citation`).
- The source sentence mixes bold text and an email address in brackets, and the model appears to drop or join parts of it.
- Before change 2 the model refused outright, because the policy gives a procedure rather than a yes/no.

**Q11 (hotel within limit + who approves the claim): `partially_correct` in every version.**
- The hotel limit is always answered correctly.
- The approval rule (05 #3: line manager **and** Finance above NGN 50,000) was **never retrieved**. `scripts/inspect_retrieval.py` shows why: the section shares almost no words with the question (no "claim", "approve" or "hotel"), so it ranks about 40th in BM25. The question's hotel details also pull both searches towards the travel policy.
- Worded the way the policy states it, the need ranks 05 #3 first in both searches. Offline replays showed that stemming or stopword removal would still not bring it into the final 8.
- The remaining levers (many more slots, or steering the splitter) would be tuning to one test question, so this is left as a **known limitation**.
- Change 3 stopped the answer from inventing an approver: in the final system, the approval part is dropped or flagged as not covered.

**Claim-checker false alarms.** In a few runs the checker rejected correct sentences, which were then dropped. Examples: Q12's calculated "repay NGN 150,000" (rated by the checker model, not the number guard), and part of Q9's detail. Treating `partly_supported` as a failure is strict on purpose: it trades some completeness for no unsupported statements.

## Known limitations

- **Vocabulary gap in retrieval.** A rule worded very differently from the question (Q11's approvals) can be missed. A small top-k makes this worse.
- **Exact-quote brittleness.** Sentences with inline Markdown or brackets are hard for the model to copy exactly (Q10).
- **Run-to-run variation at temperature 0.** This shows on Q10, Q11 and occasionally on extra detail in Q9 and Q12.
- **Over-strict claim checking** can drop correct detail.
- **Cost and latency.** Up to 6 model calls per question (median about 3 s).
- **BM25 has no stopword removal or stemming.** Common words like "the" or "who" can match unrelated sections.
- **Only 10 dev questions** were used for calibration.

## Results (final system, `results.csv`)

One fresh run of the final code. Outcomes were judged by hand against the policies, and failures are explained in the `notes` column. The three change 3 runs, the same pipeline before the final fixes (which don't change which questions are answered), scored 11 / 3 / 1, 10 / 3 / 2 and 11 / 3 / 1 (`eval/results_change3_*.csv`). The run-to-run difference is entirely Q10.

| Outcome | Count | Questions |
|---|---|---|
| **Correct answers** | **10** | Q1–Q9, Q12 |
| **Correct refusals** | **3** | Q13 (pension), Q14 (dog), Q15 (stock options) |
| **Failures** | **2** | Q10 `wrong_refusal` (the section was retrieved; in every logged run, refusals came from quote verification failing twice), Q11 `partially_correct` (the approval rule 05 #3 was not retrieved) |

