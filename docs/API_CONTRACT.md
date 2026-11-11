# API Contract — Project 1: Company Policy Q&A Assistant

Your service must follow this contract exactly. We run every submission ourselves and test it the same way, so a different request or response shape will fail our tests even if the answers are good.

**Mandatory requirements.** A submission is not graded if:
- the service cannot be started by following your README, or does not follow this contract; or
- any API key, token or password appears anywhere in the repository, **including the Git history**. Read keys from environment variables and include a `.env.example` file.

## 1. Running the service

- The service listens on **port 8000** by default (make it configurable with an environment variable).
- All requests and responses use `Content-Type: application/json`.
- Each request must return within **60 seconds**.
- Your README must show how to start the service in a few commands.

## 2. `GET /health`

Lets us check that the service is up before testing.

**Response `200`:**
```json
{ "status": "ok" }
```

## 3. `POST /ask`

### Request

```json
{ "question": "How much paternity leave do I get?" }
```

| Field | Type | Required | Rules |
|---|---|---|---|
| `question` | string | yes | 1–1000 characters |

### Response `200` — answered

```json
{
  "question": "I stayed in a hotel costing NGN 70,000 per night on a domestic work trip. Is that within the limit?",
  "answered": true,
  "answer": "Yes. The domestic hotel limit is NGN 80,000 per night, so NGN 70,000 is within it.",
  "citations": [
    {
      "doc_id": "04_business_travel_policy",
      "section": "4",
      "quote": "Domestic hotels: maximum **NGN 80,000 per night**."
    }
  ]
}
```

### Response `200` — refused (answer not in the documents)

```json
{
  "question": "Can I bring my dog to the office?",
  "answered": false,
  "answer": "The policy documents do not contain information about this.",
  "citations": []
}
```

### Fields

| Field | Type | Rules |
|---|---|---|
| `question` | string | The question exactly as received. |
| `answered` | boolean | `true` if the answer comes from the documents; `false` if the documents do not contain it. |
| `answer` | string | The answer in plain language. When `answered` is `false`, a short statement that the documents do not cover the question. Do not add a guess. |
| `citations` | array | At least 1 item when `answered` is `true`. Must be empty (`[]`) when `answered` is `false`. |
| `citations[].doc_id` | string | The policy file name **without** `.md`, e.g. `01_annual_leave_policy`. |
| `citations[].section` | string | The top-level section number from the document heading, e.g. `"4"` for `## 4. Accommodation`. |
| `citations[].quote` | string | Text copied **exactly** from that document that supports the answer. We check this automatically, so it must appear word for word in the source file. Markdown symbols (`**`, `#`, `-`) and extra spaces are ignored in the check. |

### Citation rules

- Every fact in the answer must be supported by at least one citation.
- If an answer uses more than one document, cite each document used.
- Cite only text the answer actually relies on, not every retrieved chunk.

### Errors

| Status | When | Body |
|---|---|---|
| `422` (or `400`) | `question` missing, empty, or too long | `{ "error": "<short message>" }` |
| `500` | Internal failure (e.g. the model provider is unavailable) | `{ "error": "<short message>" }` |

A failed request must return an error status. It must not return `200` with `answered: false`, because that would look like a correct refusal.

## 4. Results file

Run your assistant on all 15 questions in `test_questions.csv` and submit `results.csv` with one row per question:

| Column | Content |
|---|---|
| `id` | Question id from `test_questions.csv` |
| `question` | The question |
| `answered` | `true` / `false`, from your API response |
| `answer` | The answer text from your API response |
| `citations` | `doc_id#section` values separated by `;`, e.g. `04_business_travel_policy#4;05_expense_reimbursement_policy#3` |
| `outcome` | Your own judgement, one of: `correct`, `partially_correct`, `incorrect`, `correct_refusal`, `wrong_refusal` (refused but the answer was in the documents), `missed_refusal` (answered but the documents do not contain it) |
| `notes` | For any outcome other than `correct` or `correct_refusal`, say what went wrong and why (e.g. "retrieved the travel policy but not the expense policy") |

At the end of your README, give the summary counts: correct answers, correct refusals, and failures.

We re-grade your results ourselves. Honest reporting of failures counts in your favour.
