"""OpenAI calls: embeddings and the grounded-answer call (Structured Outputs, temperature 0)."""

from __future__ import annotations

import json
from typing import Protocol

from openai import OpenAI

from app.documents import Section

SYSTEM_PROMPT = """You answer employee questions about Kestrel Nova Ltd company policies, using ONLY the policy \
sections provided in the user message.

1. Answer every part of the question that the sections address. A section addresses a question when it states a \
rule, limit, deadline, procedure, condition or responsible person that applies to the employee's situation, even if \
it does not give a direct yes or no. In that case, explain what the rule means for the employee.
2. Be complete: include every requirement, limit, condition and exception the sections give for the question.
3. Put any part of the question that the sections do not address in "not_covered", as a short description of what \
is missing. Leave it empty if nothing is missing. If the sections address none of the question, leave "answer" \
empty and return no citations. Do not use outside knowledge and do not guess.
4. Every fact in the answer must be supported by a citation. For each citation give:
   - "doc_id": exactly as shown in the section label (e.g. 04_business_travel_policy)
   - "section": the section number shown in the label (e.g. "4")
   - "quote": a short passage copied CHARACTER FOR CHARACTER from that section. Do not paraphrase, \
shorten words, change numbers or join text from different places. One sentence or bullet is ideal.
5. If the answer uses more than one document, cite each document used.
6. Write in plain, short language. If the question needs a calculation, state the result and the rule it comes \
from."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "section": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["doc_id", "section", "quote"],
                "additionalProperties": False,
            },
        },
        "not_covered": {"type": "string"},
    },
    "required": ["answer", "citations", "not_covered"],
    "additionalProperties": False,
}


SPLIT_PROMPT = """You prepare employee questions for a search over company policy documents. Do not answer them.

If the question asks about several distinct things, split it into standalone sub-questions, at most {max_parts}. \
Each sub-question must make sense on its own: repeat any needed details (amounts, situation) from the original. \
If the question asks about only one thing, return it unchanged as the only item."""

SPLIT_SCHEMA = {
    "type": "object",
    "properties": {"sub_questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["sub_questions"],
    "additionalProperties": False,
}


CHECK_PROMPT = """You check an answer against company policy sections. You are not told the question.

The answer has been split into numbered sentences. For each sentence, decide whether the policy sections \
support it:
- "supported": the sections state this, in the same or equivalent words.
- "partly_supported": the sections state part of it, or state something similar for a different situation.
- "not_supported": the sections do not state this.
- "contradicted": the sections state the opposite.
A sentence that only says the documents do not cover something counts as "supported".

For "supported" sentences, give the doc_id and section number from the section label, and in "evidence" copy \
CHARACTER FOR CHARACTER the shortest passage from that section that supports the sentence. For any other verdict, \
leave doc_id, section and evidence empty. Return one label for every sentence, using its number as "index"."""

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "verdict": {"type": "string",
                                "enum": ["supported", "partly_supported", "not_supported", "contradicted"]},
                    "doc_id": {"type": "string"},
                    "section": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["index", "verdict", "doc_id", "section", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def format_context(sections: list[Section]) -> str:
    blocks = []
    for s in sections:
        blocks.append(
            f'<section doc_id="{s.doc_id}" section="{s.section}" title="{s.title}">\n'
            f"{s.raw_text}\n</section>"
        )
    return "\n\n".join(blocks)


def build_user_message(question: str, sections: list[Section], feedback: str | None = None) -> str:
    message = f"Policy sections:\n\n{format_context(sections)}\n\nQuestion: {question}"
    if feedback:
        message += f"\n\nCorrection needed: {feedback}"
    return message


class AnswerModel(Protocol):
    # `timeout` (seconds) is set per call from the request's remaining time budget.
    def check_answer(self, sentences: list[str], sections: list[Section], timeout: float | None = None) -> list[dict]: ...

    def split_question(self, question: str, max_parts: int, timeout: float | None = None) -> list[str]: ...

    def answer(
        self, question: str, sections: list[Section], feedback: str | None = None, timeout: float | None = None
    ) -> dict: ...


class OpenAIClient:
    """Implements both the Embedder and AnswerModel protocols."""

    def __init__(self, api_key: str, chat_model: str, embed_model: str, timeout: float) -> None:
        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=1)
        self.chat_model = chat_model
        self.embed_model = embed_model

    def _opts(self, timeout: float | None) -> dict:
        return {} if timeout is None else {"timeout": timeout}

    def embed(self, texts: list[str], timeout: float | None = None) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.embed_model, input=texts, **self._opts(timeout))
        return [item.embedding for item in response.data]

    def check_answer(self, sentences: list[str], sections: list[Section], timeout: float | None = None) -> list[dict]:
        numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(sentences))
        response = self._client.chat.completions.create(
            model=self.chat_model,
            temperature=0,
            messages=[
                {"role": "system", "content": CHECK_PROMPT},
                {"role": "user", "content": f"Policy sections:\n\n{format_context(sections)}\n\n"
                                            f"Answer sentences:\n{numbered}"},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "sentence_checks", "strict": True, "schema": CHECK_SCHEMA},
            },
            **self._opts(timeout),
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Model returned an empty response")
        return json.loads(content)["labels"]

    def split_question(self, question: str, max_parts: int, timeout: float | None = None) -> list[str]:
        response = self._client.chat.completions.create(
            model=self.chat_model,
            temperature=0,
            messages=[
                {"role": "system", "content": SPLIT_PROMPT.format(max_parts=max_parts)},
                {"role": "user", "content": question},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "sub_questions", "strict": True, "schema": SPLIT_SCHEMA},
            },
            **self._opts(timeout),
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Model returned an empty response")
        parts = [p.strip() for p in json.loads(content)["sub_questions"] if p.strip()]
        return parts[:max_parts]

    def answer(
        self, question: str, sections: list[Section], feedback: str | None = None, timeout: float | None = None
    ) -> dict:
        response = self._client.chat.completions.create(
            model=self.chat_model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(question, sections, feedback)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "policy_answer", "strict": True, "schema": ANSWER_SCHEMA},
            },
            **self._opts(timeout),
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Model returned an empty response")
        return json.loads(content)
