"""FastAPI service implementing API_CONTRACT.md: GET /health and POST /ask."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import load_settings
from app.documents import load_sections
from app.retrieval import HybridIndex
from app.service import PolicyQA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("policy_qa")


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def check_length(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        if len(value) > 1000:
            raise ValueError("question must be at most 1000 characters")
        return value  # returned unchanged: the response echoes the question exactly as received


def build_default_qa() -> PolicyQA:
    from app.llm import OpenAIClient

    settings = load_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    client = OpenAIClient(
        settings.openai_api_key, settings.chat_model, settings.embed_model, settings.llm_timeout_seconds
    )
    sections = load_sections(settings.policies_dir)
    index = HybridIndex(
        sections,
        client,
        settings.chroma_path,
        settings.collection_name,
        settings.candidates_per_retriever,
        settings.rrf_k,
    )
    logger.info(
        "Indexed %d sections from %s (chat=%s, embed=%s, top_k=%d, top_k_split=%d, gate=%.3f)",
        len(sections), settings.policies_dir, settings.chat_model, settings.embed_model,
        settings.top_k, settings.top_k_split, settings.gate_threshold,
    )
    return PolicyQA(
        index, client, settings.top_k, settings.gate_threshold,
        settings.top_k_split, settings.max_sub_questions,
        settings.request_budget_seconds, settings.llm_timeout_seconds,
    )


def create_app(qa_factory=build_default_qa) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.qa = qa_factory()
        yield

    app = FastAPI(title="Kestrel Nova Policy Q&A", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        errors = exc.errors()
        message = "invalid request"
        if errors:
            err = errors[0]
            if err.get("type") == "missing":
                message = "question is required"
            elif err.get("type") == "json_invalid":
                message = "request body must be valid JSON"
            else:
                message = str(err.get("msg", message)).removeprefix("Value error, ")
        return JSONResponse(status_code=422, content={"error": message})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})

    @app.exception_handler(Exception)
    async def internal_error(_: Request, exc: Exception):
        logger.exception("Internal error while answering")
        return JSONResponse(status_code=500, content={"error": "internal error: the answer could not be generated"})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/ask")
    def ask(body: AskRequest, request: Request):
        result = request.app.state.qa.ask(body.question)
        logger.info("ask %s", json.dumps({"question": body.question, "answered": result.answered, **result.trace}))
        return {
            "question": body.question,
            "answered": result.answered,
            "answer": result.answer,
            "citations": [
                {"doc_id": c.doc_id, "section": c.section, "quote": c.quote} for c in result.citations
            ],
        }

    return app


app = create_app()
