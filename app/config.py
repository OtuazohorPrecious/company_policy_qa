"""Runtime settings, read from environment variables (and a local .env file if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root if it exists. Real environment variables win.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    chat_model: str
    embed_model: str
    port: int
    top_k: int
    top_k_split: int
    max_sub_questions: int
    candidates_per_retriever: int
    rrf_k: int
    gate_threshold: float
    policies_dir: Path
    chroma_path: Path
    collection_name: str
    llm_timeout_seconds: float
    request_budget_seconds: float


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        port=_int_env("PORT", 8000),
        top_k=_int_env("TOP_K", 5),
        top_k_split=_int_env("TOP_K_SPLIT", 8),
        max_sub_questions=_int_env("MAX_SUB_QUESTIONS", 3),
        candidates_per_retriever=_int_env("CANDIDATES_PER_RETRIEVER", 10),
        rrf_k=_int_env("RRF_K", 60),
        # Placeholder until calibrated with scripts/calibrate_gate.py.
        gate_threshold=_float_env("GATE_THRESHOLD", 0.0),
        policies_dir=Path(os.getenv("POLICIES_DIR", PROJECT_ROOT / "docs" / "policies")),
        chroma_path=Path(os.getenv("CHROMA_PATH", PROJECT_ROOT / ".chroma")),
        collection_name=os.getenv("CHROMA_COLLECTION", "policy_sections"),
        llm_timeout_seconds=_float_env("LLM_TIMEOUT_SECONDS", 20.0),
        request_budget_seconds=_float_env("REQUEST_BUDGET_SECONDS", 45.0),
    )
