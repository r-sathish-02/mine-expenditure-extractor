"""
settings.py
-----------
Project-wide settings, loaded from environment variables with sensible defaults.

Uses a simple dataclass instead of pydantic-settings to keep dependencies light.
Directories referenced here are created on import so downstream modules can
rely on their existence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


# ------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent
PDF_INPUT_DIR: Path = ROOT_DIR / "pdfs"
OUTPUT_DIR: Path = ROOT_DIR / "outputs"
CACHE_DIR: Path = ROOT_DIR / "cache"
VECTOR_DIR: Path = ROOT_DIR / "vector_index"
LOG_DIR: Path = ROOT_DIR / "logs"

for _d in (PDF_INPUT_DIR, OUTPUT_DIR, CACHE_DIR, VECTOR_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------
# Runtime configuration
# ------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Bundle of tunable configuration values."""

    # --- Parsing -------------------------------------------------------
    # Whether to invoke an LLM to clean up messy tables.
    # When False the pipeline uses only layout-preserving text extraction
    # (still fast and usually enough for technical reports).
    enable_llm_table_enhancement: bool = field(
        default_factory=lambda: _env_bool("LLM_TABLE_ENHANCE", True)
    )
    # Minimum number of aligned numeric rows on a page to flag it as a
    # candidate for LLM table cleanup.
    table_heuristic_min_numeric_rows: int = 3

    # --- Chunking ------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))

    # --- Embeddings ----------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))

    # --- Vector store --------------------------------------------------
    vector_collection: str = "mine_reports"

    # --- Retrieval -----------------------------------------------------
    search_top_k: int = 15
    final_top_k: int = 6
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- LLM -----------------------------------------------------------
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_base_url: str = field(
        default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    )
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 2048))

    # --- Misc ----------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


# A single shared instance — import this everywhere.
settings = Settings()
