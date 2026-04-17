"""
mine_extractor.llm_client
-------------------------
Thin wrapper around Groq's OpenAI-compatible endpoint.

Why a custom wrapper instead of LangChain's ChatGroq?
  * Lets us tune retries and timeouts precisely.
  * Removes an indirection layer (easier to debug prompts).
  * Avoids pulling in the full LangChain dependency tree just for chat.

A single ``GroqClient`` can be shared across the whole pipeline; requests
are stateless so it is safe to reuse.
"""

from __future__ import annotations

import time
from typing import Any

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from mine_extractor.logging_config import get_logger
from settings import settings

logger = get_logger(__name__)


class GroqClient:
    """Minimal chat client with automatic retries on transient errors."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")

        self._client = OpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens

    # ----------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None = None,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
    ) -> str:
        """Send a chat request and return the assistant text."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except (RateLimitError, APIConnectionError, APIError) as exc:
                if attempt >= max_retries:
                    logger.error("LLM request failed after %d attempts: %s", attempt, exc)
                    raise
                wait = backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "LLM request failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
