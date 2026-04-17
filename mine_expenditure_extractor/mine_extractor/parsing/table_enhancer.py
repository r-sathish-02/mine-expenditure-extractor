"""
mine_extractor.parsing.table_enhancer
--------------------------------------
When a page looks like it contains tabular data that won't survive a
naive text dump, we ask an LLM to rewrite the page as markdown — plain
text stays as prose, aligned numeric blocks become markdown tables.

This is the "LLM-based parser" step.  It is invoked *only* on pages
flagged by ``page_classifier.is_table_candidate``, so the vast majority
of pages never hit the LLM.  That keeps the pipeline fast while still
handling complex tables accurately.
"""

from __future__ import annotations

from mine_extractor.llm_client import GroqClient
from mine_extractor.logging_config import get_logger

logger = get_logger(__name__)


_SYSTEM_PROMPT = (
    "You convert a single page of a mining / geological technical report "
    "into clean markdown. "
    "Follow these rules strictly:\n"
    "  1. Preserve ALL numbers, units and currency symbols exactly as they appear.\n"
    "  2. Detect aligned tabular data (columns of numbers, cost breakdowns, "
    "reserves tables, etc.) and render it as a proper markdown table with "
    "a header row and a separator row.\n"
    "  3. Keep section headings (e.g. '21.2 Capital Cost Estimates') as markdown "
    "headings with the appropriate depth (##, ### ...).\n"
    "  4. Keep narrative prose as ordinary paragraphs.\n"
    "  5. Do NOT paraphrase, summarise, translate or invent any content. "
    "If a row has a missing value, leave the cell empty.\n"
    "  6. Do NOT wrap the output in ``` code fences. Return the markdown directly."
)


class TableAwareMarkdownEnhancer:
    """Converts a raw page of text into cleaned markdown using an LLM."""

    def __init__(self, client: GroqClient | None = None) -> None:
        self._client = client or GroqClient(temperature=0.0, max_tokens=3500)

    # ----------------------------------------------------------------
    def enhance(self, page_text: str, page_number: int | None = None) -> str:
        """Return a cleaned markdown version of ``page_text``."""
        if not page_text.strip():
            return page_text

        user_msg = f"Page number: {page_number}\n\n" if page_number is not None else ""
        user_msg += "=== RAW PAGE TEXT START ===\n"
        user_msg += page_text
        user_msg += "\n=== RAW PAGE TEXT END ===\n\nReturn the cleaned markdown now."

        try:
            result = self._client.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
            )
        except Exception as exc:  # pragma: no cover — network/LLM failures
            logger.warning(
                "Table enhancement failed on page %s (%s) — keeping raw text",
                page_number,
                exc,
            )
            return page_text

        cleaned = result.strip()
        # Strip accidental code fences (the system prompt forbids them, but
        # smaller models slip them in anyway).
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        if not cleaned:
            return page_text
        return cleaned
