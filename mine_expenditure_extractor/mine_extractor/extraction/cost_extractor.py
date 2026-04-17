"""
mine_extractor.extraction.cost_extractor
-----------------------------------------
Given a selected mine, pull capital and operating cost figures out of
the document and return a validated :class:`MineExpenditure` object.

Strategy:
    1. Run a battery of expenditure-oriented queries against the vector
       index (restricted to the target PDF).
    2. Deduplicate hits and order them by page number so tables keep
       their natural flow in the prompt.
    3. Ask the LLM to return a strict JSON breakdown.  The JSON is then
       validated with Pydantic; if the model returns garbage we record a
       low-confidence result instead of raising.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from mine_extractor.extraction.prompts import (
    COST_EXTRACTOR_SYSTEM,
    COST_EXTRACTOR_USER_TEMPLATE,
    format_snippets,
)
from mine_extractor.extraction.schemas import MineExpenditure
from mine_extractor.llm_client import GroqClient
from mine_extractor.logging_config import get_logger
from mine_extractor.retrieval.searcher import Searcher

logger = get_logger(__name__)


_COST_QUERIES: tuple[str, ...] = (
    "capital cost estimate life of mine breakdown",
    "LOM capital expenditures mining equipment plant infrastructure",
    "operating cost estimate per tonne mining processing transportation",
    "total capex and opex figures for the mine",
    "sustaining capital and reclamation cost",
    "economic analysis cash cost summary",
)


class CostExtractor:
    """Extract structured expenditure information for a mine."""

    def __init__(
        self,
        searcher: Searcher,
        llm: GroqClient | None = None,
    ) -> None:
        self.searcher = searcher
        self.llm = llm or GroqClient()

    # ----------------------------------------------------------------
    def extract(
        self,
        source_name: str,
        mine_name: str,
        *,
        per_query_k: int = 4,
    ) -> tuple[MineExpenditure, list[dict]]:
        """
        Extract expenditure data and return (result, context_snippets).

        The snippets are returned alongside the result so the caller can
        persist them for audit and debugging.
        """
        snippets = self._gather_context(source_name, mine_name, per_query_k=per_query_k)
        if not snippets:
            logger.warning("No context to extract from for %s / %s", source_name, mine_name)
            fallback = MineExpenditure(
                mine_name=mine_name,
                confidence="low",
                caveats="No relevant excerpts retrieved — index may be empty.",
                source_document=source_name,
            )
            return fallback, []

        prompt = COST_EXTRACTOR_USER_TEMPLATE.format(
            source_name=source_name,
            mine_name=mine_name,
            context=format_snippets(snippets),
        )
        raw = self.llm.chat(
            messages=[
                {"role": "system", "content": COST_EXTRACTOR_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(raw)
            # LLM sometimes forgets these top-level hints; inject them.
            data.setdefault("mine_name", mine_name)
            data.setdefault("source_document", source_name)
            result = MineExpenditure.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error("Could not parse cost extractor output: %s\n%s", exc, raw[:400])
            result = MineExpenditure(
                mine_name=mine_name,
                confidence="low",
                caveats=f"LLM returned unparseable JSON: {exc}",
                source_document=source_name,
            )

        logger.info(
            "Cost extraction for %s / %s → confidence=%s",
            source_name,
            mine_name,
            result.confidence,
        )
        return result, snippets

    # ----------------------------------------------------------------
    def _gather_context(self, source_name: str, mine_name: str, *, per_query_k: int) -> list[dict]:
        """Collect a deduplicated, page-ordered context block."""
        seen: set[tuple[str, int, int]] = set()
        snippets: list[dict] = []

        # Include a mine-name-specific query in addition to the generic ones
        queries = [*_COST_QUERIES, f"{mine_name} total capital and operating cost"]

        for query in queries:
            hits = self.searcher.find(
                query,
                source=source_name,
                initial_k=max(per_query_k * 2, 10),
                final_k=per_query_k,
            )
            for hit in hits:
                key = (
                    str(hit.metadata.get("source", "?")),
                    int(hit.metadata.get("page", 0)),
                    int(hit.metadata.get("chunk_index", 0)),
                )
                if key in seen:
                    continue
                seen.add(key)
                snippets.append(
                    {
                        "source": hit.metadata.get("source"),
                        "page": hit.metadata.get("page"),
                        "chunk_index": hit.metadata.get("chunk_index"),
                        "text": hit.text,
                        "score": hit.metadata.get("rerank_score", hit.score),
                    }
                )

        # Page-ordered context reads more naturally for the LLM.
        snippets.sort(key=lambda s: (int(s.get("page") or 0), int(s.get("chunk_index") or 0)))
        return snippets
