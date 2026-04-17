"""
mine_extractor.extraction.mine_detector
----------------------------------------
Identify the mines discussed in a single indexed document and choose
the primary one for downstream expenditure extraction.

Approach:
    1. Run a handful of targeted queries against the vector index,
       restricted to chunks from the target PDF.
    2. Collapse the results into a single context block.
    3. Ask the LLM to enumerate every mine it sees and pick the primary.
"""

from __future__ import annotations

import json

from mine_extractor.extraction.prompts import (
    MINE_IDENTIFIER_SYSTEM,
    MINE_IDENTIFIER_USER_TEMPLATE,
    format_snippets,
)
from mine_extractor.extraction.schemas import MineList
from mine_extractor.llm_client import GroqClient
from mine_extractor.logging_config import get_logger
from mine_extractor.retrieval.searcher import Searcher

logger = get_logger(__name__)


_DISCOVERY_QUERIES: tuple[str, ...] = (
    "name of the mine or operation described in this report",
    "mine location and operating company",
    "primary commodity produced by the operation",
    "summary of the project and property description",
)


class MineDetector:
    """Identify the mine(s) in a document and choose the primary one."""

    def __init__(
        self,
        searcher: Searcher,
        llm: GroqClient | None = None,
    ) -> None:
        self.searcher = searcher
        self.llm = llm or GroqClient()

    # ----------------------------------------------------------------
    def detect(self, source_name: str, *, per_query_k: int = 4) -> MineList:
        """Return the list of candidate mines + the selected primary."""
        snippets = self._gather_context(source_name, per_query_k=per_query_k)
        if not snippets:
            logger.warning("No retrievable context for %s", source_name)
            return MineList(candidates=[], primary_mine="unknown", reasoning="no context")

        prompt = MINE_IDENTIFIER_USER_TEMPLATE.format(
            source_name=source_name,
            context=format_snippets(snippets),
        )
        raw = self.llm.chat(
            messages=[
                {"role": "system", "content": MINE_IDENTIFIER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(raw)
            result = MineList.model_validate(data)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse mine-identifier output: %s\n%s", exc, raw[:400])
            result = MineList(candidates=[], primary_mine="unknown", reasoning=str(exc))

        logger.info(
            "Mine detection for %s → primary=%r (candidates=%d)",
            source_name,
            result.primary_mine,
            len(result.candidates),
        )
        return result

    # ----------------------------------------------------------------
    def _gather_context(self, source_name: str, *, per_query_k: int) -> list[dict]:
        """Collect a deduplicated set of snippets for the identifier prompt."""
        seen: set[tuple[str, int, int]] = set()
        snippets: list[dict] = []

        for query in _DISCOVERY_QUERIES:
            hits = self.searcher.find(
                query,
                source=source_name,
                initial_k=max(per_query_k * 2, 8),
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
                        "text": hit.text,
                        "score": hit.metadata.get("rerank_score", hit.score),
                    }
                )

        return snippets
