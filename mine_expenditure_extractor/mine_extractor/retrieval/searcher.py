"""
mine_extractor.retrieval.searcher
----------------------------------
Two-stage retriever: vector index → cross-encoder rerank.

Exposed as a small class so the pipelines can swap the reranker off
(e.g. for lightweight debug runs) without touching the callers.
"""

from __future__ import annotations

from mine_extractor.indexing.vector_index import SearchHit, VectorIndex
from mine_extractor.logging_config import get_logger
from mine_extractor.retrieval.reranker import CrossEncoderReranker
from settings import settings

logger = get_logger(__name__)


class Searcher:
    """Retrieve + rerank chunks for a query."""

    def __init__(
        self,
        vector_index: VectorIndex,
        reranker: CrossEncoderReranker | None = None,
        *,
        use_rerank: bool = True,
    ) -> None:
        self.index = vector_index
        self.reranker = reranker or CrossEncoderReranker()
        self.use_rerank = use_rerank

    # ----------------------------------------------------------------
    def find(
        self,
        query: str,
        *,
        source: str | None = None,
        initial_k: int | None = None,
        final_k: int | None = None,
    ) -> list[SearchHit]:
        initial_k = initial_k or settings.search_top_k
        final_k = final_k or settings.final_top_k

        candidates = self.index.search(query, k=initial_k, source=source)
        if not candidates:
            return []

        if self.use_rerank:
            return self.reranker.rerank(query, candidates, top_k=final_k)
        return candidates[:final_k]
