"""
mine_extractor.retrieval.reranker
----------------------------------
Cross-encoder reranker for improving precision after initial vector
retrieval.

Unlike a bi-encoder, a cross-encoder sees the query and the candidate
together and can reason about fine-grained relevance. It is slower per
pair, so we only use it on the small set returned by vector search.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from sentence_transformers import CrossEncoder

from mine_extractor.indexing.vector_index import SearchHit
from mine_extractor.logging_config import get_logger
from settings import settings

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    logger.info("Loading cross-encoder: %s", model_name)
    return CrossEncoder(model_name)


class CrossEncoderReranker:
    """Rerank :class:`SearchHit` objects using a cross-encoder."""

    def __init__(self, model_name: str = settings.reranker_model) -> None:
        self.model_name = model_name

    @property
    def model(self) -> CrossEncoder:
        return _load_cross_encoder(self.model_name)

    # ----------------------------------------------------------------
    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        top_k: int = settings.final_top_k,
    ) -> list[SearchHit]:
        if not hits:
            return []
        pairs = [(query, h.text) for h in hits]
        scores = self.model.predict(pairs).tolist()

        scored = []
        for hit, score in zip(hits, scores, strict=False):
            hit.metadata = {**hit.metadata, "rerank_score": round(float(score), 4)}
            # Replace the bi-encoder score with the rerank score for sorting,
            # but keep the original under "vector_score" for transparency.
            hit.metadata.setdefault("vector_score", round(hit.score, 4))
            scored.append((hit, float(score)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        top = [hit for hit, _ in scored[:top_k]]
        logger.info(
            "Reranked %d → top %d (best score=%.3f)",
            len(hits),
            len(top),
            scored[0][1] if scored else 0.0,
        )
        return top
