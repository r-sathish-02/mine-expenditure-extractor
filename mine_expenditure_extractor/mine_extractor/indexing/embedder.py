"""
mine_extractor.indexing.embedder
--------------------------------
Lightweight wrapper around sentence-transformers for producing
document and query embeddings.

The model is loaded lazily the first time it is needed so that commands
that don't touch embeddings (e.g. CLI help, summary printing) stay snappy.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from mine_extractor.logging_config import get_logger
from settings import settings

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model(model_name: str, device: str) -> SentenceTransformer:
    logger.info("Loading embedding model '%s' on %s", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    logger.info("Embedding model ready")
    return model


class Embedder:
    """Encode text as normalised vectors suitable for cosine similarity."""

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        device: str = settings.embedding_device,
    ) -> None:
        self.model_name = model_name
        self.device = device

    @property
    def model(self) -> SentenceTransformer:
        return _load_model(self.model_name, self.device)

    # ----------------------------------------------------------------
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()
