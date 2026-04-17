"""
mine_extractor.indexing.vector_index
-------------------------------------
Persistent Chroma vector index for mine report chunks.

Uses the native chromadb client (rather than the LangChain Chroma
wrapper) to keep dependencies minimal and the API explicit.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from mine_extractor.indexing.chunker import Chunk
from mine_extractor.indexing.embedder import Embedder
from mine_extractor.logging_config import get_logger
from settings import VECTOR_DIR, settings

logger = get_logger(__name__)


@dataclass
class SearchHit:
    text: str
    metadata: dict
    score: float  # cosine similarity in [0, 1]


class VectorIndex:
    """Thin wrapper around chromadb with project defaults."""

    def __init__(
        self,
        collection_name: str = settings.vector_collection,
        persist_dir: Path = VECTOR_DIR,
        embedder: Embedder | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or Embedder()

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorIndex open (collection=%s, items=%d)",
            self.collection_name,
            self.count(),
        )

    # ----------------------------------------------------------------
    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:  # pragma: no cover
            return 0

    def clear(self) -> None:
        """Wipe the collection and recreate it empty."""
        with contextlib.suppress(Exception):
            self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Collection '%s' cleared", self.collection_name)

    def delete_source(self, source: str) -> int:
        """Remove every chunk that came from ``source`` (e.g. a PDF filename)."""
        try:
            matches = self._collection.get(where={"source": source})
        except Exception as exc:
            logger.warning("Could not query source %s: %s", source, exc)
            return 0
        ids = matches.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    # ----------------------------------------------------------------
    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Embed and upsert chunks. Returns the number of items written."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [self._chunk_id(c) for c in chunks]

        logger.info("Embedding %d chunk(s)...", len(chunks))
        vectors = self.embedder.embed_documents(texts)

        # upsert so re-indexing the same PDF is idempotent.
        self._collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunk(s). Collection total: %d", len(chunks), self.count())
        return len(chunks)

    # ----------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        source: str | None = None,
    ) -> list[SearchHit]:
        """
        Cosine similarity search.

        When ``source`` is supplied, results are restricted to chunks
        coming from that PDF — useful when we want to answer a question
        about a single document.
        """
        k = k or settings.search_top_k
        if self.count() == 0:
            logger.warning("Vector index is empty")
            return []

        query_vec = self.embedder.embed_query(query)
        where = {"source": source} if source else None

        raw = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(k, self.count()),
            where=where,
        )

        hits: list[SearchHit] = []
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        for text, meta, dist in zip(docs, metas, dists, strict=False):
            # Chroma returns cosine DISTANCE (1 - similarity). Convert back.
            similarity = max(0.0, 1.0 - float(dist))
            hits.append(SearchHit(text=text, metadata=dict(meta or {}), score=similarity))

        logger.info(
            "Vector search returned %d hit(s) for '%s%s'",
            len(hits),
            query[:60],
            "..." if len(query) > 60 else "",
        )
        return hits

    def list_sources(self) -> list[str]:
        """Return the distinct ``source`` values currently indexed."""
        try:
            data = self._collection.get()
        except Exception:  # pragma: no cover
            return []
        seen: set[str] = set()
        for meta in data.get("metadatas") or []:
            if meta and "source" in meta:
                seen.add(str(meta["source"]))
        return sorted(seen)

    # ----------------------------------------------------------------
    @staticmethod
    def _chunk_id(chunk: Chunk) -> str:
        """Deterministic id so repeated indexing is idempotent."""
        meta = chunk.metadata
        key = f"{meta.get('source', '?')}|p{meta.get('page', 0)}|c{meta.get('chunk_index', 0)}|{chunk.text[:64]}"
        return hashlib.md5(key.encode()).hexdigest()
