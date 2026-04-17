"""
mine_extractor.pipeline.ingest_pipeline
----------------------------------------
End-to-end ingestion: PDF → markdown → chunks → vector index.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from mine_extractor.indexing.chunker import MarkdownChunker
from mine_extractor.indexing.vector_index import VectorIndex
from mine_extractor.logging_config import get_logger
from mine_extractor.parsing.pdf_to_markdown import PdfMarkdownParser

logger = get_logger(__name__)


@dataclass
class IngestReport:
    pdf_path: Path
    pages: int
    chunks: int
    llm_enhanced_pages: int

    def to_dict(self) -> dict:
        return {
            "pdf": str(self.pdf_path),
            "pages": self.pages,
            "chunks": self.chunks,
            "llm_enhanced_pages": self.llm_enhanced_pages,
        }


class IngestPipeline:
    """
    Compose the parser, chunker and vector index into a single runnable.
    """

    def __init__(
        self,
        parser: PdfMarkdownParser | None = None,
        chunker: MarkdownChunker | None = None,
        vector_index: VectorIndex | None = None,
    ) -> None:
        self.parser = parser or PdfMarkdownParser()
        self.chunker = chunker or MarkdownChunker()
        self.vector_index = vector_index or VectorIndex()

    # ----------------------------------------------------------------
    def ingest_one(self, pdf_path: str | Path) -> IngestReport:
        path = Path(pdf_path).resolve()
        parsed = self.parser.parse(path)
        chunks = self.chunker.split(parsed)

        # Refresh — if the file was ingested before, drop old chunks.
        deleted = self.vector_index.delete_source(parsed.source_name)
        if deleted:
            logger.info("Replaced %d existing chunk(s) for %s", deleted, parsed.source_name)

        self.vector_index.add_chunks(chunks)

        return IngestReport(
            pdf_path=path,
            pages=len(parsed.pages),
            chunks=len(chunks),
            llm_enhanced_pages=sum(1 for p in parsed.pages if p.enhanced_by_llm),
        )

    # ----------------------------------------------------------------
    def ingest_many(self, pdf_paths: Iterable[str | Path]) -> list[IngestReport]:
        results: list[IngestReport] = []
        for pdf_path in pdf_paths:
            try:
                results.append(self.ingest_one(pdf_path))
            except Exception as exc:
                logger.exception("Failed to ingest %s: %s", pdf_path, exc)
        logger.info(
            "Ingested %d/%d PDFs",
            len(results),
            len(list(pdf_paths) if isinstance(pdf_paths, list) else results),
        )
        return results
