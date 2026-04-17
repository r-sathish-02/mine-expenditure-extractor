"""
mine_extractor.indexing.chunker
-------------------------------
Split parsed markdown into retrieval chunks.

The splitter is markdown-aware: it tries to keep tables intact (never
splitting mid-row) and respects the ``## Page N`` markers inserted by
the parser so each chunk retains the page number it came from.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from mine_extractor.logging_config import get_logger
from mine_extractor.parsing.pdf_to_markdown import ParsedDocument
from settings import settings

logger = get_logger(__name__)

_PAGE_HEADER_RE = re.compile(r"^##\s*Page\s+(\d+)\s*$", re.MULTILINE)
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class Chunk:
    text: str
    metadata: dict

    def to_dict(self) -> dict:
        return {"text": self.text, "metadata": self.metadata}


class MarkdownChunker:
    """
    Split a :class:`ParsedDocument` into small text chunks, each tagged
    with the source filename, page number and a local chunk index.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Separators prioritise markdown structure, then paragraphs, then
        # sentences. Empty string is the final fallback.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    # ----------------------------------------------------------------
    def split(self, doc: ParsedDocument) -> list[Chunk]:
        """Split a parsed document into chunks with rich metadata."""
        chunks: list[Chunk] = []

        # Work page by page so the page number metadata stays accurate.
        for page in doc.pages:
            page_chunks = self._split_one_page(page.markdown)
            for i, text in enumerate(page_chunks):
                chunks.append(
                    Chunk(
                        text=text,
                        metadata={
                            "source": doc.source_name,
                            "page": page.page_number,
                            "chunk_index": i,
                            "enhanced_by_llm": page.enhanced_by_llm,
                        },
                    )
                )

        logger.info(
            "Chunked %s into %d chunk(s) across %d page(s)",
            doc.source_name,
            len(chunks),
            len(doc.pages),
        )
        return chunks

    # ----------------------------------------------------------------
    def _split_one_page(self, md: str) -> list[str]:
        """
        Split a single page of markdown while keeping tables whole.

        Markdown tables (contiguous runs of lines starting with ``|``) are
        treated as atomic blocks so they survive retrieval intact.
        """
        if not md.strip():
            return []

        blocks = list(self._segment_blocks(md))

        pieces: list[str] = []
        for block_type, block_text in blocks:
            if block_type == "table":
                pieces.append(block_text)
            else:
                pieces.extend(self._splitter.split_text(block_text))
        # Drop whitespace-only pieces
        return [p.strip() for p in pieces if p.strip()]

    @staticmethod
    def _segment_blocks(md: str) -> Iterable[tuple[str, str]]:
        """Yield (block_type, text) segments — block_type is 'table' or 'text'."""
        lines = md.splitlines()
        i = 0
        buf: list[str] = []
        mode = "text"

        def flush(current_mode: str) -> tuple[str, str] | None:
            if buf:
                return current_mode, "\n".join(buf).strip()
            return None

        while i < len(lines):
            line = lines[i]
            if _TABLE_LINE_RE.match(line):
                if mode != "table":
                    out = flush(mode)
                    if out:
                        yield out
                    buf.clear()
                    mode = "table"
                buf.append(line)
            else:
                if mode == "table":
                    out = flush(mode)
                    if out:
                        yield out
                    buf.clear()
                    mode = "text"
                buf.append(line)
            i += 1

        out = flush(mode)
        if out:
            yield out
