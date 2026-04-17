"""
mine_extractor.parsing.pdf_to_markdown
---------------------------------------
Convert a PDF into a structured markdown document optimised for
downstream chunking and retrieval.

Pipeline per page:
    1. Pull layout-preserving text with PyMuPDF  (very fast).
    2. If the page looks tabular, optionally ask an LLM to turn aligned
       columns into proper markdown tables.
    3. Wrap every page in a ``## Page N`` heading so retrieval results
       can be traced back to page numbers.

Why PyMuPDF instead of Marker?
    * ~20-50x faster on typical technical reports.
    * No GPU requirement.
    * No large model downloads.
    * Loses fancy figure-captioning but preserves all text + spatial layout,
      which is what matters for structured data extraction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz  # PyMuPDF

from mine_extractor.logging_config import get_logger
from mine_extractor.parsing.page_classifier import is_table_candidate
from mine_extractor.parsing.table_enhancer import TableAwareMarkdownEnhancer
from settings import CACHE_DIR, settings

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------
@dataclass
class ParsedPage:
    page_number: int
    markdown: str
    enhanced_by_llm: bool


@dataclass
class ParsedDocument:
    source_name: str
    source_path: str
    pages: list[ParsedPage]

    @property
    def full_markdown(self) -> str:
        parts: list[str] = []
        for page in self.pages:
            parts.append(f"## Page {page.page_number}\n")
            parts.append(page.markdown.rstrip())
            parts.append("")  # blank line between pages
        return "\n".join(parts).strip() + "\n"


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------
class PdfMarkdownParser:
    """
    Convert PDFs to markdown with optional LLM table cleanup.

    Parameters
    ----------
    use_llm_for_tables:
        When True, pages flagged as tabular by the heuristic classifier
        are reformatted by an LLM. Defaults to settings value.
    cache_dir:
        Parsed pages are cached on disk (keyed by PDF content hash) so
        re-ingestion is essentially free for unchanged files.
    """

    def __init__(
        self,
        *,
        use_llm_for_tables: bool | None = None,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        self.use_llm_for_tables = (
            settings.enable_llm_table_enhancement
            if use_llm_for_tables is None
            else use_llm_for_tables
        )
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._enhancer: TableAwareMarkdownEnhancer | None = None  # lazy

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------
    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """Parse ``pdf_path`` into a :class:`ParsedDocument`."""
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        cached = self._load_cache(path)
        if cached is not None:
            logger.info("Using cached parse of %s", path.name)
            return cached

        logger.info("Parsing %s", path.name)
        doc = fitz.open(str(path))
        pages: list[ParsedPage] = []
        llm_calls = 0

        try:
            for idx, page in enumerate(doc, start=1):
                raw = self._extract_page_text(page)
                use_llm = self.use_llm_for_tables and is_table_candidate(
                    raw, min_numeric_rows=settings.table_heuristic_min_numeric_rows
                )

                if use_llm:
                    if self._enhancer is None:
                        self._enhancer = TableAwareMarkdownEnhancer()
                    md = self._enhancer.enhance(raw, page_number=idx)
                    llm_calls += 1
                    logger.debug("Page %d: enhanced by LLM", idx)
                else:
                    md = raw

                pages.append(ParsedPage(page_number=idx, markdown=md, enhanced_by_llm=use_llm))
        finally:
            doc.close()

        logger.info(
            "Parsed %d pages from %s (LLM calls: %d)",
            len(pages),
            path.name,
            llm_calls,
        )

        parsed = ParsedDocument(source_name=path.name, source_path=str(path), pages=pages)
        self._save_cache(path, parsed)
        return parsed

    # ----------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------
    @staticmethod
    def _extract_page_text(page: fitz.Page) -> str:
        """
        Extract text while preserving approximate spatial layout.

        PyMuPDF's ``blocks`` mode returns a list of text blocks with
        bounding boxes. We sort them top-to-bottom, left-to-right and
        join with newlines — this keeps multi-column and column-aligned
        tables readable to downstream consumers.
        """
        # Using "blocks" instead of "text" yields better structure for
        # side-by-side table cells and multi-column layouts.
        blocks = page.get_text("blocks") or []
        # Each block is (x0, y0, x1, y1, text, block_no, block_type).
        text_blocks = [b for b in blocks if len(b) >= 5 and isinstance(b[4], str)]
        text_blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
        chunks = [b[4].rstrip() for b in text_blocks if b[4].strip()]
        raw = "\n".join(chunks)
        # Collapse excessive blank lines but keep paragraph breaks.
        lines = [ln.rstrip() for ln in raw.splitlines()]
        cleaned: list[str] = []
        blank = 0
        for ln in lines:
            if not ln.strip():
                blank += 1
                if blank > 1:
                    continue
            else:
                blank = 0
            cleaned.append(ln)
        return "\n".join(cleaned).strip()

    # ---------------- cache helpers ---------------------------------
    def _cache_key(self, pdf_path: Path) -> str:
        """Stable cache key derived from path, size and mtime."""
        stat = pdf_path.stat()
        seed = f"{pdf_path.name}|{stat.st_size}|{int(stat.st_mtime)}"
        seed += f"|llm={self.use_llm_for_tables}"
        return hashlib.sha1(seed.encode()).hexdigest()

    def _cache_file(self, pdf_path: Path) -> Path:
        return self._cache_dir / f"{self._cache_key(pdf_path)}.json"

    def _load_cache(self, pdf_path: Path) -> ParsedDocument | None:
        cache_file = self._cache_file(pdf_path)
        if not cache_file.exists():
            return None
        try:
            with cache_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            pages = [ParsedPage(**p) for p in data["pages"]]
            return ParsedDocument(
                source_name=data["source_name"],
                source_path=data["source_path"],
                pages=pages,
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Ignoring invalid cache file %s: %s", cache_file, exc)
            return None

    def _save_cache(self, pdf_path: Path, parsed: ParsedDocument) -> None:
        cache_file = self._cache_file(pdf_path)
        try:
            data = {
                "source_name": parsed.source_name,
                "source_path": parsed.source_path,
                "pages": [asdict(p) for p in parsed.pages],
            }
            with cache_file.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
        except OSError as exc:
            logger.warning("Could not write cache file %s: %s", cache_file, exc)
