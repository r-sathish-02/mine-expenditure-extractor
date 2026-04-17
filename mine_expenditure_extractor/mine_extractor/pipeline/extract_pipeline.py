"""
mine_extractor.pipeline.extract_pipeline
-----------------------------------------
Per-document extraction: identify the primary mine, then pull its
capital and operating cost breakdown.

Results are persisted as JSON (for downstream systems) *and* as a
human-readable markdown report in ``outputs/``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from mine_extractor.extraction.cost_extractor import CostExtractor
from mine_extractor.extraction.mine_detector import MineDetector
from mine_extractor.extraction.schemas import DocumentResult
from mine_extractor.indexing.vector_index import VectorIndex
from mine_extractor.llm_client import GroqClient
from mine_extractor.logging_config import get_logger
from mine_extractor.retrieval.searcher import Searcher
from settings import OUTPUT_DIR

logger = get_logger(__name__)


class ExtractPipeline:
    """Run mine detection + cost extraction for one or many documents."""

    def __init__(
        self,
        vector_index: VectorIndex | None = None,
        searcher: Searcher | None = None,
        llm: GroqClient | None = None,
        output_dir: Path = OUTPUT_DIR,
    ) -> None:
        self.vector_index = vector_index or VectorIndex()
        self.searcher = searcher or Searcher(self.vector_index)
        self.llm = llm or GroqClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mine_detector = MineDetector(searcher=self.searcher, llm=self.llm)
        self.cost_extractor = CostExtractor(searcher=self.searcher, llm=self.llm)

    # ----------------------------------------------------------------
    def run_for_source(self, source_name: str) -> DocumentResult:
        logger.info("── Extraction start: %s ──", source_name)
        mine_list = self.mine_detector.detect(source_name)

        primary = mine_list.primary_mine or (
            mine_list.candidates[0].name if mine_list.candidates else "unknown"
        )

        expenditure, snippets = self.cost_extractor.extract(source_name, primary)
        expenditure.source_document = source_name
        # Prefer the richer location/operator info from the mine detector
        # when the cost extractor did not populate them.
        for cand in mine_list.candidates:
            if cand.name.lower() == primary.lower():
                expenditure.location = expenditure.location or cand.location
                expenditure.owner_operator = expenditure.owner_operator or cand.owner_operator
                break

        result = DocumentResult(
            source_document=source_name,
            mines_identified=mine_list.candidates,
            selected_mine=primary,
            extraction=expenditure,
            retrieval_snippets=snippets,
        )

        self._persist(result)
        logger.info("── Extraction done: %s ──", source_name)
        return result

    # ----------------------------------------------------------------
    def run_for_all_indexed(self) -> list[DocumentResult]:
        sources = self.vector_index.list_sources()
        if not sources:
            logger.warning("No documents indexed — nothing to extract.")
            return []
        logger.info("Running extraction across %d indexed PDFs", len(sources))
        return [self.run_for_source(s) for s in sources]

    # ----------------------------------------------------------------
    def _persist(self, result: DocumentResult) -> None:
        stem = Path(result.source_document).stem
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        json_path = self.output_dir / f"{stem}.json"
        md_path = self.output_dir / f"{stem}.md"

        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(
                result.model_dump(),
                fh,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        md_path.write_text(self._render_markdown(result, ts), encoding="utf-8")

        logger.info("Wrote %s and %s", json_path.name, md_path.name)

    # ----------------------------------------------------------------
    @staticmethod
    def _render_markdown(result: DocumentResult, timestamp: str) -> str:
        exp = result.extraction
        lines: list[str] = []
        lines.append(f"# Expenditure Extraction — {exp.mine_name}")
        lines.append("")
        lines.append(f"- **Source document:** `{result.source_document}`")
        lines.append(f"- **Generated:** {timestamp}")
        if exp.location:
            lines.append(f"- **Location:** {exp.location}")
        if exp.owner_operator:
            lines.append(f"- **Operator:** {exp.owner_operator}")
        if exp.report_type:
            lines.append(f"- **Report type:** {exp.report_type}")
        if exp.currency:
            lines.append(f"- **Currency:** {exp.currency}")
        lines.append(f"- **Confidence:** {exp.confidence}")
        if exp.caveats:
            lines.append(f"- **Caveats:** {exp.caveats}")
        lines.append("")

        if result.mines_identified:
            lines.append("## Mines Identified")
            lines.append("")
            lines.append("| Name | Location | Operator | Commodity |")
            lines.append("|---|---|---|---|")
            for m in result.mines_identified:
                lines.append(
                    f"| {m.name} | {m.location or ''} | {m.owner_operator or ''} | {m.commodity or ''} |"
                )
            lines.append("")
            lines.append(f"**Selected primary mine:** {result.selected_mine}")
            lines.append("")

        def _render_block(title: str, summary) -> None:
            if summary is None:
                return
            lines.append(f"## {title}")
            lines.append("")
            if summary.total:
                t = summary.total
                amt = "—" if t.amount is None else f"{t.amount:,}"
                units = t.units or ""
                period = f" ({t.period})" if t.period else ""
                lines.append(f"**Total:** {amt} {units}{period}")
                if t.notes:
                    lines.append(f"_Notes: {t.notes}_")
                lines.append("")
            if summary.breakdown:
                lines.append("| Category | Amount | Units | Period |")
                lines.append("|---|---:|---|---|")
                for item in summary.breakdown:
                    amt = "" if item.amount is None else f"{item.amount:,}"
                    lines.append(
                        f"| {item.category} | {amt} | {item.units or ''} | {item.period or ''} |"
                    )
                lines.append("")
            if summary.source_pages:
                pages = ", ".join(str(p) for p in summary.source_pages)
                lines.append(f"_Source pages: {pages}_")
                lines.append("")

        _render_block("Capital Costs", exp.capital_costs)
        _render_block("Operating Costs", exp.operating_costs)

        return "\n".join(lines).rstrip() + "\n"
