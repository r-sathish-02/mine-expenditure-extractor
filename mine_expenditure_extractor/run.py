"""
run.py
------
Command-line entry point for the mine expenditure extractor.

Typical workflow
----------------
1. Put one or more PDFs in ``pdfs/``.
2. Run ``python run.py ingest`` to parse + index everything.
3. Run ``python run.py extract`` to produce JSON + markdown reports in
   ``outputs/``.

Subcommands:
    ingest       Parse PDFs, chunk, embed and add to the vector index.
    extract      Identify the primary mine per PDF and pull expenditure.
    process      ingest + extract, in one shot.
    list         List indexed documents.
    clear        Wipe the vector index.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mine_extractor.indexing.vector_index import VectorIndex
from mine_extractor.logging_config import configure_logging, get_logger
from mine_extractor.pipeline.extract_pipeline import ExtractPipeline
from mine_extractor.pipeline.ingest_pipeline import IngestPipeline
from settings import PDF_INPUT_DIR

logger = get_logger(__name__)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def _discover_pdfs(paths: list[str] | None) -> list[Path]:
    """Expand CLI paths into a deduplicated list of PDF files."""
    if not paths:
        found = sorted(PDF_INPUT_DIR.glob("*.pdf"))
        if not found:
            raise FileNotFoundError(f"No PDFs supplied and none found in '{PDF_INPUT_DIR}/'.")
        return found

    out: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            for pdf in sorted(p.glob("*.pdf")):
                if pdf not in seen:
                    out.append(pdf)
                    seen.add(pdf)
        elif p.suffix.lower() == ".pdf" and p.exists():
            if p not in seen:
                out.append(p)
                seen.add(p)
        else:
            logger.warning("Skipping invalid path: %s", raw)
    if not out:
        raise FileNotFoundError("No valid PDF paths resolved from arguments.")
    return out


# ----------------------------------------------------------------
# Subcommand handlers
# ----------------------------------------------------------------
def cmd_ingest(args: argparse.Namespace) -> None:
    pdfs = _discover_pdfs(args.files)
    pipeline = IngestPipeline()
    reports = pipeline.ingest_many(pdfs)

    print("\nIngestion summary")
    print("-" * 60)
    for r in reports:
        print(
            f"  {r.pdf_path.name:<50} pages={r.pages:<4} "
            f"chunks={r.chunks:<4} llm_pages={r.llm_enhanced_pages}"
        )
    print(f"\nTotal indexed chunks: {pipeline.vector_index.count()}\n")


def cmd_extract(args: argparse.Namespace) -> None:
    pipeline = ExtractPipeline()
    sources = pipeline.vector_index.list_sources()

    if args.source:
        if args.source not in sources:
            print(
                f"Error: '{args.source}' is not indexed. Indexed: {sources}",
                file=sys.stderr,
            )
            sys.exit(2)
        results = [pipeline.run_for_source(args.source)]
    else:
        results = pipeline.run_for_all_indexed()

    print("\nExtraction summary")
    print("-" * 60)
    for r in results:
        exp = r.extraction
        cap = exp.capital_costs.total if exp.capital_costs and exp.capital_costs.total else None
        op = (
            exp.operating_costs.total if exp.operating_costs and exp.operating_costs.total else None
        )
        cap_s = f"{cap.amount:,} {cap.units}" if cap and cap.amount is not None else "—"
        op_s = f"{op.amount} {op.units}" if op and op.amount is not None else "—"
        print(f"  {r.source_document}")
        print(f"    selected mine : {r.selected_mine}")
        print(f"    capital total : {cap_s}")
        print(f"    operating tot : {op_s}")
        print(f"    confidence    : {exp.confidence}")
    print()


def cmd_process(args: argparse.Namespace) -> None:
    cmd_ingest(args)
    cmd_extract(args)


def cmd_list(_args: argparse.Namespace) -> None:
    idx = VectorIndex()
    sources = idx.list_sources()
    print(f"\nIndexed documents ({len(sources)}):")
    for s in sources:
        print(f"  - {s}")
    print(f"Total chunks: {idx.count()}\n")


def cmd_clear(_args: argparse.Namespace) -> None:
    confirm = input("This will delete ALL indexed vectors. Type 'yes' to confirm: ")
    if confirm.strip().lower() == "yes":
        VectorIndex().clear()
        print("Vector index cleared.")
    else:
        print("Cancelled.")


# ----------------------------------------------------------------
# Argparse setup
# ----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mine-extractor",
        description="Parse mining technical reports and extract expenditure data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Parse + index PDFs")
    p_ingest.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="PDF files or directories. Defaults to ./pdfs/*.pdf",
    )

    # extract
    p_extract = sub.add_parser("extract", help="Run mine + expenditure extraction")
    p_extract.add_argument(
        "--source",
        default=None,
        help="Run for a single source filename (as indexed). Defaults to all.",
    )

    # process (ingest + extract)
    p_proc = sub.add_parser("process", help="ingest + extract in one go")
    p_proc.add_argument("--files", nargs="+", default=None)
    p_proc.add_argument("--source", default=None)

    # list
    sub.add_parser("list", help="List indexed documents")

    # clear
    sub.add_parser("clear", help="Clear the vector index")

    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "ingest": cmd_ingest,
        "extract": cmd_extract,
        "process": cmd_process,
        "list": cmd_list,
        "clear": cmd_clear,
    }
    try:
        handlers[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Command failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
