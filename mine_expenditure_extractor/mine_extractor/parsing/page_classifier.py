"""
mine_extractor.parsing.page_classifier
---------------------------------------
Heuristic detection for pages that likely contain structured tabular
data. Used to decide whether to pay the price of LLM table cleanup.

The checks are intentionally cheap and permissive — it is cheaper to
send a false positive to the LLM than to miss a real table.
"""

from __future__ import annotations

import re

_NUMBER_PATTERN = re.compile(r"[-+]?\(?\$?\d[\d,]*(?:\.\d+)?%?\)?")
_TABLE_HINT_WORDS = re.compile(
    r"\btable\s*\d+[\.\-:]|"
    r"\b(capital|operating|opex|capex|cost|expenditure|"
    r"reserves?|resources?|production|throughput|total)\b",
    re.IGNORECASE,
)


def page_has_aligned_numeric_rows(page_text: str, min_rows: int = 3) -> bool:
    """
    Return True if at least ``min_rows`` lines contain multiple numbers
    with consistent spacing — a strong signal for a plain-text table.
    """
    hits = 0
    for line in page_text.splitlines():
        # A qualifying line looks like "Mining equipment   $1,922,582":
        # it has some label text, at least one number, and the number
        # must be substantial (not a page/section index).
        nums = _NUMBER_PATTERN.findall(line)
        if (
            len(nums) >= 1
            and len(line.strip()) > 6
            and re.search(r"[A-Za-z]", line)
            and re.search(r"\d[\d,]{3,}|\d+\.\d+|\$\d", line)
        ):
            hits += 1
            if hits >= min_rows:
                return True
    return False


def page_mentions_tables(page_text: str) -> bool:
    """Heuristic: text contains table/cost vocabulary."""
    return bool(_TABLE_HINT_WORDS.search(page_text))


def is_table_candidate(page_text: str, min_numeric_rows: int = 3) -> bool:
    """Combined heuristic used by the parsing pipeline."""
    if not page_text or len(page_text) < 40:
        return False
    return page_mentions_tables(page_text) and page_has_aligned_numeric_rows(
        page_text, min_rows=min_numeric_rows
    )
