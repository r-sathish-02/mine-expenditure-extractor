"""
mine_extractor.extraction.prompts
----------------------------------
Prompt templates used by the extraction pipeline.

Kept in a single module so prompt engineering is easy to iterate on
without touching pipeline logic.
"""

from __future__ import annotations

# ============================================================
# Mine identification
# ============================================================
MINE_IDENTIFIER_SYSTEM = """\
You are an expert mining-industry analyst. Your job is to read excerpts \
from a technical report and identify every mine, operation, project or \
deposit that is described in it.

Rules:
- Return ONLY valid JSON matching the requested schema. No prose, no code fences.
- If two names refer to the same asset (e.g. "Greenhills Coal Operation" \
and "GHO"), pick the more formal full name.
- If the excerpts mention related sister operations only in passing \
(e.g. nearby mines owned by the same company), list them as candidates \
but do NOT pick them as the primary mine.
- The "primary_mine" is the asset the report is fundamentally about.
"""

MINE_IDENTIFIER_USER_TEMPLATE = """\
Below are excerpts retrieved from a technical report titled `{source_name}`.

{context}

Return JSON with this exact shape:
{{
  "candidates": [
    {{
      "name": "string",
      "location": "string or null",
      "owner_operator": "string or null",
      "commodity": "string or null",
      "mentions": integer or null
    }}
  ],
  "primary_mine": "string",
  "reasoning": "short string (1-2 sentences) explaining your pick"
}}
"""


# ============================================================
# Expenditure extraction
# ============================================================
COST_EXTRACTOR_SYSTEM = """\
You are a meticulous mining-finance analyst. You extract capital and \
operating cost figures from technical report excerpts and return them \
as strict JSON.

Hard rules:
- Output ONLY JSON matching the schema. No narration, no code fences.
- Preserve numbers exactly as printed. If a figure is in thousands or \
millions, keep it that way and record the unit string verbatim.
- Never fabricate a number. If a category is not reported, omit it from \
the breakdown — do NOT invent zeros.
- If the report gives both per-tonne operating costs and a total, record \
the per-tonne figure as the total and note the unit in the `units` field.
- If only partial information is available set `confidence` to "low" and \
describe the gap in `caveats`.
- Cite the page numbers where you found the totals in `source_pages`.
"""

COST_EXTRACTOR_USER_TEMPLATE = """\
Document: `{source_name}`
Target mine: `{mine_name}`

Relevant excerpts retrieved from the document (each is tagged with its \
source filename and page number):

{context}

Extract the capital and operating cost information for the target mine. \
Return JSON with this schema:

{{
  "mine_name": "string",
  "location": "string or null",
  "owner_operator": "string or null",
  "report_type": "string or null",
  "currency": "string or null",
  "capital_costs": {{
     "total":     {{ "category": "Total capital", "amount": number or null, "units": "string or null", "period": "string or null", "notes": "string or null" }} or null,
     "breakdown": [ {{ "category": "...", "amount": number or null, "units": "...", "period": "...", "notes": "..." }} ],
     "source_pages": [ integer, ... ]
  }},
  "operating_costs": {{
     "total":     {{ "category": "Total operating", "amount": number or null, "units": "string", "period": "string or null", "notes": "string or null" }} or null,
     "breakdown": [ {{ "category": "...", "amount": number or null, "units": "...", "period": "...", "notes": "..." }} ],
     "source_pages": [ integer, ... ]
  }},
  "confidence": "high" | "medium" | "low",
  "caveats": "string or null"
}}
"""


def format_snippets(snippets: list[dict]) -> str:
    """Render retrieved snippets as a numbered, citation-friendly block."""
    lines: list[str] = []
    for i, snip in enumerate(snippets, start=1):
        src = snip.get("source", "?")
        page = snip.get("page", "?")
        text = (snip.get("text") or "").strip()
        lines.append(f"--- Snippet {i} | source={src} | page={page} ---")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()
