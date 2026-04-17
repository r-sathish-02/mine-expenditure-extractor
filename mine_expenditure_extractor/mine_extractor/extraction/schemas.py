"""
mine_extractor.extraction.schemas
----------------------------------
Pydantic models that define the structured output we expect from the
LLM extraction stage. Using Pydantic gives us automatic validation,
clean JSON serialisation, and a natural place to document each field.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ----------------------------------------------------------------
# Mine identification
# ----------------------------------------------------------------
class MineCandidate(BaseModel):
    """A mine / operation / project mentioned in a document."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Full name of the mine/operation as it appears")
    location: str | None = Field(
        default=None, description="Geographic location (state/province, country)"
    )
    owner_operator: str | None = Field(default=None, description="Operating company, if identified")
    commodity: str | None = Field(
        default=None, description="Primary commodity (coal, gold, copper, ...)"
    )
    mentions: int | None = Field(
        default=None,
        description="Rough count of how often this mine is referenced",
    )


class MineList(BaseModel):
    """Top-level response from the mine identification step."""

    model_config = ConfigDict(extra="ignore")

    candidates: list[MineCandidate] = Field(default_factory=list)
    primary_mine: str = Field(description="Name of the primary mine the report is about")
    reasoning: str | None = Field(
        default=None, description="Short rationale for picking the primary mine"
    )


# ----------------------------------------------------------------
# Expenditure extraction
# ----------------------------------------------------------------
class CostItem(BaseModel):
    """A single row in a cost breakdown."""

    model_config = ConfigDict(extra="ignore")

    category: str = Field(description="Name of the cost category")
    amount: float | None = Field(default=None, description="Numeric amount, parsed from the source")
    units: str | None = Field(
        default=None,
        description="Units as stated in the source, e.g. 'thousand CAD', 'USD/t'",
    )
    period: str | None = Field(
        default=None,
        description="Time period the number covers if mentioned (e.g. 'LOM 2020-2065')",
    )
    notes: str | None = Field(default=None, description="Anything relevant the reader should know")


class CostSummary(BaseModel):
    """Either a capital or operating cost block."""

    model_config = ConfigDict(extra="ignore")

    total: CostItem | None = Field(default=None, description="The reported total, if any")
    breakdown: list[CostItem] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)


class MineExpenditure(BaseModel):
    """Structured expenditure extraction for a single mine."""

    model_config = ConfigDict(extra="ignore")

    mine_name: str
    location: str | None = None
    owner_operator: str | None = None
    report_type: str | None = Field(
        default=None,
        description="e.g. 'NI 43-101 Technical Report', 'JORC Report', ...",
    )
    currency: str | None = Field(
        default=None,
        description="Primary currency used in the cost figures (CAD, USD, AUD, ...)",
    )
    capital_costs: CostSummary | None = None
    operating_costs: CostSummary | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
    caveats: str | None = Field(
        default=None,
        description="Anything that makes the numbers uncertain "
        "(missing figures, mixed units, forecast vs actual, etc.)",
    )
    source_document: str | None = None


# ----------------------------------------------------------------
# Per-document result bundle (what we save to disk)
# ----------------------------------------------------------------
class DocumentResult(BaseModel):
    """The final, per-PDF output blob."""

    model_config = ConfigDict(extra="ignore")

    source_document: str
    mines_identified: list[MineCandidate] = Field(default_factory=list)
    selected_mine: str
    extraction: MineExpenditure
    retrieval_snippets: list[dict] = Field(
        default_factory=list,
        description="The retrieved chunks passed to the LLM, for audit.",
    )
