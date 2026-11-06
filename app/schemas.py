"""The fixed shapes everything passes through.

Think of these as the order form and the receipt: the same fields every time,
so any question and any report can be compared side by side.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ABTestSpec(BaseModel):
    """A complete, machine-checkable description of one A/B test analysis.

    The language model's only job is to fill this in. It cannot add fields or
    pick a method that is not in the catalog.
    """

    analysis_type: Literal["ab_test"] = "ab_test"
    group_col: str = Field(description="Column saying which version each row saw")
    control: str = Field(description="Value in group_col for the baseline version")
    treatment: str = Field(description="Value in group_col for the new version")
    outcome_col: str = Field(description="Column holding the result being measured")


class Interpretation(BaseModel):
    """What the translator returns: a spec plus how it got there."""

    spec: ABTestSpec
    readback: str  # plain-English restatement, built from the spec by code
    source: Literal["llm", "heuristic"]
    notes: list[str] = []


Mode = Literal["quick", "right_answer", "research"]


class Check(BaseModel):
    name: str
    status: Literal["pass", "warn", "fail", "info"]
    message: str


class Robustness(BaseModel):
    """One independent cross-check of the main result."""

    key: str = ""
    name: str
    description: str
    result: str
    agrees: bool
    p_value: float = 1.0


class Segment(BaseModel):
    """The same comparison, run inside one slice of the data (e.g. country = UK)."""

    column: str
    level: str
    n_control: int
    n_treatment: int
    diff: float
    ci_low: float
    ci_high: float
    p_raw: float
    p_adj: float
    flagged: bool


class Assumption(BaseModel):
    """Something that must be true for a result to hold.

    ok     = we checked it in the data and it looks fine
    warn   = we checked it and there is a concern
    manual = the data can't tell us; the person has to confirm it
    """

    text: str
    status: Literal["ok", "warn", "manual"]


class MethodInfo(BaseModel):
    name: str
    assumptions: list[Assumption]


class AuditItem(BaseModel):
    """One number in the report: where it came from and what it means."""

    label: str
    value: str
    method: str
    code: str
    technical: str = ""   # the textbook-correct definition
    plain: str = ""       # what it means in practice


class ArmSummary(BaseModel):
    name: str
    n: int
    estimate: float
    ci_low: float
    ci_high: float


class Report(BaseModel):
    # Layer 1: the answer
    headline: str
    answer: str
    how_sure: str
    verdict: Literal["clear_difference", "no_clear_difference", "inconclusive", "blocked"]
    causal_label: str
    # Layer 2: the evidence
    checks: list[Check]
    arms: list[ArmSummary] = []
    effect: Optional[dict] = None
    outcome_kind: Optional[Literal["rate", "average"]] = None
    outcome_label: str = ""
    cannot_tell: list[str] = []
    # Mode-specific extras
    mode: Mode = "quick"
    caveats: list[str] = []            # quick: listed at the end
    review_items: list[str] = []       # right_answer: things to resolve before relying on it
    robustness: list[Robustness] = []  # right_answer: independent cross-checks
    robust_summary: str = ""
    segments: list[Segment] = []       # research: subgroup scan
    segment_note: str = ""
    # Layer 3: the method
    method_summary: str = ""
    methods: list[MethodInfo] = []     # assumptions, grouped by method
    repro: dict = {}                   # data fingerprint, library versions, seed
    audit: list[AuditItem] = []
    script: str = ""
    spec: Optional[ABTestSpec] = None
