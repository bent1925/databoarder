"""The fixed shapes everything passes through.

Think of these as the order form and the receipt: the same fields every time,
so any question and any report can be compared side by side.
"""
from __future__ import annotations

import datetime as _dt
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, field_validator


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


class BeforeAfterSpec(BaseModel):
    """Did a number change after a date? One row per day (or several rows per day).

    The person supplies the date. We never search for the date that looks best,
    because that would manufacture changes out of noise.
    """

    analysis_type: Literal["before_after"] = "before_after"
    date_col: str = Field(description="Column holding the date of each row")
    change_date: str = Field(default="", description="First day of 'after', as YYYY-MM-DD")
    outcome_col: str = Field(description="Column holding the result being measured")
    aggregate: Literal["average", "total"] = Field(
        default="average", description="With several rows per day: use the day's average or the day's total")

    @field_validator("change_date")
    @classmethod
    def _iso(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            return ""
        try:
            return _dt.date.fromisoformat(v[:10]).isoformat()
        except ValueError:
            raise ValueError("change_date must look like 2026-08-30")


class DiDSpec(BaseModel):
    """Before/after, but with a comparison group that never got the change.

    We compare how the gap between the two groups moved. Anything that hit both groups
    (season, news, holidays) cancels out.
    """

    analysis_type: Literal["diff_in_diff"] = "diff_in_diff"
    date_col: str
    change_date: str = ""
    outcome_col: str
    aggregate: Literal["average", "total"] = "average"
    group_col: str = Field(description="Column saying which group each row belongs to")
    treated: str = Field(description="Value for the group that got the change")
    comparison: str = Field(description="Value for the group that never got it")

    @field_validator("change_date")
    @classmethod
    def _iso(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            return ""
        try:
            return _dt.date.fromisoformat(v[:10]).isoformat()
        except ValueError:
            raise ValueError("change_date must look like 2026-08-30")


class ObservationalSpec(BaseModel):
    """Two groups that people sorted themselves into. Compares people who look alike.

    Never read as cause and effect: it can only adjust for columns that are in the file.
    """

    analysis_type: Literal["observational"] = "observational"
    group_col: str
    control: str
    treatment: str
    outcome_col: str
    covariates: list[str] = Field(default_factory=list, description="Columns to make people comparable on")
    assignment: Literal["chose", "unsure"] = "chose"


def _spec_tag(v) -> str:
    if isinstance(v, dict):
        return v.get("analysis_type", "ab_test")
    return getattr(v, "analysis_type", "ab_test")


# Either kind of question. A missing analysis_type means A/B test (how v1 specs were written).
AnySpec = Annotated[
    Union[Annotated[ABTestSpec, Tag("ab_test")], Annotated[BeforeAfterSpec, Tag("before_after")],
          Annotated[DiDSpec, Tag("diff_in_diff")], Annotated[ObservationalSpec, Tag("observational")]],
    Discriminator(_spec_tag),
]


class Interpretation(BaseModel):
    """What the translator returns: a spec plus how it got there."""

    spec: AnySpec
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
    weak: bool = False      # not contradicted, but too uncertain to confirm
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
    analysis: Literal["ab_test", "before_after", "diff_in_diff", "observational"] = "ab_test"
    design: str = ""                   # how people got the change, as the person told us
    adjusted_for: list[str] = []       # observational: columns people were matched up on
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
    outcome_kind: Optional[Literal["rate", "average", "total"]] = None
    outcome_label: str = ""
    cannot_tell: list[str] = []
    # Mode-specific extras
    mode: Mode = "quick"
    caveats: list[str] = []            # quick: listed at the end
    prep: list[str] = []               # data-preparation steps applied before analysis (also in the script)
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
    series: Optional[dict] = None      # before/after: the daily line for the chart
    spec: Optional[AnySpec] = None


# --------------------------------------------------------------------------- the data stage

class _RecipeStep(BaseModel):
    """One fixed step. Steps start as proposed, and nothing runs until the person has confirmed them all."""
    model_config = ConfigDict(extra="forbid")
    status: Literal["proposed", "confirmed"] = "proposed"


class ReadTable(_RecipeStep):
    op: Literal["read_table"] = "read_table"
    file: str
    sheet: Optional[str] = None
    delimiter: str = ","
    header_row: int = Field(ge=0, description="Position of the row holding the column names, counting from 0")
    footer_rows: int = Field(ge=0, description="Rows to leave out at the bottom, such as a total line")
    into: str


class ChooseColumns(_RecipeStep):
    op: Literal["choose_columns"] = "choose_columns"
    table: str
    columns: dict[str, str] = Field(min_length=1, description="Current name -> name to use")


class Stack(_RecipeStep):
    op: Literal["stack"] = "stack"
    tables: list[str] = Field(min_length=1)
    into: str


class CleanKey(_RecipeStep):
    op: Literal["clean_key"] = "clean_key"
    table: str
    column: str
    rule: Literal["digits_only", "trim", "lowercase"]
    into: str


class FilterRows(_RecipeStep):
    op: Literal["filter"] = "filter"
    table: str
    column: str
    test: Literal["equals", "not_equals", "starts_with", "not_starts_with",
                  "greater_than", "at_least", "less_than", "at_most"]
    value: str
    compare_as: Optional[Literal["number", "date"]] = None
    date_order: Optional[Literal["day_first", "month_first"]] = None


class Measure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    how: Literal["sum", "mean", "max", "min", "any_positive", "count_rows"]
    column: Optional[str] = None
    into: str


class RollUp(_RecipeStep):
    op: Literal["roll_up"] = "roll_up"
    table: str
    by: str
    measures: list[Measure] = Field(min_length=1)
    into: str


class MatchTables(_RecipeStep):
    op: Literal["match"] = "match"
    left: str
    right: str
    left_key: str
    right_key: str
    bring: dict[str, str] = Field(min_length=1, description="Column in the right table -> name to use")


class FlagMembership(_RecipeStep):
    op: Literal["flag_membership"] = "flag_membership"
    table: str
    key: str
    other: str
    other_key: str
    into: str
    yes: str
    no: str


class FillMissing(_RecipeStep):
    """Has no default on purpose: what an empty value means is the person's decision."""
    op: Literal["fill_missing"] = "fill_missing"
    table: str
    column: str
    value: str


class Calculate(_RecipeStep):
    op: Literal["calculate"] = "calculate"
    table: str
    first: str
    operation: Literal["multiply", "add", "subtract", "divide"]
    second: str
    into: str


class ReshapeLong(_RecipeStep):
    """Turns one column per date (or per item) into one row per date."""
    op: Literal["reshape_long"] = "reshape_long"
    table: str
    id_columns: list[str] = Field(min_length=1)
    value_columns: list[str] = Field(min_length=1)
    names_to: str
    values_to: str
    into: str


class WindowMeasure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    how: Literal["sum", "count_rows", "any_event"]
    column: Optional[str] = None
    into: str

    @field_validator("column")
    @classmethod
    def _sum_needs_a_column(cls, value, info):
        if info.data.get("how") == "sum" and not value:
            raise ValueError("A sum needs a column to add up.")
        return value


class CountWithinWindow(_RecipeStep):
    """Measures each row's events from its start date. The window length and the no-events value have
    no defaults on purpose: both can change the result, so the person has to state them."""
    op: Literal["count_within_window"] = "count_within_window"
    table: str
    key: str
    start_column: str
    default_start: Optional[str] = Field(default=None, description="Start date for rows without one, "
                                         "such as the campaign date for people who weren't sent anything")
    events: str
    event_key: str
    event_date: str
    days: int = Field(ge=1, description="Calendar days in the window, counting the start day")
    measures: list[WindowMeasure] = Field(min_length=1)
    no_events_value: str = Field(description="Value for a row with no events in its window")
    date_order: Optional[Literal["day_first", "month_first"]] = None


RecipeStep = Annotated[
    Union[ReadTable, ChooseColumns, Stack, CleanKey, FilterRows, RollUp, MatchTables, FlagMembership, FillMissing,
          Calculate, ReshapeLong, CountWithinWindow],
    Field(discriminator="op"),
]


class DataRecipe(BaseModel):
    """How messy files become one analysis table. It is plain data: fixed code carries out each step."""
    model_config = ConfigDict(extra="forbid")
    steps: list[RecipeStep] = Field(min_length=1)
    output: str = Field(description="Name of the table that becomes the analysis table")
