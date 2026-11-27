"""Science mode: the choices a person confirms, and the report that comes back.

Every field that changes the answer is a choice the person makes on screen. The engine gets these as a plain dict.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schemas import Assumption, AuditItem, Check

Assignment = Literal["random", "not_random", "unsure"]


class CompareGroupsSpec(BaseModel):
    """Do the groups differ? Two or more groups of measurements."""
    model_config = ConfigDict(extra="forbid")
    analysis: Literal["compare_groups"] = "compare_groups"
    layout: Literal["wide", "long"] = Field(description="wide: one column per group. long: a column names the group")
    groups: list[str] = Field(min_length=2, description="wide: the group columns. long: the group values. In order")
    group_col: Optional[str] = None
    value_col: Optional[str] = None
    subject_col: Optional[str] = Field(default=None, description="Which subject each row is (needed for long + paired)")
    paired: bool = Field(description="The same subjects were measured in every group")
    method: Literal["means", "ranks"] = Field(description="means: t test / ANOVA. ranks: rank-based tests")
    equal_spread: bool = Field(default=False, description="Assume every group has the same spread (SD)")
    followup: Literal["all_pairs", "vs_control", "none"] = "none"
    control: Optional[str] = None
    subset_col: Optional[str] = Field(default=None, description="Only use rows where this column ...")
    subset_value: Optional[str] = Field(default=None, description="... has this value (such as Day = 12)")
    assignment: Assignment = "unsure"

    @model_validator(mode="after")
    def _fits(self):
        if self.subset_col and self.subset_value in (None, ""):
            raise ValueError(f"Choose which value of '{self.subset_col}' to use.")
        if self.subset_col and self.subset_col in (self.group_col, self.value_col, self.subject_col):
            raise ValueError("The column used to pick rows can't also be the group, measurement or subject column.")
        if self.layout == "long" and not (self.group_col and self.value_col):
            raise ValueError("Choose the column that names the group and the column holding the measurement.")
        if self.followup == "vs_control" and self.control not in self.groups:
            raise ValueError("Choose which group is the control.")
        return self


class ContingencySpec(BaseModel):
    """Are two sets of categories related? Counts in a table."""
    model_config = ConfigDict(extra="forbid")
    analysis: Literal["contingency"] = "contingency"
    layout: Literal["rows", "counts"] = Field(description="rows: one row per subject. counts: a table of counts")
    row_col: Optional[str] = None
    col_col: Optional[str] = None
    label_col: Optional[str] = None
    rows: list[str] = Field(min_length=2, description="The row categories. For a 2x2, the first is compared with the second")
    cols: list[str] = Field(min_length=2, description="The outcome categories (counts layout: the count columns)")
    event: Optional[str] = Field(default=None, description="For a 2x2: the outcome whose risk is compared")
    assignment: Assignment = "unsure"

    @model_validator(mode="after")
    def _fits(self):
        if self.layout == "rows" and not (self.row_col and self.col_col):
            raise ValueError("Choose the two columns to cross.")
        if self.layout == "counts" and not self.label_col:
            raise ValueError("Choose the column that names each row.")
        return self


class CorrelationSpec(BaseModel):
    """Do two measurements move together?"""
    model_config = ConfigDict(extra="forbid")
    analysis: Literal["correlation"] = "correlation"
    x_col: str
    y_col: str
    method: Literal["pearson", "spearman"]


class TimeCourseSpec(BaseModel):
    """How do groups change over time? Each subject is in one group and measured at every time point."""
    model_config = ConfigDict(extra="forbid")
    analysis: Literal["time_course"] = "time_course"
    group_col: str
    time_col: str
    value_col: str
    subject_col: str
    groups: list[str] = Field(min_length=2)
    times: list[str] = Field(min_length=2, description="The time points to use, in order")
    followup: Literal["all_pairs", "vs_control", "none"] = "none"
    control: Optional[str] = None
    assignment: Assignment = "unsure"

    @model_validator(mode="after")
    def _fits(self):
        if self.followup == "vs_control" and self.control not in self.groups:
            raise ValueError("Choose which group is the control.")
        return self


ScienceSpec = Annotated[Union[CompareGroupsSpec, ContingencySpec, CorrelationSpec, TimeCourseSpec],
                        Field(discriminator="analysis")]


class Finding(BaseModel):
    """One row of the results table, as Prism would list it."""
    label: str
    value: str


class ScienceReport(BaseModel):
    analysis: str
    verdict: Literal["clear_difference", "no_clear_difference", "blocked"]
    headline: str
    answer: str
    how_sure: str
    causal_label: str = ""
    readback: str = ""
    checks: list[Check] = []
    assumptions: list[Assumption] = []
    method_name: str = ""
    method_why: str = ""
    results: list[Finding] = []         # the main test and effect sizes, Prism-style
    result: dict = {}                   # everything the engine returned (groups, pairs, counts, points)
    cannot_tell: list[str] = []
    audit: list[AuditItem] = []
    prep: list[str] = []
    repro: dict = {}
    script: str = ""
    data_name: str = ""
    spec: Optional[dict] = None
