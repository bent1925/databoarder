"""Data checks that run BEFORE any analysis.

Like a pre-flight checklist: if something is wrong with the aircraft, we say so
before takeoff instead of after landing.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

from .schemas import ABTestSpec, Check

MIN_PER_ARM_FAIL = 30
MIN_PER_ARM_WARN = 200
ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|^user$|^customer$|^account$)", re.I)


def is_idlike(df: pd.DataFrame, col: str) -> bool:
    """A column that names a person/row rather than describing them."""
    if ID_NAME.search(col):
        return True
    s = df[col]
    return pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) >= 0.9 * max(len(df), 1)


def find_id_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if ID_NAME.search(col) and df[col].nunique(dropna=True) >= 0.5 * max(len(df), 1):
            return col
    return None


def detect_outcome_kind(s: pd.Series) -> str:
    """'binary' if the column holds only two distinct values that look like yes/no."""
    vals = set(pd.Series(s).dropna().unique().tolist())
    if vals and vals <= {0, 1, 0.0, 1.0, True, False}:
        return "binary"
    return "continuous"


def validate_spec_against_data(df: pd.DataFrame, spec: ABTestSpec) -> list[str]:
    """Return a list of problems; empty means the spec fits the data."""
    problems = []
    for col in (spec.group_col, spec.outcome_col):
        if col not in df.columns:
            problems.append(f"Column '{col}' is not in your file.")
    if problems:
        return problems
    levels = set(df[spec.group_col].dropna().astype(str).unique())
    for label in (spec.control, spec.treatment):
        if label not in levels:
            problems.append(f"'{label}' is not a value in column '{spec.group_col}'.")
    if spec.control == spec.treatment:
        problems.append("Control and treatment must be different groups.")
    if spec.outcome_col == spec.group_col:
        problems.append("The outcome and the group column cannot be the same column.")
    if not problems and not pd.api.types.is_numeric_dtype(df[spec.outcome_col]):
        problems.append(f"Outcome column '{spec.outcome_col}' is not numeric.")
    return problems


def run_checks(df: pd.DataFrame, spec: ABTestSpec) -> list[Check]:
    g = df[spec.group_col].astype(str)
    sub = df[g.isin([spec.control, spec.treatment])].copy()
    sub["_g"] = sub[spec.group_col].astype(str)
    checks: list[Check] = []

    # 1. Enough people in each group?
    counts = sub["_g"].value_counts()
    n_c, n_t = int(counts.get(spec.control, 0)), int(counts.get(spec.treatment, 0))
    smallest = min(n_c, n_t)
    if smallest < MIN_PER_ARM_FAIL:
        checks.append(Check(name="Sample size", status="fail",
                            message=f"The smaller group has only {smallest} rows. That is too few to say anything reliable."))
    elif smallest < MIN_PER_ARM_WARN:
        checks.append(Check(name="Sample size", status="warn",
                            message=f"The smaller group has {smallest} rows. Only large differences will be detectable."))
    else:
        checks.append(Check(name="Sample size", status="pass",
                            message=f"{n_c:,} in {spec.control} and {n_t:,} in {spec.treatment}."))

    # 2. Missing outcomes
    missing = int(sub[spec.outcome_col].isna().sum())
    share = missing / max(len(sub), 1)
    if share > 0.10:
        checks.append(Check(name="Missing results", status="warn",
                            message=f"{share:.0%} of rows have no value for '{spec.outcome_col}'. They are left out, which can bias the answer."))
    elif missing:
        checks.append(Check(name="Missing results", status="pass",
                            message=f"{missing:,} rows ({share:.1%}) have no value and were left out."))
    else:
        checks.append(Check(name="Missing results", status="pass", message="No missing values."))

    # 3. Repeat people (only checkable when the file has an ID column)
    idc = find_id_col(sub)
    if idc is None:
        checks.append(Check(name="Repeat people", status="info",
                            message="No ID column found, so we couldn't check whether the same person appears more than once."))
    else:
        repeats = int(sub[idc].duplicated(keep=False).sum())
        both = int((sub.groupby(idc)["_g"].nunique() > 1).sum())
        if both:
            checks.append(Check(name="Repeat people", status="warn",
                                message=f"{both:,} people ('{idc}') appear in both groups, which muddies the comparison."))
        elif repeats / max(len(sub), 1) > 0.01:
            checks.append(Check(name="Repeat people", status="warn",
                                message=f"{repeats:,} rows belong to people who appear more than once. Repeats can make results look more certain than they are."))
        else:
            checks.append(Check(name="Repeat people", status="pass",
                                message=f"Each person appears once (by '{idc}')."))

    # 4. Was the split roughly even? (Sample ratio mismatch against a 50/50 plan.)
    total = n_c + n_t
    if total > 0:
        p = stats.binomtest(n_t, total, 0.5).pvalue
        if p < 0.001:
            checks.append(Check(name="Group balance", status="warn",
                                message=f"Groups are {n_c:,} vs {n_t:,}. If the plan was an even split, something went wrong in assignment and the result may be unreliable."))
        else:
            checks.append(Check(name="Group balance", status="pass",
                                message=f"Groups are {n_c:,} vs {n_t:,}, consistent with an even split."))

    # 5. Outliers (only for continuous outcomes)
    y = sub[spec.outcome_col].dropna()
    if detect_outcome_kind(y) == "continuous" and len(y) > 20:
        skew = float(stats.skew(y))
        if abs(skew) > 3:
            checks.append(Check(name="Extreme values", status="warn",
                                message=f"A few very large values pull the average around (skew {skew:.1f}). The average is less steady than usual."))
        else:
            checks.append(Check(name="Extreme values", status="pass",
                                message="No extreme values dominating the average."))
    return checks


def has_blocking_failure(checks: list[Check]) -> bool:
    return any(c.status == "fail" for c in checks)
