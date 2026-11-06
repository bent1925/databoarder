"""Data checks that run BEFORE any analysis.

Like a pre-flight checklist: if something is wrong, we say so before takeoff.
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
    """'binary' if the column holds only 0/1 (yes/no) values."""
    vals = set(pd.Series(s).dropna().unique().tolist())
    if vals and vals <= {0, 1, 0.0, 1.0, True, False}:
        return "binary"
    return "continuous"


def validate_spec_against_data(df: pd.DataFrame, spec: ABTestSpec) -> list[str]:
    """Return a list of problems; empty means the spec fits the data."""
    problems = []
    for col in (spec.group_col, spec.outcome_col):
        if col not in df.columns:
            problems.append(f"Column '{col}' isn't in your file.")
    if problems:
        return problems
    levels = set(df[spec.group_col].dropna().astype(str).unique())
    for label in (spec.control, spec.treatment):
        if label not in levels:
            problems.append(f"'{label}' isn't a value of '{spec.group_col}'.")
    if spec.control == spec.treatment:
        problems.append("Pick two different groups.")
    if spec.outcome_col == spec.group_col:
        problems.append("The outcome can't be the group column.")
    if not problems and not pd.api.types.is_numeric_dtype(df[spec.outcome_col]):
        problems.append(f"'{spec.outcome_col}' isn't a number.")
    return problems


def run_checks(df: pd.DataFrame, spec: ABTestSpec) -> list[Check]:
    g = df[spec.group_col].astype(str)
    sub_all = df[g.isin([spec.control, spec.treatment])].copy()
    sub_all["_g"] = sub_all[spec.group_col].astype(str)
    sub = sub_all.dropna(subset=[spec.outcome_col])  # rows the analysis will actually use
    checks: list[Check] = []

    # 1. Enough rows in each group?
    counts = sub["_g"].value_counts()
    n_c, n_t = int(counts.get(spec.control, 0)), int(counts.get(spec.treatment, 0))
    smallest = min(n_c, n_t)
    if smallest < MIN_PER_ARM_FAIL:
        checks.append(Check(name="Sample size", status="fail",
                            message=f"Smallest group has {smallest} rows. Too few."))
    elif smallest < MIN_PER_ARM_WARN:
        checks.append(Check(name="Sample size", status="warn",
                            message=f"Smallest group has {smallest} rows. Only big gaps will show."))
    else:
        checks.append(Check(name="Sample size", status="pass", message=f"{n_c:,} vs {n_t:,} usable rows."))

    # 2. Missing results
    missing = int(sub_all[spec.outcome_col].isna().sum())
    share = missing / max(len(sub_all), 1)
    if share > 0.10:
        checks.append(Check(name="Missing results", status="warn",
                            message=f"{share:.0%} missing. Left out; may skew the answer."))
    elif missing:
        checks.append(Check(name="Missing results", status="pass",
                            message=f"{missing:,} missing ({share:.1%}). Left out."))
    else:
        checks.append(Check(name="Missing results", status="pass", message="None missing."))

    # 2b. Is the missing spread evenly across groups?
    if missing:
        tot = sub_all["_g"].value_counts()
        tc, tt = int(tot.get(spec.control, 0)), int(tot.get(spec.treatment, 0))
        mc = int(sub_all.loc[sub_all["_g"] == spec.control, spec.outcome_col].isna().sum())
        mt = int(sub_all.loc[sub_all["_g"] == spec.treatment, spec.outcome_col].isna().sum())
        if tc and tt:
            p = float(stats.fisher_exact([[mc, tc - mc], [mt, tt - mt]])[1])
            if p < 0.01:
                checks.append(Check(name="Missing balance", status="warn",
                                    message=f"Missing differs by group ({mc / tc:.1%} vs {mt / tt:.1%}). May skew the answer."))
            else:
                checks.append(Check(name="Missing balance", status="pass",
                                    message="Missing results are spread evenly."))

    # 3. Repeat people (only checkable with an ID column)
    idc = find_id_col(sub)
    if idc is None:
        checks.append(Check(name="Repeat people", status="info",
                            message="No ID column. Can't check for repeats."))
    else:
        repeats = int(sub[idc].duplicated(keep=False).sum())
        both = int((sub.groupby(idc)["_g"].nunique() > 1).sum())
        if both:
            checks.append(Check(name="Repeat people", status="warn",
                                message=f"{both:,} people ('{idc}') are in both groups."))
        elif repeats / max(len(sub), 1) > 0.01:
            checks.append(Check(name="Repeat people", status="warn",
                                message=f"{repeats:,} rows are repeat people. Can overstate certainty."))
        else:
            checks.append(Check(name="Repeat people", status="pass", message=f"Everyone appears once ('{idc}')."))

    # 4. Was the split roughly even? (against a 50/50 plan)
    total = n_c + n_t
    if total > 0:
        p = stats.binomtest(n_t, total, 0.5).pvalue
        if p < 0.001:
            checks.append(Check(name="Group balance", status="warn",
                                message=f"Uneven split ({n_c:,} vs {n_t:,}). If 50/50 was planned, assignment may have gone wrong."))
        else:
            checks.append(Check(name="Group balance", status="pass", message="Split looks even."))

    # 5. Extreme values (continuous outcomes only)
    y = sub[spec.outcome_col]
    if detect_outcome_kind(y) == "continuous" and len(y) > 20:
        skew = float(stats.skew(y))
        if abs(skew) > 3:
            checks.append(Check(name="Extreme values", status="warn",
                                message=f"A few huge values pull the average (skew {skew:.1f})."))
        else:
            checks.append(Check(name="Extreme values", status="pass", message="No huge values dominate."))
    return checks


def has_blocking_failure(checks: list[Check]) -> bool:
    return any(c.status == "fail" for c in checks)
