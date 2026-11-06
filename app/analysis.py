"""Picks the right recipe for the question. The front desk: one door, two kitchens."""
from __future__ import annotations

import pandas as pd

from . import ab_test, before_after
from .checks import validate_spec_against_data
from .schemas import Report


def validate_spec(df: pd.DataFrame, spec) -> list[str]:
    if spec.analysis_type == "before_after":
        return before_after.validate_spec_ba(df, spec)
    return validate_spec_against_data(df, spec)


def analyze(df: pd.DataFrame, spec, csv_name: str = "your_file.csv", mode: str = "quick",
            data_sha: str = "") -> Report:
    if spec.analysis_type == "before_after":
        return before_after.analyze(df, spec, csv_name, mode, data_sha)
    return ab_test.analyze(df, spec, csv_name, mode, data_sha)
