"""Picks the right recipe for the question. The front desk: one door, two kitchens."""
from __future__ import annotations

import re

import pandas as pd

from . import ab_test, before_after, did, observational
from .checks import validate_spec_against_data
from .prep import script_block
from .schemas import Report


def validate_spec(df: pd.DataFrame, spec) -> list[str]:
    if spec.analysis_type == "before_after":
        return before_after.validate_spec_ba(df, spec)
    if spec.analysis_type == "diff_in_diff":
        return did.validate_spec_did(df, spec)
    if spec.analysis_type == "observational":
        return observational.validate_spec_obs(df, spec)
    return validate_spec_against_data(df, spec)


def analyze(df: pd.DataFrame, spec, csv_name: str = "your_file.csv", mode: str = "quick",
            data_sha: str = "", prep_steps: list | None = None,
            source: dict | None = None, source_script: str = "") -> Report:
    """`df` is the prepared data; `prep_steps` (from app.prep) are added to the script so it starts from the raw CSV.

    `source` records how a table was built from several files, and `source_script` rebuilds it from them."""
    if spec.analysis_type == "before_after":
        rep = before_after.analyze(df, spec, csv_name, mode, data_sha)
    elif spec.analysis_type == "diff_in_diff":
        rep = did.analyze(df, spec, csv_name, mode, data_sha)
    elif spec.analysis_type == "observational":
        rep = observational.analyze(df, spec, csv_name, mode, data_sha)
    else:
        rep = ab_test.analyze(df, spec, csv_name, mode, data_sha)
    if prep_steps and rep.script:
        rep.prep = [s.text for s in prep_steps]
        rep.script = re.sub(r"^(df = pd\.read_csv\(.*\)\n)", lambda m: m.group(1) + script_block(prep_steps),
                            rep.script, count=1, flags=re.M)
    if source:
        rep.repro = {**rep.repro, "source": source}
        if rep.script:
            rep.script = f"{source_script}\n\n{rep.script}"
    return rep
