"""Gets a file ready to analyze. Every change is a plain line of code that also goes at the top of the
reproduction script, so the downloaded script cleans the raw CSV exactly as the app did.

Rules: check first, fix only what is unambiguous, say what changed, never touch the original file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

YES = {"yes", "y", "true", "t"}
NO = {"no", "n", "false", "f"}
_MONEYISH = re.compile(r"[$,%\s]")


@dataclass
class Step:
    text: str          # plain English, shown to the person
    code: str          # runs on `df`, goes into the script


def _numeric_text(s: pd.Series) -> bool:
    """A text column whose values are really numbers like '$1,200' or '12%'."""
    vals = s.dropna().astype(str)
    if len(vals) < 5 or not vals.str.contains(r"\d").all():
        return False
    parsed = pd.to_numeric(vals.str.replace(_MONEYISH, "", regex=True), errors="coerce")
    plain = pd.to_numeric(vals, errors="coerce")
    return bool(parsed.notna().mean() >= 0.95 and plain.isna().mean() > 0.5)   # has $ , % or spaces to remove


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Step], list[str]]:
    """Returns (clean copy, applied steps, things only the person can decide)."""
    df = df.copy()
    steps: list[Step] = []
    notes: list[str] = []

    tidy = {c: str(c).strip() for c in df.columns if str(c) != str(c).strip()}
    if tidy:
        df = df.rename(columns=tidy)
        steps.append(Step(f"Trimmed stray spaces from {len(tidy)} column name{'s' if len(tidy) > 1 else ''}.",
                          "df.columns = [str(c).strip() for c in df.columns]"))

    text_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
    padded = [c for c in text_cols if df[c].dropna().astype(str).map(lambda v: v != v.strip()).any()]
    for c in padded:
        df[c] = df[c].where(df[c].isna(), df[c].astype(str).str.strip())
        steps.append(Step(f"'{c}': trimmed spaces around values.",
                          f"df[{c!r}] = df[{c!r}].where(df[{c!r}].isna(), df[{c!r}].astype(str).str.strip())"))

    for c in text_cols:
        s = df[c]
        if _numeric_text(s):
            df[c] = pd.to_numeric(s.astype(str).str.replace(_MONEYISH, "", regex=True), errors="coerce")
            steps.append(Step(f"'{c}': turned text like '$1,200' into numbers.",
                              f"df[{c!r}] = pd.to_numeric(df[{c!r}].astype(str).str.replace(r'[$,%\\s]', '', regex=True), errors='coerce')"))
            continue
        low = s.dropna().astype(str).str.lower()
        levels = set(low.unique())
        if levels and levels <= (YES | NO) and levels & YES and levels & NO and \
                _looks_like_outcome(c, df):
            m = {k: 1 for k in YES} | {k: 0 for k in NO}
            df[c] = s.astype(str).str.lower().map(m).where(s.notna())
            steps.append(Step(f"'{c}': turned yes/no into 1/0.",
                              f"df[{c!r}] = df[{c!r}].astype(str).str.lower().map({m!r}).where(df[{c!r}].notna())"))

    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        df = df.drop(columns=empty)
        steps.append(Step(f"Dropped {len(empty)} completely empty column{'s' if len(empty) > 1 else ''}: {', '.join(empty)}.",
                          f"df = df.drop(columns={empty!r})"))

    dups = int(df.duplicated().sum())
    if dups:
        notes.append(f"{dups:,} rows are exact copies of another row. They were kept: that's normal when rows are "
                     "events, and an error if each row should be one person. Remove them yourself if that's wrong.")
    dated = [c for c in df.columns if re.fullmatch(r"\d{4}[-_/]\d{1,2}([-_/]\d{1,2})?|[A-Za-z]{3,9}[ _-]?\d{2,4}", str(c))]
    if len(dated) >= 3:
        notes.append(f"Columns like {', '.join(map(str, dated[:3]))} look like one column per date. "
                     "This tool needs one row per person (or per day) with a date column. Reshape the file first.")
    return df, steps, notes


def _looks_like_outcome(col: str, df: pd.DataFrame) -> bool:
    """Only flip yes/no to 1/0 when it's not obviously a group label (kept as words in group columns)."""
    return not re.search(r"group|arm|variant|version|treat|program|plan|segment|joined|chose|opt", str(col), re.I)


def script_block(steps: list[Step]) -> str:
    if not steps:
        return ""
    lines = ["", "# --- Data preparation (the app made these same changes) ---"]
    lines += [f"# {s.text}\n{s.code}" for s in steps]
    return "\n".join(lines) + "\n"
