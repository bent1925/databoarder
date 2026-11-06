"""Turns a plain-English question into an ABTestSpec.

The waiter: it listens to a messy order and writes it on the standard order
form. If a language-model key is available it does the listening; otherwise a
simple keyword guesser stands in. Either way the result is checked against the
real columns and values before anything runs.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

from .analysis import validate_spec
from .before_after import daily_series, parse_dates
from .checks import detect_outcome_kind, is_idlike
from .interpret import humanize
from .schemas import ABTestSpec, BeforeAfterSpec, Interpretation

MODEL = os.environ.get("DASHBOARDER_MODEL", "claude-sonnet-5-5")

GROUP_HINTS = re.compile(r"variant|group|arm|treat|test|experiment|cohort|bucket|cell|version", re.I)
OUTCOME_HINTS = re.compile(r"convert|purchas|revenue|sales|click|signup|retain|churn|order|spend|amount|outcome|success", re.I)
CONTROL_HINTS = re.compile(r"control|baseline|base|holdout|old|current|existing|^a$", re.I)
DATE_HINTS = re.compile(r"date|day|time|week|month|period|^dt$|_at$", re.I)
TIME_WORDS = re.compile(r"\b(after|before|since|launch\w*|releas\w*|rolled out|roll-out|rollout|went live|go live|"
                        r"over time|trend|ever since|change[sd]? (on|in|at)|redesign\w*|price (rise|increase|cut))\b", re.I)
MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
ISO_IN_TEXT = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
MD_IN_TEXT = re.compile(rf"\b({MONTHS})[a-z]*\.? (\d{{1,2}})(?:st|nd|rd|th)?(?:,? (\d{{4}}))?\b", re.I)
DM_IN_TEXT = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)? ({MONTHS})[a-z]*\.?(?:,? (\d{{4}}))?\b", re.I)


def is_date_col(s: pd.Series) -> bool:
    """A text column whose values read as calendar dates (checked on a sample)."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    vals = s.dropna()
    if len(vals) == 0:
        return False
    sample = vals.sample(min(len(vals), 200), random_state=0)
    d = parse_dates(sample)
    return bool(d.notna().mean() >= 0.9 and (d.dropna().dt.year >= 1900).all())


def date_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if is_date_col(df[c])]
    cols.sort(key=lambda c: (0 if DATE_HINTS.search(c) else 1))
    return cols


def profile_columns(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in df.columns:
        s = df[col]
        nun = int(s.nunique(dropna=True))
        kind = "number" if pd.api.types.is_numeric_dtype(s) else ("date" if is_date_col(s) else "text")
        row = {
            "name": col,
            "type": kind,
            "unique_values": nun,
            "examples": [str(v) for v in s.dropna().unique()[:6]],
            "missing": int(s.isna().sum()),
        }
        if kind == "date":
            d = parse_dates(s).dropna()
            if len(d):
                row.update(date_min=d.min().date().isoformat(), date_max=d.max().date().isoformat(),
                           rows_per_day=round(len(d) / max(d.nunique(), 1), 2))
        rows.append(row)
    return rows


def readback(df: pd.DataFrame, spec) -> str:
    """Plain-English restatement, written by code from the spec (never by the LLM)."""
    if spec.analysis_type == "before_after":
        return _readback_ba(df, spec)
    g = df[spec.group_col].astype(str)
    n_c, n_t = int((g == spec.control).sum()), int((g == spec.treatment).sum())
    kind = detect_outcome_kind(df[spec.outcome_col])
    measure = f"yes/no rate of '{spec.outcome_col}'" if kind == "binary" else f"average '{spec.outcome_col}'"
    return (f"Is '{spec.treatment}' different from '{spec.control}'? "
            f"Measured by {measure} · {n_t:,} vs {n_c:,} rows.")


def _readback_ba(df: pd.DataFrame, spec: BeforeAfterSpec) -> str:
    if spec.date_col not in df.columns or spec.outcome_col not in df.columns:
        return "Pick the date column and the result to measure."
    if not spec.change_date:
        return "Pick the date the change happened."
    d = parse_dates(df[spec.date_col]).dropna()
    ts = pd.Timestamp(spec.change_date)
    n_b, n_a = int(d[d < ts].nunique()), int(d[d >= ts].nunique())
    if spec.aggregate == "total":
        measure = f"daily total of '{spec.outcome_col}'"
    elif detect_outcome_kind(df[spec.outcome_col]) == "binary":
        measure = f"yes/no rate of '{spec.outcome_col}'"
    else:
        measure = f"average '{spec.outcome_col}'"
    return (f"Did the {measure} change after {ts.strftime('%b')} {ts.day}, {ts.year}? "
            f"{n_b:,} days before vs {n_a:,} after.")


# ----------------------------------------------------------------------------- LLM

def _llm_spec(question: str, profile: list[dict]) -> dict | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        client = anthropic.Anthropic()
        tool = {
            "name": "fill_order_form",
            "description": "Fill in the analysis form using only columns and values that exist in the data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string", "enum": ["ab_test", "before_after"],
                        "description": "ab_test = compare two groups shown different versions at the same time. "
                                       "before_after = did a number change after a date."},
                    "outcome_col": {"type": "string"},
                    "group_col": {"type": "string", "description": "ab_test only"},
                    "control": {"type": "string", "description": "ab_test only: baseline value in group_col"},
                    "treatment": {"type": "string", "description": "ab_test only: new-version value in group_col"},
                    "date_col": {"type": "string", "description": "before_after only: a column of type date"},
                    "change_date": {"type": "string",
                                    "description": "before_after only: YYYY-MM-DD, ONLY if the question states the date; "
                                                   "otherwise an empty string. Never guess a date."},
                    "aggregate": {"type": "string", "enum": ["average", "total"],
                                  "description": "before_after only: with several rows per day, use each day's "
                                                 "average (default) or total."},
                },
                "required": ["analysis_type", "outcome_col"],
            },
        }
        msg = client.messages.create(
            model=MODEL, max_tokens=500, tools=[tool],
            tool_choice={"type": "tool", "name": "fill_order_form"},
            messages=[{"role": "user", "content": (
                "A business user asked a question about their data. Translate it into the analysis form. "
                "Use ONLY the column names and values listed. Do not compute anything.\n\n"
                f"Question: {question}\n\nColumns:\n{json.dumps(profile, indent=1)}")}],
        )
        for block in msg.content:
            if block.type == "tool_use":
                return dict(block.input)
    except Exception as e:  # network, auth, etc. Fall back quietly but tell the user.
        return {"_error": f"{type(e).__name__}"}
    return None


# -------------------------------------------------------------------------- heuristic

def _has_ab_structure(df: pd.DataFrame, group: str) -> bool:
    """Does this column look like 'which version did they see'? Names or values must say so."""
    if GROUP_HINTS.search(group):
        return True
    return any(CONTROL_HINTS.search(str(v)) for v in df[group].dropna().unique()[:10])


def _group_choice(df: pd.DataFrame) -> str | None:
    n = len(df)
    cols = list(df.columns)

    def idlike(c):
        return c.lower().endswith("id") or df[c].nunique() == n
    cands = [c for c in cols if 2 <= df[c].nunique() <= 6 and not idlike(c)
             and not pd.api.types.is_float_dtype(df[c]) and not is_date_col(df[c])]
    hinted = [c for c in cands if GROUP_HINTS.search(c)]
    pick = (hinted or [c for c in cands if not pd.api.types.is_numeric_dtype(df[c])] or cands or [None])[0]
    return pick


def _ab_heuristic(question: str, df: pd.DataFrame) -> dict:
    q = question.lower()
    n = len(df)
    cols = list(df.columns)

    def idlike(c):
        return c.lower().endswith("id") or df[c].nunique() == n

    group = _group_choice(df) or cols[0]
    levels = [str(v) for v in df[group].dropna().unique()]
    control = next((l for l in levels if CONTROL_HINTS.search(l)), sorted(levels)[0])
    mentioned = [l for l in levels if l.lower() in q and l != control]
    treatment = (mentioned or [l for l in levels if l != control] or levels)[0]

    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c != group and not idlike(c)]
    named = [c for c in numeric if c.lower() in q or humanize(c).lower() in q]
    hinted_o = [c for c in numeric if OUTCOME_HINTS.search(c)]
    outcome = (named or hinted_o or numeric or [cols[-1]])[0]
    return {"analysis_type": "ab_test", "group_col": group, "control": control, "treatment": treatment,
            "outcome_col": outcome}


def _heuristic_spec(question: str, df: pd.DataFrame) -> dict:
    """Backwards-compatible entry: the A/B guesser."""
    return _ab_heuristic(question, df)


def find_date_in_text(question: str, df_dates: pd.Series) -> str:
    """A date the person typed in the question, as YYYY-MM-DD ('' if none). Years default to the data's."""
    m = ISO_IN_TEXT.search(question)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    lo, hi = df_dates.min(), df_dates.max()
    for rx, order in ((MD_IN_TEXT, "md"), (DM_IN_TEXT, "dm")):
        m = rx.search(question)
        if not m:
            continue
        mon, day, year = (m.group(1), m.group(2), m.group(3)) if order == "md" else (m.group(2), m.group(1), m.group(3))
        years = [int(year)] if year else sorted({hi.year, lo.year}, reverse=True)
        for y in years:
            ts = pd.to_datetime(f"{mon[:3]} {int(day)} {y}", format="%b %d %Y", errors="coerce")
            if pd.notna(ts) and lo < ts <= hi:
                return ts.date().isoformat()
        return ""
    return ""


def _ba_heuristic(question: str, df: pd.DataFrame, date_col: str) -> dict:
    q = question.lower()
    cols = list(df.columns)
    n = len(df)

    def idlike(c):
        return c.lower().endswith("id") or df[c].nunique() == n
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and not idlike(c)]
    named = [c for c in numeric if c.lower() in q or humanize(c).lower() in q]
    hinted_o = [c for c in numeric if OUTCOME_HINTS.search(c)]
    outcome = (named or hinted_o or numeric or [cols[-1]])[0]
    dates = parse_dates(df[date_col]).dropna()
    return {"analysis_type": "before_after", "date_col": date_col, "outcome_col": outcome,
            "change_date": find_date_in_text(question, dates) if len(dates) else "", "aggregate": "average"}


def _pick_heuristic(question: str, df: pd.DataFrame) -> dict:
    dcols = date_columns(df)
    group = _group_choice(df)
    ab_ok = group is not None and _has_ab_structure(df, group)
    if dcols:
        dates = parse_dates(df[dcols[0]]).dropna()
        typed_date = bool(len(dates)) and bool(find_date_in_text(question, dates))
        wants_time = bool(TIME_WORDS.search(question)) or typed_date
        if wants_time or not ab_ok:
            return _ba_heuristic(question, df, dcols[0])
    return _ab_heuristic(question, df)


# ---------------------------------------------------------------------------- public

def _build_spec(raw: dict):
    t = raw.get("analysis_type", "ab_test")
    if t == "before_after":
        return BeforeAfterSpec(date_col=raw["date_col"], outcome_col=raw["outcome_col"],
                               change_date=raw.get("change_date", "") or "",
                               aggregate=raw.get("aggregate") or "average")
    return ABTestSpec(group_col=raw["group_col"], control=raw["control"], treatment=raw["treatment"],
                      outcome_col=raw["outcome_col"])


def interpret(question: str, df: pd.DataFrame) -> Interpretation:
    notes: list[str] = []
    source = "heuristic"
    raw = _llm_spec(question, profile_columns(df))
    spec = None
    if raw and "_error" not in raw:
        try:
            cand = _build_spec(raw)
            problems = validate_spec(df, cand)
            if cand.analysis_type == "before_after" and cand.change_date and problems and \
                    "outside your data" in problems[0]:
                notes.append(f"The date {cand.change_date} isn't in your data, so please pick it.")
                cand = cand.model_copy(update={"change_date": ""})
                problems = validate_spec(df, cand)
            if cand.analysis_type == "before_after" and not cand.change_date:
                problems = [p for p in problems if p != "Pick the date the change happened."]
            if problems:
                notes.append("The AI misread your data, so columns were guessed: " + " ".join(problems))
            else:
                spec, source = cand, "llm"
        except Exception:
            notes.append("The AI answered oddly, so columns were guessed.")
    elif raw and "_error" in raw:
        notes.append(f"AI unreachable ({raw['_error']}), so columns were guessed.")
    if spec is None:
        spec = _build_spec(_pick_heuristic(question, df))
        if not os.environ.get("ANTHROPIC_API_KEY"):
            notes.append("No AI key: columns were guessed from names. Please check.")
    if spec.analysis_type == "before_after" and not spec.change_date:
        notes.append("Pick the date the change happened.")
    return Interpretation(spec=spec, readback=readback(df, spec), source=source, notes=notes)


# ------------------------------------------------------------- prepared questions

def _ab_suggestions(df: pd.DataFrame, mode: str, limit: int) -> list[dict]:
    group = _group_choice(df)
    if group is None or not _has_ab_structure(df, group):
        return []
    base = _ab_heuristic("", df)
    group, control, treatment = base["group_col"], base["control"], base["treatment"]
    if control == treatment:
        return []
    outcomes = [c for c in df.columns
                if c != group and pd.api.types.is_numeric_dtype(df[c]) and not is_idlike(df, c)]
    outcomes.sort(key=lambda c: (0 if c == base["outcome_col"] else 1 if OUTCOME_HINTS.search(c) else 2))
    out = []
    for o in outcomes[:limit]:
        spec = ABTestSpec(group_col=group, control=control, treatment=treatment, outcome_col=o)
        if validate_spec(df, spec):
            continue
        what = f"{humanize(o)} rate" if detect_outcome_kind(df[o]) == "binary" else f"average {humanize(o)}"
        tail = ", and for whom?" if mode == "research" else "?"
        out.append({"text": f"Did '{treatment}' change {what} vs '{control}'{tail}", "spec": spec.model_dump()})
    return out


def _ba_suggestions(df: pd.DataFrame, mode: str, limit: int) -> list[dict]:
    dcols = date_columns(df)
    if not dcols:
        return []
    dcol = dcols[0]
    outcomes = [c for c in df.columns if c != dcol and pd.api.types.is_numeric_dtype(df[c]) and not is_idlike(df, c)]
    outcomes.sort(key=lambda c: (1 if OUTCOME_HINTS.search(c) is None else 0))
    out = []
    for o in outcomes[:limit]:
        spec = BeforeAfterSpec(date_col=dcol, outcome_col=o, change_date="")
        what = f"{humanize(o)} rate" if detect_outcome_kind(df[o]) == "binary" else f"average {humanize(o)}"
        tail = ", and for whom?" if mode == "research" else "?"
        out.append({"text": f"Did {what} change after a date I pick{tail}", "spec": spec.model_dump(),
                    "needs_date": True})
    return out


def suggest_questions(df: pd.DataFrame, mode: str, limit: int = 4) -> list[dict]:
    """Ready-made questions built from the file's own columns.

    Each one carries a finished spec, so picking one skips the guessing step.
    A-vs-B questions appear only when the file really has a version column; before/after
    questions appear when the file has a date column.
    """
    try:
        ab = _ab_suggestions(df, mode, limit)
        ba = _ba_suggestions(df, mode, 2)
    except Exception:
        return []
    return ab[: max(limit - len(ba), 2) if ba else limit] + ba
