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
from .observational import POST_HINTS, default_covariates
from .interpret import humanize
from .design import DESIGNS
from .privacy import masked_profile
from .schemas import ABTestSpec, BeforeAfterSpec, DiDSpec, Interpretation, ObservationalSpec

MODEL = os.environ.get("DASHBOARDER_MODEL", "claude-sonnet-5-5")

GROUP_HINTS = re.compile(r"variant|group|arm|treat|test|experiment|cohort|bucket|cell|version|member|join|enrol|opt|subscri|adopt|loyal|expos", re.I)
SELF_HINTS = re.compile(r"member|join|enrol|opt|subscri|adopt|loyal|signed|expos|self", re.I)
OUTCOME_HINTS = re.compile(r"convert|purchas|revenue|sales|click|signup|sign_up|signed|renew|retain|churn|order|spend|amount|outcome|success", re.I)
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
    if spec.analysis_type in ("before_after", "diff_in_diff"):
        return _readback_ba(df, spec)
    if spec.analysis_type == "observational":
        return _readback_obs(df, spec)
    g = df[spec.group_col].astype(str)
    n_c, n_t = int((g == spec.control).sum()), int((g == spec.treatment).sum())
    kind = detect_outcome_kind(df[spec.outcome_col])
    measure = f"yes/no rate of '{spec.outcome_col}'" if kind == "binary" else f"average '{spec.outcome_col}'"
    return (f"Is '{spec.treatment}' different from '{spec.control}'? "
            f"Measured by {measure} · {n_t:,} vs {n_c:,} rows.")


def _measure(df: pd.DataFrame, spec) -> str:
    if getattr(spec, "aggregate", "average") == "total":
        return f"daily total of '{spec.outcome_col}'"
    if detect_outcome_kind(df[spec.outcome_col]) == "binary":
        return f"yes/no rate of '{spec.outcome_col}'"
    return f"average '{spec.outcome_col}'"


def _readback_ba(df: pd.DataFrame, spec) -> str:
    if spec.date_col not in df.columns or spec.outcome_col not in df.columns:
        return "Pick the date column and the result to measure."
    if not spec.change_date:
        return "Pick the date the change happened."
    d = parse_dates(df[spec.date_col]).dropna()
    ts = pd.Timestamp(spec.change_date)
    n_b, n_a = int(d[d < ts].nunique()), int(d[d >= ts].nunique())
    when = f"{ts.strftime('%b')} {ts.day}, {ts.year}"
    if spec.analysis_type == "diff_in_diff":
        return (f"Did the {_measure(df, spec)} change more for '{spec.treated}' than for '{spec.comparison}' "
                f"after {when}? {n_b:,} days before vs {n_a:,} after.")
    return f"Did the {_measure(df, spec)} change after {when}? {n_b:,} days before vs {n_a:,} after."


def _readback_obs(df: pd.DataFrame, spec: ObservationalSpec) -> str:
    g = df[spec.group_col].astype(str)
    n_c, n_t = int((g == spec.control).sum()), int((g == spec.treatment).sum())
    cols = ", ".join(spec.covariates[:4]) + (f" +{len(spec.covariates) - 4} more" if len(spec.covariates) > 4 else "")
    like = f", comparing people alike on {cols}" if spec.covariates else ""
    return (f"Is '{spec.treatment}' linked to a different {_measure(df, spec)} than '{spec.control}'{like}? "
            f"{n_t:,} vs {n_c:,} rows.")


# ----------------------------------------------------------------------------- LLM

def _llm_spec(question: str, profile: list[dict], design: str | None = None) -> dict | None:
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
                        "type": "string", "enum": ["ab_test", "before_after", "diff_in_diff", "observational"],
                        "description": "ab_test = two groups, people split at random. before_after = did a number change "
                                       "after a date, for everyone. diff_in_diff = a date, and one group never got the "
                                       "change. observational = two groups people chose themselves."},
                    "outcome_col": {"type": "string"},
                    "group_col": {"type": "string", "description": "ab_test, observational, diff_in_diff"},
                    "control": {"type": "string", "description": "ab_test/observational: baseline value in group_col"},
                    "treatment": {"type": "string", "description": "ab_test/observational: new-version value in group_col"},
                    "treated": {"type": "string", "description": "diff_in_diff: group that got the change"},
                    "comparison": {"type": "string", "description": "diff_in_diff: group that never got it"},
                    "date_col": {"type": "string", "description": "before_after, diff_in_diff: a column of type date"},
                    "change_date": {"type": "string",
                                    "description": "before_after, diff_in_diff: YYYY-MM-DD, ONLY if the question states "
                                                   "the date; otherwise an empty string. Never guess a date."},
                    "aggregate": {"type": "string", "enum": ["average", "total"],
                                  "description": "before_after, diff_in_diff: with several rows per day, use each day's "
                                                 "average (default) or total."},
                },
                "required": ["analysis_type", "outcome_col"],
            },
        }
        fixed = ""
        if design:
            fixed = (f"The person already said how people got the change: \"{DESIGNS[design]['label']}\". "
                     f"Use analysis_type = {DESIGNS[design]['analysis']}.\n\n")
        msg = client.messages.create(
            model=MODEL, max_tokens=800, tools=[tool],
            # This model does not accept a forced tool_choice, so the prompt asks for the tool and we check for it.
            messages=[{"role": "user", "content": (
                "A business user asked a question about their data. Translate it into the analysis form. "
                "Use ONLY the column names and values listed. Do not compute anything. "
                "Answer by calling the fill_order_form tool, and nothing else.\n\n" + fixed +
                f"Question: {question}\n\nColumns:\n{json.dumps(profile, indent=1)}")}],
        )
        for block in msg.content:
            if block.type == "tool_use":
                raw = dict(block.input)
                if design:
                    raw["analysis_type"] = DESIGNS[design]["analysis"]
                return raw
    except Exception as e:  # network, auth, etc. Fall back quietly but tell the user.
        return {"_error": f"{type(e).__name__}: {str(e)[:300]}"}
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
    return (hinted or [c for c in cands if not pd.api.types.is_numeric_dtype(df[c])] or cands or [None])[0]


def _levels_pair(df: pd.DataFrame, group: str, question: str = "") -> tuple[str, str]:
    """(baseline, other) values of a group column; the baseline looks like control/baseline/no."""
    q = question.lower()
    levels = [str(v) for v in df[group].dropna().unique()]
    base = next((l for l in levels if CONTROL_HINTS.search(l)), None) or \
        next((l for l in sorted(levels) if l.lower() in ("no", "false", "0", "none")), None) or sorted(levels)[0]
    mentioned = [l for l in levels if l.lower() in q and l != base]
    other = (mentioned or [l for l in sorted(levels) if l != base] or levels)[0]
    return base, other


def _outcome_choice(df: pd.DataFrame, question: str, exclude: tuple = ()) -> str:
    q = question.lower()
    n = len(df)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude
               and not (c.lower().endswith("id") or df[c].nunique() == n)]
    stems = {w[:5] for w in re.findall(r"[a-z]{4,}", q)}

    def score(c: str) -> tuple:
        words = [w for w in re.findall(r"[a-z]{4,}", c.lower())]
        hits = sum(w[:5] in stems for w in words)
        named = c.lower() in q or humanize(c).lower() in q
        return (named, hits, bool(POST_HINTS.search(c)), bool(OUTCOME_HINTS.search(c)))
    best = sorted(numeric, key=score, reverse=True)
    return (best or [df.columns[-1]])[0]


def _ab_heuristic(question: str, df: pd.DataFrame, analysis: str = "ab_test", assignment: str = "chose") -> dict:
    group = _group_choice(df) or df.columns[0]
    control, treatment = _levels_pair(df, group, question)
    outcome = _outcome_choice(df, question, (group,))
    out = {"analysis_type": analysis, "group_col": group, "control": control, "treatment": treatment,
           "outcome_col": outcome}
    if analysis == "observational":
        out["assignment"] = assignment
    return out


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


def _ba_heuristic(question: str, df: pd.DataFrame, date_col: str, with_group: bool = False) -> dict:
    group = _group_choice(df) if with_group else None
    outcome = _outcome_choice(df, question, tuple(c for c in (date_col, group) if c))
    dates = parse_dates(df[date_col]).dropna()
    out = {"analysis_type": "before_after", "date_col": date_col, "outcome_col": outcome,
           "change_date": find_date_in_text(question, dates) if len(dates) else "", "aggregate": "average"}
    if with_group and group:
        comparison, treated = _levels_pair(df, group, question)
        out.update(analysis_type="diff_in_diff", group_col=group, treated=treated, comparison=comparison)
    return out


def _heuristic_for_design(question: str, df: pd.DataFrame, design: str) -> dict:
    if design == "random":
        return _ab_heuristic(question, df)
    if design in ("chose", "unsure"):
        return _ab_heuristic(question, df, "observational", design)
    dcols = date_columns(df)
    if design == "one_date":
        return _ba_heuristic(question, df, dcols[0])
    return _ba_heuristic(question, df, dcols[0], with_group=True)


def _pick_heuristic(question: str, df: pd.DataFrame) -> dict:
    dcols = date_columns(df)
    group = _group_choice(df)
    ab_ok = group is not None and _has_ab_structure(df, group)
    if dcols:
        dates = parse_dates(df[dcols[0]]).dropna()
        typed_date = bool(len(dates)) and bool(find_date_in_text(question, dates))
        if bool(TIME_WORDS.search(question)) or typed_date or not ab_ok:
            return _ba_heuristic(question, df, dcols[0])
    return _ab_heuristic(question, df)


# ------------------------------------------------------------------ how people got it

STRICT_AB_NAME = re.compile(r"variant|experiment|\bab\b|a_b|bucket|cell|test_?group|\barm\b|assign", re.I)


def design_options(df: pd.DataFrame) -> dict:
    """Which 'how did people get it?' answers the file can support, and the likeliest one."""
    group, dcols = _group_choice(df), date_columns(df)
    need_group, need_date = "Needs a column that splits people into groups.", "Needs a date column."
    ok = {"random": bool(group), "chose": bool(group), "unsure": bool(group), "one_date": bool(dcols),
          "one_date_comparison": bool(group and dcols)}
    why = {"random": need_group, "chose": need_group, "unsure": need_group, "one_date": need_date,
           "one_date_comparison": need_group if not group else need_date}
    default = None
    if group and dcols:
        if SELF_HINTS.search(group):
            default = "chose"
        elif STRICT_AB_NAME.search(group):
            default = "random"           # named like an experiment arm
        elif any(CONTROL_HINTS.search(str(v)) for v in df[group].dropna().unique()[:10]):
            default = "one_date_comparison"   # a control/holdout level plus dates: usually a staged rollout
        else:
            default = "one_date"         # the column is probably just a segment (device, region)
    elif dcols:
        default = "one_date"
    elif group:
        default = "chose" if SELF_HINTS.search(group) or not _has_ab_structure(df, group) else "random"
    return {"options": [{"key": k, **{f: v for f, v in DESIGNS[k].items() if f in ("emoji", "title", "sub")},
                         "ok": ok[k], "why": "" if ok[k] else why[k]} for k in DESIGNS],
            "default": default}


# ---------------------------------------------------------------------------- public

def _build_spec(raw: dict, df: pd.DataFrame, design: str | None = None):
    t = raw.get("analysis_type", "ab_test")
    if t == "before_after":
        return BeforeAfterSpec(date_col=raw["date_col"], outcome_col=raw["outcome_col"],
                               change_date=raw.get("change_date", "") or "", aggregate=raw.get("aggregate") or "average")
    if t == "diff_in_diff":
        return DiDSpec(date_col=raw["date_col"], outcome_col=raw["outcome_col"], group_col=raw["group_col"],
                       treated=raw["treated"], comparison=raw["comparison"],
                       change_date=raw.get("change_date", "") or "", aggregate=raw.get("aggregate") or "average")
    if t == "observational":
        base = ObservationalSpec(group_col=raw["group_col"], control=raw["control"], treatment=raw["treatment"],
                                 outcome_col=raw["outcome_col"],
                                 assignment=design if design in ("chose", "unsure") else raw.get("assignment", "chose"))
        return base.model_copy(update={"covariates": default_covariates(df, base)})
    return ABTestSpec(group_col=raw["group_col"], control=raw["control"], treatment=raw["treatment"],
                      outcome_col=raw["outcome_col"])


def interpret(question: str, df: pd.DataFrame, design: str | None = None) -> Interpretation:
    notes: list[str] = []
    source = "heuristic"
    raw = _llm_spec(question, masked_profile(df, profile_columns(df)), design)
    spec = None
    if raw and "_error" not in raw:
        try:
            cand = _build_spec(raw, df, design)
            problems = validate_spec(df, cand)
            dated = cand.analysis_type in ("before_after", "diff_in_diff")
            if dated and cand.change_date and problems and "outside your data" in problems[0]:
                notes.append(f"The date {cand.change_date} isn't in your data, so please pick it.")
                cand = cand.model_copy(update={"change_date": ""})
                problems = validate_spec(df, cand)
            if dated and not cand.change_date:
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
        guess = _heuristic_for_design(question, df, design) if design else _pick_heuristic(question, df)
        spec = _build_spec(guess, df, design)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            notes.append("No AI key: columns were guessed from names. Please check.")
    if spec.analysis_type in ("before_after", "diff_in_diff") and not spec.change_date:
        notes.append("Pick the date the change happened.")
    return Interpretation(spec=spec, readback=readback(df, spec), source=source, notes=notes)


# ------------------------------------------------------------- prepared questions

def _what(df: pd.DataFrame, o: str) -> str:
    return f"{humanize(o)} rate" if detect_outcome_kind(df[o]) == "binary" else f"average {humanize(o)}"


def _outcomes(df: pd.DataFrame, exclude: tuple, first: str | None, limit: int, after_choice: bool = False) -> list[str]:
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c]) and not is_idlike(df, c)]
    if after_choice:   # for "they chose": results that happen after the choice come first; past_* columns are background
        cols = [c for c in cols if POST_HINTS.search(c)] + [c for c in cols if not POST_HINTS.search(c) and not re.search("past|prior|before|previous|age", c, re.I)]
        return cols[:limit]
    cols.sort(key=lambda c: (0 if c == first else 1 if OUTCOME_HINTS.search(c) else 2))
    return cols[:limit]


def _ab_suggestions(df: pd.DataFrame, mode: str, limit: int, gate: bool = True,
                    analysis: str = "ab_test", assignment: str = "chose") -> list[dict]:
    group = _group_choice(df)
    if group is None or (gate and not _has_ab_structure(df, group)):
        return []
    base = _ab_heuristic("", df)
    group, control, treatment = base["group_col"], base["control"], base["treatment"]
    if control == treatment:
        return []
    tail = ", and for whom?" if mode == "research" else "?"
    out = []
    for o in _outcomes(df, (group,), base["outcome_col"], limit, after_choice=analysis == "observational"):
        if analysis == "observational":
            spec = ObservationalSpec(group_col=group, control=control, treatment=treatment, outcome_col=o,
                                     assignment=assignment)
            spec = spec.model_copy(update={"covariates": default_covariates(df, spec)})
            if not spec.covariates:
                continue
            text = f"Is '{treatment}' linked to a different {_what(df, o)} than '{control}'{tail}"
        else:
            spec = ABTestSpec(group_col=group, control=control, treatment=treatment, outcome_col=o)
            text = f"Did '{treatment}' change {_what(df, o)} vs '{control}'{tail}"
        if validate_spec(df, spec):
            continue
        out.append({"text": text, "spec": spec.model_dump()})
    return out


def _ba_suggestions(df: pd.DataFrame, mode: str, limit: int, with_group: bool = False) -> list[dict]:
    dcols = date_columns(df)
    if not dcols:
        return []
    dcol = dcols[0]
    tail = ", and for whom?" if mode == "research" else "?"
    out = []
    if with_group:
        group = _group_choice(df)
        if group is None:
            return []
        comparison, treated = _levels_pair(df, group)
        for o in _outcomes(df, (dcol, group), None, limit):
            spec = DiDSpec(date_col=dcol, outcome_col=o, change_date="", group_col=group, treated=treated,
                           comparison=comparison)
            out.append({"text": f"Did {_what(df, o)} change more for '{treated}' than '{comparison}' after a date I pick{tail}",
                        "spec": spec.model_dump(), "needs_date": True})
        return out
    for o in _outcomes(df, (dcol,), None, limit):
        spec = BeforeAfterSpec(date_col=dcol, outcome_col=o, change_date="")
        out.append({"text": f"Did {_what(df, o)} change after a date I pick{tail}", "spec": spec.model_dump(),
                    "needs_date": True})
    return out


def suggest_questions(df: pd.DataFrame, mode: str, limit: int = 4, design: str | None = None) -> list[dict]:
    """Ready-made questions built from the file's own columns.

    With a design (how people got the change) the list is for that design only. Without one,
    A-vs-B questions appear when the file has a version column and before/after ones when it has dates.
    """
    try:
        if design == "random":
            return _ab_suggestions(df, mode, limit, gate=False)
        if design in ("chose", "unsure"):
            return _ab_suggestions(df, mode, limit, gate=False, analysis="observational", assignment=design)
        if design == "one_date":
            return _ba_suggestions(df, mode, limit)
        if design == "one_date_comparison":
            return _ba_suggestions(df, mode, limit, with_group=True)
        ab = _ab_suggestions(df, mode, limit)
        ba = _ba_suggestions(df, mode, 2)
    except Exception:
        return []
    return ab[: max(limit - len(ba), 2) if ba else limit] + ba
