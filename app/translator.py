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

from .checks import detect_outcome_kind, is_idlike, validate_spec_against_data
from .interpret import humanize
from .schemas import ABTestSpec, Interpretation

MODEL = os.environ.get("DASHBOARDER_MODEL", "claude-sonnet-5-5")

GROUP_HINTS = re.compile(r"variant|group|arm|treat|test|experiment|cohort|bucket|cell|version", re.I)
OUTCOME_HINTS = re.compile(r"convert|purchas|revenue|sales|click|signup|retain|churn|order|spend|amount|outcome|success", re.I)
CONTROL_HINTS = re.compile(r"control|baseline|base|holdout|old|current|existing|^a$", re.I)


def profile_columns(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in df.columns:
        s = df[col]
        nun = int(s.nunique(dropna=True))
        rows.append({
            "name": col,
            "type": "number" if pd.api.types.is_numeric_dtype(s) else "text",
            "unique_values": nun,
            "examples": [str(v) for v in s.dropna().unique()[:6]],
            "missing": int(s.isna().sum()),
        })
    return rows


def readback(df: pd.DataFrame, spec: ABTestSpec) -> str:
    """Plain-English restatement, written by code from the spec (never by the LLM)."""
    g = df[spec.group_col].astype(str)
    n_c, n_t = int((g == spec.control).sum()), int((g == spec.treatment).sum())
    kind = detect_outcome_kind(df[spec.outcome_col])
    measure = (f"the share of rows where '{spec.outcome_col}' is 1 (a rate)" if kind == "binary"
               else f"the average of '{spec.outcome_col}'")
    return (f"You want to know whether '{spec.treatment}' ({n_t:,} rows) did better or worse than "
            f"'{spec.control}' ({n_c:,} rows), measured by {measure}. "
            f"Groups come from the column '{spec.group_col}'.")


# ----------------------------------------------------------------------------- LLM

def _llm_spec(question: str, profile: list[dict]) -> dict | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        client = anthropic.Anthropic()
        tool = {
            "name": "fill_order_form",
            "description": "Fill in the A/B test analysis form using only columns and values that exist in the data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "group_col": {"type": "string"},
                    "control": {"type": "string", "description": "baseline value in group_col"},
                    "treatment": {"type": "string", "description": "new-version value in group_col"},
                    "outcome_col": {"type": "string"},
                },
                "required": ["group_col", "control", "treatment", "outcome_col"],
            },
        }
        msg = client.messages.create(
            model=MODEL, max_tokens=400, tools=[tool],
            tool_choice={"type": "tool", "name": "fill_order_form"},
            messages=[{"role": "user", "content": (
                "A business user asked a question about their data. Translate it into the A/B test form. "
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

def _heuristic_spec(question: str, df: pd.DataFrame) -> dict:
    q = question.lower()
    n = len(df)
    cols = list(df.columns)

    def idlike(c):
        return c.lower().endswith("id") or df[c].nunique() == n

    # group column: few unique values, name hint wins
    cands = [c for c in cols if 2 <= df[c].nunique() <= 6 and not idlike(c)
             and not pd.api.types.is_float_dtype(df[c])]
    hinted = [c for c in cands if GROUP_HINTS.search(c)]
    group = (hinted or [c for c in cands if not pd.api.types.is_numeric_dtype(df[c])] or cands or [cols[0]])[0]

    levels = [str(v) for v in df[group].dropna().unique()]
    control = next((l for l in levels if CONTROL_HINTS.search(l)), sorted(levels)[0])
    mentioned = [l for l in levels if l.lower() in q and l != control]
    treatment = (mentioned or [l for l in levels if l != control] or levels)[0]

    # outcome column: mentioned in the question > name hint > first numeric
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c != group and not idlike(c)]
    named = [c for c in numeric if c.lower() in q or humanize(c).lower() in q]
    hinted_o = [c for c in numeric if OUTCOME_HINTS.search(c)]
    outcome = (named or hinted_o or numeric or [cols[-1]])[0]
    return {"group_col": group, "control": control, "treatment": treatment, "outcome_col": outcome}


# ---------------------------------------------------------------------------- public

def interpret(question: str, df: pd.DataFrame) -> Interpretation:
    notes: list[str] = []
    source = "heuristic"
    raw = _llm_spec(question, profile_columns(df))
    spec = None
    if raw and "_error" not in raw:
        try:
            cand = ABTestSpec(**raw)
            problems = validate_spec_against_data(df, cand)
            if problems:
                notes.append("The AI's first reading didn't match your data, so a simpler guess was used: " + " ".join(problems))
            else:
                spec, source = cand, "llm"
        except Exception:
            notes.append("The AI's answer wasn't in the expected form, so a simpler guess was used.")
    elif raw and "_error" in raw:
        notes.append(f"The AI service was unreachable ({raw['_error']}); used a simpler guess instead.")
    if spec is None:
        spec = ABTestSpec(**_heuristic_spec(question, df))
        if not os.environ.get("ANTHROPIC_API_KEY"):
            notes.append("No AI key is set, so columns were guessed from their names. Please check the choices below.")
    return Interpretation(spec=spec, readback=readback(df, spec), source=source, notes=notes)


# ------------------------------------------------------------- prepared questions

def suggest_questions(df: pd.DataFrame, mode: str, limit: int = 4) -> list[dict]:
    """Ready-made questions built from the file's own columns.

    Each one carries a finished spec, so picking one skips the guessing step.
    """
    try:
        base = _heuristic_spec("", df)
    except Exception:
        return []
    group, control, treatment = base["group_col"], base["control"], base["treatment"]
    if control == treatment:
        return []
    outcomes = [c for c in df.columns
                if c != group and pd.api.types.is_numeric_dtype(df[c]) and not is_idlike(df, c)]
    outcomes.sort(key=lambda c: (0 if c == base["outcome_col"] else 1 if OUTCOME_HINTS.search(c) else 2))
    out = []
    for o in outcomes[:limit]:
        spec = ABTestSpec(group_col=group, control=control, treatment=treatment, outcome_col=o)
        if validate_spec_against_data(df, spec):
            continue
        what = f"the {humanize(o)} rate" if detect_outcome_kind(df[o]) == "binary" else f"average {humanize(o)}"
        tail = ", and for whom?" if mode == "research" else "?"
        out.append({"text": f"Did '{treatment}' change {what} compared with '{control}'{tail}",
                    "spec": spec.model_dump()})
    return out
