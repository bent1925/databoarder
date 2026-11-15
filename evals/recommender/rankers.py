"""Rankers the harness can score. Each takes the table and what the person has said, never the labels.

A ranker returns what it would show (in order), every question it chose from (its candidates), and, when it decides
the goal can't be answered, a reason from `cases.REASONS`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.analysis import validate_spec
from app.schemas import ABTestSpec, BeforeAfterSpec, DiDSpec, ObservationalSpec
from app.translator import suggest_questions

SHOWN = 4          # the question screen shows up to four suggestions today
POOL = 50          # enough to list everything today's suggestions can produce

SPECS = {"ab_test": ABTestSpec, "observational": ObservationalSpec, "before_after": BeforeAfterSpec,
         "diff_in_diff": DiDSpec}


@dataclass
class Ranking:
    shown: list[dict]
    pool: list[dict]
    cannot_answer: str | None = None
    texts: list[str] = field(default_factory=list)


def baseline_a(df: pd.DataFrame, goal: str, design: str | None, confirmed: dict | None) -> Ranking:
    """Today's prepared questions, in today's order. It ignores the goal, which is the point of measuring it."""
    shown = suggest_questions(df, "answer", limit=SHOWN, design=design, confirmed=confirmed)
    pool = suggest_questions(df, "answer", limit=POOL, design=design, confirmed=confirmed)
    return Ranking(shown=[q["spec"] for q in shown], pool=[q["spec"] for q in pool], texts=[q["text"] for q in shown])


RANKERS = {"baseline_a": baseline_a}


def infeasible(df: pd.DataFrame, spec: dict) -> str | None:
    """Why the data can't support this question, or None. A ranker must never show one that fails.

    Date questions are offered before the person picks the date ("after a date I pick"), so a missing date is
    allowed here, as it is on the question screen. Everything else must pass the same checks as running it."""
    try:
        model = SPECS[spec["analysis_type"]].model_validate(spec)
    except Exception as e:
        return f"not a valid question: {e}"
    found = [p for p in validate_spec(df, model) if p != "Pick the date the change happened."]
    if found:
        return found[0]
    if model.analysis_type == "observational" and not model.covariates:
        return "no background columns to compare people on"
    return None
