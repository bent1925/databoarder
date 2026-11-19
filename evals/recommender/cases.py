"""Labelled cases for the question recommender: loading, checking, and expanding them.

One JSON file per dataset in evals/recommender_cases/. Each hand-written goal has paraphrases, and each dataset is also
used with its columns renamed, so one goal becomes several cases. Cases from the same goal share a `goal_id`, and all
cases from one dataset share a `dataset`, so a train/test split can keep them on the same side.

Labels describe questions by their fields, not by a candidate ID, so they don't depend on how candidates are
generated. Fields a label leaves out match anything. For example, a goal that doesn't say how people got the change
is labelled without `analysis_type`, so the A/B and "they chose" versions of the same comparison score alike. The
ranker must not prefer a design because it makes a stronger claim, and the labels mustn't either.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.prep import prepare
from app.translator import DESIGNS

ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = ROOT / "evals" / "recommender_cases"

# Why a goal can't be answered from the table. Later phases show these in plain words.
REASONS = {
    "no_dates": "The goal is about change over time, but the table has no date column.",
    "no_comparison_group": "The goal compares people who got the change with people who didn't, but nothing in the "
                           "table says who got it, or everyone did.",
    "missing_outcome": "The result the goal asks about isn't measured in the table.",
    "decision": "The goal asks what to do (\"should we...?\"). Data can show what happened, but the choice also "
                "depends on costs and priorities that aren't in the table.",
}
MATCH_FIELDS = {"analysis_type", "outcome_col", "group_col", "date_col", "treated", "base", "levels", "effect_by"}
COLUMN_FIELDS = ("outcome_col", "group_col", "date_col", "effect_by")
ANALYSIS_TYPES = {"ab_test", "observational", "before_after", "diff_in_diff"}


@dataclass
class Case:
    case_id: str          # goal_id, paraphrase number, and "r" when columns are renamed
    goal_id: str
    dataset: str
    goal: str
    renamed: bool
    design: str | None
    confirmed: dict | None
    labels: list[dict] = field(default_factory=list)
    unanswerable: str | None = None


def load_files(folder: Path = CASES_DIR) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]


def read_table(path: Path) -> pd.DataFrame:
    """The table as the app sees it after upload: read, then the same automatic clean-up."""
    df, _, _ = prepare(pd.read_csv(path))
    return df


def _rename_value(value, renames: dict):
    if isinstance(value, list):
        return [renames.get(v, v) for v in value]
    return renames.get(value, value)


def _rename_labels(labels: list[dict], renames: dict) -> list[dict]:
    return [{**label, "match": {f: (_rename_value(v, renames) if f in COLUMN_FIELDS else v)
                                for f, v in label["match"].items()}} for label in labels]


def _rename_confirmed(confirmed: dict | None, renames: dict) -> dict | None:
    return None if confirmed is None else {k: renames.get(v, v) for k, v in confirmed.items()}


def expand(spec: dict) -> list[Case]:
    """Every case one dataset file gives: each goal and paraphrase, with original and renamed columns."""
    out = []
    renames = spec.get("renames", {})
    for goal in spec["goals"]:
        texts = [goal["goal"], *goal.get("paraphrases", [])]
        for renamed in (False, True) if renames else (False,):
            labels = goal.get("labels", [])
            confirmed = goal.get("confirmed")
            if renamed:
                labels, confirmed = _rename_labels(labels, renames), _rename_confirmed(confirmed, renames)
            for i, text in enumerate(texts):
                out.append(Case(case_id=f"{goal['id']}#{i}{'r' if renamed else ''}", goal_id=goal["id"],
                                dataset=spec["dataset"], goal=text, renamed=renamed, design=goal.get("design"),
                                confirmed=confirmed, labels=labels, unanswerable=goal.get("unanswerable")))
    return out


def problems(spec: dict, df: pd.DataFrame | None) -> list[str]:
    """Everything wrong with one dataset's case file. Column checks need the table, so they're skipped without it."""
    out = []
    name = spec.get("dataset", "?")
    cols = set(df.columns) if df is not None else None
    renames = spec.get("renames", {})
    if cols is not None:
        out += [f"{name}: rename of missing column '{c}'" for c in renames if c not in cols]
        clash = set(renames.values()) & (cols - set(renames))
        out += [f"{name}: rename collides with existing column '{c}'" for c in sorted(clash)]
    seen = set()
    for goal in spec.get("goals", []):
        gid = goal.get("id", "?")
        where = f"{name}/{gid}"
        if gid in seen:
            out.append(f"{where}: duplicate id")
        seen.add(gid)
        paras = goal.get("paraphrases", [])
        if not 3 <= len(paras) <= 5:
            out.append(f"{where}: needs 3-5 paraphrases, has {len(paras)}")
        if len({p.strip().lower() for p in [goal["goal"], *paras]}) != len(paras) + 1:
            out.append(f"{where}: repeated wording")
        if goal.get("design") not in (None, *DESIGNS):
            out.append(f"{where}: unknown design {goal.get('design')!r}")
        reason = goal.get("unanswerable")
        if reason is not None and reason not in REASONS:
            out.append(f"{where}: unknown reason {reason!r}")
        labels = goal.get("labels", [])
        if reason is None and not any(label.get("grade") == 2 for label in labels):
            out.append(f"{where}: an answerable goal needs at least one grade-2 label")
        if reason is not None and any(label.get("grade") == 2 for label in labels):
            out.append(f"{where}: an unanswerable goal can only have grade-1 (closest) labels")
        for c in (goal.get("confirmed") or {}).values():
            if cols is not None and c not in cols:
                out.append(f"{where}: confirmed column '{c}' isn't in the table")
        for label in labels:
            out += [f"{where}: {p}" for p in _label_problems(label, df)]
    return out


def _label_problems(label: dict, df: pd.DataFrame | None) -> list[str]:
    out = []
    if label.get("grade") not in (1, 2):
        out.append(f"grade must be 1 or 2, not {label.get('grade')!r}")
    match = label.get("match", {})
    if not match:
        out.append("label matches every question")
    if not label.get("why"):
        out.append("label needs a 'why' for the reviewer")
    bad = set(match) - MATCH_FIELDS
    if bad:
        out.append(f"unknown match fields {sorted(bad)}")
    types = match.get("analysis_type")
    for t in types if isinstance(types, list) else [types] if types else []:
        if t not in ANALYSIS_TYPES:
            out.append(f"unknown analysis_type {t!r}")
    if df is None:
        return out
    for f in COLUMN_FIELDS:
        v = match.get(f)
        for c in v if isinstance(v, list) else [v] if v else []:
            if c not in df.columns:
                out.append(f"column '{c}' isn't in the table")
    groups = match.get("group_col")
    groups = groups if isinstance(groups, list) else [groups] if groups else []
    levels = [*([match["treated"]] if "treated" in match else []), *([match["base"]] if "base" in match else []),
              *match.get("levels", [])]
    if levels and not groups:
        out.append("levels need a group_col")
    for g in groups:
        if g in df.columns:
            have = {str(v) for v in df[g].dropna().unique()}
            out += [f"level '{lv}' isn't in '{g}'" for lv in levels if str(lv) not in have]
    return out
