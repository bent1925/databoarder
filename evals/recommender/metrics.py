"""Scoring a ranked list of questions against one case's graded labels.

Grades: 2 answers the goal well, 1 answers it acceptably, 0 (anything no label matches) doesn't answer it.

Each label is credited once. If two shown questions match the same label (the same comparison under two designs, say),
only the higher one earns its grade, because the second tells the person nothing new. This also keeps the ideal list
(one entry per label) a true upper bound, so NDCG stays between 0 and 1.
"""
from __future__ import annotations

import math

GOOD = 2


def key_of(spec: dict) -> dict:
    """The fields a label can match on, the same for every design.

    A/B and "they chose" questions call their levels control/treatment; date-with-comparison ones call them
    comparison/treated. Labels use `treated` and `base` for both, or `levels` when the order doesn't matter."""
    treated = spec.get("treatment") or spec.get("treated")
    base = spec.get("control") or spec.get("comparison")
    key = {"analysis_type": spec.get("analysis_type"), "outcome_col": spec.get("outcome_col"),
           "group_col": spec.get("group_col"), "date_col": spec.get("date_col"),
           "treated": None if treated is None else str(treated), "base": None if base is None else str(base)}
    key["levels"] = frozenset((key["treated"], key["base"])) if treated is not None and base is not None else None
    return key


def matches(key: dict, match: dict) -> bool:
    for field, want in match.items():
        got = key.get(field)
        if field == "levels":
            if got != frozenset(str(v) for v in want):
                return False
        elif isinstance(want, list):
            if got not in want:
                return False
        elif got != want:
            return False
    return True


def grades(shown: list[dict], labels: list[dict]) -> list[int]:
    """The grade each shown question earns, in order, crediting each label at most once."""
    used: set[int] = set()
    out = []
    for spec in shown:
        key = key_of(spec)
        best, best_i = 0, None
        for i, label in enumerate(labels):
            if i not in used and label["grade"] > best and matches(key, label["match"]):
                best, best_i = label["grade"], i
        if best_i is not None:
            used.add(best_i)
        out.append(best)
    return out


def ideal(labels: list[dict]) -> list[int]:
    return sorted((label["grade"] for label in labels), reverse=True)


def recall_at(g: list[int], k: int) -> float:
    """1 if a well-fitting question is in the top k, else 0. Averaged over cases, the share of cases with one."""
    return float(any(x >= GOOD for x in g[:k]))


def dcg(g: list[int], k: int) -> float:
    return sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(g[:k]))


def ndcg_at(g: list[int], ideal_grades: list[int], k: int) -> float:
    best = dcg(ideal_grades, k)
    return dcg(g, k) / best if best else 0.0


def reciprocal_rank(g: list[int]) -> float:
    """1 / the position of the first well-fitting question; 0 if none is shown."""
    return next((1 / (i + 1) for i, x in enumerate(g) if x >= GOOD), 0.0)


def score_answerable(shown: list[dict], pool: list[dict], labels: list[dict], k_recall: int = 3,
                     k_ndcg: int = 5) -> dict:
    """Metrics for one answerable case. `pool` is every question the ranker could have shown (its candidates)."""
    g = grades(shown, labels)
    return {"candidate_recall": recall_at(grades(pool, labels), len(pool)),
            f"recall@{k_recall}": recall_at(g, k_recall),
            f"ndcg@{k_ndcg}": ndcg_at(g, ideal(labels), k_ndcg),
            "mrr": reciprocal_rank(g)}
