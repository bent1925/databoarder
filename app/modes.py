"""What changes between the three kinds of work.

  quick         -> same analysis, plus an honest list of caveats at the end.
  right_answer  -> cross-checks the result with independent methods; shaky things go on top.
  research      -> also scans subgroups, corrected so we don't chase noise.

The main analysis is identical in all three; modes only add to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, test_proportions_2indep

from .checks import is_idlike
from .interpret import chance_phrase
from .schemas import ABTestSpec, Report, Robustness, Segment

ALPHA = 0.05
MIN_SEG = 50      # rows needed in EACH group within a slice
MAX_LEVELS = 8    # columns with more distinct values aren't treated as slices
N_PERM = 2000
SEED = 0


def effect(c, t, kind: str) -> dict:
    """Difference (treatment minus control), its 95% range, and p-value."""
    c, t = np.asarray(c, float), np.asarray(t, float)
    if kind == "binary":
        xc, nc, xt, nt = int(c.sum()), len(c), int(t.sum()), len(t)
        lo, hi = confint_proportions_2indep(xt, nt, xc, nc, method="newcomb", compare="diff")
        p = float(test_proportions_2indep(xt, nt, xc, nc, method="score", compare="diff", correction=False).pvalue)
        return {"diff": xt / nt - xc / nc, "lo": float(lo), "hi": float(hi), "p": p if np.isfinite(p) else 1.0}
    res = stats.ttest_ind(t, c, equal_var=False)
    ci = res.confidence_interval(0.95)
    p = float(res.pvalue)
    return {"diff": float(t.mean() - c.mean()), "lo": float(ci.low), "hi": float(ci.high),
            "p": p if np.isfinite(p) else 1.0}


# ---------------------------------------------------------------- right answer

def _permutation_p(c: np.ndarray, t: np.ndarray, seed: int = SEED) -> float:
    """Shuffle who was in which group; count how often chance alone beats the observed gap."""
    rng = np.random.default_rng(seed)  # fixed seed: same data -> same answer
    pooled, nt = np.concatenate([c, t]), len(t)
    obs = abs(t.mean() - c.mean())
    hits = 0
    for _ in range(N_PERM):
        rng.shuffle(pooled)
        if abs(pooled[:nt].mean() - pooled[nt:].mean()) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (N_PERM + 1)


def _verdict_text(p: float, main_clear: bool) -> tuple[str, bool]:
    agrees = (p < ALPHA) == main_clear
    word = "Same conclusion" if agrees else "Different conclusion"
    return f"{word}. If nothing changed: {chance_phrase(p)}.", agrees


def robustness_checks(c: np.ndarray, t: np.ndarray, kind: str, main_clear: bool) -> list[Robustness]:
    out: list[Robustness] = []

    if kind == "binary":
        xc, nc, xt, nt = int(c.sum()), len(c), int(t.sum()), len(t)
        p = float(stats.fisher_exact([[xt, nt - xt], [xc, nc - xc]])[1])
        key, name, desc = "fisher", "Exact test (Fisher)", "A different test with fewer assumptions."
    else:
        p = float(stats.mannwhitneyu(t, c, alternative="two-sided").pvalue)
        key, name = "mannwhitney", "Rank test (Mann–Whitney)"
        desc = "Who tends to score higher? Extremes can't dominate."
    text, agrees = _verdict_text(p, main_clear)
    out.append(Robustness(key=key, name=name, description=desc, result=text, agrees=agrees, p_value=p))

    p = _permutation_p(c, t)
    text, agrees = _verdict_text(p, main_clear)
    out.append(Robustness(key="shuffle", name="Shuffle test",
                          description=f"Reshuffles groups {N_PERM:,} times to see how often chance matches.",
                          result=text, agrees=agrees, p_value=p))

    if kind == "continuous":
        pooled = np.concatenate([c, t])
        lo_q, hi_q = np.percentile(pooled, [1, 99])
        c2, t2 = c[(c >= lo_q) & (c <= hi_q)], t[(t >= lo_q) & (t <= hi_q)]
        if len(c2) > 30 and len(t2) > 30:
            e = effect(c2, t2, "continuous")
            same_dir = np.sign(e["diff"]) == np.sign(t.mean() - c.mean())
            text, agrees = _verdict_text(e["p"], main_clear)
            out.append(Robustness(key="trim", name="Without extreme values",
                                  description="Reruns without the top and bottom 1%.",
                                  result=text, agrees=bool(agrees and same_dir), p_value=e["p"]))
    return out


# -------------------------------------------------------------------- research

def segment_scan(sub: pd.DataFrame, spec: ABTestSpec, kind: str) -> tuple[list[Segment], str]:
    gs = sub[spec.group_col].astype(str)
    rows: list[dict] = []
    for col in sub.columns:
        if col in (spec.group_col, spec.outcome_col) or is_idlike(sub, col):
            continue
        s = sub[col]
        # Slice only on category columns (country, device...). Numeric columns are usually
        # results (purchased, revenue), and slicing on a result gives meaningless comparisons.
        if pd.api.types.is_numeric_dtype(s) or not (2 <= s.nunique(dropna=True) <= MAX_LEVELS):
            continue
        s_str = s.astype(str)
        for level in sorted(s.dropna().astype(str).unique()):
            m = s_str == level
            c = sub.loc[m & (gs == spec.control), spec.outcome_col]
            t = sub.loc[m & (gs == spec.treatment), spec.outcome_col]
            if len(c) < MIN_SEG or len(t) < MIN_SEG:
                continue
            e = effect(c, t, kind)
            rows.append({"column": col, "level": level, "n_control": len(c), "n_treatment": len(t),
                         "diff": e["diff"], "ci_low": e["lo"], "ci_high": e["hi"], "p_raw": e["p"]})
    if not rows:
        return [], "No slice was big enough to compare."

    p_adj = multipletests([r["p_raw"] for r in rows], method="holm")[1]
    segs = [Segment(**r, p_adj=float(pa), flagged=bool(pa < ALPHA)) for r, pa in zip(rows, p_adj)]
    segs.sort(key=lambda s: s.p_raw)
    m, flagged = len(segs), sum(s.flagged for s in segs)
    note = f"{m} slices checked, adjusted for that. "
    note += (f"{flagged} stand{'s' if flagged == 1 else ''} out: ideas to test, not conclusions."
             if flagged else "None stand out.")
    return segs, note


# ------------------------------------------------------------------------ quick

def quick_caveats(rep: Report) -> list[str]:
    cav = [c.message for c in rep.checks if c.status == "warn"]
    cav.append("Quick look: not cross-checked.")
    cav.append("Cause-and-effect only if assignment was random.")
    if rep.verdict == "no_clear_difference":
        cav.append("'No clear difference' isn't 'no effect'. The test may be too small.")
    cav.append("Big decision? Rerun as 'Need the right answer'.")
    return cav


# ------------------------------------------------------------------------ entry

def apply_mode(rep: Report, sub: pd.DataFrame, spec: ABTestSpec, kind: str, mode: str) -> Report:
    rep.mode = mode
    gs = sub[spec.group_col].astype(str)
    c = sub.loc[gs == spec.control, spec.outcome_col].astype(float).to_numpy()
    t = sub.loc[gs == spec.treatment, spec.outcome_col].astype(float).to_numpy()
    main_clear = rep.verdict == "clear_difference"

    if mode == "quick":
        rep.caveats = quick_caveats(rep)
    elif mode == "right_answer":
        rep.robustness = robustness_checks(c, t, kind, main_clear)
        bad = [r for r in rep.robustness if not r.agrees]
        n = len(rep.robustness)
        if bad:
            rep.robust_summary = f"Not robust: {len(bad)} of {n} cross-checks disagree. Don't rely on this yet."
        else:
            rep.robust_summary = f"Holds up: all {n} cross-checks agree."
        rep.review_items = [c_.message for c_ in rep.checks if c_.status == "warn"]
        if bad:
            rep.review_items.append(rep.robust_summary)
    elif mode == "research":
        rep.segments, rep.segment_note = segment_scan(sub, spec, kind)
    return rep
