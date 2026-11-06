"""What changes between the three kinds of work.

  quick         -> same analysis, plus an honest list of caveats at the end.
  right_answer  -> cross-checks the result with independent methods and puts
                   anything shaky at the top.
  research      -> also scans subgroups, with a correction so we don't chase noise.

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


def effect(c, t, kind: str) -> dict:
    """Difference (treatment minus control), its 95% range and p-value."""
    c, t = np.asarray(c, float), np.asarray(t, float)
    if kind == "binary":
        xc, nc, xt, nt = int(c.sum()), len(c), int(t.sum()), len(t)
        lo, hi = confint_proportions_2indep(xt, nt, xc, nc, method="newcomb", compare="diff")
        p = float(test_proportions_2indep(xt, nt, xc, nc, method="wald", compare="diff").pvalue)
        return {"diff": xt / nt - xc / nc, "lo": float(lo), "hi": float(hi), "p": p if np.isfinite(p) else 1.0}
    res = stats.ttest_ind(t, c, equal_var=False)
    ci = res.confidence_interval(0.95)
    p = float(res.pvalue)
    return {"diff": float(t.mean() - c.mean()), "lo": float(ci.low), "hi": float(ci.high),
            "p": p if np.isfinite(p) else 1.0}


# ---------------------------------------------------------------- right answer

def _permutation_p(c: np.ndarray, t: np.ndarray, seed: int = 0) -> float:
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
    sig = p < ALPHA
    word = "Reaches the same conclusion" if sig == main_clear else "Reaches a different conclusion"
    return f"{word} (a gap this big would show up by chance {chance_phrase(p)}).", sig == main_clear


def robustness_checks(c: np.ndarray, t: np.ndarray, kind: str, main_clear: bool) -> list[Robustness]:
    out: list[Robustness] = []

    if kind == "binary":
        xc, nc, xt, nt = int(c.sum()), len(c), int(t.sum()), len(t)
        p = float(stats.fisher_exact([[xt, nt - xt], [xc, nc - xc]])[1])
        name, desc = "Exact test (Fisher)", "A different test that leans on fewer assumptions about the data."
    else:
        p = float(stats.mannwhitneyu(t, c, alternative="two-sided").pvalue)
        name = "Rank-based test (Mann-Whitney)"
        desc = "Asks whether one group tends to have higher values, so a few extreme values can't dominate."
    text, agrees = _verdict_text(p, main_clear)
    out.append(Robustness(name=name, description=desc, result=text, agrees=agrees))

    p = _permutation_p(c, t)
    text, agrees = _verdict_text(p, main_clear)
    out.append(Robustness(
        name="Shuffle test",
        description=f"Randomly reshuffles who was in which group {N_PERM:,} times to see how often chance alone produces a gap this big.",
        result=text, agrees=agrees))

    if kind == "continuous":
        pooled = np.concatenate([c, t])
        lo_q, hi_q = np.percentile(pooled, [1, 99])
        c2, t2 = c[(c >= lo_q) & (c <= hi_q)], t[(t >= lo_q) & (t <= hi_q)]
        if len(c2) > 30 and len(t2) > 30:
            e = effect(c2, t2, "continuous")
            same_dir = np.sign(e["diff"]) == np.sign(t.mean() - c.mean())
            text, agrees = _verdict_text(e["p"], main_clear)
            out.append(Robustness(
                name="Without extreme values",
                description="Re-runs the comparison after removing the top and bottom 1% of values.",
                result=text, agrees=bool(agrees and same_dir)))
    return out


# -------------------------------------------------------------------- research

def segment_scan(sub: pd.DataFrame, spec: ABTestSpec, kind: str) -> tuple[list[Segment], str]:
    gs = sub[spec.group_col].astype(str)
    rows: list[dict] = []
    for col in sub.columns:
        if col in (spec.group_col, spec.outcome_col) or is_idlike(sub, col):
            continue
        s = sub[col]
        if pd.api.types.is_float_dtype(s) or not (2 <= s.nunique(dropna=True) <= MAX_LEVELS):
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
        return [], "No slices of your data were large enough to compare reliably."

    p_adj = multipletests([r["p_raw"] for r in rows], method="holm")[1]
    segs = [Segment(**r, p_adj=float(pa), flagged=bool(pa < ALPHA)) for r, pa in zip(rows, p_adj)]
    segs.sort(key=lambda s: s.p_raw)
    m = len(segs)
    flagged = sum(s.flagged for s in segs)
    note = (f"We compared {m} slices. If nothing were going on, about {m * ALPHA:.1f} of them would still look "
            f"'significant' by luck at the usual 5% cutoff, so the chance figures below are adjusted for that. ")
    note += (f"{flagged} slice{'s' if flagged != 1 else ''} still stand{'s' if flagged == 1 else ''} out after adjusting. "
             "Treat them as ideas to test, not conclusions." if flagged
             else "None stand out after adjusting, so there's no good evidence the effect differs by slice.")
    return segs, note


# ------------------------------------------------------------------------ quick

def quick_caveats(rep: Report) -> list[str]:
    cav = [c.message for c in rep.checks if c.status == "warn"]
    cav.append("This is a quick look. The result wasn't cross-checked with other methods, so treat it as a "
               "directional read, not a final answer.")
    cav.append("Read it as cause-and-effect only if people were assigned to versions at random.")
    if rep.verdict == "no_clear_difference":
        cav.append("'No clear difference' is not proof of no effect. The test may simply be too small to see it.")
    cav.append("If this will drive a real decision, rerun it as 'Need the right answer'.")
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
            rep.robust_summary = (f"Not robust: {len(bad)} of {n} cross-checks reach{'es' if len(bad) == 1 else ''} a different conclusion. "
                                  "Don't rely on this result until that is understood.")
        else:
            rep.robust_summary = f"Holds up: all {n} cross-checks reach the same conclusion."
        rep.review_items = [c_.message for c_ in rep.checks if c_.status == "warn"]
        if bad:
            rep.review_items.append(rep.robust_summary)
    elif mode == "research":
        rep.segments, rep.segment_note = segment_scan(sub, spec, kind)
    return rep
