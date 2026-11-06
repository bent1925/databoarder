"""Two groups that people sorted themselves into: compare people who look alike.

This can NEVER be read as cause and effect. It makes the groups comparable on the columns that are
in the file; anything that is not in the file (motivation, intent, taste) is still mixed in.
That is why the verdict is worded "goes with", never "raised", and why the report says so.

Main answer: the difference after adjusting for the chosen columns (linear regression with
heteroskedasticity-robust HC3 errors). Cross-checks (right answer): inverse-probability weighting,
only people with a realistic chance of being in either group, and the unadjusted gap.
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .before_after import _cross_text, parse_dates
from .checks import detect_outcome_kind, has_blocking_failure, is_idlike, run_checks
from .interpret import chance_phrase, fmt_num, fmt_pct, humanize
from .knowledge import attach_method_info_obs, item
from .modes import ALPHA, MAX_LEVELS, MIN_SEG, SEED, effect
from .schemas import ArmSummary, Check, ObservationalSpec, Report, Robustness, Segment
from .script import build_obs_script

Z_POWER = 1.96 + 0.8416
MAX_COV_LEVELS = 20
PS_CLIP = (0.05, 0.95)
OVERLAP_KEEP = (0.10, 0.90)


# ------------------------------------------------------------------ columns to compare on

def _datelike(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return False
    vals = s.dropna()
    if len(vals) == 0:
        return False
    d = parse_dates(vals.sample(min(len(vals), 200), random_state=0))
    return bool(d.notna().mean() >= 0.9 and (d.dropna().dt.year >= 1900).all())


# Columns that look like things measured AFTER the choice. Adjusting for them would hide part of the real effect.
POST_HINTS = re.compile(r"next|after|post|future|follow|\d+\s*d$|_\d+d|\d+_?days|renew|retain|churn|convert|purchas"
                        r"|outcome|result|revenue", re.I)


def covariate_candidates(df: pd.DataFrame, spec) -> list[dict]:
    """Columns that could be used to compare people, and whether we'd tick them by default."""
    g = df[spec.group_col].astype(str)
    sub = df[g.isin([spec.control, spec.treatment])]
    out = []
    for col in df.columns:
        if col in (spec.group_col, spec.outcome_col) or is_idlike(df, col):
            continue
        s = sub[col]
        if s.isna().mean() > 0.30 or s.nunique(dropna=True) < 2 or _datelike(s):
            continue
        if not (pd.api.types.is_numeric_dtype(s) or s.nunique(dropna=True) <= MAX_COV_LEVELS):
            continue
        post = bool(POST_HINTS.search(col))
        out.append({"name": col, "checked": not post, "note": "Might be measured after they chose" if post else ""})
    return out


def default_covariates(df: pd.DataFrame, spec) -> list[str]:
    """Columns that describe the person before they chose (not IDs, dates, or likely outcomes)."""
    return [c["name"] for c in covariate_candidates(df, spec) if c["checked"]]


def validate_spec_obs(df: pd.DataFrame, spec: ObservationalSpec) -> list[str]:
    for col in (spec.group_col, spec.outcome_col, *spec.covariates):
        if col not in df.columns:
            return [f"Column '{col}' isn't in your file."]
    levels = set(df[spec.group_col].dropna().astype(str).unique())
    probs = [f"'{v}' isn't a value of '{spec.group_col}'." for v in (spec.control, spec.treatment) if v not in levels]
    if probs:
        return probs
    if spec.control == spec.treatment:
        return ["Pick two different groups."]
    if spec.outcome_col == spec.group_col:
        return ["The result can't be the group column."]
    if not pd.api.types.is_numeric_dtype(df[spec.outcome_col]):
        return [f"'{spec.outcome_col}' isn't a number."]
    if {spec.group_col, spec.outcome_col} & set(spec.covariates):
        return ["The group and the result can't be used to compare people."]
    return []


# --------------------------------------------------------------------------- the maths

def _design(frame: pd.DataFrame, covs: list[str]) -> pd.DataFrame:
    parts = []
    for c in covs:
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s):
            sd = s.std(ddof=0)
            if sd > 0:
                parts.append(((s - s.mean()) / sd).rename(c))
        else:
            dm = pd.get_dummies(s.astype(str), prefix=c, drop_first=True, dtype=float)
            parts.append(dm.loc[:, dm.std(ddof=0) > 0])
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)


def adjusted(frame: pd.DataFrame, spec, covs: list[str]) -> dict:
    """Treatment minus control, comparing like with like. Also the adjusted average per group."""
    t = (frame[spec.group_col].astype(str) == spec.treatment).to_numpy(float)
    y = frame[spec.outcome_col].to_numpy(float)
    Xc = _design(frame, covs).to_numpy(float)
    X = sm.add_constant(np.column_stack([t, Xc]), has_constant="add")
    res = sm.OLS(y, X).fit(cov_type="HC3", use_t=True)
    lo, hi = res.conf_int()[1]
    p = float(res.pvalues[1])
    xbar = Xc.mean(axis=0) if Xc.shape[1] else np.zeros(0)
    v0 = np.concatenate([[1.0, 0.0], xbar])
    v1 = np.concatenate([[1.0, 1.0], xbar])
    arms = []
    for v in (v0, v1):
        ci = res.t_test(v).conf_int()[0]
        arms.append((float(v @ res.params), float(ci[0]), float(ci[1])))
    return {"diff": float(res.params[1]), "lo": float(lo), "hi": float(hi), "p": p if np.isfinite(p) else 1.0,
            "se": float(res.bse[1]), "arms": arms}


def propensity(frame: pd.DataFrame, spec, covs: list[str]):
    """Chance of being in the treatment group, given the columns. None if it can't be fitted."""
    t = (frame[spec.group_col].astype(str) == spec.treatment).to_numpy(float)
    X = sm.add_constant(_design(frame, covs).to_numpy(float), has_constant="add")
    try:
        fit = sm.GLM(t, X, family=sm.families.Binomial()).fit()
        ps = np.asarray(fit.predict(X), float)
    except Exception:
        return t, None
    return t, ps if np.all(np.isfinite(ps)) else None


def ipw(frame: pd.DataFrame, spec, t, ps) -> dict:
    ps = np.clip(ps, *PS_CLIP)
    w = t / ps + (1 - t) / (1 - ps)
    y = frame[spec.outcome_col].to_numpy(float)
    res = sm.WLS(y, sm.add_constant(t), weights=w).fit(cov_type="HC1", use_t=True)
    lo, hi = res.conf_int()[1]
    p = float(res.pvalues[1])
    return {"diff": float(res.params[1]), "lo": float(lo), "hi": float(hi), "p": p if np.isfinite(p) else 1.0}


def evalue(diff, lo, hi, base, sd, binary) -> tuple[float, float]:
    """How strong would a hidden factor have to be, tied to both choosing and the result, to erase this?"""
    base = max(base, 1e-6)

    def rr(x):
        r = max((base + x) / base, 1e-9) if binary else float(np.exp(0.91 * x / sd))
        return 1 / r if r < 1 else r

    def ev(r):
        return r + np.sqrt(r * (r - 1))
    point = ev(rr(diff))
    ci = 1.0 if lo <= 0 <= hi else ev(rr(lo if diff > 0 else hi))
    return float(point), float(ci)


def _smd(frame: pd.DataFrame, spec, covs: list[str]) -> dict[str, float]:
    """How different the two groups were to start with, per column (0.1 = small, 0.5 = big)."""
    t = (frame[spec.group_col].astype(str) == spec.treatment).to_numpy(bool)
    out = {}
    for c in covs:
        X = _design(frame, [c]).to_numpy(float)
        if X.shape[1] == 0:
            continue
        a, b = X[t], X[~t]
        den = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2)
        den = np.where(den > 0, den, np.nan)
        d = np.abs(a.mean(axis=0) - b.mean(axis=0)) / den
        out[c] = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
    return out


# -------------------------------------------------------------------------------- report

def analyze(df: pd.DataFrame, spec: ObservationalSpec, csv_name: str = "your_file.csv", mode: str = "quick",
            data_sha: str = "") -> Report:
    problems = validate_spec_obs(df, spec)
    if problems:
        return _blocked("Can't answer yet.", problems, spec, mode)
    covs = list(spec.covariates) or default_covariates(df, spec)
    if not covs:
        return _blocked("Can't answer yet.", ["No other columns to compare people on."], spec, mode)
    spec = spec.model_copy(update={"covariates": covs})

    g = df[spec.group_col].astype(str)
    d0 = df[g.isin([spec.control, spec.treatment])].dropna(subset=[spec.outcome_col])
    kind_raw = detect_outcome_kind(d0[spec.outcome_col])
    sub = d0.dropna(subset=covs)
    gs = sub[spec.group_col].astype(str)
    n_c, n_t = int((gs == spec.control).sum()), int((gs == spec.treatment).sum())

    checks = [c for c in run_checks(df, spec) if c.name != "Group balance"]
    lost = len(d0) - len(sub)
    if lost:
        share = lost / max(len(d0), 1)
        checks.append(Check(name="Rows used", status="warn" if share > 0.2 else "pass",
                            message=f"{lost:,} rows left out for missing values in the compared columns ({share:.0%})."))
    if has_blocking_failure(checks):
        return _blocked("Can't answer yet.", [c.message for c in checks if c.status == "fail"], spec, mode, checks)

    e = adjusted(sub, spec, covs)
    t_arr, ps = propensity(sub, spec, covs)
    y_all = sub[spec.outcome_col].to_numpy(float)
    ev_pt, ev_ci = evalue(e["diff"], e["lo"], e["hi"], e["arms"][0][0], float(y_all.std(ddof=1)), kind_raw == "binary")

    smd = _smd(sub, spec, covs)
    worst = max(smd, key=smd.get) if smd else None
    if worst and smd[worst] >= 0.1:
        checks.append(Check(name="Groups differ", status="info",
                            message=f"They started furthest apart on '{worst}'. Adjusted for it."))
    else:
        checks.append(Check(name="Groups differ", status="pass", message="Already alike on the compared columns."))
    if ps is None:
        checks.append(Check(name="Comparable people", status="info", message="Couldn't measure overlap."))
    else:
        extreme = float(np.mean((ps < PS_CLIP[0]) | (ps > PS_CLIP[1])))
        checks.append(Check(
            name="Comparable people", status="warn" if extreme > 0.05 else "pass",
            message=(f"{extreme:.0%} of people almost always (or never) chose it. Few true comparisons."
                     if extreme > 0.05 else "Plenty of comparable people on both sides.")))
    checks.append(Check(name="Hidden factors", status="info",
                        message=f"A hidden factor would need to be about {ev_pt:.1f}× tied to both choosing and "
                                f"the result to erase this."))

    kind = "rate" if kind_raw == "binary" else "average"
    unit_pts = kind == "rate"
    label = f"{humanize(spec.outcome_col)} rate" if unit_pts else f"average {humanize(spec.outcome_col)}"

    rep = _assemble(spec, covs, e, ev_pt, ev_ci, n_c, n_t, kind, label, unit_pts, checks, sub)
    rep = _apply_mode(rep, sub, spec, covs, kind_raw, e, t_arr, ps, unit_pts, mode)
    rep = attach_method_info_obs(rep, mode, checks, covs)

    versions = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                "scipy": scipy.__version__, "statsmodels": statsmodels.__version__}
    rep.repro = {"data_sha256": data_sha, "rows_used": int(len(sub)), "seed": SEED, "versions": versions}
    header = (f"# Dashboarder reproduction script\n# Data fingerprint (sha256): {data_sha or 'n/a'}\n"
              f"# Built with: " + ", ".join(f"{k} {v}" for k, v in versions.items()) + "\n\n")
    rep.script = build_obs_script(spec, kind_raw, csv_name, mode, header)
    return rep


def _fmt(unit_pts: bool):
    def d(x):
        return f"{abs(x) * 100:.1f} pts" if unit_pts else fmt_num(abs(x))

    def ds(x):
        return ("+" if x >= 0 else "−") + d(x)

    def val(x):
        return fmt_pct(x) if unit_pts else fmt_num(x)
    return d, ds, val


def _assemble(spec, covs, e, ev_pt, ev_ci, n_c, n_t, kind, label, unit_pts, checks, sub) -> Report:
    diff, lo, hi, p_value = e["diff"], e["lo"], e["hi"], e["p"]
    clear = (lo > 0 or hi < 0) and p_value < ALPHA
    d, ds, val = _fmt(unit_pts)
    (m0, l0, h0), (m1, l1, h1) = e["arms"]
    rel = diff / m0 if m0 != 0 else float("nan")
    mde = Z_POWER * e["se"]
    arms = [ArmSummary(name=spec.control, n=n_c, estimate=m0, ci_low=l0, ci_high=h0),
            ArmSummary(name=spec.treatment, n=n_t, estimate=m1, ci_low=l1, ci_high=h1)]

    named = f"{spec.control} {val(m0)} → {spec.treatment} {val(m1)}"
    rel_txt = "" if rel != rel else f" ({'+' if rel >= 0 else '−'}{abs(rel) * 100:.1f}%)"
    cols_txt = f"Compared people alike on {len(covs)} column{'s' if len(covs) != 1 else ''}."
    warn_n = sum(1 for c in checks if c.status == "warn")

    if clear:
        verdict = "clear_difference"
        headline = f"{spec.treatment} goes with a {'higher' if diff > 0 else 'lower'} {label}."
        answer = f"{named} · {ds(diff)}{rel_txt}"
        how_sure = (f"Likely range: {ds(lo)} to {ds(hi)}. If there were no difference, a gap this big "
                    f"would appear {chance_phrase(p_value)}. {cols_txt}")
    else:
        verdict = "no_clear_difference"
        headline = f"No clear difference in {label} between like-for-like people."
        answer = f"{named}. " + ("Leaning that way, but not conclusive." if p_value < 0.2 else "Too close to call.")
        how_sure = (f"Likely range: {ds(lo)} to {ds(hi)}. This only spots gaps of {d(mde)} or more. "
                    f"Not proof of no effect. {cols_txt}")
    if warn_n:
        how_sure += f" {warn_n} data warning{'s' if warn_n > 1 else ''}: see checks."

    causal = ("Not cause and effect. People who chose may differ in ways this file doesn't show, "
              "and adjusting can't fix that.")
    if spec.assignment == "unsure":
        causal += " You weren't sure how people got it, so we assumed they chose."
    cannot = ["Why the groups differ.", "Anything that isn't a column in your file.",
              "What would happen if others switched."]

    cc = detect_outcome_kind(sub[spec.outcome_col])
    ci_code = "sm.OLS(y, [1, treated, covariates]).fit(cov_type='HC3', use_t=True).conf_int()"
    audit = [
        item("Matched on", ", ".join(covs), "obs_cols", "Columns made comparable", "covs = " + repr(covs)),
        item(f"{spec.control} (adjusted)", val(m0), "obs_level", "Predicted average at the typical mix of people",
             "[1, 0, x̄] @ params"),
        item(f"{spec.treatment} (adjusted)", val(m1), "obs_level", "Predicted average at the typical mix of people",
             "[1, 1, x̄] @ params"),
        item("Difference", ds(diff), "obs_diff", "Treatment coefficient", "params[1]"),
        *([item("Relative change", f"{'+' if rel >= 0 else '−'}{abs(rel) * 100:.1f}%", "rel", "Difference ÷ control",
                "params[1] / control_adjusted")] if rel == rel else []),
        item("95% range for the difference", f"{ds(lo)} to {ds(hi)}", "obs_ci", "Robust (HC3) t interval", ci_code),
        item("p-value", f"{p_value:.4g}", "obs_p", "Robust (HC3) t-test on the treatment coefficient", "pvalues[1]"),
        item("Smallest gap this can spot", d(mde), "mde", "2.80 × standard error", "2.8016 * bse[1]"),
        item("'Clear' rule", "Met" if clear else "Not met", "rule", "Range excludes 0 and p < 0.05",
             "(low > 0 or high < 0) and p < 0.05"),
        item("Hidden-factor strength", f"{ev_pt:.2f}×", "evalue", "E-value (risk-ratio scale)",
             "evalue(diff, low, high, base, sd, binary)"),
    ]
    return Report(
        analysis="observational", design=spec.assignment, headline=headline, answer=answer, how_sure=how_sure,
        verdict=verdict, causal_label=causal, checks=checks, arms=arms,
        effect={"diff": diff, "ci_low": lo, "ci_high": hi, "p_value": p_value, "rel": None if rel != rel else rel,
                "mde": mde, "unit_pts": unit_pts, "evalue": ev_pt, "evalue_ci": ev_ci},
        outcome_kind=kind, outcome_label=label, cannot_tell=cannot, audit=audit, spec=spec, adjusted_for=covs,
    )


def _blocked(headline, reasons, spec, mode, checks=None) -> Report:
    return Report(analysis="observational", design=getattr(spec, "assignment", "chose"), headline=headline,
                  answer=" ".join(reasons), how_sure="", verdict="blocked", causal_label="", mode=mode,
                  checks=checks or [Check(name="Question fits data", status="fail", message=r) for r in reasons],
                  cannot_tell=[], spec=spec)


# --------------------------------------------------------------------------------- modes

def robustness_obs(sub, spec, covs, kind_raw, main_clear, main_diff, t_arr, ps, unit_pts):
    _, ds, _ = _fmt(unit_pts)
    out: list[Robustness] = []
    skipped: list[str] = []

    if ps is None:
        skipped.append("Weighting check skipped: couldn't model who chose what.")
        skipped.append("Comparable-people check skipped: couldn't model who chose what.")
    else:
        w = ipw(sub, spec, t_arr, ps)
        text, agrees, weak = _cross_text(w, main_clear, main_diff, ds)
        out.append(Robustness(key="ipw", name="Weighting check",
                              description="Gives people who look unlikely to have chosen it more weight, to balance the groups.",
                              result=text, agrees=agrees, weak=weak, p_value=w["p"]))
        keep = (ps >= OVERLAP_KEEP[0]) & (ps <= OVERLAP_KEEP[1])
        kept = sub[keep]
        gk = kept[spec.group_col].astype(str)
        if (gk == spec.control).sum() >= 30 and (gk == spec.treatment).sum() >= 30:
            o = adjusted(kept, spec, covs)
            text, agrees, weak = _cross_text(o, main_clear, main_diff, ds)
            out.append(Robustness(key="overlap", name="Only comparable people",
                                  description="Drops people who almost always (or never) chose it, then repeats.",
                                  result=text, agrees=agrees, weak=weak, p_value=o["p"]))
        else:
            skipped.append("Comparable-people check skipped: too few people left on one side.")

    c = sub.loc[sub[spec.group_col].astype(str) == spec.control, spec.outcome_col].to_numpy(float)
    t = sub.loc[sub[spec.group_col].astype(str) == spec.treatment, spec.outcome_col].to_numpy(float)
    u = effect(c, t, "binary" if kind_raw == "binary" else "continuous")
    text, agrees, weak = _cross_text(u, main_clear, main_diff, ds)
    out.append(Robustness(key="unadjusted", name="Without matching people up",
                          description="The raw gap. If it differs a lot, who chose it matters a lot.",
                          result=text, agrees=agrees, weak=weak, p_value=u["p"]))
    return out, skipped


def segment_scan_obs(sub: pd.DataFrame, spec, covs, kind_raw) -> tuple[list[Segment], str]:
    gs = sub[spec.group_col].astype(str)
    rows = []
    for col in sub.columns:
        if col in (spec.group_col, spec.outcome_col) or is_idlike(sub, col):
            continue
        s = sub[col]
        if pd.api.types.is_numeric_dtype(s) or not (2 <= s.nunique(dropna=True) <= MAX_LEVELS):
            continue
        s_str = s.astype(str)
        covs_here = [c for c in covs if c != col]
        if not covs_here:
            continue
        for level in sorted(s.dropna().astype(str).unique()):
            m = s_str == level
            n_c, n_t = int((m & (gs == spec.control)).sum()), int((m & (gs == spec.treatment)).sum())
            if n_c < MIN_SEG or n_t < MIN_SEG:
                continue
            e = adjusted(sub[m], spec, covs_here)
            rows.append({"column": col, "level": level, "n_control": n_c, "n_treatment": n_t, "diff": e["diff"],
                         "ci_low": e["lo"], "ci_high": e["hi"], "p_raw": e["p"]})
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


def _apply_mode(rep: Report, sub, spec, covs, kind_raw, e, t_arr, ps, unit_pts, mode) -> Report:
    rep.mode = mode
    main_clear = rep.verdict == "clear_difference"
    if mode == "quick":
        cav = [c.message for c in rep.checks if c.status == "warn"]
        cav += ["Quick look: not cross-checked.",
                "Not cause and effect: hidden differences between people could explain it.",
                "Adjusted only for columns in the file."]
        if rep.verdict == "no_clear_difference":
            cav.append("'No clear difference' isn't 'no effect'. The data may be too small.")
        cav.append("Big decision? Rerun as 'Need the right answer'.")
        rep.caveats = cav
    elif mode == "right_answer":
        rep.robustness, skipped = robustness_obs(sub, spec, covs, kind_raw, main_clear, e["diff"], t_arr, ps, unit_pts)
        bad = [r for r in rep.robustness if not r.agrees]
        weak = [r for r in rep.robustness if r.weak]
        n = len(rep.robustness)
        if bad:
            rep.robust_summary = f"Not robust: {len(bad)} of {n} cross-checks disagree. Don't rely on this yet."
        elif weak:
            rep.robust_summary = (f"Holds up so far: {n - len(weak)} of {n} cross-checks confirm it; "
                                  f"{len(weak)} can't (too little data).")
        else:
            rep.robust_summary = f"Holds up: all {n} cross-checks agree."
        rep.review_items = [c.message for c in rep.checks if c.status == "warn"] + skipped
        if bad:
            rep.review_items.append(rep.robust_summary)
    elif mode == "research":
        rep.segments, rep.segment_note = segment_scan_obs(sub, spec, covs, kind_raw)
    return rep
