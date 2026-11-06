"""The A/B test analysis: one tested recipe, no improvisation.

Every number here comes from ordinary statistics code. The language model never
touches it.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from .checks import detect_outcome_kind, has_blocking_failure, run_checks, validate_spec_against_data
from .interpret import chance_phrase, fmt_num, fmt_pct, humanize
from .knowledge import attach_method_info, item
from .modes import SEED, apply_mode, effect
from .schemas import ABTestSpec, ArmSummary, Check, Report
from .script import build_script

Z_POWER = 1.96 + 0.8416  # 95% confidence + 80% power


def analyze(df: pd.DataFrame, spec: ABTestSpec, csv_name: str = "your_file.csv",
            mode: str = "quick", data_sha: str = "") -> Report:
    problems = validate_spec_against_data(df, spec)
    if problems:
        r = _blocked("Can't answer yet.", problems, spec)
        r.mode = mode
        return r

    checks = run_checks(df, spec)
    if has_blocking_failure(checks):
        r = _blocked("Can't answer yet.", [c.message for c in checks if c.status == "fail"], spec, checks)
        r.mode = mode
        return r

    g = df[spec.group_col].astype(str)
    sub = df[g.isin([spec.control, spec.treatment])].dropna(subset=[spec.outcome_col])
    gs = sub[spec.group_col].astype(str)
    c = sub.loc[gs == spec.control, spec.outcome_col].astype(float).to_numpy()
    t = sub.loc[gs == spec.treatment, spec.outcome_col].astype(float).to_numpy()
    kind = detect_outcome_kind(sub[spec.outcome_col])

    rep = _binary(spec, c, t, checks) if kind == "binary" else _continuous(spec, c, t, checks)
    rep = apply_mode(rep, sub, spec, kind, mode)
    rep = attach_method_info(rep, kind, mode, checks, c, t)

    versions = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                "scipy": scipy.__version__, "statsmodels": statsmodels.__version__}
    rep.repro = {"data_sha256": data_sha, "rows_used": int(len(sub)), "seed": SEED, "versions": versions}
    header = (f"# Dashboarder reproduction script\n"
              f"# Data fingerprint (sha256): {data_sha or 'n/a'}\n"
              f"# Built with: " + ", ".join(f"{k} {v}" for k, v in versions.items()) + "\n\n")
    rep.script = build_script(spec, kind, csv_name, mode, header)
    return rep


# --------------------------------------------------------------------------- binary

def _binary(spec, c, t, checks) -> Report:
    nc, nt = len(c), len(t)
    xc, xt = int(c.sum()), int(t.sum())
    pc, pt = xc / nc, xt / nt
    e = effect(c, t, "binary")
    rel = e["diff"] / pc if pc > 0 else float("nan")
    se = float(np.sqrt(pc * (1 - pc) / nc + pt * (1 - pt) / nt))

    arms = []
    for name, x, n in ((spec.control, xc, nc), (spec.treatment, xt, nt)):
        a, b = proportion_confint(x, n, method="wilson")
        arms.append(ArmSummary(name=name, n=n, estimate=x / n, ci_low=float(a), ci_high=float(b)))

    extra = [
        item(f"{spec.control} rate", fmt_pct(pc), "rate", "Successes ÷ rows", f"c.sum() / len(c)  # {xc} / {nc}"),
        item(f"{spec.treatment} rate", fmt_pct(pt), "rate", "Successes ÷ rows", f"t.sum() / len(t)  # {xt} / {nt}"),
    ]
    return _assemble(spec=spec, kind="rate", label=f"{humanize(spec.outcome_col)} rate", checks=checks, arms=arms,
                     e=e, rel=rel, mde=Z_POWER * se, unit_pts=True, method_kind="binary", audit_extra=extra,
                     arm_ci_code="proportion_confint(x, n, method='wilson')")


# ------------------------------------------------------------------------ continuous

def _continuous(spec, c, t, checks) -> Report:
    nc, nt = len(c), len(t)
    mc, mt = float(c.mean()), float(t.mean())
    e = effect(c, t, "continuous")
    rel = e["diff"] / mc if mc != 0 else float("nan")
    se = float(np.sqrt(t.var(ddof=1) / nt + c.var(ddof=1) / nc))

    arms = []
    for name, s in ((spec.control, c), (spec.treatment, t)):
        m, sem = float(s.mean()), float(stats.sem(s))
        h = sem * stats.t.ppf(0.975, len(s) - 1)
        arms.append(ArmSummary(name=name, n=len(s), estimate=m, ci_low=m - h, ci_high=m + h))

    extra = [
        item(f"{spec.control} average", fmt_num(mc), "mean", "Mean of the outcome", f"c.mean()  # n={nc}"),
        item(f"{spec.treatment} average", fmt_num(mt), "mean", "Mean of the outcome", f"t.mean()  # n={nt}"),
    ]
    return _assemble(spec=spec, kind="average", label=f"average {humanize(spec.outcome_col)}", checks=checks,
                     arms=arms, e=e, rel=rel, mde=Z_POWER * se, unit_pts=False, method_kind="continuous",
                     audit_extra=extra, arm_ci_code="mean ± t(0.975, n-1) × sem")


# ---------------------------------------------------------------------------- shared

def _assemble(*, spec, kind, label, checks, arms, e, rel, mde, unit_pts, method_kind, audit_extra,
              arm_ci_code) -> Report:
    diff, lo, hi, p_value = e["diff"], e["lo"], e["hi"], e["p"]
    clear = (lo > 0 or hi < 0) and p_value < 0.05

    def d(x):  # unsigned amount
        return f"{abs(x) * 100:.1f} pts" if unit_pts else fmt_num(abs(x))

    def ds(x):  # signed amount
        return ("+" if x >= 0 else "−") + d(x)

    def val(x):
        return fmt_pct(x) if unit_pts else fmt_num(x)

    direction = "raised" if diff > 0 else "lowered"
    a_c, a_t = arms[0].estimate, arms[1].estimate
    named = f"{spec.control} {val(a_c)} → {spec.treatment} {val(a_t)}"
    rel_txt = "" if rel != rel else f" ({'+' if rel >= 0 else '−'}{abs(rel) * 100:.1f}%)"
    warn_n = sum(1 for c in checks if c.status == "warn")

    if clear:
        verdict = "clear_difference"
        headline = f"{spec.treatment} {direction} the {label}."
        answer = f"{named} · {ds(diff)}{rel_txt}"
        how_sure = (f"Likely range: {ds(lo)} to {ds(hi)}. If nothing changed, a gap this big "
                    f"would appear {chance_phrase(p_value)}.")
    else:
        verdict = "no_clear_difference"
        headline = f"No clear difference in {label}."
        answer = f"{named}. " + ("Leaning that way, but not conclusive." if p_value < 0.2 else "Too close to call.")
        how_sure = (f"Likely range: {ds(lo)} to {ds(hi)}. This test only spots gaps of {d(mde)} or more. "
                    f"Not proof of no effect.")
    if warn_n:
        how_sure += f" {warn_n} data warning{'s' if warn_n > 1 else ''}: see checks."

    causal = ("Cause and effect only if people got versions at random. "
              "If they chose, it just shows the two moved together.")
    cannot = ["Why it happened.", "Whether it lasts.", "People outside this file."]

    ci_key, p_key = ("ci_binary", "p_binary") if method_kind == "binary" else ("ci_mean", "p_mean")
    ci_code = ("confint_proportions_2indep(xt, nt, xc, nc, method='newcomb')" if method_kind == "binary"
               else "stats.ttest_ind(t, c, equal_var=False).confidence_interval(0.95)")
    p_code = ("test_proportions_2indep(xt, nt, xc, nc, method='score').pvalue" if method_kind == "binary"
              else "stats.ttest_ind(t, c, equal_var=False).pvalue")
    arm_ci = [item(f"{a.name} range", f"{val(a.ci_low)} to {val(a.ci_high)}", "arm_ci",
                   "95% interval" + (" (Wilson)" if method_kind == "binary" else " (t)"), arm_ci_code)
              for a in arms]

    audit = [
        *audit_extra, *arm_ci,
        item("Difference", ds(diff), "diff", "Treatment minus control", "t.mean() - c.mean()"),
        *([item("Relative change", f"{'+' if rel >= 0 else '−'}{abs(rel) * 100:.1f}%", "rel",
                "Difference ÷ control", "(t.mean() - c.mean()) / c.mean()")] if rel == rel else []),
        item("95% range for the difference", f"{ds(lo)} to {ds(hi)}", ci_key,
             "Newcombe interval" if method_kind == "binary" else "Welch t interval", ci_code),
        item("p-value", f"{p_value:.4g}", p_key,
             "Pooled z-test" if method_kind == "binary" else "Welch t-test", p_code),
        item("Smallest gap this test can spot", d(mde), "mde", "2.80 × standard error", "2.8016 * se_diff"),
        item("'Clear' rule", "Met" if clear else "Not met", "rule", "Range excludes 0 and p < 0.05",
             "(low > 0 or high < 0) and p < 0.05"),
    ]

    return Report(
        headline=headline, answer=answer, how_sure=how_sure, verdict=verdict, causal_label=causal,
        checks=checks, arms=arms,
        effect={"diff": diff, "ci_low": lo, "ci_high": hi, "p_value": p_value, "rel": None if rel != rel else rel,
                "mde": mde, "unit_pts": unit_pts},
        outcome_kind=kind, outcome_label=label, cannot_tell=cannot, audit=audit, spec=spec,
    )


def _blocked(headline: str, reasons: list[str], spec, checks: list[Check] | None = None) -> Report:
    return Report(
        headline=headline, answer=" ".join(reasons), how_sure="", verdict="blocked",
        causal_label="", checks=checks or [Check(name="Question fits data", status="fail", message=r) for r in reasons],
        cannot_tell=[], spec=spec,
    )
