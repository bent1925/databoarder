"""The A/B test analysis: one tested recipe, no improvisation.

Every number here comes from ordinary statistics code. The language model never
touches it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    test_proportions_2indep,
)

from .checks import detect_outcome_kind, has_blocking_failure, run_checks, validate_spec_against_data
from .interpret import chance_phrase, fmt_num, fmt_pct, humanize
from .modes import apply_mode
from .schemas import ABTestSpec, ArmSummary, AuditItem, Check, Report

ALPHA = 0.05
Z_POWER = 1.96 + 0.8416  # 95% confidence + 80% power


def _script(spec: ABTestSpec, kind: str, csv_name: str) -> str:
    """A standalone Python script an analyst can run to reproduce the report."""
    head = f'''import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, test_proportions_2indep

df = pd.read_csv({csv_name!r})
df = df[df[{spec.group_col!r}].astype(str).isin([{spec.control!r}, {spec.treatment!r}])]
df = df.dropna(subset=[{spec.outcome_col!r}])
c = df.loc[df[{spec.group_col!r}].astype(str) == {spec.control!r}, {spec.outcome_col!r}]
t = df.loc[df[{spec.group_col!r}].astype(str) == {spec.treatment!r}, {spec.outcome_col!r}]
'''
    if kind == "binary":
        body = '''
low, high = confint_proportions_2indep(t.sum(), len(t), c.sum(), len(c), method="newcomb")
p_value = test_proportions_2indep(t.sum(), len(t), c.sum(), len(c), method="wald").pvalue
print("control rate", round(c.mean(), 4), "| treatment rate", round(t.mean(), 4))
print("difference", round(t.mean() - c.mean(), 4), "| 95% range", (round(float(low), 4), round(float(high), 4)), "| p-value", round(float(p_value), 5))
'''
    else:
        body = '''
res = stats.ttest_ind(t, c, equal_var=False)
ci = res.confidence_interval(0.95)
print("control mean", round(c.mean(), 4), "| treatment mean", round(t.mean(), 4))
print("difference", round(t.mean() - c.mean(), 4), "| 95% range", (round(float(ci.low), 4), round(float(ci.high), 4)), "| p-value", round(float(res.pvalue), 5))
'''
    return head + body


def analyze(df: pd.DataFrame, spec: ABTestSpec, csv_name: str = "your_file.csv", mode: str = "quick") -> Report:
    problems = validate_spec_against_data(df, spec)
    if problems:
        r = _blocked("The question doesn't fit your data.", problems, spec)
        r.mode = mode
        return r

    checks = run_checks(df, spec)
    if has_blocking_failure(checks):
        r = _blocked("There isn't enough data to answer this reliably.",
                     [c.message for c in checks if c.status == "fail"], spec, checks)
        r.mode = mode
        return r

    g = df[spec.group_col].astype(str)
    sub = df[g.isin([spec.control, spec.treatment])].dropna(subset=[spec.outcome_col])
    gs = sub[spec.group_col].astype(str)
    c = sub.loc[gs == spec.control, spec.outcome_col].astype(float)
    t = sub.loc[gs == spec.treatment, spec.outcome_col].astype(float)
    kind = detect_outcome_kind(sub[spec.outcome_col])

    rep = _binary(spec, c, t, checks, csv_name) if kind == "binary" else _continuous(spec, c, t, checks, csv_name)
    return apply_mode(rep, sub, spec, kind, mode)


# --------------------------------------------------------------------------- binary

def _binary(spec, c, t, checks, csv_name) -> Report:
    nc, nt = len(c), len(t)
    xc, xt = int(c.sum()), int(t.sum())
    pc, pt = xc / nc, xt / nt
    diff = pt - pc
    lo, hi = confint_proportions_2indep(xt, nt, xc, nc, method="newcomb", compare="diff")
    p_value = float(test_proportions_2indep(xt, nt, xc, nc, method="wald", compare="diff").pvalue)
    rel = diff / pc if pc > 0 else float("nan")
    se = float(np.sqrt(pc * (1 - pc) / nc + pt * (1 - pt) / nt))
    mde = Z_POWER * se

    arms = []
    for name, x, n in ((spec.control, xc, nc), (spec.treatment, xt, nt)):
        a, b = proportion_confint(x, n, method="wilson")
        arms.append(ArmSummary(name=name, n=n, estimate=x / n, ci_low=float(a), ci_high=float(b)))

    label = f"{humanize(spec.outcome_col)} rate"
    clear = (lo > 0 or hi < 0) and p_value < ALPHA
    return _assemble(
        spec=spec, kind="rate", label=label, checks=checks, arms=arms,
        diff=diff, lo=float(lo), hi=float(hi), p_value=p_value, rel=rel, mde=mde, clear=clear,
        unit_pts=True, csv_name=csv_name, method_kind="binary",
        method_summary=("Compared the share of rows with a 1 in each group. The gap's range comes from the "
                        "Newcombe method; the chance statement comes from a two-proportion z-test."),
        audit_extra=[
            AuditItem(label=f"{spec.control} rate", value=fmt_pct(pc), method="Successes divided by rows",
                      code=f"c.sum() / len(c)  # {xc} / {nc}"),
            AuditItem(label=f"{spec.treatment} rate", value=fmt_pct(pt), method="Successes divided by rows",
                      code=f"t.sum() / len(t)  # {xt} / {nt}"),
        ],
    )


# ------------------------------------------------------------------------ continuous

def _continuous(spec, c, t, checks, csv_name) -> Report:
    nc, nt = len(c), len(t)
    mc, mt = float(c.mean()), float(t.mean())
    diff = mt - mc
    res = stats.ttest_ind(t, c, equal_var=False)
    ci = res.confidence_interval(0.95)
    lo, hi = float(ci.low), float(ci.high)
    p_value = float(res.pvalue)
    rel = diff / mc if mc != 0 else float("nan")
    se = float(np.sqrt(t.var(ddof=1) / nt + c.var(ddof=1) / nc))
    mde = Z_POWER * se

    arms = []
    for name, s in ((spec.control, c), (spec.treatment, t)):
        m, sem = float(s.mean()), float(stats.sem(s))
        h = sem * stats.t.ppf(0.975, len(s) - 1)
        arms.append(ArmSummary(name=name, n=len(s), estimate=m, ci_low=m - h, ci_high=m + h))

    label = f"average {humanize(spec.outcome_col)}"
    clear = (lo > 0 or hi < 0) and p_value < ALPHA
    return _assemble(
        spec=spec, kind="average", label=label, checks=checks, arms=arms,
        diff=diff, lo=lo, hi=hi, p_value=p_value, rel=rel, mde=mde, clear=clear,
        unit_pts=False, csv_name=csv_name, method_kind="continuous",
        method_summary=("Compared the average in each group using Welch's t-test, which does not assume the "
                        "two groups are equally spread out."),
        audit_extra=[
            AuditItem(label=f"{spec.control} average", value=fmt_num(mc), method="Mean of the outcome",
                      code=f"c.mean()  # n={nc}"),
            AuditItem(label=f"{spec.treatment} average", value=fmt_num(mt), method="Mean of the outcome",
                      code=f"t.mean()  # n={nt}"),
        ],
    )


# ---------------------------------------------------------------------------- shared

def _assemble(*, spec, kind, label, checks, arms, diff, lo, hi, p_value, rel, mde, clear,
              unit_pts, csv_name, method_kind, method_summary, audit_extra) -> Report:
    def d(x):  # format a difference
        return f"{abs(x) * 100:.1f} percentage points" if unit_pts else fmt_num(abs(x))

    def d_signed(x):
        s = "+" if x >= 0 else "−"
        return f"{s}{abs(x) * 100:.1f} pts" if unit_pts else f"{s}{fmt_num(abs(x))}"

    direction = "raised" if diff > 0 else "lowered"
    a_c, a_t = arms[0].estimate, arms[1].estimate
    from_to = (f"{fmt_pct(a_c)} → {fmt_pct(a_t)}" if unit_pts else f"{fmt_num(a_c)} → {fmt_num(a_t)}")
    rel_txt = "" if (rel != rel) else f" ({'+' if rel >= 0 else '−'}{abs(rel) * 100:.1f}% relative)"
    warn_n = sum(1 for c in checks if c.status == "warn")

    if clear:
        verdict = "clear_difference"
        headline = f"{spec.treatment} {direction} the {label}."
        answer = (f"Yes. Compared with {spec.control}, {spec.treatment} {direction} the {label} by "
                  f"{d(diff)}{rel_txt}: {from_to}.")
        how_sure = (f"We're 95% sure the true change is between {d_signed(lo)} and {d_signed(hi)}. "
                    f"If there were really no difference, a gap this big would show up by chance {chance_phrase(p_value)}.")
    else:
        verdict = "no_clear_difference"
        headline = f"No clear difference in the {label}."
        t_val, c_val = from_to.split(" → ")[1], from_to.split(" → ")[0]
        if p_value < 0.2:
            answer = (f"Not conclusive. {spec.treatment} came out at {t_val} versus {c_val} for {spec.control}, "
                      f"which points toward a difference, but chance alone could still produce a gap this size: "
                      f"it would show up {chance_phrase(p_value)} even if nothing changed.")
        else:
            answer = (f"We can't tell them apart. {spec.treatment} came out at {t_val} versus "
                      f"{c_val} for {spec.control}, but a gap that size is easily explained by chance.")
        how_sure = (f"The true change could be anywhere from {d_signed(lo)} to {d_signed(hi)}. "
                    f"This test could only reliably detect a difference of about {d(mde)} or more; "
                    f"anything smaller would be invisible to it. That is not the same as proof of no effect.")
    if warn_n:
        how_sure += f" Note: {warn_n} data check{'s' if warn_n > 1 else ''} raised a warning, so read this with extra care."

    causal = ("Cause or coincidence? If people were assigned to a version at random (like a coin flip), this "
              "difference can be read as caused by the change. If people chose their own version, it only "
              "shows the two moved together, and something else could explain it. The data can't tell us which happened.")
    cannot = [
        "Why the difference exists. This shows what changed, not the reason.",
        "Whether it will last. The result covers only the period in this file.",
        "Anything about people not in this file.",
    ]

    audit = audit_extra + [
        AuditItem(label="Difference", value=d_signed(diff), method="Treatment minus control",
                  code="t.mean() - c.mean()"),
        *([AuditItem(label="Relative change", value=f"{'+' if rel >= 0 else '−'}{abs(rel) * 100:.1f}%",
                     method="Difference divided by the control value", code="(t.mean() - c.mean()) / c.mean()")]
          if rel == rel else []),
        AuditItem(label="95% range for the difference", value=f"{d_signed(lo)} to {d_signed(hi)}",
                  method=("Newcombe interval" if method_kind == "binary" else "Welch t interval"),
                  code=("confint_proportions_2indep(t.sum(), len(t), c.sum(), len(c), method='newcomb')"
                        if method_kind == "binary" else "stats.ttest_ind(t, c, equal_var=False).confidence_interval(0.95)")),
        AuditItem(label="Chance of a gap this big if nothing changed (p-value)", value=f"{p_value:.4g}",
                  method=("Two-proportion z-test" if method_kind == "binary" else "Welch t-test"),
                  code=("test_proportions_2indep(t.sum(), len(t), c.sum(), len(c)).pvalue"
                        if method_kind == "binary" else "stats.ttest_ind(t, c, equal_var=False).pvalue")),
        AuditItem(label="Smallest difference this test can reliably detect", value=d(mde),
                  method="2.8 × standard error (95% confidence, 80% power)", code="2.8016 * se_diff"),
    ]

    return Report(
        headline=headline, answer=answer, how_sure=how_sure, verdict=verdict, causal_label=causal,
        checks=checks, arms=arms,
        effect={"diff": diff, "ci_low": lo, "ci_high": hi, "p_value": p_value, "rel": None if rel != rel else rel,
                "mde": mde, "unit_pts": unit_pts},
        outcome_kind=kind, outcome_label=label, cannot_tell=cannot,
        method_summary=method_summary, audit=audit,
        script=_script(spec, method_kind, csv_name), spec=spec,
    )


def _blocked(headline: str, reasons: list[str], spec, checks: list[Check] | None = None) -> Report:
    return Report(
        headline=headline, answer=" ".join(reasons), how_sure="No analysis was run.",
        verdict="blocked", causal_label="", checks=checks or [
            Check(name="Question fits data", status="fail", message=r) for r in reasons],
        cannot_tell=[], spec=spec,
    )
