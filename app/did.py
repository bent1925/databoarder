"""Before/after with a comparison group (difference-in-differences).

Same whole-week method as before/after, applied to the GAP between two groups:
for each week, treated average minus comparison average. If the gap moved after
the date, the change did something that the comparison group didn't feel.
Anything that hit both groups (holidays, season, news) cancels out of the gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .before_after import (Blocks, _agrees, _fmt, _pretty, clean_list, daily_series, make_blocks, parse_dates,
                           repro_block, robustness_ba, run_checks_ba, segment_scan_ba, smooth_sides)
from .checks import detect_outcome_kind
from .interpret import chance_phrase, humanize
from .knowledge import attach_method_info_ba, item
from .modes import ALPHA, effect
from .schemas import ArmSummary, Check, DiDSpec, Report
from .script import build_ba_script

Z_POWER = 1.96 + 0.8416


def validate_spec_did(df: pd.DataFrame, spec: DiDSpec) -> list[str]:
    for col in (spec.date_col, spec.outcome_col, spec.group_col):
        if col not in df.columns:
            return [f"Column '{col}' isn't in your file."]
    if len({spec.date_col, spec.outcome_col, spec.group_col}) < 3:
        return ["Date, group and result must be three different columns."]
    if not pd.api.types.is_numeric_dtype(df[spec.outcome_col]):
        return [f"'{spec.outcome_col}' isn't a number."]
    if pd.api.types.is_numeric_dtype(df[spec.date_col]):
        return [f"'{spec.date_col}' isn't a date."]
    levels = set(df[spec.group_col].dropna().astype(str).unique())
    probs = [f"'{v}' isn't a value of '{spec.group_col}'." for v in (spec.treated, spec.comparison) if v not in levels]
    if probs:
        return probs
    if spec.treated == spec.comparison:
        return ["Pick two different groups."]
    d = parse_dates(df[spec.date_col])
    nonblank = int(df[spec.date_col].notna().sum())
    if nonblank == 0 or d.notna().sum() < 0.9 * nonblank:
        return [f"Couldn't read '{spec.date_col}' as dates."]
    if not spec.change_date:
        return ["Pick the date the change happened."]
    ts = pd.Timestamp(spec.change_date)
    lo, hi = d.min(), d.max()
    if not (lo < ts <= hi):
        return [f"{spec.change_date} is outside your data ({lo.date()} to {hi.date()})."]
    g = df[spec.group_col].astype(str)
    for level in (spec.treated, spec.comparison):
        dl = d[g == level].dropna()
        if not ((dl < ts).any() and (dl >= ts).any()):
            return [f"'{level}' needs rows both before and after {spec.change_date}."]
    return []


def group_daily(df: pd.DataFrame, spec: DiDSpec, level: str) -> pd.Series:
    return daily_series(df[df[spec.group_col].astype(str) == level], spec)


def gap_series(df: pd.DataFrame, spec: DiDSpec) -> pd.Series:
    """Treated minus comparison, per calendar day. NaN if either group has no value that day."""
    return (group_daily(df, spec, spec.treated) - group_daily(df, spec, spec.comparison)).sort_index()


def _group_mean(cal: pd.Series, starts: list) -> float:
    vals = [float(cal.loc[s: s + pd.Timedelta(days=6)].mean()) for s in starts]
    return float(np.mean(vals)) if vals else float("nan")


def analyze(df: pd.DataFrame, spec: DiDSpec, csv_name: str = "your_file.csv", mode: str = "quick",
            data_sha: str = "") -> Report:
    problems = validate_spec_did(df, spec)
    if problems:
        return _blocked("Can't answer yet.", problems, spec, mode)

    change = pd.Timestamp(spec.change_date)
    sub = df[df[spec.group_col].astype(str).isin([spec.treated, spec.comparison])]
    d_t, d_c = group_daily(sub, spec, spec.treated), group_daily(sub, spec, spec.comparison)
    gap = (d_t - d_c).sort_index()
    b = make_blocks(gap, change)

    valid = sub.loc[parse_dates(sub[spec.date_col]).notna() & sub[spec.outcome_col].notna()]
    per_day = len(valid) / max(len(d_t.dropna()) + len(d_c.dropna()), 1)
    kind = "total" if spec.aggregate == "total" else (
        "rate" if detect_outcome_kind(df[spec.outcome_col]) == "binary" else "average")
    unit_pts = kind == "rate"

    checks = run_checks_ba(sub, spec, gap, b, kind, unit_pts, gap=True)
    checks.append(_comparison_check(d_c, change, spec, unit_pts))
    if any(c.status == "fail" for c in checks):
        return _blocked("Can't answer yet.", [c.message for c in checks if c.status == "fail"], spec, mode, checks)

    name = humanize(spec.outcome_col)
    label = {"rate": f"{name} rate", "total": f"daily total {name}",
             "average": f"daily {name}" if per_day <= 1.05 else f"average {name}"}[kind]

    rep = _assemble(spec, change, b, d_t, d_c, kind, label, unit_pts, checks)
    rep = _apply_mode(rep, sub, spec, change, b, unit_pts, mode)
    rep = attach_method_info_ba(rep, mode, checks, len(b.before), len(b.after), did=True)
    rep.repro, header = repro_block(len(valid), len(b.before), len(b.after), data_sha, csv_name)
    rep.script = build_ba_script(spec, csv_name, mode, header)
    return rep


def _comparison_check(d_c: pd.Series, change, spec, unit_pts) -> Check:
    """Did the comparison group move on its own? If so, other things changed. The gap still handles it."""
    bl = make_blocks(d_c, change)
    if len(bl.before) < 3 or len(bl.after) < 3:
        return Check(name="Comparison group", status="info", message="Too few full weeks to see if it moved.")
    e = effect(bl.before, bl.after, "continuous")
    _, ds, _ = _fmt(unit_pts)
    if e["p"] < ALPHA:
        return Check(name="Comparison group", status="info",
                     message=f"'{spec.comparison}' also moved ({ds(e['diff'])}). Other things changed; the gap view accounts for it.")
    return Check(name="Comparison group", status="pass", message=f"'{spec.comparison}' stayed steady.")


def _assemble(spec, change, b: Blocks, d_t, d_c, kind, label, unit_pts, checks) -> Report:
    bef, aft = b.before, b.after
    e = effect(bef, aft, "continuous")          # Welch on weekly gaps; diff = gap after - gap before
    diff, lo, hi, p_value = e["diff"], e["lo"], e["hi"], e["p"]
    clear = (lo > 0 or hi < 0) and p_value < ALPHA
    d, ds, val = _fmt(unit_pts)

    gb, ga = float(bef.mean()), float(aft.mean())
    se = float(np.sqrt(bef.var(ddof=1) / len(bef) + aft.var(ddof=1) / len(aft)))
    mde = Z_POWER * se

    arms = []
    for nm, s in (("Gap before", bef), ("Gap after", aft)):
        m, sem = float(s.mean()), float(stats.sem(s))
        h = sem * stats.t.ppf(0.975, len(s) - 1)
        arms.append(ArmSummary(name=nm, n=len(s), estimate=m, ci_low=m - h, ci_high=m + h))

    cal_t, cal_c = d_t.reindex(b.cal.index), d_c.reindex(b.cal.index)
    t_before, t_after = _group_mean(cal_t, b.before_starts), _group_mean(cal_t, b.after_starts)
    c_before, c_after = _group_mean(cal_c, b.before_starts), _group_mean(cal_c, b.after_starts)

    pretty = _pretty(change)
    named = f"{spec.treated} minus {spec.comparison}: {ds(gb)} before → {ds(ga)} after"
    warn_n = sum(1 for c in checks if c.status == "warn")
    weeks = f"Based on {len(bef)} weeks before and {len(aft)} after."

    if clear:
        verdict = "clear_difference"
        headline = f"Relative to {spec.comparison}, {spec.treated}'s {label} {'rose' if diff > 0 else 'fell'} after {pretty}."
        answer = f"{named} · {ds(diff)}"
        how_sure = (f"Likely range: {ds(lo)} to {ds(hi)}. If the gap hadn't changed, a shift this big "
                    f"would appear {chance_phrase(p_value)}. {weeks}")
    else:
        verdict = "no_clear_difference"
        headline = f"No clear change in {spec.treated}'s {label} relative to {spec.comparison} after {pretty}."
        answer = f"{named}. " + ("Leaning that way, but not conclusive." if p_value < 0.2 else "Too close to call.")
        how_sure = (f"Likely range: {ds(lo)} to {ds(hi)}. This check only spots shifts of {d(mde)} or more. "
                    f"Not proof of no effect. {weeks}")
    if warn_n:
        how_sure += f" {warn_n} data warning{'s' if warn_n > 1 else ''}: see checks."

    causal = ("Cause and effect if, without the change, both groups would have moved in parallel. "
              "Anything that hit only one group at that date looks the same.")
    cannot = ["Why the gap moved.", "Whether it lasts.", "Anything outside these dates."]

    audit = [
        item(f"{spec.treated} before", val(t_before), "ba_level", "Mean of weekly averages, before", "treated_before.mean()"),
        item(f"{spec.treated} after", val(t_after), "ba_level", "Mean of weekly averages, after", "treated_after.mean()"),
        item(f"{spec.comparison} before", val(c_before), "ba_level", "Mean of weekly averages, before", "comparison_before.mean()"),
        item(f"{spec.comparison} after", val(c_after), "ba_level", "Mean of weekly averages, after", "comparison_after.mean()"),
        item("Gap before", ds(gb), "did_gap", "Mean weekly (treated − comparison), before", "gap_before.mean()"),
        item("Gap after", ds(ga), "did_gap", "Mean weekly (treated − comparison), after", "gap_after.mean()"),
        item("Change in the gap", ds(diff), "did_diff", "Gap after minus gap before", "gap_after.mean() - gap_before.mean()"),
        item("95% range for the change", f"{ds(lo)} to {ds(hi)}", "did_ci", "Welch t interval on weekly gaps",
             "stats.ttest_ind(gap_after, gap_before, equal_var=False).confidence_interval(0.95)"),
        item("p-value", f"{p_value:.4g}", "did_p", "Welch t-test on weekly gaps",
             "stats.ttest_ind(gap_after, gap_before, equal_var=False).pvalue"),
        item("Smallest change this check can spot", d(mde), "ba_mde", "2.80 × standard error", "2.8016 * se_diff"),
        item("'Clear' rule", "Met" if clear else "Not met", "rule", "Range excludes 0 and p < 0.05",
             "(low > 0 or high < 0) and p < 0.05"),
        item("Weeks used", f"{len(bef)} before, {len(aft)} after", "ba_weeks", "Whole 7-day weeks counted from the date",
             "weeks where both groups have all 7 days"),
    ]

    smooth_t, smooth_c = smooth_sides(cal_t, change), smooth_sides(cal_c, change)
    series = {
        "dates": [x.date().isoformat() for x in b.cal.index], "change_date": change.date().isoformat(),
        "groups": [{"name": str(spec.treated), "values": clean_list(cal_t.to_numpy(float)),
                    "smooth": clean_list(smooth_t.to_numpy(float)), "before": t_before, "after": t_after},
                   {"name": str(spec.comparison), "values": clean_list(cal_c.to_numpy(float)),
                    "smooth": clean_list(smooth_c.to_numpy(float)), "before": c_before, "after": c_after}],
        "unit_pts": unit_pts,
    }
    return Report(
        analysis="diff_in_diff", design="one_date_comparison", headline=headline, answer=answer, how_sure=how_sure,
        verdict=verdict, causal_label=causal, checks=checks, arms=arms,
        effect={"diff": diff, "ci_low": lo, "ci_high": hi, "p_value": p_value, "rel": None, "mde": mde,
                "unit_pts": unit_pts},
        outcome_kind=kind, outcome_label=label, cannot_tell=cannot, audit=audit, spec=spec, series=series,
    )


def _blocked(headline, reasons, spec, mode, checks=None) -> Report:
    return Report(analysis="diff_in_diff", design="one_date_comparison", headline=headline,
                  answer=" ".join(reasons), how_sure="", verdict="blocked", causal_label="", mode=mode,
                  checks=checks or [Check(name="Question fits data", status="fail", message=r) for r in reasons],
                  cannot_tell=[], spec=spec)


def _apply_mode(rep: Report, sub, spec, change, b: Blocks, unit_pts, mode) -> Report:
    rep.mode = mode
    main_clear = rep.verdict == "clear_difference"
    if mode == "quick":
        cav = [c.message for c in rep.checks if c.status == "warn"]
        cav += ["Quick look: not cross-checked.",
                "Shows the gap moved, not why. Anything that hit only one group at that date looks the same."]
        if rep.verdict == "no_clear_difference":
            cav.append("'No clear change' isn't 'no effect'. Too few weeks may hide it.")
        cav.append("Big decision? Rerun as 'Need the right answer'.")
        rep.caveats = cav
    elif mode == "right_answer":
        rep.robustness, skipped = robustness_ba(b, change, main_clear, rep.effect["diff"], unit_pts, noun="gap")
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
        rep.segments, rep.segment_note = segment_scan_ba(
            sub, spec, change, daily_fn=lambda part: gap_series(part, spec), skip=(spec.group_col,))
    return rep
