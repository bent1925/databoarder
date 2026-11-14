"""The before/after analysis: did a number change after a date?

How it works, in one picture: cut the calendar into whole weeks, counting
outward from the change date (weeks before it, weeks after it). Average each
week. Then compare the "before" weeks with the "after" weeks.

Why whole weeks: every week contains each weekday exactly once, so weekend dips
cancel out, and the week-to-week numbers are close to independent even when
neighbouring days move together. (We simulated both; day-by-day models with
robust errors under-cover when days are strongly linked, week blocks do not.)

The date comes from the person. We never search for the date that looks best,
because that would manufacture changes out of noise.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .checks import detect_outcome_kind, is_idlike
from .interpret import chance_phrase, fmt_num, fmt_pct, humanize
from .knowledge import attach_method_info_ba, item
from .modes import ALPHA, MAX_LEVELS, SEED, effect
from .schemas import ArmSummary, BeforeAfterSpec, Check, Report, Robustness, Segment
from .script import build_ba_script

Z_POWER = 1.96 + 0.8416
MIN_WEEKS_FAIL = 3     # complete weeks needed on EACH side
MIN_WEEKS_WARN = 6
WINDOW_WEEKS = 4       # "narrow window" cross-check
MIN_SEG_WEEKS = 4      # weeks needed on each side inside a slice
HAC_LAGS = 7           # day-by-day cross-check


# ------------------------------------------------------------------ reading the data

def parse_dates(s: pd.Series) -> pd.Series:
    """Calendar day of each row (UTC, time of day dropped). Unreadable -> NaT."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pd.to_datetime(s, errors="coerce", utc=True)
    return d.dt.tz_localize(None).dt.normalize()


def validate_spec_ba(df: pd.DataFrame, spec: BeforeAfterSpec) -> list[str]:
    problems = []
    for col in (spec.date_col, spec.outcome_col):
        if col not in df.columns:
            problems.append(f"Column '{col}' isn't in your file.")
    if problems:
        return problems
    if spec.date_col == spec.outcome_col:
        return ["The result can't be the date column."]
    if not pd.api.types.is_numeric_dtype(df[spec.outcome_col]):
        return [f"'{spec.outcome_col}' isn't a number."]
    if pd.api.types.is_numeric_dtype(df[spec.date_col]):
        return [f"'{spec.date_col}' isn't a date."]
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
    return []


def daily_series(df: pd.DataFrame, spec: BeforeAfterSpec) -> pd.Series:
    """One number per calendar day: that day's average (or total) of the result."""
    frame = pd.DataFrame({"d": parse_dates(df[spec.date_col]),
                          "y": pd.to_numeric(df[spec.outcome_col], errors="coerce")}).dropna(subset=["d"])
    g = frame.groupby("d")["y"]
    s = g.mean() if spec.aggregate == "average" else g.sum(min_count=1)
    return s.sort_index()


@dataclass
class Blocks:
    before: np.ndarray        # weekly averages, oldest first
    after: np.ndarray         # weekly averages, oldest first
    before_starts: list
    after_starts: list
    skipped_before: int       # weeks dropped because a day was missing
    skipped_after: int
    cal: pd.Series            # full calendar of daily values (NaN = missing day)


def make_blocks(daily: pd.Series, change: pd.Timestamp) -> Blocks:
    cal = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    first, last = cal.index[0], cal.index[-1]
    one = pd.Timedelta(days=1)
    after, a_starts, skip_a = [], [], 0
    k = 0
    while change + pd.Timedelta(days=7 * k + 6) <= last:
        start = change + pd.Timedelta(days=7 * k)
        v = cal.loc[start: start + 6 * one]
        if v.notna().all():
            after.append(float(v.mean())); a_starts.append(start)
        else:
            skip_a += 1
        k += 1
    before, b_starts, skip_b = [], [], 0
    k = 1
    while change - pd.Timedelta(days=7 * k) >= first:
        start = change - pd.Timedelta(days=7 * k)
        v = cal.loc[start: start + 6 * one]
        if v.notna().all():
            before.append(float(v.mean())); b_starts.append(start)
        else:
            skip_b += 1
        k += 1
    return Blocks(np.array(before[::-1], float), np.array(after, float), b_starts[::-1], a_starts,
                  skip_b, skip_a, cal)


# --------------------------------------------------------------- the other estimators

def trend_adjusted(before: np.ndarray, after: np.ndarray) -> dict:
    """Level change at the date after allowing the metric to drift at a steady pace."""
    nb, na = len(before), len(after)
    y = np.concatenate([before, after])
    post = np.concatenate([np.zeros(nb), np.ones(na)])
    time = np.concatenate([np.arange(-nb, 0), np.arange(0, na)]).astype(float)
    res = sm.OLS(y, np.column_stack([np.ones(len(y)), post, time])).fit()
    lo, hi = res.conf_int()[1]
    return {"diff": float(res.params[1]), "lo": float(lo), "hi": float(hi), "p": float(res.pvalues[1])}


def daily_model(cal: pd.Series, change: pd.Timestamp) -> dict | None:
    """Every day, weekday indicators, Newey-West errors. None if weekday pattern can't be fitted."""
    obs = cal.dropna()
    if len(obs) < 28:
        return None
    wd = pd.get_dummies(obs.index.day_of_week, drop_first=True).to_numpy(float)
    post = (obs.index >= change).astype(float)
    X = np.column_stack([np.ones(len(obs)), post, wd])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    res = sm.OLS(obs.to_numpy(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
                                             use_t=True)
    lo, hi = res.conf_int()[1]
    return {"diff": float(res.params[1]), "lo": float(lo), "hi": float(hi), "p": float(res.pvalues[1])}


def pre_trend(before: np.ndarray) -> dict | None:
    """Was the metric already drifting before the date? Slope per week and its p-value."""
    n = len(before)
    if n < 4:
        return None
    res = sm.OLS(before, np.column_stack([np.ones(n), np.arange(n, dtype=float)])).fit()
    return {"slope": float(res.params[1]), "p": float(res.pvalues[1])}


def week_link(before: np.ndarray, after: np.ndarray) -> float | None:
    """Lag-1 correlation of weekly averages (each side centred on its own mean)."""
    xs, ys = [], []
    for side in (before, after):
        if len(side) >= 2:
            z = side - side.mean()
            xs.extend(z[:-1]); ys.extend(z[1:])
    if len(xs) < 8:
        return None
    xs, ys = np.array(xs), np.array(ys)
    den = np.sqrt((xs ** 2).sum() * (ys ** 2).sum())
    return float((xs * ys).sum() / den) if den > 0 else 0.0


# ------------------------------------------------------------------------- checks

def run_checks_ba(df, spec, daily: pd.Series, b: Blocks, kind: str, unit_pts: bool, gap: bool = False) -> list[Check]:
    checks: list[Check] = []
    nb, na = len(b.before), len(b.after)
    smallest = min(nb, na)
    if smallest < MIN_WEEKS_FAIL:
        checks.append(Check(name="Weeks of data", status="fail",
                            message=f"{nb} full weeks before, {na} after. Need {MIN_WEEKS_FAIL}+ on each side."))
    elif smallest < MIN_WEEKS_WARN:
        checks.append(Check(name="Weeks of data", status="warn",
                            message=f"{nb} weeks before, {na} after. Only big changes will show."))
    else:
        checks.append(Check(name="Weeks of data", status="pass", message=f"{nb} weeks before, {na} after."))

    # Unreadable dates / missing results (row level)
    d = parse_dates(df[spec.date_col])
    bad_dates = int(d.isna().sum() - df[spec.date_col].isna().sum())
    if bad_dates:
        checks.append(Check(name="Dates", status="warn", message=f"{bad_dates:,} rows have unreadable dates. Left out."))
    miss_rows = int(df[spec.outcome_col].isna().sum())
    share = miss_rows / max(len(df), 1)
    if share > 0.10:
        checks.append(Check(name="Missing results", status="warn",
                            message=f"{share:.0%} missing. Left out; may skew the answer."))
    elif miss_rows:
        checks.append(Check(name="Missing results", status="pass",
                            message=f"{miss_rows:,} missing ({share:.1%}). Left out."))
    else:
        checks.append(Check(name="Missing results", status="pass", message="None missing."))

    # Missing days
    cal = b.cal
    n_missing = int(cal.isna().sum())
    skipped = b.skipped_before + b.skipped_after
    total_weeks = nb + na + skipped
    if n_missing:
        bad = n_missing / len(cal) > 0.10 or (total_weeks and skipped / total_weeks > 0.20)
        checks.append(Check(name="Missing days", status="warn" if bad else "pass",
                            message=f"{n_missing} days have no data; {skipped} partial week{'s' if skipped != 1 else ''} left out."))
    else:
        checks.append(Check(name="Missing days", status="pass", message="No gaps."))

    # Was it already moving?
    pt = pre_trend(b.before)
    if pt is None:
        checks.append(Check(name="Steady before", status="info",
                            message="Too few weeks before to tell if the groups moved in parallel." if gap
                            else "Too few weeks before to tell if it was already drifting."))
    elif pt["p"] < ALPHA:
        mv = pt["slope"]
        amt = f"{abs(mv) * 100:.1f} pts" if unit_pts else fmt_num(abs(mv))
        if gap:
            msg = (f"The gap was already {'widening' if mv > 0 else 'narrowing'} before: about {amt} a week. "
                   f"The groups weren't moving in parallel.")
        else:
            msg = (f"Already {'rising' if mv > 0 else 'falling'} before: about {amt} a week. "
                   f"A before/after gap may just be that drift.")
        checks.append(Check(name="Steady before", status="warn", message=msg))
    else:
        checks.append(Check(name="Steady before", status="pass",
                            message="Groups moved in parallel before the date." if gap else "Flat before the date."))

    # Do neighbouring weeks move together?
    r = week_link(b.before, b.after)
    if r is None:
        checks.append(Check(name="Weeks related", status="info", message="Too few weeks to tell."))
    elif r > 0.5:
        checks.append(Check(name="Weeks related", status="warn",
                            message=f"Weeks move together (r = {r:.2f}). Ranges may be too narrow."))
    else:
        checks.append(Check(name="Weeks related", status="pass", message=f"Weeks look separate (r = {r:.2f})."))

    # Huge days
    if kind != "rate" and len(daily) > 20:
        sk = float(stats.skew(daily.dropna()))
        if abs(sk) > 3:
            checks.append(Check(name="Extreme values", status="warn",
                                message=f"A few huge days pull the average (skew {sk:.1f})."))
        else:
            checks.append(Check(name="Extreme values", status="pass", message="No huge days dominate."))
    return checks


# ------------------------------------------------------------------------- report

def _fmt(unit_pts: bool):
    def d(x):   # unsigned amount
        return f"{abs(x) * 100:.1f} pts" if unit_pts else fmt_num(abs(x))

    def ds(x):  # signed amount
        return ("+" if x >= 0 else "−") + d(x)

    def val(x):
        return fmt_pct(x) if unit_pts else fmt_num(x)
    return d, ds, val


def _pretty(ts: pd.Timestamp) -> str:
    return f"{ts.strftime('%b')} {ts.day}, {ts.year}"


def clean_list(v) -> list:
    return [None if not np.isfinite(x) else float(x) for x in v]


def smooth_sides(cal: pd.Series, change: pd.Timestamp) -> pd.Series:
    """7-day centred average, computed separately on each side so it never crosses the date."""
    smooth = pd.Series(index=cal.index, dtype=float)
    for part in (cal[cal.index < change], cal[cal.index >= change]):
        if len(part):
            smooth.loc[part.index] = part.rolling(7, center=True, min_periods=4).mean()
    return smooth


def repro_block(n_rows: int, nb: int, na: int, data_sha: str, csv_name: str) -> tuple[dict, str]:
    versions = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                "scipy": scipy.__version__, "statsmodels": statsmodels.__version__}
    repro = {"data_sha256": data_sha, "rows_used": int(n_rows), "weeks_used": f"{nb} before, {na} after",
             "seed": SEED, "versions": versions}
    header = (f"# Dashboarder reproduction script\n"
              f"# Data fingerprint (sha256): {data_sha or 'n/a'}\n"
              f"# Built with: " + ", ".join(f"{k} {v}" for k, v in versions.items()) + "\n\n")
    return repro, header


def _series(b: Blocks, change: pd.Timestamp, a_mean: float, b_mean: float, unit_pts: bool) -> dict:
    cal = b.cal
    smooth = smooth_sides(cal, change)
    clean = clean_list
    return {
        "dates": [d.date().isoformat() for d in cal.index],
        "values": clean(cal.to_numpy(float)),
        "smooth": clean(smooth.to_numpy(float)),
        "change_date": change.date().isoformat(),
        "before_mean": b_mean, "after_mean": a_mean,
        "before_from": b.before_starts[0].date().isoformat() if b.before_starts else None,
        "after_to": (b.after_starts[-1] + pd.Timedelta(days=6)).date().isoformat() if b.after_starts else None,
        "unit_pts": unit_pts,
    }


def analyze(df: pd.DataFrame, spec: BeforeAfterSpec, csv_name: str = "your_file.csv",
            mode: str = "quick", data_sha: str = "") -> Report:
    problems = validate_spec_ba(df, spec)
    if problems:
        return _blocked("Can't answer yet.", problems, spec, mode)

    change = pd.Timestamp(spec.change_date)
    daily = daily_series(df, spec)
    b = make_blocks(daily, change)

    valid_rows = df.loc[parse_dates(df[spec.date_col]).notna() & df[spec.outcome_col].notna()]
    per_day = len(valid_rows) / max(daily.notna().sum(), 1)
    if spec.aggregate == "total":
        kind = "total"
    else:
        kind = "rate" if detect_outcome_kind(df[spec.outcome_col]) == "binary" else "average"
    unit_pts = kind == "rate"

    checks = run_checks_ba(df, spec, daily, b, kind, unit_pts)
    if any(c.status == "fail" for c in checks):
        return _blocked("Can't answer yet.", [c.message for c in checks if c.status == "fail"], spec, mode, checks)

    name = humanize(spec.outcome_col)
    label = {"rate": f"{name} rate", "total": f"daily total {name}",
             "average": f"daily {name}" if per_day <= 1.05 else f"average {name}"}[kind]

    rep = _assemble_ba(spec, change, b, kind, label, unit_pts, checks)
    rep = _apply_mode_ba(rep, df, spec, change, b, kind, unit_pts, mode)
    rep = attach_method_info_ba(rep, mode, checks, len(b.before), len(b.after))

    rep.repro, header = repro_block(len(valid_rows), len(b.before), len(b.after), data_sha, csv_name)
    rep.script = build_ba_script(spec, csv_name, mode, header)
    return rep


def _assemble_ba(spec, change, b: Blocks, kind, label, unit_pts, checks) -> Report:
    bef, aft = b.before, b.after
    e = effect(bef, aft, "continuous")          # Welch on weekly averages; diff = after - before
    diff, lo, hi, p_value = e["diff"], e["lo"], e["hi"], e["p"]
    clear = (lo > 0 or hi < 0) and p_value < ALPHA
    d, ds, val = _fmt(unit_pts)

    mb, ma = float(bef.mean()), float(aft.mean())
    rel = diff / mb if mb != 0 else float("nan")
    se = float(np.sqrt(bef.var(ddof=1) / len(bef) + aft.var(ddof=1) / len(aft)))
    mde = Z_POWER * se

    arms = []
    for nm, s in (("Before", bef), ("After", aft)):
        m, sem = float(s.mean()), float(stats.sem(s))
        h = sem * stats.t.ppf(0.975, len(s) - 1)
        arms.append(ArmSummary(name=nm, n=len(s), estimate=m, ci_low=m - h, ci_high=m + h))

    pretty = _pretty(change)
    named = f"Before {val(mb)} → After {val(ma)}"
    rel_txt = "" if rel != rel else f" ({'+' if rel >= 0 else '−'}{abs(rel) * 100:.1f}%)"
    warn_n = sum(1 for c in checks if c.status == "warn")
    weeks = f"Based on {len(bef)} weeks before and {len(aft)} after."

    if clear:
        verdict = "clear_difference"
        headline = f"The {label} {'rose' if diff > 0 else 'fell'} after {pretty}."
        answer = f"{named} · {ds(diff)}{rel_txt}"
        how_sure = (f"Likely range: {ds(lo)} to {ds(hi)}. If nothing changed, a gap this big "
                    f"would appear {chance_phrase(p_value)}. {weeks}")
    else:
        verdict = "no_clear_difference"
        headline = f"No clear change in {label} after {pretty}."
        answer = f"{named}. " + ("Leaning that way, but not conclusive." if p_value < 0.2 else "Too close to call.")
        how_sure = (f"Likely range: {ds(lo)} to {ds(hi)}. This check only spots changes of {d(mde)} or more. "
                    f"Not proof of no effect. {weeks}")
    if warn_n:
        how_sure += f" {warn_n} data warning{'s' if warn_n > 1 else ''}: see checks."

    causal = ("Shows the number moved after this date, not why. "
              "Anything else that changed around then looks the same.")
    cannot = ["What caused it.", "Whether it lasts.", "Anything outside these dates."]

    audit = [
        item("Before average", val(mb), "ba_level", "Mean of weekly averages, before", f"before.mean()  # {len(bef)} weeks"),
        item("After average", val(ma), "ba_level", "Mean of weekly averages, after", f"after.mean()  # {len(aft)} weeks"),
        *[item(f"{a.name} range", f"{val(a.ci_low)} to {val(a.ci_high)}", "ba_arm_ci", "95% interval (t)",
               "mean ± t(0.975, n-1) × sem") for a in arms],
        item("Change", ds(diff), "ba_diff", "After minus before", "after.mean() - before.mean()"),
        *([item("Relative change", f"{'+' if rel >= 0 else '−'}{abs(rel) * 100:.1f}%", "rel", "Change ÷ before",
                "(after.mean() - before.mean()) / before.mean()")] if rel == rel else []),
        item("95% range for the change", f"{ds(lo)} to {ds(hi)}", "ba_ci", "Welch t interval on weekly averages",
             "stats.ttest_ind(after, before, equal_var=False).confidence_interval(0.95)"),
        item("p-value", f"{p_value:.4g}", "ba_p", "Welch t-test on weekly averages",
             "stats.ttest_ind(after, before, equal_var=False).pvalue"),
        item("Smallest change this check can spot", d(mde), "ba_mde", "2.80 × standard error", "2.8016 * se_diff"),
        item("'Clear' rule", "Met" if clear else "Not met", "rule", "Range excludes 0 and p < 0.05",
             "(low > 0 or high < 0) and p < 0.05"),
        item("Weeks used", f"{len(bef)} before, {len(aft)} after", "ba_weeks", "Whole 7-day weeks counted from the date",
             "weeks start at change_date ± 7k days; weeks with a missing day are skipped"),
    ]
    return Report(
        analysis="before_after", headline=headline, answer=answer, how_sure=how_sure, verdict=verdict,
        causal_label=causal, design="one_date", checks=checks, arms=arms,
        effect={"diff": diff, "ci_low": lo, "ci_high": hi, "p_value": p_value, "rel": None if rel != rel else rel,
                "mde": mde, "unit_pts": unit_pts},
        outcome_kind=kind, outcome_label=label, cannot_tell=cannot, audit=audit, spec=spec,
        series=_series(b, change, ma, mb, unit_pts),
    )


def _blocked(headline, reasons, spec, mode, checks=None) -> Report:
    return Report(analysis="before_after", headline=headline, answer=" ".join(reasons), how_sure="",
                  verdict="blocked", causal_label="", mode=mode,
                  checks=checks or [Check(name="Question fits data", status="fail", message=r) for r in reasons],
                  cannot_tell=[], spec=spec)


# ------------------------------------------------------------------------- modes

def _agrees(p, diff, main_clear, main_diff) -> bool:
    return (p < ALPHA) == main_clear and (not main_clear or np.sign(diff) == np.sign(main_diff))


def _cross_text(e, main_clear, main_diff, ds) -> tuple[str, bool, bool]:
    """Compare a cross-check with the main result -> (text, agrees, weak).

    weak = same direction, main answer sits inside this check's range, but this check can't
    rule out zero (it has less data to work with). Not a contradiction, not a confirmation.
    """
    agrees = _agrees(e["p"], e["diff"], main_clear, main_diff)
    weak = (not agrees and main_clear and e["p"] >= ALPHA and np.sign(e["diff"]) == np.sign(main_diff)
            and e["lo"] <= main_diff <= e["hi"])
    word = "Same conclusion" if agrees else "Can't confirm: same direction, too uncertain" if weak \
        else "Different conclusion"
    return f"{word}. {ds(e['diff'])}; if nothing changed: {chance_phrase(e['p'])}.", agrees or weak, weak


def robustness_ba(b: Blocks, change, main_clear, main_diff, unit_pts,
                  noun: str = "number") -> tuple[list[Robustness], list[str]]:
    _, ds, _ = _fmt(unit_pts)
    out: list[Robustness] = []
    skipped: list[str] = []

    pre = b.before
    if len(pre) >= 2 * MIN_WEEKS_FAIL:
        h = len(pre) // 2
        e = effect(pre[:h], pre[h:], "continuous")
        found = e["p"] < ALPHA
        text = ("Found a 'change' at a fake date. Treat the main result with care."
                if found else "Found nothing at a fake date. Good.")
        out.append(Robustness(key="placebo", name="Fake-date test",
                              description="Pretends the change happened earlier, using only weeks before the real date.",
                              result=f"{text} If nothing changed: {chance_phrase(e['p'])}.",
                              agrees=not found, p_value=e["p"]))
    else:
        skipped.append(f"Fake-date test skipped: needs {2 * MIN_WEEKS_FAIL}+ weeks before the date.")

    e = trend_adjusted(b.before, b.after)
    text, agrees, weak = _cross_text(e, main_clear, main_diff, ds)
    out.append(Robustness(key="its", name="Allowing for a trend",
                          description=f"Lets the {noun} have been drifting already, then asks if there is still a jump.",
                          result=text, agrees=agrees, weak=weak, p_value=e["p"]))

    if len(b.before) > WINDOW_WEEKS and len(b.after) > WINDOW_WEEKS:
        e = effect(b.before[-WINDOW_WEEKS:], b.after[:WINDOW_WEEKS], "continuous")
        text, agrees, weak = _cross_text(e, main_clear, main_diff, ds)
        out.append(Robustness(key="window", name="Nearby weeks only",
                              description=f"Uses only the {WINDOW_WEEKS} weeks on each side of the date.",
                              result=text, agrees=agrees, weak=weak, p_value=e["p"]))
    else:
        skipped.append(f"Nearby-weeks check skipped: needs more than {WINDOW_WEEKS} weeks on each side.")

    e = daily_model(b.cal, change)
    if e is not None:
        text, agrees, weak = _cross_text(e, main_clear, main_diff, ds)
        out.append(Robustness(key="daily", name="Day-by-day model",
                              description="Uses every day, adjusting for weekday patterns.",
                              result=text, agrees=agrees, weak=weak, p_value=e["p"]))
    else:
        skipped.append("Day-by-day model skipped: weekday pattern couldn't be fitted.")
    return out, skipped


def segment_scan_ba(df: pd.DataFrame, spec, change, daily_fn=None, skip=()) -> tuple[list[Segment], str]:
    daily_fn = daily_fn or (lambda part: daily_series(part, spec))
    rows = []
    for col in df.columns:
        if col in (spec.date_col, spec.outcome_col, *skip) or is_idlike(df, col):
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or not (2 <= s.nunique(dropna=True) <= MAX_LEVELS):
            continue
        s_str = s.astype(str)
        for level in sorted(s.dropna().astype(str).unique()):
            part = df[s_str == level]
            if part[spec.outcome_col].notna().sum() == 0:
                continue
            daily = daily_fn(part) if len(part) else None
            if daily is None or daily.notna().sum() == 0:
                continue
            bl = make_blocks(daily, change)
            if bl.cal.index[0] >= change or bl.cal.index[-1] < change:
                continue
            if len(bl.before) < MIN_SEG_WEEKS or len(bl.after) < MIN_SEG_WEEKS:
                continue
            e = effect(bl.before, bl.after, "continuous")
            rows.append({"column": col, "level": level, "n_control": len(bl.before), "n_treatment": len(bl.after),
                         "diff": e["diff"], "ci_low": e["lo"], "ci_high": e["hi"], "p_raw": e["p"]})
    if not rows:
        return [], "No slice had enough weeks to compare."
    p_adj = multipletests([r["p_raw"] for r in rows], method="holm")[1]
    segs = [Segment(**r, p_adj=float(pa), flagged=bool(pa < ALPHA)) for r, pa in zip(rows, p_adj)]
    segs.sort(key=lambda s: s.p_raw)
    m, flagged = len(segs), sum(s.flagged for s in segs)
    note = f"{m} slices checked, adjusted for that. "
    note += (f"{flagged} stand{'s' if flagged == 1 else ''} out: ideas to test, not conclusions."
             if flagged else "None stand out.")
    return segs, note


def quick_caveats_ba(rep: Report) -> list[str]:
    cav = [c.message for c in rep.checks if c.status == "warn"]
    cav.append("Quick look: not cross-checked.")
    cav.append("Shows what moved, not why. Anything else that changed that day looks the same.")
    if rep.verdict == "no_clear_difference":
        cav.append("'No clear change' isn't 'no effect'. Too few weeks may hide it.")
    cav.append("Big decision? Rerun as 'Need the right answer'.")
    return cav


def _apply_mode_ba(rep: Report, df, spec, change, b: Blocks, kind, unit_pts, mode) -> Report:
    rep.mode = mode
    main_clear = rep.verdict == "clear_difference"
    if mode == "quick":
        rep.caveats = quick_caveats_ba(rep)
    elif mode == "right_answer":
        rep.robustness, skipped = robustness_ba(b, change, main_clear, rep.effect["diff"], unit_pts)
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
        rep.segments, rep.segment_note = segment_scan_ba(df, spec, change)
    return rep
