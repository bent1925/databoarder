"""Single source of truth for what each statistic MEANS and what must be TRUE.

Every number in a report gets two descriptions:
  technical - the textbook-correct definition
  plain     - what it means in practice
and every method lists the assumptions its result depends on. Assumptions we can
test in the data carry a checked status; the rest are marked "you confirm".
"""
from __future__ import annotations

import numpy as np

from .schemas import Assumption, AuditItem, Check, MethodInfo, Report

STATS: dict[str, dict[str, str]] = {
    "rate": {
        "technical": "Sample proportion: successes ÷ rows in the group.",
        "plain": "Share of people who did it.",
    },
    "mean": {
        "technical": "Sample mean of the outcome in the group.",
        "plain": "The typical value per person.",
    },
    "arm_ci": {
        "technical": "95% interval for one group: Wilson score interval (rates) or t-interval for the mean (averages).",
        "plain": "Where this group's true value likely sits.",
    },
    "diff": {
        "technical": "Treatment estimate minus control estimate.",
        "plain": "How far the new version moved the number.",
    },
    "rel": {
        "technical": "Difference ÷ control estimate.",
        "plain": "The change as a share of the starting point. Shaky if the starting point is near 0.",
    },
    "ci_binary": {
        "technical": "95% Newcombe hybrid-score interval for a difference of two independent proportions, built from Wilson score intervals. Across repeated samples, about 95% of such intervals contain the true difference.",
        "plain": "Likely range for the true change. Wide = unsure. Includes 0 = could be no change.",
    },
    "ci_mean": {
        "technical": "95% interval for a difference in means from Welch's t-distribution (Welch–Satterthwaite degrees of freedom). Across repeated samples, about 95% of such intervals contain the true difference.",
        "plain": "Likely range for the true change. Wide = unsure. Includes 0 = could be no change.",
    },
    "p_binary": {
        "technical": "Two-sided p-value of the pooled two-proportion z-test (score test) of H0: both groups have the same true rate. P(a result at least this extreme | H0).",
        "plain": "If nothing changed, how often a gap this big would appear. Small = hard to blame on chance. Not the chance the result is wrong.",
    },
    "p_mean": {
        "technical": "Two-sided p-value of Welch's t-test of H0: both groups have the same true mean. P(a result at least this extreme | H0).",
        "plain": "If nothing changed, how often a gap this big would appear. Small = hard to blame on chance. Not the chance the result is wrong.",
    },
    "mde": {
        "technical": "Smallest true difference detected with 80% power at two-sided α = 0.05: (1.96 + 0.84) × standard error of the difference ≈ 2.80 × SE, using the observed SE.",
        "plain": "Real gaps smaller than this will probably be missed.",
    },
    "rule": {
        "technical": "Called 'clear' only if the 95% range excludes 0 and p < 0.05.",
        "plain": "'Clear' needs both signals to agree.",
    },
    "fisher": {
        "technical": "Fisher's exact test, two-sided p-value under the hypergeometric distribution with margins fixed. H0: equal true rates.",
        "plain": "A second opinion that doesn't lean on large-sample math.",
    },
    "mannwhitney": {
        "technical": "Mann–Whitney U (rank-sum) test, two-sided. H0: a random value from one group is equally likely to be above or below one from the other.",
        "plain": "Does one group tend to score higher? Extreme values barely matter.",
    },
    "shuffle": {
        "technical": "Monte Carlo permutation test: (hits + 1) ÷ 2,001, where hits = random relabelings (seed 0) with |mean difference| ≥ observed.",
        "plain": "Reshuffle who got which version. How often does chance alone match this gap?",
    },
    "trim": {
        "technical": "Welch's t-test after dropping pooled values below the 1st or above the 99th percentile.",
        "plain": "The same question without the extreme values.",
    },
    "ba_level": {
        "technical": "Mean of weekly averages. Each week is 7 consecutive days counted outward from the change date; each day's value is that day's average (or total) of the result. Every weekday appears once per week, so weekday patterns cancel.",
        "plain": "The typical day in this period, with weekend dips evened out.",
    },
    "ba_arm_ci": {
        "technical": "95% t-interval for the mean of the weekly averages in that period (n − 1 degrees of freedom, n = number of weeks).",
        "plain": "Where the typical day in this period likely sits.",
    },
    "ba_diff": {
        "technical": "Mean of after-weeks minus mean of before-weeks (weekly averages).",
        "plain": "How far the typical day moved after the date.",
    },
    "ba_ci": {
        "technical": "95% interval for the difference of two means from Welch's t-distribution, computed on weekly averages (Welch–Satterthwaite degrees of freedom). Across repeated samples, about 95% of such intervals contain the true change.",
        "plain": "Likely range for the true change. Wide = unsure. Includes 0 = could be no change.",
    },
    "ba_p": {
        "technical": "Two-sided p-value of Welch's t-test on weekly averages of H0: the mean level is the same before and after. P(a result at least this extreme | H0).",
        "plain": "If nothing changed, how often a gap this big would appear. Small = hard to blame on chance. Not the chance the result is wrong.",
    },
    "ba_mde": {
        "technical": "Smallest true change detected with 80% power at two-sided α = 0.05: (1.96 + 0.84) × standard error of the difference ≈ 2.80 × SE, using the observed SE.",
        "plain": "Real changes smaller than this will probably be missed.",
    },
    "ba_weeks": {
        "technical": "Whole 7-day blocks starting at change_date + 7k days (after) or change_date − 7k days (before). A block with any missing day is skipped; days outside whole blocks are unused here.",
        "plain": "How many full weeks the comparison rests on. More weeks = steadier answer.",
    },
    "obs_cols": {
        "technical": "Columns used to make people comparable. Numeric columns are standardised (mean 0, SD 1); category columns become 0/1 indicators with the alphabetically first level as the reference.",
        "plain": "What we matched people up on.",
    },
    "obs_level": {
        "technical": "Predicted average outcome for the group from the regression, with the other columns held at their average over everyone (linear-model g-computation).",
        "plain": "What each group's average would be if both had the same kind of people.",
    },
    "obs_diff": {
        "technical": "Coefficient on the treatment indicator in an OLS regression of the outcome on treatment and the chosen columns. For yes/no outcomes this is a linear-probability model (a difference in probability).",
        "plain": "The gap left after comparing like with like. Not cause and effect.",
    },
    "obs_ci": {
        "technical": "95% interval for the treatment coefficient using HC3 heteroskedasticity-robust standard errors and the t-distribution.",
        "plain": "Likely range for the gap. Wide = unsure. Includes 0 = could be no gap.",
    },
    "obs_p": {
        "technical": "Two-sided p-value of the t-test (HC3 robust) of H0: the treatment coefficient is 0. P(a result at least this extreme | H0).",
        "plain": "If there were no gap among like-for-like people, how often one this big would appear. Small = hard to blame on chance. Not the chance the result is wrong.",
    },
    "evalue": {
        "technical": "E-value (VanderWeele & Ding): the minimum risk-ratio-scale association an unmeasured factor would need with both group membership and the outcome, beyond the compared columns, to fully explain away the estimate. For averages, RR ≈ exp(0.91 × d) with d = difference ÷ SD of the outcome.",
        "plain": "How strong a hidden factor would have to be to erase this. Bigger = harder to explain away.",
    },
    "ipw": {
        "technical": "Weighted least squares of the outcome on treatment, with weights 1/p for treated and 1/(1−p) for control, where p is the logistic-regression chance of being treated given the compared columns (clipped to 5%–95%). HC1 robust errors; ignores uncertainty in p, which makes the range slightly wide.",
        "plain": "Balances the groups by weighting people instead of adjusting with a model. A second route to the same question.",
    },
    "overlap": {
        "technical": "The adjusted regression repeated only on people whose estimated chance of being treated is between 10% and 90%.",
        "plain": "Only compares people who could realistically have gone either way.",
    },
    "unadjusted": {
        "technical": "The raw difference with no adjustment (Newcombe interval and score z-test for rates; Welch t-test for averages).",
        "plain": "The gap without matching people up. A big difference from the adjusted gap means who chose it matters a lot.",
    },
    "did_gap": {
        "technical": "Mean over whole weeks of (treated group's daily value − comparison group's daily value). Each week is 7 consecutive days counted outward from the change date; weeks where either group is missing a day are skipped.",
        "plain": "How far apart the two groups are on a typical day.",
    },
    "did_diff": {
        "technical": "Difference-in-differences: mean weekly gap after − mean weekly gap before = (treated after − treated before) − (comparison after − comparison before).",
        "plain": "How much the gap between the groups moved after the date. Anything that hit both groups cancels out.",
    },
    "did_ci": {
        "technical": "95% interval for the change in the weekly gap from Welch's t-distribution (Welch–Satterthwaite degrees of freedom). Across repeated samples, about 95% of such intervals contain the true change.",
        "plain": "Likely range for the true change in the gap. Wide = unsure. Includes 0 = could be no change.",
    },
    "did_p": {
        "technical": "Two-sided p-value of Welch's t-test on weekly gaps of H0: the mean gap is the same before and after. P(a result at least this extreme | H0).",
        "plain": "If the gap hadn't changed, how often a shift this big would appear. Small = hard to blame on chance. Not the chance the result is wrong.",
    },
    "placebo": {
        "technical": "Welch's t-test on the before-period weeks only, split into an earlier and a later half at a pretend change date. H0: equal means. A valid method should rarely reject here.",
        "plain": "Pick a fake date when nothing happened. If we 'find' a change anyway, drift or noise is fooling us.",
    },
    "its": {
        "technical": "Ordinary least squares of weekly averages on an intercept, an after-the-date indicator and a linear week index (one slope across the date). The indicator's coefficient is the level shift at the date after allowing a steady trend. Conventional t-based interval, n − 3 degrees of freedom.",
        "plain": "Lets the number have been drifting already, then asks whether there is still a jump at the date.",
    },
    "window": {
        "technical": "Welch's t-test on weekly averages using only the 4 weeks nearest the date on each side.",
        "plain": "Looks only at weeks right around the date, so slow drift and far-away events matter less.",
    },
    "daily": {
        "technical": "OLS on every observed day with an after-the-date indicator and weekday indicators; Newey–West (Bartlett kernel, 7 lags, n/(n − k) correction) standard errors; t-based p-value with n − k degrees of freedom.",
        "plain": "Uses every single day, allowing for weekday patterns and for neighbouring days moving together.",
    },
    "holm": {
        "technical": "Holm–Bonferroni step-down adjusted p-values. Keeps the chance of at least one false positive across all slices (family-wise error rate) at 5%.",
        "plain": "Chance figures corrected for checking many slices at once.",
    },
}


def item(label: str, value: str, key: str, method: str, code: str) -> AuditItem:
    s = STATS[key]
    return AuditItem(label=label, value=value, method=method, code=code,
                     technical=s["technical"], plain=s["plain"])


# ---------------------------------------------------------------- assumptions

def _status(checks: list[Check], name: str, default: str = "manual") -> str:
    for c in checks:
        if c.name == name:
            return {"pass": "ok", "warn": "warn", "info": "manual", "fail": "warn"}[c.status]
    return default


def _independent(checks):
    return Assumption(text="Each row is a different person", status=_status(checks, "Repeat people"))


def _missing(checks):
    names = [c for c in checks if c.name in ("Missing results", "Missing balance")]
    if any(c.status == "warn" for c in names):
        st = "warn"
    elif any(c.name == "Missing balance" for c in names):
        st = "ok"
    else:
        st = "ok" if all("None" in c.message for c in names) else "manual"
    return Assumption(text="Missing results don't depend on the version", status=st)


MANUAL = {
    "random": Assumption(text="Versions were assigned at random", status="manual"),
    "nointerference": Assumption(text="One person's version doesn't affect another's", status="manual"),
    "samemeasure": Assumption(text="Both groups were measured the same way", status="manual"),
    "once": Assumption(text="One planned test, looked at once at the end", status="manual"),
}


def build_methods(kind: str, mode: str, checks: list[Check], c: np.ndarray, t: np.ndarray) -> list[MethodInfo]:
    ind, miss = _independent(checks), _missing(checks)
    methods: list[MethodInfo] = []

    alike = next((c.status for c in checks if c.name == "Groups alike"), None)
    random_a = Assumption(text="Versions were assigned at random",
                          status="warn" if alike == "warn" else "manual")
    MANUAL = {**globals()["MANUAL"], "random": random_a}
    if kind == "binary":
        enough = min(c.sum(), len(c) - c.sum(), t.sum(), len(t) - t.sum()) >= 10
        main = [ind,
                Assumption(text="At least 10 yes and 10 no in each group", status="ok" if enough else "warn"),
                miss, MANUAL["random"], MANUAL["nointerference"], MANUAL["samemeasure"], MANUAL["once"]]
        methods.append(MethodInfo(name="Main test (Newcombe range + z-test)", assumptions=main))
    else:
        main = [ind,
                Assumption(text="Enough rows that averages are steady (no huge values dominating)",
                           status=_status(checks, "Extreme values", "ok")),
                miss, MANUAL["random"], MANUAL["nointerference"], MANUAL["samemeasure"], MANUAL["once"]]
        methods.append(MethodInfo(name="Main test (Welch t-test)", assumptions=main))

    if mode == "right_answer":
        if kind == "binary":
            methods.append(MethodInfo(name="Exact test (Fisher)", assumptions=[
                ind, Assumption(text="Outcome is yes/no", status="ok")]))
        else:
            methods.append(MethodInfo(name="Rank test (Mann–Whitney)", assumptions=[
                ind, Assumption(text="Group shapes look alike (to read it as a shift)", status="manual")]))
        methods.append(MethodInfo(name="Shuffle test", assumptions=[
            ind, Assumption(text="Versions were assigned at random, so labels are swappable", status="manual")]))
        if kind == "continuous":
            methods.append(MethodInfo(name="Without extreme values", assumptions=[
                ind, Assumption(text="Extreme values are noise, not part of the real effect", status="manual")]))

    if mode == "research":
        methods.append(MethodInfo(name="Slice scan (Holm)", assumptions=[
            Assumption(text="Every slice looked at is counted", status="ok"),
            Assumption(text="50+ rows per group in each slice", status="ok"),
            Assumption(text="Slice columns describe people (country), not results", status="manual"),
            Assumption(text="Slices weren't picked after seeing the results", status="manual"),
            Assumption(text="Main-test assumptions hold inside each slice", status="manual"),
        ]))
    return methods


def attach_method_info(rep: Report, kind: str, mode: str, checks: list[Check],
                       c: np.ndarray, t: np.ndarray) -> Report:
    rep.methods = build_methods(kind, mode, checks, c, t)
    for r in rep.robustness:
        if r.key in STATS:
            rep.audit.append(item(f"{r.name}: p-value", f"{r.p_value:.4g}", r.key,
                                  "See Reproduce script", "# see script"))
    if mode == "research" and rep.segments:
        rep.audit.append(item("Slices checked", str(len(rep.segments)), "holm",
                              "Holm step-down over all slices", "multipletests(p, method='holm')"))
    return rep


# ----------------------------------------------------------------- before / after

def build_methods_ba(mode: str, checks: list[Check], nb: int, na: int, did: bool = False) -> list[MethodInfo]:
    st = lambda name, default="manual": _status(checks, name, default)  # noqa: E731
    weeks_ok = Assumption(text="3+ full weeks on each side", status="ok" if min(nb, na) >= 3 else "warn")
    gaps = Assumption(text="Few missing days", status=st("Missing days", "ok"))
    flat = Assumption(text=("The two groups moved in parallel before the date" if did
                            else "The number was flat before the date (no steady rise or fall)"),
                      status=st("Steady before"))
    linked = Assumption(text="One week doesn't strongly push the next", status=st("Weeks related"))
    manual = [
        Assumption(text=("Nothing else changed for only one group at the date" if did
                         else "Nothing else changed around the date"), status="manual"),
        Assumption(text="The date was chosen before looking at the results", status="manual"),
        Assumption(text="The change took effect on that date (no slow roll-out)", status="manual"),
        *([Assumption(text="The comparison group wasn't affected by the change", status="manual")] if did else []),
    ]
    main = [weeks_ok, gaps, flat, linked,
            Assumption(text="No huge days dominate", status=st("Extreme values", "ok")), *manual]
    methods = [MethodInfo(name=("Main test (weekly gap between groups, Welch t-test)" if did
                                else "Main test (week-by-week, Welch t-test)"), assumptions=main)]

    if mode == "right_answer":
        methods += [
            MethodInfo(name="Fake-date test", assumptions=[
                Assumption(text="6+ full weeks before the date", status="ok" if nb >= 6 else "warn"),
                Assumption(text="Nothing happened on the fake date", status="manual")]),
            MethodInfo(name="Allowing for a trend", assumptions=[
                weeks_ok,
                Assumption(text="Any earlier drift would have continued at the same pace", status="manual"),
                manual[0]]),
            MethodInfo(name="Nearby weeks only", assumptions=[
                Assumption(text="More than 4 weeks on each side", status="ok" if min(nb, na) > 4 else "warn"),
                Assumption(text="Nothing else changed in those weeks", status="manual")]),
            MethodInfo(name="Day-by-day model", assumptions=[
                Assumption(text="Weekday pattern is the same before and after", status="manual"),
                Assumption(text="Day-to-day links fade within 7 days", status="manual"),
                manual[0]]),
        ]
    if mode == "research":
        methods.append(MethodInfo(name="Slice scan (Holm)", assumptions=[
            Assumption(text="Every slice looked at is counted", status="ok"),
            Assumption(text="4+ full weeks on each side in every slice", status="ok"),
            Assumption(text="Slice columns describe people or places, not results", status="manual"),
            Assumption(text="Slices weren't picked after seeing the results", status="manual"),
            Assumption(text="Main-test assumptions hold inside each slice", status="manual"),
        ]))
    return methods


def attach_method_info_ba(rep: Report, mode: str, checks: list[Check], nb: int, na: int, did: bool = False) -> Report:
    rep.methods = build_methods_ba(mode, checks, nb, na, did)
    for r in rep.robustness:
        if r.key in STATS:
            rep.audit.append(item(f"{r.name}: p-value", f"{r.p_value:.4g}", r.key, "See Reproduce script", "# see script"))
    if mode == "research" and rep.segments:
        rep.audit.append(item("Slices checked", str(len(rep.segments)), "holm",
                              "Holm step-down over all slices", "multipletests(p, method='holm')"))
    return rep


# ------------------------------------------------------------------- observational

def build_methods_obs(mode: str, checks: list[Check], covs: list[str]) -> list[MethodInfo]:
    ind = _independent(checks)
    comparable = Assumption(text="Plenty of comparable people on both sides", status=_status(checks, "Comparable people"))
    manual = [
        Assumption(text="Nothing outside the file drives both who chose it and the result", status="manual"),
        Assumption(text="The compared columns were set before people chose (not caused by it)", status="manual"),
    ]
    main = [ind, _missing(checks), comparable,
            Assumption(text="Effects of the compared columns add up (a straight-line model is close enough)", status="manual"),
            *manual]
    methods = [MethodInfo(name="Main test (like-for-like regression)", assumptions=main)]
    if mode == "right_answer":
        methods += [
            MethodInfo(name="Weighting check", assumptions=[ind, comparable, manual[0]]),
            MethodInfo(name="Only comparable people", assumptions=[ind, manual[0]]),
            MethodInfo(name="Without matching people up", assumptions=[ind]),
        ]
    if mode == "research":
        methods.append(MethodInfo(name="Slice scan (Holm)", assumptions=[
            Assumption(text="Every slice looked at is counted", status="ok"),
            Assumption(text="50+ rows per group in each slice", status="ok"),
            Assumption(text="Slice columns describe people, not results", status="manual"),
            Assumption(text="Slices weren't picked after seeing the results", status="manual"),
            manual[0]]))
    return methods


def attach_method_info_obs(rep: Report, mode: str, checks: list[Check], covs: list[str]) -> Report:
    rep.methods = build_methods_obs(mode, checks, covs)
    for r in rep.robustness:
        if r.key in STATS:
            rep.audit.append(item(f"{r.name}: p-value", f"{r.p_value:.4g}", r.key, "See Reproduce script", "# see script"))
    if mode == "research" and rep.segments:
        rep.audit.append(item("Slices checked", str(len(rep.segments)), "holm",
                              "Holm step-down over all slices", "multipletests(p, method='holm')"))
    return rep
