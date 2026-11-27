"""Science mode: turns the engine's numbers into words.

Everything here is fixed code. It writes the read-back sentence before anything runs, and the report afterwards.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from ..interpret import chance_phrase
from ..schemas import Assumption, AuditItem, Check
from .schemas import Finding, ScienceReport

# ------------------------------------------------------------------------------------------------ formatting


def num(x) -> str:
    """Four significant figures, halves rounded up, trailing zeros dropped: exactly what the page shows for the same
    number (JavaScript's toPrecision), so a value never reads differently in two places."""
    if x is None:
        return "n/a"
    if x == 0:
        return "0"
    if not 1e-3 <= abs(x) < 1e6:
        return f"{x:.2e}".replace("e-0", "e-").replace("e+0", "e+")
    exact = Decimal(x)
    text = format(exact.quantize(Decimal(1).scaleb(exact.adjusted() - 3), rounding=ROUND_HALF_UP), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def p_is(p) -> str:
    """'P = 0.0123', or 'P < 0.0001'."""
    text = pval(p)
    return f"P {text}" if text.startswith("<") else f"P = {text}"


def pval(p) -> str:
    if p is None:
        return "n/a"
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def signed(x) -> str:
    return "n/a" if x is None else ("+" if x >= 0 else "−") + num(abs(x))


def _listing(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ------------------------------------------------------------------------------------------------ method names

def main_test(spec: dict) -> str:
    k, paired, ranks = len(spec.get("groups", [])), spec.get("paired"), spec.get("method") == "ranks"
    if paired:
        if k == 2:
            return "Wilcoxon signed-rank test" if ranks else "paired t test"
        return "Friedman test" if ranks else "repeated-measures one-way ANOVA, Greenhouse-Geisser corrected"
    if k == 2:
        return "Mann-Whitney test" if ranks else ("unpaired t test" if spec.get("equal_spread") else "Welch's t test")
    return "Kruskal-Wallis test" if ranks else ("ordinary one-way ANOVA" if spec.get("equal_spread") else "Welch's ANOVA")


def followup_test(spec: dict) -> str:
    if len(spec.get("groups", [])) < 3 or spec.get("followup", "none") == "none":
        return ""
    if spec.get("paired"):
        return ("Wilcoxon signed-rank tests, Holm-adjusted" if spec["method"] == "ranks"
                else "paired t tests, Bonferroni-adjusted")
    if spec["method"] == "ranks":
        return "Dunn's test, Holm-adjusted"
    if not spec.get("equal_spread"):
        return "Games-Howell test"
    return "Tukey's test" if spec["followup"] == "all_pairs" else "Dunnett's test"


def _a(name: str) -> str:
    """'a paired t test', 'an ordinary one-way ANOVA', "Welch's ANOVA"."""
    if "'s " in name:
        return name
    return ("an " if name[0].lower() in "aeiou" else "a ") + name


def method_why(spec: dict) -> str:
    k, paired, ranks = len(spec["groups"]), spec["paired"], spec["method"] == "ranks"
    bits = []
    bits.append("The same subjects were measured in every group, so each subject is compared with itself."
                if paired else "Each group holds different subjects.")
    if ranks:
        bits.append("You chose to compare ranks, so the test asks whether values in one group tend to be higher, "
                    "and a few extreme values barely matter.")
    else:
        bits.append("You chose to compare averages, which assumes the "
                    + ("differences within each subject are" if paired else "values in each group are")
                    + " roughly bell-shaped (or the groups are large).")
        if not paired:
            bits.append("Spread is assumed equal across groups, as in the classic test." if spec.get("equal_spread")
                        else "Groups may have different spreads, so the Welch version is used. It's barely less "
                        "powerful when spreads happen to be equal.")
        if paired and k > 2:
            bits.append("The Greenhouse-Geisser correction is always applied, so the test doesn't rely on every "
                        "pair of conditions varying equally.")
    if k > 2:
        bits.append("With more than two groups, one overall test asks whether any group differs, so you aren't "
                    "running a separate test for every pair.")
        f = followup_test(spec)
        if f:
            target = "each group is compared with the control" if spec["followup"] == "vs_control" \
                else "every pair of groups is compared"
            bits.append(f"Then {target} ({f}), with P values adjusted for the number of comparisons.")
    return " ".join(bits)


# ------------------------------------------------------------------------------------------------ read-back

ASSIGN = {"random": "Subjects were assigned to groups at random.",
          "not_random": "Subjects were not assigned to groups at random.",
          "unsure": "Not sure whether subjects were assigned at random, so it's treated as not random."}


def readback(spec: dict) -> str:
    a = spec["analysis"]
    if a == "compare_groups":
        where = (f"the columns {_listing([repr(g) for g in spec['groups']])}" if spec["layout"] == "wide"
                 else f"'{spec['value_col']}' for the groups {_listing([repr(g) for g in spec['groups']])} "
                      f"of '{spec['group_col']}'")
        text = (f"Compare {where}, " + ("measured on the same subjects" if spec["paired"] else "each a separate set of "
                                        "subjects") + f", using {_a(main_test(spec))}.")
        f = followup_test(spec)
        if f:
            text += (f" Then compare each group with '{spec['control']}'" if spec["followup"] == "vs_control"
                     else " Then compare every pair of groups") + f" ({f})."
        if spec.get("subset_col"):
            text += f" Only rows where '{spec['subset_col']}' is {spec['subset_value']} are used."
        text += " Empty cells are left out."
        if spec["paired"]:
            text += " Subjects not measured in every group are left out."
        return text + " " + ASSIGN[spec.get("assignment", "unsure")]
    if a == "contingency":
        if spec["layout"] == "counts":
            src = f"the counts in {_listing([repr(c) for c in spec['cols']])} for {_listing([repr(r) for r in spec['rows']])}"
        else:
            src = (f"how many rows have each value of '{spec['row_col']}' ({_listing([repr(r) for r in spec['rows']])}) "
                   f"and of '{spec['col_col']}' ({_listing([repr(c) for c in spec['cols']])})")
        text = f"Count {src}."
        if len(spec["rows"]) == 2 and len(spec["cols"]) == 2:
            text += (f" Test with Fisher's exact test, and compare the share with '{spec.get('event') or spec['cols'][0]}' "
                     f"in '{spec['rows'][0]}' against '{spec['rows'][1]}' (relative risk, odds ratio, difference).")
        else:
            text += " Test with the chi-square test."
        if spec["layout"] == "rows":
            text += " Rows with other or empty values are left out."
        return text + " " + ASSIGN[spec.get("assignment", "unsure")]
    if a == "time_course":
        text = (f"Follow each '{spec['subject_col']}' over '{spec['time_col']}' ({_listing(spec['times'])}) and compare "
                f"how '{spec['value_col']}' changes in the groups {_listing([repr(g) for g in spec['groups']])} of "
                f"'{spec['group_col']}', using a two-way repeated-measures ANOVA (group × {spec['time_col'].lower()}, "
                "Greenhouse-Geisser corrected).")
        if spec.get("followup", "none") != "none":
            who = (f"each group with '{spec['control']}'" if spec["followup"] == "vs_control" else "every pair of groups")
            text += (f" Then, at each {spec['time_col'].lower()}, compare {who} (Welch t tests, Holm-adjusted across "
                     "all of those comparisons).")
        text += f" Subjects missing any of these {spec['time_col'].lower()}s are left out."
        return text + " " + ASSIGN[spec.get("assignment", "unsure")]
    name = "Pearson (straight-line)" if spec["method"] == "pearson" else "Spearman (rank)"
    return (f"Measure how closely '{spec['y_col']}' moves with '{spec['x_col']}' using {name} correlation. "
            "Rows missing either value are left out.")


# ------------------------------------------------------------------------------------------------ definitions

STATS = {
    "mean_ci": ("Mean ± t(0.975, n−1) × SD/√n.", "Where this group's true average likely sits."),
    "sd": ("Sample standard deviation (n − 1 denominator).", "How spread out the values in this group are."),
    "sem": ("SD ÷ √n.", "How precisely the average is pinned down. Not the spread of the values."),
    "median": ("Middle value (50th percentile, linear interpolation).", "Half the values are above, half below."),
    "p": ("Two-sided P value: the probability of a result at least this extreme if the null hypothesis is true.",
          "If the groups truly didn't differ, how often a result like this would turn up. Small = hard to blame on "
          "chance. Not the chance the result is wrong."),
    "p_adj": ("P value adjusted for the number of comparisons by the named method.",
              "Like the P value, but allowing for having made several comparisons."),
    "mean_diff": ("Mean of the second group minus the mean of the first, with its 95% confidence interval.",
                  "How far apart the averages are, and the likely range for the true gap."),
    "median_diff": ("Median of the second group minus the median of the first (paired: median of the differences).",
                    "How far apart the middle values are. Descriptive: no interval is given."),
    "superiority": ("U ÷ (n₁ × n₂): the probability that a random value from the second group beats one from the "
                    "first (ties count half).", "0.5 means neither group tends to be higher."),
    "eta_squared": ("η² = between-group sum of squares ÷ total sum of squares.",
                    "Share of all the variation that lines up with the groups. 0 = none, 1 = all of it."),
    "omega_squared": ("ω² = (SSB − (k−1)·MSW) ÷ (SST + MSW), truncated at 0. Less biased than η².",
                      "Like η², corrected for the tendency to overstate in small samples."),
    "eta_squared_partial": ("Partial η² = SS(conditions) ÷ (SS(conditions) + SS(error)), subject differences removed.",
                            "Share of the within-subject variation that lines up with the conditions."),
    "kendall_w": ("Kendall's W = χ²(Friedman) ÷ (n(k−1)).", "0 = conditions don't rank subjects consistently, 1 = "
                  "every subject ranks the conditions the same way."),
    "epsilon_squared": ("ε² = H ÷ (n − 1).", "Rough share of the variation in ranks that lines up with the groups."),
    "epsilon_pooled": ("Greenhouse-Geisser ε from the covariance of the time points within groups, pooled across "
                       "groups and double-centred. The time and group × time degrees of freedom are multiplied by it.",
                       "1 = every pair of time points varies equally; smaller values make the test more cautious."),
    "epsilon": ("Greenhouse-Geisser ε from the double-centred covariance matrix; both degrees of freedom are "
                "multiplied by it.", "1 = conditions vary equally; smaller values make the test more cautious."),
    "relative_risk": ("Risk in the first row ÷ risk in the second, 95% interval on the log scale (Katz).",
                      "How many times as likely the outcome is in the first group."),
    "odds_ratio": ("Sample odds ratio (ad)/(bc), 95% interval on the log scale (Woolf).",
                   "How the odds compare. Close to the relative risk only when the outcome is rare."),
    "risk_diff": ("Difference of two proportions, 95% Newcombe hybrid-score interval.",
                  "The gap in percentage points."),
    "cramers_v": ("Cramér's V = √(χ² ÷ (n × (min(rows, cols) − 1))).", "0 = no relationship, 1 = perfectly linked."),
    "r": ("Pearson's r, 95% interval from the Fisher z transform.",
          "−1 to +1: how close the points are to a straight line, and which way it slopes."),
    "rho": ("Spearman's rank correlation, 95% interval from the Fisher z transform with the Bonett-Wright standard "
            "error.", "−1 to +1: whether one tends to rise as the other rises, in any steady shape."),
}


def _item(label: str, value: str, key: str, code: str) -> AuditItem:
    technical, plain = STATS[key]
    return AuditItem(label=label, value=value, method=key, code=code, technical=technical, plain=plain)


# ------------------------------------------------------------------------------------------------ the report

def _causal(spec: dict) -> str:
    if spec["analysis"] == "correlation":
        return "Moving together isn't the same as one causing the other: a third thing could drive both."
    who = spec.get("assignment", "unsure")
    if who == "random":
        return ("You said subjects were assigned at random, so a clear difference can be put down to what the groups "
                "received, as long as the experiment ran as planned.")
    return ("Subjects weren't assigned at random" if who == "not_random" else
            "It isn't known whether subjects were assigned at random") + (
        ", so this shows the groups differ, not why. Something else that differs between the groups could explain it.")


def _groups_report(spec: dict, r: dict) -> dict:
    names = spec["groups"]
    k, ranks, paired = len(names), spec["method"] == "ranks", spec["paired"]
    main = r["tests"][0]
    e = r["effect"]
    g = {x["name"]: x for x in r["groups"]}
    checks, assume, findings, audit = [], [], [], []

    # --- words
    if k == 2:
        a, b = names
        if ranks:
            mid = f"Medians: {a} {num(g[a]['median'])}, {b} {num(g[b]['median'])}"
            gap = f"{'median difference' if paired else 'difference in medians'} {signed(e['diff'])}"
            answer = f"{mid} ({gap})."
            higher = (e["diff"] or 0) > 0 if paired else (e.get("superiority") or 0.5) > 0.5
            headline = (f"{b} tended to be {'higher' if higher else 'lower'} than {a}." if r["clear"]
                        else f"No clear difference between {a} and {b}.")
        else:
            answer = (f"Means: {a} {num(g[a]['mean'])}, {b} {num(g[b]['mean'])}. "
                      f"{'Mean of the differences' if paired else 'Difference'} {signed(e['diff'])} "
                      f"(95% CI {signed(e['ci_low'])} to {signed(e['ci_high'])}).")
            headline = (f"{b} averaged {'higher' if e['diff'] > 0 else 'lower'} than {a}." if r["clear"]
                        else f"No clear difference between the averages of {a} and {b}.")
    else:
        what = "values" if ranks else "averages"
        headline = (f"The groups' {what} differ." if r["clear"] else f"No clear difference between the groups' {what}.")
        answer = ", ".join(f"{n} {num(g[n]['median' if ranks else 'mean'])}" for n in names)
        answer = ("Medians: " if ranks else "Means: ") + answer + "."
        if r["pairs"]:
            clear = [f"{p['b']} vs {p['a']}" for p in r["pairs"] if p["clear"]]
            answer += (f" Clear in the follow-up tests: {_listing(clear)}." if clear
                       else " No pair is clearly different in the follow-up tests.")
            if clear and not r["clear"]:
                checks.append(Check(name="Overall vs pairs", status="warn", message=(
                    "A pair looks clearly different although the overall test didn't. Treat that pair as a lead to "
                    "check, not a finding.")))
            elif r["clear"] and not clear:
                checks.append(Check(name="Overall vs pairs", status="info", message=(
                    "The overall test found a difference but no single pair is clear. That happens when several small "
                    "gaps add up, or the gap is between combinations of groups.")))
    how = (f"{p_is(main['p'])}. If the groups truly didn't differ, a result like this would turn up "
           f"{chance_phrase(main['p'])}.")
    if not r["clear"]:
        how += " Not proof that the groups are the same: small or noisy samples often miss real differences."

    # --- results table
    stat = (f"{main['statistic_name']}" + (f"({main['df']})" if main["df"] else "") + f" = {num(main['statistic'])}"
            if main["statistic"] is not None else "")
    findings.append(Finding(label="Test", value=main["name"]))
    if stat:
        findings.append(Finding(label="Test statistic", value=stat))
    findings.append(Finding(label="P value", value=pval(main["p"])))
    findings.append(Finding(label="Clear difference at P < 0.05?", value="Yes" if r["clear"] else "No"))
    if main.get("epsilon") is not None:
        findings.append(Finding(label="Greenhouse-Geisser ε", value=num(main["epsilon"])))
        audit.append(_item("Greenhouse-Geisser ε", num(main["epsilon"]), "epsilon", 'result["tests"][0]["epsilon"]'))
    if e.get("kind") == "mean_diff":
        findings.append(Finding(label=f"Difference ({e['b']} − {e['a']})", value=f"{signed(e['diff'])} "
                                f"(95% CI {signed(e['ci_low'])} to {signed(e['ci_high'])})"))
        audit.append(_item("Difference and 95% CI", f"{signed(e['diff'])} ({signed(e['ci_low'])} to "
                           f"{signed(e['ci_high'])})", "mean_diff", 'result["effect"]["diff"], ["ci_low"], ["ci_high"]'))
    elif e.get("kind") == "median_diff":
        findings.append(Finding(label=f"Median difference ({e['b']} − {e['a']})", value=signed(e["diff"])))
        audit.append(_item("Median difference", signed(e["diff"]), "median_diff", 'result["effect"]["diff"]'))
        if e.get("superiority") is not None:
            findings.append(Finding(label=f"Chance a {e['b']} value beats a {e['a']} value", value=num(e["superiority"])))
            audit.append(_item("Probability of superiority", num(e["superiority"]), "superiority",
                               'result["effect"]["superiority"]'))
    else:
        label = {"eta_squared": "η² (effect size)", "eta_squared_partial": "Partial η² (effect size)",
                 "kendall_w": "Kendall's W (effect size)", "epsilon_squared": "ε² (effect size)"}[e["kind"]]
        findings.append(Finding(label=label, value=num(e["value"])))
        audit.append(_item(label, num(e["value"]), e["kind"], 'result["effect"]["value"]'))
        if e.get("omega_squared") is not None:
            findings.append(Finding(label="ω² (effect size)", value=num(e["omega_squared"])))
            audit.append(_item("ω²", num(e["omega_squared"]), "omega_squared", 'result["effect"]["omega_squared"]'))
    audit.insert(0, _item("P value", pval(main["p"]), "p", 'result["tests"][0]["p"]'))
    for i, x in enumerate(r["groups"]):
        audit.append(_item(f"{x['name']}: mean and 95% CI", f"{num(x['mean'])} ({num(x['ci_low'])} to "
                           f"{num(x['ci_high'])}), n = {x['n']}", "mean_ci", f'result["groups"][{i}]["mean"]'))
        audit.append(_item(f"{x['name']}: SD", num(x["sd"]), "sd", f'result["groups"][{i}]["sd"]'))
        audit.append(_item(f"{x['name']}: SEM", num(x["sem"]), "sem", f'result["groups"][{i}]["sem"]'))
        audit.append(_item(f"{x['name']}: median", num(x["median"]), "median", f'result["groups"][{i}]["median"]'))
    for i, p in enumerate(r["pairs"]):
        audit.append(_item(f"{p['b']} vs {p['a']}: adjusted P", pval(p["p_adj"]), "p_adj", f'result["pairs"][{i}]["p_adj"]'))

    # --- checks
    kept = r["left_out"].get("rows_kept")
    if kept:
        checks.append(Check(name="Part of the data", status="info", message=(
            f"Only rows where {kept['column']} is {kept['value']} were used ({kept['kept']} of {kept['of']}). Pick the "
            f"{kept['column'].lower()} before looking at results: testing each one separately and reporting the "
            "clearest makes a false alarm much more likely.")))
    dropped = sum(r["left_out"]["empty_values"].values())
    if dropped:
        per = ", ".join(f"{n}: {c}" for n, c in r["left_out"]["empty_values"].items() if c)
        checks.append(Check(name="Empty cells", status="info", message=f"{dropped} empty cell{'s' if dropped > 1 else ''} "
                            f"left out ({per}). If a value is missing for a reason linked to the result (an animal "
                            "died, a sample failed), leaving it out can bias the answer."))
    if r["left_out"]["incomplete_subjects"]:
        n = r["left_out"]["incomplete_subjects"]
        checks.append(Check(name="Incomplete subjects", status="warn", message=(
            f"{n} subject{'s were' if n > 1 else ' was'} not measured in every group and left out entirely. A "
            "mixed-effects model can use them; it isn't available yet.")))
    small = [x["name"] for x in r["groups"] if x["n"] < 5]
    if small:
        checks.append(Check(name="Small groups", status="warn", message=(
            f"{_listing(small)} {'has' if len(small) == 1 else 'have'} fewer than 5 values. Only large differences can "
            "be detected, and the shape of the values can't really be checked.")))
    far = [f"{x['name']} ({x['far_values']})" for x in r["groups"] if x["far_values"]]
    if far:
        checks.append(Check(name="Far-out values", status="info", message=(
            f"Values far outside the rest of their group: {_listing(far)}. They were kept. Check they aren't typing "
            "or measurement errors; never remove them just because they change the answer.")))
    shape_ok = True
    if not ranks:
        if paired:
            pn = r["paired_normality"]["p"]
            skewed = [f"the {'differences' if k == 2 else 'within-subject differences'} ({p_is(pn)})"] \
                if pn is not None and pn < 0.05 else []
        else:
            skewed = [f"{x['name']} ({p_is(x['normality']['p'])})" for x in r["groups"]
                      if x["normality"]["p"] is not None and x["normality"]["p"] < 0.05]
        shape_ok = not skewed
        if skewed:
            checks.append(Check(name="Shape of the values", status="warn", message=(
                f"Values in {_listing(skewed)} don't look bell-shaped (Shapiro-Wilk). Averages can mislead here; "
                "consider comparing ranks instead. Decide on scientific grounds, not on which gives the smaller P.")))
        else:
            checks.append(Check(name="Shape of the values", status="info", message=(
                ("The differences don't look clearly skewed" if paired else "No group looks clearly skewed") +
                " (Shapiro-Wilk). With small groups this check rarely objects, so it "
                "doesn't prove the values are bell-shaped.")))
        ratio = r.get("spread_ratio")
        if not paired and ratio is not None and ratio > 2:
            status = "warn" if spec.get("equal_spread") else "info"
            msg = (f"The largest SD is {num(ratio)} times the smallest. "
                   + ("The equal-spread assumption looks doubtful: consider the Welch version." if spec.get("equal_spread")
                      else "The Welch version you're using doesn't assume equal spread."))
            checks.append(Check(name="Spread", status=status, message=msg))

    # --- assumptions
    assume.append(Assumption(status="manual", text=(
        "Each subject is independent: different animals, patients or cultures, not technical replicates of the same "
        "sample counted as separate values. If they are replicates, average them per subject first.")))
    if paired:
        assume.append(Assumption(status="manual", text="The rows (or subject labels) really match the same subject "
                                 "across groups."))
    if not ranks:
        subject = "The differences between conditions are" if paired else "Values in each group are"
        assume.append(Assumption(status="manual" if shape_ok else "warn", text=(
            f"{subject} roughly bell-shaped, or the groups are large. A check found no clear problem, which isn't proof."
            if shape_ok else f"{subject} roughly bell-shaped, but at least one group looks skewed (see checks).")))
        if not paired and spec.get("equal_spread"):
            ratio = r.get("spread_ratio") or 1
            assume.append(Assumption(status="ok" if ratio <= 2 else "warn",
                                     text=f"Groups have similar spread (largest SD ÷ smallest = {num(ratio)})."))
        if paired and k > 2:
            assume.append(Assumption(status="ok", text=(
                f"Not assumed: equal variation between every pair of conditions (sphericity). The Greenhouse-Geisser "
                f"correction is applied (ε = {num(main.get('epsilon'))}).")))
    else:
        assume.append(Assumption(status="manual", text=(
            "To read the result as a shift in the middle value, the groups should have similar shapes. Otherwise it "
            "says only that one group tends to have higher values.")))

    cannot = ["Whether the difference matters scientifically: judge that from its size, not from the P value.",
              "Subjects or conditions that aren't in this data.",
              "If you run several analyses on this data, these P values aren't adjusted for that."]
    return {"headline": headline, "answer": answer, "how_sure": how, "checks": checks, "assumptions": assume,
            "results": findings, "audit": audit, "cannot": cannot, "method": main_test(spec), "why": method_why(spec)}


def _table_report(spec: dict, r: dict) -> dict:
    main = r["tests"][0]
    e = r["effect"]
    checks, assume, findings, audit = [], [], [], []
    findings.append(Finding(label="Test", value=main["name"]))
    if main["statistic"] is not None:
        findings.append(Finding(label="Test statistic", value=f"χ²({main['df']}) = {num(main['statistic'])}"))
    findings.append(Finding(label="P value", value=pval(main["p"])))
    findings.append(Finding(label="Clear relationship at P < 0.05?", value="Yes" if r["clear"] else "No"))
    audit.append(_item("P value", pval(main["p"]), "p", 'result["tests"][0]["p"]'))
    a_name, b_name = r["rows"][0], r["rows"][1]
    if e["kind"] == "two_by_two":
        ev = r["event"]
        answer = (f"'{ev}': {num(e['risk_a'] * 100)}% of {a_name} vs {num(e['risk_b'] * 100)}% of {b_name}. "
                  f"Relative risk {num(e['relative_risk'])} (95% CI {num(e['rr_low'])} to {num(e['rr_high'])}).")
        higher = (e["risk_diff"] or 0) > 0
        headline = (f"'{ev}' was {'more' if higher else 'less'} common in {a_name} than in {b_name}." if r["clear"]
                    else f"No clear difference in how common '{ev}' was.")
        for label, key, lo, hi, kind in (("Relative risk", "relative_risk", "rr_low", "rr_high", "relative_risk"),
                                         ("Odds ratio", "odds_ratio", "or_low", "or_high", "odds_ratio")):
            val = f"{num(e[key])} (95% CI {num(e[lo])} to {num(e[hi])})"
            findings.append(Finding(label=f"{label} ({a_name} vs {b_name})", value=val))
            audit.append(_item(label, val, kind, f'result["effect"]["{key}"]'))
        rd = (f"{signed(e['risk_diff'] * 100)} pts (95% CI {signed(e['rd_low'] * 100)} to "
              f"{signed(e['rd_high'] * 100)} pts)")
        findings.append(Finding(label="Difference in risk", value=rd))
        audit.append(_item("Difference in risk", rd, "risk_diff", 'result["effect"]["risk_diff"]'))
        if e["relative_risk"] is None or e["odds_ratio"] is None:
            checks.append(Check(name="Zero counts", status="info", message=(
                "A cell is 0, so a ratio can't be estimated. Fisher's test and the difference in risk still work.")))
    else:
        answer = f"Cramér's V = {num(e['value'])} across {len(r['rows'])} × {len(r['cols'])} categories."
        headline = ("The categories are related." if r["clear"] else "No clear relationship between the categories.")
        findings.append(Finding(label="Cramér's V (effect size)", value=num(e["value"])))
        audit.append(_item("Cramér's V", num(e["value"]), "cramers_v", 'result["effect"]["value"]'))
        if (r["expected_small_share"] or 0) > 0.2 or (r["expected_min"] or 0) < 1:
            checks.append(Check(name="Small counts", status="warn", message=(
                "More than 1 in 5 cells expect fewer than 5 counts (or one expects fewer than 1), so the chi-square P "
                "value may be off. Combining sparse categories, decided before looking at the result, can help.")))
        assume.append(Assumption(status="ok" if (r["expected_small_share"] or 0) <= 0.2 else "warn",
                                 text="Expected counts are large enough for the chi-square approximation."))
    how = (f"{p_is(main['p'])}. If the categories were unrelated, a pattern like this would turn up "
           f"{chance_phrase(main['p'])}.")
    if not r["clear"]:
        how += " Not proof there's no relationship."
    if r["left_out"]:
        checks.append(Check(name="Rows left out", status="info", message=(
            f"{r['left_out']} row{'s' if r['left_out'] > 1 else ''} had another value or an empty cell in one of the "
            "two columns and weren't counted.")))
    assume.insert(0, Assumption(status="manual", text=(
        "Each subject is counted once, in one cell. Counting the same animal or patient twice breaks every test here.")))
    cannot = ["Whether the difference matters scientifically: judge that from its size, not from the P value.",
              "If you run several analyses on this data, these P values aren't adjusted for that."]
    return {"headline": headline, "answer": answer, "how_sure": how, "checks": checks, "assumptions": assume,
            "results": findings, "audit": audit, "cannot": cannot, "method": main["name"],
            "why": ("Fisher's exact test works out the P value exactly, so it's reliable even with small counts."
                    if e["kind"] == "two_by_two" else
                    "The chi-square test compares the counts with what you'd expect if the categories were unrelated.")}


def _correlation_report(spec: dict, r: dict) -> dict:
    main, e = r["tests"][0], r["effect"]
    sym = "r" if spec["method"] == "pearson" else "Spearman's rs"
    val = f"{num(e['value'])} (95% CI {num(e['ci_low'])} to {num(e['ci_high'])})"
    direction = "rises" if e["value"] > 0 else "falls"
    headline = (f"'{spec['y_col']}' tends to {direction[:-1]} as '{spec['x_col']}' rises." if r["clear"]
                else f"No clear relationship between '{spec['x_col']}' and '{spec['y_col']}'.")
    answer = f"{sym} = {val}, from {r['n']} pairs."
    how = (f"{p_is(main['p'])}. If the two weren't related, a pattern like this would turn up "
           f"{chance_phrase(main['p'])}.")
    findings = [Finding(label="Test", value=main["name"]), Finding(label=sym, value=val),
                Finding(label="P value", value=pval(main["p"])), Finding(label="Pairs used", value=str(r["n"]))]
    audit = [_item(sym, val, "r" if spec["method"] == "pearson" else "rho", 'result["effect"]["value"]'),
             _item("P value", pval(main["p"]), "p", 'result["tests"][0]["p"]')]
    if spec["method"] == "pearson":
        findings.append(Finding(label="r²", value=num(e["r_squared"])))
        findings.append(Finding(label="Best straight line", value=f"y = {num(e['slope'])} × x {'+' if e['intercept'] >= 0 else '−'} {num(abs(e['intercept']))}"))
    checks = []
    if r["n"] < 10:
        checks.append(Check(name="Few pairs", status="warn", message=(
            f"Only {r['n']} pairs. The interval is wide and one point can change the answer.")))
    if r["left_out"]:
        checks.append(Check(name="Rows left out", status="info", message=f"{r['left_out']} rows were missing a value."))
    assume = [Assumption(status="manual", text="Each pair comes from a different, independent subject."),
              Assumption(status="manual", text=(
                  "Neither measurement was used to calculate the other (that builds in a correlation)."))]
    if spec["method"] == "pearson":
        assume.append(Assumption(status="manual", text=(
            "The relationship is roughly a straight line. Look at the plot: a curve or a few far-out points can make r "
            "misleading. Spearman doesn't need a straight line.")))
    cannot = ["Whether one causes the other.", "What happens outside the range of values measured."]
    return {"headline": headline, "answer": answer, "how_sure": how, "checks": checks, "assumptions": assume,
            "results": findings, "audit": audit, "cannot": cannot, "method": main["name"],
            "why": ("Pearson measures how close the points are to a straight line." if spec["method"] == "pearson"
                    else "Spearman works on ranks, so it catches any steady rise or fall and isn't thrown by a few "
                    "extreme values.")}


def _time_report(spec: dict, r: dict) -> dict:
    inter, group, time = r["tests"]
    when = spec["time_col"].lower()
    checks, findings, audit = [], [], []

    # --- words: the interaction is the question "did the groups change differently?"
    last = r["times"][-1]
    ends = ", ".join(f"{m['group']} {num(m['points'][-1]['mean'])}" for m in r["means"])
    if r["clear"]:
        headline = f"The groups changed differently over {when}s."
    elif (group["p"] or 1) < 0.05:
        headline = f"The groups differ overall, but there's no clear sign they changed differently over {when}s."
    else:
        headline = f"No clear difference in how the groups changed over {when}s."
    answer = f"Averages at {spec['time_col']} {last}: {ends}."
    if r["pairs"]:
        clear = [p for p in r["pairs"] if p["clear"]]
        if clear:
            by_time: dict = {}
            for p in clear:
                by_time.setdefault(p["time"], []).append(p["b"] if spec["followup"] == "vs_control" else f"{p['b']} vs {p['a']}")
            against = f" (each group vs {spec['control']})" if spec["followup"] == "vs_control" else ""
            answer += f" Clear in the follow-up tests{against}: " + "; ".join(
                f"{spec['time_col']} {t}: {_listing(v)}" for t, v in by_time.items()) + "."
        else:
            answer += " No single comparison at any one " + when + " is clear in the follow-up tests."
    how = (f"Group × {when}: {p_is(inter['p'])}. If the groups truly changed in the same way, a pattern like this "
           f"would turn up {chance_phrase(inter['p'])}.")
    if not r["clear"]:
        how += " Not proof that they changed the same way: small groups often miss real differences."
    if r["clear"]:
        how += (f" Because the groups changed differently, the overall group and {when} effects below average over "
                f"different patterns: read the {when}-by-{when} comparisons instead.")

    # --- results table, Prism-style
    findings.append(Finding(label="Test", value=f"Two-way repeated-measures ANOVA (Greenhouse-Geisser ε = {num(r['epsilon'])})"))
    for t, label in ((inter, f"Group × {when} (do the groups change differently?)"),
                     (group, "Group (do the groups differ, averaged over time?)"),
                     (time, f"{spec['time_col']} (does the value change over time, averaged over groups?)")):
        findings.append(Finding(label=label, value=f"F({t['df']}) = {num(t['statistic'])}, P {pval(t['p']) if pval(t['p']).startswith('<') else '= ' + pval(t['p'])}, partial η² = {num(t['partial_eta_squared'])}"))
        audit.append(_item(f"{label}: P", pval(t["p"]), "p", f'result["tests"][{r["tests"].index(t)}]["p"]'))
        audit.append(_item(f"{t['name']}: partial η²", num(t["partial_eta_squared"]), "eta_squared_partial",
                           f'result["tests"][{r["tests"].index(t)}]["partial_eta_squared"]'))
    findings.append(Finding(label="Subjects used", value=" · ".join(f"{m['group']}: {m['n']}" for m in r["means"])))
    audit.append(_item("Greenhouse-Geisser ε", num(r["epsilon"]), "epsilon_pooled", 'result["epsilon"]'))
    for i, m in enumerate(r["means"]):
        for j, pt in enumerate(m["points"]):
            audit.append(_item(f"{m['group']}, {spec['time_col']} {pt['time']}: mean and 95% CI",
                               f"{num(pt['mean'])} ({num(pt['ci_low'])} to {num(pt['ci_high'])})", "mean_ci",
                               f'result["means"][{i}]["points"][{j}]["mean"]'))
    for i, p in enumerate(r["pairs"]):
        audit.append(_item(f"{spec['time_col']} {p['time']}: {p['b']} vs {p['a']}, adjusted P", pval(p["p_adj"]),
                           "p_adj", f'result["pairs"][{i}]["p_adj"]'))

    # --- checks
    lo = r["left_out"]
    if lo["incomplete_subjects"]:
        per = ", ".join(f"{g}: {c}" for g, c in lo["incomplete_by_group"].items() if c)
        checks.append(Check(name="Incomplete subjects", status="warn", message=(
            f"{lo['incomplete_subjects']} subject{'s were' if lo['incomplete_subjects'] > 1 else ' was'} missing at "
            f"least one {when} and left out entirely ({per}). If they're missing for a reason linked to the result "
            "(an animal culled because its tumour grew too big), leaving them out biases the answer. A mixed-effects "
            "model can use their partial data; it isn't available yet.")))
    small = [m["group"] for m in r["means"] if m["n"] < 5]
    if small:
        checks.append(Check(name="Small groups", status="warn", message=(
            f"{_listing(small)} {'has' if len(small) == 1 else 'have'} fewer than 5 subjects. Only large differences "
            "can be detected.")))
    checks.append(Check(name="Sphericity", status="info", message=(
        f"Greenhouse-Geisser ε = {num(r['epsilon'])} (1 means every pair of {when}s varies equally). The correction is "
        "always applied, so this assumption isn't needed." + (" ε is well below 1 here, so the correction matters."
                                                             if (r["epsilon"] or 1) < 0.75 else ""))))
    pn = r["residual_normality"]["p"]
    if pn is not None and pn < 0.05:
        checks.append(Check(name="Shape of the values", status="warn", message=(
            f"What's left after removing each subject's level and each group's time course doesn't look bell-shaped "
            f"(Shapiro-Wilk {p_is(pn)}). With many values this check objects to small departures that don't matter; "
            "with skewed measurements (sizes and concentrations often are), analysing their logarithm is common. Decide that "
            "on scientific grounds, before looking at results.")))
    else:
        checks.append(Check(name="Shape of the values", status="info", message=(
            "The leftover variation doesn't look clearly skewed (Shapiro-Wilk). That doesn't prove it's bell-shaped.")))
    if (r["spread_ratio"] or 1) > 3:
        checks.append(Check(name="Spread", status="warn", message=(
            f"At some {when}, one group's SD is {num(r['spread_ratio'])} times another's. The ANOVA assumes groups "
            "vary similarly at each time; the follow-up Welch tests don't.")))
    if r["pairs"]:
        checks.append(Check(name="Follow-up comparisons", status="info", message=(
            f"{len(r['pairs'])} comparisons ({len(r['times'])} {when}s × {len(r['pairs']) // len(r['times'])} "
            f"pair{'s' if len(r['pairs']) // len(r['times']) > 1 else ''}), Holm-adjusted together. Leaving out "
            f"{when}s you don't care about (such as the first day, before treatment) gives the rest more power, but "
            "choose them before looking.")))
    assume = [
        Assumption(status="manual", text="Each subject belongs to one group, and different subjects are independent "
                   "(different animals or patients, not technical replicates)."),
        Assumption(status="manual" if pn is None or pn >= 0.05 else "warn", text=(
            "The leftover variation is roughly bell-shaped." + (" A check found no clear problem, which isn't proof."
                                                                 if pn is None or pn >= 0.05 else " See checks."))),
        Assumption(status="ok", text=(f"Not assumed: equal variation between every pair of {when}s (sphericity). "
                                      f"The Greenhouse-Geisser correction is applied (ε = {num(r['epsilon'])}).")),
        Assumption(status="manual", text=(f"The groups vary in a similar way over {when}s (their covariance "
                                          "patterns are alike). It isn't checked here.")),
    ]
    cannot = ["Whether the difference matters scientifically: judge that from its size, not from the P value.",
              f"What happens after the last {when} measured, or between the {when}s measured.",
              "If you run several analyses on this data, these P values aren't adjusted for that."]
    return {"headline": headline, "answer": answer, "how_sure": how, "checks": checks, "assumptions": assume,
            "results": findings, "audit": audit, "cannot": cannot,
            "method": "two-way repeated-measures ANOVA",
            "why": (f"Each subject was measured at every {when}, and each belongs to one group. This test separates "
                    f"three questions: do the groups change differently over {when}s (the group × {when} "
                    f"interaction, usually the one that matters), do they differ overall, and does the value change "
                    f"over {when}s. Comparing each {when} with separate tests instead would ignore that the same "
                    f"subjects were measured each time, and multiply the chances of a false alarm.")}


def build(spec: dict, result: dict) -> ScienceReport:
    maker = {"compare_groups": _groups_report, "contingency": _table_report, "correlation": _correlation_report,
             "time_course": _time_report}
    parts = maker[spec["analysis"]](spec, result)
    return ScienceReport(
        analysis=spec["analysis"], verdict="clear_difference" if result["clear"] else "no_clear_difference",
        headline=parts["headline"], answer=parts["answer"], how_sure=parts["how_sure"], causal_label=_causal(spec),
        readback=readback(spec), checks=parts["checks"], assumptions=parts["assumptions"],
        method_name=parts["method"], method_why=parts["why"], results=parts["results"], result=result,
        cannot_tell=parts["cannot"], audit=parts["audit"], spec=spec)
