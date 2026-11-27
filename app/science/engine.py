"""Science mode: the fixed analysis engine.

This file is copied word for word into every reproduction script, so it must only import numpy, pandas, scipy and
statsmodels, and nothing from the app. `run(df, spec)` takes the prepared table and the confirmed choices (a plain
dict) and returns every number in the report as plain data. Nothing here is random, and nothing here picks a method
on its own: the choices in `spec` decide which tests run.

Analyses:
  compare_groups  t tests, one-way ANOVA (ordinary or Welch), repeated-measures ANOVA (Greenhouse-Geisser),
                  Mann-Whitney, Kruskal-Wallis, Wilcoxon signed-rank, Friedman, and their follow-up tests
                  (Tukey, Dunnett, Games-Howell, Dunn, paired tests with Bonferroni or Holm corrections).
  contingency     Fisher's exact test, relative risk, odds ratio and risk difference (2x2); chi-square and
                  Cramer's V (bigger tables).
  correlation     Pearson or Spearman, with a 95% interval.
  time_course     two-way repeated-measures ANOVA (group x time, Greenhouse-Geisser), with Welch t tests between
                  groups at each time, Holm-adjusted.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.proportion import confint_proportions_2indep

ALPHA = 0.05
SEED = 0      # Dunnett's test integrates its probabilities by quasi-Monte Carlo; the seed keeps them fixed


class EngineError(ValueError):
    """The table and the choices don't fit together. The message says what to change."""


def _f(x) -> float | None:
    """A plain float for the report, or None when it isn't a finite number."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


# ----------------------------------------------------------------------------- tidy the table into groups

def _labels(s: pd.Series) -> pd.Series:
    """Group and subject labels as text, so 1 and 1.0 are the same label."""
    def one(v):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    return s.map(one).where(s.notna())


def only_rows(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict | None]:
    """Keeps only the rows where one column has one value (such as Day = 12), when that was chosen."""
    col = spec.get("subset_col")
    if not col:
        return df, None
    if col not in df.columns:
        raise EngineError(f"Column '{col}' isn't in your table.")
    keep = _labels(df[col]) == str(spec.get("subset_value"))
    if not keep.any():
        raise EngineError(f"No rows have '{col}' equal to {spec.get('subset_value')!r}.")
    return df[keep], {"column": col, "value": spec.get("subset_value"), "kept": int(keep.sum()), "of": len(df)}


def tidy_groups(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    """One row per value: columns `group`, `value` and `subject`. Also says what was left out and why."""
    groups = list(spec["groups"])
    df, kept = only_rows(df, spec)
    if len(groups) < 2:
        raise EngineError("Choose at least two groups to compare.")
    if len(set(groups)) != len(groups):
        raise EngineError("Each group can only be chosen once.")
    if spec["layout"] == "wide":
        missing = [g for g in groups if g not in df.columns]
        if missing:
            raise EngineError(f"Column '{missing[0]}' isn't in your table.")
        subject = (_labels(df[spec["subject_col"]]) if spec.get("subject_col")
                   else pd.Series([f"row {i + 2}" for i in range(len(df))], index=df.index))
        parts = []
        for g in groups:
            if not pd.api.types.is_numeric_dtype(df[g]):
                raise EngineError(f"Column '{g}' has values that aren't numbers, so it can't be one of the groups.")
            parts.append(pd.DataFrame({"group": g, "value": df[g].astype(float), "subject": subject}))
        long = pd.concat(parts, ignore_index=True)
    else:
        for key in ("group_col", "value_col"):
            if spec.get(key) not in df.columns:
                raise EngineError(f"Column '{spec.get(key)}' isn't in your table.")
        if not pd.api.types.is_numeric_dtype(df[spec["value_col"]]):
            raise EngineError(f"'{spec['value_col']}' has values that aren't numbers.")
        if spec.get("paired") and not spec.get("subject_col"):
            raise EngineError("For repeated measurements, choose the column that says which subject each row is.")
        if spec.get("subject_col") and spec["subject_col"] not in df.columns:
            raise EngineError(f"Column '{spec['subject_col']}' isn't in your table.")
        label = _labels(df[spec["group_col"]])
        long = pd.DataFrame({"group": label, "value": df[spec["value_col"]].astype(float),
                             "subject": (_labels(df[spec["subject_col"]]) if spec.get("subject_col")
                                         else pd.Series([f"row {i + 2}" for i in range(len(df))], index=df.index))})
        unknown = [g for g in groups if g not in set(label.dropna())]
        if unknown:
            raise EngineError(f"'{unknown[0]}' isn't one of the values of '{spec['group_col']}'.")
        long = long[long["group"].isin(groups)].reset_index(drop=True)

    left_out = {"empty_values": {g: int(long.loc[long["group"] == g, "value"].isna().sum()) for g in groups},
                "incomplete_subjects": 0, "rows_kept": kept}
    everyone = long["subject"].dropna().unique()        # including subjects whose every value is empty
    long = long.dropna(subset=["value"])
    if spec.get("paired"):
        if long["subject"].isna().any():
            raise EngineError("Some rows have no subject label. Fill them in, or don't treat the data as repeated.")
        repeats = long.duplicated(subset=["subject", "group"])
        if repeats.any():
            row = long[repeats].iloc[0]
            raise EngineError(f"Subject '{row['subject']}' has more than one value for '{row['group']}'. "
                              "Repeated measurements need exactly one value per subject in each group.")
        counts = long.groupby("subject")["group"].nunique().reindex(everyone, fill_value=0)
        complete = counts[counts == len(groups)].index
        left_out["incomplete_subjects"] = int((counts < len(groups)).sum())
        long = long[long["subject"].isin(complete)]
    long = long.assign(group=pd.Categorical(long["group"], categories=groups, ordered=True))
    long = long.sort_values(["group", "subject"], kind="stable").reset_index(drop=True)
    return long, left_out


def _wide_paired(long: pd.DataFrame, groups: list[str]) -> np.ndarray:
    """Rows are subjects, columns are groups (in order). Only complete subjects reach here."""
    table = long.pivot(index="subject", columns="group", values="value")
    return table[groups].to_numpy(float)


# ----------------------------------------------------------------------------- describe each group

def describe(values: np.ndarray) -> dict:
    n = len(values)
    mean = float(np.mean(values)) if n else float("nan")
    sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    half = stats.t.ppf(0.975, n - 1) * sem if n > 1 else float("nan")
    q1, median, q3 = (np.percentile(values, [25, 50, 75]) if n else (float("nan"),) * 3)
    return {"n": n, "mean": _f(mean), "sd": _f(sd), "sem": _f(sem), "ci_low": _f(mean - half),
            "ci_high": _f(mean + half), "median": _f(median), "q1": _f(q1), "q3": _f(q3),
            "min": _f(np.min(values)) if n else None, "max": _f(np.max(values)) if n else None}


def _normality(values: np.ndarray) -> dict:
    """Shapiro-Wilk. Shown for information only: it never picks the test."""
    if len(values) < 3 or np.ptp(values) == 0:
        return {"test": "Shapiro-Wilk", "p": None, "note": "too few values or all the same"}
    res = stats.shapiro(values)
    return {"test": "Shapiro-Wilk", "W": _f(res.statistic), "p": _f(res.pvalue), "note": ""}


def _far_values(values: np.ndarray) -> int:
    """How many values sit more than 3 interquartile ranges outside the middle half. Counted, never removed."""
    if len(values) < 4:
        return 0
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return int(((values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)).sum()) if iqr > 0 else 0


# ----------------------------------------------------------------------------- follow-up tests

def games_howell(samples: list[np.ndarray], names: list[str], pairs: list[tuple[int, int]]) -> list[dict]:
    """Games-Howell: pairwise Welch comparisons judged against the studentized range for k groups."""
    k = len(samples)
    out = []
    for i, j in pairs:
        a, b = samples[i], samples[j]
        va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
        se = math.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        diff = float(np.mean(b) - np.mean(a))
        q = abs(diff) / (se / math.sqrt(2))
        p = float(stats.studentized_range.sf(q, k, df))
        crit = float(stats.studentized_range.ppf(1 - ALPHA, k, df)) * se / math.sqrt(2)
        out.append({"a": names[i], "b": names[j], "diff": _f(diff), "ci_low": _f(diff - crit),
                    "ci_high": _f(diff + crit), "p_adj": _f(min(p, 1.0))})
    return out


def dunn(samples: list[np.ndarray], names: list[str], pairs: list[tuple[int, int]]) -> list[dict]:
    """Dunn's test on the ranks of all values together, with the tie correction, then Holm-adjusted."""
    pooled = np.concatenate(samples)
    ranks = stats.rankdata(pooled)
    n_total = len(pooled)
    _, tie_counts = np.unique(pooled, return_counts=True)
    tie_term = float(np.sum(tie_counts ** 3 - tie_counts)) / (12 * (n_total - 1))
    sizes = [len(s) for s in samples]
    starts = np.cumsum([0] + sizes[:-1])
    mean_rank = [float(ranks[s:s + n].mean()) for s, n in zip(starts, sizes)]
    raw, rows = [], []
    for i, j in pairs:
        se = math.sqrt((n_total * (n_total + 1) / 12 - tie_term) * (1 / sizes[i] + 1 / sizes[j]))
        z = (mean_rank[j] - mean_rank[i]) / se
        raw.append(2 * stats.norm.sf(abs(z)))
        rows.append({"a": names[i], "b": names[j], "diff": _f(np.median(samples[j]) - np.median(samples[i])),
                     "mean_rank_diff": _f(mean_rank[j] - mean_rank[i]), "z": _f(z), "ci_low": None, "ci_high": None})
    adjusted = multipletests(raw, method="holm")[1]
    for row, p_raw, p in zip(rows, raw, adjusted):
        row["p_raw"], row["p_adj"] = _f(p_raw), _f(p)
    return rows


def paired_followups(wide: np.ndarray, names: list[str], pairs: list[tuple[int, int]], method: str) -> list[dict]:
    """Paired t tests with Bonferroni-adjusted p-values and matching intervals, or Wilcoxon signed-rank with Holm."""
    m = len(pairs)
    rows, raw = [], []
    for i, j in pairs:
        d = wide[:, j] - wide[:, i]
        if method == "means":
            res = stats.ttest_rel(wide[:, j], wide[:, i])
            ci = res.confidence_interval(1 - ALPHA / m)
            rows.append({"a": names[i], "b": names[j], "diff": _f(d.mean()), "ci_low": _f(ci.low),
                         "ci_high": _f(ci.high), "p_raw": _f(res.pvalue), "p_adj": _f(min(res.pvalue * m, 1.0))})
        else:
            p = float(stats.wilcoxon(wide[:, j], wide[:, i]).pvalue) if np.any(d != 0) else 1.0
            raw.append(p)
            rows.append({"a": names[i], "b": names[j], "diff": _f(np.median(d)), "ci_low": None, "ci_high": None,
                         "p_raw": _f(p)})
    if method != "means":
        for row, p in zip(rows, multipletests(raw, method="holm")[1]):
            row["p_adj"] = _f(p)
    return rows


def _pairs(names: list[str], followup: str, control: str | None) -> list[tuple[int, int]]:
    if followup == "none":
        return []
    if followup == "vs_control":
        c = names.index(control)
        return [(c, j) for j in range(len(names)) if j != c]
    return list(itertools.combinations(range(len(names)), 2))


# ----------------------------------------------------------------------------- compare groups

def _sphericity(wide: np.ndarray) -> float:
    """Greenhouse-Geisser epsilon from the double-centred covariance of the repeated measurements."""
    k = wide.shape[1]
    if k < 3:
        return 1.0
    s = np.cov(wide, rowvar=False)
    centre = np.eye(k) - np.ones((k, k)) / k
    sc = centre @ s @ centre
    denom = (k - 1) * float(np.sum(sc * sc))
    return float(np.trace(sc) ** 2 / denom) if denom > 0 else 1.0


def compare_groups(df: pd.DataFrame, spec: dict) -> dict:
    names = list(spec["groups"])
    long, left_out = tidy_groups(df, spec)
    paired, method, k = bool(spec.get("paired")), spec["method"], len(names)
    followup = spec.get("followup", "none") if k > 2 else "none"
    if followup == "vs_control" and spec.get("control") not in names:
        raise EngineError("Choose which group is the control.")
    samples = [long.loc[long["group"] == g, "value"].to_numpy(float) for g in names]
    for g, s in zip(names, samples):
        if len(s) < 2:
            raise EngineError(f"'{g}' has fewer than 2 values left, so it can't be compared."
                              + (" With repeated measurements, only subjects measured in every group are used."
                                 if paired else ""))
    if all(np.ptp(s) == 0 for s in samples) and len({float(s[0]) for s in samples}) == 1:
        raise EngineError("Every value is the same, so there is nothing to compare.")

    groups = [{"name": g, **describe(s), "normality": _normality(s), "far_values": _far_values(s),
               "values": [float(v) for v in s]} for g, s in zip(names, samples)]
    result = {"analysis": "compare_groups", "groups": groups, "left_out": left_out, "tests": [], "pairs": [],
              "effect": {}}
    pairs = _pairs(names, followup, spec.get("control"))

    if paired:
        wide = _wide_paired(long, names)
        result["n_subjects"] = int(wide.shape[0])
        if wide.shape[0] < 3:
            raise EngineError("Fewer than 3 subjects were measured in every group, so this can't be tested.")
        # Paired tests assume the within-subject differences are bell-shaped, not each group's raw values.
        within = (wide[:, 1] - wide[:, 0] if k == 2
                  else (wide - wide.mean(axis=1, keepdims=True) - wide.mean(axis=0) + wide.mean()).ravel())
        result["paired_normality"] = _normality(within)
        if k == 2:
            d = wide[:, 1] - wide[:, 0]
            if method == "means":
                res = stats.ttest_rel(wide[:, 1], wide[:, 0])
                ci = res.confidence_interval(0.95)
                result["tests"].append({"key": "paired_t", "name": "Paired t test", "statistic_name": "t",
                                        "statistic": _f(res.statistic), "df": f"{int(res.df)}", "p": _f(res.pvalue)})
                result["effect"] = {"kind": "mean_diff", "a": names[0], "b": names[1], "diff": _f(d.mean()),
                                    "ci_low": _f(ci.low), "ci_high": _f(ci.high),
                                    "sd_diff": _f(np.std(d, ddof=1))}
            else:
                res = stats.wilcoxon(wide[:, 1], wide[:, 0]) if np.any(d != 0) else None
                result["tests"].append({"key": "wilcoxon", "name": "Wilcoxon signed-rank test", "statistic_name": "W",
                                        "statistic": _f(res.statistic) if res else None, "df": "",
                                        "p": _f(res.pvalue) if res else 1.0})
                result["effect"] = {"kind": "median_diff", "a": names[0], "b": names[1], "diff": _f(np.median(d))}
        elif method == "means":
            n = wide.shape[0]
            grand = wide.mean()
            ss_cond = n * float(np.sum((wide.mean(axis=0) - grand) ** 2))
            ss_subj = k * float(np.sum((wide.mean(axis=1) - grand) ** 2))
            ss_total = float(np.sum((wide - grand) ** 2))
            ss_err = ss_total - ss_cond - ss_subj
            df1, df2 = k - 1, (k - 1) * (n - 1)
            f = (ss_cond / df1) / (ss_err / df2) if ss_err > 0 else float("inf")
            eps = _sphericity(wide)
            p_gg = float(stats.f.sf(f, df1 * eps, df2 * eps))
            result["tests"].append({"key": "rm_anova", "name": "Repeated-measures one-way ANOVA (Greenhouse-Geisser)",
                                    "statistic_name": "F", "statistic": _f(f),
                                    "df": f"{df1 * eps:.2f}, {df2 * eps:.2f}", "p": _f(p_gg),
                                    "p_uncorrected": _f(stats.f.sf(f, df1, df2)), "epsilon": _f(eps)})
            result["effect"] = {"kind": "eta_squared_partial", "value": _f(ss_cond / (ss_cond + ss_err))}
        else:
            res = stats.friedmanchisquare(*[wide[:, j] for j in range(k)])
            result["tests"].append({"key": "friedman", "name": "Friedman test", "statistic_name": "χ²",
                                    "statistic": _f(res.statistic), "df": f"{k - 1}", "p": _f(res.pvalue)})
            result["effect"] = {"kind": "kendall_w", "value": _f(res.statistic / (wide.shape[0] * (k - 1)))}
        if pairs:
            result["pairs"] = paired_followups(wide, names, pairs, method)
            result["pair_method"] = ("paired t tests, Bonferroni-adjusted" if method == "means"
                                     else "Wilcoxon signed-rank tests, Holm-adjusted")
    else:
        equal = bool(spec.get("equal_spread"))
        if k == 2:
            a, b = samples
            if method == "means":
                res = stats.ttest_ind(b, a, equal_var=equal)
                ci = res.confidence_interval(0.95)
                result["tests"].append({"key": "student_t" if equal else "welch_t",
                                        "name": "Unpaired t test" if equal else "Welch's t test",
                                        "statistic_name": "t", "statistic": _f(res.statistic),
                                        "df": f"{res.df:.0f}" if equal else f"{res.df:.2f}", "p": _f(res.pvalue)})
                result["effect"] = {"kind": "mean_diff", "a": names[0], "b": names[1],
                                    "diff": _f(b.mean() - a.mean()), "ci_low": _f(ci.low), "ci_high": _f(ci.high)}
            else:
                res = stats.mannwhitneyu(b, a, alternative="two-sided")
                result["tests"].append({"key": "mann_whitney", "name": "Mann-Whitney test", "statistic_name": "U",
                                        "statistic": _f(res.statistic), "df": "", "p": _f(res.pvalue)})
                result["effect"] = {"kind": "median_diff", "a": names[0], "b": names[1],
                                    "diff": _f(np.median(b) - np.median(a)),
                                    "superiority": _f(res.statistic / (len(a) * len(b)))}
        elif method == "means":
            pooled = np.concatenate(samples)
            ss_between = sum(len(s) * (s.mean() - pooled.mean()) ** 2 for s in samples)
            ss_total = float(np.sum((pooled - pooled.mean()) ** 2))
            if equal:
                res = stats.f_oneway(*samples)
                df1, df2 = k - 1, len(pooled) - k
                ms_within = (ss_total - ss_between) / df2
                result["tests"].append({"key": "anova", "name": "Ordinary one-way ANOVA", "statistic_name": "F",
                                        "statistic": _f(res.statistic), "df": f"{df1}, {df2}", "p": _f(res.pvalue)})
                omega = (ss_between - df1 * ms_within) / (ss_total + ms_within)
                result["effect"] = {"kind": "eta_squared", "value": _f(ss_between / ss_total),
                                    "omega_squared": _f(max(omega, 0.0))}
            else:
                res = anova_oneway(samples, use_var="unequal")
                result["tests"].append({"key": "welch_anova", "name": "Welch's ANOVA", "statistic_name": "F",
                                        "statistic": _f(res.statistic), "df": f"{res.df[0]:.0f}, {res.df[1]:.2f}",
                                        "p": _f(res.pvalue)})
                result["effect"] = {"kind": "eta_squared", "value": _f(ss_between / ss_total)}
            if pairs and equal and followup == "all_pairs":
                tk = stats.tukey_hsd(*samples)
                ci = tk.confidence_interval(1 - ALPHA)
                # scipy reports group i minus group j; the report shows b minus a.
                result["pairs"] = [{"a": names[i], "b": names[j], "diff": _f(-tk.statistic[i, j]),
                                    "ci_low": _f(-ci.high[i, j]), "ci_high": _f(-ci.low[i, j]),
                                    "p_adj": _f(tk.pvalue[i, j])} for i, j in pairs]
                result["pair_method"] = "Tukey's test"
            elif pairs and equal:
                c = names.index(spec["control"])
                others = [j for j in range(k) if j != c]
                dn = stats.dunnett(*[samples[j] for j in others], control=samples[c], rng=np.random.default_rng(SEED))
                ci = dn.confidence_interval(1 - ALPHA)
                result["pairs"] = [{"a": names[c], "b": names[j], "diff": _f(samples[j].mean() - samples[c].mean()),
                                    "ci_low": _f(ci.low[n]), "ci_high": _f(ci.high[n]), "p_adj": _f(dn.pvalue[n])}
                                   for n, j in enumerate(others)]
                result["pair_method"] = "Dunnett's test"
            elif pairs:
                result["pairs"] = games_howell(samples, names, pairs)
                result["pair_method"] = "Games-Howell test" + (
                    " (adjusted for every pair, so slightly cautious when only comparing with the control)"
                    if followup == "vs_control" else "")
        else:
            res = stats.kruskal(*samples)
            n_total = sum(len(s) for s in samples)
            result["tests"].append({"key": "kruskal", "name": "Kruskal-Wallis test", "statistic_name": "H",
                                    "statistic": _f(res.statistic), "df": f"{k - 1}", "p": _f(res.pvalue)})
            result["effect"] = {"kind": "epsilon_squared", "value": _f(res.statistic / (n_total - 1))}
            if pairs:
                result["pairs"] = dunn(samples, names, pairs)
                result["pair_method"] = "Dunn's test, Holm-adjusted"
        if method == "means":
            sds = [s.std(ddof=1) for s in samples]
            result["spread_ratio"] = _f(max(sds) / min(sds)) if min(sds) > 0 else None
    for row in result["pairs"]:
        row["clear"] = bool(row["p_adj"] is not None and row["p_adj"] < ALPHA)
    main = result["tests"][0]
    result["main_p"] = main["p"]
    result["clear"] = bool(main["p"] is not None and main["p"] < ALPHA)
    return result


# ----------------------------------------------------------------------------- contingency tables

def count_table(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, int]:
    """The table of counts, rows and columns in the chosen order, and how many rows were left out."""
    rows, cols = list(spec["rows"]), list(spec["cols"])
    if len(rows) < 2 or len(cols) < 2:
        raise EngineError("A table of counts needs at least two rows and two columns.")
    if spec["layout"] == "counts":
        if spec.get("label_col") not in df.columns:
            raise EngineError(f"Column '{spec.get('label_col')}' isn't in your table.")
        labels = _labels(df[spec["label_col"]])
        if labels.duplicated().any():
            raise EngineError(f"'{labels[labels.duplicated()].iloc[0]}' appears twice in '{spec['label_col']}'.")
        table = pd.DataFrame(index=rows, columns=cols, dtype=float)
        for r in rows:
            if r not in set(labels.dropna()):
                raise EngineError(f"'{r}' isn't one of the values of '{spec['label_col']}'.")
            for c in cols:
                if c not in df.columns:
                    raise EngineError(f"Column '{c}' isn't in your table.")
                v = df.loc[labels == r, c].iloc[0]
                if pd.isna(v) or not isinstance(v, (int, float, np.integer, np.floating)) or v < 0 or v != int(v):
                    raise EngineError(f"The count for '{r}' in '{c}' must be a whole number, 0 or more.")
                table.loc[r, c] = int(v)
        return table.astype(int), 0
    for key in ("row_col", "col_col"):
        if spec.get(key) not in df.columns:
            raise EngineError(f"Column '{spec.get(key)}' isn't in your table.")
    if spec["row_col"] == spec["col_col"]:
        raise EngineError("Choose two different columns.")
    a, b = _labels(df[spec["row_col"]]), _labels(df[spec["col_col"]])
    for value, col, column in [(r, a, spec["row_col"]) for r in rows] + [(c, b, spec["col_col"]) for c in cols]:
        if value not in set(col.dropna()):
            raise EngineError(f"'{value}' isn't one of the values of '{column}'.")
    keep = a.isin(rows) & b.isin(cols)
    table = pd.crosstab(a[keep], b[keep]).reindex(index=rows, columns=cols, fill_value=0)
    return table.astype(int), int((~keep).sum())


def contingency(df: pd.DataFrame, spec: dict) -> dict:
    table, left_out = count_table(df, spec)
    counts = table.to_numpy()
    if counts.sum() == 0:
        raise EngineError("Every count is 0.")
    if (counts.sum(axis=1) == 0).any() or (counts.sum(axis=0) == 0).any():
        raise EngineError("A whole row or column has no counts. Leave it out of the comparison.")
    result = {"analysis": "contingency", "rows": list(table.index), "cols": list(table.columns),
              "counts": counts.tolist(), "left_out": left_out, "tests": [], "effect": {}}
    row_tot = counts.sum(axis=1, keepdims=True)
    result["row_percent"] = (counts / row_tot).tolist()
    expected = stats.contingency.expected_freq(counts)
    result["expected_min"] = _f(expected.min())
    result["expected_small_share"] = _f(float((expected < 5).mean()))
    if counts.shape == (2, 2):
        event = spec.get("event") or result["cols"][0]
        if event not in result["cols"]:
            raise EngineError("Choose which column counts as the outcome happening.")
        e = result["cols"].index(event)
        other = 1 - e
        (a, b), (c, d) = (counts[0, e], counts[0, other]), (counts[1, e], counts[1, other])
        fisher = stats.fisher_exact([[a, b], [c, d]])
        result["tests"].append({"key": "fisher", "name": "Fisher's exact test", "statistic_name": "",
                                "statistic": None, "df": "", "p": _f(fisher.pvalue)})
        rr = stats.contingency.relative_risk(int(a), int(a + b), int(c), int(c + d))
        rr_ci = rr.confidence_interval(0.95)
        orr = stats.contingency.odds_ratio([[a, b], [c, d]], kind="sample")
        or_ci = orr.confidence_interval(0.95)
        lo, hi = confint_proportions_2indep(a, a + b, c, c + d, method="newcomb", compare="diff")
        result["event"] = event
        result["effect"] = {"kind": "two_by_two", "risk_a": _f(a / (a + b)), "risk_b": _f(c / (c + d)),
                            "relative_risk": _f(rr.relative_risk), "rr_low": _f(rr_ci.low), "rr_high": _f(rr_ci.high),
                            "odds_ratio": _f(orr.statistic), "or_low": _f(or_ci.low), "or_high": _f(or_ci.high),
                            "risk_diff": _f(a / (a + b) - c / (c + d)), "rd_low": _f(lo), "rd_high": _f(hi)}
    else:
        chi = stats.chi2_contingency(counts, correction=False)
        n = counts.sum()
        v = math.sqrt(chi.statistic / (n * (min(counts.shape) - 1)))
        result["tests"].append({"key": "chi_square", "name": "Chi-square test", "statistic_name": "χ²",
                                "statistic": _f(chi.statistic), "df": f"{chi.dof}", "p": _f(chi.pvalue)})
        result["effect"] = {"kind": "cramers_v", "value": _f(v)}
    result["main_p"] = result["tests"][0]["p"]
    result["clear"] = bool(result["main_p"] is not None and result["main_p"] < ALPHA)
    return result


# ----------------------------------------------------------------------------- correlation

def correlation(df: pd.DataFrame, spec: dict) -> dict:
    x_col, y_col = spec["x_col"], spec["y_col"]
    for col in (x_col, y_col):
        if col not in df.columns:
            raise EngineError(f"Column '{col}' isn't in your table.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise EngineError(f"'{col}' has values that aren't numbers.")
    if x_col == y_col:
        raise EngineError("Choose two different columns.")
    both = df[[x_col, y_col]].astype(float)
    left_out = int(both.isna().any(axis=1).sum())
    both = both.dropna()
    x, y = both[x_col].to_numpy(), both[y_col].to_numpy()
    n = len(x)
    if n < 4:
        raise EngineError("Fewer than 4 rows have both values, so a correlation can't be estimated.")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise EngineError("One of the columns has the same value in every row.")
    result = {"analysis": "correlation", "n": n, "left_out": left_out, "x": [float(v) for v in x],
              "y": [float(v) for v in y], "tests": []}
    if spec["method"] == "pearson":
        res = stats.pearsonr(x, y)
        ci = res.confidence_interval(0.95)
        fit = stats.linregress(x, y)
        result["tests"].append({"key": "pearson", "name": "Pearson correlation", "statistic_name": "r",
                                "statistic": _f(res.statistic), "df": f"{n - 2}", "p": _f(res.pvalue)})
        result["effect"] = {"kind": "r", "value": _f(res.statistic), "ci_low": _f(ci.low), "ci_high": _f(ci.high),
                            "r_squared": _f(res.statistic ** 2), "slope": _f(fit.slope),
                            "intercept": _f(fit.intercept)}
    else:
        res = stats.spearmanr(x, y)
        rho = float(res.statistic)
        # Fisher z interval with the Bonett-Wright (2000) standard error for Spearman's rho.
        se = math.sqrt((1 + rho ** 2 / 2) / (n - 3))
        z = math.atanh(max(min(rho, 0.999999), -0.999999))
        crit = stats.norm.ppf(0.975)
        result["tests"].append({"key": "spearman", "name": "Spearman correlation", "statistic_name": "rs",
                                "statistic": _f(rho), "df": "", "p": _f(res.pvalue)})
        result["effect"] = {"kind": "rho", "value": _f(rho), "ci_low": _f(math.tanh(z - crit * se)),
                            "ci_high": _f(math.tanh(z + crit * se))}
    result["main_p"] = result["tests"][0]["p"]
    result["clear"] = bool(result["main_p"] is not None and result["main_p"] < ALPHA)
    return result


# ----------------------------------------------------------------------------- time course (two-way repeated measures)

def _time_key(label: str):
    """Days sort as numbers (2 before 10); other labels keep their order."""
    try:
        return (0, float(label), "")
    except ValueError:
        return (1, 0.0, label)


def time_course(df: pd.DataFrame, spec: dict) -> dict:
    """Two-way repeated-measures ANOVA: groups of different subjects, each subject measured at every time point.

    Sums of squares are the classic split-plot ones (with unequal group sizes, the time effect weights each group by
    its size). The time and group-by-time tests use the Greenhouse-Geisser correction, with epsilon from the
    covariance of the time points within groups, pooled across groups. Follow-ups compare groups at each time with
    Welch t tests, Holm-adjusted across every comparison made."""
    groups, times = list(spec["groups"]), [str(t) for t in spec["times"]]
    if len(groups) < 2:
        raise EngineError("Choose at least two groups.")
    if len(times) < 2:
        raise EngineError("Choose at least two time points.")
    for key in ("group_col", "time_col", "value_col", "subject_col"):
        if spec.get(key) not in df.columns:
            raise EngineError(f"Column '{spec.get(key)}' isn't in your table.")
    if len({spec["group_col"], spec["time_col"], spec["value_col"], spec["subject_col"]}) < 4:
        raise EngineError("The group, time, measurement and subject columns must all be different.")
    if not pd.api.types.is_numeric_dtype(df[spec["value_col"]]):
        raise EngineError(f"'{spec['value_col']}' has values that aren't numbers.")
    long = pd.DataFrame({"group": _labels(df[spec["group_col"]]), "time": _labels(df[spec["time_col"]]),
                         "subject": _labels(df[spec["subject_col"]]), "value": df[spec["value_col"]].astype(float)})
    for value, column, name in [(g, "group", spec["group_col"]) for g in groups] + \
                               [(t, "time", spec["time_col"]) for t in times]:
        if value not in set(long[column].dropna()):
            raise EngineError(f"'{value}' isn't one of the values of '{name}'.")
    long = long[long["group"].isin(groups) & long["time"].isin(times)]
    if long["subject"].isna().any():
        raise EngineError(f"Some rows have no '{spec['subject_col']}', so they can't be followed over time.")
    homes = long.groupby("subject")["group"].nunique()
    if (homes > 1).any():
        raise EngineError(f"'{homes[homes > 1].index[0]}' appears in more than one group. Each subject must belong "
                          "to one group (if subjects are labelled 1, 2, 3 in every group, the labels need the group "
                          "name too).")
    repeats = long.duplicated(subset=["subject", "time"])
    if repeats.any():
        row = long[repeats].iloc[0]
        raise EngineError(f"'{row['subject']}' has more than one value at {spec['time_col']} {row['time']}.")
    empty = int(long["value"].isna().sum())
    # Pivot before dropping empty values, so a subject with no values at all still counts as incomplete.
    wide = long.pivot(index="subject", columns="time", values="value").reindex(columns=times)
    complete = wide.dropna()
    home = long.drop_duplicates("subject").set_index("subject")["group"]
    long = long.dropna(subset=["value"])
    left_out = {"empty_values": empty, "incomplete_subjects": int(len(wide) - len(complete)),
                "incomplete_by_group": {g: int(((home.reindex(wide.index) == g) & wide.isna().any(axis=1)).sum())
                                        for g in groups}}
    blocks = [complete[home.reindex(complete.index) == g].to_numpy(float) for g in groups]
    for g, y in zip(groups, blocks):
        if len(y) < 2:
            raise EngineError(f"'{g}' has fewer than 2 subjects measured at every chosen time point, so it can't be "
                              "compared. Only subjects measured at every chosen time are used.")
    a, b = len(groups), len(times)
    n = np.array([len(y) for y in blocks])
    ns = int(n.sum())
    allv = np.vstack(blocks)
    grand = allv.mean()
    g_mean = np.array([y.mean() for y in blocks])                          # group means over subjects and times
    t_mean = allv.mean(axis=0)                                             # time means over all subjects
    cell = np.array([y.mean(axis=0) for y in blocks])                      # group x time means
    ss_group = b * float(np.sum(n * (g_mean - grand) ** 2))
    ss_subj = b * float(sum(np.sum((y.mean(axis=1) - y.mean()) ** 2) for y in blocks))
    ss_time = ns * float(np.sum((t_mean - grand) ** 2))
    ss_gt = float(np.sum(n[:, None] * (cell - g_mean[:, None] - t_mean[None, :] + grand) ** 2))
    resid = np.vstack([y - y.mean(axis=1, keepdims=True) - c[None, :] + y.mean() for y, c in zip(blocks, cell)])
    ss_err = float(np.sum(resid ** 2))
    df_group, df_subj = a - 1, ns - a
    df_time, df_gt, df_err = b - 1, (a - 1) * (b - 1), (ns - a) * (b - 1)
    if df_subj < 1 or ss_err <= 0 or ss_subj <= 0:
        raise EngineError("There's no variation left to test against: too few subjects, or values that are all the same.")
    # Greenhouse-Geisser epsilon from the pooled within-group covariance of the time points.
    pooled = sum((len(y) - 1) * np.cov(y, rowvar=False) for y in blocks) / df_subj
    centre = np.eye(b) - np.ones((b, b)) / b
    sc = centre @ pooled @ centre
    eps = float(np.trace(sc) ** 2 / ((b - 1) * np.sum(sc * sc))) if b > 2 and np.sum(sc * sc) > 0 else 1.0
    ms_err = ss_err / df_err

    def within(name, key, ss, df):
        f = (ss / df) / ms_err
        return {"key": key, "name": name, "statistic_name": "F", "statistic": _f(f),
                "df": f"{df * eps:.2f}, {df_err * eps:.2f}", "p": _f(stats.f.sf(f, df * eps, df_err * eps)),
                "p_uncorrected": _f(stats.f.sf(f, df, df_err)), "partial_eta_squared": _f(ss / (ss + ss_err))}

    f_group = (ss_group / df_group) / (ss_subj / df_subj)
    tests = [within("Group × time", "interaction", ss_gt, df_gt),
             {"key": "group", "name": "Group", "statistic_name": "F", "statistic": _f(f_group),
              "df": f"{df_group}, {df_subj}", "p": _f(stats.f.sf(f_group, df_group, df_subj)),
              "partial_eta_squared": _f(ss_group / (ss_group + ss_subj))},
             within("Time", "time", ss_time, df_time)]
    table = [{"source": "Group", "ss": _f(ss_group), "df": df_group},
             {"source": "Subjects within groups", "ss": _f(ss_subj), "df": df_subj},
             {"source": "Time", "ss": _f(ss_time), "df": df_time},
             {"source": "Group × time", "ss": _f(ss_gt), "df": df_gt},
             {"source": "Residual", "ss": _f(ss_err), "df": df_err}]

    # Follow-ups: at each time, compare groups with Welch t tests, Holm-adjusted across every comparison.
    pairs = []
    for t in range(b):
        for i, j in _pairs(groups, spec.get("followup", "none"), spec.get("control")):
            x, y = blocks[i][:, t], blocks[j][:, t]
            res = stats.ttest_ind(y, x, equal_var=False)
            p = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0
            # Rounded far below anything shown, so equal means give exactly 0 rather than 4e-16.
            pairs.append({"time": times[t], "a": groups[i], "b": groups[j], "diff": _f(round(float(y.mean() - x.mean()), 12)),
                          "ci_low": None, "ci_high": None, "p_raw": _f(p)})
    if pairs:
        for row, p in zip(pairs, multipletests([r["p_raw"] for r in pairs], method="holm")[1]):
            row["p_adj"] = _f(p)
            row["clear"] = bool(p < ALPHA)
    means = [{"group": g, "n": int(len(y)),
              "points": [{"time": times[t], "mean": _f(y[:, t].mean()), "sd": _f(np.std(y[:, t], ddof=1)),
                          "sem": _f(np.std(y[:, t], ddof=1) / math.sqrt(len(y))),
                          "ci_low": _f(y[:, t].mean() - stats.t.ppf(0.975, len(y) - 1) * np.std(y[:, t], ddof=1) / math.sqrt(len(y))),
                          "ci_high": _f(y[:, t].mean() + stats.t.ppf(0.975, len(y) - 1) * np.std(y[:, t], ddof=1) / math.sqrt(len(y)))}
                         for t in range(b)],
              "subjects": [[float(v) for v in row] for row in y]} for g, y in zip(groups, blocks)]
    sds = [np.std(y, axis=0, ddof=1) for y in blocks]
    spread = np.array(sds)
    result = {"analysis": "time_course", "groups": groups, "times": times, "tests": tests, "anova_table": table,
              "epsilon": _f(eps), "means": means, "pairs": pairs, "left_out": left_out, "n_subjects": ns,
              "residual_normality": _normality(resid.ravel()),
              "spread_ratio": _f(float(np.max(spread / spread.min(axis=0)))) if spread.min() > 0 else None,
              "pair_method": "Welch t tests at each time, Holm-adjusted across all comparisons" if pairs else ""}
    result["main_p"] = tests[0]["p"]
    result["clear"] = bool(tests[0]["p"] is not None and tests[0]["p"] < ALPHA)
    return result


def run(df: pd.DataFrame, spec: dict) -> dict:
    """Every number in the report, from the prepared table and the confirmed choices."""
    kinds = {"compare_groups": compare_groups, "contingency": contingency, "correlation": correlation,
             "time_course": time_course}
    if spec.get("analysis") not in kinds:
        raise EngineError("Unknown analysis.")
    return kinds[spec["analysis"]](df, spec)
