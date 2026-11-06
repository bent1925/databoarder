"""Builds a standalone Python script that reproduces EVERY number in a report.

It uses the same recipes as the app (the tests run it and compare the output).
"""
from __future__ import annotations

from .schemas import ABTestSpec

SEED = 0
N_PERM = 2000


def build_script(spec: ABTestSpec, kind: str, csv_name: str, mode: str, header: str = "") -> str:
    g, ctl, trt, y = spec.group_col, spec.control, spec.treatment, spec.outcome_col
    out = [f'''{header}import numpy as np
import pandas as pd
from scipy import stats
''']

    if kind == "binary":
        out.append('''from statsmodels.stats.proportion import confint_proportions_2indep, test_proportions_2indep
''')
    if mode == "research":
        out.append('''import re
from statsmodels.stats.multitest import multipletests
''')

    out.append(f'''
df = pd.read_csv({csv_name!r})
df = df[df[{g!r}].astype(str).isin([{ctl!r}, {trt!r}])]
df = df.dropna(subset=[{y!r}])
is_c = df[{g!r}].astype(str) == {ctl!r}
c = df.loc[is_c, {y!r}].to_numpy(float)
t = df.loc[~is_c, {y!r}].to_numpy(float)

''')

    if kind == "binary":
        out.append('''def effect(c, t):
    xc, nc, xt, nt = int(c.sum()), len(c), int(t.sum()), len(t)
    low, high = confint_proportions_2indep(xt, nt, xc, nc, method="newcomb", compare="diff")
    p = test_proportions_2indep(xt, nt, xc, nc, method="score", compare="diff", correction=False).pvalue
    return xt / nt - xc / nc, float(low), float(high), float(p) if np.isfinite(p) else 1.0
''')
    else:
        out.append('''def effect(c, t):
    res = stats.ttest_ind(t, c, equal_var=False)   # Welch
    ci = res.confidence_interval(0.95)
    p = float(res.pvalue)
    return float(t.mean() - c.mean()), float(ci.low), float(ci.high), p if np.isfinite(p) else 1.0
''')

    out.append('''
diff, low, high, p = effect(c, t)
print(f"control_mean={c.mean():.4f} treatment_mean={t.mean():.4f}")
print(f"diff={diff:.4f} low={low:.4f} high={high:.4f} p={p:.5f}")
''')

    if mode == "right_answer":
        out.append('''
# --- cross-checks ---
''')
        if kind == "binary":
            out.append('''xc, nc, xt, nt = int(c.sum()), len(c), int(t.sum()), len(t)
print(f"fisher_p={stats.fisher_exact([[xt, nt - xt], [xc, nc - xc]])[1]:.5f}")
''')
        else:
            out.append('''print(f"mannwhitney_p={stats.mannwhitneyu(t, c, alternative='two-sided').pvalue:.5f}")
''')
        out.append(f'''
rng = np.random.default_rng({SEED})            # fixed seed: same data -> same answer
pooled, obs, hits = np.concatenate([c, t]), abs(t.mean() - c.mean()), 0
for _ in range({N_PERM}):
    rng.shuffle(pooled)
    if abs(pooled[:len(t)].mean() - pooled[len(t):].mean()) >= obs - 1e-12:
        hits += 1
print(f"shuffle_p={{(hits + 1) / {N_PERM + 1}:.5f}}")
''')
        if kind == "continuous":
            out.append('''
lo_q, hi_q = np.percentile(np.concatenate([c, t]), [1, 99])
c2, t2 = c[(c >= lo_q) & (c <= hi_q)], t[(t >= lo_q) & (t <= hi_q)]
print(f"trimmed_p={effect(c2, t2)[3]:.5f}")
''')

    if mode == "research":
        out.append(f'''
# --- slice scan (same difference, inside each slice) ---
ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|^user$|^customer$|^account$)", re.I)
def idlike(col):
    return bool(ID_NAME.search(col)) or (pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() >= 0.9 * len(df))

rows = []
for col in df.columns:
    if col in ({g!r}, {y!r}) or idlike(col):
        continue
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) or not (2 <= s.nunique() <= 8):
        continue   # slice only on category columns; numeric columns are usually results
    for level in sorted(s.dropna().astype(str).unique()):
        m = s.astype(str) == level
        cs = df.loc[m & is_c, {y!r}].to_numpy(float)
        ts = df.loc[m & ~is_c, {y!r}].to_numpy(float)
        if len(cs) >= 50 and len(ts) >= 50:
            d, lo, hi, p_raw = effect(cs, ts)
            rows.append((col, level, d, p_raw))
if rows:
    p_adj = multipletests([r[3] for r in rows], method="holm")[1]
    for (col, level, d, p_raw), pa in sorted(zip(rows, p_adj), key=lambda x: x[0][3]):
        print(f"slice {{col}}={{level}} diff={{d:.4f}} p_raw={{p_raw:.5f}} p_adj={{pa:.5f}}")
''')
    return "".join(out)


# --------------------------------------------------------------------- before / after

def build_ba_script(spec, csv_name: str, mode: str, header: str = "") -> str:
    """Before/after, and before/after with a comparison group (the 'gap' version)."""
    d, y, change, agg = spec.date_col, spec.outcome_col, spec.change_date, spec.aggregate
    agg_line = "g.mean()" if agg == "average" else "g.sum(min_count=1)"
    did = spec.analysis_type == "diff_in_diff"
    series_fn = "gap_series" if did else "daily_series"
    out = [f'''{header}import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
''']
    if mode == "research":
        out.append('''import re
from statsmodels.stats.multitest import multipletests
''')
    gap_fn = ""
    if did:
        gap_fn = f'''def gap_series(frame):
    """Treated minus comparison, per calendar day (NaN if either group has no value)."""
    g = frame[{spec.group_col!r}].astype(str)
    return (daily_series(frame[g == {spec.treated!r}]) - daily_series(frame[g == {spec.comparison!r}])).sort_index()

'''
    out.append(f'''
df = pd.read_csv({csv_name!r})
change = pd.Timestamp({change!r})          # the first day of "after" (you chose it)
{"df = df[df[" + repr(spec.group_col) + "].astype(str).isin([" + repr(spec.treated) + ", " + repr(spec.comparison) + "])]" if did else ""}

def parse_dates(s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_localize(None).dt.normalize()

def daily_series(frame):
    """One number per calendar day: that day's {agg} of the result."""
    f = pd.DataFrame({{"d": parse_dates(frame[{d!r}]), "y": pd.to_numeric(frame[{y!r}], errors="coerce")}}).dropna(subset=["d"])
    g = f.groupby("d")["y"]
    return {agg_line}.sort_index()

{gap_fn}def make_blocks(daily):
    """Whole 7-day weeks counted outward from the change date. Weeks with a missing day are skipped."""
    cal = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    one = pd.Timedelta(days=1)
    after, k = [], 0
    while change + pd.Timedelta(days=7 * k + 6) <= cal.index[-1]:
        v = cal.loc[change + pd.Timedelta(days=7 * k): change + pd.Timedelta(days=7 * k + 6)]
        if v.notna().all():
            after.append(float(v.mean()))
        k += 1
    before, k = [], 1
    while change - pd.Timedelta(days=7 * k) >= cal.index[0]:
        v = cal.loc[change - pd.Timedelta(days=7 * k): change - pd.Timedelta(days=7 * k) + 6 * one]
        if v.notna().all():
            before.append(float(v.mean()))
        k += 1
    return np.array(before[::-1]), np.array(after), cal

def effect(c, t):
    res = stats.ttest_ind(t, c, equal_var=False)   # Welch
    ci = res.confidence_interval(0.95)
    p = float(res.pvalue)
    return float(t.mean() - c.mean()), float(ci.low), float(ci.high), p if np.isfinite(p) else 1.0

before, after, cal = make_blocks({series_fn}(df))
diff, low, high, p = effect(before, after)
print(f"weeks_before={{len(before)}} weeks_after={{len(after)}}")
print(f"before_mean={{before.mean():.4f}} after_mean={{after.mean():.4f}}")   # {"weekly gap (treated - comparison)" if did else "typical day"}
print(f"diff={{diff:.4f}} low={{low:.4f}} high={{high:.4f}} p={{p:.5f}}")
''')

    if mode == "right_answer":
        out.append('''
# --- cross-checks ---
if len(before) >= 6:                      # fake date: only weeks before the real date
    h = len(before) // 2
    print(f"placebo_p={effect(before[:h], before[h:])[3]:.5f}")

n_b, n_a = len(before), len(after)
y = np.concatenate([before, after])       # allow a steady trend, then look for a jump at the date
X = np.column_stack([np.ones(n_b + n_a), np.r_[np.zeros(n_b), np.ones(n_a)], np.r_[np.arange(-n_b, 0), np.arange(0, n_a)]])
fit = sm.OLS(y, X).fit()
print(f"its_diff={fit.params[1]:.4f} its_p={fit.pvalues[1]:.5f}")

if n_b > 4 and n_a > 4:                   # only the 4 weeks nearest the date
    print(f"window_p={effect(before[-4:], after[:4])[3]:.5f}")

obs = cal.dropna()                        # every day, weekday indicators, Newey-West errors
X = np.column_stack([np.ones(len(obs)), (obs.index >= change).astype(float),
                     pd.get_dummies(obs.index.weekday, drop_first=True).to_numpy(float)])
if len(obs) >= 28 and np.linalg.matrix_rank(X) == X.shape[1]:
    fit = sm.OLS(obs.to_numpy(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": 7, "use_correction": True}, use_t=True)
    print(f"daily_diff={fit.params[1]:.4f} daily_p={fit.pvalues[1]:.5f}")
''')

    if mode == "research":
        skip = f", {spec.group_col!r}" if did else ""
        out.append(f'''
# --- slice scan (same comparison, inside each slice) ---
ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|^user$|^customer$|^account$)", re.I)
def idlike(col):
    return bool(ID_NAME.search(col)) or (pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() >= 0.9 * len(df))

rows = []
for col in df.columns:
    if col in ({d!r}, {y!r}{skip}) or idlike(col):
        continue
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) or not (2 <= s.nunique() <= 8):
        continue   # slice only on category columns; numeric columns are usually results
    for level in sorted(s.dropna().astype(str).unique()):
        part = df[s.astype(str) == level]
        if part[{y!r}].notna().sum() == 0:
            continue
        daily = {series_fn}(part)
        if daily.notna().sum() == 0 or daily.index.min() >= change or daily.index.max() < change:
            continue
        b, a, _ = make_blocks(daily)
        if len(b) >= 4 and len(a) >= 4:
            dd, lo, hi, p_raw = effect(b, a)
            rows.append((col, level, dd, p_raw))
if rows:
    p_adj = multipletests([r[3] for r in rows], method="holm")[1]
    for (col, level, dd, p_raw), pa in sorted(zip(rows, p_adj), key=lambda x: x[0][3]):
        print(f"slice {{col}}={{level}} diff={{dd:.4f}} p_raw={{p_raw:.5f}} p_adj={{pa:.5f}}")
''')
    return "".join(out)


# ------------------------------------------------------------------------ observational

_OBS_BODY = '''
group, control, treatment, outcome = __GROUP__, __CONTROL__, __TREATMENT__, __OUTCOME__
covs = __COVS__                               # the columns people were made comparable on

df = pd.read_csv(__CSV__)
df = df[df[group].astype(str).isin([control, treatment])].dropna(subset=[outcome])
df = df.dropna(subset=covs)

def design(frame, cols):
    """Numbers are standardised; categories become 0/1 columns (first level is the reference)."""
    parts = []
    for c in cols:
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s):
            sd = s.std(ddof=0)
            if sd > 0:
                parts.append(((s - s.mean()) / sd).rename(c))
        else:
            dm = pd.get_dummies(s.astype(str), prefix=c, drop_first=True, dtype=float)
            parts.append(dm.loc[:, dm.std(ddof=0) > 0])
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)

def adjusted(frame, cols):
    """Treatment minus control, comparing like with like (OLS, HC3 robust errors)."""
    t = (frame[group].astype(str) == treatment).to_numpy(float)
    y = frame[outcome].to_numpy(float)
    Xc = design(frame, cols).to_numpy(float)
    X = sm.add_constant(np.column_stack([t, Xc]), has_constant="add")
    res = sm.OLS(y, X).fit(cov_type="HC3", use_t=True)
    lo, hi = res.conf_int()[1]
    p = float(res.pvalues[1])
    xbar = Xc.mean(axis=0) if Xc.shape[1] else np.zeros(0)
    base = float(np.concatenate([[1.0, 0.0], xbar]) @ res.params)      # adjusted control average
    return float(res.params[1]), float(lo), float(hi), p if np.isfinite(p) else 1.0, base

def evalue(diff, lo, hi, base, sd, binary):
    """How strong a hidden factor must be (tied to both choosing and the result) to erase the gap."""
    base = max(base, 1e-6)
    def rr(x):
        r = max((base + x) / base, 1e-9) if binary else float(np.exp(0.91 * x / sd))
        return 1 / r if r < 1 else r
    ev = lambda r: r + np.sqrt(r * (r - 1))
    return ev(rr(diff)), (1.0 if lo <= 0 <= hi else ev(rr(lo if diff > 0 else hi)))

diff, low, high, p, base = adjusted(df, covs)
print(f"rows_used={len(df)}")
print(f"diff={diff:.4f} low={low:.4f} high={high:.4f} p={p:.5f}")
print(f"evalue={evalue(diff, low, high, base, df[outcome].std(ddof=1), __BINARY__)[0]:.4f}")
'''

_OBS_BINARY_EFFECT = '''
from statsmodels.stats.proportion import confint_proportions_2indep, test_proportions_2indep
def raw_effect(c, t):
    xc, nc, xt, nt = int(c.sum()), len(c), int(t.sum()), len(t)
    low, high = confint_proportions_2indep(xt, nt, xc, nc, method="newcomb", compare="diff")
    p = test_proportions_2indep(xt, nt, xc, nc, method="score", compare="diff", correction=False).pvalue
    return xt / nt - xc / nc, float(low), float(high), float(p) if np.isfinite(p) else 1.0
'''

_OBS_CONT_EFFECT = '''
def raw_effect(c, t):
    res = stats.ttest_ind(t, c, equal_var=False)   # Welch
    ci = res.confidence_interval(0.95)
    p = float(res.pvalue)
    return float(t.mean() - c.mean()), float(ci.low), float(ci.high), p if np.isfinite(p) else 1.0
'''

_OBS_CROSS = '''
# --- cross-checks ---
t = (df[group].astype(str) == treatment).to_numpy(float)
X = sm.add_constant(design(df, covs).to_numpy(float), has_constant="add")
try:                                           # chance of choosing it, given the columns
    ps = np.asarray(sm.GLM(t, X, family=sm.families.Binomial()).fit().predict(X), float)
except Exception:
    ps = None
if ps is not None and np.all(np.isfinite(ps)):
    pc = np.clip(ps, 0.05, 0.95)
    w = t / pc + (1 - t) / (1 - pc)            # weight people who look unlikely to have chosen it
    res = sm.WLS(df[outcome].to_numpy(float), sm.add_constant(t), weights=w).fit(cov_type="HC1", use_t=True)
    pw = float(res.pvalues[1])
    print(f"ipw_diff={res.params[1]:.4f} ipw_p={pw if np.isfinite(pw) else 1.0:.5f}")
    kept = df[(ps >= 0.10) & (ps <= 0.90)]     # only people who could realistically have gone either way
    gk = kept[group].astype(str)
    if (gk == control).sum() >= 30 and (gk == treatment).sum() >= 30:
        print(f"overlap_p={adjusted(kept, covs)[3]:.5f}")

is_c = df[group].astype(str) == control
print(f"unadjusted_diff={raw_effect(df.loc[is_c, outcome].to_numpy(float), df.loc[~is_c, outcome].to_numpy(float))[0]:.4f} "
      f"unadjusted_p={raw_effect(df.loc[is_c, outcome].to_numpy(float), df.loc[~is_c, outcome].to_numpy(float))[3]:.5f}")
'''

_OBS_SLICES = '''
# --- slice scan (same comparison, inside each slice) ---
ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|^user$|^customer$|^account$)", re.I)
def idlike(col):
    return bool(ID_NAME.search(col)) or (pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() >= 0.9 * len(df))

rows = []
for col in df.columns:
    if col in (group, outcome) or idlike(col):
        continue
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) or not (2 <= s.nunique() <= 8):
        continue   # slice only on category columns; numeric columns are usually results
    here = [c for c in covs if c != col]
    if not here:
        continue
    for level in sorted(s.dropna().astype(str).unique()):
        m = s.astype(str) == level
        g = df[group].astype(str)
        if (m & (g == control)).sum() >= 50 and (m & (g == treatment)).sum() >= 50:
            d, lo, hi, p_raw, _ = adjusted(df[m], here)
            rows.append((col, level, d, p_raw))
if rows:
    p_adj = multipletests([r[3] for r in rows], method="holm")[1]
    for (col, level, d, p_raw), pa in sorted(zip(rows, p_adj), key=lambda x: x[0][3]):
        print(f"slice {col}={level} diff={d:.4f} p_raw={p_raw:.5f} p_adj={pa:.5f}")
'''


def build_obs_script(spec, kind: str, csv_name: str, mode: str, header: str = "") -> str:
    binary = kind == "binary"
    out = [f"{header}import numpy as np\nimport pandas as pd\nimport statsmodels.api as sm\nfrom scipy import stats\n"]
    if mode == "research":
        out.append("import re\nfrom statsmodels.stats.multitest import multipletests\n")
    if mode == "right_answer":
        out.append(_OBS_BINARY_EFFECT if binary else _OBS_CONT_EFFECT)
    body = (_OBS_BODY.replace("__GROUP__", repr(spec.group_col)).replace("__CONTROL__", repr(spec.control))
            .replace("__TREATMENT__", repr(spec.treatment)).replace("__OUTCOME__", repr(spec.outcome_col))
            .replace("__COVS__", repr(list(spec.covariates))).replace("__CSV__", repr(csv_name))
            .replace("__BINARY__", repr(binary)))
    out.append(body)
    if mode == "right_answer":
        out.append(_OBS_CROSS)
    if mode == "research":
        out.append(_OBS_SLICES)
    return "".join(out)
