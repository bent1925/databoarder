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
