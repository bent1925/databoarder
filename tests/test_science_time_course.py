"""Science mode: two-way repeated-measures ANOVA (group x time), checked against independent calculations."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from fastapi.testclient import TestClient
from scipy import stats
from statsmodels.stats.anova import anova_lm

from app.main import app
from app.science import engine, report

client = TestClient(app)


def course(sizes=(5, 6, 4), days=(1, 4, 8, 11), slopes=(0.12, 0.06, 0.0), seed=11):
    """Groups of different sizes, every animal measured on every day, groups diverging over time."""
    rng = np.random.default_rng(seed)
    rows = []
    for g, (n, slope) in enumerate(zip(sizes, slopes)):
        for a in range(n):
            level = rng.normal(2.5, 0.3)
            for d in days:
                rows.append({"Group": f"G{g}", "Mouse": f"G{g}-{a}", "Day": d,
                             "Volume": level + slope * d + rng.normal(0, 0.15 + 0.01 * d)})
    return pd.DataFrame(rows)


def spec_for(df, **kw):
    spec = dict(analysis="time_course", group_col="Group", time_col="Day", value_col="Volume", subject_col="Mouse",
                groups=sorted(df.Group.unique()), times=[str(d) for d in sorted(df.Day.unique())], followup="none",
                control=None, assignment="random")
    spec.update(kw)
    return spec


def test_sums_of_squares_match_a_model_with_a_term_for_each_subject():
    df = course()
    r = engine.run(df, spec_for(df))
    ss = {row["source"]: row["ss"] for row in r["anova_table"]}
    # Each effect is the drop in leftover variation when its term is added to a model with a term for every subject.
    # (Residual sums of squares are used because that model is rank-deficient, which can upset sequential tables.)
    rss = [smf.ols(f, data=df).fit().ssr for f in ("Volume ~ C(Mouse)", "Volume ~ C(Mouse) + C(Day)",
                                                  "Volume ~ C(Mouse) + C(Day) + C(Group):C(Day)")]
    assert ss["Time"] == pytest.approx(rss[0] - rss[1])
    assert ss["Group × time"] == pytest.approx(rss[1] - rss[2])
    assert ss["Residual"] == pytest.approx(rss[2])
    per_mouse = df.groupby(["Mouse", "Group"], as_index=False)["Volume"].mean()
    between = anova_lm(smf.ols("Volume ~ C(Group)", data=per_mouse).fit())
    group = next(t for t in r["tests"] if t["key"] == "group")
    assert group["statistic"] == pytest.approx(between.loc["C(Group)", "F"])
    assert group["p"] == pytest.approx(between.loc["C(Group)", "PR(>F)"])
    inter = r["tests"][0]
    days, groups, mice = df.Day.nunique(), df.Group.nunique(), df.Mouse.nunique()
    df_gt, df_err = (groups - 1) * (days - 1), (mice - groups) * (days - 1)
    assert inter["statistic"] == pytest.approx(((rss[1] - rss[2]) / df_gt) / (rss[2] / df_err))


def test_with_two_days_the_interaction_is_an_anova_on_each_animals_change():
    df = course(days=(1, 12))
    r = engine.run(df, spec_for(df))
    change = df.pivot(index=["Mouse", "Group"], columns="Day", values="Volume").reset_index()
    change["delta"] = change[12] - change[1]
    ref = stats.f_oneway(*[change.loc[change.Group == g, "delta"] for g in sorted(df.Group.unique())])
    assert r["epsilon"] == 1.0
    assert r["tests"][0]["statistic"] == pytest.approx(ref.statistic)
    assert r["tests"][0]["p"] == pytest.approx(ref.pvalue)


def test_greenhouse_geisser_epsilon_from_orthonormal_contrasts():
    df = course()
    r = engine.run(df, spec_for(df))
    days = sorted(df.Day.unique())
    blocks = [df[df.Group == g].pivot(index="Mouse", columns="Day", values="Volume")[days].to_numpy()
              for g in sorted(df.Group.unique())]
    pooled = sum((len(y) - 1) * np.cov(y, rowvar=False) for y in blocks) / (sum(len(y) for y in blocks) - len(blocks))
    k = len(days)
    contrasts = np.linalg.qr(np.column_stack([np.ones(k), np.eye(k)[:, :-1]]))[0][:, 1:]
    lam = np.linalg.eigvalsh(contrasts.T @ pooled @ contrasts)
    assert r["epsilon"] == pytest.approx(lam.sum() ** 2 / ((k - 1) * (lam ** 2).sum()))
    assert 1 / (k - 1) <= r["epsilon"] <= 1


def test_follow_ups_compare_groups_at_each_day_holm_adjusted():
    df = course()
    r = engine.run(df, spec_for(df, followup="vs_control", control="G2"))
    assert len(r["pairs"]) == 2 * 4 and {p["a"] for p in r["pairs"]} == {"G2"}
    raw = []
    for p in r["pairs"]:
        x = df[(df.Group == p["a"]) & (df.Day == int(p["time"]))].Volume
        y = df[(df.Group == p["b"]) & (df.Day == int(p["time"]))].Volume
        assert p["p_raw"] == pytest.approx(stats.ttest_ind(y, x, equal_var=False).pvalue)
        raw.append(p["p_raw"])
    order, held = np.argsort(raw), 0.0
    for rank, i in enumerate(order):
        held = max(held, min(1.0, raw[i] * (len(raw) - rank)))
        assert r["pairs"][i]["p_adj"] == pytest.approx(held)
    assert len(engine.run(df, spec_for(df, followup="all_pairs"))["pairs"]) == 3 * 4


def test_the_planted_divergence_is_found_and_worded_as_the_main_question():
    df = course(sizes=(8, 8, 8))
    r = engine.run(df, spec_for(df))
    assert r["clear"] and r["main_p"] == r["tests"][0]["p"]
    rep = report.build(spec_for(df), r)
    assert rep.headline == "The groups changed differently over days."
    assert "read the day-by-day comparisons" in rep.how_sure
    flat = course(sizes=(8, 8, 8), slopes=(0.05, 0.05, 0.05))
    assert not engine.run(flat, spec_for(flat))["clear"]


def test_animals_missing_a_day_are_left_out_and_counted():
    df = course()
    df.loc[(df.Mouse == "G1-2") & (df.Day == 8), "Volume"] = np.nan
    r = engine.run(df, spec_for(df))
    assert r["left_out"]["incomplete_subjects"] == 1 and r["left_out"]["incomplete_by_group"]["G1"] == 1
    assert [m["n"] for m in r["means"]] == [5, 5, 4]
    assert any(c.name == "Incomplete subjects" for c in report.build(spec_for(df), r).checks)


def test_leaving_out_a_day_keeps_animals_that_missed_only_that_day():
    df = course()
    df.loc[(df.Mouse == "G1-2") & (df.Day == 1), "Volume"] = np.nan
    r = engine.run(df, spec_for(df, times=["4", "8", "11"]))
    assert r["left_out"]["incomplete_subjects"] == 0 and r["times"] == ["4", "8", "11"]


@pytest.mark.parametrize("change, message", [
    (lambda d: d.assign(Mouse=d.Mouse.str[-1]), "more than one group"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "more than one value"),
    (lambda d: d[~d.Group.eq("G2") | d.Mouse.eq("G2-0")], "fewer than 2 subjects"),
])
def test_tables_that_dont_fit_are_stopped_with_a_reason(change, message):
    with pytest.raises(engine.EngineError, match=message):
        engine.run(change(course()), spec_for(course()))


def test_readback():
    text = report.readback(spec_for(course(), followup="vs_control", control="G0"))
    assert "two-way repeated-measures ANOVA" in text and "Greenhouse-Geisser" in text
    assert "at each day, compare each group with 'G0'" in text and "Holm" in text


def test_a_long_csv_is_guessed_and_the_script_reproduces_it(tmp_path):
    df = course()
    raw = df.to_csv(index=False).encode()
    d = client.post("/api/science/upload", files={"file": ("course.csv", raw, "text/csv")}).json()
    assert d["guesses"]["time_course"] == {"group_col": "Group", "time_col": "Day", "value_col": "Volume",
                                           "subject_col": "Mouse"}
    spec = spec_for(df, followup="all_pairs")
    rep = client.post("/api/science/run", json={"dataset_id": d["dataset_id"], "spec": spec})
    assert rep.status_code == 200, rep.text
    rep = rep.json()
    (tmp_path / "course.csv").write_bytes(raw)
    (tmp_path / "reproduce.py").write_text(rep["script"])
    done = subprocess.run([sys.executable, "-I", "reproduce.py"], cwd=tmp_path, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr
    again = json.loads(done.stdout)
    assert again["tests"] == rep["result"]["tests"] and again["pairs"] == rep["result"]["pairs"]
