"""Science mode: every number checked against an independent calculation, the guards that stop a wrong analysis,
the reproduction script, and the web routes."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from fastapi.testclient import TestClient
from scipy import stats
from statsmodels.stats.anova import AnovaRM, anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from app.main import app
from app.science import engine, report
from app.science.routes import pasted_to_csv

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "static" / "science"
client = TestClient(app)


def groups_spec(groups, **kw):
    spec = dict(analysis="compare_groups", layout="wide", groups=list(groups), paired=False, method="means",
                equal_spread=False, followup="none", control=None, assignment="unsure")
    spec.update(kw)
    return spec


@pytest.fixture
def three():
    rng = np.random.default_rng(7)
    return pd.DataFrame({"A": rng.normal(10, 2, 9), "B": rng.normal(13, 2, 9), "C": rng.normal(11, 5, 9)})


def long_of(df):
    return df.melt(var_name="g", value_name="y").dropna()


# ---------------------------------------------------------------------------- independent groups

def test_ordinary_anova_matches_a_linear_model(three):
    r = engine.run(three, groups_spec("ABC", equal_spread=True))
    table = anova_lm(smf.ols("y ~ C(g)", data=long_of(three)).fit())
    assert r["tests"][0]["statistic"] == pytest.approx(table.loc["C(g)", "F"])
    assert r["main_p"] == pytest.approx(table.loc["C(g)", "PR(>F)"])
    ss_b, ss_e = table.loc["C(g)", "sum_sq"], table.loc["Residual", "sum_sq"]
    assert r["effect"]["value"] == pytest.approx(ss_b / (ss_b + ss_e))


def test_welch_anova_matches_welchs_formula(three):
    r = engine.run(three, groups_spec("ABC"))
    s = [three[c].to_numpy() for c in "ABC"]
    w = np.array([len(x) / x.var(ddof=1) for x in s])
    m = np.array([x.mean() for x in s])
    k = len(s)
    mw = (w * m).sum() / w.sum()
    lam = sum((1 - wi / w.sum()) ** 2 / (len(x) - 1) for wi, x in zip(w, s))
    f = ((w * (m - mw) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) * lam / (k ** 2 - 1))
    df2 = (k ** 2 - 1) / (3 * lam)
    assert r["tests"][0]["statistic"] == pytest.approx(f)
    assert r["main_p"] == pytest.approx(stats.f.sf(f, k - 1, df2))


def test_tukey_matches_statsmodels(three):
    r = engine.run(three, groups_spec("ABC", equal_spread=True, followup="all_pairs"))
    ref = pairwise_tukeyhsd(long_of(three)["y"], long_of(three)["g"]).summary().data[1:]
    for (a, b, diff, p, lo, hi, _), row in zip(ref, r["pairs"]):
        assert (row["a"], row["b"]) == (a, b)
        assert row["diff"] == pytest.approx(diff, abs=1e-3)
        assert row["p_adj"] == pytest.approx(p, abs=1e-3)
        assert (row["ci_low"], row["ci_high"]) == pytest.approx((lo, hi), abs=1e-3)


def test_dunnett_compares_each_group_with_the_control(three):
    r = engine.run(three, groups_spec("ABC", equal_spread=True, followup="vs_control", control="B"))
    assert [(p["a"], p["b"]) for p in r["pairs"]] == [("B", "A"), ("B", "C")]
    for p in r["pairs"]:
        assert p["diff"] == pytest.approx(three[p["b"]].mean() - three["B"].mean())
        assert p["ci_low"] < p["diff"] < p["ci_high"]
        assert (p["ci_low"] > 0 or p["ci_high"] < 0) == (p["p_adj"] < 0.05)


def test_games_howell_with_two_groups_is_welchs_t_test(three):
    s = [three["A"].to_numpy(), three["C"].to_numpy()]
    row = engine.games_howell(s, ["A", "C"], [(0, 1)])[0]
    welch = stats.ttest_ind(s[1], s[0], equal_var=False)
    assert row["p_adj"] == pytest.approx(welch.pvalue, rel=1e-4)
    ci = welch.confidence_interval(0.95)
    assert (row["ci_low"], row["ci_high"]) == pytest.approx((ci.low, ci.high), rel=1e-4)


def test_dunn_with_two_groups_is_the_rank_sum_z_test():
    a, b = np.array([1, 2, 2, 3, 5, 8, 9.0]), np.array([4, 6, 6, 7, 10, 11, 12, 13.0])
    row = engine.dunn([a, b], ["a", "b"], [(0, 1)])[0]
    mw = stats.mannwhitneyu(b, a, method="asymptotic", use_continuity=False)
    assert row["p_raw"] == pytest.approx(mw.pvalue, rel=1e-6)


def test_kruskal_wallis_and_dunn_holm(three):
    r = engine.run(three, groups_spec("ABC", method="ranks", followup="all_pairs"))
    assert r["main_p"] == pytest.approx(stats.kruskal(three["A"], three["B"], three["C"]).pvalue)
    raw = [p["p_raw"] for p in r["pairs"]]
    order = np.argsort(raw)
    held = 0.0
    for rank, i in enumerate(order):                       # Holm, by hand
        held = max(held, min(1.0, raw[i] * (len(raw) - rank)))
        assert r["pairs"][i]["p_adj"] == pytest.approx(held)


def test_two_groups(three):
    welch = engine.run(three, groups_spec("AB"))
    ref = stats.ttest_ind(three["B"], three["A"], equal_var=False)
    assert welch["main_p"] == pytest.approx(ref.pvalue)
    assert welch["effect"]["diff"] == pytest.approx(three["B"].mean() - three["A"].mean())
    student = engine.run(three, groups_spec("AB", equal_spread=True))
    assert student["main_p"] == pytest.approx(stats.ttest_ind(three["B"], three["A"]).pvalue)
    mw = engine.run(three, groups_spec("AB", method="ranks"))
    assert mw["main_p"] == pytest.approx(stats.mannwhitneyu(three["B"], three["A"]).pvalue)
    assert mw["pairs"] == [] and "pair_method" not in mw


def test_follow_ups_are_ignored_for_two_groups(three):
    r = engine.run(three, groups_spec("AB", followup="all_pairs"))
    assert r["pairs"] == []


def test_long_layout_gives_the_same_answer_as_wide(three):
    long = long_of(three).rename(columns={"g": "treatment", "y": "weight"})
    wide = engine.run(three, groups_spec("ABC", equal_spread=True, followup="all_pairs"))
    longr = engine.run(long, groups_spec("ABC", layout="long", group_col="treatment", value_col="weight",
                                         equal_spread=True, followup="all_pairs"))
    assert longr["main_p"] == pytest.approx(wide["main_p"])
    assert [p["p_adj"] for p in longr["pairs"]] == pytest.approx([p["p_adj"] for p in wide["pairs"]])


# ---------------------------------------------------------------------------- repeated measures

@pytest.fixture
def visits():
    rng = np.random.default_rng(3)
    base = rng.normal(150, 10, 10)
    return pd.DataFrame({"id": [f"p{i}" for i in range(10)], "t0": base + rng.normal(0, 3, 10),
                         "t1": base - 3 + rng.normal(0, 3, 10), "t2": base - 6 + rng.normal(0, 6, 10)})


def test_repeated_measures_anova_matches_statsmodels(visits):
    r = engine.run(visits, groups_spec(["t0", "t1", "t2"], paired=True, subject_col="id"))
    long = visits.melt(id_vars="id", var_name="visit", value_name="bp")
    ref = AnovaRM(long, "bp", "id", within=["visit"]).fit().anova_table
    assert r["tests"][0]["statistic"] == pytest.approx(ref["F Value"].iloc[0])
    assert r["tests"][0]["p_uncorrected"] == pytest.approx(ref["Pr > F"].iloc[0])


def test_greenhouse_geisser_epsilon_from_orthonormal_contrasts(visits):
    wide = visits[["t0", "t1", "t2"]].to_numpy()
    k = wide.shape[1]
    helmert = np.linalg.qr(np.column_stack([np.ones(k), np.eye(k)[:, :-1]]))[0][:, 1:]   # orthonormal contrasts
    lam = np.linalg.eigvalsh(helmert.T @ np.cov(wide, rowvar=False) @ helmert)
    expected = lam.sum() ** 2 / ((k - 1) * (lam ** 2).sum())
    assert engine._sphericity(wide) == pytest.approx(expected)
    assert 1 / (k - 1) <= expected <= 1


def test_paired_two_groups_and_bonferroni_intervals(visits):
    r = engine.run(visits, groups_spec(["t0", "t2"], paired=True))
    assert r["main_p"] == pytest.approx(stats.ttest_rel(visits["t2"], visits["t0"]).pvalue)
    full = engine.run(visits, groups_spec(["t0", "t1", "t2"], paired=True, followup="all_pairs"))
    for p in full["pairs"]:
        assert p["p_adj"] == pytest.approx(min(1, 3 * stats.ttest_rel(visits[p["b"]], visits[p["a"]]).pvalue))
        assert (p["ci_low"] > 0 or p["ci_high"] < 0) == (p["p_adj"] < 0.05)


def test_friedman_and_wilcoxon(visits):
    r = engine.run(visits, groups_spec(["t0", "t1", "t2"], paired=True, method="ranks", followup="vs_control",
                                       control="t0"))
    assert r["main_p"] == pytest.approx(stats.friedmanchisquare(visits.t0, visits.t1, visits.t2).pvalue)
    w = engine.run(visits, groups_spec(["t0", "t1"], paired=True, method="ranks"))
    assert w["main_p"] == pytest.approx(stats.wilcoxon(visits.t1, visits.t0).pvalue)


def test_subjects_missing_a_visit_are_left_out_and_counted(visits):
    visits.loc[2, "t1"] = np.nan
    r = engine.run(visits, groups_spec(["t0", "t1", "t2"], paired=True))
    assert r["n_subjects"] == 9 and r["left_out"]["incomplete_subjects"] == 1
    rep = report.build(groups_spec(["t0", "t1", "t2"], paired=True), r)
    assert any(c.name == "Incomplete subjects" for c in rep.checks)


def test_long_paired_needs_one_value_per_subject_and_group():
    df = pd.DataFrame({"mouse": ["m1", "m1", "m1", "m2", "m2", "m3", "m3"],
                       "day": ["d0", "d7", "d7", "d0", "d7", "d0", "d7"], "y": [1, 2, 3, 1, 2, 1, 2.0]})
    spec = groups_spec(["d0", "d7"], layout="long", group_col="day", value_col="y", subject_col="mouse", paired=True)
    with pytest.raises(engine.EngineError, match="more than one value"):
        engine.run(df, spec)
    with pytest.raises(engine.EngineError, match="which subject"):
        engine.run(df, {**spec, "subject_col": None})


# ---------------------------------------------------------------------------- guards and missing values

def test_empty_cells_are_left_out_and_reported(three):
    three.loc[[0, 4], "B"] = np.nan
    r = engine.run(three, groups_spec("ABC"))
    assert r["left_out"]["empty_values"] == {"A": 0, "B": 2, "C": 0}
    assert r["groups"][1]["n"] == 7
    rep = report.build(groups_spec("ABC"), r)
    assert any(c.name == "Empty cells" and "2 empty cells" in c.message for c in rep.checks)


@pytest.mark.parametrize("change, message", [
    (lambda d: d.assign(B=["x"] * len(d)), "aren't numbers"),
    (lambda d: d.assign(B=[1.0] + [np.nan] * (len(d) - 1)), "fewer than 2 values"),
    (lambda d: d.drop(columns="B"), "isn't in your table"),
])
def test_bad_tables_are_stopped_with_a_reason(three, change, message):
    with pytest.raises(engine.EngineError, match=message):
        engine.run(change(three), groups_spec("ABC"))


def test_the_normality_check_never_changes_the_test():
    rng = np.random.default_rng(1)
    skewed = pd.DataFrame({"A": rng.lognormal(0, 1.2, 30), "B": rng.lognormal(0.5, 1.2, 30)})
    spec = groups_spec("AB")
    r = engine.run(skewed, spec)
    assert r["tests"][0]["key"] == "welch_t"                       # still the test that was chosen
    rep = report.build(spec, r)
    assert any(c.name == "Shape of the values" and c.status == "warn" for c in rep.checks)


def test_a_passing_shape_check_is_not_presented_as_proof(three):
    rep = report.build(groups_spec("ABC"), engine.run(three, groups_spec("ABC")))
    shape = next(c for c in rep.checks if c.name == "Shape of the values")
    if shape.status == "info":
        assert "doesn't prove" in shape.message


def test_paired_shape_check_looks_at_the_differences():
    base = np.array([10, 50, 100, 400, 800, 1600, 3200, 6400.0])        # very skewed levels
    df = pd.DataFrame({"pre": base, "post": base + np.array([1, -1, 2, 0, 1, -2, 1, 0])})
    r = engine.run(df, groups_spec(["pre", "post"], paired=True))
    assert r["groups"][0]["normality"]["p"] < 0.05
    rep = report.build(groups_spec(["pre", "post"], paired=True), r)
    assert not any(c.name == "Shape of the values" and c.status == "warn" for c in rep.checks)


# ---------------------------------------------------------------------------- contingency

def counts_spec(**kw):
    spec = dict(analysis="contingency", layout="counts", label_col="Group", rows=["Vaccinated", "Placebo"],
                cols=["Infected", "Not infected"], event="Infected", assignment="random")
    spec.update(kw)
    return spec


def test_two_by_two():
    df = pd.read_csv(SAMPLES / "infection_counts.csv")
    r = engine.run(df, counts_spec())
    assert r["main_p"] == pytest.approx(stats.fisher_exact([[8, 92], [30, 70]]).pvalue)
    e = r["effect"]
    assert e["relative_risk"] == pytest.approx(0.08 / 0.30)
    assert e["odds_ratio"] == pytest.approx((8 * 70) / (92 * 30))
    se = math.sqrt(1 / 8 - 1 / 100 + 1 / 30 - 1 / 100)
    assert (e["rr_low"], e["rr_high"]) == pytest.approx((0.08 / 0.3 * math.exp(-1.959964 * se),
                                                         0.08 / 0.3 * math.exp(1.959964 * se)), rel=1e-4)
    flipped = engine.run(df, counts_spec(event="Not infected"))
    assert flipped["effect"]["relative_risk"] == pytest.approx(0.92 / 0.70)
    assert flipped["main_p"] == pytest.approx(r["main_p"])


def test_counting_rows_gives_the_same_table():
    df = pd.DataFrame({"arm": ["V"] * 100 + ["P"] * 100 + [None] * 3,
                       "outcome": ["inf"] * 8 + ["ok"] * 92 + ["inf"] * 30 + ["ok"] * 70 + ["inf"] * 3})
    r = engine.run(df, dict(analysis="contingency", layout="rows", row_col="arm", col_col="outcome", rows=["V", "P"],
                            cols=["inf", "ok"], event="inf"))
    assert r["counts"] == [[8, 92], [30, 70]] and r["left_out"] == 3


def test_bigger_tables_use_chi_square():
    df = pd.DataFrame({"g": ["a", "b", "c"], "x": [10, 20, 30], "y": [30, 20, 12], "z": [5, 5, 9]})
    spec = dict(analysis="contingency", layout="counts", label_col="g", rows=["a", "b", "c"], cols=["x", "y", "z"])
    r = engine.run(df, spec)
    chi = stats.chi2_contingency(df[["x", "y", "z"]].to_numpy(), correction=False)
    assert r["main_p"] == pytest.approx(chi.pvalue)
    assert r["effect"]["value"] == pytest.approx(math.sqrt(chi.statistic / (df[["x", "y", "z"]].to_numpy().sum() * 2)))


def test_counts_must_be_whole_numbers():
    df = pd.DataFrame({"Group": ["Vaccinated", "Placebo"], "Infected": [8.5, 30], "Not infected": [92, 70]})
    with pytest.raises(engine.EngineError, match="whole number"):
        engine.run(df, counts_spec())


# ---------------------------------------------------------------------------- correlation

def test_correlation():
    df = pd.read_csv(SAMPLES / "gene_expression.csv")
    x, y = df["mRNA (TPM)"], df["Protein (ng/mL)"]
    p = engine.run(df, dict(analysis="correlation", x_col="mRNA (TPM)", y_col="Protein (ng/mL)", method="pearson"))
    assert p["effect"]["value"] == pytest.approx(np.corrcoef(x, y)[0, 1])
    z, se = np.arctanh(p["effect"]["value"]), 1 / math.sqrt(len(x) - 3)
    assert (p["effect"]["ci_low"], p["effect"]["ci_high"]) == pytest.approx(
        (np.tanh(z - 1.959964 * se), np.tanh(z + 1.959964 * se)), rel=1e-4)
    s = engine.run(df, dict(analysis="correlation", x_col="mRNA (TPM)", y_col="Protein (ng/mL)", method="spearman"))
    assert s["effect"]["value"] == pytest.approx(stats.spearmanr(x, y).statistic)


# ---------------------------------------------------------------------------- wording

def test_wording_follows_how_subjects_were_assigned(three):
    r = engine.run(three, groups_spec("AB"))
    assert "put down to what the groups received" in report.build(groups_spec("AB", assignment="random"), r).causal_label
    for who in ("not_random", "unsure"):
        assert "not why" in report.build(groups_spec("AB", assignment=who), r).causal_label


def test_readback_names_the_test_and_the_follow_up():
    text = report.readback(groups_spec("ABC", equal_spread=True, followup="vs_control", control="A",
                                       assignment="random"))
    assert "ordinary one-way ANOVA" in text and "Dunnett's test" in text and "'A'" in text and "at random" in text
    assert "Welch's ANOVA" in report.readback(groups_spec("ABC", followup="none"))
    assert "Greenhouse-Geisser" in report.readback(groups_spec("ABC", paired=True))


# ---------------------------------------------------------------------------- planted answers in the samples

def run_sample(name, spec):
    raw = (SAMPLES / name).read_bytes()
    d = client.post("/api/science/upload", files={"file": (name, raw, "text/csv")}).json()
    out = client.post("/api/science/run", json={"dataset_id": d["dataset_id"], "spec": spec})
    assert out.status_code == 200, out.text
    return d, out.json()


TUMOUR = groups_spec(["Vehicle", "Low dose", "Mid dose", "High dose"], equal_spread=True, followup="vs_control",
                     control="Vehicle", assignment="random")
PRESSURE = groups_spec(["Baseline", "Week 4", "Week 8"], paired=True, subject_col="Patient", followup="vs_control",
                       control="Baseline", assignment="not_random")
SAMPLE_SPECS = [
    ("tumour_volume.csv", TUMOUR),
    ("tumour_volume.csv", {**TUMOUR, "method": "ranks", "followup": "all_pairs"}),
    ("tumour_volume.csv", {**TUMOUR, "equal_spread": False, "followup": "all_pairs"}),
    ("blood_pressure.csv", PRESSURE),
    ("blood_pressure.csv", {**PRESSURE, "method": "ranks"}),
    ("blood_pressure.csv", {**PRESSURE, "groups": ["Baseline", "Week 8"]}),
    ("infection_counts.csv", counts_spec()),
    ("gene_expression.csv", dict(analysis="correlation", x_col="mRNA (TPM)", y_col="Protein (ng/mL)",
                                 method="pearson")),
]


def test_samples_give_their_planted_answers():
    _, t = run_sample("tumour_volume.csv", TUMOUR)
    assert t["verdict"] == "clear_difference"
    clear = {p["b"] for p in t["result"]["pairs"] if p["clear"]}
    assert {"High dose", "Mid dose"} <= clear and "Low dose" not in clear
    assert t["result"]["left_out"]["empty_values"]["Mid dose"] == 1
    _, bp = run_sample("blood_pressure.csv", PRESSURE)
    assert bp["verdict"] == "clear_difference" and bp["result"]["left_out"]["incomplete_subjects"] == 1
    assert bp["result"]["pairs"][-1]["diff"] < -4
    _, inf = run_sample("infection_counts.csv", counts_spec())
    assert inf["verdict"] == "clear_difference" and inf["result"]["effect"]["relative_risk"] < 0.5
    _, ge = run_sample("gene_expression.csv", SAMPLE_SPECS[-1][1])
    assert ge["verdict"] == "clear_difference" and 0.3 < ge["result"]["effect"]["value"] < 0.9


def _close(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _close(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _close(x, y)
    elif isinstance(a, float):
        assert b == pytest.approx(a, rel=1e-9, abs=1e-12)
    else:
        assert a == b


@pytest.mark.parametrize("name, spec", SAMPLE_SPECS)
def test_the_script_reproduces_every_number(tmp_path, name, spec):
    d, rep = run_sample(name, spec)
    (tmp_path / name).write_bytes(client.get(f"/api/science/datasets/{d['dataset_id']}/data").content)
    (tmp_path / "reproduce.py").write_text(rep["script"])
    done = subprocess.run([sys.executable, "-I", "reproduce.py"], cwd=tmp_path, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr
    _close(rep["result"], json.loads(done.stdout))


def test_the_script_refuses_a_different_file(tmp_path):
    d, rep = run_sample("tumour_volume.csv", TUMOUR)
    (tmp_path / "tumour_volume.csv").write_text((SAMPLES / "tumour_volume.csv").read_text().replace("703", "704"))
    (tmp_path / "reproduce.py").write_text(rep["script"])
    done = subprocess.run([sys.executable, "-I", "reproduce.py"], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode != 0 and "isn't the file the report used" in done.stderr


def test_the_script_repeats_the_tidying(tmp_path):
    raw = b"Control,Treated\n$4.10,$5.30\n$3.80,$6.00\n$4.40,$5.10\n$4.00,$5.90\n$4.20,$5.50\n"
    d = client.post("/api/science/upload", files={"file": ("money.csv", raw, "text/csv")}).json()
    assert d["prep"]["steps"]
    spec = groups_spec(["Control", "Treated"])
    rep = client.post("/api/science/run", json={"dataset_id": d["dataset_id"], "spec": spec}).json()
    (tmp_path / "money.csv").write_bytes(raw)
    (tmp_path / "reproduce.py").write_text(rep["script"])
    done = subprocess.run([sys.executable, "-I", "reproduce.py"], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    _close(rep["result"], json.loads(done.stdout))


# ---------------------------------------------------------------------------- web routes

def test_pasted_cells_keep_their_values():
    raw = pasted_to_csv("Control\tTreated, high\n4.1\t5.3\n3.8\t\n\n")
    assert raw == b'Control,"Treated, high"\n4.1,5.3\n3.8,\n'
    d = client.post("/api/science/paste", json={"text": "a\tb\n1\t2\n3\t4\n5\t7"}).json()
    assert d["rows"] == 3 and d["guesses"]["compare_groups"]["layout"] == "wide"
    assert client.post("/api/science/paste", json={"text": "  \n"}).status_code == 400


def test_file_names_are_made_safe():
    d = client.post("/api/science/upload", files={"file": ('../../evil"\r\nX: y.csv', b"a,b\n1,2\n3,4\n", "text/csv")}).json()
    assert "/" not in d["name"] and '"' not in d["name"] and "\n" not in d["name"]


def test_files_that_cant_be_read_get_a_way_forward():
    out = client.post("/api/science/upload", files={"file": ("plate.xlsx", b"PK..", "application/octet-stream")})
    assert out.status_code == 400 and "save it as .xlsx" in out.json()["detail"]
    out = client.post("/api/science/upload", files={"file": ("old.xls", b"..", "application/octet-stream")})
    assert out.status_code == 400 and ".xlsx" in out.json()["detail"]
    out = client.post("/api/science/upload", files={"file": ("study.numbers", b"..", "application/octet-stream")})
    assert out.status_code == 400 and "Export To > Excel" in out.json()["detail"]


def test_readback_reports_problems_before_running():
    d = client.post("/api/science/paste", json={"text": "a\tb\n1\t\n\t2\n3\t4\n5\t6"}).json()
    rb = client.post("/api/science/readback", json={"dataset_id": d["dataset_id"],
                                                     "spec": groups_spec("ab", paired=True)}).json()
    assert "Fewer than 3 subjects" in rb["problem"]
    out = client.post("/api/science/run", json={"dataset_id": d["dataset_id"], "spec": groups_spec("ab", paired=True)})
    assert out.status_code == 400


def test_choices_that_dont_fit_are_refused_plainly():
    d = client.post("/api/science/paste", json={"text": "a\tb\tc\n1\t2\t3\n3\t4\t5\n5\t6\t9"}).json()
    out = client.post("/api/science/run", json={"dataset_id": d["dataset_id"],
                                                "spec": groups_spec("abc", followup="vs_control", control=None)})
    assert out.status_code == 400 and "control" in out.json()["detail"]
    out = client.post("/api/science/run", json={"dataset_id": d["dataset_id"], "spec": {**groups_spec("abc"), "extra": 1}})
    assert out.status_code == 400


def test_the_page_is_served():
    page = client.get("/science")
    assert page.status_code == 200 and "Databoarder Lab" in page.text
    assert "/science" in client.get("/").text


@pytest.mark.parametrize("x, shown", [(762.25, "762.3"), (1.0005, "1"), (143.0, "143"), (12345.6, "12350"),
                                      (9999.5, "10000"), (0.012345, "0.01235"), (0.0001234, "1.23e-4"),
                                      (-44.625, "-44.63"), (2.5e6, "2.50e+6")])
def test_numbers_are_rounded_as_the_page_rounds_them(x, shown):
    assert report.num(x) == shown          # the same as (+x.toPrecision(4)).toString() / toExponential(2) in JS


def test_tiny_p_values_read_naturally():
    assert report.p_is(0.00001) == "P < 0.0001" and report.p_is(0.0123) == "P = 0.0123"


def test_a_subject_with_every_value_empty_still_counts_as_incomplete():
    df = pd.DataFrame({"id": ["a", "b", "c", "d", "e"], "pre": [1, 2, 3, 4, np.nan], "post": [2, 3, 5, 6, np.nan]})
    r = engine.run(df, groups_spec(["pre", "post"], paired=True, subject_col="id"))
    assert r["left_out"]["incomplete_subjects"] == 1 and r["n_subjects"] == 4
