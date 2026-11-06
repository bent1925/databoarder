import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import stats

from app.ab_test import analyze
from app.checks import validate_spec_against_data
from app.interpret import unsupported_numbers
from app.main import app
from app.schemas import ABTestSpec
from app.translator import interpret

SPEC = ABTestSpec(group_col="g", control="A", treatment="B", outcome_col="y")


def binary_df(xc, nc, xt, nt):
    return pd.DataFrame({
        "g": ["A"] * nc + ["B"] * nt,
        "y": [1] * xc + [0] * (nc - xc) + [1] * xt + [0] * (nt - xt),
    })


def test_binary_known_answer():
    r = analyze(binary_df(100, 1000, 150, 1000), SPEC)
    assert r.verdict == "clear_difference"
    assert r.effect["diff"] == pytest.approx(0.05)
    assert r.effect["ci_low"] < 0.05 < r.effect["ci_high"]
    assert r.effect["ci_low"] > 0  # excludes zero
    assert r.effect["p_value"] < 0.001
    assert [a.estimate for a in r.arms] == pytest.approx([0.10, 0.15])


def test_binary_null_case():
    r = analyze(binary_df(100, 1000, 102, 1000), SPEC)
    assert r.verdict == "no_clear_difference"
    assert r.effect["ci_low"] < 0 < r.effect["ci_high"]
    assert "close to call" in r.answer.lower()


def test_continuous_matches_scipy_directly():
    rng = np.random.default_rng(1)
    a, b = rng.normal(10, 2, 400), rng.normal(10.6, 2, 450)
    df = pd.DataFrame({"g": ["A"] * 400 + ["B"] * 450, "y": np.r_[a, b]})
    r = analyze(df, SPEC)
    ref = stats.ttest_ind(b, a, equal_var=False)
    ci = ref.confidence_interval(0.95)
    assert r.effect["diff"] == pytest.approx(b.mean() - a.mean())
    assert r.effect["p_value"] == pytest.approx(ref.pvalue)
    assert r.effect["ci_low"] == pytest.approx(ci.low) and r.effect["ci_high"] == pytest.approx(ci.high)
    assert r.outcome_kind == "average"


def test_too_little_data_is_blocked():
    r = analyze(binary_df(2, 10, 5, 10), SPEC)
    assert r.verdict == "blocked"
    assert r.effect is None


def test_spec_must_match_data():
    df = binary_df(10, 100, 10, 100)
    bad = ABTestSpec(group_col="nope", control="A", treatment="B", outcome_col="y")
    assert validate_spec_against_data(df, bad)
    assert analyze(df, bad).verdict == "blocked"
    bad2 = ABTestSpec(group_col="g", control="A", treatment="Z", outcome_col="y")
    assert validate_spec_against_data(df, bad2)


def test_uneven_split_warns():
    r = analyze(binary_df(100, 1000, 100, 2000), SPEC)
    assert any(c.name == "Group balance" and c.status == "warn" for c in r.checks)


def test_answer_numbers_all_trace_to_audit_trail():
    r = analyze(binary_df(100, 1000, 150, 1000), SPEC)
    audit_text = " ".join(a.value for a in r.audit)
    assert unsupported_numbers(r.answer, audit_text) == set()
    assert unsupported_numbers("it rose by 99.9 points", audit_text) == {"99.9"}


def test_script_reproduces_report(tmp_path):
    df = binary_df(100, 1000, 150, 1000)
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    r = analyze(df, SPEC, csv_name=str(csv))
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(r.script, {})
    assert "diff=0.0500" in buf.getvalue()


def test_heuristic_translator_on_sample(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    df = pd.read_csv("sample_data/checkout_test.csv")
    i = interpret("Did the new checkout page increase purchases?", df)
    assert i.source == "heuristic"
    assert (i.spec.group_col, i.spec.control, i.spec.treatment, i.spec.outcome_col) == \
        ("variant", "control", "new_checkout", "purchased")
    assert "new_checkout" in i.readback and "control" in i.readback


def test_api_end_to_end(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = TestClient(app)
    with open("sample_data/checkout_test.csv", "rb") as f:
        up = c.post("/api/upload", files={"file": ("checkout_test.csv", f, "text/csv")}).json()
    ask = c.post("/api/ask", json={"dataset_id": up["dataset_id"], "question": "Did checkout increase purchases?"}).json()
    run = c.post("/api/run", json={"dataset_id": up["dataset_id"], "spec": ask["spec"]})
    assert run.status_code == 200
    body = run.json()
    assert body["verdict"] in {"clear_difference", "no_clear_difference"}
    assert body["audit"] and body["script"]
    assert c.get("/").status_code == 200
    assert c.post("/api/upload", files={"file": ("x.csv", b"not,a\n", "text/csv")}).status_code == 400


# ------------------------------------------------------------------ modes

def sample_spec():
    from app.translator import suggest_questions
    df = pd.read_csv("sample_data/checkout_test.csv")
    return df, ABTestSpec(**suggest_questions(df, "quick")[0]["spec"])


def test_quick_has_caveats_and_no_extras():
    df, spec = sample_spec()
    r = analyze(df, spec, mode="quick")
    assert r.mode == "quick" and r.caveats and not r.robustness and not r.segments
    assert any("Quick look" in c for c in r.caveats)


def test_right_answer_crosschecks_agree_on_clear_effect():
    r = analyze(binary_df(100, 1000, 150, 1000), SPEC, mode="right_answer")
    assert r.verdict == "clear_difference"
    assert len(r.robustness) == 2 and all(x.agrees for x in r.robustness)
    assert r.robust_summary.startswith("Holds up") and r.review_items == []


def test_right_answer_flags_disagreement(monkeypatch):
    import app.modes as modes
    monkeypatch.setattr(modes, "_permutation_p", lambda c, t, seed=0: 0.5)  # force one check to disagree
    r = analyze(binary_df(100, 1000, 150, 1000), SPEC, mode="right_answer")
    assert r.robust_summary.startswith("Not robust")
    assert any("Not robust" in i for i in r.review_items)


def test_right_answer_surfaces_warnings_at_top():
    r = analyze(binary_df(100, 1000, 100, 2000), SPEC, mode="right_answer")  # lopsided split
    assert any("Uneven split" in i for i in r.review_items)


def test_research_scan_corrects_for_many_looks():
    rng = np.random.default_rng(5)  # seed chosen so one noise slice looks 'significant' before correction
    n = 4000
    df = pd.DataFrame({"g": rng.choice(["A", "B"], n), "y": (rng.random(n) < 0.1).astype(int),
                       "country": rng.choice(list("UVWXYZ"), n)})
    # one slice with a real effect
    boost = (df.g == "B") & (df.country == "U") & (rng.random(n) < 0.2)
    df.loc[boost, "y"] = 1
    r = analyze(df, SPEC, mode="research")
    assert len(r.segments) == 6
    top = r.segments[0]
    assert (top.column, top.level) == ("country", "U") and top.flagged
    assert all(s.p_adj >= s.p_raw for s in r.segments)       # correction only ever makes p larger
    assert [s.level for s in r.segments if s.flagged] == ["U"]
    # a noise slice that passes the naive 5% bar but is correctly NOT flagged after correction
    assert any(s.p_raw < 0.05 and not s.flagged for s in r.segments)
    assert "slices" in r.segment_note


def test_null_data_rarely_flags_slices():
    rng = np.random.default_rng(3)
    n = 4000
    df = pd.DataFrame({"g": rng.choice(["A", "B"], n), "y": (rng.random(n) < 0.1).astype(int),
                       "country": rng.choice(list("UVWXYZ"), n), "device": rng.choice(["m", "d"], n)})
    r = analyze(df, SPEC, mode="research")
    assert not any(s.flagged for s in r.segments)


def test_repeat_people_check_needs_an_id_column():
    df = binary_df(100, 1000, 100, 1000)
    assert [c.status for c in analyze(df, SPEC).checks if c.name == "Repeat people"] == ["info"]
    df["user_id"] = np.arange(len(df))
    assert [c.status for c in analyze(df, SPEC).checks if c.name == "Repeat people"] == ["pass"]
    df.loc[df.index[:5], "user_id"] = df.loc[df.index[-5:], "user_id"].to_numpy()  # same people in both groups
    assert [c.status for c in analyze(df, SPEC).checks if c.name == "Repeat people"] == ["warn"]


def test_suggestions_are_valid_and_mode_aware(monkeypatch):
    from app.translator import suggest_questions
    df = pd.read_csv("sample_data/checkout_test.csv")
    qs = suggest_questions(df, "quick")
    assert 2 <= len(qs) <= 4
    assert all(not validate_spec_against_data(df, ABTestSpec(**q["spec"])) for q in qs)
    assert "user_id" not in [q["spec"]["outcome_col"] for q in qs]
    assert suggest_questions(df, "research")[0]["text"].endswith("and for whom?")


def test_api_modes_and_suggestions(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = TestClient(app)
    with open("sample_data/checkout_test.csv", "rb") as f:
        up = c.post("/api/upload", files={"file": ("checkout_test.csv", f, "text/csv")}).json()
    sug = c.get("/api/suggestions", params={"dataset_id": up["dataset_id"], "mode": "research"}).json()["questions"]
    for mode in ("quick", "right_answer", "research"):
        body = c.post("/api/run", json={"dataset_id": up["dataset_id"], "spec": sug[0]["spec"], "mode": mode}).json()
        assert body["mode"] == mode
    assert c.post("/api/run", json={"dataset_id": up["dataset_id"], "spec": sug[0]["spec"], "mode": "bogus"}).status_code == 422


# ------------------------------------------------- correctness (independent re-derivations)

def _wilson(x, n, z=1.959963984540054):
    p = x / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def _newcombe(x1, n1, x2, n2):
    """Newcombe (1998) method 10 for p1 - p2, hand-coded from the paper's formula."""
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = _wilson(x1, n1)
    l2, u2 = _wilson(x2, n2)
    d = p1 - p2
    return d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


@pytest.mark.parametrize("xc,nc,xt,nt", [(100, 1000, 150, 1000), (311, 3047, 346, 2953), (5, 80, 14, 95)])
def test_binary_range_and_p_match_hand_formulas(xc, nc, xt, nt):
    r = analyze(binary_df(xc, nc, xt, nt), SPEC)
    lo, hi = _newcombe(xt, nt, xc, nc)
    assert r.effect["ci_low"] == pytest.approx(lo, abs=1e-6)
    assert r.effect["ci_high"] == pytest.approx(hi, abs=1e-6)
    # pooled two-proportion z-test, written out by hand
    pp = (xc + xt) / (nc + nt)
    z = (xt / nt - xc / nc) / np.sqrt(pp * (1 - pp) * (1 / nc + 1 / nt))
    assert r.effect["p_value"] == pytest.approx(2 * (1 - stats.norm.cdf(abs(z))), abs=1e-9)


def test_welch_range_matches_hand_formula():
    rng = np.random.default_rng(11)
    a, b = rng.gamma(2, 3, 300), rng.gamma(2, 3.4, 350)
    df = pd.DataFrame({"g": ["A"] * 300 + ["B"] * 350, "y": np.r_[a, b]})
    r = analyze(df, SPEC)
    v1, v2 = b.var(ddof=1) / len(b), a.var(ddof=1) / len(a)
    se = np.sqrt(v1 + v2)
    dof = (v1 + v2) ** 2 / (v1 ** 2 / (len(b) - 1) + v2 ** 2 / (len(a) - 1))   # Welch-Satterthwaite
    half = stats.t.ppf(0.975, dof) * se
    d = b.mean() - a.mean()
    assert (r.effect["ci_low"], r.effect["ci_high"]) == pytest.approx((d - half, d + half), abs=1e-9)
    assert r.effect["mde"] == pytest.approx((1.959964 + 0.841621) * se, rel=1e-4)


def test_95_percent_range_has_about_95_percent_coverage():
    """Simulation: the range should catch the true difference ~95% of the time."""
    rng = np.random.default_rng(2024)
    hit, runs, true_diff = 0, 400, 0.03
    for _ in range(runs):
        xc, xt = rng.binomial(600, 0.10), rng.binomial(600, 0.10 + true_diff)
        r = analyze(binary_df(xc, 600, xt, 600), SPEC)
        hit += r.effect["ci_low"] <= true_diff <= r.effect["ci_high"]
    assert 0.92 <= hit / runs <= 0.98


def test_false_alarm_rate_is_about_5_percent_when_nothing_changed():
    rng = np.random.default_rng(99)
    alarms, runs = 0, 400
    for _ in range(runs):
        xc, xt = rng.binomial(800, 0.12), rng.binomial(800, 0.12)
        alarms += analyze(binary_df(xc, 800, xt, 800), SPEC).verdict == "clear_difference"
    assert alarms / runs < 0.08


# ------------------------------------------------------------ reproducibility

def _run_script(script: str) -> str:
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(script, {"__name__": "__repro__"})
    return buf.getvalue()


def _kv(text: str, prefix: str = "") -> dict:
    import re
    out = {}
    for line in text.splitlines():
        if line.startswith("slice ") or prefix and not line.startswith(prefix):
            continue
        out.update({k: float(v) for k, v in re.findall(r"(\w+)=(-?[\d.]+(?:e-?\d+)?)", line)})
    return out


@pytest.mark.parametrize("which", [0, 1])           # 0 = yes/no outcome, 1 = numeric outcome
@pytest.mark.parametrize("mode", ["quick", "right_answer", "research"])
def test_script_reproduces_every_number(which, mode):
    from app.translator import suggest_questions
    path = "sample_data/checkout_test.csv"
    df = pd.read_csv(path)
    spec = ABTestSpec(**suggest_questions(df, mode)[which]["spec"])
    rep = analyze(df, spec, csv_name=path, mode=mode, data_sha="abc123")
    got = _kv(_run_script(rep.script))
    assert got["diff"] == pytest.approx(rep.effect["diff"], abs=1e-4)
    assert got["low"] == pytest.approx(rep.effect["ci_low"], abs=1e-4)
    assert got["high"] == pytest.approx(rep.effect["ci_high"], abs=1e-4)
    assert got["p"] == pytest.approx(rep.effect["p_value"], abs=1e-5)
    for r in rep.robustness:
        key = {"fisher": "fisher_p", "mannwhitney": "mannwhitney_p", "shuffle": "shuffle_p", "trim": "trimmed_p"}[r.key]
        assert got[key] == pytest.approx(r.p_value, abs=1e-5), key
    if mode == "research":
        import re
        lines = [l for l in _run_script(rep.script).splitlines() if l.startswith("slice ")]
        assert len(lines) == len(rep.segments) > 0
        by = {(s.column, s.level): s for s in rep.segments}
        for l in lines:
            col, level = re.match(r"slice (\w+)=(\S+)", l).groups()
            vals = dict(re.findall(r"(\w+)=(-?[\d.]+)", l))
            assert float(vals["p_raw"]) == pytest.approx(by[(col, level)].p_raw, abs=1e-5)
            assert float(vals["p_adj"]) == pytest.approx(by[(col, level)].p_adj, abs=1e-5)
    assert "abc123" in rep.script and rep.repro["data_sha256"] == "abc123"


def test_same_data_gives_identical_report():
    from app.translator import suggest_questions
    df = pd.read_csv("sample_data/checkout_test.csv")
    spec = ABTestSpec(**suggest_questions(df, "quick")[1]["spec"])
    a = analyze(df, spec, mode="right_answer").model_dump()
    b = analyze(df, spec, mode="right_answer").model_dump()
    assert a == b


# ----------------------------------------------------- definitions and assumptions

@pytest.mark.parametrize("mode", ["quick", "right_answer", "research"])
def test_every_number_has_both_definitions_and_every_method_has_assumptions(mode):
    from app.translator import suggest_questions
    df = pd.read_csv("sample_data/checkout_test.csv")
    for q in suggest_questions(df, mode)[:2]:
        rep = analyze(df, ABTestSpec(**q["spec"]), mode=mode)
        assert rep.audit and all(a.technical and a.plain for a in rep.audit)
        assert rep.methods and all(m.assumptions for m in rep.methods)
        assert all(a.status in {"ok", "warn", "manual"} for m in rep.methods for a in m.assumptions)
        assert any("random" in a.text for a in rep.methods[0].assumptions)      # the one we can never check
        assert len(rep.methods) == {"quick": 1, "right_answer": 3 if q["spec"]["outcome_col"] == "purchased" else 4,
                                    "research": 2}[mode]


def test_assumption_status_follows_the_data_checks():
    df = binary_df(100, 1000, 100, 1000)
    df["user_id"] = np.arange(len(df))
    ok = {a.text: a.status for a in analyze(df, SPEC).methods[0].assumptions}
    assert ok["Each row is a different person"] == "ok"
    df.loc[df.index[:5], "user_id"] = df.loc[df.index[-5:], "user_id"].to_numpy()
    bad = {a.text: a.status for a in analyze(df, SPEC).methods[0].assumptions}
    assert bad["Each row is a different person"] == "warn"
    tiny = {a.text: a.status for a in analyze(binary_df(3, 100, 40, 100), SPEC).methods[0].assumptions}
    assert tiny["At least 10 yes and 10 no in each group"] == "warn"


def test_uneven_missing_is_flagged():
    df = binary_df(100, 1000, 100, 1000).astype({"y": float})
    df.loc[df.index[:150], "y"] = np.nan          # all the missing rows are in group A
    r = analyze(df, SPEC)
    assert any(c.name == "Missing balance" and c.status == "warn" for c in r.checks)
    assert {a.text: a.status for a in r.methods[0].assumptions}["Missing results don't depend on the version"] == "warn"


def test_slices_never_use_numeric_result_columns():
    df = pd.read_csv("sample_data/checkout_test.csv")
    rep = analyze(df, ABTestSpec(group_col="variant", control="control", treatment="new_checkout",
                                 outcome_col="revenue"), mode="research")
    assert {s.column for s in rep.segments} == {"country", "device"}      # not purchased / sessions
    assert all(np.isfinite(s.p_raw) and np.isfinite(s.p_adj) for s in rep.segments)
