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
    assert "can't tell" in r.answer.lower()


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
    assert "0.05" in buf.getvalue()


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
    assert any("quick look" in c for c in r.caveats)


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
    assert any("Groups are" in i for i in r.review_items)


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
