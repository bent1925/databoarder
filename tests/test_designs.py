import contextlib
import io
import re

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.analysis import analyze
from app.main import app
from app.observational import covariate_candidates
from app.schemas import ABTestSpec, DiDSpec, ObservationalSpec
from app.translator import design_options, interpret, suggest_questions

ROLL = pd.read_csv("sample_data/regional_rollout.csv")
LOY = pd.read_csv("sample_data/loyalty_program.csv")
CHANGE = "2026-08-30"


def did_spec(outcome="signed_up", agg="average"):
    return DiDSpec(date_col="visit_date", outcome_col=outcome, change_date=CHANGE, group_col="city_group",
                   treated="pilot", comparison="holdout", aggregate=agg)


def obs_spec(outcome="spend_next_90d", covs=None):
    s = ObservationalSpec(group_col="joined_program", control="no", treatment="yes", outcome_col=outcome)
    return s.model_copy(update={"covariates": covs} if covs is not None else {})


# ------------------------------------------------------------- difference-in-differences

def test_did_range_covers_planted_effect_and_naive_before_after_overstates():
    rep = analyze(ROLL, did_spec())
    e = rep.effect
    assert e["ci_low"] * 100 <= 2.4 <= e["ci_high"] * 100          # planted extra effect: +2.4 pts
    pilot = ROLL[ROLL.city_group == "pilot"]
    naive = pilot[pilot.visit_date >= CHANGE].signed_up.mean() - pilot[pilot.visit_date < CHANGE].signed_up.mean()
    assert naive * 100 > e["ci_high"] * 100 - 0.5 or naive * 100 > 3.8   # shared +2 pts shock inflates the pilot-only view
    assert rep.analysis == "diff_in_diff" and rep.design == "one_date_comparison"


def test_did_flags_slices_in_research_mode():
    rep = analyze(ROLL, did_spec(), mode="research")
    assert any(s.flagged and s.column == "device" for s in rep.segments)


@pytest.mark.parametrize("mode", ["quick", "right_answer", "research"])
@pytest.mark.parametrize("which", ["did", "obs"])
def test_scripts_reproduce_the_report(which, mode, tmp_path):
    df, spec, name = (ROLL, did_spec(), "r.csv") if which == "did" else (LOY, obs_spec(), "l.csv")
    df.to_csv(tmp_path / name, index=False)
    rep = analyze(df, spec, csv_name=str(tmp_path / name), mode=mode)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(rep.script, "repro", "exec"), {"__name__": "__main__"})
    out = buf.getvalue()
    nums = {}
    for k, v in re.findall(r"(?<![\w_])(diff|low|high)=(-?[\d.]+)", out):
        nums.setdefault(k, v)
    assert abs(float(nums["diff"]) - rep.effect["diff"]) < 1e-3
    assert abs(float(nums["low"]) - rep.effect["ci_low"]) < 1e-3
    assert abs(float(nums["high"]) - rep.effect["ci_high"]) < 1e-3


# ------------------------------------------------------------------------ observational

def test_adjusted_gap_is_far_below_raw_gap_and_never_says_raised():
    rep = analyze(LOY, obs_spec(), mode="right_answer")
    raw = LOY[LOY.joined_program == "yes"].spend_next_90d.mean() - LOY[LOY.joined_program == "no"].spend_next_90d.mean()
    assert 0 < rep.effect["diff"] < raw * 0.5
    assert rep.effect["ci_low"] <= 5.0 + 5 and rep.effect["ci_high"] >= 5.0          # planted +$5 within reach
    text = (rep.headline + rep.answer + (rep.causal_label or "")).lower()
    assert "goes with" in text and "raised" not in text
    assert "cause" in (rep.causal_label or "").lower()
    assert rep.adjusted_for
    assert any("hidden" in c.name.lower() or "hidden" in c.message.lower() for c in rep.checks)   # E-value shown


def test_post_choice_columns_are_not_ticked_by_default():
    cands = {c["name"]: c for c in covariate_candidates(LOY, obs_spec("spend_next_90d"))}
    assert "renewed" in cands and not cands["renewed"]["checked"]
    assert {"region", "device", "age"} <= {n for n, c in cands.items() if c["checked"]}
    rep = analyze(LOY, obs_spec())
    assert "renewed" not in rep.adjusted_for


def test_balance_check_warns_when_random_is_claimed_for_self_selected_data():
    spec = ABTestSpec(group_col="joined_program", control="no", treatment="yes", outcome_col="spend_next_90d")
    rep = analyze(LOY, spec)
    bal = [c for c in rep.checks if c.name == "Groups alike"]
    assert bal and bal[0].status in ("warn", "fail")


# ----------------------------------------------------------------------- design routing

def test_design_options_defaults_and_availability():
    d = design_options(LOY)
    assert d["default"] == "chose"
    ok = {o["key"]: o["ok"] for o in d["options"]}
    assert ok["random"] and not ok["one_date"] and not ok["one_date_comparison"]
    assert design_options(pd.read_csv("sample_data/signup_page_change.csv"))["default"] == "one_date"


@pytest.mark.parametrize("design,kind", [("random", "ab_test"), ("chose", "observational"), ("unsure", "observational"),
                                         ("one_date", "before_after"), ("one_date_comparison", "diff_in_diff")])
def test_design_decides_the_method(design, kind, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert interpret("did it work?", ROLL, design).spec.analysis_type == kind
    assert all(q["spec"]["analysis_type"] == kind for q in suggest_questions(ROLL, "quick", design=design))


def test_unsure_is_treated_as_chose():
    spec = interpret("x", LOY, "unsure").spec
    assert spec.assignment == "unsure"
    assert analyze(LOY, spec).analysis == "observational"


def test_suggested_outcomes_for_chose_come_after_the_choice():
    texts = [q["text"] for q in suggest_questions(LOY, "quick", design="chose")]
    assert texts and "past spend" not in texts[0] and "next 90d" in texts[0]


def test_api_design_flow():
    c = TestClient(app)
    with open("sample_data/loyalty_program.csv", "rb") as f:
        up = c.post("/api/upload", files={"file": ("l.csv", f, "text/csv")}).json()
    assert up["designs"]["default"] == "chose"
    ds = up["dataset_id"]
    bad = c.post("/api/ask", json={"dataset_id": ds, "question": "q", "design": "one_date"})
    assert bad.status_code == 400
    ok = c.post("/api/ask", json={"dataset_id": ds, "question": "q", "design": "chose"}).json()
    assert ok["spec"]["analysis_type"] == "observational" and ok["spec"]["covariates"]
    cov = c.get("/api/covariates", params=dict(dataset_id=ds, group_col="joined_program", control="no",
                                               treatment="yes", outcome_col="spend_next_90d")).json()["columns"]
    assert any(x["name"] == "region" and x["checked"] for x in cov)
    r = c.post("/api/run", json={"dataset_id": ds, "spec": ok["spec"], "mode": "quick"}).json()
    assert r["analysis"] == "observational" and r["design"] == "chose"
    q = c.get("/api/suggestions", params=dict(dataset_id=ds, mode="quick", design="chose")).json()["questions"]
    assert q


# ------------------------------------------- LaLonde benchmark (known randomized answer ~ +$1,794)

def test_a_copy_of_the_group_column_is_never_offered_for_comparing_people():
    d = LOY.copy()
    d["joined_copy"] = (d.joined_program == "yes").astype(int)
    assert "joined_copy" not in {c["name"] for c in covariate_candidates(d, obs_spec())}


def test_lalonde_observational_is_honest_about_poor_overlap():
    import os
    path = "sample_data/lalonde_cps.csv"
    if not os.path.exists(path):
        pytest.skip("run tools/make_lalonde_sample.py (needs causaldata)")
    df = pd.read_csv(path)
    spec = ObservationalSpec(group_col="trained", control="no", treatment="yes", outcome_col="re78")
    assert not any(c["name"] == "trained" for c in covariate_candidates(df, spec))
    rep = analyze(df, spec.model_copy(update={"covariates": ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]}),
                  mode="right_answer")
    raw = df[df.trained == "yes"].re78.mean() - df[df.trained == "no"].re78.mean()
    assert raw < -5000                                          # naive comparison says training HURT earnings
    assert rep.effect["ci_low"] <= 1794 <= rep.effect["ci_high"]   # adjusted range reaches the randomized answer
    assert rep.verdict != "clear_difference" or rep.effect["diff"] > 0   # never a confident wrong-signed claim
    assert rep.review_items                                       # and it warns: overlap is poor / checks disagree


def test_poor_overlap_is_said_up_front():
    import os
    if not os.path.exists("sample_data/lalonde_cps.csv"):
        pytest.skip("sample missing")
    df = pd.read_csv("sample_data/lalonde_cps.csv")
    spec = ObservationalSpec(group_col="trained", control="no", treatment="yes", outcome_col="re78",
                             covariates=["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"])
    assert "truly alike" in analyze(df, spec).how_sure
    assert "truly alike" not in analyze(LOY, obs_spec()).how_sure


@pytest.mark.parametrize("sample,default", [("checkout_test", "random"), ("signup_page_change", "one_date"),
                                            ("regional_rollout", "one_date_comparison"), ("loyalty_program", "chose")])
def test_design_default_per_sample(sample, default):
    assert design_options(pd.read_csv(f"sample_data/{sample}.csv"))["default"] == default


def test_offline_translator_passes_the_eval_cases(monkeypatch):
    """The keyword fallback must also get the eval questions right (the real-model run is evals/translator_cases.py)."""
    from evals.translator_cases import CASES
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for f, design, q, want in CASES:
        got = interpret(q, pd.read_csv(f"sample_data/{f}.csv"), design).spec.model_dump()
        assert {k: got.get(k) for k in want} == want, q
