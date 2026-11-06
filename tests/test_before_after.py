import contextlib
import io
import re

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import stats

from app.analysis import analyze
from app.before_after import daily_series, make_blocks
from app.main import app
from app.modes import effect
from app.schemas import BeforeAfterSpec
from app.translator import interpret, suggest_questions

SAMPLE = "sample_data/signup_page_change.csv"
CHANGE = "2026-08-30"


def spec(outcome="signed_up", change=CHANGE, agg="average"):
    return BeforeAfterSpec(date_col="visit_date", outcome_col=outcome, change_date=change, aggregate=agg)


def daily_df(values, start="2026-01-01", name="y"):
    return pd.DataFrame({"day": pd.date_range(start, periods=len(values)).strftime("%Y-%m-%d"), name: values})


def dspec(change, name="y"):
    return BeforeAfterSpec(date_col="day", outcome_col=name, change_date=change)


# ------------------------------------------------------------ hand-coded correctness

def test_numbers_match_a_hand_calculation():
    rng = np.random.default_rng(3)
    vals = rng.normal(100, 5, 84) + np.r_[np.zeros(42), np.full(42, 8.0)]
    df = daily_df(vals)
    rep = analyze(df, dspec("2026-02-12"))                 # day 42 = first "after" day
    wk = vals.reshape(12, 7).mean(axis=1)
    b, a = wk[:6], wk[6:]
    diff = a.mean() - b.mean()
    se = np.sqrt(b.var(ddof=1) / 6 + a.var(ddof=1) / 6)
    dfree = se ** 4 / ((b.var(ddof=1) / 6) ** 2 / 5 + (a.var(ddof=1) / 6) ** 2 / 5)   # Welch-Satterthwaite
    h = stats.t.ppf(0.975, dfree) * se
    p = 2 * stats.t.sf(abs(diff / se), dfree)
    e = rep.effect
    assert e["diff"] == pytest.approx(diff) and e["ci_low"] == pytest.approx(diff - h)
    assert e["ci_high"] == pytest.approx(diff + h) and e["p_value"] == pytest.approx(p)
    assert e["mde"] == pytest.approx(2.8016 * se, rel=1e-4)
    assert rep.arms[0].estimate == pytest.approx(b.mean()) and rep.arms[1].estimate == pytest.approx(a.mean())


def test_weeks_are_counted_outward_from_the_date_and_skip_gaps():
    vals = np.arange(60, dtype=float)
    df = daily_df(vals).drop(index=[10])                    # a missing day inside one week
    blocks = make_blocks(daily_series(df, dspec("2026-01-31")), pd.Timestamp("2026-01-31"))
    assert len(blocks.after) == 4 and len(blocks.before) == 3 and blocks.skipped_before == 1
    assert blocks.after[0] == pytest.approx(np.arange(30, 37).mean())
    assert blocks.before[-1] == pytest.approx(np.arange(23, 30).mean())


def test_weekday_swings_cannot_fake_a_change():
    # Weekends are +40 every single week. The after-period starts mid-week, so a naive
    # day-count average would be biased; whole weeks cancel it exactly.
    wd = pd.date_range("2026-01-01", periods=98).weekday
    vals = 100 + np.where(wd >= 5, 40, 0).astype(float)
    rep = analyze(daily_df(vals), dspec("2026-02-18"))     # a Wednesday
    assert rep.effect["diff"] == pytest.approx(0, abs=1e-9)
    assert rep.verdict == "no_clear_difference"


def test_total_vs_average_aggregation():
    rows = []
    for i, day in enumerate(pd.date_range("2026-01-01", periods=70)):
        n = 3 if i < 35 else 6                              # twice as many orders after, same size
        rows += [{"day": day.strftime("%Y-%m-%d"), "amount": 10.0} for _ in range(n)]
    df = pd.DataFrame(rows)
    avg = analyze(df, BeforeAfterSpec(date_col="day", outcome_col="amount", change_date="2026-02-05", aggregate="average"))
    tot = analyze(df, BeforeAfterSpec(date_col="day", outcome_col="amount", change_date="2026-02-05", aggregate="total"))
    assert avg.effect["diff"] == pytest.approx(0, abs=1e-9)
    assert tot.effect["diff"] == pytest.approx(30.0) and tot.outcome_kind == "total"


# -------------------------------------------------------------- statistical honesty

def _ar1(rng, n, rho):
    e = np.zeros(n)
    e[0] = rng.normal()
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal() * np.sqrt(1 - rho ** 2)
    return e


@pytest.mark.parametrize("rho", [0.0, 0.4])
def test_ranges_cover_the_truth_about_95_percent_even_when_days_are_linked(rho):
    rng = np.random.default_rng(11)
    hit = clear = 0
    runs = 400
    for _ in range(runs):
        wd = np.arange(84) % 7
        y = 5 * np.isin(wd, [5, 6]) + _ar1(rng, 84, rho) + np.r_[np.zeros(42), np.full(42, 0.0)]
        rep = analyze(daily_df(y), dspec("2026-02-12"))
        hit += rep.effect["ci_low"] <= 0 <= rep.effect["ci_high"]
        clear += rep.verdict == "clear_difference"
    assert hit / runs > 0.92
    assert clear / runs < 0.08


def test_real_change_is_usually_found():
    rng = np.random.default_rng(5)
    found = 0
    for _ in range(100):
        y = _ar1(rng, 112, 0.3) + np.r_[np.zeros(56), np.full(56, 1.2)]
        found += analyze(daily_df(y), dspec("2026-02-26")).verdict == "clear_difference"
    assert found >= 90


def test_rising_trend_is_flagged_and_fake_date_test_catches_it():
    rng = np.random.default_rng(2)
    y = np.arange(112) * 0.3 + rng.normal(0, 1, 112)       # steady climb, no jump at all
    rep = analyze(daily_df(y), dspec("2026-02-26"), mode="right_answer")
    assert any(c.name == "Steady before" and c.status == "warn" for c in rep.checks)
    by = {r.key: r for r in rep.robustness}
    assert by["placebo"].agrees is False                    # "found" a change at a fake date
    assert "Not robust" in rep.robust_summary


def test_trend_adjustment_rescues_a_real_jump_on_top_of_a_trend():
    rng = np.random.default_rng(4)
    y = np.arange(112) * 0.05 + rng.normal(0, 0.5, 112) + np.r_[np.zeros(56), np.full(56, 6.0)]
    rep = analyze(daily_df(y), dspec("2026-02-26"), mode="right_answer")
    its = {r.key: r for r in rep.robustness}["its"]
    assert its.agrees and not its.weak


def test_planted_effects_in_the_sample_are_recovered():
    df = pd.read_csv(SAMPLE)
    rep = analyze(df, spec(), mode="research")
    assert rep.verdict == "clear_difference" and 0.012 < rep.effect["diff"] < 0.03     # truth: ~+1.8 pts
    flagged = {(s.column, s.level) for s in rep.segments if s.flagged}
    assert ("device", "mobile") in flagged                  # biggest planted effect
    assert not any(c == "channel" for c, _ in flagged)      # no planted effect there


def test_same_data_gives_identical_report():
    df = pd.read_csv(SAMPLE)
    a = analyze(df, spec(), mode="right_answer").model_dump()
    assert a == analyze(df, spec(), mode="right_answer").model_dump()


# --------------------------------------------------------------------- guard rails

@pytest.mark.parametrize("change, text", [("", "Pick the date"), ("2025-01-01", "outside your data"),
                                          ("2027-01-01", "outside your data")])
def test_bad_dates_are_blocked_with_a_reason(change, text):
    rep = analyze(pd.read_csv(SAMPLE), spec(change=change))
    assert rep.verdict == "blocked" and text in rep.answer


def test_too_few_weeks_is_blocked():
    rep = analyze(pd.read_csv(SAMPLE), spec(change="2026-07-10"))
    assert rep.verdict == "blocked" and "Need 3+" in rep.answer


def test_wrong_columns_are_blocked():
    df = pd.read_csv(SAMPLE)
    assert analyze(df, spec(outcome="device")).verdict == "blocked"
    assert analyze(df, BeforeAfterSpec(date_col="device", outcome_col="signed_up", change_date=CHANGE)).verdict == "blocked"
    assert analyze(df, BeforeAfterSpec(date_col="nope", outcome_col="signed_up", change_date=CHANGE)).verdict == "blocked"


def test_sample_checks_see_the_two_outage_days():
    rep = analyze(pd.read_csv(SAMPLE), spec())
    c = {c.name: c for c in rep.checks}
    assert "2 days" in c["Missing days"].message


def test_quick_mode_lists_caveats_and_does_not_cross_check():
    rep = analyze(pd.read_csv(SAMPLE), spec(), mode="quick")
    assert rep.caveats and not rep.robustness and not rep.segments


# ----------------------------------------------------------------- reproducibility

def _run_script(script: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(script, {"__name__": "__repro__"})
    return buf.getvalue()


@pytest.mark.parametrize("outcome", ["signed_up", "pages_viewed"])
@pytest.mark.parametrize("mode", ["quick", "right_answer", "research"])
def test_script_reproduces_every_number(outcome, mode):
    df = pd.read_csv(SAMPLE)
    rep = analyze(df, spec(outcome), csv_name=SAMPLE, mode=mode, data_sha="abc123")
    text = _run_script(rep.script)
    got = {}
    for line in text.splitlines():
        if not line.startswith("slice "):
            got.update({k: float(v) for k, v in re.findall(r"(\w+)=(-?[\d.]+(?:e-?\d+)?)", line)})
    e = rep.effect
    assert got["diff"] == pytest.approx(e["diff"], abs=1e-4) and got["low"] == pytest.approx(e["ci_low"], abs=1e-4)
    assert got["high"] == pytest.approx(e["ci_high"], abs=1e-4) and got["p"] == pytest.approx(e["p_value"], abs=1e-5)
    assert got["before_mean"] == pytest.approx(rep.arms[0].estimate, abs=1e-4)
    assert got["after_mean"] == pytest.approx(rep.arms[1].estimate, abs=1e-4)
    key = {"placebo": "placebo_p", "its": "its_p", "window": "window_p", "daily": "daily_p"}
    for r in rep.robustness:
        assert got[key[r.key]] == pytest.approx(r.p_value, abs=1e-5), r.key
    if mode == "research":
        lines = [l for l in text.splitlines() if l.startswith("slice ")]
        assert len(lines) == len(rep.segments) > 0
        by = {(s.column, s.level): s for s in rep.segments}
        for l in lines:
            col, level = re.match(r"slice (\w+)=(\S+)", l).groups()
            vals = dict(re.findall(r"(\w+)=(-?[\d.]+)", l))
            assert float(vals["p_raw"]) == pytest.approx(by[(col, level)].p_raw, abs=1e-5)
            assert float(vals["p_adj"]) == pytest.approx(by[(col, level)].p_adj, abs=1e-5)
    assert "abc123" in rep.script


def test_script_reproduces_total_aggregation():
    rows = [{"day": d.strftime("%Y-%m-%d"), "amount": float(1 + (i % 3))}
            for i, d in enumerate(pd.date_range("2026-01-01", periods=70)) for _ in range(2)]
    pd.DataFrame(rows).to_csv("/tmp/_total.csv", index=False)
    df = pd.read_csv("/tmp/_total.csv")
    s = BeforeAfterSpec(date_col="day", outcome_col="amount", change_date="2026-02-05", aggregate="total")
    rep = analyze(df, s, csv_name="/tmp/_total.csv")
    got = dict(re.findall(r"(\w+)=(-?[\d.]+)", _run_script(rep.script)))
    assert float(got["after_mean"]) == pytest.approx(rep.arms[1].estimate, abs=1e-4)


# ----------------------------------------------- definitions, assumptions, the front door

@pytest.mark.parametrize("mode", ["quick", "right_answer", "research"])
def test_every_number_has_both_definitions_and_every_method_has_assumptions(mode):
    rep = analyze(pd.read_csv(SAMPLE), spec(), mode=mode)
    assert rep.audit and all(a.technical and a.plain for a in rep.audit)
    assert rep.methods and all(m.assumptions for m in rep.methods)
    assert any(a.status == "manual" for a in rep.methods[0].assumptions)
    if mode == "right_answer":
        assert {m.name for m in rep.methods} >= {"Fake-date test", "Allowing for a trend", "Nearby weeks only", "Day-by-day model"}


def test_translator_reads_dates_from_the_question_without_inventing_any(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    df = pd.read_csv(SAMPLE)
    i = interpret("Did signups change after Aug 30?", df)
    assert i.spec.analysis_type == "before_after" and i.spec.change_date == CHANGE
    j = interpret("What happened to signups since the redesign?", df)
    assert j.spec.analysis_type == "before_after" and j.spec.change_date == ""      # never guessed
    k = interpret("Did the new checkout page increase purchases?", pd.read_csv("sample_data/checkout_test.csv"))
    assert k.spec.analysis_type == "ab_test"


def test_suggestions_match_the_file():
    ba = suggest_questions(pd.read_csv(SAMPLE), "quick")
    assert ba and all(q["spec"]["analysis_type"] == "before_after" and q["needs_date"] for q in ba)
    ab = suggest_questions(pd.read_csv("sample_data/checkout_test.csv"), "quick")
    assert ab and all(q["spec"]["analysis_type"] == "ab_test" for q in ab)


def test_api_runs_both_kinds_and_old_style_specs_still_work():
    c = TestClient(app)
    up = c.post("/api/upload", files={"file": ("s.csv", open(SAMPLE, "rb"), "text/csv")}).json()
    assert [x["type"] for x in up["columns"]][0] == "date" and up["columns"][0]["rows_per_day"] > 1
    body = {"dataset_id": up["dataset_id"], "mode": "right_answer", "spec": spec().model_dump()}
    r = c.post("/api/run", json=body).json()
    assert r["analysis"] == "before_after" and r["series"]["change_date"] == CHANGE and r["robustness"]
    assert "days before" in c.post("/api/readback", json=body).json()["readback"]
    ck = c.post("/api/upload", files={"file": ("c.csv", open("sample_data/checkout_test.csv", "rb"), "text/csv")}).json()
    old = {"dataset_id": ck["dataset_id"], "spec": {"group_col": "variant", "control": "control",
                                                    "treatment": "new_checkout", "outcome_col": "purchased"}}
    assert c.post("/api/run", json=old).json()["analysis"] == "ab_test"
