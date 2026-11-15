"""The recommender's labelled cases and scoring harness (evals/recommender). No network, no API key."""
import pandas as pd
import pytest

from evals.recommender import __main__ as run
from evals.recommender import cases as C
from evals.recommender.metrics import grades, key_of, matches, ndcg_at, recall_at, reciprocal_rank, score_answerable
from evals.recommender.rankers import RANKERS, infeasible

AB = {"analysis_type": "ab_test", "group_col": "variant", "control": "control", "treatment": "new",
      "outcome_col": "purchased"}
OBS = {**AB, "analysis_type": "observational", "covariates": ["device"], "assignment": "chose"}
DID = {"analysis_type": "diff_in_diff", "date_col": "d", "change_date": "", "outcome_col": "signed_up",
       "group_col": "city", "treated": "pilot", "comparison": "holdout", "aggregate": "average"}


@pytest.fixture(scope="module")
def loaded():
    found, _ = run.load()
    return found


# ------------------------------------------------------------------ metrics

def test_key_reads_levels_the_same_for_every_design():
    assert key_of(AB)["treated"] == "new" and key_of(AB)["base"] == "control"
    assert key_of(DID)["treated"] == "pilot" and key_of(DID)["base"] == "holdout"
    assert key_of(DID)["levels"] == frozenset({"pilot", "holdout"})
    assert key_of({"analysis_type": "before_after", "outcome_col": "y", "date_col": "d"})["levels"] is None


def test_matching():
    k = key_of(AB)
    assert matches(k, {"outcome_col": "purchased"})
    assert matches(k, {"levels": ["new", "control"]}) and matches(k, {"levels": ["control", "new"]})
    assert not matches(k, {"treated": "control"})
    assert matches(k, {"analysis_type": ["ab_test", "observational"]})
    assert not matches(k, {"analysis_type": "before_after"})
    assert not matches(k, {"outcome_col": "revenue"})


def test_a_label_is_credited_once():
    """The same comparison under two designs tells the person nothing new the second time."""
    labels = [{"grade": 2, "match": {"outcome_col": "purchased"}}]
    assert grades([AB, OBS], labels) == [2, 0]


def test_each_question_takes_its_best_unused_label():
    labels = [{"grade": 1, "match": {"group_col": "variant"}}, {"grade": 2, "match": {"outcome_col": "purchased"}}]
    assert grades([AB, OBS], labels) == [2, 1]


def test_ranking_metrics():
    assert recall_at([0, 0, 2, 0], 3) == 1.0 and recall_at([0, 0, 0, 2], 3) == 0.0
    assert recall_at([1, 1, 1], 3) == 0.0          # acceptable isn't good enough for the headline number
    assert reciprocal_rank([0, 2]) == 0.5 and reciprocal_rank([1, 0]) == 0.0
    assert ndcg_at([2, 1], [2, 1], 5) == pytest.approx(1.0)
    assert 0 < ndcg_at([1, 2], [2, 1], 5) < 1
    assert ndcg_at([], [2], 5) == 0.0 and ndcg_at([], [], 5) == 0.0


def test_candidate_recall_uses_the_whole_pool():
    revenue = {**AB, "outcome_col": "revenue"}
    labels = [{"grade": 2, "match": {"outcome_col": "purchased"}}]
    s = score_answerable(shown=[revenue], pool=[revenue, AB], labels=labels)
    assert s["candidate_recall"] == 1.0 and s["recall@3"] == 0.0


# ------------------------------------------------------------------ case files

def test_case_files_have_no_problems(loaded):
    assert loaded, "no case files found"
    assert run.check(loaded) == []


def test_problems_are_caught():
    df = pd.DataFrame({"g": ["a", "b"], "y": [0, 1]})
    spec = {"dataset": "t", "renames": {"missing": "x", "y": "g"}, "goals": [
        {"id": "t1", "goal": "q", "paraphrases": ["q1"], "design": "sideways",
         "labels": [{"grade": 3, "match": {"outcome_col": "nope", "group_col": "g", "levels": ["a", "c"], "colour": 1}}]},
        {"id": "t1", "goal": "q", "paraphrases": ["Q", "r", "s"], "unanswerable": "too_hard",
         "labels": [{"grade": 2, "match": {"outcome_col": "y"}, "why": "w"}]},
    ]}
    found = "\n".join(C.problems(spec, df))
    for expected in ("rename of missing column 'missing'", "collides with existing column 'g'", "duplicate id",
                     "needs 3-5 paraphrases", "repeated wording", "unknown design", "unknown reason",
                     "needs at least one grade-2", "only have grade-1", "grade must be 1 or 2", "needs a 'why'",
                     "unknown match fields ['colour']", "column 'nope' isn't in the table", "level 'c' isn't in 'g'"):
        assert expected in found, expected


def test_renamed_cases_follow_the_renamed_columns():
    spec = {"dataset": "t", "renames": {"y": "outcome_value", "g": "arm"}, "goals": [
        {"id": "t1", "goal": "q", "paraphrases": ["a", "b", "c"], "confirmed": {"group_col": "g"},
         "labels": [{"grade": 2, "match": {"outcome_col": ["y", "z"], "group_col": "g", "levels": ["1", "0"]}}]}]}
    cases = C.expand(spec)
    assert len(cases) == 8 and len({c.case_id for c in cases}) == 8
    assert {c.goal_id for c in cases} == {"t1"}
    renamed = [c for c in cases if c.renamed]
    assert renamed[0].labels[0]["match"] == {"outcome_col": ["outcome_value", "z"], "group_col": "arm",
                                             "levels": ["1", "0"]}
    assert renamed[0].confirmed == {"group_col": "arm"}
    assert [c for c in cases if not c.renamed][0].labels[0]["match"]["group_col"] == "g"


def test_every_dataset_has_cases_it_cannot_answer():
    for spec in C.load_files():
        assert any(g.get("unanswerable") for g in spec["goals"]), spec["dataset"]


# ------------------------------------------------------------------ hard rule: nothing infeasible is shown

def test_infeasible_check():
    df = pd.DataFrame({"variant": ["control", "new"] * 10, "purchased": [0, 1] * 10, "device": ["m", "d"] * 10})
    assert infeasible(df, AB) is None
    assert infeasible(df, {**AB, "outcome_col": "missing"})
    assert infeasible(df, {**AB, "treatment": "other"})
    assert infeasible(df, {**OBS, "covariates": []}) == "no background columns to compare people on"
    dated = pd.DataFrame({"d": pd.date_range("2026-01-01", periods=20).astype(str), "y": range(20)})
    assert infeasible(dated, {"analysis_type": "before_after", "date_col": "d", "outcome_col": "y",
                              "change_date": ""}) is None      # the person picks the date later
    assert infeasible(dated, {"analysis_type": "before_after", "date_col": "d", "outcome_col": "y",
                              "change_date": "2030-01-01"})


@pytest.mark.parametrize("name", sorted(RANKERS))
def test_rankers_never_show_anything_infeasible(name, loaded):
    ranker = RANKERS[name]
    for spec, df, df_renamed in loaded:
        for goal in spec["goals"]:
            for table, confirmed in ((df, goal.get("confirmed")),
                                     (df_renamed, C._rename_confirmed(goal.get("confirmed"), spec.get("renames", {})))):
                for s in ranker(table, goal["goal"], goal.get("design"), confirmed).shown:
                    assert infeasible(table, s) is None, (spec["dataset"], goal["id"], s)


def test_saved_result_fingerprints_each_table(tmp_path, monkeypatch, capsys):
    """A score must be traceable to the exact table, including ones that aren't in git."""
    import hashlib
    import json
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    assert run.main(["--dataset", "checkout_test"]) == 0
    saved = json.loads(next(tmp_path.glob("recommender-*.json")).read_text())
    [entry] = saved["datasets"]
    raw = (C.ROOT / "sample_data" / "checkout_test.csv").read_bytes()
    assert entry["dataset"] == "checkout_test" and entry["sha256"] == hashlib.sha256(raw).hexdigest()
    assert entry["rows"] == 6000


def test_harness_runs_end_to_end(capsys):
    assert run.main(["--dataset", "checkout_test", "--no-save"]) == 0
    out = capsys.readouterr().out
    assert "baseline_a" in out and "recall@3" in out and "renamed columns" in out
