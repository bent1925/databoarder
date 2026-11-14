"""The routes for putting several files together, from upload through to a report."""
import importlib.util
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main, sources
from app.schemas import ABTestSpec, ObservationalSpec

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("test_ingest_helpers", ROOT / "tests" / "test_ingest.py")
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)
tool = helpers.tool

client = TestClient(main.app)
PEOPLE = "id,group\nC-1,a\nC-2,b\nC-3,a\nC-4,b\n"
ORDERS = "id,amount\n1,5\n1,6\n2,7\n"


def csv_file(name, text):
    return ("files", (name, text.encode(), "text/csv"))


def add(*files):
    response = client.post("/api/sources", files=list(files))
    assert response.status_code == 200, response.text
    return response.json()


def confirmed(steps):
    return [{**step, "status": "confirmed"} for step in steps]


def people_steps(reads):
    return reads + [
        {"op": "clean_key", "table": "people", "column": "id", "rule": "digits_only", "into": "pid"},
        {"op": "roll_up", "table": "orders", "by": "id", "into": "per",
         "measures": [{"how": "sum", "column": "amount", "into": "spend"}]},
        {"op": "match", "left": "people", "right": "per", "left_key": "pid", "right_key": "id", "bring": {"spend": "spend"}},
        {"op": "fill_missing", "table": "people", "column": "spend", "value": "0"},
    ]


def started():
    got = add(csv_file("people.csv", PEOPLE), csv_file("orders.csv", ORDERS))
    return got["source_id"], got["steps"]


def confirm(source_id, steps, output="people", unit="pid", group="group", time_column=None):
    return client.post(f"/api/sources/{source_id}/confirm", json={
        "recipe": {"steps": steps, "output": output}, "unit_column": unit, "group_column": group, "time_column": time_column})


def test_upload_reports_tables_and_suggests_read_steps_unconfirmed():
    got = add(csv_file("people.csv", PEOPLE), csv_file("orders.csv", ORDERS))
    assert [t["rows"] for t in got["tables"]] == [4, 3]
    assert [s["into"] for s in got["steps"]] == ["people", "orders"]
    assert {s["status"] for s in got["steps"]} == {"proposed"}


def test_folders_are_stripped_from_file_names():
    got = add(csv_file("../../evil/people.csv", PEOPLE))
    assert got["steps"][0]["file"] == "people.csv"


@pytest.mark.parametrize("files,message", [
    ([csv_file("a.exe", "x")], "isn't a type"),
    ([csv_file("a.csv", PEOPLE), csv_file("a.csv", PEOPLE)], "More than one file"),
])
def test_bad_uploads_are_explained(files, message):
    response = client.post("/api/sources", files=files)
    assert response.status_code == 400 and message in response.json()["detail"]


def test_too_many_or_too_large_is_explained(monkeypatch):
    monkeypatch.setattr(sources, "MAX_FILES", 1)
    response = client.post("/api/sources", files=[csv_file("a.csv", PEOPLE), csv_file("b.csv", PEOPLE)])
    assert "the limit is 1" in response.json()["detail"]
    monkeypatch.setattr(sources, "MAX_FILES", 20)
    monkeypatch.setattr(sources, "MAX_SOURCE_BYTES", 5)
    response = client.post("/api/sources", files=[csv_file("a.csv", PEOPLE)])
    assert "larger than" in response.json()["detail"]


def test_unreadable_excel_is_explained():
    pytest.importorskip("openpyxl")
    response = client.post("/api/sources", files=[("files", ("a.xlsx", b"not a workbook", "application/octet-stream"))])
    assert response.status_code == 400 and "Couldn't read one of those files" in response.json()["detail"]


def test_preview_works_before_confirming_and_saves_nothing():
    source_id, reads = started()
    before = len(main.DATASETS)
    response = client.post(f"/api/sources/{source_id}/preview",
                           json={"recipe": {"steps": people_steps(reads), "output": "people"}})
    body = response.json()
    assert response.status_code == 200 and body["error"] is None
    assert body["rows"] == 4 and body["columns"] == ["id", "group", "pid", "spend"]
    assert body["texts"][0].startswith("Read people.csv")
    assert "orders" in body["tables"] and len(main.DATASETS) == before


def test_preview_says_which_step_failed_and_keeps_earlier_work():
    source_id, reads = started()
    bad = reads + [{"op": "fill_missing", "table": "people", "column": "nope", "value": "0"}]
    body = client.post(f"/api/sources/{source_id}/preview", json={"recipe": {"steps": bad, "output": "people"}}).json()
    assert body["failed_step"] == 3 and "no column called 'nope'" in body["error"]
    assert body["sample"] is None and body["tables"]["people"] == ["id", "group"] and len(body["texts"]) == 3


def test_incomplete_step_gets_a_plain_message():
    source_id, reads = started()
    bad = reads + [{"op": "fill_missing", "table": "people", "column": "group"}]
    response = client.post(f"/api/sources/{source_id}/preview", json={"recipe": {"steps": bad, "output": "people"}})
    assert response.status_code == 400 and response.json()["detail"] == "Step 3: 'value' needs a valid value."


def test_unexpected_failure_inside_a_step_is_contained(monkeypatch):
    from app import recipe_ops

    def boom(*args):
        raise KeyError("x")

    monkeypatch.setitem(recipe_ops.OPERATIONS, "stack", boom)
    source_id, reads = started()
    steps = reads + [{"op": "stack", "tables": ["people"], "into": "t"}]
    body = client.post(f"/api/sources/{source_id}/preview", json={"recipe": {"steps": steps, "output": "t"}}).json()
    assert body["failed_step"] == 3 and "couldn't be carried out" in body["error"]


def test_confirm_refuses_unconfirmed_steps():
    source_id, reads = started()
    response = confirm(source_id, people_steps(reads))
    assert response.status_code == 400 and "still need your confirmation" in response.json()["detail"]


def test_confirm_refuses_repeated_units():
    source_id, reads = started()
    response = confirm(source_id, confirmed(reads), output="orders", unit="id", group="amount")
    assert response.status_code == 400 and "isn't one row per 'id'" in response.json()["detail"]
    assert "roll-up" in response.json()["detail"]


def test_confirm_needs_a_way_to_compare():
    source_id, reads = started()
    response = confirm(source_id, confirmed(people_steps(reads)), group=None)
    assert response.status_code == 400 and "who got the campaign" in response.json()["detail"]


def test_confirm_stops_when_everyone_is_in_one_group():
    got = add(csv_file("people.csv", "id,group\n1,a\n2,a\n"))
    response = confirm(got["source_id"], confirmed(got["steps"]), unit="id")
    assert response.status_code == 400 and "no one to compare" in response.json()["detail"]


def test_confirm_stops_when_time_has_only_one_date():
    got = add(csv_file("people.csv", "id,day,v\n1,2024-01-01,3\n2,2024-01-01,4\n"))
    response = confirm(got["source_id"], confirmed(got["steps"]), unit="id", group=None, time_column="day")
    assert response.status_code == 400 and "two different dates" in response.json()["detail"]


def test_confirm_accepts_dates_instead_of_groups():
    got = add(csv_file("days.csv", "day,v\n2024-01-01,3\n2024-01-02,4\n"))
    response = confirm(got["source_id"], confirmed(got["steps"]), output="days", unit="day", group=None, time_column="day")
    assert response.status_code == 200 and response.json()["source"]["time_column"] == "day"


def test_oversized_final_table_is_explained(monkeypatch):
    monkeypatch.setattr(main, "MAX_BYTES", 5)
    source_id, reads = started()
    response = confirm(source_id, confirmed(people_steps(reads)))
    assert response.status_code == 413 and "larger than 25 MB" in response.json()["detail"]


def test_confirmed_table_downloads_exactly_what_was_fingerprinted():
    source_id, reads = started()
    result = confirm(source_id, confirmed(people_steps(reads))).json()
    download = client.get(f"/api/datasets/{result['dataset_id']}/snapshot")
    assert download.status_code == 200
    assert main.hashlib.sha256(download.content).hexdigest() == result["source"]["table_sha256"]
    assert main.DATASETS[result["dataset_id"]]["sha"] == result["source"]["table_sha256"]


def test_plain_uploads_have_no_snapshot():
    response = client.post("/api/upload", files={"file": ("a.csv", PEOPLE.encode(), "text/csv")})
    assert client.get(f"/api/datasets/{response.json()['dataset_id']}/snapshot").status_code == 404


def test_gone_files_get_a_way_forward():
    response = client.post("/api/sources/nope/preview", json={"recipe": {"steps": [
        {"op": "stack", "tables": ["a"], "into": "b"}], "output": "b"}})
    assert response.status_code == 404 and "upload them again" in response.json()["detail"]


def test_old_folders_are_deleted(monkeypatch):
    store = sources.SourceStore()
    source_id = store.add([("a.csv", b"x,y\n1,2\n")])
    folder = store.folder(source_id)
    monkeypatch.setattr(sources.time, "time", lambda: time.monotonic() + 1e12)
    with pytest.raises(sources.RecipeError, match="upload them again"):
        store.folder(source_id)
    assert not folder.exists()


def test_oldest_folders_are_pushed_out(monkeypatch):
    monkeypatch.setattr(sources, "MAX_SOURCES", 2)
    store = sources.SourceStore()
    first = store.add([("a.csv", b"x\n")])
    folder = store.folder(first)
    store.add([("a.csv", b"x\n")])
    store.add([("a.csv", b"x\n")])
    assert not folder.exists()


def _run_on(dataset_id, outcome="spend"):
    spec = ABTestSpec(group_col="group", control="a", treatment="b", outcome_col=outcome).model_dump()
    return client.post("/api/run", json={"dataset_id": dataset_id, "spec": spec, "mode": "quick"})


@pytest.fixture
def hillstrom_pull(tmp_path):
    pytest.importorskip("openpyxl")
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(5)
    n = 400
    visit = rng.random(n) < 0.4
    bought = visit & (rng.random(n) < 0.5)
    answer = tool.answer_table(pd.DataFrame({
        "recency": rng.integers(1, 13, n), "history_segment": rng.choice(["1) $0 - $100", "2) $100 - $200"], n),
        "history": rng.integers(3000, 90000, n) / 100, "mens": rng.integers(0, 2, n), "womens": rng.integers(0, 2, n),
        "zip_code": rng.choice(["Urban", "Rural"], n), "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(["Web", "Phone"], n), "segment": rng.choice(["Mens E-Mail", "No E-Mail"], n),
        "visit": visit.astype(int), "conversion": bought.astype(int),
        "spend": np.where(bought, rng.integers(2999, 49900, n) / 100, 0.0)}))
    tool.write_pull(tool.build_pull(answer), tmp_path)
    return tmp_path, answer


JOURNEY_SPECS = {
    "random": ABTestSpec(group_col="segment", control="No E-Mail", treatment="Mens E-Mail", outcome_col="visit"),
    "chose": ObservationalSpec(group_col="segment", control="No E-Mail", treatment="Mens E-Mail", outcome_col="spend",
                               covariates=["recency", "history", "newbie"]),
}


def _confirm_hillstrom(folder):
    uploads = [("files", (path.name, path.read_bytes(), "application/octet-stream")) for path in sorted(folder.iterdir())]
    got = client.post("/api/sources", files=uploads).json()
    customer = next(s for s in got["steps"] if s["file"] == tool.XLSX_NAME)
    assert (customer["header_row"], customer["footer_rows"]) == (3, 2)
    return client.post(f"/api/sources/{got['source_id']}/confirm", json={
        "recipe": {"steps": confirmed(helpers.hillstrom_steps()), "output": "customers"},
        "unit_column": "customer_id", "group_column": "emailed"}).json()


def _numbers(text):
    return re.findall(r"-?\d+\.\d+", text)


@pytest.mark.parametrize("mode", ["quick", "right_answer", "research"])
@pytest.mark.parametrize("design", ["random", "chose"])
def test_whole_journey_report_matches_a_plain_upload_and_script_reproduces_it(hillstrom_pull, tmp_path_factory,
                                                                             mode, design):
    folder, answer = hillstrom_pull
    result = _confirm_hillstrom(folder)
    spec = JOURNEY_SPECS[design].model_dump()
    report = client.post("/api/run", json={"dataset_id": result["dataset_id"], "spec": spec, "mode": mode}).json()

    frozen = client.get(f"/api/datasets/{result['dataset_id']}/snapshot").content
    plain_id = client.post("/api/upload", files={"file": (main.FROZEN_NAME, frozen, "text/csv")}).json()["dataset_id"]
    plain = client.post("/api/run", json={"dataset_id": plain_id, "spec": spec, "mode": mode}).json()
    assert report["effect"] == plain["effect"]
    assert report["repro"]["source"]["table_sha256"] == report["repro"]["data_sha256"]
    assert len(report["repro"]["source"]["inputs"]) == 4

    (folder / "repro.py").write_text(report["script"])
    rebuilt = subprocess.run([sys.executable, "repro.py"], cwd=folder, check=True, capture_output=True, text=True)
    assert (folder / main.FROZEN_NAME).read_bytes() == frozen

    other = tmp_path_factory.mktemp("plain")
    (other / main.FROZEN_NAME).write_bytes(frozen)
    (other / "repro.py").write_text(plain["script"])
    direct = subprocess.run([sys.executable, "repro.py"], cwd=other, check=True, capture_output=True, text=True)
    assert rebuilt.stdout == direct.stdout
    effect = report["effect"]
    printed = {float(n) for n in _numbers(rebuilt.stdout)}
    for key in ("diff", "ci_low", "ci_high"):
        assert any(abs(n - effect[key]) < 1e-3 for n in printed), (key, effect[key], rebuilt.stdout)


def _hillstrom_script(folder):
    result = _confirm_hillstrom(folder)
    spec = JOURNEY_SPECS["random"].model_dump()
    report = client.post("/api/run", json={"dataset_id": result["dataset_id"], "spec": spec, "mode": "quick"}).json()
    (folder / "repro.py").write_text(report["script"])
    return report


def test_script_stops_when_an_original_file_has_changed(hillstrom_pull):
    folder, _ = hillstrom_pull
    report = _hillstrom_script(folder)
    changed = folder / report["repro"]["source"]["inputs"][0]["file"]
    changed.write_bytes(changed.read_bytes() + b"\n")
    run = subprocess.run([sys.executable, "repro.py"], cwd=folder, capture_output=True, text=True)
    assert run.returncode != 0 and "isn't the same file the report used" in run.stderr
    assert not (folder / main.FROZEN_NAME).exists()


def test_script_stops_when_an_original_file_is_missing(hillstrom_pull):
    folder, _ = hillstrom_pull
    report = _hillstrom_script(folder)
    (folder / report["repro"]["source"]["inputs"][-1]["file"]).unlink()
    run = subprocess.run([sys.executable, "repro.py"], cwd=folder, capture_output=True, text=True)
    assert run.returncode != 0 and "isn't in this folder" in run.stderr


def test_script_stops_when_the_rebuilt_table_differs(hillstrom_pull):
    folder, _ = hillstrom_pull
    report = _hillstrom_script(folder)
    script = report["script"].replace(f"TABLE_SHA256 = {report['repro']['source']['table_sha256']!r}",
                                      "TABLE_SHA256 = '0' * 64")
    (folder / "repro.py").write_text(script)
    run = subprocess.run([sys.executable, "repro.py"], cwd=folder, capture_output=True, text=True)
    assert run.returncode != 0 and "doesn't match the one the report analyzed" in run.stderr


def test_report_records_the_excel_reader_version(hillstrom_pull):
    folder, _ = hillstrom_pull
    report = _hillstrom_script(folder)
    import openpyxl
    assert report["repro"]["source"]["versions"] == {"openpyxl": openpyxl.__version__}
    assert f"openpyxl {openpyxl.__version__}" in report["script"]


def test_confirmed_labels_reach_the_analysis_and_the_script_unchanged(tmp_path):
    people = "id,group,region,visits\n" + "".join(f"C-{i},{'01' if i % 2 else '1'},{'NA' if i % 3 else 'EU'},{i % 7}\n"
                                                  for i in range(1, 61))
    got = add(csv_file("people.csv", people))
    steps = confirmed(got["steps"] + [{"op": "clean_key", "table": "people", "column": "id", "rule": "digits_only",
                                       "into": "pid"}])
    result = confirm(got["source_id"], steps).json()
    analyzed = main.DATASETS[result["dataset_id"]]["df"]
    assert sorted(analyzed["group"].unique()) == ["01", "1"] and "NA" in set(analyzed["region"])

    spec = ABTestSpec(group_col="group", control="01", treatment="1", outcome_col="visits").model_dump()
    report = client.post("/api/run", json={"dataset_id": result["dataset_id"], "spec": spec, "mode": "quick"}).json()
    assert "error" not in report and "**READ_OPTIONS" in report["script"]
    (tmp_path / "people.csv").write_text(people)
    (tmp_path / "repro.py").write_text(report["script"])
    subprocess.run([sys.executable, "repro.py"], cwd=tmp_path, check=True, capture_output=True, text=True)


def test_an_original_named_like_the_finished_table_is_not_overwritten():
    got = add(csv_file("people.csv", PEOPLE), csv_file("analysis_table.csv", ORDERS))
    reads = [{**step, "into": "orders"} if step["file"] == "analysis_table.csv" else step for step in got["steps"]]
    result = confirm(got["source_id"], confirmed(people_steps(reads))).json()
    assert result["name"] == "analysis_table_2.csv"
    assert "open('analysis_table_2.csv', 'wb')" in main.DATASETS[result["dataset_id"]]["source_script"]


def test_page_has_the_new_screen():
    page = client.get("/").text
    assert 'id="m1"' in page and "Put them together first" in page


# ------------------------------------------------- answers from building the table carry into the question

# 'cohort' looks like a group column by name, but the person says 'got_email' shows who got the campaign.
EMAILED = "id,cohort,got_email,signup_date,spend\n" + "".join(
    f"{i},{'spring' if i % 3 else 'fall'},{'yes' if i % 2 else 'no'},2024-03-{1 + i % 20:02d},{(i * 7) % 50}\n"
    for i in range(1, 81))


# Two date columns: the person says purchases happened on 'bought_on', not 'signup_date'.
TWO_DATES = "id,cohort,signup_date,bought_on,spend\n" + "".join(
    f"{i},{'spring' if i % 3 else 'fall'},2024-01-{1 + i % 20:02d},2024-03-{1 + i % 20:02d},{(i * 7) % 50}\n"
    for i in range(1, 81))


def _built(group=None, time_column=None, content=EMAILED):
    got = add(csv_file("people.csv", content))
    response = confirm(got["source_id"], confirmed(got["steps"]), unit="id", group=group, time_column=time_column)
    assert response.status_code == 200, response.text
    return response.json()


def test_plain_upload_has_nothing_confirmed():
    result = client.post("/api/upload", files={"file": ("people.csv", EMAILED.encode(), "text/csv")}).json()
    assert result["confirmed"] is None
    first = client.get(f"/api/suggestions?dataset_id={result['dataset_id']}&mode=quick&design=random").json()
    assert first["questions"][0]["spec"]["group_col"] == "cohort"     # without an answer, the name is a guess


def test_confirmed_group_column_is_used_for_every_suggestion():
    result = _built(group="got_email")
    assert result["confirmed"] == {"group_col": "got_email", "date_col": None}
    for design in ("random", "chose", "unsure", "one_date_comparison"):
        questions = client.get(f"/api/suggestions?dataset_id={result['dataset_id']}&mode=quick&design={design}").json()
        assert questions["questions"], design
        assert {q["spec"]["group_col"] for q in questions["questions"]} == {"got_email"}, design


def test_confirmed_group_column_makes_comparing_groups_the_starting_point():
    designs = _built(group="got_email")["designs"]
    assert designs["default"] == "chose"           # they weren't asked how people were picked, so don't assume random
    assert all(option["ok"] for option in designs["options"])


def test_typed_question_uses_the_confirmed_group_without_ai():
    result = _built(group="got_email")
    interp = client.post("/api/ask", json={"dataset_id": result["dataset_id"], "question": "Did spend go up?",
                                            "design": "random"}).json()
    assert interp["spec"]["group_col"] == "got_email"
    assert interp["spec"]["outcome_col"] == "spend"


def test_ai_cannot_quietly_swap_the_confirmed_group(monkeypatch):
    from app import translator
    seen = {}

    def fake(question, profile, design, confirmed=None):
        seen["confirmed"] = confirmed
        return {"analysis_type": "ab_test", "group_col": "cohort", "control": "fall", "treatment": "spring",
                "outcome_col": "spend"}

    monkeypatch.setattr(translator, "_llm_spec", fake)
    result = _built(group="got_email")
    interp = client.post("/api/ask", json={"dataset_id": result["dataset_id"], "question": "Did spring spend more?",
                                            "design": "random"}).json()
    assert seen["confirmed"]["group_col"] == "got_email"
    assert interp["spec"]["group_col"] == "got_email"
    assert any("you said 'got_email' shows who got the campaign" in note for note in interp["notes"])


def test_everyone_got_it_rules_out_group_comparisons_and_uses_the_chosen_date():
    result = _built(time_column="signup_date")
    assert result["confirmed"] == {"group_col": None, "date_col": "signup_date"}
    options = {option["key"]: option for option in result["designs"]["options"]}
    assert result["designs"]["default"] == "one_date" and options["one_date"]["ok"]
    for key in ("random", "chose", "unsure", "one_date_comparison"):
        assert not options[key]["ok"] and "everyone got the campaign" in options[key]["why"]
    refused = client.post("/api/ask", json={"dataset_id": result["dataset_id"], "question": "x", "design": "random"})
    assert refused.status_code == 400
    questions = client.get(f"/api/suggestions?dataset_id={result['dataset_id']}&mode=quick&design=one_date").json()
    assert {q["spec"]["date_col"] for q in questions["questions"]} == {"signup_date"}


def _ask_with_fake_ai(monkeypatch, result, answer, design):
    from app import translator
    monkeypatch.setattr(translator, "_llm_spec", lambda question, profile, design, confirmed=None: answer)
    response = client.post("/api/ask", json={"dataset_id": result["dataset_id"], "question": "Did spend go up?",
                                              "design": design})
    assert response.status_code == 200, response.text
    return response.json()


def test_ai_cannot_quietly_swap_the_confirmed_date(monkeypatch):
    result = _built(time_column="bought_on", content=TWO_DATES)
    for design in ("one_date", None):
        interp = _ask_with_fake_ai(monkeypatch, result, {"analysis_type": "before_after", "date_col": "signup_date",
                                                         "outcome_col": "spend", "change_date": ""}, design)
        assert interp["spec"]["date_col"] == "bought_on", design
        assert any("you said 'bought_on' is the date column" in note for note in interp["notes"]), design


def test_ai_answer_using_the_confirmed_date_is_kept(monkeypatch):
    result = _built(time_column="bought_on", content=TWO_DATES)
    interp = _ask_with_fake_ai(monkeypatch, result, {"analysis_type": "before_after", "date_col": "bought_on",
                                                     "outcome_col": "spend", "change_date": ""}, "one_date")
    assert interp["source"] == "llm" and interp["spec"]["date_col"] == "bought_on"
    assert not any("you said" in note for note in interp["notes"])


def test_ai_cannot_compare_groups_when_everyone_got_it(monkeypatch):
    # With no design picked yet, the AI chooses the kind of analysis, so it could split people into groups.
    result = _built(time_column="bought_on", content=TWO_DATES)
    interp = _ask_with_fake_ai(monkeypatch, result, {"analysis_type": "ab_test", "group_col": "cohort",
                                                     "control": "fall", "treatment": "spring",
                                                     "outcome_col": "spend"}, None)
    assert interp["spec"]["analysis_type"] == "before_after"
    assert interp["spec"]["date_col"] == "bought_on"
    assert any("you said everyone got the campaign" in note for note in interp["notes"])
