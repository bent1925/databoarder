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
from app.schemas import ABTestSpec

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


def test_whole_journey_report_matches_a_plain_upload_and_script_reproduces_it(hillstrom_pull, tmp_path_factory):
    folder, answer = hillstrom_pull
    uploads = [("files", (path.name, path.read_bytes(), "application/octet-stream")) for path in sorted(folder.iterdir())]
    got = client.post("/api/sources", files=uploads).json()
    customer = next(s for s in got["steps"] if s["file"] == tool.XLSX_NAME)
    assert (customer["header_row"], customer["footer_rows"]) == (3, 2)

    result = client.post(f"/api/sources/{got['source_id']}/confirm", json={
        "recipe": {"steps": confirmed(helpers.hillstrom_steps()), "output": "customers"},
        "unit_column": "customer_id", "group_column": "emailed"}).json()
    spec = ABTestSpec(group_col="segment", control="No E-Mail", treatment="Mens E-Mail", outcome_col="visit").model_dump()
    report = client.post("/api/run", json={"dataset_id": result["dataset_id"], "spec": spec, "mode": "quick"}).json()

    frozen = client.get(f"/api/datasets/{result['dataset_id']}/snapshot").content
    plain_id = client.post("/api/upload", files={"file": (main.FROZEN_NAME, frozen, "text/csv")}).json()["dataset_id"]
    plain = client.post("/api/run", json={"dataset_id": plain_id, "spec": spec, "mode": "quick"}).json()
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
    assert rebuilt.stdout == direct.stdout and re.search(r"\d", rebuilt.stdout)


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
