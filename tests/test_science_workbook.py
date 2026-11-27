"""Science mode: reading Excel workbooks with one sheet per group, by fixed rules, with minimal input."""
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.science import engine, report, sheets, workbook

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)
GROUPS = ["Vehicle Control", "Drug 25mg/kg", "Drug 50mg/kg"]
DAYS = [1, 4, 8, 11, 12]


def make_book(*, typo=True, missing_labels=True, extra=None) -> bytes:
    """Laid out like an export from Numbers: a title row, a sheet that only lists the groups, one sheet per group
    (days across, animals down, the group name beside the first animal), a blank sheet at the end."""
    rng = np.random.default_rng(5)
    book = openpyxl.Workbook()
    first = book.active
    first.title = "Sheet 1 - Table 2"
    first["A1"] = "Table 2"
    for i, g in enumerate(GROUPS):
        first.cell(row=3, column=2 + i, value=g)
    for n, g in enumerate(GROUPS):
        ws = book.create_sheet(f"Sheet 1 - Table {n + 3}")
        ws["A1"] = f"Table {n + 3}"
        for j, d in enumerate(DAYS):
            label = "Day11" if (d == 11 and n == 0) else f"Day {d}"
            if not (missing_labels and n == 2 and d != 12):
                ws.cell(row=2, column=2 + j, value=label)
        ws.cell(row=3, column=1, value="drug 25mg/lg" if (typo and n == 1) else g)
        for a in range(5):
            for j, d in enumerate(DAYS):
                ws.cell(row=3 + a, column=2 + j, value=round(2.5 + 0.1 * d * (1 - 0.4 * n) + rng.normal(0, .2), 2))
    if extra:
        extra(book)
    blank = book.create_sheet("Sheet 1 - Table 9")
    blank["A1"] = "Table 9"
    book.create_sheet("Empty")
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


def test_one_sheet_per_group_is_read_without_any_input():
    found = workbook.propose(make_book())
    summary, plan = found["summary"], found["plan"]
    assert [g["name"] for g in summary["groups"]] == GROUPS
    assert summary["times"] == DAYS and summary["time_name"] == "Day"
    assert {s["sheet"] for s in summary["skipped"]} == {"Sheet 1 - Table 2", "Sheet 1 - Table 9", "Empty"}
    table = sheets.build_table(make_book(), plan)
    assert list(table.columns) == ["Group", "Animal", "Day", "Value"]
    assert len(table) == 3 * 5 * 5 and table["Value"].notna().all()
    assert table.groupby("Group")["Animal"].nunique().to_dict() == {g: 5 for g in GROUPS}


def test_what_was_inferred_is_listed_for_checking():
    checks = workbook.propose(make_book())["summary"]["checks"]
    assert any("no labels above columns B, C, D and E" in c and "Day 1, Day 4, Day 8 and Day 11" in c for c in checks)
    assert any("'drug 25mg/lg'" in c and "Using 'Drug 25mg/kg'" in c for c in checks)
    assert workbook.propose(make_book(typo=False, missing_labels=False))["summary"]["checks"] == []


def test_the_sheets_own_spelling_can_be_kept():
    found = workbook.propose(make_book(), {"keep_spelling": True})
    assert "drug 25mg/lg" in [g["name"] for g in found["summary"]["groups"]]
    assert not any("Using" in c for c in found["summary"]["checks"])


def test_text_in_a_number_cell_is_reported_and_left_empty():
    def dead(book):
        book["Sheet 1 - Table 3"]["F5"] = "dead"
    found = workbook.propose(make_book(extra=dead))
    assert any("cell F5 says 'dead'" in c for c in found["summary"]["checks"])
    table = sheets.build_table(make_book(extra=dead), found["plan"])
    assert table["Value"].isna().sum() == 1


def test_labels_that_disagree_stop_with_a_reason():
    def clash(book):
        book["Sheet 1 - Table 4"]["C2"] = "Day 5"
    with pytest.raises(workbook.WorkbookError, match="labelled differently"):
        workbook.propose(make_book(extra=clash))


def test_a_workbook_without_numbers_is_refused():
    book = openpyxl.Workbook()
    book.active["A1"] = "Nothing here"
    out = io.BytesIO()
    book.save(out)
    with pytest.raises(workbook.WorkbookError, match="numbers"):
        workbook.propose(out.getvalue())


def test_a_single_ordinary_sheet_reads_as_a_table():
    book = openpyxl.Workbook()
    ws = book.active
    ws.append(["Mouse", "Control", "Treated"])
    for i in range(6):
        ws.append([f"m{i}", 4 + i * .1, 5 + i * .2])
    other = book.create_sheet("Notes")
    other["B4"], other["C4"] = 1, 2
    out = io.BytesIO()
    book.save(out)
    found = workbook.propose(out.getvalue())
    assert found["plan"]["kind"] == "single_sheet"
    table = sheets.build_table(out.getvalue(), found["plan"])
    assert list(table.columns) == ["Mouse", "Control", "Treated"] and len(table) == 6
    assert found["summary"]["other_sheets"] == ["Notes"]
    assert workbook.propose(out.getvalue(), {"sheet": "Notes"})["plan"]["sheet"] == "Notes"


# ---------------------------------------------------------------------------- one day at a time

def day_spec(day="12", **kw):
    spec = dict(analysis="compare_groups", layout="long", groups=GROUPS, group_col="Group", value_col="Value",
                subject_col="Animal", subset_col="Day", subset_value=day, paired=False, method="means",
                equal_spread=False, followup="vs_control", control="Vehicle Control", assignment="random")
    spec.update(kw)
    return spec


def test_comparing_groups_on_one_day():
    table = sheets.build_table(make_book(), workbook.propose(make_book())["plan"])
    r = engine.run(table, day_spec())
    assert r["left_out"]["rows_kept"] == {"column": "Day", "value": "12", "kept": 15, "of": 75}
    assert [g["n"] for g in r["groups"]] == [5, 5, 5]
    expected = table[table.Day == 12].groupby("Group", sort=False)["Value"].mean()
    assert [g["mean"] for g in r["groups"]] == pytest.approx([expected[g] for g in GROUPS])
    rep = report.build(day_spec(), r)
    assert "Only rows where 'Day' is 12" in rep.readback
    assert any(c.name == "Part of the data" and "false alarm" in c.message for c in rep.checks)


def test_comparing_days_within_one_group():
    table = sheets.build_table(make_book(), workbook.propose(make_book())["plan"])
    spec = day_spec(groups=["1", "12"], group_col="Day", subset_col="Group", subset_value="Vehicle Control",
                    paired=True, followup="none", control=None)
    r = engine.run(table, spec)
    assert r["n_subjects"] == 5 and r["tests"][0]["key"] == "paired_t"


def test_the_row_picking_column_cant_be_the_group_column():
    d = upload_book()
    client.post(f"/api/science/datasets/{d['dataset_id']}/confirm")
    out = client.post("/api/science/run", json={"dataset_id": d["dataset_id"], "spec": day_spec(subset_col="Group")})
    assert out.status_code == 400 and "can't also be" in out.json()["detail"]
    out = client.post("/api/science/run", json={"dataset_id": d["dataset_id"], "spec": day_spec(day=None)})
    assert out.status_code == 400 and "which value of 'Day'" in out.json()["detail"]


# ---------------------------------------------------------------------------- through the web routes

def upload_book(raw=None, name="House comparison.xlsx"):
    raw = raw or make_book()
    out = client.post("/api/science/upload", files={"file": (name, raw, "application/octet-stream")})
    assert out.status_code == 200, out.text
    return out.json()


def test_nothing_runs_until_the_reading_is_confirmed():
    d = upload_book()
    assert d["workbook"]["confirmed"] is False and d["workbook"]["checks"]
    assert d["guesses"]["compare_groups"]["subset_col"] == "Day"
    assert [m["group"] for m in d["workbook"]["means"]] == GROUPS
    blocked = client.post("/api/science/run", json={"dataset_id": d["dataset_id"], "spec": day_spec()})
    assert blocked.status_code == 400 and "That's right" in blocked.json()["detail"]
    ok = client.post(f"/api/science/datasets/{d['dataset_id']}/confirm").json()
    assert ok["workbook"]["confirmed"] is True
    assert client.post("/api/science/run", json={"dataset_id": d["dataset_id"], "spec": day_spec()}).status_code == 200


def test_reading_again_with_the_sheets_spelling():
    d = upload_book()
    again = client.post(f"/api/science/datasets/{d['dataset_id']}/reread", json={"keep_spelling": True}).json()
    assert again["dataset_id"] != d["dataset_id"] and again["workbook"]["confirmed"] is False
    assert "drug 25mg/lg" in [g["name"] for g in again["workbook"]["groups"]]


def test_tidy_table_download():
    d = upload_book()
    tidy = pd.read_csv(io.BytesIO(client.get(f"/api/science/datasets/{d['dataset_id']}/tidy").content))
    assert list(tidy.columns) == ["Group", "Animal", "Day", "Value"] and len(tidy) == 75


def test_the_script_rebuilds_the_table_from_the_workbook(tmp_path):
    raw = make_book()
    d = upload_book(raw)
    client.post(f"/api/science/datasets/{d['dataset_id']}/confirm")
    rep = client.post("/api/science/run", json={"dataset_id": d["dataset_id"], "spec": day_spec()}).json()
    assert any("no labels above columns" in line for line in rep["prep"])
    (tmp_path / "House comparison.xlsx").write_bytes(client.get(f"/api/science/datasets/{d['dataset_id']}/data").content)
    (tmp_path / "reproduce.py").write_text(rep["script"])
    done = subprocess.run([sys.executable, "-I", "reproduce.py"], cwd=tmp_path, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr
    again = json.loads(done.stdout)
    assert again["main_p"] == pytest.approx(rep["result"]["main_p"], rel=1e-12)
    assert again["pairs"] == rep["result"]["pairs"]
    (tmp_path / "House comparison.xlsx").write_bytes(make_book(typo=False))
    done = subprocess.run([sys.executable, "-I", "reproduce.py"], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode != 0 and "isn't the file the report used" in done.stderr


REAL = ROOT / "sample_data" / "House comparison LA.xlsx"


@pytest.mark.skipif(not REAL.exists(), reason="the House comparison workbook isn't in sample_data/ (it isn't in git)")
def test_the_house_comparison_workbook():
    found = workbook.propose(REAL.read_bytes())
    summary = found["summary"]
    assert [g["name"] for g in summary["groups"]] == ["Vehicle Control", "RIPTAC 25mg/kg", "RIPTAC 50mg/kg",
                                                      "RIPTAC 25mg/kg + Tala", "RIPTAC 50mg/kg + Tala"]
    assert summary["times"] == [1, 4, 8, 11, 12] and summary["subjects_per_group"] == [5] * 5
    assert [s["why"] for s in summary["skipped"]] == ["only labels, no numbers", "blank"]
    assert len(summary["checks"]) == 2          # Table 7's missing day labels, and Table 6's "25mg/lg + tala"
    table = sheets.build_table(REAL.read_bytes(), found["plan"])
    assert table.shape == (125, 4) and table["Value"].notna().all()


# ---------------------------------------------------------------------------- from the Codex review of PR #14

def book_of(fill) -> bytes:
    book = openpyxl.Workbook()
    book.remove(book.active)
    fill(book)
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


def numeric_days(book, word="Day"):
    for g in ("Ctrl", "Drug"):
        ws = book.create_sheet(g)
        ws.append([word, 1, 4, 8])
        for a in range(3):
            ws.append([f"{g}{a}", 2.1 + a, 2.55 + a, 3.05 + a])


def test_a_first_row_of_day_numbers_is_read_as_labels_and_shown_for_checking():
    raw = book_of(numeric_days)
    found = workbook.propose(raw)
    summary = found["summary"]
    assert summary["times"] == [1, 4, 8] and summary["time_name"] == "Day" and summary["first_row"] == "labels"
    assert any("Row 1 on Ctrl and Drug (1, 4, 8) is read as labels" in c for c in summary["checks"])
    table = sheets.build_table(raw, found["plan"])
    assert len(table) == 2 * 3 * 3 and "Day" not in set(table["Animal"])     # no fabricated animal
    flipped = workbook.propose(raw, {"first_row": "values"})["summary"]
    assert flipped["subjects_per_group"] == [4, 4] and flipped["first_row"] == "values"


def test_the_same_rising_row_on_every_sheet_counts_as_labels_without_a_header_word():
    def fill(book):
        for g in ("Ctrl", "Drug"):
            ws = book.create_sheet(g)
            ws.append([None, 1, 4, 8])
            for a in range(3):
                ws.append([g if a == 0 else None, 2 + a, 3 + a, 4 + a])
    found = workbook.propose(book_of(fill))
    assert found["summary"]["times"] == [1, 4, 8] and found["summary"]["subjects_per_group"] == [3, 3]


def test_measurements_that_happen_to_rise_are_not_taken_for_labels():
    def fill(book):
        for g in ("Ctrl", "Drug"):
            ws = book.create_sheet(g)
            for a in range(3):
                ws.append([g if a == 0 else None, 2.1 + a, 2.6 + a, 3.4 - a])
    found = workbook.propose(book_of(fill))
    assert found["summary"]["subjects_per_group"] == [3, 3] and found["summary"]["first_row"] in (None, "values")


def test_an_animal_with_no_numbers_is_kept_counted_and_listed():
    def fill(book):
        for g in ("Ctrl", "Drug"):
            ws = book.create_sheet(g)
            ws.append(["Mouse", "Day 1", "Day 4"])
            ws.append(["m1", 2, 3])
            ws.append(["m2", "dead", None])
            ws.append(["m3", 2.5, 3.1])
            ws.append(["m4", 2.2, 2.9])
            ws.append(["m5", None, None])                  # last row, label only
    raw = book_of(fill)
    found = workbook.propose(raw)
    assert found["summary"]["subjects_per_group"] == [5, 5]
    checks = found["summary"]["checks"]
    assert any("row 3 (m2) has no numbers" in c for c in checks) and any("row 6 (m5) has no numbers" in c for c in checks)
    table = sheets.build_table(raw, found["plan"])
    spec = dict(analysis="time_course", group_col="Group", time_col="Day", value_col="Value", subject_col="Animal",
                groups=["Ctrl", "Drug"], times=["1", "4"], followup="none")
    r = engine.run(table, spec)
    assert r["left_out"]["incomplete_subjects"] == 4 and [m["n"] for m in r["means"]] == [3, 3]


def test_an_ordinary_sheet_keeps_every_column():
    def fill(book):
        ws = book.create_sheet("Data")
        ws.append(["Subject", "Group", "Day", "Value", "Note"])
        for i in range(6):
            ws.append([f"s{i // 2}", "A" if i < 4 else "B", 1 + 3 * (i % 2), 2.0 + i, "ok"])
    raw = book_of(fill)
    table = sheets.build_table(raw, workbook.propose(raw)["plan"])
    assert list(table.columns) == ["Subject", "Group", "Day", "Value", "Note"]
    assert table["Subject"].tolist() == ["s0", "s0", "s1", "s1", "s2", "s2"]
    assert pd.api.types.is_numeric_dtype(table["Value"]) and not pd.api.types.is_numeric_dtype(table["Note"])


def test_reading_again_keeps_earlier_choices():
    d = upload_book()
    kept = client.post(f"/api/science/datasets/{d['dataset_id']}/reread", json={"keep_spelling": True}).json()
    again = client.post(f"/api/science/datasets/{kept['dataset_id']}/reread", json={"first_row": "values"}).json()
    assert "drug 25mg/lg" in [g["name"] for g in again["workbook"]["groups"]]


def test_a_labelled_day_with_no_numbers_is_kept_and_listed():
    def fill(book):
        for g in ("Ctrl", "Drug"):
            ws = book.create_sheet(g)
            ws.append(["Mouse", "Day 1", "Day 4", "Day 8", "Notes"])
            for a in range(3):
                ws.append([f"m{a}", 2 + a, 3.5 + 1.3 * a + (0.4 if g == "Drug" else 0) * (a % 2),
                           "dead" if a == 0 else None, "fine"])
    raw = book_of(fill)
    found = workbook.propose(raw)
    summary = found["summary"]
    assert summary["times"] == [1, 4, 8]                                   # "Notes" isn't a time point
    assert any("(Day 8) has no numbers on any sheet" in c for c in summary["checks"])
    assert any("cell D2 says 'dead'" in c for c in summary["checks"])
    table = sheets.build_table(raw, found["plan"])
    assert table.loc[table.Day == 8, "Value"].isna().all() and len(table) == 2 * 3 * 3
    spec = dict(analysis="time_course", group_col="Group", time_col="Day", value_col="Value", subject_col="Animal",
                groups=["Ctrl", "Drug"], times=["1", "4", "8"], followup="none")
    with pytest.raises(engine.EngineError, match="fewer than 2 subjects measured at every chosen time"):
        engine.run(table, spec)
    assert engine.run(table, {**spec, "times": ["1", "4"]})["n_subjects"] == 6


def test_a_long_table_whose_only_category_is_the_day_still_loads():
    df = pd.DataFrame({"Subject": [f"s{i}" for i in range(15) for _ in (0, 1)], "Day": ["Day 1", "Day 4"] * 15,
                       "Value": np.arange(30.0)})
    out = client.post("/api/science/upload", files={"file": ("t.csv", df.to_csv(index=False).encode(), "text/csv")})
    assert out.status_code == 200 and "time_course" not in out.json()["guesses"]
