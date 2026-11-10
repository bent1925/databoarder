"""The data stage. The main test: rebuild the Hillstrom answer table from its messy pieces."""
import hashlib
import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.ingest import build_table, inspect_file, inspect_files, require_comparison, script_block
from app.main import DATASETS, app
from app.recipe_ops import RecipeError
from app.schemas import DataRecipe

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("make_messy_hillstrom", ROOT / "tools" / "make_messy_hillstrom.py")
tool = importlib.util.module_from_spec(_spec)
sys.modules["make_messy_hillstrom"] = tool
_spec.loader.exec_module(tool)
REAL_PULL = ROOT / "local_data" / "hillstrom"


def recipe(steps, output, status="confirmed"):
    return DataRecipe(steps=[{"status": status, **step} for step in steps], output=output)


def hillstrom_steps():
    def read(name, into):
        return {"op": "read_table", "file": name, "header_row": 0, "footer_rows": 0, "into": into}

    def key(table, column):
        return {"op": "clean_key", "table": table, "column": column, "rule": "digits_only", "into": "customer_id"}

    def flag(other, into):
        return {"op": "flag_membership", "table": "customers", "key": "customer_id", "other": other,
                "other_key": "customer_id", "into": into, "yes": "1", "no": "0"}

    return [
        {"op": "read_table", "file": tool.XLSX_NAME, "header_row": 3, "footer_rows": 2, "into": "customers"},
        read("email_sends.csv", "sends"), read("site_visits.csv", "visits"), read("purchases.csv", "purchases"),
        key("customers", tool.CUSTOMER_HEADER), key("sends", "cust_id"), key("visits", "visitor_id"),
        key("purchases", "customer_ref"),
        {"op": "roll_up", "table": "purchases", "by": "customer_id", "into": "per_customer",
         "measures": [{"how": "sum", "column": "amount", "into": "spend"}]},
        {"op": "match", "left": "customers", "right": "sends", "left_key": "customer_id", "right_key": "customer_id",
         "bring": {"campaign": "segment"}},
        {"op": "fill_missing", "table": "customers", "column": "segment", "value": tool.NO_EMAIL},
        flag("visits", "visit"), flag("per_customer", "conversion"), flag("sends", "emailed"),
        {"op": "match", "left": "customers", "right": "per_customer", "left_key": "customer_id",
         "right_key": "customer_id", "bring": {"spend": "spend"}},
        {"op": "fill_missing", "table": "customers", "column": "spend", "value": "0"},
    ]


def hillstrom_recipe():
    return recipe(hillstrom_steps(), "customers")


def as_analyzed(built):
    return pd.read_csv(io.BytesIO(built.csv_bytes)).sort_values("customer_id").reset_index(drop=True)


@pytest.fixture
def answer():
    rng = np.random.default_rng(3)
    n = 500
    visit = rng.random(n) < 0.3
    bought = visit & (rng.random(n) < 0.5)
    return tool.answer_table(pd.DataFrame({
        "recency": rng.integers(1, 13, n), "history_segment": rng.choice(["1) $0 - $100", "2) $100 - $200"], n),
        "history": rng.integers(3000, 90000, n) / 100, "mens": rng.integers(0, 2, n),
        "womens": rng.integers(0, 2, n), "zip_code": rng.choice(["Urban", "Rural"], n),
        "newbie": rng.integers(0, 2, n), "channel": rng.choice(["Web", "Phone"], n),
        "segment": rng.choice(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], n),
        "visit": visit.astype(int), "conversion": bought.astype(int),
        "spend": np.where(bought, rng.integers(2999, 49900, n) / 100, 0.0),
    }))


@pytest.fixture
def pull(answer, tmp_path):
    pytest.importorskip("openpyxl")
    tool.write_pull(tool.build_pull(answer), tmp_path)
    return tmp_path


def test_hillstrom_pull_rebuilds_the_answer_table(answer, pull):
    assert tool.same_table(as_analyzed(build_table(hillstrom_recipe(), pull)), answer)


@pytest.mark.skipif(not (REAL_PULL / "pull_full").exists(), reason="the real Hillstrom pull hasn't been generated")
def test_real_hillstrom_pull_rebuilds_the_answer_key():
    pytest.importorskip("openpyxl")
    key = pd.read_csv(REAL_PULL / "answer_key" / "analysis_table.csv")
    assert tool.same_table(as_analyzed(build_table(hillstrom_recipe(), REAL_PULL / "pull_full")), key)


def test_treated_only_pull_stops_and_says_what_to_do(answer, tmp_path):
    pytest.importorskip("openpyxl")
    tool.write_pull(tool.treated_only(tool.build_pull(answer)), tmp_path)
    built = build_table(hillstrom_recipe(), tmp_path)
    with pytest.raises(RecipeError, match="no one to compare them with.*didn't get the campaign"):
        require_comparison(built.table, "emailed")


def test_comparison_check_passes_with_two_groups():
    require_comparison(pd.DataFrame({"emailed": ["1", "0"]}), "emailed")


def test_nothing_runs_until_every_step_is_confirmed(answer, pull):
    steps = [{"status": "confirmed", **step} for step in hillstrom_steps()]
    steps[3]["status"] = "proposed"
    with pytest.raises(RecipeError, match="Step 4 still need"):
        build_table(DataRecipe(steps=steps, output="customers"), pull)


def test_steps_start_as_proposed():
    plan = DataRecipe(steps=[{"op": "stack", "tables": ["a", "b"], "into": "c"}], output="c")
    assert plan.steps[0].status == "proposed"


def test_script_rebuilds_the_identical_frozen_table(pull):
    plan = hillstrom_recipe()
    built = build_table(plan, pull)
    (pull / "rebuild.py").write_text(script_block(plan))
    subprocess.run([sys.executable, "rebuild.py"], cwd=pull, check=True, capture_output=True, text=True)
    assert hashlib.sha256((pull / "analysis_table.csv").read_bytes()).hexdigest() == built.sha256


def test_frozen_table_goes_through_the_normal_upload_path(pull):
    built = build_table(hillstrom_recipe(), pull)
    response = TestClient(app).post("/api/upload", files={"file": ("frozen.csv", built.csv_bytes, "text/csv")})
    assert response.status_code == 200
    assert DATASETS[response.json()["dataset_id"]]["sha"] == built.sha256


def test_inputs_are_fingerprinted(pull):
    built = build_table(hillstrom_recipe(), pull)
    assert {entry["file"] for entry in built.inputs} == {tool.XLSX_NAME, "email_sends.csv", "site_visits.csv",
                                                         "purchases.csv"}


def test_log_shows_match_losses_in_plain_words(answer, pull):
    built = build_table(hillstrom_recipe(), pull)
    matched = int(answer.emailed.sum())
    match_step = next(entry for entry in built.log if entry["text"].startswith("Match 'customers' to 'sends'"))
    assert match_step["note"] == f"{matched:,} of {len(answer):,} rows found a match. 0 rows in 'sends' matched nothing."


def test_finds_excel_header_and_footer(pull):
    found = inspect_file(pull / tool.XLSX_NAME)
    assert len(found) == 1
    assert (found[0].header_row, found[0].footer_rows) == (3, 2)
    assert found[0].columns[0] == tool.CUSTOMER_HEADER


def test_every_sheet_is_found(tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(writer, sheet_name="One", index=False)
        pd.DataFrame({"a": [5], "b": [6]}).to_excel(writer, sheet_name="Two", index=False)
    assert [(item.sheet, item.rows) for item in inspect_file(path)] == [("One", 2), ("Two", 1)]


def test_several_files_are_inspected(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n")
    (tmp_path / "b.csv").write_text("p;q\n1;2\n")
    found = inspect_files([tmp_path / "a.csv", tmp_path / "b.csv"])
    assert [(item.file, item.delimiter, item.columns) for item in found] == [("a.csv", ",", ["x", "y"]),
                                                                             ("b.csv", ";", ["p", "q"])]


def test_csv_with_junk_rows_and_text_preserved(tmp_path):
    (tmp_path / "a.csv").write_text("Report title\n\nid,country\n00123,NA\n00456,UK\nTotal: 2\n")
    assert (inspect_file(tmp_path / "a.csv")[0].header_row, inspect_file(tmp_path / "a.csv")[0].footer_rows) == (2, 1)
    plan = recipe([{"op": "read_table", "file": "a.csv", "header_row": 2, "footer_rows": 1, "into": "t"}], "t")
    table = build_table(plan, tmp_path).table
    assert table["id"].tolist() == ["00123", "00456"] and table["country"].tolist() == ["NA", "UK"]


def test_unknown_steps_and_fields_are_rejected():
    with pytest.raises(ValidationError):
        DataRecipe(steps=[{"op": "run_python", "code": "1"}], output="x")
    with pytest.raises(ValidationError):
        DataRecipe(steps=[{"op": "stack", "tables": ["a"], "into": "c", "extra": 1}], output="c")


@pytest.mark.parametrize("name", ["../secret.csv", "/etc/passwd", "sub/a.csv"])
def test_files_outside_the_upload_are_refused(tmp_path, name):
    plan = recipe([{"op": "read_table", "file": name, "header_row": 0, "footer_rows": 0, "into": "t"}], "t")
    with pytest.raises(RecipeError, match="isn't one of the uploaded files"):
        build_table(plan, tmp_path)


def _two_tables(tmp_path):
    (tmp_path / "a.csv").write_text("id,v\n1,10\n2,20\n3,oops\n")
    (tmp_path / "b.csv").write_text("id,w\n1,x\n1,y\n")
    return [{"op": "read_table", "file": "a.csv", "header_row": 0, "footer_rows": 0, "into": "a"},
            {"op": "read_table", "file": "b.csv", "header_row": 0, "footer_rows": 0, "into": "b"}]


def test_unknown_column_gets_a_clear_message(tmp_path):
    plan = recipe(_two_tables(tmp_path) + [{"op": "fill_missing", "table": "a", "column": "nope", "value": "0"}], "a")
    with pytest.raises(RecipeError, match="no column called 'nope'"):
        build_table(plan, tmp_path)


def test_match_refuses_to_multiply_rows(tmp_path):
    plan = recipe(_two_tables(tmp_path) + [{"op": "match", "left": "a", "right": "b", "left_key": "id",
                                            "right_key": "id", "bring": {"w": "w"}}], "a")
    with pytest.raises(RecipeError, match="would multiply rows. Roll that table up"):
        build_table(plan, tmp_path)


def test_roll_up_refuses_text_it_cannot_add(tmp_path):
    plan = recipe(_two_tables(tmp_path) + [{"op": "roll_up", "table": "a", "by": "id", "into": "r",
                                            "measures": [{"how": "sum", "column": "v", "into": "t"}]}], "r")
    with pytest.raises(RecipeError, match="needs to be numbers.*'oops'"):
        build_table(plan, tmp_path)


def test_roll_up_reports_rows_with_no_key(tmp_path):
    (tmp_path / "a.csv").write_text("id,v\n1,10\n1,20\n,30\n")
    plan = recipe([{"op": "read_table", "file": "a.csv", "header_row": 0, "footer_rows": 0, "into": "a"},
                   {"op": "roll_up", "table": "a", "by": "id", "into": "r",
                    "measures": [{"how": "sum", "column": "v", "into": "t"}]}], "r")
    built = build_table(plan, tmp_path)
    assert built.table["t"].tolist() == [30.0]
    assert built.log[1]["note"] == "Left out 1 row with no 'id'."


@pytest.mark.parametrize("raw,clean", [("C-00123", "123"), ("00123", "123"), ("13085.0", "13085"), ("C00123", "123"),
                                       ("0", "0")])
def test_ids_written_differently_become_the_same(tmp_path, raw, clean):
    (tmp_path / "a.csv").write_text(f"id,v\n{raw},1\n")
    plan = recipe([{"op": "read_table", "file": "a.csv", "header_row": 0, "footer_rows": 0, "into": "a"},
                   {"op": "clean_key", "table": "a", "column": "id", "rule": "digits_only", "into": "k"}], "a")
    assert build_table(plan, tmp_path).table["k"].tolist() == [clean]


def test_ids_without_digits_are_counted_not_hidden(tmp_path):
    (tmp_path / "a.csv").write_text("id,v\nunknown,1\n5,2\n")
    plan = recipe([{"op": "read_table", "file": "a.csv", "header_row": 0, "footer_rows": 0, "into": "a"},
                   {"op": "clean_key", "table": "a", "column": "id", "rule": "digits_only", "into": "k"}], "a")
    assert build_table(plan, tmp_path).log[1]["note"] == "1 value couldn't be read as an ID."


def test_stack_and_date_filter_resolve_overlapping_sheets(tmp_path):
    (tmp_path / "y1.csv").write_text("d,v\n2010-11-30,1\n2010-12-05,2\n")
    (tmp_path / "y2.csv").write_text("d,v\n2010-12-05,2\n2011-01-01,3\n")
    read = [{"op": "read_table", "file": f"{n}.csv", "header_row": 0, "footer_rows": 0, "into": n} for n in ("y1", "y2")]
    plan = recipe(read + [
        {"op": "filter", "table": "y1", "column": "d", "test": "less_than", "value": "2010-12-01", "compare_as": "date"},
        {"op": "stack", "tables": ["y1", "y2"], "into": "all"}], "all")
    assert build_table(plan, tmp_path).table["v"].tolist() == ["1", "2", "3"]


def test_stack_refuses_different_columns(tmp_path):
    plan = recipe(_two_tables(tmp_path) + [{"op": "stack", "tables": ["a", "b"], "into": "c"}], "c")
    with pytest.raises(RecipeError, match="different columns"):
        build_table(plan, tmp_path)


def test_number_comparison_needs_a_stated_type(tmp_path):
    plan = recipe(_two_tables(tmp_path) + [{"op": "filter", "table": "a", "column": "v", "test": "at_least",
                                            "value": "10"}], "a")
    with pytest.raises(RecipeError, match="compares numbers or dates"):
        build_table(plan, tmp_path)
