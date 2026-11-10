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

from app.ingest import (build_table, describe, inspect_file, inspect_files, output_name, read_frozen,
                        require_comparison, script_block)
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


def windowed_hillstrom_steps():
    """Builds the outcomes from dated events, the way a real campaign pull would need."""
    steps = hillstrom_steps()[:8] + [
        {"op": "match", "left": "customers", "right": "sends", "left_key": "customer_id", "right_key": "customer_id",
         "bring": {"campaign": "segment", "send_date": "start"}},
        {"op": "fill_missing", "table": "customers", "column": "segment", "value": tool.NO_EMAIL},
        {"op": "flag_membership", "table": "customers", "key": "customer_id", "other": "sends",
         "other_key": "customer_id", "into": "emailed", "yes": "1", "no": "0"}]
    common = {"op": "count_within_window", "table": "customers", "key": "customer_id", "start_column": "start",
              "default_start": "2008-03-03", "event_key": "customer_id", "days": tool.OUTCOME_WINDOW_DAYS,
              "no_events_value": "0"}
    return steps + [
        {**common, "events": "visits", "event_date": "visit_date", "measures": [{"how": "any_event", "into": "visit"}]},
        {**common, "events": "purchases", "event_date": "order_date",
         "measures": [{"how": "any_event", "into": "conversion"}, {"how": "sum", "column": "amount", "into": "spend"}]},
    ]


def test_hillstrom_outcomes_rebuilt_from_dated_events(answer, pull):
    assert tool.same_table(as_analyzed(build_table(recipe(windowed_hillstrom_steps(), "customers"), pull)), answer)


@pytest.mark.skipif(not (REAL_PULL / "pull_full").exists(), reason="the real Hillstrom pull hasn't been generated")
def test_real_hillstrom_outcomes_rebuilt_from_dated_events():
    pytest.importorskip("openpyxl")
    key = pd.read_csv(REAL_PULL / "answer_key" / "analysis_table.csv")
    plan = recipe(windowed_hillstrom_steps(), "customers")
    assert tool.same_table(as_analyzed(build_table(plan, REAL_PULL / "pull_full")), key)


def test_script_rebuilds_a_table_made_with_a_window(pull):
    plan = recipe(windowed_hillstrom_steps(), "customers")
    built = build_table(plan, pull)
    (pull / "rebuild.py").write_text(script_block(plan))
    subprocess.run([sys.executable, "rebuild.py"], cwd=pull, check=True, capture_output=True, text=True)
    assert hashlib.sha256((pull / "analysis_table.csv").read_bytes()).hexdigest() == built.sha256


def _window_plan(tmp_path, people, events, **changes):
    (tmp_path / "p.csv").write_text(people)
    (tmp_path / "e.csv").write_text(events)
    step = {"op": "count_within_window", "table": "p", "key": "id", "start_column": "start", "events": "e",
            "event_key": "id", "event_date": "when", "days": 7, "no_events_value": "0",
            "measures": [{"how": "count_rows", "into": "n"}, {"how": "sum", "column": "amount", "into": "total"}]}
    return recipe([{"op": "read_table", "file": "p.csv", "header_row": 0, "footer_rows": 0, "into": "p"},
                   {"op": "read_table", "file": "e.csv", "header_row": 0, "footer_rows": 0, "into": "e"},
                   {**step, **changes}], "p")


EVENTS = ("id,when,amount\n1,2024-01-09 08:00:00,5\n1,2024-01-10 09:00:00,1\n1,2024-01-16 23:59:00,2\n"
          "1,2024-01-17 00:00:00,100\n9,2024-01-11,7\n")


def test_window_counts_only_events_inside_it(tmp_path):
    built = build_table(_window_plan(tmp_path, "id,start\n1,2024-01-10 09:00:00\n2,2024-01-10\n", EVENTS), tmp_path)
    assert built.table["n"].tolist() == [2, "0"] and built.table["total"].tolist() == [3.0, "0"]


def test_window_says_what_each_measure_calculates(tmp_path):
    plan = _window_plan(tmp_path, "id,start\n1,2024-01-10\n", EVENTS, measures=[
        {"how": "count_rows", "into": "n"}, {"how": "sum", "column": "amount", "into": "total"},
        {"how": "any_event", "into": "bought"}])
    text = describe(plan.steps[2])
    assert "n (number of events)" in text and "total (total of 'amount')" in text
    assert "bought (1 if any event, otherwise 0)" in text


def test_window_results_are_read_back_as_numbers(tmp_path):
    built = build_table(_window_plan(tmp_path, "id,start\n1,2024-01-10 09:00:00\n2,2024-01-10\n", EVENTS), tmp_path)
    assert built.read_options["dtype"] == {"start": "str"}
    assert read_frozen(built)["total"].tolist() == [3.0, 0.0]


def test_reading_back_keeps_labels_that_look_like_numbers_or_missing(tmp_path):
    (tmp_path / "p.csv").write_text("id,group,region,visits\n001,01,NA,3\n002,1,EU,4\n003,1.0,NA,5\n")
    built = build_table(recipe([{"op": "read_table", "file": "p.csv", "header_row": 0, "footer_rows": 0,
                                 "into": "p"}], "p"), tmp_path)
    table = read_frozen(built)
    assert table["id"].tolist() == ["001", "002", "003"] and table["group"].tolist() == ["01", "1", "1.0"]
    assert table["region"].tolist() == ["NA", "EU", "NA"] and table["visits"].tolist() == [3, 4, 5]


def test_frozen_table_never_takes_an_original_file_name():
    assert output_name(["people.csv"]) == "analysis_table.csv"
    assert output_name(["Analysis_Table.csv", "analysis_table_2.csv"]) == "analysis_table_3.csv"


def test_window_reports_what_it_left_out(tmp_path):
    built = build_table(_window_plan(tmp_path, "id,start\n1,2024-01-10 09:00:00\n2,2024-01-10\n", EVENTS), tmp_path)
    note = built.log[2]["note"]
    assert note.startswith("2 of 5 events fell inside the window.")
    assert "1 event before it started." in note and "1 event after it ended." in note
    assert "1 event belonged to no row in the table." in note


def test_window_length_and_no_events_value_are_required():
    base = {"op": "count_within_window", "table": "p", "key": "id", "start_column": "s", "events": "e",
            "event_key": "id", "event_date": "w", "measures": [{"how": "count_rows", "into": "n"}]}
    with pytest.raises(ValidationError):
        DataRecipe(steps=[{**base, "no_events_value": "0"}], output="p")
    with pytest.raises(ValidationError):
        DataRecipe(steps=[{**base, "days": 7}], output="p")


def test_window_leaves_rows_without_a_start_empty_not_zero(tmp_path):
    built = build_table(_window_plan(tmp_path, "id,start\n1,2024-01-10\n2,\n", EVENTS), tmp_path)
    assert built.table["n"].iloc[0] == 2 and pd.isna(built.table["n"].iloc[1])
    assert "1 row had no start date and left empty." in built.log[2]["note"]


def test_window_default_start_covers_the_comparison_group(tmp_path):
    plan = _window_plan(tmp_path, "id,start\n1,2024-01-10\n1000,\n", "id,when,amount\n1000,2024-01-12,4\n",
                        default_start="2024-01-10", days=2)
    assert build_table(plan, tmp_path).table["n"].tolist() == ["0", "0"]     # the 12th is outside a 2-day window
    plan = _window_plan(tmp_path, "id,start\n1,2024-01-10\n1000,\n", "id,when,amount\n1000,2024-01-11,4\n",
                        default_start="2024-01-10", days=2)
    assert build_table(plan, tmp_path).table["n"].tolist() == ["0", 1]


def test_window_warns_when_data_ends_too_soon(tmp_path):
    plan = _window_plan(tmp_path, "id,start\n1,2024-01-10\n", "id,when,amount\n1,2024-01-11,1\n", days=30)
    assert ("1 row had a window running past the last event date, 2024-01-11, so may be undercounted."
            in build_table(plan, tmp_path).log[2]["note"])


def test_window_refuses_repeated_units(tmp_path):
    with pytest.raises(RecipeError, match="Roll that table up"):
        build_table(_window_plan(tmp_path, "id,start\n1,2024-01-10\n1,2024-01-11\n", EVENTS), tmp_path)


def test_ambiguous_dates_must_be_stated(tmp_path):
    people = "id,start\n1,03/04/2024\n"
    events = "id,when,amount\n1,2024-03-05,1\n1,2024-04-05,2\n"
    with pytest.raises(RecipeError, match="could be March or April"):
        build_table(_window_plan(tmp_path, people, events), tmp_path)
    march = build_table(_window_plan(tmp_path, people, events, date_order="month_first", days=7), tmp_path)
    april = build_table(_window_plan(tmp_path, people, events, date_order="day_first", days=7), tmp_path)
    assert march.table["total"].tolist() == [1.0] and april.table["total"].tolist() == [2.0]


def test_day_first_setting_does_not_swap_unambiguous_dates(tmp_path):
    (tmp_path / "d.csv").write_text("d,v\n03/04/2024,1\n2024-04-05,2\n2024-05-04,3\n")
    plan = recipe([{"op": "read_table", "file": "d.csv", "header_row": 0, "footer_rows": 0, "into": "t"},
                   {"op": "filter", "table": "t", "column": "d", "test": "less_than", "value": "2024-04-30",
                    "compare_as": "date", "date_order": "day_first"}], "t")
    assert build_table(plan, tmp_path).table["v"].tolist() == ["1", "2"]


def _totals_then(tmp_path, **filter_step):
    (tmp_path / "o.csv").write_text("id,amount\n1,4\n1,6\n2,0\n3,10.00\n4,10.5\n")
    return recipe([{"op": "read_table", "file": "o.csv", "header_row": 0, "footer_rows": 0, "into": "o"},
                   {"op": "roll_up", "table": "o", "by": "id", "into": "per",
                    "measures": [{"how": "sum", "column": "amount", "into": "total"}]},
                   {"op": "filter", "table": "per", "column": "total", **filter_step}], "per")


@pytest.mark.parametrize("filter_step,kept", [
    ({"test": "equals", "value": "10"}, ["1", "3"]),
    ({"test": "not_equals", "value": "0"}, ["1", "3", "4"]),
    ({"test": "starts_with", "value": "10"}, ["1", "3", "4"]),
    ({"test": "equals", "value": "10.00"}, []),
    ({"test": "equals", "value": "10.00", "compare_as": "number"}, ["1", "3"]),
    ({"test": "not_equals", "value": "0.0", "compare_as": "number"}, ["1", "3", "4"]),
])
def test_equality_filters_work_on_numbers_made_by_earlier_steps(tmp_path, filter_step, kept):
    assert build_table(_totals_then(tmp_path, **filter_step), tmp_path).table["id"].tolist() == kept


def test_equality_filter_on_text_and_missing_values_is_unchanged(tmp_path):
    (tmp_path / "t.csv").write_text("id,code\n1,10.0\n2,10\n3,\n")
    plan = recipe([{"op": "read_table", "file": "t.csv", "header_row": 0, "footer_rows": 0, "into": "t"},
                   {"op": "filter", "table": "t", "column": "code", "test": "not_equals", "value": "10"}], "t")
    assert build_table(plan, tmp_path).table["id"].tolist() == ["1", "3"]


def test_date_equality_uses_the_date_order(tmp_path):
    (tmp_path / "d.csv").write_text("d,v\n03/04/2024,1\n2024-04-03,2\n2024-03-04,3\n")
    plan = recipe([{"op": "read_table", "file": "d.csv", "header_row": 0, "footer_rows": 0, "into": "t"},
                   {"op": "filter", "table": "t", "column": "d", "test": "equals", "value": "2024-04-03",
                    "compare_as": "date", "date_order": "day_first"}], "t")
    assert build_table(plan, tmp_path).table["v"].tolist() == ["1", "2"]


def _date_filter(tmp_path, value, date_order):
    (tmp_path / "d.csv").write_text("d,v\n20/03/2024,1\n10/04/2024,2\n")
    return recipe([{"op": "read_table", "file": "d.csv", "header_row": 0, "footer_rows": 0, "into": "t"},
                   {"op": "filter", "table": "t", "column": "d", "test": "at_least", "value": value,
                    "compare_as": "date", "date_order": date_order}], "t")


@pytest.mark.parametrize("date_order,kept", [("day_first", ["2"]), ("month_first", ["1", "2"])])
def test_typed_cutoff_follows_the_stated_date_order(tmp_path, date_order, kept):
    assert build_table(_date_filter(tmp_path, "03/04/2024", date_order), tmp_path).table["v"].tolist() == kept


def test_ambiguous_cutoff_must_be_stated(tmp_path):
    (tmp_path / "d.csv").write_text("d,v\n2024-03-20,1\n2024-04-10,2\n")
    plan = recipe([{"op": "read_table", "file": "d.csv", "header_row": 0, "footer_rows": 0, "into": "t"},
                   {"op": "filter", "table": "t", "column": "d", "test": "at_least", "value": "03/04/2024",
                    "compare_as": "date"}], "t")
    with pytest.raises(RecipeError, match="could be March or April"):
        build_table(plan, tmp_path)
    plan.steps[1].value = "2024-04-03"
    assert build_table(plan, tmp_path).table["v"].tolist() == ["2"]


def test_fallback_start_follows_the_stated_date_order(tmp_path):
    people, events = "id,start\n1,\n", "id,when,amount\n1,2024-03-05,1\n1,2024-04-05,2\n"
    april = _window_plan(tmp_path, people, events, date_order="day_first", default_start="03/04/2024", days=7)
    assert build_table(april, tmp_path).table["total"].tolist() == [2.0]


def test_descriptions_show_date_choices(tmp_path):
    text = describe(_date_filter(tmp_path, "03/04/2024", "day_first").steps[1])
    assert "(as a date)" in text and "day first, as 3 April 2024" in text
    window = describe(_window_plan(tmp_path, "id,start\n1,2024-01-10\n", EVENTS, date_order="month_first").steps[2])
    assert "dated by 'when'" in window and "same 'id'" in window and "month first, as 4 March 2024" in window


def test_numeric_style_dates_are_understood(tmp_path):
    plan = _window_plan(tmp_path, "id,start\n1,20240110\n", "id,when,amount\n1,20240112,3\n")
    assert build_table(plan, tmp_path).table["total"].tolist() == [3.0]


def _sales(tmp_path, body="q,p\n3,1.10\n2,0\n"):
    (tmp_path / "s.csv").write_text(body)
    return {"op": "read_table", "file": "s.csv", "header_row": 0, "footer_rows": 0, "into": "s"}


def test_calculate_multiplies_without_float_noise(tmp_path):
    plan = recipe([_sales(tmp_path), {"op": "calculate", "table": "s", "first": "q", "operation": "multiply",
                                      "second": "p", "into": "revenue"}], "s")
    assert build_table(plan, tmp_path).table["revenue"].tolist() == [3.3, 0.0]


def test_calculate_says_when_it_cannot(tmp_path):
    plan = recipe([_sales(tmp_path), {"op": "calculate", "table": "s", "first": "q", "operation": "divide",
                                      "second": "p", "into": "r"}], "s")
    built = build_table(plan, tmp_path)
    assert pd.isna(built.table["r"].iloc[1])
    assert built.log[1]["note"] == "1 row couldn't be calculated and left empty."


def test_calculate_refuses_text_and_existing_names(tmp_path):
    plan = recipe([_sales(tmp_path, "q,p\nabc,1\n"), {"op": "calculate", "table": "s", "first": "q",
                                                      "operation": "add", "second": "p", "into": "r"}], "s")
    with pytest.raises(RecipeError, match="needs to be numbers"):
        build_table(plan, tmp_path)
    plan = recipe([_sales(tmp_path), {"op": "calculate", "table": "s", "first": "q", "operation": "add",
                                      "second": "p", "into": "q"}], "s")
    with pytest.raises(RecipeError, match="already has a column"):
        build_table(plan, tmp_path)


def test_reshape_turns_date_columns_into_rows(tmp_path):
    (tmp_path / "w.csv").write_text("store,2024-01,2024-02\nA,1,\nB,3,4\n")
    plan = recipe([{"op": "read_table", "file": "w.csv", "header_row": 0, "footer_rows": 0, "into": "w"},
                   {"op": "reshape_long", "table": "w", "id_columns": ["store"],
                    "value_columns": ["2024-01", "2024-02"], "names_to": "month", "values_to": "sales",
                    "into": "long"}], "long")
    built = build_table(plan, tmp_path)
    assert built.table.shape == (4, 3)
    assert built.table[built.table.store == "B"]["sales"].tolist() == ["3", "4"]
    assert built.log[1]["note"] == "1 new row with an empty value."
    assert (built.log[1]["rows_before"], built.log[1]["rows_after"]) == (2, 4)


def test_reshape_refuses_a_name_already_in_use(tmp_path):
    (tmp_path / "w.csv").write_text("store,a\nA,1\n")
    plan = recipe([{"op": "read_table", "file": "w.csv", "header_row": 0, "footer_rows": 0, "into": "w"},
                   {"op": "reshape_long", "table": "w", "id_columns": ["store"], "value_columns": ["a"],
                    "names_to": "store", "values_to": "v", "into": "long"}], "long")
    with pytest.raises(RecipeError, match="already one of the columns"):
        build_table(plan, tmp_path)


def test_number_comparison_needs_a_stated_type(tmp_path):
    plan = recipe(_two_tables(tmp_path) + [{"op": "filter", "table": "a", "column": "v", "test": "at_least",
                                            "value": "10"}], "a")
    with pytest.raises(RecipeError, match="compares numbers or dates"):
        build_table(plan, tmp_path)
