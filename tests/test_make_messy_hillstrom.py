"""The fake analyst pull must break apart cleanly and rebuild exactly. Uses made-up rows, so no download."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parent.parent / "tools" / "make_messy_hillstrom.py"
_spec = importlib.util.spec_from_file_location("make_messy_hillstrom", PATH)
tool = importlib.util.module_from_spec(_spec)
sys.modules["make_messy_hillstrom"] = tool
_spec.loader.exec_module(tool)


@pytest.fixture
def table():
    rng = np.random.default_rng(1)
    n = 600
    visit = rng.random(n) < 0.3
    bought = visit & (rng.random(n) < 0.5)
    source = pd.DataFrame({
        "recency": rng.integers(1, 13, n),
        "history_segment": rng.choice(["1) $0 - $100", "2) $100 - $200"], n),
        "history": rng.integers(3000, 90000, n) / 100,
        "mens": rng.integers(0, 2, n),
        "womens": rng.integers(0, 2, n),
        "zip_code": rng.choice(["Urban", "Rural"], n),
        "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(["Web", "Phone"], n),
        "segment": rng.choice(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], n),
        "visit": visit.astype(int),
        "conversion": bought.astype(int),
        "spend": np.where(bought, rng.integers(2999, 49900, n) / 100, 0.0),
    })
    return tool.answer_table(source)


def test_absent_means_zero_files_hold_only_positives(table):
    pull = tool.build_pull(table)
    assert set(tool._digits(pull["site_visits"]["visitor_id"])) == set(table.loc[table.visit == 1, "customer_id"])
    assert set(tool._digits(pull["purchases"]["customer_ref"])) == set(table.loc[table.spend > 0, "customer_id"])
    assert set(tool._digits(pull["email_sends"]["cust_id"])) == set(table.loc[table.emailed == 1, "customer_id"])


def test_order_amounts_add_back_to_real_spend_exactly(table):
    purchases = tool.build_pull(table)["purchases"]
    cents = (purchases["amount"] * 100).round().astype(int).groupby(tool._digits(purchases["customer_ref"])).sum()
    expected = (table.set_index("customer_id")["spend"] * 100).round().astype(int)
    assert (purchases["amount"] > 0).all()
    assert cents.to_dict() == expected[expected > 0].to_dict()


def test_each_file_writes_ids_differently(table):
    pull = tool.build_pull(table)
    assert pull["customers"][tool.CUSTOMER_HEADER].str.fullmatch(r"C-\d{5}").all()
    assert pull["email_sends"]["cust_id"].str.fullmatch(r"[1-9]\d*").all()
    assert pull["site_visits"]["visitor_id"].str.fullmatch(r"\d{5}").all()
    assert pull["purchases"]["customer_ref"].str.fullmatch(r"C\d{5}").all()


def test_files_cannot_be_matched_by_row_order(table):
    customers = tool.build_pull(table)["customers"]
    assert not tool._digits(customers[tool.CUSTOMER_HEADER]).is_monotonic_increasing


def test_same_seed_gives_same_pull(table):
    first, second = tool.build_pull(table), tool.build_pull(table)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_treated_only_pull_has_no_comparison_group(table):
    pull = tool.treated_only(tool.build_pull(table))
    emailed = set(table.loc[table.emailed == 1, "customer_id"])
    assert set(tool._digits(pull["customers"][tool.CUSTOMER_HEADER])) == emailed
    assert set(tool._digits(pull["site_visits"]["visitor_id"])) <= emailed
    assert set(tool._digits(pull["purchases"]["customer_ref"])) <= emailed


def test_written_files_rebuild_the_answer_table_exactly(table, tmp_path):
    pytest.importorskip("openpyxl")
    tool.write_pull(tool.build_pull(table), tmp_path)
    assert tool.same_table(tool.rebuild(tmp_path), table)


def test_excel_title_and_total_rows_are_not_data(table, tmp_path):
    pytest.importorskip("openpyxl")
    tool.write_pull(tool.build_pull(table), tmp_path)
    raw = pd.read_excel(tmp_path / tool.XLSX_NAME, header=None, dtype=object)
    assert raw.iloc[0, 0] == "Customer attribute export"
    assert str(raw.dropna(how="all").iloc[-1, 0]).startswith("Total rows")
    assert len(tool.read_customers(tmp_path / tool.XLSX_NAME)) == len(table)


def test_wrong_source_file_is_refused(tmp_path):
    bad = tmp_path / "not_hillstrom.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit, match="not the expected Hillstrom file"):
        tool.read_source(bad)


def test_expected_results_are_labeled_as_estimates(table):
    results = tool.expected_results(table)
    assert "not the exact true effect" in results["note"]
    assert len(results["comparisons"]) == 9
    assert all(r["ci95_low"] <= r["difference"] <= r["ci95_high"] for r in results["comparisons"])


@pytest.mark.parametrize("outcome", ["visit", "spend"])
def test_expected_results_match_the_app(table, outcome):
    from app.analysis import analyze
    from app.schemas import ABTestSpec

    spec = ABTestSpec(group_col="segment", control="No E-Mail", treatment="Mens E-Mail", outcome_col=outcome)
    report = analyze(table, spec).model_dump()
    expected = next(r for r in tool.expected_results(table)["comparisons"]
                    if r["treatment"] == "Mens E-Mail" and r["outcome"] == outcome)
    effect = report["effect"]
    assert effect["diff"] == pytest.approx(expected["difference"])
    assert effect["ci_low"] == pytest.approx(expected["ci95_low"])
    assert effect["ci_high"] == pytest.approx(expected["ci95_high"])
