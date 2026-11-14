"""The one-screen combine flow: suggested answers, what blocks a table, and building from confirmed answers."""
import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import combine, main
from app.schemas import ABTestSpec, TableAnswers

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("test_ingest_helpers_combine", ROOT / "tests" / "test_ingest.py")
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)
tool = helpers.tool

client = TestClient(main.app)


def csv_file(name, text):
    return ("files", (name, text.encode(), "text/csv"))


def upload(*files):
    response = client.post("/api/sources", files=list(files))
    assert response.status_code == 200, response.text
    return response.json()["source_id"]


def suggest(source_id, question=""):
    response = client.post(f"/api/sources/{source_id}/answers/suggest", json={"question": question})
    assert response.status_code == 200, response.text
    return response.json()


def preview(source_id, answers):
    return client.post(f"/api/sources/{source_id}/answers/preview", json={"answers": answers})


def confirm(source_id, answers, question=""):
    return client.post(f"/api/sources/{source_id}/answers/confirm", json={"answers": answers, "question": question})


def confirm_all(answers):
    for name in ("people", "campaign", "window", "missing", "match", "returns"):
        if answers.get(name):
            answers[name]["status"] = "confirmed"
    return answers


# ------------------------------------------------------------------------- a small pull written by hand

PEOPLE = "id,age_band\n" + "".join(f"C-{i:03d},{20 + 10 * (i % 4)}\n" for i in range(1, 41))
SENDS = "cust,version,sent\n" + "".join(f"{i},{'A' if i % 2 else 'B'},2024-03-01\n" for i in range(1, 21))
# Customer 3 (sent) orders on days 1 and 9; customer 25 (not sent) on day 2; customer 30 (not sent) on day 12.
ORDERS = ("customer_ref,order_date,amount\n3,2024-03-01,10.50\n3,2024-03-09,4.50\n25,2024-03-02,7.25\n"
          "30,2024-03-12,3.75\n")


def small(orders=ORDERS, people=PEOPLE, sends=SENDS):
    files = [csv_file("people.csv", people), csv_file("orders.csv", orders)]
    if sends is not None:
        files.insert(1, csv_file("email_sends.csv", sends))
    return upload(*files)


def test_suggestions_find_each_file_and_stay_suggestions():
    got = suggest(small())
    answers = got["answers"]
    assert (answers["people"]["table"], answers["people"]["id_column"]) == ("people", "id")
    campaign = answers["campaign"]
    assert (campaign["kind"], campaign["table"], campaign["id_column"]) == ("list", "email_sends", "cust")
    assert (campaign["versions_column"], campaign["date_column"]) == ("version", "sent")
    assert [(e["table"], e["id_column"], e["date_column"], e["totals"]) for e in answers["events"]] == [
        ("orders", "customer_ref", "order_date", {"amount": "total_amount"})]
    assert answers["window"] == {"status": "suggested", "by": "code", "start_date": "2024-03-01", "days": 12,
                                 "date_order": None}
    assert got["options"]["window_days"] == [7, 12]
    assert {answers[name]["status"] for name in ("people", "campaign", "window", "missing")} == {"suggested"}


def test_question_can_pick_the_window():
    assert suggest(small(), "Did people buy more within a week?")["answers"]["window"]["days"] == 7


def test_preview_shows_numbers_and_every_step_without_building():
    source_id = small()
    shown = preview(source_id, suggest(source_id)["answers"]).json()
    assert shown["error"] is None and shown["blocked"] == []
    assert shown["waiting"] == ["people", "campaign", "window", "missing"] and not shown["ready"]
    numbers = {n["key"]: n["value"] for n in shown["numbers"]}
    assert numbers["people"] == "40" and numbers["group:Not sent"] == "50%" and numbers["any_orders"] == "7.5%"
    assert any("keeping only the digits" in fix["text"] for fix in shown["fixes"])
    assert shown["group_column"] == "campaign_group"


def test_a_shorter_window_changes_what_counts():
    source_id = small()
    answers = suggest(source_id)["answers"]
    answers["window"]["days"] = 7
    numbers = {n["key"]: n["value"] for n in preview(source_id, answers).json()["numbers"]}
    assert numbers["any_orders"] == "5.0%"          # customer 30's order on day 12 falls outside
    assert numbers["total_amount"] == f"{(10.50 + 7.25) / 40:,.2f}"    # customer 3's day-9 order too


def test_a_window_past_the_end_of_the_data_is_refused():
    source_id = small()
    answers = confirm_all(suggest(source_id)["answers"])
    answers["window"]["days"] = 30
    shown = preview(source_id, answers).json()
    assert "end on 2024-03-12, so a 30-day window isn't complete" in shown["blocked"][0]
    assert confirm(source_id, answers).status_code == 400


def test_confirm_refuses_answers_still_suggested():
    source_id = small()
    response = confirm(source_id, suggest(source_id)["answers"])
    assert response.status_code == 400 and "Still to confirm: people, campaign, window, missing" in response.json()["detail"]


def test_confirmed_answers_build_the_table_and_record_the_decisions():
    source_id = small()
    response = confirm(source_id, confirm_all(suggest(source_id)["answers"]), "Did the email work?")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["confirmed"] == {"group_col": "campaign_group", "date_col": None}
    source = result["source"]
    assert source["answers"]["campaign"]["status"] == "confirmed" and source["question"] == "Did the email work?"
    assert source["decisions"][0].startswith("Who belongs in the table: everyone in people")
    table = main.DATASETS[result["dataset_id"]]["df"].set_index("person_id")
    assert table.loc[3, "campaign_group"] == "A" and table.loc[25, "campaign_group"] == "Not sent"
    assert table.loc[3, "total_amount"] == 15.0 and table.loc[30, "any_orders"] == 1 and table.loc[1, "any_orders"] == 0


def test_answers_naming_made_up_tables_or_columns_are_refused():
    source_id = small()
    answers = suggest(source_id)["answers"]
    answers["events"][0]["totals"] = {"profit": "total_profit"}
    response = preview(source_id, answers)
    assert response.status_code == 400 and response.json()["detail"] == "The table 'orders' has no column called 'profit'."
    answers = suggest(source_id)["answers"]
    answers["people"]["table"] = "secrets"
    assert "no table called 'secrets'" in preview(source_id, answers).json()["detail"]


def test_missing_rows_can_wait_for_the_analyst():
    source_id = small()
    answers = confirm_all(suggest(source_id)["answers"])
    answers["missing"]["means"] = "ask"
    shown = preview(source_id, answers).json()
    assert not shown["ready"] and "People missing from orders" in shown["note"]
    assert confirm(source_id, answers).status_code == 400


def test_everyone_on_the_campaign_list_blocks_and_drafts_a_note():
    sends = "cust,version,sent\n" + "".join(f"{i},A,2024-03-01\n" for i in range(1, 41))
    source_id = small(sends=sends)
    answers = confirm_all(suggest(source_id)["answers"])
    shown = preview(source_id, answers).json()
    assert "there's no one to compare with" in shown["blocked"][0]
    assert "people who were held out of the campaign" in shown["note"]
    assert "no one to compare with" in confirm(source_id, answers).json()["detail"]


def test_many_unmatched_rows_become_a_choice():
    orders = ORDERS + "901,2024-03-02,5.00\n902,2024-03-03,5.00\n"
    source_id = small(orders=orders)
    answers = confirm_all(suggest(source_id)["answers"])
    shown = preview(source_id, answers).json()
    assert shown["waiting"] == ["match"] and "2 of 6 rows in orders match no one" in shown["risks"][0]["text"]
    answers["match"] = {"status": "confirmed", "by": "person", "action": "ask"}
    shown = preview(source_id, answers).json()
    assert not shown["ready"] and "don't match anyone in people" in shown["note"]
    answers["match"]["action"] = "leave_out"
    assert preview(source_id, answers).json()["ready"]
    assert confirm(source_id, answers).status_code == 200


def test_ids_that_differ_only_by_letters_block_the_table():
    orders = "customer_ref,order_date,amount\nB-003,2024-03-01,10.50\nB-025,2024-03-02,7.25\n"
    source_id = small(orders=orders)
    shown = preview(source_id, confirm_all(suggest(source_id)["answers"])).json()
    assert "different letters" in shown["blocked"][0] and "A-123 and B-123" in shown["note"]


def test_negative_amounts_become_a_choice_and_can_be_left_out():
    source_id = small(orders=ORDERS + "3,2024-03-02,-4.00\n")
    answers = confirm_all(suggest(source_id)["answers"])
    shown = preview(source_id, answers).json()
    assert shown["waiting"] == ["returns"] and "negative 'amount'" in shown["risks"][0]["text"]
    answers["returns"] = {"status": "confirmed", "by": "person", "action": "count"}
    counted = {n["key"]: n["value"] for n in preview(source_id, answers).json()["numbers"]}
    answers["returns"]["action"] = "leave_out"
    left_out = {n["key"]: n["value"] for n in preview(source_id, answers).json()["numbers"]}
    assert counted["total_amount"] == f"{(10.50 + 4.50 + 7.25 + 3.75 - 4.00) / 40:,.2f}"
    assert left_out["total_amount"] == f"{(10.50 + 4.50 + 7.25 + 3.75) / 40:,.2f}"


def test_people_listed_twice_block_the_table():
    source_id = small()
    answers = confirm_all(suggest(source_id)["answers"])
    answers["people"] = {"status": "confirmed", "by": "person", "table": "orders", "id_column": "customer_ref"}
    answers["events"] = []
    shown = preview(source_id, answers).json()
    assert any("appear more than once in orders" in text for text in shown["blocked"])


def test_a_group_column_in_the_people_file_is_used_when_there_is_no_list():
    people = "id,got_email,age_band\n" + "".join(f"C-{i:03d},{'yes' if i <= 20 else 'no'},{20 + i % 3}\n"
                                                 for i in range(1, 41))
    source_id = small(people=people, sends=None)
    got = suggest(source_id)
    assert got["answers"]["campaign"] == {"status": "suggested", "by": "code", "kind": "column", "table": None,
                                          "id_column": None, "versions_column": "got_email", "versions": "separate",
                                          "none_label": "Not sent", "date_column": None}
    assert "earliest event date, 2024-03-01" in got["notes"][0]
    result = confirm(source_id, confirm_all(got["answers"])).json()
    assert result["confirmed"]["group_col"] == "got_email"


def test_files_that_dont_say_who_got_the_campaign_stop_with_a_note():
    people = "id,age_band\n" + "".join(f"C-{i:03d},{20 + i % 3}\n" for i in range(1, 41))
    got = suggest(small(people=people, sends=None))
    assert got["answers"] is None and "don't show who got the campaign" in got["blocked"][0]
    assert "list of people who got the campaign" in got["note"]


# ------------------------------------------------------------------------------ the Hillstrom test pull

@pytest.fixture
def hillstrom(tmp_path):
    pytest.importorskip("openpyxl")
    rng = np.random.default_rng(11)
    n = 300
    visit = rng.random(n) < 0.4
    bought = visit & (rng.random(n) < 0.5)
    answer = tool.answer_table(pd.DataFrame({
        "recency": rng.integers(1, 13, n), "history_segment": rng.choice(["1) $0 - $100", "2) $100 - $200"], n),
        "history": rng.integers(3000, 90000, n) / 100, "mens": rng.integers(0, 2, n), "womens": rng.integers(0, 2, n),
        "zip_code": rng.choice(["Urban", "Rural"], n), "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(["Web", "Phone"], n), "segment": rng.choice(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], n),
        "visit": visit.astype(int), "conversion": bought.astype(int),
        "spend": np.where(bought, rng.integers(2999, 49900, n) / 100, 0.0)}))
    pull = tool.build_pull(answer)
    tool.write_pull(pull, tmp_path / "full")
    tool.write_pull(tool.treated_only(pull), tmp_path / "treated")
    return tmp_path, answer


def _upload_folder(folder):
    return upload(*[("files", (path.name, path.read_bytes(), "application/octet-stream"))
                    for path in sorted(folder.iterdir())])


ANSWER_NAMES = {"person_id": "customer_id", "campaign_group": "segment", "got_campaign": "emailed",
                "any_site_visits": "visit", "any_purchases": "conversion", "total_amount": "spend"}


def test_hillstrom_suggestions_rebuild_the_answer_table(hillstrom):
    folder, answer = hillstrom
    source_id = _upload_folder(folder / "full")
    got = suggest(source_id)
    answers = got["answers"]
    assert answers["people"]["table"] == "customer_export" and answers["campaign"]["table"] == "email_sends"
    assert answers["campaign"]["versions_column"] == "campaign"
    assert sorted(e["table"] for e in answers["events"]) == ["purchases", "site_visits"]
    assert preview(source_id, answers).json()["blocked"] == []

    result = confirm(source_id, confirm_all(answers)).json()
    frozen = client.get(f"/api/datasets/{result['dataset_id']}/snapshot").content
    rebuilt = pd.read_csv(io.BytesIO(frozen)).rename(columns=ANSWER_NAMES)
    rebuilt["segment"] = rebuilt["segment"].replace("Not sent", tool.NO_EMAIL)
    rebuilt = rebuilt.sort_values("customer_id").reset_index(drop=True)
    columns = list(answer.columns)
    pd.testing.assert_frame_equal(rebuilt[columns], answer[columns], check_dtype=False)


def test_hillstrom_with_only_emailed_people_is_blocked(hillstrom):
    folder, _ = hillstrom
    source_id = _upload_folder(folder / "treated")
    shown = preview(source_id, confirm_all(suggest(source_id)["answers"])).json()
    assert "there's no one to compare with" in shown["blocked"][0] and shown["note"]


def test_script_from_answers_lists_the_decisions_and_reproduces_the_report(hillstrom):
    folder, _ = hillstrom
    source_id = _upload_folder(folder / "full")
    result = confirm(source_id, confirm_all(suggest(source_id)["answers"])).json()
    spec = ABTestSpec(group_col="campaign_group", control="Not sent", treatment="Mens E-Mail",
                      outcome_col="any_site_visits").model_dump()
    report = client.post("/api/run", json={"dataset_id": result["dataset_id"], "spec": spec, "mode": "quick"}).json()
    script = report["script"]
    assert "# You decided:\n#   - Who belongs in the table: everyone in customer_export" in script

    (folder / "full" / "repro.py").write_text(script)
    run = subprocess.run([sys.executable, "repro.py"], cwd=folder / "full", check=True, capture_output=True, text=True)
    frozen = client.get(f"/api/datasets/{result['dataset_id']}/snapshot").content
    assert (folder / "full" / main.FROZEN_NAME).read_bytes() == frozen
    assert f"{report['effect']['diff']:.4f}"[:5] in run.stdout


def test_compiled_recipe_reads_only_the_files_it_uses(tmp_path):
    (tmp_path / "people.csv").write_text(PEOPLE)
    (tmp_path / "email_sends.csv").write_text(SENDS)
    (tmp_path / "orders.csv").write_text(ORDERS)
    (tmp_path / "notes.csv").write_text("topic,text\na,b\nc,d\n")
    paths = sorted(tmp_path.iterdir())
    answers = TableAnswers.model_validate(combine.suggest(paths)["answers"])
    steps, tables = combine.read_files(paths)
    recipe = combine.compile_recipe(answers, steps, tables)
    assert sorted(step.file for step in recipe.steps if step.op == "read_table") == [
        "email_sends.csv", "orders.csv", "people.csv"]
    assert {step.status for step in recipe.steps} == {"proposed"}
