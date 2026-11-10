"""AI-proposed recipes and masking. The model is replaced with a stand-in, so no data goes anywhere."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main, proposer, translator
from app.privacy import masked_profile, safe_examples, shape_of, should_mask
from app.schemas import DataRecipe

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("test_ingest_for_proposer", ROOT / "tests" / "test_ingest.py")
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)
tool = helpers.tool
client = TestClient(main.app)

PEOPLE = "customer,email,group,age\n" + "".join(f"C-{i:05d},p{i}@mail.example,{'a' if i % 2 else 'b'},{30 + i % 9}\n"
                                                 for i in range(40))
ORDERS = "customer_ref,amount\nC00001,5.25\nC00002,7.5\n"


def csv_file(name, text):
    return ("files", (name, text.encode(), "text/csv"))


def started():
    got = client.post("/api/sources", files=[csv_file("people.csv", PEOPLE), csv_file("orders.csv", ORDERS)]).json()
    return got["source_id"], got["steps"]


GOOD = [{"op": "clean_key", "table": "people", "column": "customer", "rule": "digits_only", "into": "cid"},
        {"op": "clean_key", "table": "orders", "column": "customer_ref", "rule": "digits_only", "into": "cid"}]


def stub(monkeypatch, answer):
    sent = []

    def fake(payload):
        sent.append(payload)
        return answer

    monkeypatch.setattr(proposer, "_ask_model", fake)
    return sent


def ask(source_id, steps, goal=""):
    return client.post(f"/api/sources/{source_id}/propose", json={"steps": steps, "goal": goal})


# ---- masking

@pytest.mark.parametrize("name,values,masked", [
    ("customer_id", ["1", "2", "3"], True),
    ("Customer ID", ["C-1", "C-2"], True),
    ("cust_ref", ["a", "b"], True),
    ("contact", ["a@b.co", "c@d.co"], True),
    ("mobile", [f"0770090{i:04d}" for i in range(30)], True),
    ("notes", [f"comment number {i}" for i in range(30)], True),
    ("order_key", [str(1000 + i) for i in range(30)], True),
    ("segment", ["a", "b"] * 15, False),
    ("campaign_name", ["Spring", "Fall"] * 15, False),
    ("zip_code", ["Urban", "Rural"] * 15, False),
    ("amount", [f"{i}.25" for i in range(30)], False),
    ("signed_up", ["2024-01-01 09:15:00", "2024-01-02 11:20:00"] * 15, False),
])
def test_what_gets_masked(name, values, masked):
    assert should_mask(name, pd.Series(values)) is masked


def test_masked_ids_keep_their_format_only():
    assert shape_of("C-00123") == "X-99999" and shape_of("ab@cd.com") == "<email>"
    examples, masked = safe_examples("customer_id", pd.Series(["C-00123", "C-00456"]))
    assert masked and examples == ["X-99999"]


def test_typed_questions_no_longer_send_real_ids(monkeypatch):
    seen = {}

    def fake(question, profile, design):
        seen["profile"] = profile
        return None

    monkeypatch.setattr(translator, "_llm_spec", fake)
    df = pd.DataFrame({"customer_id": [f"C-{i:05d}" for i in range(40)], "group": ["a", "b"] * 20, "spend": range(40)})
    translator.interpret("Did b spend more?", df)
    text = json.dumps(seen["profile"])
    assert "C-00001" not in text and "X-99999" in text
    assert next(c for c in seen["profile"] if c["name"] == "group")["examples"] == ["a", "b"]


def test_browser_still_sees_real_values():
    df = pd.DataFrame({"customer_id": ["C-1", "C-2"], "v": [1, 2]})
    profile = translator.profile_columns(df)
    masked_profile(df, profile)
    assert profile[0]["examples"] == ["C-1", "C-2"]


# ---- proposals

def test_model_sees_structure_but_no_identifying_values(monkeypatch):
    sent = stub(monkeypatch, {"steps": GOOD, "output": "people", "assumptions": []})
    source_id, reads = started()
    assert ask(source_id, reads, "Did the emails work?").status_code == 200
    text = json.dumps(sent[0])
    assert "C-00001" not in text and "C00001" not in text and "mail.example" not in text
    assert "X-99999" in text and "X99999" in text and "<email>" in text
    assert sent[0]["goal"] == "Did the emails work?" and {t["table"] for t in sent[0]["tables"]} == {"people", "orders"}
    assert "rows" in sent[0]["tables"][0] and "sample" not in text


def test_proposals_are_never_confirmed_even_if_the_model_says_so(monkeypatch):
    stub(monkeypatch, {"steps": [{**GOOD[0], "status": "confirmed"}], "output": "people", "assumptions": []})
    source_id, reads = started()
    assert [s["status"] for s in ask(source_id, reads).json()["steps"]] == ["proposed"]


def test_proposals_come_back_as_proposed_with_assumptions(monkeypatch):
    stub(monkeypatch, {"steps": GOOD, "output": "people", "assumptions": ["No order means no purchase."]})
    source_id, reads = started()
    body = ask(source_id, reads).json()
    assert {s["status"] for s in body["steps"]} == {"proposed"} and body["output"] == "people"
    assert body["assumptions"] == ["No order means no purchase."] and body["problem"] is None


def test_a_proposed_recipe_cannot_be_used_until_confirmed(monkeypatch):
    stub(monkeypatch, {"steps": GOOD, "output": "people", "assumptions": []})
    source_id, reads = started()
    body = ask(source_id, reads).json()
    response = client.post(f"/api/sources/{source_id}/confirm", json={
        "recipe": {"steps": [{**s, "status": "confirmed"} for s in reads] + body["steps"], "output": "people"},
        "unit_column": "cid", "group_column": "group"})
    assert response.status_code == 400 and "still need your confirmation" in response.json()["detail"]


@pytest.mark.parametrize("bad", [
    [{"op": "read_table", "file": "people.csv", "header_row": 0, "footer_rows": 0, "into": "x"}],
    [{"op": "run_python", "code": "import os"}],
    "not a list",
])
def test_steps_outside_the_fixed_set_are_rejected(monkeypatch, bad):
    stub(monkeypatch, {"steps": bad, "output": "x", "assumptions": []})
    source_id, reads = started()
    response = ask(source_id, reads)
    assert response.status_code == 502 and "by hand" in response.json()["detail"]


def test_a_broken_suggestion_keeps_only_the_steps_that_ran(monkeypatch):
    broken = [GOOD[0], {"op": "fill_missing", "table": "people", "column": "nope", "value": "0"}, GOOD[1]]
    stub(monkeypatch, {"steps": broken, "output": "people", "assumptions": []})
    source_id, reads = started()
    body = ask(source_id, reads).json()
    assert len(body["steps"]) == 1 and "only the 1 before it" in body["problem"] and "'nope'" in body["problem"]


def test_unknown_final_table_is_left_for_the_person(monkeypatch):
    stub(monkeypatch, {"steps": GOOD, "output": "missing", "assumptions": []})
    source_id, reads = started()
    assert ask(source_id, reads).json()["output"] == ""


def test_only_read_steps_can_be_sent(monkeypatch):
    stub(monkeypatch, {"steps": GOOD, "output": "people", "assumptions": []})
    source_id, reads = started()
    response = ask(source_id, reads + GOOD)
    assert response.status_code == 400 and "remove any steps you added" in response.json()["detail"]


def test_no_key_says_how_to_continue(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    source_id, reads = started()
    response = ask(source_id, reads)
    assert response.status_code == 502 and "add the steps by hand" in response.json()["detail"]


def test_ai_is_told_what_each_step_does():
    text = json.dumps(proposer._tool())
    assert "removes leading zeros" in text and "Every left row is kept" in text
    assert "counting the start day" in text and "doesn't remove overlapping rows" in text
    assert "Don't tell the person a step has a limit" in proposer.SYSTEM


def test_tool_schema_cannot_confirm_or_read():
    text = json.dumps(proposer._tool())
    assert '"status"' not in text and "read_table" not in text and "discriminator" not in text
    assert {"steps", "output", "assumptions"} == set(proposer._tool()["input_schema"]["required"])


def test_hillstrom_proposal_rebuilds_the_answer_once_confirmed(monkeypatch, tmp_path):
    pytest.importorskip("openpyxl")
    import numpy as np

    rng = np.random.default_rng(9)
    n = 300
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
    reads = helpers.hillstrom_steps()[:4]
    later = helpers.hillstrom_steps()[4:]
    sent = stub(monkeypatch, {"steps": later, "output": "customers", "assumptions": []})
    result = proposer.propose(DataRecipe(steps=reads, output="customers"), tmp_path)
    assert "C-00001" not in json.dumps(sent[0])
    final = DataRecipe(steps=[{**s, "status": "confirmed"} for s in reads + result["steps"]], output="customers")
    assert tool.same_table(helpers.as_analyzed(helpers.build_table(final, tmp_path)), answer)
