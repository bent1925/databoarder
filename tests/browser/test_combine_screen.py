"""The combine screen in a real browser: upload a messy pull, confirm the choices, and reach the question screen.

Needs Playwright and its Chromium browser:
    .venv/bin/pip install -r requirements-dev.txt
    .venv/bin/python -m playwright install chromium
Without them these tests are skipped, so the rest of the suite still runs anywhere.
"""
import importlib.util
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sync_api = pytest.importorskip("playwright.sync_api", reason="Playwright isn't installed (see this file's docstring)")
pytest.importorskip("openpyxl")
import uvicorn  # noqa: E402

from app import main  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("make_messy_hillstrom_browser", ROOT / "tools" / "make_messy_hillstrom.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


@pytest.fixture(scope="module")
def pulls(tmp_path_factory):
    rng = np.random.default_rng(21)
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
    folder = tmp_path_factory.mktemp("pulls")
    pull = tool.build_pull(answer)
    tool.write_pull(pull, folder / "full")
    tool.write_pull(tool.treated_only(pull), folder / "treated")
    return {name: sorted(str(path) for path in (folder / name).iterdir()) for name in ("full", "treated")}


@pytest.fixture(scope="module")
def server():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    running = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not running.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    running.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        try:
            launched = playwright.chromium.launch()
        except Exception as problem:     # the browser itself hasn't been downloaded
            pytest.skip(f"Chromium for Playwright isn't installed ({type(problem).__name__}). See this file's docstring.")
        yield launched
        launched.close()


@pytest.fixture
def page(browser, server):
    context = browser.new_context()
    opened = context.new_page()
    errors = []
    opened.on("pageerror", lambda error: errors.append(str(error)))
    # A 400 the page handles (like "say whether the day or the month comes first") is logged by the browser too.
    opened.on("console", lambda message: message.type == "error" and not message.text.startswith(
        "Failed to load resource") and errors.append(message.text))
    opened.goto(server)
    yield opened
    context.close()
    assert errors == [], errors


def start(page, files, question=""):
    page.click("#openmerge")
    if question:
        page.fill("#cq", question)
    page.set_input_files("#cfile", files)


def tiles(page):
    page.wait_for_selector("#ctiles")
    return dict(zip(page.locator("#ctiles .k").all_inner_texts(), page.locator("#ctiles .v").all_inner_texts()))


def test_confirm_all_builds_the_table_and_reaches_the_questions(page, pulls):
    start(page, pulls["full"], "Did the email get more people to visit?")
    numbers = tiles(page)
    assert numbers["People"] == "300" and "Not sent" in numbers
    assert page.locator(".choice").count() == 4
    assert page.locator(".choice .pill").all_inner_texts() == ["Suggested"] * 4
    assert page.is_disabled("#cuse")

    page.click("#call")
    page.wait_for_function("!document.querySelector('#cuse').disabled")
    assert page.locator(".choice .pill").all_inner_texts() == ["✓ Confirmed by you"] * 4
    page.click("#cuse")
    page.wait_for_selector("#s3:not(.hidden)")
    info = page.inner_text("#fileinfo")
    assert "campaign_group says who got the campaign" in info
    assert page.locator("#decided li").count() == 4
    assert page.input_value("#qother") == "Did the email get more people to visit?"


def test_changing_the_window_changes_the_numbers(page, pulls):
    start(page, pulls["full"])
    before = tiles(page)["Any site visits"]
    window = page.locator('[data-choice="window"]')
    window.get_by_role("button", name="Change").click()
    window.locator("label.copt").filter(has_text="7 days").click()
    page.wait_for_function(f"document.querySelector('#ctiles') && !document.querySelector('#ctiles').innerText.includes('{before}')")
    assert tiles(page)["Any site visits"] != before
    page.locator('[data-choice="window"]').get_by_role("button", name="Use this answer").click()
    page.wait_for_selector('[data-choice="window"].done')


def test_asking_the_analyst_shows_a_note_and_waits(page, pulls):
    start(page, pulls["full"])
    tiles(page)
    missing = page.locator('[data-choice="missing"]')
    missing.get_by_role("button", name="Change").click()
    missing.locator("label.copt").filter(has_text="Ask my analyst").click()
    page.wait_for_selector("#cnote")
    assert "People missing from" in page.inner_text("#cletter")
    page.click("#call")
    page.wait_for_selector('[data-choice="people"].done')
    assert page.is_disabled("#cuse")


def test_a_pull_with_only_emailed_people_is_blocked(page, pulls):
    start(page, pulls["treated"])
    page.wait_for_selector("[data-blocked]")
    assert "no one to compare with" in page.inner_text("[data-blocked]")
    assert "held out of the campaign" in page.inner_text("#cletter")
    assert page.is_disabled("#cuse")


def test_the_step_by_step_builder_is_still_there(page, pulls):
    start(page, pulls["full"])
    tiles(page)
    page.click("#cbuilder")
    page.wait_for_selector("#m1:not(.hidden) #msteps .mstep")
    assert page.is_hidden("#c1")


def test_a_table_that_fails_to_build_says_why_and_how_to_proceed(page, tmp_path):
    people = "id,age_band\n" + "".join(f"C-{i:03d},{20 + 10 * (i % 4)}\n" for i in range(1, 41))
    sends = "cust,version,sent\n" + "".join(f"{i},{'A' if i % 2 else 'B'},2024-03-01\n" for i in range(1, 21))
    orders = "customer_ref,order_date,amount\n" + "".join(f"{i},2024-03-0{1 + i % 5},{i}.50\n" for i in range(1, 21))
    orders += "21,2024-03-02,N/A\n"
    files = []
    for name, text in (("people.csv", people), ("email_sends.csv", sends), ("orders.csv", orders)):
        (tmp_path / name).write_text(text)
        files.append(str(tmp_path / name))

    def total_the_amount(route):     # an answer whose total can't be worked out, as if a person had picked it
        reply = route.fetch()
        body = reply.json()
        body["answers"]["events"][0]["totals"] = {"amount": "total_amount"}
        route.fulfill(response=reply, body=json.dumps(body))

    page.route("**/answers/suggest", total_the_amount)
    start(page, files)
    page.wait_for_selector("#cproblem")
    problem = page.inner_text("#cproblem")
    assert "couldn't be put together" in problem and "'amount' needs to be numbers" in problem
    assert "Build it step by step instead" in problem
    assert page.locator("#ctiles").count() == 0
    page.click("#call")
    page.wait_for_selector('[data-choice="people"].done')
    assert page.is_disabled("#cuse") and page.inner_text("#cwhy") == "Fix the problem above first"


def test_choosing_the_date_order_updates_the_windows_on_offer(page, tmp_path):
    people = "id,age_band\n" + "".join(f"C-{i:03d},{20 + 10 * (i % 4)}\n" for i in range(1, 41))
    sends = "cust,version,sent\n" + "".join(f"{i},{'A' if i % 2 else 'B'},2024-03-01\n" for i in range(1, 21))
    orders = "customer_ref,order_date,amount\n3,02/03/2024,10.50\n25,03/04/2024,7.25\n"
    files = []
    for name, text in (("people.csv", people), ("email_sends.csv", sends), ("orders.csv", orders)):
        (tmp_path / name).write_text(text)
        files.append(str(tmp_path / name))
    start(page, files)
    page.wait_for_selector("#c-order")
    assert "could be March or April" in page.inner_text("#cproblem")
    page.select_option("#c-order", "day_first")
    page.wait_for_selector("#ctiles")
    window = page.locator('[data-choice="window"]')
    window.get_by_role("button", name="Change").click()
    labels = window.locator("label.copt").all_inner_texts()
    assert any(label.startswith("34 days: 2024-03-01 to 2024-04-03") for label in labels)
    assert any(label.startswith("30 days") for label in labels)

