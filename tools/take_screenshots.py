"""Retakes the README screenshots in docs/screenshots/ from the running app, so they can be refreshed after UI changes.

    .venv/bin/python tools/take_screenshots.py

Needs Playwright's Chromium (see tests/browser/). Starts the app on a free local port with no API key, so only
fixed code runs: the ready-made questions and the combine screen's suggestions don't use the AI. Uses the samples in
sample_data/, plus the messy Hillstrom pull in local_data/hillstrom/pull_full/ for the combine screen (skipped,
with a message, if local_data/ isn't there).
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("APP_PASSWORD", None)

import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from app import main  # noqa: E402

OUT = ROOT / "docs" / "screenshots"
PULL = ROOT / "local_data" / "hillstrom" / "pull_full"
SAMPLE_BUTTON = {"checkout_test.csv": "#sample", "signup_page_change.csv": "#sample2",
                 "regional_rollout.csv": "#sample3", "loyalty_program.csv": "#sample4"}


def serve() -> tuple[uvicorn.Server, str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    return server, f"http://127.0.0.1:{port}"


def set_up(page, url, sample, mode, design):
    """Loads a sample, picks how careful and how people got it, and waits for the ready-made questions."""
    page.goto(url)
    page.click(SAMPLE_BUTTON[sample])
    page.wait_for_selector("#s2:not(.hidden) .mode")
    page.click(f'#modes [data-m="{mode}"]')
    if design:
        page.click(f'#designs [data-d="{design}"]')
    page.wait_for_selector("#qlist .qopt:not([data-id=other])")
    page.wait_for_load_state("networkidle")


def report(page, change_date=None):
    """Presses Go; confirms the read-back when it's shown; waits for the report."""
    page.click("#go")
    page.wait_for_selector("#report:not(.hidden), #s4:not(.hidden)")
    if page.is_visible("#s4") and not page.is_visible("#report"):
        if change_date and page.is_visible("#f_when") and not page.input_value("#f_when"):
            page.fill("#f_when", change_date)
            page.dispatch_event("#f_when", "change")
            page.wait_for_load_state("networkidle")
        page.click("#runbtn")
    page.wait_for_selector("#report:not(.hidden) .tabbar")
    page.wait_for_load_state("networkidle")
    return page.locator("#report")


def main_() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    server, url = serve()
    errors: list[str] = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={"width": 760, "height": 1000}, device_scale_factor=2)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))

            set_up(page, url, "loyalty_program.csv", "right_answer", "chose")
            page.screenshot(path=OUT / "1-setup.png", full_page=True)
            report(page).screenshot(path=OUT / "2-people-chose-report.png")

            set_up(page, url, "regional_rollout.csv", "right_answer", "one_date_comparison")
            report(page, change_date="2026-08-30").screenshot(path=OUT / "3-comparison-group-report.png")

            set_up(page, url, "checkout_test.csv", "quick", None)
            report(page).screenshot(path=OUT / "4-ab-test-quick.png")

            if PULL.is_dir():
                page.goto(url)
                page.click("#openmerge")
                page.fill("#cq", "Did the email get more people to visit the site within two weeks?")
                page.set_input_files("#cfile", sorted(str(p) for p in PULL.iterdir() if p.suffix in (".csv", ".xlsx")))
                page.wait_for_selector("#ctiles")
                page.wait_for_load_state("networkidle")
                page.locator("#c1").screenshot(path=OUT / "5-combine.png")
            else:
                print(f"Skipped 5-combine.png: {PULL.relative_to(ROOT)} isn't here (see the README's Run section).")
            browser.close()
    finally:
        server.should_exit = True
    if errors:
        sys.exit("The page reported errors, so the screenshots may be wrong:\n" + "\n".join(errors))
    print(f"Saved to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main_()
