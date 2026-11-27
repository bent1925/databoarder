"""Shared by the browser tests: the app running on a free port, and one Chromium for the whole run."""
import socket
import threading
import time

import pytest

sync_api = pytest.importorskip("playwright.sync_api", reason="Playwright isn't installed (see the test files' docstrings)")
import uvicorn  # noqa: E402

from app import main  # noqa: E402


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
