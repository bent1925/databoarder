import base64

from fastapi.testclient import TestClient

from app import main
from app.main import app

c = TestClient(app)


def basic(pw, user="anyone"):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}


def test_open_when_no_password_is_set(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    assert c.get("/").status_code == 200 and c.get("/healthz").json() == {"ok": True}


def test_password_protects_pages_and_api(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    for path in ("/", "/api/suggestions?dataset_id=x", "/static/index.html"):
        r = c.get(path)
        assert r.status_code == 401 and "Basic" in r.headers["www-authenticate"]
    assert c.post("/api/ask", json={"dataset_id": "x", "question": "q"}).status_code == 401
    assert c.get("/", headers=basic("wrong")).status_code == 401
    assert c.get("/", headers={"Authorization": "Basic !!!notbase64"}).status_code == 401
    assert c.get("/", headers=basic("s3cret")).status_code == 200
    assert c.get("/", headers=basic("s3cret", user="")).status_code == 200


def test_health_check_needs_no_password(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    assert c.get("/healthz").status_code == 200


def test_oldest_uploads_are_dropped(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.setattr(main, "MAX_DATASETS", 3)
    main.DATASETS.clear()
    ids = []
    for _ in range(5):
        with open("sample_data/checkout_test.csv", "rb") as f:
            ids.append(c.post("/api/upload", files={"file": ("a.csv", f, "text/csv")}).json()["dataset_id"])
    assert len(main.DATASETS) == 3 and ids[0] not in main.DATASETS and ids[-1] in main.DATASETS
