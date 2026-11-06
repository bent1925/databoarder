"""Web server: upload a CSV, ask a question, confirm the read-back, get a report."""
from __future__ import annotations

import hashlib
import io
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .analysis import analyze
from .schemas import AnySpec, Mode
from .observational import covariate_candidates
from .prep import prepare
from .schemas import ObservationalSpec
from .translator import design_options, interpret, profile_columns, readback, suggest_questions

app = FastAPI(title="Data Science Dashboarder")
STATIC = Path(__file__).resolve().parent.parent / "static"
MAX_BYTES = 25 * 1024 * 1024

# In-memory store: fine for a single-user prototype.
DATASETS: dict[str, dict] = {}


class AskBody(BaseModel):
    dataset_id: str
    question: str
    design: Optional[str] = None


class RunBody(BaseModel):
    dataset_id: str
    spec: AnySpec
    mode: Mode = "quick"


def _get(dataset_id: str) -> dict:
    ds = DATASETS.get(dataset_id)
    if not ds:
        raise HTTPException(404, "That file is no longer loaded. Please upload it again.")
    return ds


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "File is larger than 25 MB.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Couldn't read that as a CSV file.")
    if df.shape[0] < 2 or df.shape[1] < 2:
        raise HTTPException(400, "The file needs at least 2 rows and 2 columns.")
    df, steps, notes = prepare(df)
    dataset_id = uuid.uuid4().hex[:12]
    DATASETS[dataset_id] = {"df": df, "prep": steps, "name": file.filename or "your_file.csv",
                            "sha": hashlib.sha256(raw).hexdigest()}
    return {
        "dataset_id": dataset_id,
        "name": DATASETS[dataset_id]["name"],
        "rows": int(df.shape[0]),
        "columns": profile_columns(df),
        "designs": design_options(df),
        "prep": {"steps": [x.text for x in steps], "notes": notes},
        "preview": df.head(5).astype(object).where(df.head(5).notna(), None).to_dict("records"),
    }


@app.post("/api/ask")
def ask(body: AskBody):
    ds = _get(body.dataset_id)
    if body.design:
        opt = {o["key"]: o for o in design_options(ds["df"])["options"]}.get(body.design)
        if not opt:
            raise HTTPException(400, "Unknown choice.")
        if not opt["ok"]:
            raise HTTPException(400, opt["why"])
    result = interpret(body.question, ds["df"], body.design)
    return result.model_dump()


@app.post("/api/readback")
def readback_for(body: RunBody):
    """Re-render the plain-English read-back after the user edits the choices."""
    ds = _get(body.dataset_id)
    return {"readback": readback(ds["df"], body.spec)}


@app.post("/api/run")
def run(body: RunBody):
    ds = _get(body.dataset_id)
    return analyze(ds["df"], body.spec, csv_name=ds["name"], mode=body.mode,
                   data_sha=ds["sha"], prep_steps=ds["prep"]).model_dump()


@app.get("/api/suggestions")
def suggestions(dataset_id: str, mode: Mode = "quick", design: Optional[str] = None):
    ds = _get(dataset_id)
    return {"questions": suggest_questions(ds["df"], mode, design=design)}


@app.get("/api/covariates")
def covariates(dataset_id: str, group_col: str, control: str, treatment: str, outcome_col: str):
    """Columns people can be compared on, and which ones are ticked by default."""
    ds = _get(dataset_id)
    spec = ObservationalSpec(group_col=group_col, control=control, treatment=treatment, outcome_col=outcome_col)
    for col in (group_col, outcome_col):
        if col not in ds["df"].columns:
            raise HTTPException(400, "Unknown column.")
    return {"columns": covariate_candidates(ds["df"], spec)}


@app.get("/api/levels")
def levels(dataset_id: str, column: str):
    ds = _get(dataset_id)
    if column not in ds["df"].columns:
        raise HTTPException(400, "Unknown column.")
    vals = ds["df"][column].dropna().astype(str).unique().tolist()
    return {"values": sorted(vals)[:50]}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
