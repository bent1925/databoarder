"""Web server: upload a CSV or put messy files together, ask a question, confirm the read-back, get a report."""
from __future__ import annotations

import hashlib
import io
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from .analysis import analyze
from .ingest import (build_table, inspect_files, preview_recipe, require_comparison, require_one_row_per_unit,
                     require_time_span, script_block, suggest_read_steps)
from .schemas import AnySpec, DataRecipe, Mode
from .observational import covariate_candidates
from .prep import prepare
from .proposer import ProposalError, propose
from .recipe_ops import RecipeError
from .schemas import ObservationalSpec
from .security import PasswordGate
from .sources import SourceStore
from .translator import design_options, interpret, profile_columns, readback, suggest_questions

app = FastAPI(title="Data Science Dashboarder")
app.add_middleware(PasswordGate)
STATIC = Path(__file__).resolve().parent.parent / "static"
MAX_BYTES = 25 * 1024 * 1024

# In-memory store: fine for a prototype run as ONE process. Oldest uploads are dropped past MAX_DATASETS.
MAX_DATASETS = 40
DATASETS: "OrderedDict[str, dict]" = OrderedDict()
SOURCES = SourceStore()
FROZEN_NAME = "analysis_table.csv"


class AskBody(BaseModel):
    dataset_id: str
    question: str
    design: Optional[str] = None


class RunBody(BaseModel):
    dataset_id: str
    spec: AnySpec
    mode: Mode = "quick"


class PreviewBody(BaseModel):
    recipe: dict


class ProposeBody(BaseModel):
    steps: list[dict]
    goal: str = ""


class ConfirmBody(BaseModel):
    recipe: dict
    unit_column: str
    group_column: Optional[str] = None
    time_column: Optional[str] = None


def _get(dataset_id: str) -> dict:
    ds = DATASETS.get(dataset_id)
    if not ds:
        raise HTTPException(404, "That file is no longer loaded. Please upload it again.")
    return ds


def _ingest(raw: bytes, name: str, source: Optional[dict] = None, source_script: str = "") -> dict:
    """The one path every table takes into the analysis, whether uploaded directly or built from several files."""
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Couldn't read that as a CSV file.")
    if df.shape[0] < 2 or df.shape[1] < 2:
        raise HTTPException(400, "The file needs at least 2 rows and 2 columns.")
    df, steps, notes = prepare(df)
    dataset_id = uuid.uuid4().hex[:12]
    while len(DATASETS) >= MAX_DATASETS:
        DATASETS.popitem(last=False)
    DATASETS[dataset_id] = {"df": df, "prep": steps, "name": name, "sha": hashlib.sha256(raw).hexdigest(),
                            "source": source, "source_script": source_script}
    return {
        "dataset_id": dataset_id,
        "name": name,
        "rows": int(df.shape[0]),
        "columns": profile_columns(df),
        "designs": design_options(df),
        "prep": {"steps": [x.text for x in steps], "notes": notes},
        "preview": df.head(5).astype(object).where(df.head(5).notna(), None).to_dict("records"),
        "source": source,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "File is larger than 25 MB.")
    return _ingest(raw, file.filename or "your_file.csv")


def _recipe(raw: dict) -> DataRecipe:
    """Checks a recipe against the fixed steps, and says plainly what's wrong with it."""
    try:
        return DataRecipe.model_validate(raw)
    except ValidationError as problem:
        first = problem.errors()[0]
        where = [part for part in first["loc"] if isinstance(part, int)]
        name = next((str(part) for part in reversed(first["loc"]) if isinstance(part, str)), "this field")
        step = f"Step {where[0] + 1}: " if where else ""
        raise HTTPException(400, f"{step}'{name}' needs a valid value.")


@app.post("/api/sources")
def add_sources(files: list[UploadFile] = File(...)):
    """Saves several files for a short while and reports what's in them. Nothing is analyzed yet."""
    source_id = None
    try:
        source_id = SOURCES.add([(item.filename or "", item.file.read()) for item in files])
        found = inspect_files(SOURCES.paths(source_id))
    except RecipeError as problem:
        if source_id:
            SOURCES.discard(source_id)
        raise HTTPException(400, str(problem))
    except Exception:
        if source_id:
            SOURCES.discard(source_id)
        raise HTTPException(400, "Couldn't read one of those files. Check it opens normally, then try again.")
    return {"source_id": source_id, "tables": [vars(item) for item in found], "steps": suggest_read_steps(found)}


@app.post("/api/sources/{source_id}/preview")
def preview_source(source_id: str, body: PreviewBody):
    try:
        return preview_recipe(_recipe(body.recipe), SOURCES.folder(source_id))
    except RecipeError as problem:
        raise HTTPException(404, str(problem))


@app.post("/api/sources/{source_id}/propose")
def propose_steps(source_id: str, body: ProposeBody):
    """Asks the AI for suggested steps. They come back unconfirmed, for the person to review."""
    if not body.steps or any(step.get("op") != "read_table" for step in body.steps):
        raise HTTPException(400, "AI suggestions start from the files you've read, so remove any steps you added first.")
    recipe = _recipe({"steps": body.steps, "output": body.steps[0].get("into", "")})
    try:
        return propose(recipe, SOURCES.folder(source_id), body.goal)
    except RecipeError as problem:
        raise HTTPException(400, str(problem))
    except ProposalError as problem:
        raise HTTPException(502, str(problem))


@app.post("/api/sources/{source_id}/confirm")
def confirm_source(source_id: str, body: ConfirmBody):
    """Freezes the table from a fully confirmed recipe, then sends it down the normal upload path."""
    recipe = _recipe(body.recipe)
    try:
        folder = SOURCES.folder(source_id)
        built = build_table(recipe, folder)
        require_one_row_per_unit(built.table, body.unit_column)
        if body.group_column:
            require_comparison(built.table, body.group_column)
        elif body.time_column:
            require_time_span(built.table, body.time_column)
        else:
            raise RecipeError("Choose a column that says who got the campaign, or a date column for before and after.")
    except RecipeError as problem:
        raise HTTPException(400, str(problem))
    if len(built.csv_bytes) > MAX_BYTES:
        raise HTTPException(413, "The finished table is larger than 25 MB. Roll it up or keep fewer columns.")
    source = {"kind": "recipe", "inputs": built.inputs, "recipe": recipe.model_dump(), "table_sha256": built.sha256,
              "unit_column": body.unit_column, "group_column": body.group_column, "time_column": body.time_column,
              "steps": [{key: entry[key] for key in ("text", "rows_before", "rows_after", "note")} for entry in built.log]}
    result = _ingest(built.csv_bytes, FROZEN_NAME, source, script_block(recipe, FROZEN_NAME))
    path = SOURCES.snapshot_path(source_id, result["dataset_id"])
    path.write_bytes(built.csv_bytes)
    DATASETS[result["dataset_id"]]["snapshot_path"] = path
    return result


@app.get("/api/datasets/{dataset_id}/snapshot")
def download_snapshot(dataset_id: str):
    path = _get(dataset_id).get("snapshot_path")
    if not path or not path.is_file():
        raise HTTPException(404, "That table is no longer stored. Build it again from your files to download it.")
    return FileResponse(path, media_type="text/csv", filename=FROZEN_NAME)


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
    return analyze(ds["df"], body.spec, csv_name=ds["name"], mode=body.mode, data_sha=ds["sha"], prep_steps=ds["prep"],
                   source=ds.get("source"), source_script=ds.get("source_script", "")).model_dump()


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


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
