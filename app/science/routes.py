"""Science mode: web routes. Upload or paste a table, confirm the choices, get a report."""
from __future__ import annotations

import csv
import re
import hashlib
import io
import uuid
from collections import OrderedDict
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, TypeAdapter, ValidationError

from ..prep import prepare, script_block
from . import engine, report, script
from .schemas import ScienceSpec

router = APIRouter()
MAX_BYTES = 25 * 1024 * 1024
MAX_DATASETS = 40
DATASETS: "OrderedDict[str, dict]" = OrderedDict()
SPEC = TypeAdapter(ScienceSpec)


class PasteBody(BaseModel):
    text: str
    name: str = "pasted_data.csv"


class SpecBody(BaseModel):
    dataset_id: str
    spec: dict


def pasted_to_csv(text: str) -> bytes:
    """Cells pasted from a spreadsheet are separated by tabs; otherwise commas. Values are kept exactly as typed."""
    lines = [line for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise HTTPException(400, "Nothing was pasted. Copy the cells from your spreadsheet, including the header row.")
    if "\t" in lines[0]:
        rows = [line.split("\t") for line in lines]
    else:
        rows = list(csv.reader(lines))
    width = max(len(r) for r in rows)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for r in rows:
        writer.writerow([cell.strip() for cell in r] + [""] * (width - len(r)))
    return out.getvalue().encode("utf-8")


def _profile(df: pd.DataFrame) -> list[dict]:
    cols = []
    for c in df.columns:
        s = df[c]
        numeric = bool(pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s))
        levels = engine._labels(s).dropna().unique().tolist()
        cols.append({"name": str(c), "numeric": numeric, "missing": int(s.isna().sum()), "distinct": len(levels),
                     "levels": levels[:30] if len(levels) <= 30 else []})
    return cols


def guesses(df: pd.DataFrame) -> dict:
    """Starting points for the column choices, shown as guesses. Nothing here is a confirmed choice."""
    cols = _profile(df)
    labels = [c["name"] for c in cols if not c["numeric"] and c["distinct"] == len(df)]
    cats = [c["name"] for c in cols if not c["numeric"] and 2 <= c["distinct"] <= 12 and c["distinct"] < len(df)]
    num = [c["name"] for c in cols if c["numeric"]]
    out: dict = {}
    if cats and num:
        out["compare_groups"] = {"layout": "long", "group_col": cats[0], "value_col": num[0],
                                 "subject_col": labels[0] if labels else None}
    elif len(num) >= 2:
        out["compare_groups"] = {"layout": "wide", "groups": num[:8], "subject_col": labels[0] if labels else None}
    if len(cats) >= 2:
        out["contingency"] = {"layout": "rows", "row_col": cats[0], "col_col": cats[1]}
    counts = [c for c in num if pd.api.types.is_integer_dtype(df[c]) and (df[c] >= 0).all()]
    if "contingency" not in out and labels and len(counts) >= 2 and len(df) <= 12:
        out["contingency"] = {"layout": "counts", "label_col": labels[0], "cols": counts[:6]}
    if len(num) >= 2:
        out["correlation"] = {"x_col": num[0], "y_col": num[1]}
    return out


def _safe_name(name: str) -> str:
    """A file name that is safe in a download header and a script."""
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", name.replace("\\", "/").split("/")[-1]).strip(" .") or "data"
    return base[:80] if base.lower().endswith(".csv") else base[:76] + ".csv"


def _store(raw: bytes, name: str) -> dict:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Couldn't read that as a table. Save it as CSV from your spreadsheet, or paste the "
                                 "cells instead.")
    if df.shape[0] < 2 or df.shape[1] < 1:
        raise HTTPException(400, "The table needs a header row and at least 2 rows of values.")
    df, steps, notes = prepare(df)
    notes = [n for n in notes if "one column per date" not in n]   # wide tables are normal here
    dataset_id = uuid.uuid4().hex[:12]
    while len(DATASETS) >= MAX_DATASETS:
        DATASETS.popitem(last=False)
    DATASETS[dataset_id] = {"df": df, "raw": raw, "name": name, "sha": hashlib.sha256(raw).hexdigest(), "prep": steps}
    head = df.head(8)
    return {"dataset_id": dataset_id, "name": name, "rows": int(df.shape[0]), "columns": _profile(df),
            "guesses": guesses(df), "prep": {"steps": [s.text for s in steps], "notes": notes},
            "preview": head.astype(object).where(head.notna(), None).to_dict("records")}


def _get(dataset_id: str) -> dict:
    ds = DATASETS.get(dataset_id)
    if not ds:
        raise HTTPException(404, "That table is no longer loaded. Please add it again.")
    return ds


def _spec(raw: dict) -> dict:
    try:
        return SPEC.validate_python(raw).model_dump()
    except ValidationError as problem:
        first = problem.errors()[0]
        msg = first.get("msg", "")
        raise HTTPException(400, msg.removeprefix("Value error, ") if "Value error" in msg
                            else f"'{'.'.join(str(p) for p in first['loc'][1:]) or 'a choice'}' needs a valid value.")


@router.post("/api/science/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "File is larger than 25 MB.")
    name = file.filename or "data.csv"
    if name.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Excel files aren't read here yet. Save the sheet as CSV, or copy the cells and paste them.")
    return _store(raw, _safe_name(name))


@router.post("/api/science/paste")
def paste(body: PasteBody):
    raw = pasted_to_csv(body.text)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "That's more than 25 MB of cells.")
    return _store(raw, _safe_name(body.name))


@router.get("/api/science/datasets/{dataset_id}/data")
def download(dataset_id: str):
    ds = _get(dataset_id)
    return Response(ds["raw"], media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{ds["name"]}"'})


@router.post("/api/science/readback")
def readback(body: SpecBody):
    """The sentence saying what will run, written by code, and anything that stops it running."""
    ds = _get(body.dataset_id)
    spec = _spec(body.spec)
    problem: Optional[str] = None
    try:
        engine.run(ds["df"], spec)
    except engine.EngineError as e:
        problem = str(e)
    return {"readback": report.readback(spec), "problem": problem, "spec": spec}


@router.post("/api/science/run")
def run(body: SpecBody):
    ds = _get(body.dataset_id)
    spec = _spec(body.spec)
    try:
        result = engine.run(ds["df"], spec)
    except engine.EngineError as e:
        raise HTTPException(400, str(e))
    rep = report.build(spec, result)
    rep.prep = [s.text for s in ds["prep"]]
    rep.data_name = ds["name"]
    rep.repro = {"data_sha256": ds["sha"], "rows_in_file": int(ds["df"].shape[0]), "versions": script.versions(),
                 "random_steps": (f"Dunnett's test estimates its P values and intervals by quasi-Monte Carlo with a fixed "
                                   f"seed ({engine.SEED}), so they repeat exactly" if result.get("pair_method") == "Dunnett's test"
                                   else "none")}
    rep.script = script.build(spec, ds["name"], ds["sha"], script_block(ds["prep"]))
    return rep.model_dump()
