"""Science mode: turns an Excel workbook into one tidy table, following a plan of exact cell positions.

This file is copied word for word into reproduction scripts, so it only imports openpyxl and pandas. All the
guessing happens elsewhere (app/science/workbook.py) and ends up in the plan, which the person confirms. Here, the
plan is simply carried out: read these cells, from these sheets, and label them like this.

Plan kinds:
  sheet_per_group  every listed sheet is one group; its rows are subjects, its columns are time points (or
                   conditions). Gives one row per measurement: group, subject, time, value.
  single_sheet     one sheet read as an ordinary table: a header row, then one row per record.
"""
from __future__ import annotations

import io

import openpyxl
import pandas as pd


def number(value):
    """A cell as a number, or None when it's empty or isn't one (such as 'n/a' or 'dead')."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", ""))
    except ValueError:
        return None


def load(raw: bytes):
    """Cached values, not formulas: what the spreadsheet shows."""
    return openpyxl.load_workbook(io.BytesIO(raw), data_only=True, read_only=False)


def build_table(raw: bytes, plan: dict) -> pd.DataFrame:
    book = load(raw)
    if plan["kind"] == "sheet_per_group":
        rows = []
        for part in plan["sheets"]:
            sheet = book[part["sheet"]]
            for n, r in enumerate(part["rows"], start=1):
                for column in plan["columns"]:
                    rows.append({plan["group_name"]: part["group"],
                                 plan["subject_name"]: part["subjects"][n - 1] if part.get("subjects")
                                 else f"{part['group']} #{n}",
                                 plan["time_name"]: column["label"],
                                 plan["value_name"]: number(sheet.cell(row=r, column=column["col"]).value)})
        return pd.DataFrame(rows)
    sheet = book[plan["sheet"]]
    data = {}
    for column in plan["columns"]:
        cells = [sheet.cell(row=r, column=column["col"]).value for r in plan["rows"]]
        if column["numeric"]:
            data[column["name"]] = [number(v) for v in cells]
        else:
            data[column["name"]] = [None if v is None else str(v).strip() for v in cells]
    return pd.DataFrame(data)


def table_sha256(table: pd.DataFrame) -> str:
    import hashlib
    return hashlib.sha256(table.to_csv(index=False).encode("utf-8")).hexdigest()
