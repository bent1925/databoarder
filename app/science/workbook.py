"""Science mode: works out how an Excel workbook is laid out, so the person doesn't have to.

Fixed rules, no AI. The result is a plan of exact cell positions (carried out by app/science/sheets.py) and a summary
in plain words. Anything the rules had to infer, such as a missing column label or a group name spelled two ways, is
listed under `checks` for the person to confirm before anything is analyzed.

Layouts it recognises:
  - one sheet per group, rows = subjects, columns = days (or other time points or conditions). A title row such as
    "Table 3" and a group name in the first column are fine. Sheets with no numbers are skipped; a sheet listing only
    the group names is used to check their spelling and order.
  - a single sheet holding an ordinary table.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

from openpyxl.utils import get_column_letter

from .sheets import load, number

TIME = re.compile(r"^\s*(day|d|week|wk|w|hour|hr|h|minute|min|month|mo|time|t)s?\s*[\s._:#-]*\s*(-?\d+(?:\.\d+)?)\s*$", re.I)
UNITS = {"day": "Day", "d": "Day", "week": "Week", "wk": "Week", "w": "Week", "hour": "Hour", "hr": "Hour",
         "h": "Hour", "minute": "Minute", "min": "Minute", "month": "Month", "mo": "Month", "time": "Time", "t": "Time"}
TITLE = re.compile(r"^\s*(table|sheet|data)\s*\d*\s*$", re.I)


class WorkbookError(ValueError):
    """The workbook can't be read as data. The message says what to do."""


@dataclass
class Sheet:
    name: str
    kind: str                                   # blank, labels, data
    texts: list[str] = field(default_factory=list)
    rows: list[int] = field(default_factory=list)
    cols: list[int] = field(default_factory=list)
    header_row: Optional[int] = None
    headers: dict[int, str] = field(default_factory=dict)
    row_labels: dict[int, str] = field(default_factory=dict)
    label_col: Optional[int] = None
    title: Optional[str] = None
    not_numbers: list[tuple[str, str]] = field(default_factory=list)


def _text(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def read_sheet(ws) -> Sheet:
    cells = {}
    for row in ws.iter_rows():
        for c in row:
            if _text(c.value) is not None:
                cells[(c.row, c.column)] = c.value
    numeric = {k for k, v in cells.items() if number(v) is not None}
    texts = [_text(v) for k, v in sorted(cells.items()) if k not in numeric]
    if not numeric:
        return Sheet(ws.title, "blank" if len(texts) <= 1 else "labels", texts=texts)
    first_row = min(r for r, _ in numeric)
    first_col = min(c for _, c in numeric)
    rows = sorted({r for r, _ in numeric})
    cols = sorted({c for _, c in numeric})
    header_row = max((r for (r, c) in cells if r < first_row and c in cols and (r, c) not in numeric), default=None)
    headers = {c: _text(cells[(header_row, c)]) for c in cols if header_row and (header_row, c) in cells}
    text_left = [c for c in range(first_col - 1, 0, -1) if any((r, c) in cells for r in rows)]
    label_col = text_left[0] if text_left else None          # the nearest column of labels left of the numbers
    row_labels = {r: _text(cells[(r, label_col)]) for r in rows if label_col and (r, label_col) in cells}
    if label_col and header_row and (header_row, label_col) in cells:
        headers[label_col] = _text(cells[(header_row, label_col)])
    above = [_text(v) for (r, c), v in sorted(cells.items()) if r < (header_row or first_row)]
    not_numbers = [(f"{get_column_letter(c)}{r}", _text(cells[(r, c)])) for r in rows for c in cols
                   if (r, c) in cells and (r, c) not in numeric]
    return Sheet(ws.title, "data", texts=texts, rows=rows, cols=cols, header_row=header_row, headers=headers,
                 row_labels=row_labels, label_col=label_col, title=next((t for t in above if t), None), not_numbers=not_numbers)


def _time(label: Optional[str]):
    """('Day', 11) for 'Day11', (None, 4) for a plain 4, None for anything else."""
    if label is None:
        return None
    m = TIME.match(label)
    if m:
        return UNITS[m.group(1).lower()], _num(m.group(2))
    value = number(label)
    return (None, _num(value)) if value is not None else None


def _num(v):
    v = float(v)
    return int(v) if v.is_integer() else v


def _close_name(name: str, names: list[str]) -> Optional[str]:
    lowered = {n.lower(): n for n in names}
    if name.lower() in lowered:
        return lowered[name.lower()]
    match = difflib.get_close_matches(name.lower(), list(lowered), n=1, cutoff=0.85)
    return lowered[match[0]] if match else None


def propose(raw: bytes, options: Optional[dict] = None) -> dict:
    """The plan, and what it means in plain words. `options`: keep_spelling (bool), sheet (name)."""
    options = options or {}
    try:
        book = load(raw)
    except Exception:
        raise WorkbookError("Couldn't open that as an Excel workbook. Check it opens in Excel, then save it as .xlsx again.")
    sheets = [read_sheet(ws) for ws in book.worksheets]
    data = [s for s in sheets if s.kind == "data"]
    skipped = [{"sheet": s.name, "why": "blank" if s.kind == "blank" else "only labels, no numbers"}
               for s in sheets if s.kind != "data"]
    if not data:
        raise WorkbookError("None of the sheets has any numbers in it.")
    stackable = (len(data) >= 2 and not options.get("sheet")
                 and len({tuple(s.cols) for s in data}) == 1          # the same columns on every sheet
                 and all(len(s.rows) >= 2 for s in data))             # and a group of subjects on each
    if stackable:
        return _sheet_per_group(data, sheets, skipped, options)
    chosen = next((s for s in data if s.name == options.get("sheet")), data[0])
    return _single_sheet(chosen, data, skipped)


def _sheet_per_group(data: list[Sheet], sheets: list[Sheet], skipped: list[dict], options: dict) -> dict:
    checks, notes = [], []
    cols = data[0].cols
    # --- column labels: each sheet's own, and where one is missing, what the other sheets agree on
    columns, filled = [], {}
    for c in cols:
        seen = {s.name: _time(s.headers.get(c)) for s in data}
        parsed = {v for v in seen.values() if v is not None}
        if len(parsed) > 1:
            shown = ", ".join(f"{s.headers.get(c)!r} on {n}" for n, s in ((s.name, s) for s in data) if s.headers.get(c))
            raise WorkbookError(f"Column {get_column_letter(c)} is labelled differently on different sheets ({shown}). "
                                "Make the labels match, then add the workbook again.")
        if not parsed:
            label = next((s.headers[c] for s in data if s.headers.get(c)), None)
            columns.append({"col": c, "label": label or f"Column {get_column_letter(c)}", "unit": None})
            continue
        unit, value = parsed.pop()
        for name in (n for n, v in seen.items() if v is None):
            filled.setdefault(name, []).append((get_column_letter(c), f"{unit or ''} {value}".strip()))
        columns.append({"col": c, "label": value, "unit": unit})
    for name, gaps in filled.items():
        many = len(gaps) > 1
        checks.append(f"{name} has no label{'s' if many else ''} above column{'s' if many else ''} "
                      f"{_names([g[0] for g in gaps])}. The other sheets call {'them' if many else 'it'} "
                      f"{_names([g[1] for g in gaps])}, so {'they are' if many else 'it is'} read that way.")
    units = {col["unit"] for col in columns}
    timed = all(isinstance(col["label"], (int, float)) for col in columns)
    time_name = (units.pop() or "Time") if timed and len(units) == 1 else ("Time" if timed else "Condition")
    if timed:
        written = sorted({s.headers.get(c) for s in data for c in cols if s.headers.get(c)})
        odd = [w for w in written if w != f"{time_name} {_time(w)[1]}"]
        if odd:
            notes.append(f"Labels written slightly differently ({', '.join(repr(w) for w in odd[:4])}) are read by "
                         "their number.")

    # --- the group on each sheet
    listed = [t for s in sheets if s.kind == "labels" for t in s.texts if not TITLE.match(t)]
    parts = []
    for s in data:
        labels = [s.row_labels.get(r) for r in s.rows]
        named = [x for x in labels if x]
        subjects = None
        if len(named) == 1:
            group, how = named[0], "the label next to its first row"
        elif len(set(named)) == len(s.rows):
            group, how = (s.title if s.title and not TITLE.match(s.title) else s.name), "the sheet's title"
            subjects = named
        else:
            group, how = (s.title if s.title and not TITLE.match(s.title) else s.name), "the sheet's title"
            if len(set(named)) > 1:
                checks.append(f"{s.name} has several different labels down its first column "
                              f"({_names(sorted(set(named))[:3])}). The whole sheet is treated as one group.")
        if how == "the sheet's title" and group == s.name:
            how = "the sheet's name"
        canon = _close_name(group, listed) if listed else None
        if canon and canon != group:
            if options.get("keep_spelling"):
                notes.append(f"{s.name}: kept the sheet's spelling '{group}' (the list on another sheet says '{canon}').")
            else:
                checks.append(f"{s.name} names its group '{group}', but the list on "
                              f"{_names([x.name for x in sheets if x.kind == 'labels'])} says '{canon}'. "
                              f"Using '{canon}'.")
                group = canon
        parts.append({"sheet": s.name, "group": group, "rows": s.rows, "subjects": subjects, "how": how,
                      "label_cell": None})
        for cell, text in s.not_numbers:
            checks.append(f"{s.name}, cell {cell} says '{text}', which isn't a number. It's treated as empty.")
    names = [p["group"] for p in parts]
    if len(set(names)) != len(names):
        dup = next(n for n in names if names.count(n) > 1)
        raise WorkbookError(f"Two sheets are both named '{dup}'. Give each group's sheet a different name, then add "
                            "the workbook again.")
    if listed and all(n in listed for n in names):
        parts.sort(key=lambda p: listed.index(p["group"]))
        notes.append("Groups are in the order of the list on " + _names([x.name for x in sheets if x.kind == "labels"])
                     + ".")
    for p in parts:
        p.pop("label_cell")
    plan = {"kind": "sheet_per_group", "group_name": "Group", "subject_name": "Animal", "time_name": time_name,
            "value_name": "Value", "columns": [{"col": c["col"], "label": c["label"]} for c in columns],
            "sheets": [{k: p[k] for k in ("sheet", "group", "rows", "subjects")} for p in parts]}
    sizes = [len(p["rows"]) for p in parts]
    summary = {
        "layout": "One sheet per group, rows are animals, columns are " + (time_name.lower() + "s" if timed else "conditions"),
        "groups": [{"name": p["group"], "sheet": p["sheet"], "subjects": len(p["rows"]), "how": p["how"]} for p in parts],
        "times": [c["label"] for c in columns], "time_name": time_name,
        "skipped": skipped, "checks": checks, "notes": notes,
        "assumptions": [
            "Each row on a sheet is one animal, followed across every " + (time_name.lower() if timed else "column") + ".",
            "Each sheet holds different animals (no animal appears in two groups).",
        ],
        "subjects_per_group": sizes,
    }
    return {"plan": plan, "summary": summary}


def _single_sheet(s: Sheet, data: list[Sheet], skipped: list[dict]) -> dict:
    checks = []
    columns, used = [], set()
    if s.label_col:
        name = s.headers.get(s.label_col) or "Label"
        columns.append({"col": s.label_col, "name": name, "numeric": False})
        used.add(name)
    for c in s.cols:
        name = s.headers.get(c) or f"Column {get_column_letter(c)}"
        if not s.headers.get(c):
            checks.append(f"Column {get_column_letter(c)} on {s.name} has no label, so it's called '{name}'.")
        while name in used:
            name += " (2)"
        used.add(name)
        columns.append({"col": c, "name": name, "numeric": True})
    for cell, text in s.not_numbers:
        checks.append(f"{s.name}, cell {cell} says '{text}', which isn't a number. It's treated as empty.")
    others = [d.name for d in data if d.name != s.name]
    plan = {"kind": "single_sheet", "sheet": s.name, "rows": s.rows, "columns": columns}
    summary = {"layout": f"One table, from {s.name}", "sheet": s.name, "other_sheets": others,
               "skipped": skipped, "checks": checks, "notes": [], "assumptions": [], "groups": [], "times": []}
    if others:
        summary["notes"].append(f"The other sheets with numbers ({_names(others)}) are laid out differently, so only "
                                f"{s.name} is used. Pick another sheet if that's the wrong one.")
    return {"plan": plan, "summary": summary}


def _names(names: list[str]) -> str:
    names = [str(n) for n in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
