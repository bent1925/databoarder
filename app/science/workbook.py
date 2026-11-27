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
# Words that label a header row ("Animal  1  4  8", "Day  0  7  14").
HEADER_WORD = re.compile(r"^\s*(animal|mouse|mice|rat|subject|patient|sample|id|day|time|week|hour|hr|h|min|minute|"
                         r"month|visit|dose|conc|concentration|group|treatment|condition|timepoint|time point)s?"
                         r"\s*(\(.*\))?\s*(no\.?|#|id)?\s*:?\s*$", re.I)


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
    no_numbers: list[tuple[int, Optional[str]]] = field(default_factory=list)   # subject rows with no values at all
    empty_cols: list[int] = field(default_factory=list)                         # labelled columns with no values
    candidate: Optional[dict] = None          # a first row of numbers that may be labels (1, 4, 8) rather than values
    header_from_numbers: bool = False
    cells: dict = field(default_factory=dict, repr=False)


def _text(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _numeric_header(cells: dict, numeric: set) -> Optional[dict]:
    """The first row of numbers, when it could be labels rather than measurements: no text above it in those columns,
    rows of numbers below it, and values that rise steadily (1, 4, 8). `likely` says whether there's a further sign:
    a header word beside it ("Animal", "Day"), or whole numbers above measurements that aren't whole."""
    r0 = min(r for r, _ in numeric)
    cols0 = sorted(c for r, c in numeric if r == r0)
    all_cols = {c for _, c in numeric}
    if len(cols0) < 2 or not any(r > r0 for r, _ in numeric):
        return None
    if any(r < r0 and c in all_cols for (r, c) in cells):
        return None                                            # there's already a header above the numbers
    values = [number(cells[(r0, c)]) for c in cols0]
    if any(b <= a for a, b in zip(values, values[1:])):
        return None
    word = next((_text(cells[(r0, c)]) for c in range(min(cols0) - 1, 0, -1) if (r0, c) in cells
                 and HEADER_WORD.match(_text(cells[(r0, c)]))), None)
    whole = all(float(v).is_integer() for v in values)
    rest_fractional = any(not float(number(cells[k])).is_integer() for k in numeric if k[0] != r0)
    return {"row": r0, "cols": cols0, "values": [_num(v) for v in values], "word": word,
            "likely": bool(word) or (whole and rest_fractional)}


def read_sheet(ws, numeric_header: Optional[bool] = None) -> Sheet:
    """`numeric_header`: True or False decides whether a first row of rising numbers is labels; None lets the sheet's
    own signs decide."""
    cells = {}
    for row in ws.iter_rows():
        for c in row:
            if _text(c.value) is not None:
                cells[(c.row, c.column)] = c.value
    numeric = {k for k, v in cells.items() if number(v) is not None}
    texts = [_text(v) for k, v in sorted(cells.items()) if k not in numeric]
    if not numeric:
        return Sheet(ws.title, "blank" if len(texts) <= 1 else "labels", texts=texts, cells=cells)
    candidate = _numeric_header(cells, numeric)
    as_header = bool(candidate) and (candidate["likely"] if numeric_header is None else numeric_header)
    if as_header:
        numeric -= {(candidate["row"], c) for c in candidate["cols"]}
        if not numeric:
            return Sheet(ws.title, "labels", texts=texts, cells=cells)
    first_row, last_row = min(r for r, _ in numeric), max(r for r, _ in numeric)
    first_col = min(c for _, c in numeric)
    cols = sorted({c for _, c in numeric})
    header_row = max((r for (r, c) in cells if r < first_row and c in cols and (r, c) not in numeric), default=None)
    if header_row:
        # A column labelled as a time point ("Day 8") is kept even when it has no numbers (every animal blank or
        # "dead"), so the missing day is shown and counted rather than silently lost.
        cols = sorted(set(cols) | {c for (r, c) in cells if r == header_row and c > first_col and c not in cols
                                   and _time(_text(cells[(r, c)]))})
    headers = {c: _text(cells[(header_row, c)]) for c in cols if header_row and (header_row, c) in cells}
    number_rows = sorted({r for r, _ in numeric})
    text_left = [c for c in range(first_col - 1, 0, -1) if any((r, c) in cells for r in number_rows)]
    label_col = text_left[0] if text_left else None          # the nearest column of labels left of the numbers
    # Subject rows: every row in the block with anything in it, so an animal with no numbers ("dead", or blank with
    # its own label) is kept and reported rather than silently dropped.
    has = lambda r: any((r, c) in cells for c in cols) or bool(label_col and (r, label_col) in cells)  # noqa: E731
    rows = [r for r in range(first_row, last_row + 1) if has(r)]
    own_labels = label_col and len({_text(cells.get((r, label_col))) for r in number_rows} - {None}) > 1
    r = last_row + 1
    while any((r, c) in cells for c in cols) or (own_labels and (r, label_col) in cells):
        rows.append(r)
        r += 1
    row_labels = {r: _text(cells[(r, label_col)]) for r in rows if label_col and (r, label_col) in cells}
    if label_col and header_row and (header_row, label_col) in cells:
        headers[label_col] = _text(cells[(header_row, label_col)])
    above = [_text(v) for (r, c), v in sorted(cells.items()) if r < (header_row or first_row)]
    not_numbers = [(f"{get_column_letter(c)}{r}", _text(cells[(r, c)])) for r in rows for c in cols
                   if (r, c) in cells and (r, c) not in numeric]
    no_numbers = [(r, row_labels.get(r)) for r in rows if not any((r, c) in numeric for c in cols)]
    empty_cols = [c for c in cols if not any((r, c) in numeric for r in rows)]
    return Sheet(ws.title, "data", texts=texts, rows=rows, cols=cols, header_row=header_row, headers=headers,
                 row_labels=row_labels, label_col=label_col, title=next((t for t in above if t), None),
                 not_numbers=not_numbers, no_numbers=no_numbers, empty_cols=empty_cols, candidate=candidate,
                 header_from_numbers=as_header,
                 cells=cells)


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
    """The plan, and what it means in plain words.

    `options`: keep_spelling (bool), sheet (name), first_row ("labels" or "values": how to read a first row of
    rising numbers such as 1, 4, 8)."""
    options = options or {}
    try:
        book = load(raw)
    except Exception:
        raise WorkbookError("Couldn't open that as an Excel workbook. Check it opens in Excel, then save it as .xlsx again.")
    choice = {"labels": True, "values": False}.get(options.get("first_row"))
    sheets = [read_sheet(ws, choice) for ws in book.worksheets]
    found = [x.candidate for x in sheets if x.kind == "data" and x.candidate]
    if choice is None and len(found) >= 2 and len({tuple(c["values"]) for c in found}) == 1 \
            and len(found) == sum(x.kind == "data" for x in sheets) and not all(c["likely"] for c in found) \
            and all(float(v).is_integer() for v in found[0]["values"]):
        # The same rising row of whole numbers on every sheet: labels, not measurements (still shown for checking).
        # Fractional values are left alone: identical measurement rows can come from copied baseline values.
        sheets = [read_sheet(ws, True) for ws in book.worksheets]
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


def _row_checks(data: list[Sheet]) -> list[str]:
    """What was decided about rows: a first row of numbers read as labels (or not), and subjects with no numbers."""
    checks = []
    used = [x for x in data if x.header_from_numbers]
    unused = [x for x in data if x.candidate and not x.header_from_numbers]
    if used:
        c = used[0].candidate
        checks.append(f"Row {c['row']} on {_names([x.name for x in used])} ({', '.join(map(str, c['values']))}) is read "
                      "as labels for the columns, not as measurements.")
    if unused:
        c = unused[0].candidate
        checks.append(f"Row {c['row']} on {_names([x.name for x in unused])} ({', '.join(map(str, c['values']))}) "
                      "rises steadily like column labels, but is read as measurements.")
    for x in data:
        for r, label in x.no_numbers:
            checks.append(f"{x.name}, row {r}{f' ({label})' if label else ''} has no numbers. It's kept as a subject "
                          "with no values, so analyses that need every value leave it out and say so.")
    return checks


def _sheet_per_group(data: list[Sheet], sheets: list[Sheet], skipped: list[dict], options: dict) -> dict:
    checks, notes = _row_checks(data), []
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
    for c in cols:
        if all(c in x.empty_cols for x in data):
            label = next((x.headers.get(c) for x in data if x.headers.get(c)), get_column_letter(c))
            checks.append(f"Column {get_column_letter(c)} ({label}) has no numbers on any sheet. It's kept, so every "
                          f"subject counts as missing it: leave it out when choosing the {label.split()[0].lower()}s "
                          "to analyze.")
        else:
            bare = [x.name for x in data if c in x.empty_cols]
            if bare:
                label = next((x.headers.get(c) for x in data if x.headers.get(c)), get_column_letter(c))
                checks.append(f"{_names(bare)} {'has' if len(bare) == 1 else 'have'} no numbers in column "
                              f"{get_column_letter(c)} ({label}). Those subjects count as missing it.")
    for name, gaps in filled.items():
        many = len(gaps) > 1
        checks.append(f"{name} has no label{'s' if many else ''} above column{'s' if many else ''} "
                      f"{_names([g[0] for g in gaps])}. The other sheets call {'them' if many else 'it'} "
                      f"{_names([g[1] for g in gaps])}, so {'they are' if many else 'it is'} read that way.")
    word = next((x.candidate["word"] for x in data if x.header_from_numbers and x.candidate["word"]), None)
    if word and _time(f"{word} 1") and all(col["unit"] is None for col in columns):
        for col in columns:
            col["unit"] = _time(f"{word} 1")[0]
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
    labels = [x for p in parts for x in (p["subjects"] or [])]
    if len(labels) != len(set(labels)):
        # The same labels on several sheets (m1, m2, ... in every group) are different animals: name them by group.
        for p in parts:
            if p["subjects"]:
                p["subjects"] = [f"{p['group']} {x}" for x in p["subjects"]]
        notes.append("Animals are labelled the same way on different sheets, so each label is prefixed with its "
                     "group (for example 'Ctrl m1').")
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
        "first_row": _first_row(data),
    }
    return {"plan": plan, "summary": summary}


def _single_sheet(s: Sheet, data: list[Sheet], skipped: list[dict]) -> dict:
    """An ordinary table: every column with a header or a value is kept, text columns included."""
    checks = _row_checks([s])
    cells = s.cells
    head = s.header_row if s.header_row else (s.candidate["row"] if s.header_from_numbers else None)
    present = sorted({c for (r, c) in cells if r in s.rows or r == head})
    columns, used = [], set()
    for c in present:
        values = [cells[(r, c)] for r in s.rows if (r, c) in cells]
        header = _text(cells.get((head, c))) if head else None
        name = header or f"Column {get_column_letter(c)}"
        if not header:
            checks.append(f"Column {get_column_letter(c)} on {s.name} has no label, so it's called '{name}'.")
        while name in used:
            name += " (2)"
        used.add(name)
        numeric = bool(values) and all(number(v) is not None for v in values)
        if not numeric and values and sum(number(v) is not None for v in values) >= 0.8 * len(values):
            odd = [f"{get_column_letter(c)}{r}" for r in s.rows if (r, c) in cells and number(cells[(r, c)]) is None]
            checks.append(f"Column '{name}' on {s.name} is mostly numbers, but {_names(odd[:3])} "
                          f"{'isn\'t' if len(odd) == 1 else 'aren\'t'}. Those are treated as empty.")
            numeric = True
        columns.append({"col": c, "name": name, "numeric": numeric})
    others = [d.name for d in data if d.name != s.name]
    plan = {"kind": "single_sheet", "sheet": s.name, "rows": s.rows, "columns": columns}
    summary = {"layout": f"One table, from {s.name}", "sheet": s.name, "other_sheets": others,
               "skipped": skipped, "checks": checks, "notes": [], "assumptions": [], "groups": [], "times": [],
               "first_row": _first_row([s])}
    if others:
        summary["notes"].append(f"The other sheets with numbers ({_names(others)}) are laid out differently, so only "
                                f"{s.name} is used. Pick another sheet if that's the wrong one.")
    return {"plan": plan, "summary": summary}


def _first_row(data: list[Sheet]) -> Optional[str]:
    """For the screen's switch: how a first row of rising numbers was read, if there was one."""
    if any(x.header_from_numbers for x in data):
        return "labels"
    if any(x.candidate for x in data):
        return "values"
    return None


def _names(names: list[str]) -> str:
    names = [str(n) for n in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
