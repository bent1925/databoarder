"""The fixed steps that turn messy files into one analysis table.

This file is copied word for word into reproduction scripts, so the script and the app run the same
code and can't disagree. Keep it standalone: only the standard library, numpy and pandas.

Everything is read as text first, so IDs like 00123 keep their zeros and nothing is guessed. Numbers
are only converted by a step that needs them. A recipe is plain data, and nothing in it is executed.
"""
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


class RecipeError(Exception):
    """A step can't be carried out. The message is written for the person using the app."""


def _count(number, noun):
    return f"{number:,} {noun}{'' if number == 1 else 's'}"


def _text(value):
    if pd.isna(value) or value == "":
        return np.nan
    if isinstance(value, float) and value.is_integer():
        return str(int(value))     # Excel stores 13085 as 13085.0
    return str(value)


def _read_csv_rows(path, delimiter):
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with open(path, newline="", encoding=encoding) as handle:
                return list(csv.reader(handle, delimiter=delimiter))
        except UnicodeDecodeError:
            continue
    raise RecipeError(f"Couldn't read the text in {Path(path).name}. Try saving it again as a UTF-8 CSV.")


def read_grid(path, sheet=None, delimiter=","):
    """A file's cells exactly as they are, as text, with every physical row kept."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = _read_csv_rows(path, delimiter)
        width = max((len(row) for row in rows), default=0)
        grid = pd.DataFrame([row + [""] * (width - len(row)) for row in rows], dtype=object)
    elif suffix == ".xlsx":
        try:
            grid = pd.read_excel(path, sheet_name=0 if sheet is None else sheet, header=None, dtype=object)
        except ValueError as error:
            raise RecipeError(f"Couldn't find the sheet '{sheet}' in {path.name}. ({error})")
    else:
        raise RecipeError(f"{path.name} isn't a type this app can read. Save it as .csv or .xlsx and try again.")
    return grid.apply(lambda column: column.map(_text)).astype(object)


def table_from_grid(grid, header_row, footer_rows, label):
    """Uses the chosen header row, and leaves out the rows above it and the footer rows below the data."""
    if header_row >= len(grid):
        raise RecipeError(f"{label} has only {len(grid)} rows, so row {header_row + 1} can't be the header.")
    header = grid.iloc[header_row]
    body = grid.iloc[header_row + 1:len(grid) - footer_rows]
    unnamed = [position for position in range(grid.shape[1]) if pd.isna(header.iloc[position])]
    if any(body.iloc[:, position].notna().any() for position in unnamed):
        raise RecipeError(f"{label} has data in a column with no name. Check that the right header row is chosen.")
    named = [position for position in range(grid.shape[1]) if position not in unnamed]
    table = body.iloc[:, named].copy()
    table.columns = [str(header.iloc[position]).strip() for position in named]
    repeated = sorted({name for name in table.columns if list(table.columns).count(name) > 1})
    if repeated:
        raise RecipeError(f"{label} has more than one column named {', '.join(repeated)}. Rename them and try again.")
    return table.dropna(how="all").reset_index(drop=True)


def _table(tables, name):
    if name not in tables:
        raise RecipeError(f"There's no table called '{name}' at this point in the recipe.")
    return tables[name]


def _column(table, column, table_name):
    if column not in table.columns:
        raise RecipeError(f"The table '{table_name}' has no column called '{column}'.")
    return table[column]


def _numbers(series, column):
    numbers = pd.to_numeric(series, errors="coerce")
    bad = series.notna() & numbers.isna()
    if bad.any():
        raise RecipeError(f"'{column}' needs to be numbers, but {int(bad.sum()):,} values aren't, "
                          f"such as '{series[bad].iloc[0]}'. Filter those rows out first.")
    return numbers


def _read_table(step, tables, folder):
    name = step["file"]
    if Path(name).name != name:
        raise RecipeError(f"'{name}' isn't one of the uploaded files.")
    path = Path(folder) / name
    if not path.is_file():
        raise RecipeError(f"The file '{name}' isn't loaded. Please upload it again.")
    grid = read_grid(path, step.get("sheet"), step.get("delimiter", ","))
    tables[step["into"]] = table_from_grid(grid, step["header_row"], step["footer_rows"], name)
    return step["into"], 0, ""


def _choose_columns(step, tables, folder):
    table = _table(tables, step["table"])
    for column in step["columns"]:
        _column(table, column, step["table"])
    names = list(step["columns"].values())
    if len(set(names)) != len(names):
        raise RecipeError("Two columns would end up with the same name. Choose different names.")
    tables[step["table"]] = table[list(step["columns"])].rename(columns=step["columns"])
    return step["table"], len(table), ""


def _stack(step, tables, folder):
    parts = [_table(tables, name) for name in step["tables"]]
    for name, part in zip(step["tables"], parts):
        if list(part.columns) != list(parts[0].columns):
            raise RecipeError(f"'{name}' has different columns from '{step['tables'][0]}', so they can't be stacked.")
    tables[step["into"]] = pd.concat(parts, ignore_index=True)
    return step["into"], 0, ""


def _clean_key(step, tables, folder):
    table = _table(tables, step["table"]).copy()
    values = _column(table, step["column"], step["table"])
    if step["rule"] == "digits_only":
        cleaned = values.str.replace(r"\.0+$", "", regex=True).str.replace(r"\D", "", regex=True)
        cleaned = cleaned.str.replace(r"^0+(?=\d)", "", regex=True).replace("", np.nan)
    elif step["rule"] == "trim":
        cleaned = values.str.strip().replace("", np.nan)
    else:
        cleaned = values.str.strip().str.lower().replace("", np.nan)
    table[step["into"]] = cleaned
    tables[step["table"]] = table
    lost = int((values.notna() & cleaned.isna()).sum())
    return step["table"], len(table), f"{_count(lost, 'value')} couldn't be read as an ID." if lost else ""


def _filter(step, tables, folder):
    table = _table(tables, step["table"])
    values = _column(table, step["column"], step["table"])
    test, target = step["test"], step["value"]
    if test == "equals":
        keep = values == target
    elif test == "not_equals":
        keep = values != target
    elif test == "starts_with":
        keep = values.str.startswith(target, na=False)
    elif test == "not_starts_with":
        keep = ~values.str.startswith(target, na=False)
    else:
        if step.get("compare_as") == "date":
            left, right = pd.to_datetime(values, errors="coerce"), pd.Timestamp(target)
        elif step.get("compare_as") == "number":
            left, right = pd.to_numeric(values, errors="coerce"), float(target)
        else:
            raise RecipeError(f"Filtering '{step['column']}' needs to say whether it compares numbers or dates.")
        keep = {"greater_than": left > right, "at_least": left >= right,
                "less_than": left < right, "at_most": left <= right}[test]
    tables[step["table"]] = table[keep].reset_index(drop=True)
    return step["table"], len(table), ""


def _roll_up(step, tables, folder):
    table = _table(tables, step["table"])
    key = step["by"]
    no_key = int(_column(table, key, step["table"]).isna().sum())
    rows = table.dropna(subset=[key])
    groups = {}
    for measure in step["measures"]:
        if measure["how"] == "count_rows":
            groups[measure["into"]] = rows.groupby(key).size()
            continue
        numbers = _numbers(_column(rows, measure["column"], step["table"]), measure["column"])
        grouped = numbers.groupby(rows[key])
        if measure["how"] == "sum":
            groups[measure["into"]] = grouped.sum().round(10)     # removes tiny decimal noise from adding
        elif measure["how"] == "mean":
            groups[measure["into"]] = grouped.mean()
        elif measure["how"] == "max":
            groups[measure["into"]] = grouped.max()
        elif measure["how"] == "min":
            groups[measure["into"]] = grouped.min()
        else:
            groups[measure["into"]] = (numbers.fillna(0) > 0).groupby(rows[key]).max().astype(int)
    tables[step["into"]] = pd.DataFrame(groups).reset_index()
    note = f"Left out {_count(no_key, 'row')} with no '{key}'." if no_key else ""
    return step["into"], len(table), note


def _match(step, tables, folder):
    left = _table(tables, step["left"]).copy()
    right = _table(tables, step["right"])
    left_key = _column(left, step["left_key"], step["left"])
    right_key = _column(right, step["right_key"], step["right"])
    repeats = int(right_key.dropna().duplicated().sum())
    if repeats:
        raise RecipeError(f"'{step['right_key']}' repeats in '{step['right']}' ({repeats:,} extra rows), so matching "
                          "would multiply rows. Roll that table up to one row per ID first.")
    lookup = right.dropna(subset=[step["right_key"]]).set_index(step["right_key"])
    for column, new_name in step["bring"].items():
        _column(right, column, step["right"])
        if new_name in left.columns:
            raise RecipeError(f"'{step['left']}' already has a column called '{new_name}'. Choose a different name.")
        left[new_name] = left_key.map(lookup[column])
    tables[step["left"]] = left
    found = int(left_key.isin(lookup.index).sum())
    unused = int((~lookup.index.isin(left_key.dropna())).sum())
    note = f"{found:,} of {len(left):,} rows found a match. {unused:,} rows in '{step['right']}' matched nothing."
    return step["left"], len(left), note


def _flag_membership(step, tables, folder):
    table = _table(tables, step["table"]).copy()
    other = _table(tables, step["other"])
    keys = _column(table, step["key"], step["table"])
    members = set(_column(other, step["other_key"], step["other"]).dropna())
    found = keys.isin(members)
    table[step["into"]] = np.where(found, step["yes"], step["no"])
    tables[step["table"]] = table
    return step["table"], len(table), f"{int(found.sum()):,} of {len(table):,} rows are in '{step['other']}'."


def _fill_missing(step, tables, folder):
    table = _table(tables, step["table"]).copy()
    values = _column(table, step["column"], step["table"])
    filled = int(values.isna().sum())
    table[step["column"]] = values.astype(object).where(values.notna(), step["value"])
    tables[step["table"]] = table
    return step["table"], len(table), f"Set {_count(filled, 'empty value')} to {step['value']}."


OPERATIONS = {"read_table": _read_table, "choose_columns": _choose_columns, "stack": _stack, "clean_key": _clean_key,
              "filter": _filter, "roll_up": _roll_up, "match": _match, "flag_membership": _flag_membership,
              "fill_missing": _fill_missing}


def run_recipe(recipe, folder):
    """Runs a confirmed recipe. Returns the final table and, for each step, what it did to the row count."""
    waiting = [number for number, step in enumerate(recipe["steps"], 1) if step.get("status") != "confirmed"]
    if waiting:
        label = "Steps" if len(waiting) > 1 else "Step"
        verb = "need" if len(waiting) > 1 else "needs"
        raise RecipeError(f"{label} {', '.join(map(str, waiting))} still {verb} your confirmation before this can run.")
    tables, log = {}, []
    for number, step in enumerate(recipe["steps"], 1):
        name, rows_before, note = OPERATIONS[step["op"]](step, tables, folder)
        log.append({"step": number, "table": name, "rows_before": rows_before,
                    "rows_after": len(tables[name]), "note": note})
    return _table(tables, recipe["output"]).reset_index(drop=True), log


def freeze(table):
    """The final table as exact CSV bytes. These bytes are what gets fingerprinted and analyzed."""
    return table.to_csv(index=False, lineterminator="\n").encode("utf-8")
