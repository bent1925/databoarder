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


_AMBIGUOUS_DATE = r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"


def _dates(series, column, date_order=None):
    """Reads dates. Won't guess whether 03/04/2010 is March or April: that has to be stated."""
    ambiguous = series.notna() & series.astype(str).str.match(_AMBIGUOUS_DATE)
    if date_order is None and ambiguous.any():
        raise RecipeError(f"Dates in '{column}' look like 03/04/2010, which could be March or April. "
                          "Say whether the day or the month comes first.")
    # The stated order applies only to ambiguous dates: pandas would otherwise swap 2024-04-05 too.
    parsed = pd.to_datetime(series.where(~ambiguous), errors="coerce")
    failed = series.notna() & ~ambiguous & parsed.isna()
    if failed.any():
        parsed[failed] = pd.to_datetime(series[failed], errors="coerce", format="mixed")
    if ambiguous.any():
        parsed[ambiguous] = pd.to_datetime(series[ambiguous], errors="coerce", format="mixed",
                                           dayfirst=date_order == "day_first")
    return parsed


def _timestamp(value, label, date_order=None):
    """Reads one typed date by the same rules as the step's date column, so 03/04/2024 means the same in both."""
    if date_order is None and re.match(_AMBIGUOUS_DATE, str(value)):
        raise RecipeError(f"{label} '{value}' could be March or April. Say whether the day or the month comes first, "
                          "or write it like 2024-04-03.")
    parsed = _dates(pd.Series([value], dtype=object), label, date_order).iloc[0]
    if pd.isna(parsed):
        raise RecipeError(f"{label} '{value}' isn't a date this app can read. Write it like 2010-12-01.")
    return parsed


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
            left = _dates(values, step["column"], step.get("date_order"))
            right = _timestamp(target, "The date", step.get("date_order"))
        elif step.get("compare_as") == "number":
            try:
                left, right = pd.to_numeric(values, errors="coerce"), float(target)
            except ValueError:
                raise RecipeError(f"'{target}' isn't a number, so '{step['column']}' can't be compared with it.")
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


def _new_column(table, name, table_name):
    if name in table.columns:
        raise RecipeError(f"'{table_name}' already has a column called '{name}'. Choose a different name.")


def _calculate(step, tables, folder):
    table = _table(tables, step["table"]).copy()
    _new_column(table, step["into"], step["table"])
    first = _numbers(_column(table, step["first"], step["table"]), step["first"])
    second = _numbers(_column(table, step["second"], step["table"]), step["second"])
    operation = step["operation"]
    if operation == "multiply":
        result = first * second
    elif operation == "add":
        result = first + second
    elif operation == "subtract":
        result = first - second
    else:
        result = first / second.where(second != 0)
    table[step["into"]] = result.round(10)     # removes tiny decimal noise
    tables[step["table"]] = table
    blank = int(result.isna().sum())
    return step["table"], len(table), f"{_count(blank, 'row')} couldn't be calculated and left empty." if blank else ""


def _reshape_long(step, tables, folder):
    table = _table(tables, step["table"])
    for column in step["id_columns"] + step["value_columns"]:
        _column(table, column, step["table"])
    for name in (step["names_to"], step["values_to"]):
        if name in step["id_columns"]:
            raise RecipeError(f"'{name}' is already one of the columns being kept. Choose a different name.")
    tables[step["into"]] = table.melt(id_vars=step["id_columns"], value_vars=step["value_columns"],
                                      var_name=step["names_to"], value_name=step["values_to"])
    blank = int(tables[step["into"]][step["values_to"]].isna().sum())
    return step["into"], len(table), f"{_count(blank, 'new row')} with an empty value." if blank else ""


def _measure_events(measure, rows, keys):
    if measure["how"] == "count_rows":
        return rows.groupby(keys).size()
    if measure["how"] == "any_event":
        return pd.Series(1, index=keys.dropna().unique()).sort_index()
    return _numbers(rows[measure["column"]], measure["column"]).groupby(keys).sum().round(10)


def _count_within_window(step, tables, folder):
    """For each row, measures its events from its start date through the chosen number of calendar days."""
    table = _table(tables, step["table"]).copy()
    events = _table(tables, step["events"])
    for measure in step["measures"]:
        _new_column(table, measure["into"], step["table"])
        if measure["how"] == "sum":
            _column(events, measure["column"], step["events"])
    keys = _column(table, step["key"], step["table"])
    repeats = int(keys.dropna().duplicated().sum())
    if repeats:
        raise RecipeError(f"'{step['key']}' repeats in '{step['table']}' ({repeats:,} extra rows), so each row can't "
                          "have its own window. Roll that table up to one row per ID first.")
    order = step.get("date_order")
    starts = _dates(_column(table, step["start_column"], step["table"]), step["start_column"], order)
    if step.get("default_start"):
        starts = starts.fillna(_timestamp(step["default_start"], "The start date", order))
    no_start = starts.isna() & keys.notna()
    event_keys = _column(events, step["event_key"], step["events"])
    event_dates = _dates(_column(events, step["event_date"], step["events"]), step["event_date"], order)

    start_of = pd.Series(starts.to_numpy(), index=keys.to_numpy())
    start_of = start_of[start_of.index.notna()]
    event_starts = event_keys.map(start_of)
    # The window opens at the start itself and closes at midnight after the last calendar day.
    event_ends = event_starts.dt.normalize() + pd.Timedelta(days=step["days"])
    inside = (event_dates >= event_starts) & (event_dates < event_ends)

    rows = events[inside]
    for measure in step["measures"]:
        found = _measure_events(measure, rows, event_keys[inside])
        result = pd.Series(step["no_events_value"], index=table.index, dtype=object)
        has_events = keys.isin(found.index)
        result[has_events] = keys[has_events].map(found).astype(object)
        result[no_start | keys.isna()] = np.nan
        table[measure["into"]] = result
    tables[step["table"]] = table

    known = event_starts.notna()
    parts = [f"{int(inside.sum()):,} of {len(events):,} events fell inside the window."]
    for count, words in ((int((known & (event_dates < event_starts)).sum()), "before it started"),
                         (int((known & (event_dates >= event_ends)).sum()), "after it ended"),
                         (int((~event_keys.isin(start_of.index)).sum()), "belonged to no row in the table"),
                         (int((events[step["event_date"]].notna() & event_dates.isna()).sum()), "had an unreadable date")):
        if count:
            parts.append(f"{_count(count, 'event')} {words}.")
    if int(no_start.sum()):
        parts.append(f"{_count(int(no_start.sum()), 'row')} had no start date and left empty.")
    latest = event_dates.max()
    last_days = starts.dt.normalize() + pd.Timedelta(days=step["days"] - 1)
    cut_off = int((last_days > latest.normalize()).sum()) if pd.notna(latest) else 0
    if cut_off:
        parts.append(f"{_count(cut_off, 'row')} had a window running past the last event date, "
                     f"{latest:%Y-%m-%d}, so may be undercounted.")
    return step["table"], len(table), " ".join(parts)


OPERATIONS = {"read_table": _read_table, "choose_columns": _choose_columns, "stack": _stack, "clean_key": _clean_key,
              "filter": _filter, "roll_up": _roll_up, "match": _match, "flag_membership": _flag_membership,
              "fill_missing": _fill_missing, "calculate": _calculate, "reshape_long": _reshape_long,
              "count_within_window": _count_within_window}


def run_steps(recipe, folder):
    """Carries out the steps in order. If one fails, the error says which, and holds the work so far."""
    tables, log = {}, []
    for number, step in enumerate(recipe["steps"], 1):
        try:
            name, rows_before, note = OPERATIONS[step["op"]](step, tables, folder)
        except Exception as cause:
            error = cause if isinstance(cause, RecipeError) else RecipeError(
                f"This step couldn't be carried out ({type(cause).__name__}). Check the columns it uses.")
            error.step, error.tables, error.log = number, tables, log
            raise error
        log.append({"step": number, "table": name, "rows_before": rows_before,
                    "rows_after": len(tables[name]), "note": note})
    return tables, log


def run_recipe(recipe, folder):
    """Runs a confirmed recipe. Returns the final table and, for each step, what it did to the row count."""
    waiting = [number for number, step in enumerate(recipe["steps"], 1) if step.get("status") != "confirmed"]
    if waiting:
        label = "Steps" if len(waiting) > 1 else "Step"
        verb = "need" if len(waiting) > 1 else "needs"
        raise RecipeError(f"{label} {', '.join(map(str, waiting))} still {verb} your confirmation before this can run.")
    tables, log = run_steps(recipe, folder)
    return _table(tables, recipe["output"]).reset_index(drop=True), log


def freeze(table):
    """The final table as exact CSV bytes. These bytes are what gets fingerprinted and analyzed."""
    return table.to_csv(index=False, lineterminator="\n").encode("utf-8")


def read_options(table):
    """How to read the frozen table back without changing any value: pass these to pandas.read_csv.

    Only an empty cell is missing, as in the recipe, so a label like NA stays a label. A text column is
    read as numbers only when that changes nothing, so 00123 keeps its zeros and 1 and 1.0 stay apart."""
    text = []
    for column in table.columns:
        if pd.api.types.is_numeric_dtype(table[column]):
            continue
        values = table[column].dropna()
        written = values[values.map(lambda value: isinstance(value, str))]    # a step may also have made numbers
        if (pd.to_numeric(values.astype(str), errors="coerce").isna().any() or written.str.match(r"[+-]?0\d").any()
                or pd.to_numeric(written).nunique() != written.nunique()):
            text.append(str(column))
    return {"dtype": {column: "str" for column in text}, "keep_default_na": False, "na_values": [""]}
