"""The data stage: look at messy files, run a confirmed recipe, and freeze one analysis table.

AI may propose a recipe later, but it can only fill in the fixed steps in schemas.DataRecipe. The
person confirms each step, and nothing runs until they have.
"""
from __future__ import annotations

import hashlib
import io
import pprint
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .recipe_ops import RecipeError, freeze, read_grid, read_options, run_recipe, run_steps, table_from_grid
from .schemas import DataRecipe

RUNTIME_SOURCE = Path(__file__).with_name("recipe_ops.py")
FROZEN_NAME = "analysis_table.csv"
_LOOKS_NUMERIC = re.compile(r"^[-+]?[\d.,]+%?$")
_HEADER_SEARCH_ROWS = 30
_MOST_FOOTER_ROWS = 10


@dataclass
class FoundTable:
    """What a file seems to hold. The header and footer positions are guesses for the person to confirm."""
    file: str
    sheet: Optional[str]
    delimiter: str
    header_row: int
    footer_rows: int
    rows: int
    columns: list[str]


@dataclass
class BuiltTable:
    table: pd.DataFrame
    csv_bytes: bytes
    sha256: str
    inputs: list[dict] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    read_options: dict = field(default_factory=dict)    # how to read csv_bytes back without changing any value
    name: str = FROZEN_NAME                             # a file name none of the original files uses


def _guess_delimiter(path: Path) -> str:
    with open(path, "rb") as handle:
        sample = handle.read(65536)
    return max([",", ";", "\t"], key=lambda delimiter: sample.count(delimiter.encode()))


def guess_header_row(grid: pd.DataFrame) -> int:
    """The first row that is fully filled in with distinct labels rather than numbers."""
    top = grid.head(_HEADER_SEARCH_ROWS)
    width = int(top.notna().sum(axis=1).max()) if len(top) else 0
    for position in range(len(top)):
        cells = top.iloc[position].dropna().astype(str)
        if len(cells) == width and cells.is_unique and not cells.str.match(_LOOKS_NUMERIC).any():
            return position
    return 0


def guess_footer_rows(grid: pd.DataFrame, header_row: int) -> int:
    """Rows at the very end with at most half the columns filled in, such as a total line."""
    width = int(grid.iloc[header_row].notna().sum())
    filled = grid.iloc[header_row + 1:].notna().sum(axis=1).to_numpy()
    count = 0
    while count < min(_MOST_FOOTER_ROWS, len(filled)) and filled[-(count + 1)] * 2 <= width:
        count += 1
    return count


def _found(path: Path, sheet: Optional[str], delimiter: str) -> FoundTable:
    grid = read_grid(path, sheet, delimiter)
    header_row = guess_header_row(grid)
    footer_rows = guess_footer_rows(grid, header_row)
    table = table_from_grid(grid, header_row, footer_rows, path.name)
    return FoundTable(path.name, sheet, delimiter, header_row, footer_rows, len(table), list(table.columns))


def inspect_file(path: Path) -> list[FoundTable]:
    """One entry for a CSV, and one per sheet for an Excel workbook."""
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        with pd.ExcelFile(path) as workbook:
            sheets = list(workbook.sheet_names)
        return [_found(path, sheet, ",") for sheet in sheets]
    return [_found(path, None, _guess_delimiter(path))]


def inspect_files(paths: list[Path]) -> list[FoundTable]:
    return [found for path in paths for found in inspect_file(path)]


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _order(step) -> str:
    if not step.date_order:
        return ""
    first = "day" if step.date_order == "day_first" else "month"
    example = "3 April" if first == "day" else "4 March"
    return f" Dates like 03/04/2024 are read with the {first} first, as {example} 2024."


def describe(step) -> str:
    """Plain English for one step, written by code from the step's own fields."""
    kind = step.op
    if kind == "read_table":
        where = f", sheet '{step.sheet}'" if step.sheet else ""
        return (f"Read {step.file}{where} as '{step.into}', using row {step.header_row + 1} as the column names "
                f"and leaving out {step.footer_rows} row(s) at the bottom.")
    if kind == "choose_columns":
        return f"In '{step.table}', keep these columns: {', '.join(step.columns.values())}."
    if kind == "stack":
        return f"Stack {', '.join(step.tables)} on top of each other as '{step.into}'."
    if kind == "clean_key":
        rule = {"digits_only": "keeping only the digits", "trim": "trimming spaces",
                "lowercase": "trimming spaces and using lowercase"}[step.rule]
        return f"In '{step.table}', make '{step.into}' from '{step.column}' by {rule}, so IDs can be matched."
    if kind == "filter":
        words = step.test.replace("_", " ")
        as_date = " (as a date)" if step.compare_as == "date" else " (as a number)" if step.compare_as == "number" else ""
        return f"In '{step.table}', keep only rows where '{step.column}' {words} {step.value}{as_date}.{_order(step)}"
    if kind == "roll_up":
        parts = [f"{m.into} ({m.how.replace('_', ' ')}{' of ' + m.column if m.column else ''})" for m in step.measures]
        return f"Roll '{step.table}' up to one row per '{step.by}' as '{step.into}': {', '.join(parts)}."
    if kind == "match":
        return (f"Match '{step.left}' to '{step.right}' on '{step.left_key}' and '{step.right_key}', "
                f"bringing in {', '.join(step.bring.values())}. Rows without a match are kept.")
    if kind == "flag_membership":
        return (f"In '{step.table}', set '{step.into}' to {step.yes} if '{step.key}' is in '{step.other}', "
                f"and to {step.no} if it isn't.")
    if kind == "calculate":
        words = {"multiply": "multiplied by", "add": "plus", "subtract": "minus", "divide": "divided by"}[step.operation]
        return f"In '{step.table}', make '{step.into}' as '{step.first}' {words} '{step.second}'."
    if kind == "reshape_long":
        return (f"Turn the {len(step.value_columns)} columns from '{step.value_columns[0]}' to "
                f"'{step.value_columns[-1]}' in '{step.table}' into rows, as '{step.into}', with their names in "
                f"'{step.names_to}' and their values in '{step.values_to}'.")
    if kind == "count_within_window":
        fallback = f", or {step.default_start} for rows without one," if step.default_start else ""
        how = {"sum": "total of '{}'", "count_rows": "number of events", "any_event": "1 if any event, otherwise 0"}
        parts = [f"{m.into} ({how[m.how].format(m.column)})" for m in step.measures]
        return (f"For each row in '{step.table}', look at '{step.events}' rows with the same '{step.event_key}', "
                f"dated by '{step.event_date}', from '{step.start_column}'{fallback} "
                f"through {step.days} calendar day(s), counting the start day, to make "
                f"{', '.join(parts)}. Rows with no events get {step.no_events_value}.{_order(step)}")
    return f"In '{step.table}', treat empty values of '{step.column}' as {step.value}."


def build_table(recipe: DataRecipe, folder: Path) -> BuiltTable:
    """Runs a fully confirmed recipe and freezes the result."""
    folder = Path(folder)
    table, log = run_recipe(recipe.model_dump(), folder)
    for entry, step in zip(log, recipe.steps):
        entry["text"] = describe(step)
    csv_bytes = freeze(table)
    files = sorted({step.file for step in recipe.steps if step.op == "read_table"})
    inputs = [{"file": name, "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest()} for name in files]
    return BuiltTable(table, csv_bytes, hashlib.sha256(csv_bytes).hexdigest(), inputs, log, read_options(table),
                      output_name(files))


def output_name(input_files: list[str]) -> str:
    """The frozen table's file name, chosen so the reproduction script can't overwrite an original file."""
    taken = {name.lower() for name in input_files}
    name, number = FROZEN_NAME, 2
    while name.lower() in taken:
        name, number = f"analysis_table_{number}.csv", number + 1
    return name


def read_frozen(built: BuiltTable) -> pd.DataFrame:
    """The frozen table as the analysis will read it. Stops if reading it back would change any column."""
    table = pd.read_csv(io.BytesIO(built.csv_bytes), **built.read_options)
    for column in built.table.columns:
        before, after = built.table[column], table[column]
        if str(column) in built.read_options["dtype"]:
            before = before.astype(str).where(before.notna())
        elif not pd.api.types.is_numeric_dtype(before):
            before = pd.to_numeric(before.astype(str).where(before.notna()))
        if before.nunique() != after.nunique() or before.isna().sum() != after.isna().sum():
            raise RecipeError(f"Saving the finished table would change the values in '{column}', so it wasn't used. "
                              "Rename those values, or remove that column with a 'keep columns' step.")
    return table


def require_comparison(table: pd.DataFrame, group_column: str) -> None:
    """Stops when everyone is in the same group, because there would be no one to compare them with."""
    if group_column not in table.columns:
        raise RecipeError(f"The final table has no column called '{group_column}'.")
    groups = table[group_column].dropna().unique()
    if len(groups) < 2:
        only = f" ({groups[0]})" if len(groups) else ""
        raise RecipeError(f"Everyone in this data has the same value in '{group_column}'{only}, so there's no one to "
                          "compare them with. Ask your analyst to include people who didn't get the campaign, "
                          "or include data from before it started.")


def require_one_row_per_unit(table: pd.DataFrame, unit_column: str) -> None:
    """The methods assume each row is one person, order or day. Repeats would make results look too certain."""
    if unit_column not in table.columns:
        raise RecipeError(f"The final table has no column called '{unit_column}'.")
    values = table[unit_column]
    if values.isna().any():
        raise RecipeError(f"{int(values.isna().sum()):,} rows have no '{unit_column}'. Filter them out or roll them up.")
    repeats = int(values.duplicated().sum())
    if repeats:
        raise RecipeError(f"'{unit_column}' repeats, with {repeats:,} extra rows, so the table isn't one row per "
                          f"'{unit_column}'. Add a roll-up step, because repeated rows make results look more certain "
                          "than they are.")


def require_time_span(table: pd.DataFrame, time_column: str) -> None:
    """Without a comparison group, a before-and-after comparison needs more than one date."""
    if time_column not in table.columns:
        raise RecipeError(f"The final table has no column called '{time_column}'.")
    dates = pd.to_datetime(table[time_column], errors="coerce", format="mixed").dropna().dt.normalize()
    if dates.nunique() < 2:
        raise RecipeError(f"'{time_column}' doesn't hold at least two different dates, so there's nothing to compare. "
                          "Ask your analyst to include people who didn't get the campaign, or data from before it started.")


def suggest_read_steps(found: list[FoundTable]) -> list[dict]:
    """One proposed step per table found. These are guesses written by code, for the person to check."""
    steps, used = [], set()
    per_file = [item.file for item in found]
    for item in found:
        sheet = f"_{item.sheet}" if item.sheet and per_file.count(item.file) > 1 else ""
        base = re.sub(r"\W+", "_", Path(item.file).stem + sheet).strip("_") or "table"
        name, number = base, 2
        while name in used:
            name, number = f"{base}_{number}", number + 1
        used.add(name)
        steps.append({"op": "read_table", "status": "proposed", "file": item.file, "sheet": item.sheet,
                      "delimiter": item.delimiter, "header_row": item.header_row, "footer_rows": item.footer_rows,
                      "into": name})
    return steps


def preview_recipe(recipe: DataRecipe, folder: Path, sample_rows: int = 20) -> dict:
    """Shows what the steps would do, confirmed or not. Nothing is frozen or analyzed from a preview."""
    dumped = recipe.model_dump()
    for step in dumped["steps"]:
        step["status"] = "confirmed"
    error, failed_step = None, None
    try:
        tables, log = run_steps(dumped, Path(folder))
    except RecipeError as problem:
        tables, log, error, failed_step = problem.tables, problem.log, str(problem), problem.step
    final = tables.get(recipe.output) if error is None else None
    sample = None
    if final is not None:
        head = final.head(sample_rows)
        sample = head.astype(object).where(head.notna(), None).to_dict("records")
    return {"texts": [describe(step) for step in recipe.steps], "log": log, "error": error, "failed_step": failed_step,
            "tables": {name: list(table.columns) for name, table in tables.items()},
            "rows": None if final is None else len(final), "columns": None if final is None else list(final.columns),
            "sample": sample}


def script_block(recipe: DataRecipe, snapshot_name: str = FROZEN_NAME) -> str:
    """A standalone section of Python that rebuilds the frozen table from the original files."""
    dumped = recipe.model_dump()
    steps = "\n".join(f"# {number}. {_one_line(describe(step))}" for number, step in enumerate(recipe.steps, 1))
    return (f"# --- Building the analysis table from the original files ---\n"
            f"# Run this script in a folder that holds the original files.\n{steps}\n\n"
            f"{RUNTIME_SOURCE.read_text()}\n\n"
            f"RECIPE = {pprint.pformat(dumped, sort_dicts=False)}\n\n"
            f"_table, _log = run_recipe(RECIPE, '.')\n"
            f"with open({snapshot_name!r}, 'wb') as _handle:\n"
            f"    _handle.write(freeze(_table))\n"
            f"READ_OPTIONS = read_options(_table)    # read it back exactly as the app did\n")
