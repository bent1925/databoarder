"""Checks that AI-proposed recipes rebuild the right table. Needs ANTHROPIC_API_KEY and calls the real model, which costs money.

Run:  python -m evals.recipe_cases          (exit code 1 if it fails)
Needs the fake analyst pull in local_data/hillstrom. See tools/make_messy_hillstrom.py.

The goal below names the columns, so the check can compare tables directly. It tests whether the AI finds the right
steps. It doesn't test whether a marketer's vague goal would get there, or whether its assumptions are any good.
"""
import importlib.util
import io
import sys
from pathlib import Path

import pandas as pd

from app.ingest import build_table, inspect_files, suggest_read_steps
from app.proposer import propose
from app.schemas import DataRecipe

ROOT = Path(__file__).resolve().parent.parent
PULL = ROOT / "local_data" / "hillstrom"
GOAL = ("Did the emails change visits and spending? I need one row per customer with these exact columns: customer_id "
        "(digits only), segment (the campaign, or 'No E-Mail'), emailed (1 or 0), visit (1 or 0), conversion (1 or 0), "
        "spend (0 if none), and every customer attribute.")


def main() -> int:
    spec = importlib.util.spec_from_file_location("make_messy_hillstrom", ROOT / "tools" / "make_messy_hillstrom.py")
    tool = importlib.util.module_from_spec(spec)
    sys.modules["make_messy_hillstrom"] = tool
    spec.loader.exec_module(tool)

    folder = PULL / "pull_full"
    reads = suggest_read_steps(inspect_files(sorted(folder.iterdir())))
    result = propose(DataRecipe(steps=reads, output=reads[0]["into"]), folder, GOAL)
    print("Assumptions the AI listed:", *result["assumptions"], sep="\n  - ")
    if result["problem"]:
        print("FAIL:", result["problem"])
        return 1
    steps = [{**step, "status": "confirmed"} for step in reads + result["steps"]]
    built = build_table(DataRecipe(steps=steps, output=result["output"]), folder)
    got = pd.read_csv(io.BytesIO(built.csv_bytes)).sort_values("customer_id").reset_index(drop=True)
    key = pd.read_csv(PULL / "answer_key" / "analysis_table.csv")
    missing = [column for column in key.columns if column not in got.columns]
    if missing:
        print("FAIL: missing columns:", missing)
        return 1
    passed = tool.same_table(got, key)
    print("PASS: rebuilt the answer table" if passed else "FAIL: the table doesn't match the answer key")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
