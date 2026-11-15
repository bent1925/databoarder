"""Scores question rankers against the labelled cases. No API key needed, no network, same result every run.

Run:  python -m evals.recommender                 (every ranker; prints a table and saves it to evals/results/)
      python -m evals.recommender --check         (only checks the case files)
      python -m evals.recommender --show 5        (also prints the worst 5 goals per ranker, to see what it gets wrong)

Hillstrom's table isn't in git (local_data/). Without it, its cases are skipped and the saved result says so.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from . import cases as C
from .metrics import score_answerable
from .rankers import RANKERS, infeasible

RESULTS = C.ROOT / "evals" / "results"
METRICS = ["candidate_recall", "recall@3", "ndcg@5", "mrr"]


def load(only: list[str] | None = None):
    """(dataset file, original table, renamed table) for every case file whose table is present."""
    out, missing = [], []
    for spec in C.load_files():
        if only and spec["dataset"] not in only:
            continue
        path = C.ROOT / spec["path"]
        if not path.exists():
            missing.append(spec["dataset"])
            continue
        df = C.read_table(path)
        out.append((spec, df, df.rename(columns=spec.get("renames", {}))))
    return out, missing


def check(loaded) -> list[str]:
    found = []
    for spec, df, _ in loaded:
        found += C.problems(spec, df)
    ids = [g["id"] for spec, _, _ in loaded for g in spec["goals"]]
    found += [f"goal id used in two files: {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    return found


def evaluate(ranker, loaded) -> dict:
    """Per-case scores for one ranker, plus every infeasible question it showed."""
    rows, bad = [], []
    for spec, df, df_renamed in loaded:
        for case in C.expand(spec):
            table = df_renamed if case.renamed else df
            r = ranker(table, case.goal, case.design, case.confirmed)
            bad += [(case.case_id, s, why) for s in r.shown if (why := infeasible(table, s))]
            row = {"case": case, "said_cannot": r.cannot_answer}
            if case.unanswerable:
                row["reason_right"] = float(r.cannot_answer == case.unanswerable)
            else:
                row.update(score_answerable(r.shown, r.pool, case.labels))
                row["false_cannot"] = float(r.cannot_answer is not None)
            rows.append(row)
    return {"rows": rows, "infeasible": bad}


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if not r["case"].unanswerable]
    una = [r for r in rows if r["case"].unanswerable]
    out = {m: _mean(r[m] for r in ans) for m in METRICS}
    out["cant_answer_right"] = _mean(r["reason_right"] for r in una)
    out["false_cant_answer"] = _mean(r["false_cannot"] for r in ans)
    out["cases"], out["answerable"], out["unanswerable"] = len(rows), len(ans), len(una)
    return out


def breakdown(rows: list[dict]) -> dict:
    """The same summary per dataset, and for original vs renamed columns."""
    by_data, by_names = defaultdict(list), defaultdict(list)
    for r in rows:
        by_data[r["case"].dataset].append(r)
        by_names["renamed columns" if r["case"].renamed else "original columns"].append(r)
    return {k: summarize(v) for k, v in [*sorted(by_data.items()), *sorted(by_names.items(), reverse=True)]}


def _no_nan(x):
    """JSON has no NaN; a metric with no cases to average is saved as null."""
    if isinstance(x, dict):
        return {k: _no_nan(v) for k, v in x.items()}
    return None if isinstance(x, float) and x != x else x


def fmt_table(named: dict, first: str) -> str:
    cols = [*METRICS, "cant_answer_right", "false_cant_answer", "cases"]
    width = max(len(first), *(len(k) for k in named)) + 2
    lines = [first.ljust(width) + "".join(c.rjust(19) for c in cols)]
    for name, s in named.items():
        cells = [f"{s[c]:>19}" if c == "cases" else ("—".rjust(19) if s[c] != s[c] else f"{s[c]:>19.3f}") for c in cols]
        lines.append(name.ljust(width) + "".join(cells))
    return "\n".join(lines)


def worst(rows: list[dict], n: int) -> list[tuple[str, float, str]]:
    """Goals with the lowest recall@3, averaged over their paraphrases and renames."""
    by_goal = defaultdict(list)
    for r in rows:
        if not r["case"].unanswerable:
            by_goal[r["case"].goal_id].append(r)
    scored = sorted((_mean(r["recall@3"] for r in rs), gid, rs[0]["case"].goal) for gid, rs in by_goal.items())
    return [(gid, s, goal) for s, gid, goal in scored[:n]]


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "app", "evals/recommender",
                                "evals/recommender_cases"], cwd=C.ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evals.recommender")
    ap.add_argument("--check", action="store_true", help="only check the case files")
    ap.add_argument("--ranker", action="append", choices=sorted(RANKERS), help="score only this ranker")
    ap.add_argument("--dataset", action="append", help="score only this dataset")
    ap.add_argument("--show", type=int, default=0, metavar="N", help="print the N worst goals per ranker")
    ap.add_argument("--no-save", action="store_true", help="don't write to evals/results/")
    args = ap.parse_args(argv)

    loaded, missing = load(args.dataset)
    for m in missing:
        print(f"Skipped {m}: its table isn't on this computer.")
    found = check(loaded)
    if found:
        print("The case files have problems:\n  " + "\n  ".join(found))
        return 1
    if args.check:
        print(f"Case files OK: {sum(len(C.expand(s)) for s, _, _ in loaded)} cases from "
              f"{sum(len(s['goals']) for s, _, _ in loaded)} goals.")
        return 0

    results, failed = {}, False
    for name in args.ranker or RANKERS:
        res = evaluate(RANKERS[name], loaded)
        results[name] = {"summary": summarize(res["rows"]), "breakdown": breakdown(res["rows"]),
                         "infeasible_shown": len(res["infeasible"])}
        if res["infeasible"]:
            failed = True
            print(f"FAIL {name} showed {len(res['infeasible'])} questions the data can't support, e.g. "
                  f"{res['infeasible'][0]}")
        if args.show:
            print(f"\nWorst goals for {name} (recall@3):")
            for gid, s, goal in worst(res["rows"], args.show):
                print(f"  {s:.2f}  {gid}: {goal}")

    print()
    print(fmt_table({n: r["summary"] for n, r in results.items()}, "ranker"))
    for n, r in results.items():
        print(f"\n{n}, by dataset and column names:")
        print(fmt_table(r["breakdown"], "subset"))

    if not args.no_save:
        RESULTS.mkdir(exist_ok=True)
        commit = git_commit()
        out = RESULTS / f"recommender-{date.today().isoformat()}-{commit}.json"
        out.write_text(json.dumps({"date": date.today().isoformat(), "commit": commit,
                                   "datasets": [s["dataset"] for s, _, _ in loaded], "skipped": missing,
                                   "rankers": _no_nan(results)}, indent=2) + "\n")
        print(f"\nSaved {out.relative_to(C.ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
