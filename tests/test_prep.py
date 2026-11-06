import contextlib
import io
import re

import pandas as pd
import pytest

from app.analysis import analyze
from app.prep import prepare
from app.schemas import ABTestSpec


def messy(n=400, seed=1):
    import numpy as np
    rng = np.random.default_rng(seed)
    arm = rng.choice(["control", "new"], n)
    spend = np.round(rng.gamma(2, 30, n) + (arm == "new") * 6, 2)
    return pd.DataFrame({" arm ": arm, "spend": [f"${v:,.2f}" for v in spend],
                         "bought": np.where(rng.random(n) < 0.3 + 0.05 * (arm == "new"), "Yes", "No"),
                         "note": [" x " if i % 2 else "y" for i in range(n)], "blank": [None] * n})


def test_each_fix_is_applied_and_described():
    clean, steps, notes = prepare(messy())
    assert "arm" in clean.columns and "blank" not in clean.columns
    assert pd.api.types.is_numeric_dtype(clean.spend) and set(clean.bought.unique()) == {0, 1}
    assert len(steps) >= 4 and all(s.text and s.code for s in steps)


def test_group_labels_stay_as_words_and_clean_data_changes_nothing():
    df = pd.DataFrame({"joined_program": ["yes", "no"] * 10, "x": range(20)})
    clean, steps, _ = prepare(df)
    assert steps == [] and clean.equals(df)


def test_ambiguous_things_are_flagged_not_changed():
    df = pd.DataFrame({"a": [1, 1, 2, 2, 3, 3], "b": list("xxyyzz")})
    clean, steps, notes = prepare(df)
    assert len(clean) == 6 and notes and "exact copies" in notes[0]
    wide = pd.DataFrame({"id": [1, 2], "2026-01-01": [1, 2], "2026-01-02": [3, 4], "2026-01-03": [5, 6]})
    assert any("one column per date" in n for n in prepare(wide)[2])


@pytest.mark.parametrize("mode", ["quick", "right_answer"])
def test_script_starts_from_the_raw_csv_and_matches_the_report(mode, tmp_path):
    raw = messy()
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    clean, steps, _ = prepare(pd.read_csv(path))
    rep = analyze(clean, ABTestSpec(group_col="arm", control="control", treatment="new", outcome_col="spend"),
                  csv_name=str(path), mode=mode, prep_steps=steps)
    assert rep.prep and "Data preparation" in rep.script
    assert rep.script.index("read_csv") < rep.script.index("Data preparation")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(rep.script, "repro", "exec"), {"__name__": "__main__"})
    m = re.search(r"diff=(-?[\d.]+)", buf.getvalue())
    assert m and abs(float(m.group(1)) - rep.effect["diff"]) < 1e-2
