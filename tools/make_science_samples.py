"""Builds the science-mode sample tables in static/science/. Synthetic, with planted answers, fixed seed.

    python tools/make_science_samples.py

tumour_volume.csv       one column per group, different mice in each: a dose effect (High and Mid shrink tumours),
                        one empty cell. Fits one-way ANOVA with Dunnett's test against Vehicle.
blood_pressure.csv      the same 12 patients at three visits: pressure falls by about 8 mmHg by week 8, one patient
                        missed week 8. Fits repeated-measures ANOVA.
infection_counts.csv    a 2x2 table of counts: 8 of 100 vaccinated infected vs 30 of 100 on placebo.
gene_expression.csv     mRNA and protein levels for 24 samples, moderately correlated (true r about 0.6).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "static" / "science"


def tumour(rng) -> pd.DataFrame:
    means = {"Vehicle": 820, "Low dose": 760, "Mid dose": 610, "High dose": 450}
    table = pd.DataFrame({k: np.round(rng.normal(m, 0.18 * m, 8), 0) for k, m in means.items()})
    table.loc[5, "Mid dose"] = np.nan       # a mouse that was culled early
    return table


def pressure(rng) -> pd.DataFrame:
    n = 12
    base = rng.normal(148, 10, n)
    table = pd.DataFrame({"Patient": [f"P{i + 1:02d}" for i in range(n)],
                          "Baseline": np.round(base + rng.normal(0, 4, n), 0),
                          "Week 4": np.round(base - 4 + rng.normal(0, 4, n), 0),
                          "Week 8": np.round(base - 8 + rng.normal(0, 4, n), 0)})
    table.loc[9, "Week 8"] = np.nan         # missed the last visit
    return table


def infection() -> pd.DataFrame:
    return pd.DataFrame({"Group": ["Vaccinated", "Placebo"], "Infected": [8, 30], "Not infected": [92, 70]})


def expression(rng) -> pd.DataFrame:
    n = 24
    mrna = rng.lognormal(2.0, 0.5, n)
    protein = 3 + 0.9 * (mrna - mrna.mean()) / mrna.std() + rng.normal(0, 1.2, n)
    return pd.DataFrame({"Sample": [f"S{i + 1}" for i in range(n)], "mRNA (TPM)": np.round(mrna, 2),
                         "Protein (ng/mL)": np.round(protein, 2)})


def main() -> None:
    rng = np.random.default_rng(2026)
    OUT.mkdir(parents=True, exist_ok=True)
    tumour(rng).to_csv(OUT / "tumour_volume.csv", index=False)
    pressure(rng).to_csv(OUT / "blood_pressure.csv", index=False)
    infection().to_csv(OUT / "infection_counts.csv", index=False)
    expression(rng).to_csv(OUT / "gene_expression.csv", index=False)


if __name__ == "__main__":
    main()
