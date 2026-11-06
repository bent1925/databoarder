"""Builds sample_data/regional_rollout.csv: a new signup page shown only in 'pilot' cities from 2026-08-30.

Planted truth (so the report can be checked against it):
  BOTH city groups get +2.0 pts from the date (a shared seasonal shock), weekends +1.5 pts.
  Pilot cities get an EXTRA +2.4 pts on top (mobile +3.2, desktop +1.4): the true effect.
  Pilot cities started 1 pt higher. Naive before/after on the pilot alone would say ~+4.4 pts.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
days = pd.date_range("2026-07-01", periods=120, freq="D")
change = pd.Timestamp("2026-08-30")
mood = 0.0
rows = []
for day in days:
    mood = 0.5 * mood + rng.normal(0, 0.006)            # a wobble that hits both groups the same day
    weekend = day.weekday() >= 5
    for grp, base_rate, n0 in (("pilot", 0.080, 200), ("holdout", 0.070, 200)):
        n = rng.poisson(n0 * (0.75 if weekend else 1.0))
        device = rng.choice(["mobile", "desktop"], n, p=[0.55, 0.45])
        p = base_rate + (0.015 if weekend else 0.0) + mood
        p = np.full(n, p) + (0.020 if day >= change else 0.0)
        if grp == "pilot" and day >= change:
            p = p + np.where(device == "mobile", 0.032, 0.014)
        signed = (rng.random(n) < np.clip(p, 0.001, 0.99)).astype(int)
        pages = 1 + rng.poisson(3.0 + (0.2 if day >= change else 0.0) + (0.3 if grp == "pilot" and day >= change else 0.0), n).astype(float)
        rows.append(pd.DataFrame({"visit_date": day.date().isoformat(), "city_group": grp, "device": device,
                                  "signed_up": signed, "pages_viewed": pages}))
df = pd.concat(rows, ignore_index=True)
df.to_csv("sample_data/regional_rollout.csv", index=False)
df.to_csv("static/regional_rollout.csv", index=False)
print(df.shape)
