"""Builds sample_data/signup_page_change.csv: a made-up signup page redesign on 2026-08-30.

Planted truth (so the report can be checked against it):
  desktop signup rate +1.0 pts after the date, mobile +2.5 pts; channel makes no difference.
  weekends convert ~1.5 pts better; neighbouring days move together; 2 days of tracking outage.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
days = pd.date_range("2026-07-01", periods=120, freq="D")
change = pd.Timestamp("2026-08-30")
outage = {pd.Timestamp("2026-07-19"), pd.Timestamp("2026-09-20")}
mood = 0.0
rows = []
for day in days:
    mood = 0.5 * mood + rng.normal(0, 0.008)           # slow wobble shared by the whole day
    if day in outage:
        continue
    weekend = day.weekday() >= 5
    n = rng.poisson(110 if weekend else 150)
    device = rng.choice(["mobile", "desktop"], n, p=[0.55, 0.45])
    channel = rng.choice(["search", "social", "email"], n, p=[0.5, 0.3, 0.2])
    base = np.where(device == "mobile", 0.060, 0.090) + (0.015 if weekend else 0.0) + mood
    lift = np.where(device == "mobile", 0.025, 0.010) if day >= change else 0.0
    p = np.clip(base + lift, 0.001, 0.99)
    signed = (rng.random(n) < p).astype(int)
    pages = 1 + rng.poisson(3.0 + (0.3 if day >= change else 0.0), n).astype(float)
    pages[rng.random(n) < 0.01] = np.nan
    rows.append(pd.DataFrame({"visit_date": day.date().isoformat(), "device": device, "channel": channel,
                              "signed_up": signed, "pages_viewed": pages}))
df = pd.concat(rows, ignore_index=True)
df.to_csv("sample_data/signup_page_change.csv", index=False)
df.to_csv("static/signup_page_change.csv", index=False)
print(df.shape)
