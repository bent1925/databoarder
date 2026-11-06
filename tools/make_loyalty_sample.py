"""Builds sample_data/loyalty_program.csv: customers who CHOSE to join a loyalty program.

Planted truth (so the report can be checked against it):
  Joining raises 90-day spend by +$5 and renewal chance by about +6 pts.
  But big past spenders (and enthusiasts, who are NOT in the file) join far more often,
  so the raw gap is ~+$25. Adjusting for past spend removes most of it; the enthusiasm
  that isn't in the file leaves a bit of upward bias. That is exactly the honest caveat.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(21)
n = 5000
region = rng.choice(["North", "South", "East", "West"], n, p=[0.3, 0.25, 0.25, 0.2])
device = rng.choice(["mobile", "desktop"], n, p=[0.6, 0.4])
age = np.clip(rng.normal(42, 13, n), 18, 80).round()
past_spend = np.clip(rng.lognormal(4.4, 0.6, n), 5, 900).round(2)
enthusiasm = rng.normal(0, 1, n)                      # NOT saved in the file
z = (past_spend - 100) / 70
join_p = 1 / (1 + np.exp(-(-0.6 + 0.9 * z + 0.5 * enthusiasm + 0.3 * (region == "West"))))
joined = rng.random(n) < join_p
spend = 15 + 0.45 * past_spend + 0.12 * age + 7 * enthusiasm + 5.0 * joined + rng.normal(0, 25, n)
renew_p = 1 / (1 + np.exp(-(-0.3 + 0.5 * z + 0.4 * enthusiasm + 0.26 * joined)))
renewed = rng.random(n) < renew_p
df = pd.DataFrame({
    "customer_id": np.arange(1, n + 1), "joined_program": np.where(joined, "yes", "no"),
    "region": region, "device": device, "age": age.astype(int), "past_spend": past_spend,
    "spend_next_90d": np.round(np.maximum(spend, 0), 2), "renewed": renewed.astype(int),
})
df.to_csv("sample_data/loyalty_program.csv", index=False)
df.to_csv("static/loyalty_program.csv", index=False)
print(df.shape, "raw spend gap:", round(df.groupby("joined_program").spend_next_90d.mean().diff().iloc[-1], 1))
