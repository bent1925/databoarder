"""Builds sample_data/lalonde_cps.csv: LaLonde's job-training program (185 trained people) vs ~16,000 survey respondents.

Benchmark truth: the randomized experiment (nsw_mixtape) gives about +$1,794 in 1978 earnings.
Needs: pip install causaldata
"""
import pandas as pd
from causaldata import cps_mixtape, nsw_mixtape

n, c = nsw_mixtape.load_pandas().data, cps_mixtape.load_pandas().data
df = pd.concat([n[n.treat == 1], c]).reset_index(drop=True)
df.insert(0, "trained", df.treat.map({1: "yes", 0: "no"}))
df = df.drop(columns=["data_id", "treat"])
df.to_csv("sample_data/lalonde_cps.csv", index=False)
print(df.shape)
