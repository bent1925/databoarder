"""Science mode: a standalone script that reproduces every number in a report.

It checks the data file is the one the report used, makes the same preparation steps, and then runs the engine,
whose source is copied in word for word. The tests run it and compare its output with the report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels

ENGINE = Path(__file__).with_name("engine.py")


def versions() -> dict:
    return {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "statsmodels": statsmodels.__version__}


def build(spec: dict, data_name: str, data_sha: str, prep_block: str) -> str:
    engine = ENGINE.read_text().replace("from __future__ import annotations\n", "", 1)
    built = ", ".join(f"{k} {v}" for k, v in versions().items())
    return f'''from __future__ import annotations
# Databoarder science mode: reproduction script.
# Put it in the same folder as {data_name}, then run: python this_script.py
# It prints every number in the report as JSON. Nothing in it is random.
# Built with: {built}

import hashlib
import io
import json
import sys

import pandas as pd

DATA_FILE = {data_name!r}
DATA_SHA256 = {data_sha!r}

raw = open(DATA_FILE, "rb").read()
if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
    sys.exit(f"{{DATA_FILE}} isn't the file the report used (its fingerprint differs). Use the original file.")
df = pd.read_csv(io.BytesIO(raw))
{prep_block}
# --- The choices you confirmed ---
SPEC = json.loads({json.dumps(json.dumps(spec, ensure_ascii=False), ensure_ascii=False)})

# --- The analysis engine, copied word for word from the app ---
{engine}

if __name__ == "__main__":
    print(json.dumps(run(df, SPEC), indent=2, ensure_ascii=False))
'''
