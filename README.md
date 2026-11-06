# Dashboarder

Ask a business question, upload a CSV, get a plain-language A/B test report.

## Run
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...   # optional; without it, columns are guessed from names
    uvicorn app.main:app --reload   # then open http://localhost:8000
    pytest

## Design
- The LLM only translates a question into an `ABTestSpec` (app/translator.py). It never computes.
- Statistics are fixed, tested code (app/ab_test.py); data checks run first (app/checks.py).
- The read-back sentence is written by code from the spec, so the user confirms what will actually run.
- Reports have three layers: answer, evidence, method (every number + code + reproducible script).

## Modes
- **Quick and dirty** - same analysis; caveats listed at the end.
- **Need the right answer** - adds independent cross-checks (exact/rank test, shuffle test, trimmed run) and puts anything shaky at the top.
- **Research** - also scans subgroups with a Holm correction so noise isn't mistaken for findings.
Prepared questions are built from your columns and carry a finished spec; "Other" goes through the translator and is always confirmed.
