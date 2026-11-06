# Databoarder

Ask a business question, upload a CSV, get a plain-language A/B test report.

## Run
    python -m pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...   # optional; without it, columns are guessed from names
    python -m uvicorn app.main:app   # then open http://localhost:8000
    python -m pytest

## Design
- The LLM only translates a question into an `ABTestSpec` (app/translator.py). It never computes.
- Statistics are fixed, tested code (app/ab_test.py, app/modes.py); data checks run first (app/checks.py).
- The read-back sentence is written by code from the spec, so the user confirms what will actually run.
- Three kinds of work: **Quick and dirty** (caveats at the end), **Need the right answer** (independent
  cross-checks, shaky things flagged up top), **Research** (subgroup scan with Holm correction).

## Method tabs (app/knowledge.py is the single source of truth)
- **Assumptions** per method, each marked checked (data looked fine), concern, or "you confirm".
- **Technical**: textbook-correct definition of every number.
- **In practice**: what each number means, in plain words.
- **Reproduce**: data fingerprint (sha256), library versions, seed, code per number, and a standalone script
  that reproduces every number in the report (including cross-checks and slices).

## Correctness and reproducibility guards (tests/)
- Newcombe range and pooled z-test re-derived by hand; Welch range and detectable gap re-derived by hand.
- Simulations: the 95% range covers the truth about 95% of the time; false alarms stay near 5%.
- The generated script is executed in every mode and its output compared with the report.
- Same data in, identical report out (fixed seed).

## Analyses
- **A/B test** (`app/ab_test.py`): two groups shown different versions at the same time.
- **Before/after** (`app/before_after.py`): did a number change after a date? You supply the date; it is never searched for.
  Whole 7-day weeks are counted outward from the date and compared with Welch's t-test, so weekday swings cancel and
  linked neighbouring days don't fake precision (simulated coverage ~95%). Cross-checks: fake date, allowing for a trend,
  nearby weeks only, day-by-day model. Research mode scans slices. Sample: `sample_data/signup_page_change.csv`
  (made by `tools/make_signup_sample.py`; planted truth is in its docstring).
