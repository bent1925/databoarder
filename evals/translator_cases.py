"""Checks that the question -> form translation picks the right columns. Needs ANTHROPIC_API_KEY (calls the real model).

Run:  python -m evals.translator_cases          (prints a pass/fail table; exit code 1 if any fail)
Add a case whenever a real question is mistranslated. Each case states ONLY what must be right.
"""
import sys

import pandas as pd

from app.translator import interpret

CASES = [  # (file, design, question, expected field values)
    ("checkout_test", "random", "Did the new checkout raise purchases?",
     {"analysis_type": "ab_test", "treatment": "new_checkout", "outcome_col": "purchased"}),
    ("signup_page_change", "one_date", "Did signups change after Aug 30?",
     {"analysis_type": "before_after", "change_date": "2026-08-30", "outcome_col": "signed_up"}),
    ("signup_page_change", "one_date", "Did pages viewed go up after the redesign?",
     {"analysis_type": "before_after", "change_date": "", "outcome_col": "pages_viewed"}),   # no date given: must NOT invent one
    ("regional_rollout", "one_date_comparison", "Did the pilot cities sign up more than holdout cities after Aug 30?",
     {"analysis_type": "diff_in_diff", "treated": "pilot", "comparison": "holdout", "change_date": "2026-08-30",
      "outcome_col": "signed_up"}),
    ("loyalty_program", "chose", "Do loyalty members spend more in the next 90 days?",
     {"analysis_type": "observational", "treatment": "yes", "outcome_col": "spend_next_90d"}),
    ("loyalty_program", "chose", "Does joining make people renew?",
     {"analysis_type": "observational", "outcome_col": "renewed"}),
]


def main() -> int:
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY: these cases test the real model, so nothing was run.")
        return 0
    bad = 0
    for f, design, q, want in CASES:
        df = pd.read_csv(f"sample_data/{f}.csv")
        res = interpret(q, df, design)
        got = res.spec.model_dump()
        wrong = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
        bad += bool(wrong) or res.source != "llm"
        print(("PASS" if not wrong and res.source == "llm" else "FAIL"), f, "|", q, "|", wrong or f"used {res.source}; notes: {res.notes}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
