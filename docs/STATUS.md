# Where things stand

A running note of what we're working on, updated at the end of each work session.
The full plan is in [plans/data-ingestion.md](plans/data-ingestion.md).

**Last updated:** 2026-11-10

## Right now

- **Branch:** `data-stage-finish`, pushed to GitHub, **not merged**. All 259 tests pass.
- **Paused** at the end of the day. Nothing half-finished in the code.

## The project in one paragraph

Databoarder helps nontechnical people answer "did this campaign work?" questions. The current work is
**step 1, the data stage**: someone uploads the messy files their analyst sent, confirms a list of
preparation steps, and gets one clean, frozen table that the existing analysis screens use.

## Done

- **Step 0, test data:** a fake messy analyst pull built from the Hillstrom dataset, with a known answer.
- **Step 1, merged to `main` (PR #2):** uploading files, the review screen, preparation steps, AI-suggested
  steps that need confirming, and hiding IDs, emails and phone numbers from the AI.
- **On `data-stage-finish` (not merged yet):**
  - Exact library versions pinned, so results can be reproduced.
  - The downloadable script stops if the original files or the rebuilt table don't match what the report used.
    Tested in every mode.
  - The column the person confirmed while building the table (who got the campaign, or the date) now carries
    into the question screens instead of being guessed again.

## Next up, in order

1. **Click through the new screens in a browser.** Only the server side is tested.
2. **Recognize two more messy formats:** IDs Excel turned into `9.941E+14`, and codes like `XX` meaning
   "missing". The `XX` one can silently count missing values as real data.
3. **Merge `data-stage-finish`.**
4. **Step 2:** a ready-to-forward note telling the analyst exactly what's missing from a pull.

## Waiting on you

- **Real-model test of AI suggestions.** Needs an API key and costs money. Run
  `ANTHROPIC_API_KEY=… .venv/bin/python -m evals.recipe_cases`, or give me the key.
- **Deployment for real customer data.** One shared password isn't suitable. Options: a separate deployment
  per company, or running inside the customer's environment. Also shapes step 4.
- **A real analyst pull,** to check size limits and whether the AI suggestions help.

## Decided

- **People on more than one row:** a table can't be confirmed until it has one row per person. Clustered error
  estimates are on hold until a real pull needs them. (Recommended 2026-11-10; confirm if you disagree.)

## Later

- Step 3: save the analyst's query with the report.
- Step 4: warehouse connections, once there's a pilot company.
- Step 5: more kinds of analysis, chosen from what users ask.
- Tidy-up: delete the merged `data-ingestion-plan` and `plain-language-steps` branches.

## Setup notes

- Tests: `.venv/bin/python -m pytest` (install with `.venv/bin/pip install -r requirements-dev.txt`).
