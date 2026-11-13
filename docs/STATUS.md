# Where things stand

A running note of what we're working on, updated at the end of each work session.
The full plan is in [plans/data-ingestion.md](plans/data-ingestion.md).

**Last updated:** 2026-11-13

## Right now

- **Branch:** `data-stage-finish`, pushed to GitHub, **not merged**, no PR yet. All 259 tests pass.
- **Clicked through in a browser (2026-11-13).** It works, but combining files is the bottleneck: too many
  options, too noisy for a nontechnical person.

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

1. **Codex review of `data-stage-finish`.** Open a PR to `main` and fix what the review finds.
2. **Merge `data-stage-finish`.**
3. **One-screen combine flow.** Start with a clickable mockup, not code. Replace the step-by-step recipe builder
   with: upload the files, see the finished table with sanity numbers ("10,000 customers, 50% got the email"),
   and answer only the 3–4 choices that change the answer (who counts, who got the campaign, what counts as
   the result and over what window, what to do with people who don't match or appear twice). Each answer
   is shown as a suggestion until confirmed. Mechanical fixes (trimming, stacking, reading Excel) are applied
   by fixed code and listed, collapsed, with row counts. Matching IDs across files is done by code, not the AI.
   Possibly ask the question before the upload so the table is built for it.
4. **Recognize more messy formats:** IDs Excel turned into `9.941E+14`, codes like `XX` meaning "missing",
   and `YYYYMMDD` dates. The `XX` one can silently count missing values as real data. These fit the
   "mechanical fixes" part of the new flow.
5. **Step 2:** a ready-to-forward note telling the analyst exactly what's missing from a pull.

## Open question: why not just use Claude or ChatGPT?

A chatbot already combines files and answers "did it work?" easily. The pitch is "an answer you can defend":
no hidden choices, the same answer every time, no causal overclaiming, and a result someone else can check.
That only matters if users get challenged on their numbers. Check with one or two real users how they answer
these questions today, and what goes wrong. Harder-to-copy parts: the analyst note, saved queries, warehouse
connections, and monthly reruns with the same choices.

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
