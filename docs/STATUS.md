# Where things stand

A running note of what we're working on, updated at the end of each work session.
The full plan is in [plans/data-ingestion.md](plans/data-ingestion.md).

**Last updated:** 2026-11-14

## How we work

- **Keep [README.md](../README.md) up to date.** Whenever a change is merged to `main`, check whether the README
  still describes the app correctly. If it doesn't, fix it in the same PR, or in a follow-up straight after.
  (Asked for 2026-11-14, when the README had fallen behind.)

## Right now

- **Branch:** `main`. Phases 1 and 2 of the one-screen combine flow are merged (PRs #5 and #6), and question
  recommender phase 1 (PR #8, labelled cases and the scoring harness, no app change), all reviewed by Codex.
  All 314 tests pass, including 7 browser tests.
- **Recommender baseline:** `.venv/bin/python -m evals.recommender`. Today's suggestions (Baseline A) put a
  well-fitting question in the top 3 for 41% of cases (36% with renamed columns), and never say "can't answer".
  **Waiting on you:** review the labels in `evals/recommender_cases/` (about 70 goals).
  **In progress (branch `review-checkout-labels`, not yet a PR):** reviewing them together in chat.
  Done: `checkout_test`, `signup_page_change`, `regional_rollout`. Applying the regional rules back to the earlier files: `checkout_test` done
  (2026-11-18); `signup_page_change` S1–S12 and the LaLonde "should we" paraphrases still to regrade. Next: `loyalty_program`, `lalonde_cps`
  (points to resume from are in the chat: require the "compare similar people" method for the causal goals, and
  make "worth the money" can't-answer), `hillstrom`. Rules so far: near answers on a different measure, or that
  ignore the dates an effect question needs, get no credit; "should we...?" goals are can't-answer (`decision`).
- **README** brought up to date with `main` (PR #9, merged 2026-11-14).
- **Next after that:** recommender phase 2, or phase 3 of the combine flow (AI suggestions for the combine
  screen's answers, see [plans/combine-flow.md](plans/combine-flow.md)). The order isn't decided.

## The project in one paragraph

Databoarder helps nontechnical people answer "did this campaign work?" questions. The current work is
**step 1, the data stage**: someone uploads the messy files their analyst sent, confirms a list of
preparation steps, and gets one clean, frozen table that the existing analysis screens use.

## Done

- **Step 0, test data:** a fake messy analyst pull built from the Hillstrom dataset, with a known answer.
- **Step 1, merged to `main` (PR #2):** uploading files, the review screen, preparation steps, AI-suggested
  steps that need confirming, and hiding IDs, emails and phone numbers from the AI.
- **Step 1 finish, merged to `main` (PRs #3, #4, reviewed by Codex):**
  - Exact library versions pinned, so results can be reproduced.
  - The downloadable script stops if the original files or the rebuilt table don't match what the report used.
    Tested in every mode.
  - The column the person confirmed while building the table (who got the campaign, or the date) now carries
    into the question screens instead of being guessed again. If the AI picks a different group or date column,
    the confirmed one is used and a note says why. A before/after question that sets aside the confirmed group
    is allowed, with a note saying it can't show whether the campaign made the difference.

## Next up, in order

1. **One-screen combine flow.** Mockup approved 2026-11-13: [mockups/combine-flow.html](mockups/combine-flow.html)
   (published privately at https://claude.ai/artifact/81LqwpTmSo4mA9E9W6YLAu). Building it, following [plans/combine-flow.md](plans/combine-flow.md) (three PRs: server, screen, AI).
   **Phase 1 (server) merged to `main` (PR #5, reviewed by Codex, 2026-11-14):** suggested answers, previews, blocking checks,
   the analyst note, and building from confirmed answers. 290 tests pass.
   **Phase 2 (screen) merged to `main` (PR #6, reviewed by Codex, 2026-11-14):** the new combine screen replaces what
   "Put them together first" opens; the old builder is behind "Build it step by step instead". 7 Playwright browser
   tests; 300 tests pass. **Next: phase 3 (AI suggestions).**
   Decided 2026-11-13: keep "Confirm all suggestions" (the choices stay visible above it). Matching IDs by
   digits and adding up orders stay automatic, but become an extra choice when matching looks risky (people
   would merge, or many IDs don't match). The optional question asked before upload steers the suggestions and
   carries into the question screen.
   Add browser tests (Playwright) with it, since the screens have no automated tests yet. Original idea: Replace the step-by-step recipe builder
   with: upload the files, see the finished table with sanity numbers ("10,000 customers, 50% got the email"),
   and answer only the 3–4 choices that change the answer (who counts, who got the campaign, what counts as
   the result and over what window, what to do with people who don't match or appear twice). Each answer
   is shown as a suggestion until confirmed. Mechanical fixes (trimming, stacking, reading Excel) are applied
   by fixed code and listed, collapsed, with row counts. Matching IDs across files is done by code, not the AI.
   Possibly ask the question before the upload so the table is built for it.
2. **Recognize more messy formats:** IDs Excel turned into `9.941E+14`, codes like `XX` meaning "missing",
   and `YYYYMMDD` dates. The `XX` one can silently count missing values as real data. These fit the
   "mechanical fixes" part of the new flow.
3. **Step 2:** a ready-to-forward note telling the analyst exactly what's missing from a pull.
4. **Question recommender.** Planned 2026-11-14: [plans/recommender.md](plans/recommender.md). Ranks questions the
   data can answer by how well they fit the person's goal, built as a standard two-stage recommender (candidates,
   hard feasibility filter, embedding retrieval, ranking, feedback logging, offline evaluation). Six PRs.
   **Phase 1 (cases and harness) merged to `main` (PR #8, reviewed by Codex), 2026-11-14.** Where the rest goes in this order isn't decided.
   **Don't forget (Ben, 2026-11-14): kinds of question.** Effect of a change, prevalence from a sample,
   forecasting, drivers. Not an up-front question yet; infer it from the goal and the table. Phase 2 adds a `task`
   field to candidates; phase 4 adds a `different_task` "can't answer" reason and test cases. New kinds of analysis
   only if real users ask. Details: [plans/recommender.md, "Kinds of question"](plans/recommender.md#kinds-of-question-noted-2026-11-14-for-later).

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

- **The old step-by-step builder** stays, hidden behind a "Build it step by step instead" link, for pulls
  the four choices can't describe. (Decided 2026-11-14.)

- **People on more than one row:** a table can't be confirmed until it has one row per person. Clustered error
  estimates are on hold until a real pull needs them. (Recommended 2026-11-10; confirm if you disagree.)

## Later

- Step 3: save the analyst's query with the report.
- Step 4: warehouse connections, once there's a pilot company.
- Step 5: more kinds of analysis, chosen from what users ask. Candidates so far: prevalence from a sample, then
  forecasting (see "Kinds of question" in [plans/recommender.md](plans/recommender.md)).

## Setup notes

- Run locally: `.venv/bin/python -m uvicorn app.main:app --reload`, then open http://localhost:8000. Messy test
  files are in `local_data/hillstrom/pull_full/`.
- Browser tests need Chromium once: `.venv/bin/python -m playwright install chromium`. Without it they're skipped.
- Tests: `.venv/bin/python -m pytest` (install with `.venv/bin/pip install -r requirements-dev.txt`).
- **On a new computer:** clone the repo, then `python3.12 -m venv .venv`, install `requirements-dev.txt`, and
  install Chromium as above. `local_data/` isn't in git: copy it over, or regenerate the messy pull with
  `tools/make_messy_hillstrom.py` from the public Hillstrom CSV.
