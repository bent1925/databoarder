# Plan: One-screen combine flow

Status (2026-11-13): phase 1 (server) built on `combine-flow-server`; see "What phase 1 changed from this plan"
at the end. Mockup approved:
[../mockups/combine-flow.html](../mockups/combine-flow.html). Written against `main` at `0cc2608`. Line numbers
may drift.

## The problem

Clicking through the data stage on the Hillstrom test pull showed that combining files is the bottleneck.
The screen is a recipe builder: about 12 kinds of step, each with Confirm, Edit and Remove, plus an
"Add step" menu. It asks a marketer how to combine the files, when what they can actually judge is what
the finished table means.

## What changes for the person

1. **Add all the files at once,** with an optional "What do you want to find out?" box.
2. **See the finished table** with six numbers to sanity-check (people, share emailed, no-email group,
   visited, bought, spent per person) and a peek at five rows.
3. **Answer four choices,** each pre-filled with a suggestion:
   1. Who belongs in the table
   2. Who got the campaign (and whether its versions stay separate)
   3. Which results count (the time window)
   4. What a person missing from an events file means
4. **Confirm.** One at a time, or "Confirm all suggestions". Changing a choice updates the numbers.
5. **The mechanical fixes are listed, collapsed,** with a count for each one: skipped title rows, ID
   formats matched, orders added up.

When the files can't answer the question (everyone got the campaign), the page says so and drafts a note
for the analyst instead of showing the choices.

## Decisions (2026-11-13)

- **"Confirm all suggestions" stays.** The four choices are on screen above it, so every choice is still
  visible before it's confirmed. The server still requires each answer to be marked confirmed.
- **ID matching and adding up orders stay automatic, unless something looks risky.** Then they become an
  extra choice. "Risky" is defined below.
- **The optional question steers the suggestions** and carries into the question screen.

## The main idea: answers compile into the existing recipe

The person's answers become a new, small, typed object, `TableAnswers`. Fixed code (a new
`app/combine.py`) turns those answers into a `DataRecipe`, made of the same steps that exist today. Nothing
downstream changes:

```
files ──inspect──▶ found tables ──suggest (code, or AI checked by code)──▶ TableAnswers (suggested)
                                                                          │ person confirms
                                                                          ▼
                                     TableAnswers (confirmed) ──compile (fixed code)──▶ DataRecipe
                                                                          │ existing code
                                                                          ▼
                                     build_table → checks → freeze → fingerprint → script_block
```

Why this shape:
- **Reproducibility doesn't change.** The compiled recipe is what's recorded and what the script runs.
  `app/recipe_ops.py`, `build_table()`, `script_block()` and the fingerprint checks are reused unchanged, along
  with their tests.
- **AI still only fills in a fixed schema.** It now suggests answers instead of steps. That's a smaller
  space, so suggestions are easier to check and harder to get subtly wrong.
- **Every step the table needs already exists.** For Hillstrom: `read_table`, `clean_key`, `match`,
  `flag_membership`, `fill_missing` and `count_within_window`. The window step already reports events before
  or after the window, events matching no one, and windows cut off by the end of the data
  (`app/recipe_ops.py:333`).

## `TableAnswers`

Added to `app/schemas.py`, next to `DataRecipe`. Each answer has its own `status: "suggested" |
"confirmed"` and a `by: "code" | "ai" | "person"` field, so the report can say who chose what.

| Answer | Holds | Hillstrom suggestion |
|---|---|---|
| `people` | which table is the list of people, and its ID column | `customer_export.xlsx`, `Customer ID` |
| `campaign` | how we know who got it: membership in a list, a column in the people table, or "everyone" (before/after); the versions column if any; `separate` or `any` | in `email_sends.csv`; versions from `campaign`; `separate` |
| `window` | start (each person's send date, with one date for people not sent anything) and length in days | send date, 3 Mar 2008 for the rest; 14 days |
| `missing` | `none` (absent means no visit and $0) or `ask` (stop and add to the analyst note) | `none` |
| `outcomes` | per events table: the ID and date columns, plus what to measure (any event, total of a number column) | visits: any; purchases: any, and total of `amount` |
| `match` | only present when matching is risky (below) | not present |

`outcomes` isn't one of the four choices. It's read off the events files by code and shown in the table
numbers, and the main outcome is still chosen later, in the question screen (one main outcome per report,
as today).

The compiler also reads the header row and footer rows found by `inspect_files()`. Those stay automatic and
listed, as today.

## Suggesting the answers

**Code first, the same pattern as the question translator** (`app/translator.py`): fixed code makes a
suggestion that always works without an API key. AI may improve it, and its answer is checked against the
found tables before use.

How code makes the suggestion:
1. **Find ID columns.** For each column that could be an ID, clean it with the `digits_only` rule and measure
   how much it overlaps with the other tables' candidates. The columns that link the tables are the IDs.
2. **People table:** the table whose IDs are unique per row and contain the IDs found in the other tables.
3. **Campaign table:** a table with one row per person whose IDs are a subset of the people table, usually
   with a single date or a few dates (the send). A text column with 2–5 values becomes the versions.
4. **Events tables:** the rest, where IDs repeat or dates spread over a range. Number columns become
   "total of …".
5. **Window:** from the send date to the last event date in the data, offered as whole options (7, 14,
   30 days, "all the data"), and only those the data covers fully. A window from the question ("within a
   week") is picked when it's one of those.
6. **No campaign table and no group column:** "everyone" with the send date, which leads to before/after.

**The AI's part:** it's given the same masked table descriptions as today (`proposer.describe_table()`),
the question, and the code's suggestion. It may change any answer, using names of tables and columns that
exist. Anything else is rejected and the code's suggestion is kept, with a note, as in `interpret()`. Its
answers arrive as `status: "suggested", by: "ai"`. The AI can never mark an answer confirmed.

## When matching is risky (the extra choice)

Code checks these after compiling the suggestion and running a preview. Each one adds a `match` choice with
counts and plain options ("Match anyway, leaving these out" or "Ask my analyst"):

- **Two different people would merge.** Two different raw IDs in one table clean to the same key (`A-123`
  and `B-123` both become `123`).
- **Many rows don't match.** More than 1% of a campaign or events table's rows match no one in the people
  table. Any unmatched rows are always listed in the fixes, even under 1%.
- **Negative amounts.** Returns or cancellations in a number column being totalled. The choice is whether
  they count against spend.
- **Send dates spread out.** If campaign rows have send dates more than 3 days apart, people not sent
  anything have no single obvious start date, so the window start becomes part of the choice.
- **The people table repeats IDs.** This one stops instead of asking, as today ("a table can't be confirmed
  until it has one row per person").

The 1% and 3-day thresholds are guesses until there's a real pull. They live as named constants in one place.

## When the files can't answer the question

If the campaign list covers everyone in the people table, there's no comparison group. The page shows that
instead of the choices, offers before/after with its limits stated (as `require_comparison` already
explains), and drafts a note for the analyst. The note is built by fixed code from templates, one sentence
per problem: no comparison group, choice 4 answered "ask", a risky match answered "ask". This brings a
small part of step 2 (the analyst note) forward. The full step 2 is still separate.

## What gets recorded

In the dataset's `source`, next to what's recorded today:
- the confirmed `TableAnswers`, including what was suggested and by whom (`code` or `ai`)
- the question, if one was typed
- the compiled recipe (already recorded), which is what reproduction depends on

The reproduction script gets a header listing the four decisions in plain words. It then runs the compiled
recipe exactly as it does today. Recompiling isn't needed to reproduce, so a later change to the compiler
can't change an old report.

The confirmed `campaign` answer sets `group_column` or `time_column`, which already carry into the question
screens (`_confirmed()` in `app/main.py`).

## Routes

- `POST /api/sources`: unchanged. It also returns the suggested answers.
- `POST /api/sources/{id}/answers/suggest`: takes the question, and returns suggested answers, with the AI's
  if a key is set.
- `POST /api/sources/{id}/answers/preview`: takes answers in any state, and returns the six numbers, five rows,
  fixes with counts, risks, the blocked reason if any, and the analyst note.
- `POST /api/sources/{id}/answers/confirm`: refuses unless every answer is confirmed. It compiles the answers,
  then runs the same checks and freezing as `confirm_source()` today (`app/main.py:180`). That function's
  body gets factored out so both routes share it.

The preview runs the recipe on every change. That's fast for Hillstrom (64,000 people). Large pulls are a
separate, later problem.

## What happens to the current builder

- **Kept:** every recipe step, `recipe_ops.py`, `build_table()`, `script_block()`, the `/preview` and
  `/confirm` routes and their tests.
- **Hidden:** the step-by-step builder in `static/index.html` (lines 209 to about 628). A "Build it step by
  step instead" link reaches it, for pulls the four choices can't describe (Online Retail II has no campaign
  at all).
- **Replaced:** the AI step proposer (`app/proposer.py`). It becomes an answer proposer, and
  `evals/recipe_cases.py` becomes `evals/answer_cases.py`.

**Open question:** remove the builder entirely once a real pull shows whether anyone needs it. Keeping it
costs maintenance, and it's the noisy screen this plan replaces.

## Statistical points to get right

- **Both groups need the same window.** People not sent anything start on the campaign date, so both groups
  are measured over the same days. That's what `default_start` is for. Windows the data doesn't fully cover
  aren't offered.
- **Absent doesn't automatically mean zero.** That's choice 4. It's suggested, never silent, and "ask" is a
  real option.
- **Two versions means two comparisons.** Keeping the versions separate leads to Men's vs none and Women's
  vs none. The question screen already compares two groups at a time. Whether looking at both needs a
  correction outside Research mode should be checked when this is built, not assumed.
- **Attributes measured after the campaign.** The people table's columns are kept as they are, and covariates
  are still chosen in the analysis flow (`default_covariates()`). A people table exported after the campaign
  may hold attributes affected by it, so the export date should be shown when it's found. Hillstrom's export
  says "Exported 2008-03-02", the day before the send.
- **Agreeing numbers aren't proof.** The sanity numbers help someone spot a wrong table. They don't show the
  table is right, and the page shouldn't say they do.

## Tests

Server, with pytest:
- Hillstrom suggested answers, confirmed and compiled, rebuild the answer-key table. Values must match; column
  names come from the answers.
- The code's suggestion for the Hillstrom pull picks the expected people, campaign, versions and events tables.
- Each risky-matching case triggers its choice, and a clean pull triggers none.
- Treated-only pull: blocked, with the right note, and no choices.
- A 7-day window changes the counts to the expected values.
- Confirming refuses any answer still marked suggested. An AI answer naming an unknown table or column is
  rejected, the code's suggestion is kept, and a note is shown. The AI can't send `confirmed`.
- The reproduction script, built from answers, rebuilds the table and matches the report in every mode. This
  is the existing test, fed from the new route.

Browser, with Playwright (new, `tests/browser/`):
- Upload a small generated Hillstrom pull, confirm all, use the table, and see the "You confirmed this" box on
  the question screen.
- Change choice 3 and see the numbers change.
- Treated-only pull: blocked message and note.
- Answer choice 4 with "ask": the note appears and "Use this table" stays disabled.

Playwright needs `pip install playwright` and `playwright install chromium`. The browser tests skip with a
message saying so when it's missing, so `pytest` still runs anywhere.

## Order of work

Each phase is its own PR, with a Codex review.

1. **Server:** `TableAnswers`, the compiler, the code's suggestions, risk checks, the three routes, the analyst
   note templates, and server tests. No screen changes yet.
2. **Screen:** the new combine screen from the mockup, with the old builder behind a link, plus the Playwright
   tests.
3. **AI suggestions:** the answer proposer, the question steering suggestions, eval cases. Needs a real-model
   run, which needs an API key and costs money.

The messy-format work (`9.941E+14` IDs, `XX` meaning missing, `YYYYMMDD` dates) comes after. Those become
more automatic fixes in the same list.

## What this plan doesn't settle

- Whether real pulls fit the four choices. Hillstrom was built to. A real pull is still the main missing input.
- Size limits. The preview reruns the whole recipe on each change.
- Deployment for real customer data. Still waiting on a decision (see `docs/STATUS.md`).

## What phase 1 changed from this plan

- **Suggestions come only from their own route.** `POST /api/sources` is unchanged; the screen calls
  `/answers/suggest` with the question. That keeps uploads fast and lets the question shape the suggestions.
- **No "everyone got it" answer yet.** `CampaignAnswer.kind` is `list` or `column`. When the files show that
  everyone got the campaign, the table is blocked with a note for the analyst. Building a before/after table
  from these files needs a different shape (one row per day) and is left for later.
- **Some risks block instead of becoming a choice.** IDs that differ only by letters (`A-123`, `B-123`) and people
  listed twice on the campaign list stop the table, with a note for the analyst. Merging different people isn't
  a choice that should be one click away.
- **A window longer than the data covers is refused** by the server, not only left out of the options.
- **`run_steps()` can start from tables already read,** and the window step can start every row on one date. Both
  are small additions to `app/recipe_ops.py`, and old recipes run exactly as before.
- **Speed:** a preview takes about 0.4 s on the 64,000-person Hillstrom pull. The first suggestion takes about
  6 s, mostly reading the Excel file twice (once to find the header row, once to read it). That double read
  already happens in the current builder.

