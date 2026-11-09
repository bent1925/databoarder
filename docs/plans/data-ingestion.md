# Plan: Seamless data ingestion

Status: not started. Written 2026-11-09 against `main` at `23df680`.
Line numbers below are from that commit and may drift.

## The problem

Databoarder's users are marketers and other nontechnical people. They often know what they want to
learn and roughly where the data lives, but not how to get it into a form that can be analyzed.

The typical path is that a marketer asks an analyst to pull everything related to a campaign. That
pull comes back in one of two ways:
- a pile of exports (several CSV or Excel files, or one wide table), or
- a query that gets everything, which should plug into the tool.

Either way, ingestion should be seamless. Today the app accepts one CSV of up to 25 MB that must
already have one row per person, order or day.

## Decisions

- **Data preparation belongs in this tool, behind a hard boundary.** Who counts, what counts as
  treated and over what time window are analytical choices. If they happen elsewhere they go
  unrecorded. The tool is not becoming a general data-cleaning product.
- **Standardize the output, not the input.** Sources vary without limit. What every causal question
  needs does not: a unit of observation, a treatment indicator, one or more outcomes, and covariates.
- **AI proposes, fixed code runs.** AI fills in a recipe from a fixed set of steps. It never writes
  code or SQL that runs. The user confirms the recipe.
- **Seamless means no reformatting and no technical work. It does not mean no confirmation.**
  Unambiguous tidying happens automatically and is listed, as `app/prep.py` does today. The user
  answers only the questions that change the result.
- **"Any format" means any table.** CSV and Excel are in scope. PDFs and screenshots are not, because
  nobody can check the transcription.
- **Dropped from the earlier plan:** a screen where users write SQL to join uploaded files. It only
  helps someone who already writes SQL. DuckDB is no longer a planned dependency. Reconsider it only
  as an internal engine if real pulls are too large for pandas.

## Order of work

0. Collect real pulls
1. Data stage: many files in, one confirmed frozen table out
2. Analyst data request and "what's missing" message
3. Record the analyst's query
4. Run queries directly against a warehouse
5. More analyses, chosen from real usage (cutoff designs are the known gap)

**This order flips** if someone already has a clean file and a question the app can't answer: build
that analysis first.

## Rules from AGENTS.md that shape this work

- **AI can't perform analysis or silently alter choices.** Recipes are data, checked against a fixed
  schema. Suggestions stay visibly distinct from approved decisions.
- **Results must reproduce from a recorded dataset version.** Every ingestion is frozen and
  fingerprinted, and a fresh pull is a new version.
- **Reproduction scripts must include data preparation.** The script rebuilds the table from the
  original files, including joins and roll-ups.
- **Repeated observations, missing data and multiple comparisons must be handled.** Each has an
  explicit question or rule below.
- **Consequential choices must be explained, and errors must say how to proceed.**

## Step 0: Collect real pulls

Before building the recipe steps, get two or three real campaign pulls from an analyst, anonymized if
needed. Note the file types, sizes, shapes, ID columns and whether non-treated people are included.
These pulls decide which recipe steps come first, set the size limit, and become test fixtures.

## Step 1: The data stage

### Contract between the stages

The data stage produces one frozen analysis table and a record of how it was built. The analysis
stage sees only that table and record. A clean CSV can still go straight through the existing upload.

### What the user does

1. **Add data.** One drop zone takes any number of CSV and Excel files.
2. **See what was found.** Each file is described plainly: rows, columns, what it appears to be, and
   the automatic tidying applied.
3. **Answer a few questions,** asked with their own data (see below). AI-suggested answers are marked
   as suggestions, and every one can be changed.
4. **Check the result.** A preview of the final table, plus a count of rows and people after each
   step, so losses from matching or filtering are visible.
5. **Confirm.** The table is frozen and the existing flow continues.

### The questions

Ask in concrete terms: "Customer 1043 appears on 3 rows. Is each one a separate purchase?" works
better than "What's the unit of observation?" Only ask what the data can't settle.

- **Unit:** what one row of the final table represents.
- **Treatment:** who got it and when. Assigned and actually exposed are different groups, and they
  support different claims.
- **Comparison group:** a campaign list often holds only treated people. If there is no comparison
  group and no before-period, stop and explain why, then offer the message from step 2. A before-period
  alone can route to the before/after method, with its limits stated in the report.
- **How treatment was decided:** this already exists in `app/design.py`.
- **Main outcome:** chosen before any results are shown. v1 has one main outcome per report, and
  further outcomes wait until they can be corrected for multiple comparisons.
- **Outcome window:** "converted" needs a time limit, such as within 30 days of treatment.
- **Missing rows:** someone absent from the purchases file probably means zero purchases, but could
  be a failed match. This changes results, so it has no silent default.
- **Failed matches:** how many people matched across files and how many didn't. If the share is
  large, the user must acknowledge it.
- **Covariates:** only attributes measured before treatment. Columns that may be measured afterward
  are flagged, and a date alone doesn't prove which is which, so the user decides.
- **Repeated rows:** roll up to one row per unit with a chosen rule, such as sum or any-conversion.
  People who appear in both groups need an explicit decision. If rows still repeat, the data stage
  doesn't continue until step "Repeated rows" below adds a valid alternative.

Checks passing here must not be presented as proof the data is right. The report states what the
user confirmed, and what was assumed.

### The recipe

Add a `DataRecipe` to `app/schemas.py`, built like `AnySpec`: a typed list of fixed operations.

Starting set, to be trimmed or extended from step 0:
- read a table (file, Excel sheet, header row, footer rows to skip)
- keep or rename columns
- reshape wide to long
- match tables on a key, with key normalization such as trimming and lowercasing
- filter rows on a stated condition
- roll up to a unit, with a stated rule
- build a treatment indicator from a column or from membership in a list
- count an outcome within a time window
- fill missing values after a match, with a stated value

Each operation produces both plain English and pandas code, extending the `Step` pattern in
`app/prep.py:19-21`. Each carries a status of proposed or confirmed, and code refuses to run a recipe
with any proposed step.

### AI's role

AI reads column profiles and proposes a recipe and answers. Code validates the proposal against the
schema and the actual columns. Everything must also be possible by hand, so an AI failure never blocks
the user.

Add recipe proposals to the evals alongside `evals/translator_cases.py`.

### Privacy

`profile_columns` (`app/translator.py:57`) sends real example values to the model, unmasked. On an
analyst pull those could be emails or customer IDs.
- Mask identifier-like values before anything reaches the model. The browser can still show them.
- The README (line 16) and the privacy note in `static/index.html:189` describe what the AI sees, and
  say prepared questions send nothing. Proposing a recipe does send data, so update both deliberately
  and say so before it happens.
- Both places also warn against sensitive data (`README.md:93`, `static/index.html:190`). That warning
  conflicts with real customer pulls, so real data needs the deployment decision under Open questions.

### Size and storage

Today `MAX_BYTES` is 25 MB, up to 40 datasets are held in memory, and the host is one process on
Render's starter plan (`app/main.py:28-32`, `render.yaml`).
- Set the new limit from step 0.
- Keep uploaded files and snapshots on disk in a per-session temporary folder, with a total size cap
  and an expiry. Dropped sessions get the existing "no longer loaded" style of message.
- Update the privacy note at `static/index.html:187`, which says files are kept in memory.
- Measure whether pandas fits in the instance's memory on the real pulls.

### Where this touches the code

- `app/main.py:54-80`, `upload()`: move everything after `await file.read()` into a shared ingest
  function taking raw bytes, a name and optional source details. The frozen table goes through it as
  CSV bytes, so `prepare()`, the sha256 fingerprint and the `read_csv` line keep working.
- New `app/ingest.py`: file reading, operations, recipe execution, step counts.
- New routes: start a session with files, propose a recipe, preview a recipe, confirm it, and download
  the frozen table.
- `app/analysis.py:24-38`: record the source in `repro` and add the rebuild block to the script. The
  regex at line 37 depends on the form of the `df = pd.read_csv(...)` line, so leave that line alone.
- `app/script.py:29,142,253`: these stay as they are if the rebuild block writes the snapshot file
  that `read_csv` then opens.
- `static/index.html`: the new screens and a "where this data came from" report section.
- `requirements.txt`: add `openpyxl` for Excel.

### What gets recorded

In the report's `repro` dict and the script header: the sha256 of each input file, the confirmed
recipe, each answer the user gave, the sha256 of the frozen table, and library versions.

### Tests

- The frozen table and a plain upload of the same bytes give identical analysis results.
- The generated script, run against the original files, rebuilds a table with the recorded fingerprint
  and matches the report in every mode.
- A messy sample with a planted truth, following `tools/make_*_sample.py`, recovers the known effect.
- A recipe with any unconfirmed step refuses to run.
- A proposal naming unknown columns or operations is rejected.
- A treated-only pull stops with the right explanation.
- Identifier values never reach the model.
- Unreadable files and failed steps return errors the user can act on.
- The full `python -m pytest` suite still passes.

## Repeated rows (part of step 1, statistics may follow)

Today `app/checks.py:111-126` warns that repeat people "can overstate certainty" and carries on, and
only when it can find an ID column. `app/prep.py:75` flags exact duplicates without changing them.

- The roll-up in the recipe is the first answer. It is the easiest to explain.
- Consider clustered error estimates where a roll-up would throw away real information.
- Start with A/B and adjusted comparison. Before/after works in whole weeks, so check what repeated
  rows do to it and to difference-in-differences before changing either.
- Add simulations like the existing ones: 95% ranges should still cover the truth about 95% of the
  time when rows repeat.
- Direct CSV uploads keep today's warning until this is done. Decide then whether they should also be
  asked the unit question.

## Step 2: Analyst data request and "what's missing"

The marketer describes the question and how treatment was decided. The tool produces a note to
forward to the analyst. Code fills the note from a fixed template for each design in `app/design.py`:
- everyone eligible, including people who weren't treated
- a stable ID that is the same across files
- treatment assignment and date
- outcomes, with dates, far enough past the campaign
- attributes measured before the campaign

If a pull fails the checks in step 1, the tool lists what's missing in the same forwardable form. The
list is computed by code from the confirmed answers. This needs no integrations.

## Step 3: Record the analyst's query

Until step 4, the analyst runs the query and uploads the result. Add an optional field to paste the
query. It is saved with the recipe and shown in the report, labeled as supplied by the analyst and
not run by this tool, because the tool can't verify that the query produced the file.

The query text stays out of what the AI sees. In the script it is written as comments, with every
line prefixed, so query text can't become code.

## Step 4: Run queries against a warehouse

Wait for a pilot company, and build for the warehouse and security rules they actually have.

Add it as another source feeding the same data stage, so nothing downstream changes.
- A connector is the smaller part of the work. Enterprise readiness is the larger part: security
  review, single sign-on and network access.
- Queries are written by an analyst, not generated by AI. A data person approves a small set of
  sources once, and marketers choose from those.
- Use a read-only role, statement timeouts and result size limits. A result over the limit is an
  error, never silently cut off.
- Decide where credentials live. Don't put them in the dataset store or generated scripts.
- Every run creates a new frozen snapshot with its own fingerprint and timestamp. The saved query can
  be rerun for fresh data, but that is a new dataset version.
- Running inside the customer's environment using the existing `Dockerfile` may be easier than asking
  them to connect to a hosted app. It would also keep their data in place.

## Step 5: More analyses

Choose from what real users bring. Cutoff-style designs are the documented gap in `README.md:73`.
Each new method needs a spec in `app/schemas.py`, dispatch in `app/analysis.py`, a design option,
knowledge entries in `app/knowledge.py`, a reproduction script, and a sample dataset with a planted
truth.

## Open questions

- **First pilot:** who is it, and which warehouse do they use?
- **Deployment for real data:** one shared password on a hosted app isn't suitable for customer data.
  Options are a separate deployment per company, or running inside the customer's environment.
- **Size:** how big is a real "everything" pull?
- **Flow after the data stage:** how much of the existing question and design flow can be pre-filled
  from confirmed answers, while keeping suggestions distinct from decisions?
