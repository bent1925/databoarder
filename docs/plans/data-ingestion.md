# Plan: Seamless data ingestion

Status: step 1 in progress. File reading, the recipe and the frozen table exist in `app/ingest.py`, with
no routes or screens yet. Written 2026-11-09 against `main` at `23df680`.
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

0. Test data: public stand-ins now, real pulls when available
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

## Step 0: Test data

No real analyst pull is available yet, so two public datasets stand in for one. They do different
jobs, and neither replaces a real pull.

| | Hillstrom email test | UCI Online Retail II |
|---|---|---|
| **Role** | A fake analyst pull we build ourselves | One real retailer's transaction log |
| **Question it answers** | Does the data stage rebuild the correct table? | Does it cope with mess nobody planted? |
| **Known right answer?** | Yes | No |
| **Campaign or treatment?** | Yes, randomized | No |
| **Messiness** | Planted by us | Comes with the data |
| **License** | Not confirmed, so nothing derived from it is committed | CC BY 4.0, with credit |

### Hillstrom: a fake analyst pull with an answer key

**What it is.** The public Hillstrom file is one clean table of 64,000 customers, randomly assigned to
a men's email, a women's email or no email. `tools/make_messy_hillstrom.py` deliberately breaks it into
the kind of messy files an analyst might hand over. It is a simulation of an analyst's pull. No analyst
produced it.

**Real versus invented.** Customer attributes, assigned email, visits, purchases and each customer's
total spend are real and unchanged. Customer IDs, ID formats, file layout, all dates, order numbers
and how spend divides across orders are invented by the script.

**What "the right answer" means.** There are two levels, and they shouldn't be confused:
- The original file, plus an ID, is the correct **table**. The data stage must rebuild it exactly from
  the messy pieces. This is the pass/fail test.
- Comparisons computed from that table are the expected **numbers**. Emails were randomly assigned, so
  these are valid estimates with uncertainty. They are not the exact true effect of the email. If the
  table is right, the numbers follow.

**Planted problems.** Excel title and total rows; the same customer written four ways across files; an
email list that holds only emailed people; non-buyers and non-visitors absent from their files; and one
row per order. A second version holds only emailed people, and must stop with the "no comparison
group" explanation.

**What passing does not prove.** We planted these problems, so passing shows the tool handles the ones
we thought of. It says nothing about how large or strange a real pull is.

To build it: download the file named in the script, then run the command in its docstring. Output goes
to `local_data/`, which git ignores. The `answer_key/` folder must never be given to the data stage.

### Online Retail II: real retailer data

[UCI dataset 502](https://archive.ics.uci.edu/dataset/502/online+retail+ii), licensed CC BY 4.0. Credit:
Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D.

Download `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip` (zip sha256
`572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`). It holds one Excel workbook with
1,067,371 rows covering December 2009 to December 2011. The workbook is too large for the repo, so
keep it in `local_data/`. Small excerpts can be committed as test fixtures with the credit above.

**What it is for.** Real mess from a real business. It has no campaign, no treatment and no answer
key, so it can't test comparison-group questions. Results from it must not be presented as a validation
or as evidence of any effect. A simulated campaign with a known effect could be added later, labeled
as simulated.

**What profiling found:**
- **Two year sheets that overlap.** 1 to 9 December 2010 appears in both, about 22,500 rows. Stacking
  the sheets double-counts them. Removing every exact duplicate isn't the fix either: the stacked data
  has 34,335 exact repeats, more than the overlap explains, and some may be genuine repeat lines. The
  overlap should be resolved by date range, and the user should decide about the rest.
- **The unit is a product on an invoice,** with a median of 53 rows per customer.
- **Cancellations and returns.** 19,494 rows are on invoices starting with `C`, usually with negative
  quantity. Another 3,457 rows have negative quantity without being on a cancellation invoice. Whether
  these count against spend changes the outcome, so it needs a question.
- **No customer ID on 22.8% of rows.** Those rows can't be tied to a person.
- **IDs read as numbers,** so a customer appears as `13085.0`.
- **Lines that aren't products:** stock codes such as `POST`, `DOT`, `BANK CHARGES`, `ADJUST` and
  `AMAZONFEE`. A spend total has to decide whether they count.
- **Odd prices:** 6,207 rows have a price of zero or less.
- **Size:** 45.6 MB as Excel, about 95 MB as CSV and 332 MB in pandas.

**How the current app handles it, as a baseline to beat:** the workbook is rejected as over 25 MB. A
small Excel file fails with "Couldn't read that as a CSV file." A 20 MB CSV slice is accepted as 224,951
product-line rows, and before/after is offered without the unit being settled.

### Still needed from a real pull

Get one when possible, anonymized if needed. Only a real pull can settle the size limit, what analysts
actually send, and whether non-treated people are usually included.

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
- stack tables with the same columns, such as sheets split by year
- reshape wide to long
- calculate one column from two others, such as quantity times price
- match tables on a key, with key normalization such as trimming and lowercasing
- filter rows on a stated condition
- roll up to a unit, with a stated rule
- build a treatment indicator from a column or from membership in a list
- count an outcome within a time window
- fill missing values after a match, with a stated value

The time window opens at each row's start date and runs through the stated number of calendar days,
counting the start day. Its length and the value for rows with no events have no defaults, because
both change results. Rows without a start date stay empty unless a default start is stated, which is
how a comparison group that received nothing gets a window. The step reports events outside the
window, and rows whose window runs past the end of the data, since those may be undercounted.
Whether a date like 03/04/2010 has the day or month first must be stated, never guessed.

Each operation has a plain-English description. The operations themselves live in the standalone
`app/recipe_ops.py`, which is copied word for word into the reproduction script, so the app and the
script run the same code. Each step carries a status of proposed or confirmed, and code refuses to run
a recipe with any proposed step.

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

### Other formats to recognize

Dates written as `YYYYMMDD` numbers (the app currently says "Needs a date column"), IDs ruined by
scientific notation such as `9.941E+14`, and codes such as `XX` that stand for missing values.

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
