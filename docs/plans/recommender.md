# Plan: Question recommender

Status (2026-11-14): planned, not started. Written against `main` at `ff43144`. Line numbers may drift.

## The problem

Someone arrives with a table and a loose goal ("did the spring email do anything?", "is loyalty worth it?").
The question screen currently helps in two ways:

- `design_options()` (`app/translator.py:363`) says which designs the table can support and guesses the likeliest one.
- `suggest_questions()` (`app/translator.py:536`) lists up to four ready-made questions for that design, built from
  column names. **It ignores the person's goal**, and its order comes from fixed rules (a preferred outcome first,
  then names that look like outcomes).

The recommender replaces that list with questions **ranked by how well they fit the person's goal**, all of which the
data can answer. When the goal can't be answered, it says why and offers the closest question that can.

## Why build it as a standard recommender

The menu is small enough that rules could get most of the way there. This plan builds it as a standard two-stage
recommender anyway, on purpose: learning that architecture is a goal of this work. The architecture fits:
candidates come from the data, a hard filter removes what the data can't support, and a ranker orders the rest. Every
stage gets measured against a fixed set of labelled cases, so each change has to show it helps.

## What changes for the person

1. On the question screen, the "What do you want to find out?" box comes first. It's pre-filled with the optional
   question from the combine screen, if one was given.
2. Below it, the top 3–5 questions for that goal, each with a one-line reason ("uses `spend`, which matches
   'worth it'; compares members with non-members").
3. Picking one fills in the form exactly as picking a suggestion does today. Nothing runs until the person confirms
   the read-back, as now.
4. If the goal can't be answered (for example, everyone got the email, so there's no one to compare with), a
   message says so in plain words, offers the closest answerable question, and lists what to ask the analyst for.

## Rules the recommender must keep

These follow the review rules in `AGENTS.md`.

- **Recommendations are suggestions.** They stay marked as suggested until the person confirms the read-back. The
  recommender never runs anything and never changes a confirmed choice.
- **The feasibility filter is hard.** A question that fails it is never shown, whatever the ranker scores it.
  The rules live in code and have tests. They aren't learned.
- **The design is the person's answer.** The ranker never prefers a design because it gives a stronger claim.
  When the person has picked a design, or confirmed one while building the table, only that design's questions are
  candidates. Before they've picked, questions from each design the data can support may appear, labelled with the
  design they assume, and picking one still asks "how did people get it?" as today.
- **No columns measured after the change as background.** The existing `default_covariates()` and `POST_HINTS`
  rules still apply.
- **Privacy.** The goal text and column names go to a local embedding model, not an outside service. Anything sent
  to the AI uses `masked_profile()` (`app/privacy.py`), as the translator does now.

## How it works

```
goal text ─┐
table ─────┼─▶ 1. generate candidates ─▶ 2. feasibility filter ─▶ 3. retrieve top N ─▶ 4. rank ─▶ top k + reasons
confirmed ─┘   (all spec combinations)   (hard rules, tested)      (embeddings)         (scorer)   │
                                                                                                  ▼
                                                                         5. log what was shown and what was picked
```

Terms used below:
- **Candidate:** one fully specified question: a design, plus the spec fields it needs (outcome, group column,
  levels, date column), plus its plain-English text. Specs are the existing `ABTestSpec`, `BeforeAfterSpec`,
  `DiDSpec` and `ObservationalSpec` from `app/schemas.py`.
- **Retrieval:** a cheap first pass that narrows many candidates to a few dozen.
- **Ranking:** a more careful scorer that orders those few dozen.
- **Implicit feedback:** what people do (pick, edit, run, leave), as opposed to ratings they give.

### 1. Candidate generation

A new `app/recommend/candidates.py`. Today's `_ab_suggestions()` and `_ba_suggestions()` stop at `limit` outcomes and
one group column. The generator lists every combination instead:

- every outcome column that isn't an ID (`is_idlike`), for every design the data supports;
- every column that could be the group (not only `_group_choice`'s pick), with each pair of levels;
- every date column for the date designs.

Hillstrom has about 8 outcomes × 3 group pairs × 2 designs, so tens of candidates, not thousands. Retrieval matters
less here than in a real catalogue, but it's built the standard way so the stages stay separate and measurable.

### 2. Feasibility filter

A new `app/recommend/feasible.py`. It reuses what exists:
- `design_options()` availability (`ok`), and the confirmed group and date;
- `validate_spec()` (`app/analysis.py`) on each candidate;
- observational candidates need at least one background column (`default_covariates`), as `_ab_suggestions` requires
  now;
- a minimum group size, so no candidate has, say, 4 people in one group.

Each rejected candidate keeps its reason. Those reasons feed the "can't answer" message (phase 4).

### 3. Retrieval by embeddings

Embed the goal text, and each candidate's text plus the humanized names of its columns, with a small local
sentence-embedding model. Keep the top N (start with 30) by cosine similarity. The model and its version are pinned
in `requirements.txt` like every other library.

Library choice (decide in phase 2): `sentence-transformers` is the standard and most widely recognized, but pulls in
PyTorch (large install). `fastembed` runs the same kind of model on ONNX with a much smaller install. Lean:
`fastembed`, unless the install size turns out not to matter.

With an empty goal, retrieval is skipped and every feasible candidate goes to the ranker.

### 4. Ranking

Features for each (goal, candidate) pair:
- embedding similarity of goal and candidate;
- whether a word in the goal matches the outcome, group or level names (after `humanize`);
- outcome type (yes/no, amount, count) vs. words in the goal ("rate", "how much", "more people");
- whether the candidate uses the confirmed group or date column;
- whether a number in the goal ("90 days") matches a column name (`spend_next_90d`);
- group sizes and the smallest difference the data could detect (`mde`, already computed in `app/ab_test.py`);
- for "they chose": how many background columns there are to compare people on.

Not a feature: anything that rewards one design over another for being more convincing (see the rules above).

Three rankers, built in order and compared on the same test cases:
1. **Baseline A:** today's `suggest_questions()` order, so there's a "before" number.
2. **Baseline B:** a hand-weighted sum of the features.
3. **Learned ranker:** LightGBM's LambdaRank, trained on the labelled cases below. It's the standard
   learning-to-rank method. It trains offline. The trained model file is committed with a version number, and the
   same input always gives the same order.

Optional fourth (phase 6): the AI re-ranks the top 10 and writes the one-line reasons. Compare quality, cost and
speed against the learned ranker before deciding whether to keep it.

### 5. Feedback logging

A new `recommendations` log in a local SQLite file:
- **impression:** request ID, time, dataset fingerprint, ranker name and version, the goal text, and the candidates
  shown with their positions and scores;
- **pick:** which candidate, at what position;
- **edit:** which spec fields the person changed before running;
- **run / leave:** whether the analysis was run.

Logging position matters because people pick the top item partly because it's on top (position bias). A later phase
can correct for it. The goal text is the person's own data and stays on this server, like the uploaded files.

## Labelled cases and evaluation

### The cases

A new `evals/recommender_cases/`, one file per dataset:
- **Datasets:** the five in `sample_data/` plus Hillstrom (`local_data/hillstrom/`).
- **Goals:** about 10 hand-written goals per dataset, from specific ("did the new checkout raise purchases?") to
  vague ("was the redesign worth it?"), plus 2–3 goals each dataset can't answer.
- **Paraphrases:** the AI rewrites each goal 3–5 ways (vaguer, marketing jargon, typos), and someone reviews them.
  That gives roughly 300–400 cases.
- **Labels:** each case lists the candidates that answer it well (grade 2), acceptably (grade 1), or not at all
  (grade 0, the default). Unanswerable cases list the expected reason instead. Claude drafts the labels, Ben reviews
  them.
- **Column renaming:** each dataset also appears with its columns renamed (`purchased` → `converted`, `spend` →
  `revenue_usd`), so the ranker can't just memorize column names.

**Train/test split by dataset, not by case.** All paraphrases of one goal, and all goals on one dataset, stay on
the same side. Otherwise the test score measures memory, not whether it generalizes to a new table. With 6 datasets,
use leave-one-dataset-out: train on 5, test on 1, six times.

### Metrics

- **Candidate recall:** the share of cases whose best (grade 2) question survives generation and filtering.
  Retrieval and ranking can't do better than this.
- **Recall@3:** the share of cases with a grade-2 question in the top 3. This is the headline number, since the
  screen shows about 3.
- **NDCG@5:** rewards putting the best questions highest, using the grades.
- **MRR:** how far down the first good question appears, on average.
- **Unanswerable cases:** the share where the "can't answer" message appears with the right reason.
- **Must be 100% (a test, not a metric):** nothing infeasible is ever shown.

`python -m evals.recommender` prints a table, one row per ranker, one column per metric, and saves it to
`evals/results/` with the date and git commit.

### Measured once there are users

From the log: how often the top question is picked, how often a top-3 question is picked, how often the pick is edited
before running, and how often it's run. These start at zero real traffic. The point is that the measurement exists.

## Phases (one PR each)

1. **Labelled cases and harness.** Cases, labels, metrics, and Baseline A's score. No change to the app.
2. **Candidates, filter, embeddings, Baseline B.** `app/recommend/`, a `/api/recommend` endpoint, scores for
   Baseline B. The screen isn't changed yet.
3. **The screen.** The goal box and ranked list replace the suggestion list. Playwright tests. `suggest_questions()`
   stays as the fallback if the recommender errors.
4. **"Can't answer" and redirects.** Uses the filter's rejection reasons. Links into the analyst note (STATUS.md,
   step 2) when that exists.
5. **Logging and learned ranker.** The SQLite log, LambdaRank trained on the cases, and the comparison table. It
   becomes the default only if it beats Baseline B on the held-out datasets.
6. **Optional AI re-ranking.** Compare against the learned ranker on quality, cost and speed.

Later, once there's traffic: retrain from logged picks, correct for position bias, and occasionally swap positions
2 and 3 to learn what people would have picked (exploration). Any randomness is seeded and logged.

## Open questions

- **Embedding library:** `fastembed` or `sentence-transformers` (see section 3).
- **Labelling time:** reviewing about 70 hand-written goals and their labels takes a few hours of Ben's time. The
  paraphrases need only a skim.
- **Questions about subgroups** ("…and for whom?") are left out of the candidates for now. Research mode already
  scans subgroups after the main question runs.
