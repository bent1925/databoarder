# Plan: Science mode

Status (2026-11-26): phase 1 built on branch `science-mode`. It's a first try at whether Databoarder works for lab
scientists who use GraphPad Prism today (see "To do: from a talk with a potential customer" in
[../STATUS.md](../STATUS.md)).

## The idea

A separate page, `/science`, for small lab datasets. The pitch is "the right test, explained and reproducible", not
"a Prism replacement". The same rules as the rest of the app apply:

- The person makes every choice that changes the answer. Nothing is picked from the data behind their back: a
  normality check never switches the test, and columns guessed from the table are marked "guessed".
- Fixed, tested code does every calculation. There's no AI in science mode yet.
- Every report comes with a script that reproduces every number. The engine (`app/science/engine.py`) only imports
  numpy, pandas, scipy and statsmodels, and is copied into the script word for word. The tests run the script and
  compare its output with the report.

It's a separate page because the campaign screens are built around "did this campaign work?" (the design picker,
the AI translator, the four campaign analyses). Lab questions start from the shape of the experiment instead.

## Phase 1 (this branch): Prism's column and contingency tables

The analyses labs use most often. Each one was checked against an independent calculation in `tests/test_science.py`.

| Question | Choices the person makes | Tests |
|---|---|---|
| Do groups differ? | layout, groups, same subjects?, averages or ranks, equal spread?, which pairs, control, random assignment | Welch's or unpaired t test, Welch's or ordinary one-way ANOVA with Games-Howell, Tukey or Dunnett, paired t test, repeated-measures ANOVA (Greenhouse-Geisser always applied) with Bonferroni paired t tests, Mann-Whitney, Kruskal-Wallis with Dunn (Holm), Wilcoxon signed-rank, Friedman with Wilcoxon (Holm) |
| Are categories related? | rows or counts, groups, outcomes, which outcome is the event, random assignment | Fisher's exact test, relative risk, odds ratio, risk difference (2×2); chi-square and Cramér's V (bigger tables) |
| Do two measurements move together? | x, y, Pearson or Spearman | r with a Fisher-z interval; Spearman with the Bonett-Wright interval |

Data can be uploaded as CSV or pasted from a spreadsheet. It can be laid out one column per group (like Prism) or one
row per measurement.

Decisions made in phase 1 (open to change):

- **Welch is the suggested default** for unequal spread (marked "No is safer"), but the person still has to choose.
- **Repeated measures always use the Greenhouse-Geisser correction**, so sphericity is never assumed.
- **Subjects missing a time point are left out of repeated-measures tests**, and the report counts them and says so.
  A mixed-effects model (phase 2) can use them.
- **Dunnett's test uses a fixed seed.** scipy estimates its probabilities by quasi-Monte Carlo, so without a seed the
  numbers changed between runs. The report says this.
- **Exact P values, no asterisks.** Asterisks belong with the figures (phase 3).
- **Assumption checks are information, not gates.** Shapiro-Wilk results, the spread ratio and far-out values are
  shown with a note that a passing check isn't proof.

## Later phases (not started; each needs its own go-ahead)

Ordered by how often labs use them. The first step is still to ask the potential customer which Prism analyses they
actually run each month, and to reorder this list to match.

2. **Grouped tables:** two-way ANOVA, and mixed-effects models for repeated measures with missing values. This needs
   a careful choice of sums of squares (Type II or III) and of follow-up comparisons. statsmodels covers it.
3. **Figures for papers:** the same charts drawn by fixed code from the frozen data and confirmed choices, exported
   as SVG or PDF with the script that redraws them. Optional asterisks for adjusted P values.
4. **Curve fitting:** dose-response (4-parameter logistic, EC50/IC50 with intervals), Michaelis-Menten, exponential
   decay. Starting values and constraints must be visible choices, and a failed fit must be reported, never hidden.
5. **Survival:** Kaplan-Meier curves, log-rank test, Cox regression. This adds the `lifelines` library, or uses
   statsmodels' `SurvfuncRight` and `PHReg`.
6. **Regression:** linear, logistic, Poisson, with diagnostics shown as information.
7. **Diagnostics and agreement:** ROC curves with AUC intervals, Bland-Altman plots.
8. **Many variables:** PCA and clustering. These are exploratory, so the wording has to say "patterns to check",
   never findings.
9. **Planning:** sample size and power for the tests above.

Not planned: Prism's 100+ curve equations, user-written equations (that would be code written by the person, which
breaks the fixed-code rule), and integrations (Prism Cloud, LabArchives).

## Open questions

- Should the main page offer science mode as an equal choice, or keep it as a link until a real lab has tried it?
  For now it's a link.
- Prism users often analyse several outcomes from one experiment. Should there be a project-level correction for
  multiple outcomes, or a warning? For now, every report says P values aren't adjusted across analyses.
