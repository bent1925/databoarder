# Databoarder

Databoarder helps nontechnical people formulate questions and
explicitly choose analyses. AI assists with clarification and
planning; fixed code performs the analysis.

## Code Review Rules

### User control
- Flag changes that let AI-generated code perform analysis or
  silently alter analytical choices.
- Meaningful choices must be visible and confirmed before execution.
  Suggestions are allowed; they must remain distinguishable from
  approved decisions.

### Reproducibility
- Results must be reproducible from the recorded dataset version,
  approved specification, and analysis engine version.
- Flag unrecorded filtering, transformations, exclusions, defaults,
  randomness, or fresh data pulls that can change results.
- Reproduction scripts must include data preparation and reproduce
  the reported calculations.

### Statistical validity
- Flag inappropriate methods, invalid uncertainty estimates,
  misleading comparisons, and unsupported causal claims.
- Passing diagnostics or agreeing cross-checks must not be presented
  as proof that assumptions hold.
- Missing data, repeated observations, and multiple comparisons
  must be handled appropriately when relevant.

### Nontechnical usability
- Flag consequential choices whose meaning or effect is unexplained.
- Errors should explain what happened and how the user can proceed.
- Reports must distinguish observed differences, uncertainty,
  assumptions, and limitations.
