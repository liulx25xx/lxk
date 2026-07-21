# Local Result Audit

Generated without API calls from the JSON artifacts currently in this repository.

## Stable local facts

- Failure trajectories: 143.
- Type counts: {'EDIT': 28, 'LOC': 37, 'LOGIC': 70, 'PLAN': 8}.
- Mean/median post-error tail ratio: 0.818/0.875.
- Phase-4 routing scores:
  - control: 1.7604 (n=96)
  - fixed: 2.0417 (n=96)
  - oracle: 2.3333 (n=96)
- Paired percentile-bootstrap intervals (10,000 resamples):
  - EDIT: delta=1.0000, 95% CI=[0.7143, 1.3214] (n=28)
  - PLAN: delta=0.8750, 95% CI=[0.5000, 1.2500] (n=8)
  - LOC: delta=0.2667, 95% CI=[-0.1000, 0.6333] (n=30)
  - LOGIC: delta=0.1667, 95% CI=[-0.0667, 0.4333] (n=30)

## Inconsistencies that must not be presented as settled results

- The saved classifier summary reports 74.8% with 15 features, whereas the manuscript says 11 features.
- The manuscript confusion matrix yields 102/143 = 71.3%, not 74.8%.
- Policy advantage captured is 74.5% relative to control and 48.1% relative to the universal strategy.
- The repository cannot reproduce the reported classifier because its training script and predictions are missing.

## Protocol flags

- scaffolding scripts use the first four recorded turns rather than the annotated first-error prefix.
- scaffolding prompts include the gold failure type.
- scaffolding prompts include gold target files.
- the 0--3 score is a proxy next-action metric, not SWE-bench resolution.
