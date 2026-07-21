# GPU / Server Experiment Queue

This list separates experiments that cannot be completed from the stored local
JSON artifacts. Items are ordered by how much they would strengthen a NeurIPS or
StRICt submission.

## Completed locally on 2026-07-15

The stored artifacts were sufficient for leave-one-project-out scaffold
selection, proxy-component ablations, blinded-judge sensitivity, and paired
cross-category transfer with project-clustered bootstrap intervals.  Results,
code, and processed data are in `LOCAL_EXPERIMENTS.md`,
`scripts/local_offline_experiments.py`, and
`results/local_offline_experiments.json`.  These analyses are removed from the
server queue; the experiments below require new model calls, raw trajectory
prefixes, repository containers, or human annotation.

## P0: submission-critical

### 1. Remove privileged diagnostic context

- **Question:** Do category-matched scaffolds still help when the prompt does not
  disclose the gold failure type or gold target files?
- **Protocol:** Detect a first observable error from the trajectory; provide only
  the issue text and prefix available at that point. Compare control,
  `reread_file`, `step_back`, and an abstain condition on a held-out split.
- **Minimum scale:** 96 instances x 4 conditions x 3 representative models =
  1,152 model calls, with at least three repeated generations if temperature is
  nonzero.
- **Resource:** API/server inference; no local GPU is required if using hosted
  models.
- **Deliverables:** versioned prompts, raw responses, per-instance scores, token
  counts, latency/cost, and paired confidence intervals.

### 2. End-to-end repository repair

- **Question:** Does a redirect improve actual SWE-bench resolve rate rather than
  only the next-action proxy?
- **Protocol:** Resume each agent from the intervention point for a fixed step and
  token budget. Run repository tests in isolated SWE-bench containers. Compare
  no intervention, matched redirect, fixed redirect, and confidence-gated
  abstention.
- **Minimum scale:** At least 50--100 held-out instances, two agent harnesses, and
  three runs per stochastic condition.
- **Resource:** Multi-core server, Docker storage, and substantial API budget; a
  GPU is needed only for locally hosted models.
- **Primary metrics:** resolved rate, tests passed, regression rate, tokens,
  wall-clock time, and intervention harm rate.

### 3. Reproducible online structure detector

- **Question:** Can failure structure be inferred using only observations
  available before intervention?
- **Protocol:** Rebuild features from prefix-only traces; exclude gold-file
  overlap, patch status, full-trajectory counts, and post-intervention signals.
  Use repository- or project-grouped cross-validation and save every held-out
  prediction.
- **Baselines:** majority class, rule-only detector, logistic regression, random
  forest, and a small sequence/LLM classifier.
- **Resource:** CPU for tabular baselines; GPU/API only for sequence or embedding
  models.
- **Deliverables:** training script, feature schema, split manifest, confusion
  matrices, calibration curves, abstention curves, and policy value on held-out
  examples.

## P1: strong reviewer-facing additions

### 4. Directly measure trajectory structure

- Compute retry-cycle length, repeated-action rate, action-type entropy,
  edit-target entropy, test/edit alternation, and semantic drift across turns.
- Test whether these continuous features predict scaffold effect beyond the four
  retrospective labels.
- **Resource:** CPU for discrete features; one GPU or embedding API for semantic
  similarity.

### 5. Human validation of categories and proxy scores

- Use two independent annotators on a stratified sample, with a written rubric
  and blinded condition labels.
- Report category agreement, score agreement, and disagreements between humans,
  regex scoring, and LLM judges.
- **Resource:** annotation time or a labeling platform; no GPU required.

### 6. Joint held-out routing and scaffold selection

- The completed local analysis holds out projects when selecting among stored
  scaffold responses.  A stricter deployment test must also learn the
  prefix-only category/router on development projects, freeze both the router
  and scaffold policy, and generate fresh responses once on untouched projects.
- Use nested/grouped validation so neither category detection nor scaffold
  selection sees test outcomes.
- **Resource:** primarily API inference for fresh responses; CPU for analysis.

## P2: breadth and generalization

### 7. Harness, language, and benchmark transfer

- Repeat the study on at least one additional code-agent harness and one
  non-Python benchmark or repository collection.
- Preserve the same observation budget and evaluation contract across settings.
- **Resource:** server/container infrastructure and API or local-model inference.

### 8. Intervention timing and dose

- Compare first observable error, repeated-error threshold, mid-trajectory, and
  confidence-triggered intervention. Test one-sentence versus evidence-rich
  redirects under equal token budgets.
- **Resource:** API/server inference; container execution for end-to-end outcomes.

## Recommended execution order

Run P0.1 and P0.3 first because they remove the largest leakage risk. Then run a
small P0.2 pilot (10--20 instances) to validate the container and continuation
pipeline before spending on the full end-to-end experiment. Freeze prompts,
splits, and metrics before scaling the run.
