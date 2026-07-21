# AAAI-27 Submission Priorities

Updated: 2026-07-21 (Asia/Shanghai)

## Immediate objective: submit the abstract first

- AAAI-27 abstract deadline: **2026-07-21, 11:59 PM UTC-12 (AoE)**.
- Full-paper deadline: **2026-07-28, 11:59 PM UTC-12 (AoE)**.
- Supplementary material and code deadline: **2026-07-31, 11:59 PM UTC-12 (AoE)**.
- Official page: <https://aaai.org/conference/aaai/aaai-27/>

Do **not** wait for human evaluation before submitting the abstract.  The
current abstract should retain the conservative framing already used in the
paper:

> This is a controlled empirical study of when short behavioral redirects help
> or hurt code agents.  The main finding is evaluator-sensitive matched gains
> and more persistent negative transfer under mismatched redirects.  The paper
> does not yet claim a deployable controller or end-to-end software repair.

The abstract should not promise human evaluation, end-to-end repair, or an
online structure detector before those results exist.

## P0 after the abstract: remove privileged context

This is the most important missing experiment for the full paper.

### Question

Do redirects still change next-action quality when the prompt does not reveal
the gold failure category or gold target files?

### Minimum comparison

1. No-scaffold control.
2. Matched redirect selected without test-instance gold information.
3. Fixed redirect.
4. Abstain / no-intervention condition.

Use only the issue and trajectory prefix observable at intervention time.  Do
not provide the gold category, gold target files, gold patch, or later
trajectory information.  Freeze the split, prompts, and primary metric before
running the experiment.

### Where it goes

Place it immediately after the experimental setup and make it the first main
result:

```text
5.1 Experimental Setup
5.2 Leakage-Free Redirect Evaluation       <- new primary result
5.3 End-to-End Outcomes                     <- if completed
5.4 Evaluator Sensitivity and Negative Transfer
5.5 Oracle-Context Diagnostic Analysis
```

The current gold-context candidate-best and oracle results should become
diagnostic evidence, not the headline AAAI result.

## P0/P1: end-to-end repair pilot

Run this after the leakage-free experiment if infrastructure is ready.

- Resume the agent from the intervention point under a fixed step/token budget.
- Compare no intervention, matched redirect, fixed redirect, and gated
  redirect with abstention.
- Report resolved rate, tests passed, intervention harm rate, tokens/cost, and
  wall-clock time.
- A 30--50-instance pilot is useful under the deadline; a 50--100-instance
  held-out evaluation is preferable.

This result belongs in the main paper directly after the leakage-free
next-action experiment.  Do not hide it in supplementary material if it is
used to support a repair claim.

## P1 if time permits: prefix-only structure detector

This is necessary only if the paper continues to claim a deployable
structure-aware controller.  Train and evaluate using prefix-only features,
project-grouped splits, calibration, and an abstention curve.  If it cannot be
completed, keep the paper framed as a diagnostic/measurement study and weaken
controller claims accordingly.

Place a compact result in the failure-classification section; put feature
definitions and full confusion matrices in supplementary material.

## Deferred: human evaluation

Human evaluation is **not required for today's abstract submission** and should
not block the P0 experiments above.  Add it later when annotator availability
and quality control can be guaranteed.

When feasible, use two blinded annotators on a stratified sample and report
agreement plus disagreements among human ratings, the regex proxy, and the LLM
judge.  Mention the main agreement result briefly in the setup; place the full
rubric and disagreement analysis in supplementary material.

Until then, retain the current cautious wording: the blinded LLM-judge subset
is a sensitivity analysis, not human validation.

## What not to prioritize this week

- More models under the same gold-context regex protocol.
- Additional cosmetic figures or more proxy-only tables.
- A second benchmark before privileged-context leakage is addressed.
- Full human annotation before the abstract or before P0 is running.

The existing nine-model analysis is already broad enough for a diagnostic
result.  The decisive AAAI evidence is leakage-free evaluation and, if
possible, an end-to-end repair pilot.

## Minimum full-paper target

The minimum credible AAAI package is:

1. Existing failure taxonomy and negative-transfer analysis.
2. A leakage-free next-action experiment with held-out selection.
3. Independent scoring that does not reuse type-specific lexical cues.
4. Honest scope language distinguishing next-action quality from repair.
5. Preferably, a small end-to-end repair pilot.

Human evaluation remains a valuable later addition, but it is not on the
critical path for the abstract deadline.
