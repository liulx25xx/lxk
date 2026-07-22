# AAAI-27 Selective-Recovery Experiment Priorities

Updated: 2026-07-22 (Asia/Shanghai)

## Paper thesis after the EMNLP-review alignment

The paper is no longer organized around the claim that cascade type predicts a
fixed scaffold.  Its new thesis is:

> Recovery is a selective control problem.  Given only the observable trajectory
> prefix, a controller should issue a cheap redirect when its incremental value
> is positive, escalate to richer execution evidence when a prompt is
> insufficient, and abstain when intervention risk is too high.

The existing 143-trajectory and 1,400-response results are retained as an
oracle-context diagnostic study.  They support evaluator-sensitive apparent
matched gains and more persistent negative transfer.  They do not establish a
deployable controller or software repair.

## Abstract-deadline claim boundary

The current abstract may claim:

1. The selective-recovery formulation: redirect, escalate, or abstain.
2. The audited trajectory sample and controlled response count.
3. The verified proxy-ablation, LLM-judge, and paired negative-transfer results.
4. The design implication that policy value and post-action verification matter
   more than four-way category accuracy alone.

Do not claim that the selective controller improves SWE-bench resolved rate
until the experiments below are complete.  Human evaluation is deferred and
must not block the abstract.

## P0: leakage-free selective-action evaluation

This is the most important missing experiment.

### Research question

Can a policy choose between no intervention, a cheap redirect, and evidence
escalation using only information available at decision time?

### Frozen protocol

- Split by project before prompt or policy selection.
- Construct each decision state from the issue and trajectory prefix ending at
  a reproducible observable trigger.
- Exclude gold category, gold target files, gold patch, full-trajectory counts,
  and all later turns.
- Freeze prompts, action definitions, primary outcome, and split manifest before
  generating test outputs.

### Minimum action set

1. **Abstain/control:** continue the original agent without an extra directive.
2. **Universal redirect:** one fixed low-cost prompt such as `step_back`.
3. **Observable-rule redirect:** use only obvious prefix signals, such as a
   repeated edit-tool error, to select a narrow redirect.
4. **Evidence escalation:** request a targeted test/trace analysis under a
   matched additional-token budget.
5. **Selective policy:** choose among the actions above with confidence-based
   abstention.

Gold-category routing may appear only as an oracle upper bound.

### Primary reporting

- Incremental outcome relative to the same-instance abstain/control condition.
- Risk--coverage curve for the selective policy.
- Intervention harm rate: fraction of acted-on cases worse than control.
- Action distribution and abstention rate.
- Tokens, model calls, latency, and estimated cost.
- Project-clustered paired confidence intervals.

Do not use the old 0--3 regex proxy as the sole primary metric.  Prefer an
execution-grounded progress outcome.  If a next-action judge is needed for the
first stage, blind it to condition and report it separately from repository
outcomes.

## P0: end-to-end repository continuation

This experiment addresses the strongest common criticism from all three EMNLP
reviews.

- Resume the agent from each frozen decision state for the same step, token, and
  wall-clock budget.
- Compare abstain/control, always redirect, always escalate, and the selective
  policy.
- Run repository tests in isolated SWE-bench containers.
- Report resolved rate, test-progress rate, regression rate, harm rate, tokens,
  wall-clock time, and cost.
- Use 30--50 held-out instances for a deadline pilot; scale to 50--100 if the
  harness is stable.
- Repeat stochastic conditions or use temperature zero with a clearly stated
  deterministic protocol.

The end-to-end table should become the first main result.  The current
oracle-context proxy study should move after it as diagnostic evidence.

## P0/P1: calibrated action-value gate

Do not rebuild the unreproducible four-way online classifier as the headline
method.  Train or define a gate that estimates whether an action beats
abstention.

Minimum baselines:

- always abstain;
- always use the best development-set redirect;
- repeated-error rule;
- multinomial logistic regression over observable prefix features;
- random forest or gradient-boosted trees;
- optional small LLM/sequence gate.

Use project-grouped validation and save every held-out prediction.  Report
calibration, risk--coverage, per-action policy value, and confusion matrices only
as secondary diagnostics.  A low-coverage policy that avoids harm can be more
useful than a high-accuracy four-way classifier.

## P1: stronger escalation for the majority applied-patch group

The applied-but-unresolved group contains 70/143 trajectories.  The old paper
treated its weak prompt response as a behavioral frontier; the new paper should
test stronger recovery actions instead.

Compare under matched or explicitly reported budgets:

1. one-sentence `minimal_fix` / `test_analysis` redirect;
2. evidence-rich targeted test analysis;
3. execution-trace-conditioned replanning;
4. optional PRM/critic-guided next action;
5. abstain/control.

Primary evidence is downstream test or repair progress, not lexical relevance.
This result belongs directly after the end-to-end main table or as the first
mechanism analysis.

## P1: qualitative and evaluator validation

The paper now includes two artifact-grounded qualitative audit cases.  Before
submission, expand this to a small pre-specified set containing:

- a redirect that changes the next action and later helps;
- a plausible redirect that harms relative to same-instance control;
- a proxy false positive;
- an escalation that succeeds where a short redirect does not;
- a case where abstention is the correct decision.

Human evaluation remains deferred until reliable annotators are available.  If
completed, use two blinded annotators on a stratified sample and report
agreement.  Human next-action ratings complement but do not replace repository
tests.

## P2: breadth and generalization

After the P0 causal chain is working, add one additional agent harness or one
typed/compiled-language setting.  This directly addresses the single-agent,
Python-only criticism.  Do not spend the deadline budget on more models under
the same leaked regex protocol; the existing nine-model matrix is already
sufficient as a diagnostic appendix result.

## Recommended execution order

1. Freeze projects, observable trigger, action set, and primary outcomes.
2. Run a 10--20-instance end-to-end infrastructure smoke test.
3. Generate development data for the leakage-free action comparison.
4. Fit/calibrate the action-value gate and freeze its threshold.
5. Run the held-out 30--50-instance selective-recovery pilot.
6. Add the stronger escalation comparison for applied-but-unresolved cases.
7. Only then expand to another harness/language or human evaluation.

## Minimum credible AAAI package

1. Current audited diagnostic evidence and paired negative transfer.
2. Leakage-free held-out action selection with an abstention curve.
3. A small end-to-end repository continuation table.
4. At least one evidence-escalation baseline for the majority failure group.
5. Reproducible prompts, split manifest, per-instance outputs, cost, and tests.

