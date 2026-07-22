# GPU / Server Queue for Selective Recovery

Updated: 2026-07-22

This queue follows the revised paper thesis in `main_neurips.tex`: select among
redirect, evidence escalation, and abstention by expected intervention utility.
The old four-way classifier is not a submission target unless it can be rebuilt
from prefix-only features with complete held-out artifacts.

## Already completed locally

No additional model calls were needed for the following audited results:

- leave-one-project-out selection over stored responses;
- proxy-component ablations;
- paired LLM-judge sensitivity on the available subset;
- same-instance paired cross-category transfer;
- artifact-grounded qualitative proxy-failure cases;
- regenerated publication figures and project-clustered intervals.

These are documented in `LOCAL_EXPERIMENTS.md`, `LOCAL_AUDIT.md`,
`results/local_offline_experiments.json`, and the local analysis scripts.

## P0.1: leakage-free action comparison

**Question:** Which recovery action has positive incremental value when only the
issue and observable prefix are available?

**Conditions:**

1. abstain/control;
2. fixed `step_back` redirect;
3. repeated-edit-error rule with `reread_file`;
4. evidence escalation through targeted test/trace analysis;
5. learned or calibrated selective policy.

**Protocol:** project-held-out split; reproducible observable trigger; no gold
category, target files, patch, full trajectory, or future actions.  Freeze the
split, prompts, token budgets, and primary metric before the test run.

**Minimum scale:** 96 instances x 4--5 conditions x 2--3 representative models.
Use temperature zero or at least three generations for stochastic conditions.

**Resources:** hosted API or local inference server.  A GPU is needed only for
locally hosted models.  Save raw prompts, outputs, token counts, latency, costs,
and test-state manifests.

**Primary analysis:** same-instance incremental value, project-clustered paired
intervals, intervention harm rate, and risk--coverage curves.  The old regex
score may be a secondary diagnostic but cannot be the sole primary outcome.

## P0.2: end-to-end repository continuation

**Question:** Does selective recovery improve actual repository progress under
equal budget?

Resume from the frozen decision state and compare:

- abstain/control;
- always redirect;
- always escalate;
- selective policy with abstention.

Use isolated SWE-bench containers and a fixed step/token/wall-clock budget.
Report resolved rate, tests passed or newly passing, regressions, intervention
harm, tokens, latency, and cost.

**Scale:** first run a 10--20-instance infrastructure smoke test.  If stable,
run 30--50 held-out instances for the deadline and expand to 50--100 later.

**Resources:** multi-core server, Docker storage, API budget, and optionally one
GPU for local models.  This is the most infrastructure-intensive experiment.

## P0.3: prefix-only action-value gate

**Question:** Can the system predict when acting beats abstention, rather than
only predicting a retrospective failure label?

Construct features from the observable prefix only.  Candidate signals include
recognized tool errors, repeated-action counts, action-type entropy, edit-target
entropy, test/edit alternation, elapsed tokens, and recent environment feedback.
Exclude all gold and post-decision features.

**Baselines:** always abstain, always redirect, repeated-error rule, logistic
regression, random forest/gradient boosting, and an optional small LLM gate.

**Validation:** project-grouped or repository-grouped splits; threshold chosen
on development data; every held-out prediction saved.  Report calibration,
risk--coverage, action-specific value, harm, and cost.  Four-way accuracy is a
secondary diagnostic only.

**Resources:** CPU for tabular models; GPU/API for embeddings, sequence models,
or LLM gates.

## P1.1: evidence escalation for applied-but-unresolved cases

The largest operational group contains 70/143 trajectories and did not respond
clearly to short redirects under the stored proxy.  Compare:

- short `minimal_fix` or `test_analysis` redirect;
- targeted test analysis with the current failure output;
- execution-trace-conditioned replanning;
- optional critic/PRM guidance;
- abstain/control.

Use matched or explicitly reported compute budgets.  Primary outcomes are
downstream test progress and repair success.  This experiment replaces the old
unsupported claim of a universal ``scaffolding frontier.''

**Resources:** API/server inference and containers; GPU only for a local critic
or locally hosted model.

## P1.2: intervention timing and verification

Compare first observable error, repeated-error threshold, fixed mid-trajectory,
and confidence-triggered intervention.  After acting, verify repository progress
and allow continue/escalate/stop decisions.  Report the quality--cost--harm
trade-off, not only the best timing point.

## P1.3: human/evaluator validation

Human annotation is deferred.  When annotators are available, use two blinded
raters on a pre-specified stratified sample and report agreement and disagreement
with the regex proxy and LLM judge.  This requires annotation time, not a GPU.
Repository tests remain the primary recovery metric.

## P2: harness and language transfer

After P0 works, repeat the held-out selective policy on one additional agent
harness or a typed/compiled-language repository set.  Keep the observation and
budget contract fixed.  Do not prioritize additional models under the old
gold-context regex protocol.

## Execution order

1. Freeze the split, decision trigger, action set, budgets, and metrics.
2. Run the small end-to-end smoke test.
3. Run P0.1 development conditions.
4. Fit and freeze the P0.3 action-value gate.
5. Run the held-out P0.1/P0.2 selective-policy evaluation.
6. Run the stronger P1.1 escalation comparison.
7. Add timing, human evaluation, and transfer only after the core result is
   stable.

