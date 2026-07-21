# Revision candidates after the P0 reruns

These passages deliberately avoid freezing the current test-selected numbers. Replace bracketed fields only after seed provenance and dev-only selection are complete.

## Candidate title

**When Conservative GRPO Under-Updates: Learning-Rate Calibration Across Domains**

Alternative: **Calibrating GRPO Beyond Mathematics: A Controlled Multi-Domain Study**

## Candidate abstract

Group Relative Policy Optimization (GRPO) is increasingly applied beyond mathematical reasoning, often with optimization settings inherited from math-centric studies. We examine whether such settings transfer across four reasoning and knowledge domains under a shared backbone, adaptation method, data budget, and evaluation protocol. A conservative GRPO configuration yields little or inconsistent improvement over the base model, whereas learning rates selected exclusively on held-out development data improve test accuracy by [X--Y] points on [K/4] domains. The effect persists across [N] independent training seeds and [M] model families, but is not universal: gains vary substantially across component benchmarks and a small-data legal task does not benefit. Run-level diagnostics are consistent with an under-update explanation—conservative settings induce smaller policy movement—without implying that learning rate changes GRPO's divergence direction. We further audit multiple-choice evaluation through answer-option permutations and output-parser checks, and compare GRPO with compute- and data-accounted filtered self-training. Our results show that GRPO configurations do not transfer reliably across domains and should be calibrated on held-out data, with benchmark-level reporting and independent-seed uncertainty.

## Candidate contribution bullets

1. We conduct a controlled, multi-seed comparison of conservative and development-selected GRPO configurations across four domains, reporting both component-benchmark and macro-averaged results.
2. We show that conservative settings can under-update the policy outside their original calibration regime; higher learning rates often help, but the effect is domain dependent rather than universal.
3. We audit the result with answer-option permutations, parser-error measurements, paired uncertainty estimates, and run-level policy-update diagnostics.

## Claims to delete, not merely soften

- “lr=5e-7, the rate used in DeepSeekMath”
- “40x above the DeepSeekMath value”
- “consistently surpassing SFT across all domains”
- “the first controlled multi-domain comparison”
- “external signal is necessary”
- “pure reverse-KL minimization”
- “higher learning rates capture new modes / acquire new knowledge”
- “learning-rate sensitivity is consistent across five architectures” when conservative controls are absent
- “SFT perplexity predicts degradation” from three domain-level points
- “frac_reward_zero_std predicts success” from three domain-level points

## Replacement vocabulary

| Current wording | Safer replacement |
|---|---|
| math-optimized default | conservative configuration / configuration inherited from math-focused studies |
| SFT teacher demonstrations | filtered self-training or rejection-sampling fine-tuning, unless an external teacher is actually used |
| matched compute | shared backbone, LoRA configuration, training prompts, and evaluation protocol |
| mechanism | empirical pattern / under-update hypothesis |
| new knowledge acquisition | improved held-out benchmark accuracy |
| five model families | five model checkpoints from four organizations |
| OOD generalization | cross-benchmark transfer, only when train/test sets are genuinely disjoint |
| best learning rate | development-selected learning rate |

## Recommended main-results sentence pattern

> Using a learning rate selected on the development split, GRPO improves the macro-averaged test score from [BASE] to [SCORE] ([DELTA] points; [CI] hierarchical-bootstrap CI) across four domains. Improvements are strongest on [DOMAINS], while [DOMAIN/BENCHMARK] shows no reliable gain. Individual training-seed results and component-benchmark scores are reported rather than pooling all questions into a single micro average.

## Recommended limitation paragraph

> Our study isolates learning rate within a fixed LoRA-based GRPO implementation, but learning rate is only one determinant of effective policy movement. Batch size, number of updates, clipping, KL regularization, adapter capacity, and response length may produce related effects. We therefore interpret the observed threshold as implementation- and model-dependent rather than a universal constant. In addition, the non-mathematical supervised baseline uses filtered self-generated responses; conclusions about externally supervised fine-tuning require a separate teacher-data comparison. Finally, improvements on multiple-choice benchmarks do not by themselves establish acquisition of new factual knowledge.
