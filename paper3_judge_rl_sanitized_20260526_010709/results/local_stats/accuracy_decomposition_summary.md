# Accuracy Decomposition Summary

Generated from canonical cross-model rows. Values are original-order accuracy percentages.
Genuine = balanced GRPO - baseline; Shortcut = unbalanced GRPO - balanced GRPO.

| Model | Baseline | Unbal GRPO | Bal GRPO | Apparent gain | Genuine | Shortcut |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | 80.2 | 94.7 | 83.7 | +14.5 | +3.5 | +11.0 |
| Qwen3-8B | 86.0 | 88.7 | 87.4 | +2.7 | +1.4 | +1.3 |
| Mistral-7B | 65.7 | 98.4 | 81.1 | +32.7 | +15.4 | +17.4 |

## Sources

- Qwen2.5-7B: results/baseline_qwen7b/eval_results.json, results/EXP-009_full_composite/eval/eval_results.json, results/EXP-009_full_composite_s2/eval/eval_results.json, results/EXP-009_full_composite_s3/eval/eval_results.json, results/EXP-009_full_s4/eval/eval_results.json, results/EXP-009b_full_balanced/eval/eval_results.json, results/EXP-009b_full_balanced_s2/eval/eval_results.json, results/EXP-009b_full_balanced_s3/eval/eval_results.json, results/balanced_full_s4/eval/eval_results.json, results/balanced_full_s5/eval/eval_results.json
- Qwen3-8B: results/baseline_qwen3_8b/eval/eval_results.json, results/GRPO_qwen3_8b_unbalanced/eval/eval_results.json, results/GRPO_qwen3_8b_unbalanced_s2/eval/eval_results.json, results/GRPO_qwen3_8b_unbal_s3/eval/eval_results.json, results/GRPO_qwen3_8b_balanced/eval/eval_results.json, results/GRPO_qwen3_8b_balanced_s2/eval/eval_results.json
- Mistral-7B: results/baseline_mistral7b/eval/metrics.json, results/GRPO_mistral7b_unbal/eval/eval_results.json, results/GRPO_mistral7b_unbal_s2/eval/eval_results.json, results/GRPO_mistral7b_unbal_s3/eval/eval_results.json, results/GRPO_mistral7b_balanced/eval/eval_results.json, results/GRPO_mistral7b_balanced_s2/eval/eval_results.json

