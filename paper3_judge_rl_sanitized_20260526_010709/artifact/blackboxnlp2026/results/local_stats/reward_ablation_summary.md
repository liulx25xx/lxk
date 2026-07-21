# Reward Ablation Summary

Generated from local artifacts only. Values are percentages.
Qwen2.5 rows are multi-seed where available; Qwen3 rows are single-seed reward controls.

## Qwen2.5-7B

| Reward | Runs | Acc | Swap Acc | Con | First-pos | Source note |
|---|---:|---:|---:|---:|---:|---|
| Accuracy only | 3 | 94.7 +/- 0.7 | 55.7 +/- 6.4 | 59.8 +/- 7.6 | 68.1 +/- 3.6 | Multi-seed accuracy-reward GRPO on unbalanced data. |
| + Decisive | 2 | 93.9 +/- 1.1 | 61.0 +/- 5.4 | 66.3 +/- 6.5 | 65.0 +/- 3.5 | Multi-seed accuracy plus decisiveness reward on unbalanced data. |
| + Confidence proxy | 2 | 95.4 +/- 0.5 | 53.1 +/- 7.4 | 56.1 +/- 7.9 | 69.4 +/- 4.0 | Multi-seed accuracy plus fixed-confidence proxy on unbalanced data. |
| Full composite | 4 | 94.7 +/- 0.9 | 54.8 +/- 8.3 | 58.7 +/- 9.1 | 68.8 +/- 4.3 | Multi-seed full composite reward on unbalanced data. |

## Qwen3-8B

| Reward | Runs | Acc | Swap Acc | Con | First-pos | Source note |
|---|---:|---:|---:|---:|---:|---|
| Accuracy only | 1 | 88.4 | 73.5 | 78.8 | 55.2 | Single-seed accuracy-reward GRPO on unbalanced data. |
| + Decisive | 1 | 88.2 | 74.8 | 82.0 | 55.6 | Single-seed accuracy plus decisiveness reward on unbalanced data. |
| + Confidence proxy | 1 | 89.1 | 72.2 | 78.0 | 57.1 | Single-seed accuracy plus fixed-confidence proxy on unbalanced data. |
| Full composite | 1 | 87.8 | 74.4 | 80.8 | 55.5 | Single-seed full composite reward on unbalanced data. |

## Sources

- Qwen2.5-7B Accuracy only: results/EXP-006_accuracy_only/eval/eval_results.json, results/EXP-006_accuracy_s2/eval/eval_results.json, results/EXP-006_accuracy_s3/eval/eval_results.json
- Qwen2.5-7B + Decisive: results/EXP-007a_acc_decisive/eval/eval_results.json, results/EXP-007a_acc_decisive_s2/eval/eval_results.json
- Qwen2.5-7B + Confidence proxy: results/EXP-008_acc_calib/eval/eval_results.json, results/EXP-008_acc_calib_s2/eval/eval_results.json
- Qwen2.5-7B Full composite: results/EXP-009_full_composite/eval/eval_results.json, results/EXP-009_full_composite_s2/eval/eval_results.json, results/EXP-009_full_composite_s3/eval/eval_results.json, results/EXP-009_full_s4/eval/eval_results.json
- Qwen3-8B Accuracy only: results/GRPO_qwen3_8b_unbalanced/eval/eval_results.json
- Qwen3-8B + Decisive: results/GRPO_qwen3_8b_decisive_unbal/eval/eval_results.json
- Qwen3-8B + Confidence proxy: results/GRPO_qwen3_8b_calib_unbal/eval/eval_results.json
- Qwen3-8B Full composite: results/GRPO_qwen3_8b_full_composite_unbal/eval/eval_results.json

