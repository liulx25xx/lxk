# Learning Rate Sweep Summary

Generated from local artifacts only. Values are percentages.
Rows are single-seed unless a row lists more than one run.
Qwen2.5 uses the full composite reward for the sweep; Qwen3 uses the accuracy reward for the sweep.

## Qwen2.5-7B

| LR | Reward | Runs | Acc | Swap Acc | Con | First-pos | Source note |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | none | 1 | 80.2 | 75.3 | 81.5 | 49.3 | Untrained baseline. |
| 1e-6 | full | 1 | 83.7 | 76.2 | 83.3 | 51.0 | Single-seed full-reward LR control. |
| 2e-6 | full | 1 | 85.7 | 73.3 | 81.1 | 54.3 | Single-seed full-reward LR control. |
| 3e-6 | full | 1 | 90.0 | 70.2 | 75.7 | 58.2 | Single-seed full-reward LR control. |
| 5e-6 | full | 1 | 94.0 | 57.2 | 61.7 | 67.4 | Standard full-reward seed used in the single-seed LR sweep. |
| 1e-5 | full | 1 | 98.9 | 37.0 | 38.1 | 80.5 | Single-seed full-reward LR control. |

## Qwen3-8B

| LR | Reward | Runs | Acc | Swap Acc | Con | First-pos | Source note |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | none | 1 | 86.0 | 74.2 | 80.4 | 53.3 | Untrained disable-thinking baseline. |
| 1e-6 | accuracy | 1 | 89.1 | 76.2 | 82.4 | 55.3 | Single-seed accuracy-reward LR control. |
| 2e-6 | accuracy | 1 | 87.8 | 75.3 | 81.5 | 54.8 | Single-seed accuracy-reward LR control. |
| 3e-6 | accuracy | 1 | 87.8 | 75.5 | 82.6 | 54.8 | Single-seed accuracy-reward LR control. |
| 5e-6 | accuracy | 1 | 88.4 | 73.5 | 78.8 | 55.2 | Seed-42 standard accuracy-reward run; multi-seed mean is reported in the cross-model table. |
| 1e-5 | accuracy | 1 | 90.2 | 72.2 | 78.4 | 58.0 | Single-seed accuracy-reward LR control. |

## Sources

- Qwen2.5-7B baseline: results/baseline_qwen7b/eval_results.json
- Qwen2.5-7B 1e-6: results/EXP-009_full_lr1e6/eval/eval_results.json
- Qwen2.5-7B 2e-6: results/EXP-009_full_lr2e6/eval/eval_results.json
- Qwen2.5-7B 3e-6: results/EXP-009_full_lr3e6/eval/eval_results.json
- Qwen2.5-7B 5e-6: results/EXP-009_full_composite/eval/eval_results.json
- Qwen2.5-7B 1e-5: results/EXP-009_full_lr1e5/eval/eval_results.json
- Qwen3-8B baseline: results/baseline_qwen3_8b/eval/eval_results.json
- Qwen3-8B 1e-6: results/GRPO_qwen3_8b_unbal_lr1e6/eval/eval_results.json
- Qwen3-8B 2e-6: results/GRPO_qwen3_8b_unbal_lr2e6/eval/eval_results.json
- Qwen3-8B 3e-6: results/GRPO_qwen3_8b_unbal_lr3e6/eval/eval_results.json
- Qwen3-8B 5e-6: results/GRPO_qwen3_8b_unbalanced/eval/eval_results.json
- Qwen3-8B 1e-5: results/GRPO_qwen3_8b_unbal_lr1e5/eval/eval_results.json

