# Post-hoc Swap Filtering Summary

Generated from local eval artifacts only. Values are percentages.
Coverage is the fraction of examples whose original and swapped predictions agree after label flipping.
Covered accuracy is accuracy on that retained subset; random fallback accuracy assumes a random binary choice on inconsistent examples.

| Setting | Runs | Standard Acc | Coverage | Covered Acc | Random Fallback Acc | Inconsistent Orig-Correct | Source note |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | 1 | 80.2 | 81.5 | 84.2 | 77.8 | 62.7 | Untrained Qwen2.5 baseline. |
| SFT unbalanced | 1 | 100.0 | 0.0 | 0.0 | 50.0 | 100.0 | Unbalanced SFT endpoint. |
| SFT balanced | 1 | 91.3 | 87.1 | 96.2 | 90.2 | 58.6 | Balanced SFT endpoint. |
| GRPO acc-only unbalanced | 3 | 94.7 +/- 0.7 | 59.8 +/- 7.6 | 92.4 +/- 0.7 | 75.3 +/- 2.8 | 98.4 +/- 0.8 | Multi-seed unbalanced accuracy-reward GRPO. |
| GRPO full unbalanced | 4 | 94.7 +/- 0.9 | 58.7 +/- 9.1 | 92.3 +/- 0.6 | 74.9 +/- 3.8 | 97.9 +/- 0.8 | Multi-seed unbalanced full-reward GRPO. |
| GRPO full balanced | 5 | 83.7 +/- 1.1 | 84.0 +/- 0.8 | 88.8 +/- 0.6 | 82.6 +/- 0.8 | 56.8 +/- 7.2 | Multi-seed balanced full-reward GRPO. |

## Sources

- Baseline: results/baseline_qwen7b/eval_results.json
- SFT unbalanced: results/SFT_unbalanced/eval/eval_results.json
- SFT balanced: results/SFT_balanced/eval/eval_results.json
- GRPO acc-only unbalanced: results/EXP-006_accuracy_only/eval/eval_results.json, results/EXP-006_accuracy_s2/eval/eval_results.json, results/EXP-006_accuracy_s3/eval/eval_results.json
- GRPO full unbalanced: results/EXP-009_full_composite/eval/eval_results.json, results/EXP-009_full_composite_s2/eval/eval_results.json, results/EXP-009_full_composite_s3/eval/eval_results.json, results/EXP-009_full_s4/eval/eval_results.json
- GRPO full balanced: results/EXP-009b_full_balanced/eval/eval_results.json, results/EXP-009b_full_balanced_s2/eval/eval_results.json, results/EXP-009b_full_balanced_s3/eval/eval_results.json, results/balanced_full_s4/eval/eval_results.json, results/balanced_full_s5/eval/eval_results.json

