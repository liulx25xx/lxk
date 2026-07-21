# Length Confound Control Summary

Generated from local eval artifacts only. Values are percentages.
The length-confounded control is included to test whether a non-positional artifact produces the same position-shortcut collapse.

| Setting | Runs | Confound | Acc | Swap Acc | Con | First-pos | Tie | Source note |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Baseline | 1 | none | 80.2 | 75.3 | 81.5 | 49.3 | 5.1 | Untrained Qwen2.5 baseline. |
| Position-confounded GRPO-A | 3 | preferred response always first | 94.7 +/- 0.7 | 55.7 +/- 6.4 | 59.8 +/- 7.6 | 68.1 +/- 3.6 | 2.1 +/- 0.3 | Primary unbalanced accuracy-reward GRPO comparison. |
| Length-confounded GRPO | 3 | longer response preferred in training | 80.9 +/- 0.8 | 75.8 +/- 1.1 | 79.7 +/- 1.5 | 50.6 +/- 0.2 | 3.6 +/- 0.4 | Control trained on a length-confounded objective for 300 steps. |

## Sources

- Baseline: results/baseline_qwen7b/eval_results.json
- Position-confounded GRPO-A: results/EXP-006_accuracy_only/eval/eval_results.json, results/EXP-006_accuracy_s2/eval/eval_results.json, results/EXP-006_accuracy_s3/eval/eval_results.json
- Length-confounded GRPO: results/EXP-LENGTH_confounded/eval/eval_results.json, results/length_confounded_s2/eval/eval_results.json, results/length_confounded_s3/eval/eval_results.json

