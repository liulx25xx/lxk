# External Judge Summary

Generated from local eval artifacts only. Values are percentages.
Gap is original-order accuracy minus swapped-order accuracy; Bias is always-first minus always-second inconsistent rate.

| Model | Runs | Orig Acc | Swap Acc | Gap | Con | First-pos | Bias | Source note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| JudgeLRM-7B | 1 | 81.3 [77.4, 84.6] | 66.1 [61.6, 70.4] | 15.1 | 75.7 [71.6, 79.5] | 53.1 [49.8, 56.4] | 13.4 | Public JudgeLRM 7B checkpoint evaluated with the same two-order diagnostic. |
| JudgeLRM-3B | 1 | 76.8 [72.7, 80.5] | 55.9 [51.3, 60.4] | 20.9 | 67.9 [63.5, 72.1] | 56.7 [53.4, 59.9] | 15.4 | Public JudgeLRM 3B checkpoint evaluated with the same two-order diagnostic. |

## Sources

- JudgeLRM-7B: results/judgelrm_7b/eval/eval_results.json
- JudgeLRM-3B: results/judgelrm_3b/eval/eval_results.json

