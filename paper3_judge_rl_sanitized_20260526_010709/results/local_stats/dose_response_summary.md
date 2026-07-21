# Confound-Ratio and Duplication Summary

Generated from local artifacts only. Values are percentages unless marked as pp.
Consistency loss is computed as 84.0 minus each row's consistency.
Multi-run rows are formatted as mean +/- sample standard deviation; intermediate ratio rows are single-seed diagnostics.

| Condition | Pos-A | Train n | Dup. | Runs | Acc | Con | First-pos | Loss pp |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 50 mirrored balanced reference | 50 | 4178 | yes | 5 | 83.7 +/- 1.1 | 84.0 +/- 0.8 | 50.0 +/- 1.0 | 0.0 |
| 50 non-duplicated balanced | 50 | 2089 | no | 1 | 83.1 | 82.6 | 50.2 | 1.4 |
| 60 ratio | 60 | 2089 | no | 1 | 83.5 | 83.1 | 50.7 | 0.9 |
| 75 ratio | 75 | 2089 | no | 1 | 85.3 | 85.7 | 52.0 | -1.7 |
| 80 ratio | 80 | 2089 | no | 1 | 87.8 | 80.2 | 55.5 | 3.8 |
| 90 ratio | 90 | 2089 | no | 1 | 88.6 | 76.6 | 56.6 | 7.4 |
| 95 ratio | 95 | 2089 | no | 1 | 87.1 | 78.0 | 55.6 | 6.1 |
| 100 fully unbalanced | 100 | 2089 | no | 4 | 94.7 +/- 0.9 | 58.7 +/- 9.1 | 68.8 +/- 4.3 | 25.3 |

## Sources

- 50 mirrored balanced reference: Multi-seed GRPO full balanced reference used for consistency-loss normalization. Sources: results/EXP-009b_full_balanced/eval/eval_results.json, results/EXP-009b_full_balanced_s2/eval/eval_results.json, results/EXP-009b_full_balanced_s3/eval/eval_results.json, results/balanced_full_s4/eval/eval_results.json, results/balanced_full_s5/eval/eval_results.json
- 50 non-duplicated balanced: Single-seed control with one balanced orientation per training instance. Sources: results/GRPO_balanced_nodupe/eval/eval_results.json
- 60 ratio: Single-seed intermediate confound-ratio diagnostic. Sources: results/GRPO_ratio60/eval/eval_results.json
- 75 ratio: Single-seed intermediate confound-ratio diagnostic. Sources: results/GRPO_ratio75/eval/eval_results.json
- 80 ratio: Single-seed intermediate confound-ratio diagnostic. Sources: results/GRPO_ratio80/eval/eval_results.json
- 90 ratio: Single-seed intermediate confound-ratio diagnostic. Sources: results/GRPO_ratio90/eval/eval_results.json
- 95 ratio: Single-seed intermediate confound-ratio diagnostic. Sources: results/GRPO_ratio95/eval/eval_results.json
- 100 fully unbalanced: Multi-seed GRPO full unbalanced endpoint. Sources: results/EXP-009_full_composite/eval/eval_results.json, results/EXP-009_full_composite_s2/eval/eval_results.json, results/EXP-009_full_composite_s3/eval/eval_results.json, results/EXP-009_full_s4/eval/eval_results.json
