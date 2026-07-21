# Training Dynamics Summary

Generated from local artifacts only. Values are percentages; deltas are percentage points relative to the previous available checkpoint for the same variant, using the baseline as the first reference.
The paper table uses only checkpoints shared by the acc-only and full-reward trajectories; extra one-sided artifacts are retained here for auditing.

| Variant | Steps | Paper | Acc | Swap Acc | Con | First-pos | Delta Acc | Delta Con | Delta First-pos | Source note |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 0 | yes | 80.2 | 75.3 | 81.5 | 49.3 | --- | --- | --- | Untrained Qwen2.5-7B baseline shared by both trajectories. |
| Acc-only | 100 | yes | 86.0 | 74.2 | 80.8 | 53.8 | +5.8 | -0.7 | +4.5 | Accuracy-reward checkpoint used for the paired dynamics table. |
| Acc-only | 200 | yes | 89.1 | 70.4 | 76.8 | 57.1 | +3.1 | -4.0 | +3.3 | Accuracy-reward checkpoint used for the paired dynamics table. |
| Acc-only | 300 | yes | 92.2 | 60.6 | 65.3 | 63.7 | +3.1 | -11.6 | +6.6 | Accuracy-reward checkpoint used for the paired dynamics table. |
| Acc-only | 500 | yes | 94.4 | 56.8 | 60.8 | 67.4 | +2.2 | -4.5 | +3.7 | Final accuracy-reward run used as the 500-step endpoint. |
| Full | 100 | yes | 84.6 | 73.9 | 81.7 | 53.5 | +4.5 | +0.2 | +4.1 | Full-reward checkpoint used for the paired dynamics table. |
| Full | 200 | yes | 90.9 | 68.6 | 74.6 | 59.8 | +6.2 | -7.1 | +6.3 | Full-reward checkpoint used for the paired dynamics table. |
| Full | 300 | yes | 93.5 | 61.5 | 65.9 | 64.9 | +2.7 | -8.7 | +5.1 | Full-reward checkpoint used for the paired dynamics table. |
| Full | 400 | no | 95.3 | 59.2 | 63.5 | 66.7 | +1.8 | -2.4 | +1.8 | Additional full-reward-only checkpoint; omitted from the paired paper table because acc-only lacks a matching 400-step artifact. |
| Full | 500 | yes | 94.0 | 57.2 | 61.7 | 67.4 | -1.3 | -1.8 | +0.7 | Final full-reward run used as the 500-step endpoint. |

## Sources

- Baseline step 0: results/baseline_qwen7b/eval_results.json
- Acc-only step 100: results/EXP-006_accuracy_only/eval_ckpt100/eval_results.json
- Acc-only step 200: results/EXP-006_accuracy_only/eval_ckpt200/eval_results.json
- Acc-only step 300: results/EXP-006_accuracy_only/eval_ckpt300/eval_results.json
- Acc-only step 500: results/EXP-006_accuracy_only/eval/eval_results.json
- Full step 100: results/EXP-009_full_composite/eval_ckpt100/eval_results.json
- Full step 200: results/EXP-009_full_composite/eval_ckpt200/eval_results.json
- Full step 300: results/EXP-009_full_composite/eval_ckpt300/eval_results.json
- Full step 400: results/EXP-009_full_composite/eval_step400/eval_results.json
- Full step 500: results/EXP-009_full_composite/eval/eval_results.json

