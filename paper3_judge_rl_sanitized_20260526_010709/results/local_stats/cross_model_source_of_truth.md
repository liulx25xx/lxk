# Cross-Model Source of Truth

Generated from local artifacts only. Multi-run values are mean +/- sample standard deviation.
Rows marked `metrics_json` are sourced from stored metrics because the full eval_results artifact is unavailable or partial.

## Qwen2.5-7B

| Method | Data | Runs | Acc | Con | Parse fail | Source |
|---|---|---:|---:|---:|---:|---|
| Baseline | --- | 1 | 80.2 | 81.5 | 0.0 | results/baseline_qwen7b/eval_results.json |
| SFT | unbal | 1 | 100.0 | 0.0 | 0.0 | results/SFT_unbalanced/eval/eval_results.json |
| DPO | unbal | 1 | 94.2 | 54.3 | 0.0 | results/DPO_unbalanced/eval/eval_results.json |
| GRPO | unbal | 4 | 94.7 +/- 0.9 | 58.7 +/- 9.1 | 0.0 +/- 0.0 | results/EXP-009_full_composite/eval/eval_results.json<br>results/EXP-009_full_composite_s2/eval/eval_results.json<br>results/EXP-009_full_composite_s3/eval/eval_results.json<br>results/EXP-009_full_s4/eval/eval_results.json |
| SFT | bal | 1 | 91.3 | 87.1 | 0.0 | results/SFT_balanced/eval/eval_results.json |
| GRPO | bal | 5 | 83.7 +/- 1.1 | 84.0 +/- 0.8 | 0.0 +/- 0.0 | results/EXP-009b_full_balanced/eval/eval_results.json<br>results/EXP-009b_full_balanced_s2/eval/eval_results.json<br>results/EXP-009b_full_balanced_s3/eval/eval_results.json<br>results/balanced_full_s4/eval/eval_results.json<br>results/balanced_full_s5/eval/eval_results.json |

## Qwen3-8B

| Method | Data | Runs | Acc | Con | Parse fail | Source |
|---|---|---:|---:|---:|---:|---|
| Baseline | --- | 1 | 86.0 | 80.4 | 2.3 | results/baseline_qwen3_8b/eval/eval_results.json |
| SFT | unbal | 2 | 100.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | results/SFT_qwen3_8b_unbalanced/eval/eval_results.json<br>results/SFT_qwen3_8b_unbal_s2/eval/eval_results.json |
| DPO | unbal | 1 | 89.3 | 81.3 | 0.0 | results/DPO_qwen3_8b_unbalanced/eval/eval_results.json |
| GRPO | unbal | 3 | 88.7 +/- 0.3 | 80.0 +/- 1.0 | 0.4 +/- 0.6 | results/GRPO_qwen3_8b_unbalanced/eval/eval_results.json<br>results/GRPO_qwen3_8b_unbalanced_s2/eval/eval_results.json<br>results/GRPO_qwen3_8b_unbal_s3/eval/eval_results.json |
| SFT | bal | 2 | 88.5 +/- 1.1 | 89.1 +/- 0.3 | 0.0 +/- 0.0 | results/SFT_qwen3_8b_balanced/eval/eval_results.json<br>results/SFT_qwen3_8b_balanced_s2/eval/eval_results.json |
| DPO | bal | 1 | 88.6 | 85.3 | 0.0 | results/DPO_qwen3_8b_balanced/eval/eval_results.json |
| GRPO | bal | 2 | 87.4 +/- 0.8 | 82.2 +/- 0.6 | 0.0 +/- 0.0 | results/GRPO_qwen3_8b_balanced/eval/eval_results.json<br>results/GRPO_qwen3_8b_balanced_s2/eval/eval_results.json |

## Mistral-7B

| Method | Data | Runs | Acc | Con | Parse fail | Source |
|---|---|---:|---:|---:|---:|---|
| Baseline | --- | 1 | 65.7 | 64.8 | 0.1 | results/baseline_mistral7b/eval/metrics.json |
| SFT | unbal | 1 | 100.0 | 0.0 | 0.0 | results/SFT_mistral7b_unbal/eval/eval_results.json |
| SFT | bal | 1 | 92.2 | 84.9 | 0.0 | results/SFT_mistral7b_balanced/eval/eval_results.json |
| GRPO | unbal | 3 | 98.4 +/- 0.4 | 23.0 +/- 3.6 | 0.0 +/- 0.0 | results/GRPO_mistral7b_unbal/eval/eval_results.json<br>results/GRPO_mistral7b_unbal_s2/eval/eval_results.json<br>results/GRPO_mistral7b_unbal_s3/eval/eval_results.json |
| GRPO | bal | 2 | 81.1 +/- 0.9 | 68.3 +/- 3.6 | 0.0 +/- 0.0 | results/GRPO_mistral7b_balanced/eval/eval_results.json<br>results/GRPO_mistral7b_balanced_s2/eval/eval_results.json |

## Notes

- Qwen3-8B Baseline ---: Canonical Qwen3 baseline uses disable_thinking and low parse failure; excludes legacy top-level baseline_qwen3_8b/eval_results.json.
- Mistral-7B Baseline ---: The local eval_results artifact is a partial 80-sample file; metrics.json reports the canonical 449-sample run used in the paper.

