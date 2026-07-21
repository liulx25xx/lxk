# Prompt and Label Control Summary

Generated from local artifacts only. Values are percentages.
Reported rows are the interpretable prompt/label controls used in the appendix table.
Qwen3 thinking-enabled diagnostics are retained here because their high parse-failure rates explain why the paper reports disable-thinking controls.

## Reported Controls

| Model | Control | Eval mode | Acc | Swap Acc | Con | First-pos | Orig-first | Parse fail | Source note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-7B | A / B | standard prompt | 94.0 | 57.2 | 61.7 | 67.4 | 94.0 | 0.0 | Standard unbalanced-GRPO full-composite run used as the Qwen2.5 label-control anchor. |
| Qwen2.5-7B | Anti-prompt | ordering-randomized instruction | 93.8 | 58.4 | 63.9 | 66.3 | 93.8 | 0.0 | Prompt-level mitigation: instructs the judge to ignore order and notes that ordering is random. |
| Qwen2.5-7B | 1 / 2 | numeric labels | 94.7 | 56.3 | 60.8 | 67.3 | 94.7 | 0.0 | Alternative label vocabulary for the same unbalanced-GRPO checkpoint. |
| Qwen2.5-7B | Left / Right | spatial labels | 94.2 | 55.5 | 59.9 | 67.9 | 94.2 | 0.0 | Alternative label vocabulary for the same unbalanced-GRPO checkpoint. |
| Qwen3-8B | A / B | standard prompt | 88.4 | 73.5 | 78.8 | 55.2 | 88.4 | 1.1 | Standard Qwen3 unbalanced accuracy-reward run used as the label-control anchor. |
| Qwen3-8B | Anti-prompt | ordering-randomized instruction, thinking disabled | 88.0 | 73.9 | 80.2 | 55.7 | 88.0 | 0.7 | Reported anti-prompt control with thinking disabled to keep parse failures low. |
| Qwen3-8B | 1 / 2 | numeric labels, thinking disabled | 88.4 | 75.5 | 81.7 | 54.9 | 88.4 | 1.2 | Reported numeric-label control with thinking disabled to keep parse failures low. |
| Qwen3-8B | Left / Right | spatial labels, thinking disabled | 85.7 | 78.2 | 83.7 | 52.7 | 85.7 | 1.0 | Reported spatial-label control with thinking disabled to keep parse failures low. |
| Mistral-7B | A / B | standard prompt | 98.9 | 18.9 | 20.0 | 89.9 | 98.9 | 0.0 | Standard Mistral unbalanced-GRPO run used as the label-control anchor. |
| Mistral-7B | Anti-prompt | ordering-randomized instruction | 98.7 | 17.6 | 18.9 | 90.5 | 98.7 | 0.0 | Prompt-level mitigation: instructs the judge to ignore order and notes that ordering is random. |
| Mistral-7B | 1 / 2 | numeric labels | 99.6 | 11.6 | 12.0 | 93.9 | 99.6 | 0.0 | Alternative label vocabulary for the same unbalanced-GRPO checkpoint. |
| Mistral-7B | Left / Right | spatial labels | 99.1 | 14.7 | 15.6 | 92.2 | 99.1 | 0.0 | Alternative label vocabulary for the same unbalanced-GRPO checkpoint. |

## Diagnostic Qwen3 Thinking-On Controls

| Model | Control | Eval mode | Acc | Swap Acc | Con | First-pos | Orig-first | Parse fail | Source note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-8B | Anti-prompt (thinking on) | ordering-randomized instruction, thinking enabled | 56.8 | 54.3 | 77.1 | 31.6 | 56.8 | 39.0 | Diagnostic only: high parse-failure rate makes the control hard to interpret. |
| Qwen3-8B | 1 / 2 (thinking on) | numeric labels, thinking enabled | 54.3 | 53.5 | 80.8 | 29.1 | 54.3 | 42.3 | Diagnostic only: high parse-failure rate makes the control hard to interpret. |
| Qwen3-8B | Left / Right (thinking on) | spatial labels, thinking enabled | 57.5 | 54.1 | 78.2 | 31.1 | 57.5 | 39.6 | Diagnostic only: high parse-failure rate makes the control hard to interpret. |

## Sources

- Qwen2.5-7B A / B: results/EXP-009_full_composite/eval/eval_results.json
- Qwen2.5-7B Anti-prompt: results/GRPO_antiprompt_eval/eval_results.json
- Qwen2.5-7B 1 / 2: results/label_variant_numeric/eval_results.json
- Qwen2.5-7B Left / Right: results/label_variant_leftright/eval_results.json
- Qwen3-8B A / B: results/GRPO_qwen3_8b_unbalanced/eval/eval_results.json
- Qwen3-8B Anti-prompt: results/antiprompt_qwen3_nothink/eval_results.json
- Qwen3-8B 1 / 2: results/label_variant_numeric_qwen3_nothink/eval_results.json
- Qwen3-8B Left / Right: results/label_variant_leftright_qwen3_nothink/eval_results.json
- Qwen3-8B Anti-prompt (thinking on): results/antiprompt_qwen3/eval_results.json
- Qwen3-8B 1 / 2 (thinking on): results/label_variant_numeric_qwen3/eval_results.json
- Qwen3-8B Left / Right (thinking on): results/label_variant_leftright_qwen3/eval_results.json
- Mistral-7B A / B: results/GRPO_mistral7b_unbal/eval/eval_results.json
- Mistral-7B Anti-prompt: results/antiprompt_mistral/eval_results.json
- Mistral-7B 1 / 2: results/label_variant_numeric_mistral/eval_results.json
- Mistral-7B Left / Right: results/label_variant_leftright_mistral/eval_results.json

