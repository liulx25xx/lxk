# Domain Slice Summary

Generated from local artifacts only. Values are percentages.
Multi-run settings are formatted as mean +/- sample standard deviation across seeds.

Domain mapping:
- Chat: alpacaeval-easy, alpacaeval-hard, alpacaeval-length, mt-bench-easy, mt-bench-med, mt-bench-hard, llmbar-natural
- Reasoning: math-prm
- Safety: xstest-should-respond, xstest-should-refuse, refusals-offensive, refusals-dangerous, donotanswer
- Code: hep-cpp, hep-go, hep-java, hep-js, hep-python, hep-rust
- Adversarial: llmbar-adver-GPTInst, llmbar-adver-GPTOut, llmbar-adver-manual, llmbar-adver-neighbor

| Setting | Metric | Chat (n=87) | Reas. (n=69) | Safe (n=115) | Code (n=135) | Adv. (n=43) |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Orig Acc | 96.6 | 78.3 | 80.9 | 85.2 | 32.6 |
| Baseline | Con | 93.1 | 63.8 | 90.4 | 78.5 | 72.1 |
| Baseline | First-pos | 51.1 | 60.1 | 48.3 | 45.2 | 44.2 |
| GRPO unbalanced | Orig Acc | 99.4 +/- 1.1 | 98.6 +/- 0.0 | 94.1 +/- 1.3 | 96.3 +/- 1.4 | 75.0 +/- 9.2 |
| GRPO unbalanced | Con | 64.9 +/- 9.0 | 19.6 +/- 3.4 | 73.7 +/- 10.3 | 64.4 +/- 8.7 | 51.2 +/- 17.5 |
| GRPO unbalanced | First-pos | 67.1 +/- 4.7 | 90.2 +/- 1.7 | 62.5 +/- 5.6 | 63.0 +/- 3.0 | 73.3 +/- 9.4 |
| GRPO balanced | Orig Acc | 94.3 +/- 0.8 | 85.5 +/- 2.0 | 82.6 +/- 2.5 | 91.3 +/- 1.1 | 38.6 +/- 3.1 |
| GRPO balanced | Con | 91.5 +/- 0.6 | 70.7 +/- 7.2 | 88.7 +/- 2.8 | 84.1 +/- 0.8 | 77.2 +/- 5.6 |
| GRPO balanced | First-pos | 50.6 +/- 0.6 | 58.6 +/- 4.1 | 48.8 +/- 1.3 | 47.9 +/- 0.7 | 44.4 +/- 2.1 |

