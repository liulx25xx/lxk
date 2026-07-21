# Uncertainty Checks

Generated from local artifacts only. Bootstrap intervals use 10,000 resamples with seed 20260709.
Differences are `after - before` in percentage points.
Paired-instance checks resample matched test examples; seed-mean checks resample available training seeds.

## Paired Instance Bootstrap

| Comparison | Metric | Before | After | Diff [95% CI] | Pairs |
|---|---|---:|---:|---:|---:|
| Qwen2.5 Baseline -> SFT unbalanced | Orig Acc | 80.2 | 100.0 | 19.8 [16.3, 23.6] | 449 |
| Qwen2.5 Baseline -> SFT unbalanced | Swap Acc | 75.3 | 0.0 | -75.3 [-79.3, -71.3] | 449 |
| Qwen2.5 Baseline -> SFT unbalanced | Avg Acc | 77.7 | 50.0 | -27.7 [-31.0, -24.4] | 449 |
| Qwen2.5 Baseline -> SFT unbalanced | Con | 81.5 | 0.0 | -81.5 [-85.1, -78.0] | 449 |
| Qwen2.5 Baseline -> SFT unbalanced | First-pos | 49.3 | 100.0 | 50.7 [48.8, 52.6] | 449 |
| Qwen2.5 SFT unbalanced -> SFT balanced | Orig Acc | 100.0 | 91.3 | -8.7 [-11.4, -6.2] | 449 |
| Qwen2.5 SFT unbalanced -> SFT balanced | Swap Acc | 0.0 | 89.1 | 89.1 [86.2, 91.8] | 449 |
| Qwen2.5 SFT unbalanced -> SFT balanced | Avg Acc | 50.0 | 90.2 | 40.2 [38.0, 42.3] | 449 |
| Qwen2.5 SFT unbalanced -> SFT balanced | Con | 0.0 | 87.1 | 87.1 [84.0, 90.2] | 449 |
| Qwen2.5 SFT unbalanced -> SFT balanced | First-pos | 100.0 | 51.1 | -48.9 [-50.6, -47.2] | 449 |
| Qwen2.5 Baseline -> SFT balanced | Orig Acc | 80.2 | 91.3 | 11.1 [7.6, 14.7] | 449 |
| Qwen2.5 Baseline -> SFT balanced | Swap Acc | 75.3 | 89.1 | 13.8 [9.8, 17.8] | 449 |
| Qwen2.5 Baseline -> SFT balanced | Avg Acc | 77.7 | 90.2 | 12.5 [9.5, 15.6] | 449 |
| Qwen2.5 Baseline -> SFT balanced | Con | 81.5 | 87.1 | 5.6 [1.1, 10.0] | 449 |
| Qwen2.5 Baseline -> SFT balanced | First-pos | 49.3 | 51.1 | 1.8 [-0.4, 4.1] | 449 |

## Seed-Mean Bootstrap

| Comparison | Metric | Before | After | Diff [95% CI] | Runs |
|---|---|---:|---:|---:|---:|
| Qwen2.5 GRPO full unbalanced -> balanced | Orig Acc | 94.7 | 83.7 | -11.0 [-12.2, -9.9] | 4 -> 5 |
| Qwen2.5 GRPO full unbalanced -> balanced | Avg Acc | 74.7 | 82.6 | 7.8 [5.3, 11.6] | 4 -> 5 |
| Qwen2.5 GRPO full unbalanced -> balanced | Con | 58.7 | 84.0 | 25.3 [19.9, 34.4] | 4 -> 5 |
| Qwen2.5 GRPO full unbalanced -> balanced | First-pos | 68.8 | 50.0 | -18.9 [-23.2, -16.0] | 4 -> 5 |
| Qwen3 GRPO full unbalanced -> balanced | Orig Acc | 88.7 | 87.4 | -1.3 [-2.1, -0.5] | 3 -> 2 |
| Qwen3 GRPO full unbalanced -> balanced | Avg Acc | 81.3 | 81.5 | 0.2 [-0.4, 0.8] | 3 -> 2 |
| Qwen3 GRPO full unbalanced -> balanced | Con | 80.0 | 82.2 | 2.2 [1.1, 3.3] | 3 -> 2 |
| Qwen3 GRPO full unbalanced -> balanced | First-pos | 55.6 | 54.7 | -0.9 [-1.3, -0.6] | 3 -> 2 |
| Mistral GRPO full unbalanced -> balanced | Orig Acc | 98.4 | 81.1 | -17.4 [-18.3, -16.5] | 3 -> 2 |
| Mistral GRPO full unbalanced -> balanced | Avg Acc | 60.0 | 72.8 | 12.8 [9.1, 16.4] | 3 -> 2 |
| Mistral GRPO full unbalanced -> balanced | Con | 23.0 | 68.3 | 45.2 [40.4, 50.1] | 3 -> 2 |
| Mistral GRPO full unbalanced -> balanced | First-pos | 88.3 | 57.1 | -31.3 [-34.1, -28.5] | 3 -> 2 |
