# Phase A 结果记录表

本文件只填写已完成且 provenance 合格的实验。计划值、预期值和旧论文数字不得填入 result cells。

## 1. 数据与代码冻结

| 项目 | 值 | 状态 |
|---|---|---|
| Code commit | TODO | pending |
| Train manifest SHA-256 | TODO | pending |
| Dev manifest SHA-256 | TODO | pending |
| Test manifest SHA-256 | TODO | pending |
| Base model revision | TODO | pending |
| TRL/Transformers/vLLM versions | TODO | pending |
| Canonical result snapshot | TODO | pending |

## 2. Run provenance

| Run ID | Domain | LR | Train seed | Config seed verified | Independent checkpoint | Raw log | Predictions | Status |
|---|---|---:|---:|---|---|---|---|---|
| TODO | Math | 1e-6 | 42 | no | no | no | no | pending |
| TODO | Math | 1e-6 | 123 | no | no | no | no | pending |
| TODO | Math | 1e-6 | 456 | no | no | no | no | pending |
| TODO | Science | 1e-6 | 42 | no | no | no | no | pending |
| TODO | Science | 1e-6 | 123 | no | no | no | no | pending |
| TODO | Science | 1e-6 | 456 | no | no | no | no | pending |
| TODO | Medicine | 1e-6 | 42 | no | no | no | no | pending |
| TODO | Medicine | 1e-6 | 123 | no | no | no | no | pending |
| TODO | Medicine | 1e-6 | 456 | no | no | no | no | pending |
| TODO | Commonsense | 1e-6 | 42 | no | no | no | no | pending |
| TODO | Commonsense | 1e-6 | 123 | no | no | no | no | pending |
| TODO | Commonsense | 1e-6 | 456 | no | no | no | no | pending |

为 2e-5 旧 runs 建立相同记录；只有状态为 `valid` 才允许复用。

## 3. Dev selection

| Domain | Candidate LR | Seed-level dev macro | Aggregate dev macro | Selected | Selection manifest |
|---|---:|---|---:|---|---|
| Math | 1e-6 | TODO | TODO | TODO | TODO |
| Math | 2e-5 | TODO | TODO | TODO | TODO |
| Science | 1e-6 | TODO | TODO | TODO | TODO |
| Science | 2e-5 | TODO | TODO | TODO | TODO |
| Medicine | 1e-6 | TODO | TODO | TODO | TODO |
| Medicine | 2e-5 | TODO | TODO | TODO | TODO |
| Commonsense | 1e-6 | TODO | TODO | TODO | TODO |
| Commonsense | 2e-5 | TODO | TODO | TODO | TODO |

## 4. Final test results

只在 dev selection 冻结后填写。

| Domain / benchmark | Base | RFT | GRPO 1e-6 | GRPO dev-selected | Delta vs base | 95% CI | Seeds |
|---|---:|---:|---:|---:|---:|---|---:|
| Math / GSM8K | TODO | TODO | TODO | TODO | TODO | TODO | 3 |
| Science / ARC-C | TODO | TODO | TODO | TODO | TODO | TODO | 3 |
| Science / ScienceQA | TODO | TODO | TODO | TODO | TODO | TODO | 3 |
| Science / macro | TODO | TODO | TODO | TODO | TODO | TODO | 3 |
| Medicine / MedQA | TODO | TODO | TODO | TODO | TODO | TODO | 3 |
| Commonsense / ARC-E | TODO | TODO | TODO | TODO | TODO | TODO | 3 |
| Commonsense / HellaSwag | TODO | TODO | TODO | TODO | TODO | TODO | 3 |
| Commonsense / macro | TODO | TODO | TODO | TODO | TODO | TODO | 3 |

## 5. MCQ audit

| Domain | Model | Original acc. | Permutation mean±SD | Invalid rate | Manual parser agreement | Pass/Fail |
|---|---|---:|---:|---:|---:|---|
| Science | Base | TODO | TODO | TODO | TODO | TODO |
| Science | GRPO 1e-6 | TODO | TODO | TODO | TODO | TODO |
| Science | GRPO selected | TODO | TODO | TODO | TODO | TODO |
| Medicine | Base | TODO | TODO | TODO | TODO | TODO |
| Medicine | GRPO 1e-6 | TODO | TODO | TODO | TODO | TODO |
| Medicine | GRPO selected | TODO | TODO | TODO | TODO | TODO |
| Commonsense | Base | TODO | TODO | TODO | TODO | TODO |
| Commonsense | GRPO 1e-6 | TODO | TODO | TODO | TODO | TODO |
| Commonsense | GRPO selected | TODO | TODO | TODO | TODO | TODO |

## 6. Final claim gate

根据最终数据逐条勾选：

- [ ] 可以说 conservative GRPO 在多数 domain under-updates；
- [ ] 可以说 dev-selected LR 在 overall macro 上可靠提高；
- [ ] 可以说该效应在所有四个 domain 成立；只有全部成立才勾；
- [ ] 可以说超过 RFT；
- [ ] 可以说超过真正 external teacher SFT；仅在补做该 baseline 后勾；
- [ ] 可以说跨模型 LR sensitivity；仅在每个模型都有 conservative/high-LR 对照后勾；
- [ ] 可以展示真实 KL/update trajectory；仅在日志完整后勾；

没有勾选的 claim 不得写入 abstract、contribution bullets 或 figure caption。
