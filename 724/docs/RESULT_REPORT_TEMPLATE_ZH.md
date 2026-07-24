# 人工 Parser Audit 结果报告模板

状态：待人工标注完成后由统计脚本填写。所有比例同时保留分子、分母，不只写百分比。

## 1. Protocol

- Protocol version：`parser-audit-v1`
- Frozen source sample hashes：`TBD`
- Blinding/randomization seed：`TBD`
- Annotation dates：`TBD`
- Number of annotators：`TBD`
- Annotation design：`full double annotation / primary + partial review`
- Adjudicator：`TBD`
- Total source questions：600
- Total model outputs：2400
- Returned valid annotations：`TBD / expected`

## 2. 标注覆盖与一致性

| 项目 | 数量/结果 |
|---|---:|
| 双人重叠输出数 | TBD |
| 双人重叠比例 | TBD |
| Raw agreement | TBD |
| Cohen's kappa | TBD |
| 初始分歧数 | TBD |
| 仲裁数 | TBD |
| Clear | TBD |
| No answer | TBD |
| Ambiguous | TBD |
| Unreadable | TBD |

若不是全量双标，必须明确 kappa 只基于哪个预先确定的重叠子集。

## 3. Parser agreement

| Domain | Condition | Eligible n | Agreement n (%) | False extraction | Missed extraction | Wrong letter | Ambiguous/unreadable |
|---|---|---:|---:|---:|---:|---:|---:|
| Science | Base | TBD | TBD | TBD | TBD | TBD | TBD |
| Science | GRPO | TBD | TBD | TBD | TBD | TBD | TBD |
| Medicine | Base | TBD | TBD | TBD | TBD | TBD | TBD |
| Medicine | GRPO | TBD | TBD | TBD | TBD | TBD | TBD |
| Commonsense | Base | TBD | TBD | TBD | TBD | TBD | TBD |
| Commonsense | GRPO | TBD | TBD | TBD | TBD | TBD | TBD |

GRPO 行需先按三个 seed 分别统计，再给聚合结果，不能把同一道题的三个输出误当成三个独立 test items 做显著性检验。

## 4. 人工修正前后 Accuracy

| Domain | Condition | Automatic accuracy | Human-corrected accuracy | Difference | 95% CI |
|---|---|---:|---:|---:|---:|
| Science | Base | TBD | TBD | TBD | TBD |
| Science | GRPO selected | TBD | TBD | TBD | TBD |
| Medicine | Base | TBD | TBD | TBD | TBD |
| Medicine | GRPO selected | TBD | TBD | TBD | TBD |
| Commonsense | Base | TBD | TBD | TBD | TBD |
| Commonsense | GRPO selected | TBD | TBD | TBD | TBD |

## 5. 主结论稳健性

- 自动 parser 下 selected-minus-base：`TBD`
- 人工修正后 selected-minus-base：`TBD`
- 方向是否变化：`TBD`
- 置信区间是否仍排除 0：`TBD`
- Base 与 GRPO 的 parser error 是否显著不同：`TBD`
- parser/format 差异能否解释 headline gain：`TBD`

## 6. 论文可用表述

根据结果只选一个版本，并填入数字：

### Audit 通过

> A blinded manual audit of 200 randomly sampled items per MCQ domain found parser agreement of TBD% for the base model and TBD% for dev-selected GRPO. Recomputing accuracy from human-extracted answers changed the selected-minus-base effect by TBD percentage points and did not alter the conclusion.

### Parser 有偏但不足以解释结果

> Manual auditing identified a higher parser error rate for the base model (TBD%) than for GRPO (TBD%). Human correction reduced the apparent gain from TBD to TBD points, but the remaining effect was TBD.

### 结果主要由格式/parser 解释

> Human correction substantially reduced the automatic-score gain from TBD to TBD points. We therefore interpret the observed change primarily as improved answer formatting/selection rather than evidence of increased task capability.

## 7. Artifact 清单

- Blinded packets and hashes：`TBD`
- Raw annotator files：`TBD`
- Validation report：`TBD`
- Agreement report：`TBD`
- Adjudication file：`TBD`
- Private mapping hash：`TBD`
- Final per-output audit file：`TBD`
- Aggregation script/version：`TBD`
