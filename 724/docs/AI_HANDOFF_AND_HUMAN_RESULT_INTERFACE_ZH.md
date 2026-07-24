# 下一位 AI 交接与人工结果接口

版本：`handoff-v1`

日期：2026-07-24
项目：AAAI-27 GRPO calibration / parser audit

## 1. 交接目标

下一位 AI 的首要任务不是重新训练模型，而是接收 24 位标注员返回的人工 parser audit，完成数据校验、交叉复核、仲裁、解除盲化、统计分析和论文结论更新。

只有人工结果表明仍有无法由格式/parser 解释的残余增益时，才考虑启动后续 zero-variance filtering GPU 实验。

## 2. 硬约束

### GPU 节点

任何后续 GPU 任务只能使用：

- hostname：`os-node-created-r24fv`
- IP：`10.100.41.246`
- GPU：该节点上的 8 张卡

禁止在其他 Ray 节点提交训练或评测。远程任务必须同时使用 Ray node resource 约束和运行时 hostname/IP 检查。

### 数据与结论

- 不得根据 test 或人工结果重新选择 LR/checkpoint。
- 不得修改 frozen raw predictions。
- 不得把 200/domain 的人工子样本描述为整套 test 均经过人工修正。
- 不得在所有标注返回前把已完成答案共享给其他标注员。
- 不得上传 `private_do_not_share/` 给标注员。
- 不得引用旧 tracker 中的 universal/high-LR-always-wins 结论。

## 3. 当前实验状态

当前没有训练或评测任务运行，GPU 已释放。

### Canonical 主实验：4 domains

主实验为 Qwen2.5-7B-Instruct、LoRA rank 64、`1e-6` vs `2e-5`、3 个独立训练 seeds、dev-only selection：

| Domain | Benchmark | Dev-selected LR | Frozen test 结论 |
|---|---|---:|---|
| Math | GSM8K | 2e-5 | 90.47%，相对 Base +7.53pp |
| Science | ARC-Challenge | 1e-6 | 88.92%，相对 Base +1.65pp |
| Medicine | MedQA | 2e-5 | generation 58.80%，相对 Base +5.69pp |
| Commonsense | ARC-Easy | 2e-5 | +0.31pp，CI 跨 0 |

Canonical 入口：

- `results/aaai27_canonical_v2/selection_manifest.json`
- `results/aaai27_canonical_v2/main_results.csv`
- `results/aaai27_canonical_v2/final_statistics.json`
- `results/aaai27_canonical_v2/robustness_summary.json`
- `docs/aaai27/02_RESULT_TRACKER.md`

### 人工 parser audit：3 MCQ domains

人工审核只覆盖 Science、Medicine、Commonsense。Math 是数值 verifier，不属于 A/B/C/D parser audit。

### Rank / Full-FT 补实验：2 domains

Rank-16、Rank-128 和 Full-FT 只覆盖 Math、Medicine，均已完成：

- `results/aaai27_rank_ablation_v1/combined_summary.json`
- `results/aaai27_fullft_calibration_v1/combined_summary.json`

Full-FT 使用 Adafactor，而 canonical LoRA 使用默认 AdamW。Full-FT 可作为实用鲁棒性对照，不能写成只改变参数化的纯因果消融。

## 4. GitHub 公开包

仓库：`https://github.com/liulx25xx/lxk`

分支：`main`
目录：`724/`

公开包结构：

```text
724/
  README.md
  INSTRUCTIONS_ZH.md
  CHECKSUMS.sha256
  packages/
    annotator_01.zip
    ...
    annotator_24.zip
  docs/
  schemas/
```

每个 ZIP 含：

- `primary_100.jsonl`
- `qc_overlap_10.jsonl`
- `INSTRUCTIONS_ZH.md`

公开包不含 gold、模型身份、parser、correct、seed、原始 item ID 或私有映射。

## 5. 本地私有接口

项目相对路径：

```text
human_annotation_parser_audit/release_v1/private_do_not_share/
  mapping.jsonl
  generation_manifest.json
  annotator_assignment.csv
```

关键文件：

- `mapping.jsonl`：`annotation_id` 到 domain、item、model、gold 和自动 parser 的唯一 join key。
- `generation_manifest.json`：输入 hash、随机 seed、packet hash 和数量不变量。
- `annotator_assignment.csv`：匿名编号与真实标注员的协调表；不得进入公开分析数据。

如果新环境只从 GitHub clone，必须由项目负责人通过私有渠道同步该目录。不能从公开包反推或重建 private mapping。

## 6. 人工结果回传目录

所有 24 人返回后，在本地建立：

```text
human_annotation_parser_audit/returns/v1/
  raw/
    annotator_01/
      primary_100.jsonl
      qc_overlap_10.jsonl
    ...
    annotator_24/
      primary_100.jsonl
      qc_overlap_10.jsonl
  validated/
  adjudication/
  reports/
```

`raw/` 必须保存标注员原始返回文件，只读保留。任何格式修复写入 `validated/`，不得覆盖 raw。

不要边回收边把人工答案 push 到 GitHub。等 24 人全部返回、hash 冻结并确认不会继续互相影响后，再由负责人决定上传范围。

## 7. 单条返回记录合同

返回 JSONL 必须保持原始结构：

```json
{
  "annotation_id": "PA-XXXXXXXXXXXX",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "prediction": "...",
  "manual_answer": "B",
  "status": "clear",
  "notes": ""
}
```

允许状态：

- `clear`：`manual_answer` 必须为 A-E；
- `no_answer`：`manual_answer` 必须为 null；
- `ambiguous`：`manual_answer` 必须为 null；
- `unreadable`：`manual_answer` 必须为 null。

标注员只允许修改 `manual_answer`、`status`、`notes`。其他字段必须与发出的 packet 逐字一致。

Schema：

- `human_annotation_parser_audit/schemas/completed_annotation.schema.json`
- GitHub 对应：`724/schemas/completed_annotation.schema.json`

## 8. 回收数量不变量

在任何统计前必须满足：

| 项目 | 期望值 |
|---|---:|
| 标注员 | 24 |
| Primary files | 24 |
| QC files | 24 |
| Primary judgments | 2400 |
| Primary unique annotation IDs | 2400 |
| QC judgments | 240 |
| QC unique annotation IDs | 240 |
| QC IDs 是否为 primary 子集 | 全部是 |
| 每位标注员 primary | 100 |
| 每位标注员 QC | 10 |

每个 primary annotator 的私有条件分配应为：

- Base 25；
- GRPO 75；
- 三个 domain 为 33、33、34；
- 同一道题不会在同一人的 primary 中重复；
- QC reviewer 在 primary 中没有看过同一道题。

## 9. 回收校验顺序

1. 对 raw 返回文件计算 SHA-256。
2. 验证 JSONL 每行可解析。
3. 验证 annotation ID 数量、唯一性与所属 packet。
4. 对比发出 packet，确认 question/options/prediction 未变。
5. 验证 manual answer 与 status 的约束。
6. 记录缺失、重复、非法 label 和被修改的 immutable 字段。
7. 校验失败的 packet 退回原标注员修复；协调者不得猜测或代填。
8. 只有所有 packet 通过后，才进入 agreement 和解除盲化。

## 10. 交叉复核与一致率

同一个 annotation ID 的 primary 与 QC 构成一对独立判断。统一映射到 8 类标签：

```text
A, B, C, D, E, NO_ANSWER, AMBIGUOUS, UNREADABLE
```

至少报告：

- raw agreement；
- Cohen's kappa；
- 8×8 disagreement matrix；
- 各 label 数量；
- 各 domain 的 agreement；
- 240 条 overlap 的条件/domain 构成。

推荐质量目标是 raw agreement ≥95%、kappa ≥0.9，但不能只根据阈值删除分歧。所有分歧必须进入仲裁。

由于多数输出可能是 clear 单字母，kappa 会受 prevalence 影响，因此必须与 raw agreement 和分歧矩阵一起解释。

## 11. 仲裁接口

以下记录进入盲化仲裁：

- primary 与 QC 不一致；
- 任一方为 ambiguous/unreadable；
- notes 表示规范未覆盖；
- 解除盲化后，最终人工答案与自动 parser 不一致。

建议仲裁记录结构：

```json
{
  "annotation_id": "PA-XXXXXXXXXXXX",
  "primary_annotator_id": "annotator_01",
  "qc_annotator_id": "annotator_02",
  "primary_label": "B",
  "qc_label": "NO_ANSWER",
  "final_manual_answer": "B",
  "final_status": "clear",
  "adjudicator_id": "adjudicator_01",
  "reason": "The first line explicitly selects B."
}
```

仲裁文件写入：

`human_annotation_parser_audit/returns/v1/adjudication/adjudicated_annotations.jsonl`

不得覆盖两个原始判断。

## 12. 解除盲化与私有 join

所有 primary/QC 和第一轮仲裁冻结后，才使用 `mapping.jsonl` join。

私有 joined 文件至少包含：

- annotation ID；
- domain/item ID；
- model tag 与 seed；
- gold；
- permissive parser 与 strict parser；
- primary/QC/manual/adjudicated label；
- parser agreement；
- automatic correctness；
- human correctness；
- error category。

错误分类：

- `false_extraction`：人工 no-answer，parser 给字母；
- `missed_extraction`：人工有清晰答案，parser 为 null；
- `wrong_letter`：人工字母与 parser 字母不同；
- `agreement`：人工与 parser 一致；
- `excluded_ambiguous`：ambiguous/unreadable，不进入主要 agreement 分母。

该 joined 文件包含解除盲化信息，默认只保存在本地 private results，不自动 push 到 GitHub。

## 13. 统计接口

### Parser audit

分别按 domain × condition 报告：

- eligible n；
- parser agreement n/%；
- false/missed/wrong-letter 数量；
- ambiguous/unreadable 数量；
- Base 与 GRPO parser error 差值及置信区间。

### Human-corrected accuracy

人工修正只针对随机抽取的 200 items/domain：

- Base：每 domain n=200；
- 每个 GRPO training seed：每 domain n=200；
- 先按 seed 单独算 accuracy，再跨 seed 汇总；
- 同一道题的 Base/GRPO 必须配对；
- 不把 3 个 seed × 200 当成 600 个独立 test items；
- 使用 seed/item hierarchical bootstrap 或明确的 paired bootstrap。

必须在同一人工审计子集上比较：

- automatic subset accuracy；
- human-corrected subset accuracy；
- selected-minus-base 的变化；
- 95% CI。

不要把子样本修正数字替换成整套 frozen test accuracy。

## 14. 结果输出合同

建议生成：

```text
human_annotation_parser_audit/returns/v1/reports/
  validation_report.json
  agreement_summary.json
  disagreement_matrix.csv
  adjudication_summary.json
  parser_error_by_domain_condition.csv
  corrected_accuracy_by_seed.csv
  corrected_accuracy_bootstrap.json
  public_summary.json
  public_summary.md
```

`public_summary` 不包含标注员真实身份，不包含未聚合的私有 model mapping。

论文表格模板：

- `human_annotation_parser_audit/RESULT_REPORT_TEMPLATE_ZH.md`
- GitHub：`724/docs/RESULT_REPORT_TEMPLATE_ZH.md`

## 15. 论文决策规则

### 人工修正前后增益基本不变

可以写：MCQ 主结果通过 blinded parser audit，自动解析不足以解释增益。

### 增益缩小但仍存在

最适合写成：GRPO 同时改善 answer-format compliance 与任务表现；人工修正量化了两部分贡献。

### Medicine 增益接近消失

结合 parser-free logprob，论文应强调 format/capability dissociation，不得继续把 generation 增益直接称为能力提升。

### 人工一致率低

人工结果不能直接进入论文。先完成仲裁；若规则仍不稳定，重新标注相关子集。

最可信的整体主线不是“所有 domain 都提升”，而是：Math 稳健提升、Science 小幅提升、Medicine 可能格式主导、Commonsense ceiling-limited null，支持 domain-dependent calibration。

## 16. 后续 GPU 决策

人工结果分析前不启动新 GPU 实验。

如果人工修正后仍有明显残余增益，优先补 P0-D：

```text
Science + Medicine
× 1e-6 + zero-variance filtering/resampling
× seeds 42, 123, 456
= 6 个新 runs
```

现有普通 `1e-6` 和 `2e-5` 作为对照复用。Filtering 必须补采样，使生成 token、有效 group/update 预算匹配。只删除 zero-variance group 而不补足预算不是公平实验。

如果 Medicine 人工修正后增益消失，先重写论文主线，不要机械启动 filtering。当前不建议再跑更多 rank、零散 LR 或 Full-FT 网格。

## 17. 下一位 AI 的启动检查表

1. 阅读本文件和 `docs/aaai27/00_START_HERE.md`。
2. 确认当前没有实验进程，不要重复启动已完成队列。
3. 核对 GitHub `724/` 与本地 release hash。
4. 确认 24 位标注员是否全部返回。
5. 在解除盲化前完成 raw 文件冻结、schema 和 immutable-field 校验。
6. 计算 overlap agreement 并完成第一轮仲裁。
7. 使用私有 mapping join，生成 parser error 与 corrected subset accuracy。
8. 依据结果更新论文主张。
9. 只有满足决策条件时，才设计并启动 6-run filtering 实验。
10. 如启动 GPU，严格锁定 `r24fv / 10.100.41.246`。

## 18. 禁止事项

- 不根据人工结果重选 test checkpoint/LR；
- 不只分析自动 parser disagreement；
- 不删除 no-answer/ambiguous 样本以美化 agreement；
- 不把标注员当成独立训练 seed；
- 不把 200-item audit 写成 full-test manual evaluation；
- 不提前向标注员公开自动答案、其他 packet 或已回收标注；
- 不上传人员真实身份；
- 不覆盖 raw、mapping、manifest 或 frozen canonical artifacts；
- 不在 r24fv 以外的节点运行 GPU 任务。
