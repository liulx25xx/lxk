# 724 MCQ Parser 人工标注

本目录是 `parser-audit-v1` 的公开盲标分发包，共覆盖 600 道随机抽样题的 2400 个模型输出。

## 标注员领取方式

1. 向协调者确认你的匿名编号 `annotator_01` 至 `annotator_24`。
2. 只下载 `packages/` 中与你编号一致的 ZIP。
3. 阅读 ZIP 内的 `INSTRUCTIONS_ZH.md`。
4. 完成 `primary_100.jsonl` 中的 100 条主标。
5. 独立完成 `qc_overlap_10.jsonl` 中的 10 条交叉复核。
6. 保持文件名不变，将两个完成后的 JSONL 返回协调者。

每人共填写 110 条。不要查看、讨论或合并其他标注员的 packet。

## 标注目标

任务是判断模型原始输出最终表达了哪个选项，不是重新解题，也不是判断答案知识上是否正确。

正式记录只填写：

- `manual_answer`
- `status`
- `notes`

不得修改 `annotation_id`、`question`、`options` 或 `prediction`。

## 目录

- `INSTRUCTIONS_ZH.md`：可直接阅读的精简说明。
- `packages/`：24 个独立标注包。
- `docs/`：完整标注、盲化、质控和报告规范。
- `schemas/`：盲标输入与完成标注的 JSON schema。
- `CHECKSUMS.sha256`：本目录公开文件的 SHA-256。

本目录不包含模型身份、gold answer、自动 parser 结果、自动正确性或私有映射。
