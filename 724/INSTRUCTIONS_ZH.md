# MCQ 输出解析人工标注说明

规范版本：parser-audit-v1

## 你的任务

根据 `prediction` 判断模型最终明确选择了哪个选项。不要重新解题，不要判断模型答案是否正确，也不要根据自己的知识修正模型。

每条记录只填写：

- `manual_answer`：明确答案填 `"A"`、`"B"`、`"C"`、`"D"` 或 `"E"`；无法得到唯一答案时填 `null`。
- `status`：填 `"clear"`、`"no_answer"`、`"ambiguous"` 或 `"unreadable"`。
- `notes`：通常保持空字符串；只有边界情况才写一句简短说明。

不要修改 `annotation_id`、`question`、`options` 或 `prediction`。

## 四种状态

- `clear`：能确定唯一选项，`manual_answer` 必须为 A-E。
- `no_answer`：没有提交最终选项，`manual_answer` 必须为 null。
- `ambiguous`：多个冲突选项且没有最终消歧，`manual_answer` 必须为 null。
- `unreadable`：输出损坏或无法读取，`manual_answer` 必须为 null。

## 判定规则

1. `B`、`B)`、`(B)`、`Answer: B`、`The answer is B`、`\\boxed{B}` 都是 B。
2. 只写了某个选项的完整文本，并且与唯一选项精确对应，也标对应字母。
3. 最后一个明确的最终答案优先。
4. 解释支持其他答案，但模型明确写了最终字母时，按明确字母标注。
5. 只有推理、计算或背景说明，没有提交选项，标 `no_answer`。
6. 多个冲突答案且没有最终结论，标 `ambiguous`。

## 文件分工

- `primary_100.jsonl`：100 条主任务，必须全部完成。
- `qc_overlap_10.jsonl`：10 条交叉复核，按同一规则独立完成。不要讨论或寻找其他标注员的答案。

JSONL 每一行是一个独立 JSON 对象。请保持一行一个对象，不要把整个文件改成 JSON 数组。

## 完成示例

原记录：

```json
{"annotation_id":"PA-example","question":"...","options":{"A":"...","B":"...","C":"...","D":"..."},"prediction":"D) Telomerase","manual_answer":null,"status":null,"notes":""}
```

完成后：

```json
{"annotation_id":"PA-example","question":"...","options":{"A":"...","B":"...","C":"...","D":"..."},"prediction":"D) Telomerase","manual_answer":"D","status":"clear","notes":""}
```

回传前确认没有漏掉 `status`，且没有修改模型原始输出。
