# MCQ Parser 人工审计标注规范

版本：parser-audit-v1

## 1. 标注目的

本任务只检查一件事：自动 parser 是否正确读取了模型在原始输出中实际表达的最终选项。

这不是知识问答标注。标注员不需要判断模型答案在医学、科学或常识上是否正确，也不能根据自己的知识替模型纠错。

## 2. 标注单位

每个标注单元包含：

- 一个匿名 `annotation_id`；
- 题目和选项；
- 一个模型的原始输出 `prediction`；
- 待填写的 `manual_answer`、`status` 和 `notes`。

同一道题的不同模型输出必须分别判断。正式包中不会展示模型名称、gold 或自动 parser 结果。

## 3. 输出字段

### `manual_answer`

允许值：

- `"A"`、`"B"`、`"C"`、`"D"`、`"E"`：模型明确选择了该选项；
- `null`：无法得到唯一、明确的模型选择。

### `status`

必须从以下值中选择一个：

- `"clear"`：可以确定唯一选项，此时 `manual_answer` 必须为 A-E；
- `"no_answer"`：没有表达最终选项，此时 `manual_answer` 必须为 `null`；
- `"ambiguous"`：表达了多个冲突选项且没有最终消歧，此时 `manual_answer` 必须为 `null`；
- `"unreadable"`：输出损坏、乱码或上下文不足以判断，此时 `manual_answer` 必须为 `null`。

### `notes`

默认填空字符串。只有 `ambiguous`、`unreadable` 或确实需要解释的边界样本才写简短说明，不要写题目解析。

## 4. 核心判定规则

1. `B`、`B)`、`(B)`、`B. text`、`Answer: B`、`The answer is B` 和 `\\boxed{B}` 都标为 B。
2. 如果输出只写了某个选项的完整文本，并且能与唯一选项精确对应，可以标为对应字母。
3. 最后一个明确的最终答案优先。例如先讨论 A，最后写 `Therefore, the answer is D`，标为 D。
4. 如果解释内容似乎支持 D，但模型明确写 `Answer: B`，仍标为 B。不要替模型修正逻辑。
5. 只有推理、计算或背景说明，没有明确承诺某个选项，标为 `no_answer`。
6. 输出中存在多个冲突答案，且无法确认哪个是最终选择，标为 `ambiguous`。
7. 不使用 gold、题目知识或自动 parser 结果反推模型“应该想选什么”。
8. 大小写和常见标点不影响判断，例如 `b`、`B)` 和 `(b)` 均可视为 B。

## 5. 特殊情况

| 模型输出 | manual_answer | status | 原因 |
|---|---|---|---|
| `C` | C | clear | 独立选项字母 |
| `C) Telomerase` | C | clear | 明确的字母与选项文本 |
| `The answer is A. ... Therefore, final answer: D.` | D | clear | 最后的明确答案优先 |
| `Positive whiff test`，且该文本唯一对应 B | B | clear | 唯一选项文本匹配 |
| `It may be A or C.` | null | ambiguous | 无最终消歧 |
| `We calculate 2000 / 100 ...` 后输出截断 | null | no_answer | 没有表达最终选项 |
| 空字符串或无关乱码 | null | unreadable | 无法读取 |

## 6. 不要做的事

- 不要解题或查资料。
- 不要判断解释是否科学正确。
- 不要因为某个选项是 gold 就选择它。
- 不要因为自动字段说 `correct: true/false` 而改变判断。
- 不要把“解释暗示某答案”当成明确的最终选择。
- 不要擅自改写模型原始输出。

## 7. Parser agreement 的计算规则

这个字段由回收脚本计算，不由盲标员填写：

- `status=clear`：当 `manual_answer == parsed_answer` 时 agreement 为 true，否则为 false；
- `status=no_answer`：当 `parsed_answer` 也是 null 时 agreement 为 true，否则为 false；
- `status=ambiguous/unreadable`：不进入主要 agreement 分母，单独报告比例。

人工答案与 gold 的比较也在回收后由脚本完成。人工标注阶段不展示 gold。

## 8. 质量控制

推荐方案：

1. 两名标注员独立盲标同一批数据；
2. 计算原始一致率和 Cohen's kappa；
3. 对所有分歧、`ambiguous` 和 `unreadable` 样本进行第三方仲裁；
4. 保存原始双人标注，不用仲裁结果覆盖它们；
5. 报告各 domain、Base/GRPO 分组的 parser agreement，以及人工分歧率。

最低方案：一名主标注员完成全部 2400 个输出，第二人复核所有边界样本和 parser 不一致样本，并随机复核至少 20% 的普通样本。

## 9. 提交前自检

- 每个 `annotation_id` 恰好出现一次；
- `status=clear` 时 `manual_answer` 必须是 A-E；
- 其他 status 的 `manual_answer` 必须是 null；
- 不得修改 `annotation_id`、题目、选项或 `prediction`；
- JSONL 每行必须能独立解析为合法 JSON；
- 标注文件中不得新增 gold、模型名或自动 parser 字段。
