# 正式标注、回收与质量控制流程

版本：parser-audit-v1

本项目不设置新的培训或准入测试。所有参与者仍需在开始前确认已经阅读同一版本的标注规范。

## 1. 角色

- **协调者**：生成盲标包、分配 packet、保存私有映射、执行格式校验和统计。
- **标注员**：只依据题目、选项和原始输出填写人工答案，不接触私有字段。
- **仲裁者**：处理两位标注员的分歧；仲裁前同样不能看到模型身份、gold 或自动 parser。

仲裁者可以是资深标注员，但不得用题目正确答案替代“模型表达了什么”这一判断目标。

## 2. 正式标注步骤

1. 标注员收到说明文件和一个或多个匿名 packet。
2. 每条记录先读 `prediction`，必要时查看题目和选项文本。
3. 填写 `manual_answer`、`status`，仅边界情况填写 `notes`。
4. 不修改其他字段，不重新排序或删除记录。
5. 每完成一个 packet 就单独回传，不把多个版本覆盖为同名文件。

建议每次连续标注不超过 60 分钟。是否休息不进入实验指标，但长文本判断应避免疲劳造成系统性误差。

## 3. 标注中的问题处理

标注员遇到规范未覆盖的新类型时：

1. 暂时标为 `ambiguous` 或 `unreadable`；
2. 在 `notes` 中简述原因；
3. 记录 annotation ID，并在该 packet 完成后统一反馈；
4. 协调者若需要更新规则，必须发布新规范版本并将规则同步应用到此前同类样本。

不要在标注过程中针对单个样本临时查看 gold 或 parser。

## 4. 回收后的机器校验

每个回收文件必须检查：

- JSONL 可解析；
- 文件名、annotator ID、pass 和规范版本完整；
- annotation ID 没有缺失、重复或未知值；
- question、options、prediction 与发出版本逐字一致；
- `status=clear` 时 manual answer 是 A-E；
- 其他 status 时 manual answer 为 null；
- 除允许的三个标注字段外，没有内容被修改；
- 回收文件保存 SHA-256，原始版本只读保留。

校验失败的 packet 应退回修正，不能由协调者猜测或代填。

## 5. 双人一致性与复核

全量双标时，以以下 8 类标签计算一致性：

`A, B, C, D, E, NO_ANSWER, AMBIGUOUS, UNREADABLE`

报告：

- raw agreement；
- Cohen's kappa；
- 各标签数量；
- 各 domain 的一致率；
- 分歧矩阵。

最低复核方案下，报告第二人实际覆盖比例、随机普通样本覆盖比例和边界样本覆盖比例，不把部分重叠 kappa 描述成全量双标结果。

## 6. 仲裁

以下样本进入仲裁：

- 两位标注员标签不同；
- 任一标注为 `ambiguous` 或 `unreadable`；
- 格式校验通过但 notes 指出规则问题；
- 解除盲化后，manual answer 与 parser 不一致。

最后一项不是让仲裁者偏向 parser，而是对论文中实际计为 parser error 的样本进行额外质量确认。

仲裁文件应同时保存：annotator A、annotator B、adjudicated answer、adjudicated status、仲裁理由和仲裁者 ID。不得覆盖原始标注。

## 7. 解除盲化后的计算

完成仲裁后才能关联私有映射，并分别对 Base 与 GRPO 计算：

1. parser agreement；
2. false extraction：人工无答案但 parser 给出字母；
3. missed extraction：人工有明确答案但 parser 返回 null；
4. wrong-letter extraction：人工字母与 parser 字母不同；
5. ambiguous/unreadable rate；
6. 基于人工答案重新计算的 accuracy；
7. 自动 accuracy 与人工修正 accuracy 的差值；
8. Base 与 GRPO 的 parser error 差异及置信区间。

主要 agreement 分母排除 `ambiguous/unreadable`，但必须同时报告被排除数量和比例。`no_answer` 保留在分母中。

## 8. 结论门槛

- 如果 Base 与 GRPO 的人工 parser error 都很低，且修正前后主结论不变，可写 parser audit supports robustness。
- 如果 Base parser error 明显更高，但人工修正后 GRPO 增益仍存在，应报告原 parser 有偏但不足以解释全部增益。
- 如果人工修正显著缩小或消除增益，正文必须把结果限定为输出格式/parser 改善，不能写能力提升。
- 如果人工一致率不足，先改进规则并重新仲裁，不能直接报告不稳定的人工结果。

## 9. 最终必须保留的文件

- 发出的每个盲标 packet；
- 每位标注员的原始回收文件；
- 校验日志与 checksums；
- 双人分歧表；
- 仲裁文件；
- 私有映射与生成 manifest；
- 汇总统计 JSON/CSV；
- 生成论文表格所用脚本及其版本。
