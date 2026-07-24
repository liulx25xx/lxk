# 盲标数据生成与交付规范

版本：parser-audit-v1

## 1. 输入与规模

固定输入是三个随机抽样文件：

- Science：200 道题；
- Medicine：200 道题；
- Commonsense：200 道题。

每题包含 Base 和 3 个 dev-selected GRPO seed 的输出。生成器必须将嵌套结构展开为 2400 个独立标注单元。

不得重新抽样，不得只保留自动 parser 不一致的输出，也不得删除看起来过于简单的单字母输出。当前 200/domain 是固定随机样本，完整保留才能无偏估计 parser agreement。

## 2. 正式盲标单元

发给标注员的每一行只能包含：

```json
{
  "annotation_id": "PA-000001",
  "question": "Question text",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "prediction": "raw model output",
  "manual_answer": null,
  "status": null,
  "notes": ""
}
```

必须隐藏：

- domain 和 benchmark 名称；
- 原始 item ID；
- Base/GRPO 身份、learning rate、seed 和 checkpoint；
- gold answer；
- `parsed_answer` 与 `strict_parsed_answer`；
- `correct`、reward 以及任何自动判分；
- 同一道题还有哪些模型输出。

题目本身可能自然暴露学科类型，这不构成实验条件泄漏。关键是不能暴露模型身份和自动判分。

## 3. 私有映射表

协调者单独保存、禁止发给标注员的映射表：

```json
{
  "annotation_id": "PA-000001",
  "domain": "medicine",
  "item_id": "medqa_usmle_4opt:...",
  "model_tag": "base",
  "gold": "D",
  "parsed_answer": "A",
  "strict_parsed_answer": "B",
  "source_prediction_sha256": "..."
}
```

映射表与盲标包必须分目录保存。共享盲标包时不要共享私有映射目录。

## 4. ID 与随机化

1. `annotation_id` 使用无语义、不连续泄漏模型分组的随机 ID。
2. 随机化使用固定 seed，并把 seed 写入生成 manifest，以便复现。
3. 2400 条记录全局随机排序；同一道题的 4 个输出不得相邻出现。
4. 不同标注员收到相同标注单元，但记录顺序应独立随机化。
5. 不在文件名中写 domain、model 或 condition。
6. 任何重新生成都必须使用新版本号，不能覆盖已经发出的包。

## 5. 题目与选项恢复

使用私有映射中的原始 item ID，从以下 frozen test 文件关联题目与选项：

- `data/aaai27/science/test.jsonl`
- `data/aaai27/medicine/test.jsonl`
- `data/aaai27/commonsense/test.jsonl`

`question` 应移除末尾的统一答题指令；`options` 使用 positional A/B/C/D/E 与对应文本。不得更改选项顺序，因为当前人工审计针对 original-order 输出。

模型的 `prediction` 必须逐字保留。JSON 中换行可以编码为 `\n`，但标注界面应恢复为可读换行。

## 6. 分包与人员分配

推荐将 2400 条切成 8 个约 300 条的 packet。每个 packet 记录：

- packet ID 和版本；
- 记录数；
- annotation ID 列表的 SHA-256；
- 文件 SHA-256；
- 分配的 annotator/pass；
- 发出和回收时间。

如果采用全量双标，两轮都覆盖相同的 2400 个 annotation ID，但使用不同顺序。若采用最低方案，第二轮至少覆盖所有边界样本、所有自动 parser 分歧样本以及预先随机抽取的 20% 普通样本；第二轮抽样规则必须在解除盲化前冻结。

## 7. 推荐目录结构

```text
parser_audit_release_v1/
  public/
    instructions.pdf-or-md
    annotator_A/
      packet_01.jsonl
      ...
    annotator_B/
      packet_01.jsonl
      ...
  private_do_not_share/
    mapping.jsonl
    generation_manifest.json
    packet_checksums.sha256
  returned/
    raw/
    validated/
    adjudicated/
```

## 8. 生成后硬检查

- 总标注单元必须是 2400；
- 每个源 item 恰好对应 4 个 annotation ID；
- annotation ID 全局唯一；
- `public/` 中不得出现禁用字段及模型标签字符串；
- question 和 options 不得为空；prediction 字段必须存在，但允许模型产生空字符串，此时应保留并标为 no_answer；
- 每个 prediction 的 hash 必须与私有映射一致；
- 所有 JSONL 行必须通过 `schemas/blinded_item.schema.json`；
- 保存输入文件、输出文件、脚本和 manifest 的 SHA-256。

正式发包前应由协调者打开至少 20 个随机条目，人工确认换行、选项和原始输出显示正常。
