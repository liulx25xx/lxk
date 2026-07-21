# AAAI-27 补实验执行清单

> 最后整合：2026-07-21  
> 状态：待执行与跟踪  
> 对应论文：`paper/main_aaai2027.tex`  
> 核心目标：补强外部有效性和强基线比较，而不是继续堆叠局部 reward/prompt/label ablation。
> 本文件是当前唯一有效的补实验计划，并取代历史文件 `NEXT_EXPERIMENTS.md`。

## 0. 先看结论

当前实验数量并不少。已有证据覆盖：

- Qwen2.5、Qwen3、Mistral 三个模型家族；
- SFT、DPO、GRPO，以及多种 proxy reward；
- unbalanced/balanced data、无数据复制的 balanced control；
- prompt、label、learning rate、checkpoint、domain 和 length-confound 控制；
- GRPO 多种子、bootstrap uncertainty，以及公开 JudgeLRM checkpoint 的外部诊断。

真正影响 AAAI reviewer 判断的缺口是：

1. **跨数据集外部有效性不足**：训练与主要验证仍集中在 RewardBench 转换上。
2. **缺少强 position-aware mitigation 正面对比**：目前能证明 balancing 有效，但还不能充分回答它相对在线随机换位、paired consistency 或 permutation-aware 方法的优劣。
3. **生成随机性尚未完全排除**：当前每个 orientation 在 temperature 0.1 下只生成一次。
4. **部分 headline algorithm condition 仍是单次训练**：SFT/DPO 的稳定性弱于主 GRPO 结果。

因此优先级是：**第二数据集 > 强 mitigation baseline > 确定性解码复评 > SFT/DPO 多种子 > 其他扩展**。

### 0.1 历史计划合并状态

旧计划中的任务已按当前证据重新归档，避免重复安排或把已完成实验误列为待办：

| 旧计划事项 | 当前状态 | 本文件中的归属 |
|---|---|---|
| Balanced training / data augmentation | 已完成，且已有 matched-size no-duplication control | 作为现有证据，见第 2 节 |
| 中间 checkpoint 分析 | 已完成 | 不再重跑；仅在需要压缩/重组附录时引用 |
| Post-hoc swap correction/filtering | 已完成 | 作为 evaluation-time baseline，见第 2 节 |
| 真正的 paired consistency training | 尚未完成 | 合并为 permutation-aware mitigation，见第 2.3 节 |

`EXPERIMENTS.md` 是历史实验台账而非待办计划，继续保留用于追溯已有运行；新增实验的状态只在本文件更新。

---

## 1. P0：第二数据集复现（最重要的新训练实验）

### 1.1 Reviewer 问题

> 这个 shortcut 是否只来自作者对 RewardBench 的特定转换，而不是 preference-to-judge pipeline 的一般风险？

### 1.2 数据集选择标准

从一个独立 preference dataset 中选择一个，要求：

- 原始数据含明确的 `chosen` / `rejected` 或可无歧义二值化的偏好标签；
- 许可证允许研究使用和发布派生统计；
- 可以构造完全独立的 train/test split；
- 与 RewardBench 的来源尽量不同；
- 先检查近重复和 train/test 泄漏。

候选可以从 UltraFeedback-binarized、HH-RLHF 或同类公开 preference corpus 中选择，但正式运行前必须确认具体版本、许可证、字段语义和去重情况。

### 1.3 最低实验矩阵

主模型先固定为 Qwen2.5-7B-Instruct，避免把数据集效应与模型变化混在一起。

| Condition | Training data | Seeds | 必须运行 |
|---|---|---:|---|
| Base | 无训练 | 1 checkpoint | 是 |
| SFT-Unbalanced | preferred 永远在 A | 3 | 推荐 |
| SFT-Balanced | preferred 在 A/B 各 50% | 3 | 推荐 |
| GRPO-Full-Unbalanced | preferred 永远在 A | 3 | 是 |
| GRPO-Full-Balanced | preferred 在 A/B 各 50% | 3 | 是 |

如果时间或 GPU 不足，最低可接受版本只保留：Base、GRPO-Full-Unbalanced、GRPO-Full-Balanced，训练条件仍须至少 3 seeds。

### 1.4 评估协议

每个 held-out pair 都评估 original 和 swapped 两个 orientation，保持 prompt、parser 和最大生成长度与主实验一致。

必须报告：

- Original-order accuracy（越高越好，但不能单独作为结论）；
- Swap accuracy（越高越好）；
- Order-averaged accuracy（越高越好）；
- Position-swap consistency（越高越好）；
- First-position rate（越接近 50% 越好）；
- Position bias（绝对值越小越好）；
- mean ± seed standard deviation；
- 测试样本数和 parse-failure rate。

### 1.5 成功判据

不预设或“美化”具体数值。支持论文主张所需的最低模式是：

- unbalanced training 相比 base 在效应量上明显提高 first-position dependence；
- balanced training 相比 unbalanced 提高 swap accuracy/consistency，并使 first-position rate 更接近 50%；
- 结果方向在多数 seeds 中一致；
- 若第二数据集不复现，必须如实报告，并将论文结论限定为当前数据构造范围。

### 1.6 正文位置

在 `paper/main_aaai2027.tex` 的 **Balanced Data Restores Position-Invariant Judgment** 之后、当前 **Robustness Summary** 之前，新增：

```text
4.3 Cross-Dataset Generalization
```

正文放一张紧凑表：每个 dataset 只展示 Base、Unbalanced、Balanced 的 Avg / Con / First-pos。完整 SFT/GRPO、逐 seed、完整六指标放 supplementary。

预计正文占用：0.35–0.50 页。

---

## 2. P0：强 mitigation 与 matched-compute 对比

### 2.1 Reviewer 问题

> Balanced data 是否只是数据量翻倍或普通 augmentation？它相对更直接的 position-aware 方法是否仍有价值？

### 2.2 已有、无需重跑的条件

- Unbalanced GRPO；
- Mirrored balanced GRPO（2,089 → 4,178 pairs）；
- Non-duplicated balanced GRPO（2,089 pairs，已有结果）；
- Prompt reminder；
- Evaluation-time swap filtering。

其中 non-duplicated balanced control 已经表明收益不是由数据翻倍解释。该结果当前位于 `paper/supplement.tex` 的 **Confound-Ratio and Data Duplication Control**，AAAI 版应提升到正文强基线表。

### 2.3 需要新增的条件

至少增加下面第一项；资源允许时再增加第二项：

1. **Online random response order**：每次/每 epoch 动态随机 A/B 位置，保持原始样本数、训练步数和 optimizer updates 不变。
2. **Paired consistency / permutation-aware objective**：显式奖励 original/swapped verdict 的逻辑一致性，或复现一个可稳定实现的 position-aware baseline。

### 2.4 公平比较要求

所有 mitigation 对比必须尽量匹配：

- base model；
- train split；
- optimizer steps；
- effective batch size；
- LoRA configuration；
- reward weights；
- decoding protocol；
- checkpoint selection rule；
- seeds。

若 paired method 每步需要额外 forward pass，必须单独报告训练成本，不能称为严格 matched compute。

### 2.5 结果表设计

| Method | Unique pairs | Updates | Extra forward | Avg ↑ | Con ↑ | First-pos → 50 | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unbalanced | 2,089 | matched | No | TBD | TBD | TBD | 1.0× |
| Mirrored balanced | 4,178 | matched | No | existing | existing | existing | TBD |
| Non-duplicated balanced | 2,089 | matched | No | existing | existing | existing | TBD |
| Online random order | 2,089 | matched | No | TBD | TBD | TBD | TBD |
| Paired/position-aware | matched | matched | Yes/No | TBD | TBD | TBD | TBD |

`TBD` 必须由真实运行填充，不允许根据趋势补数。

### 2.6 正文位置

将当前 `paper/main_aaai2027.tex` 的 **Robustness Summary** 改为：

```text
4.4 Comparison with Position-Aware Mitigations
```

正文保留上述紧凑表和一段结论。Prompt、label、reward、learning-rate 的详细 controls 移到 supplementary，只在正文用一句话汇总。

预计正文占用：0.35–0.45 页。

---

## 3. P0：确定性解码与重复采样复评（无需重新训练）

### 3.1 Reviewer 问题

> Position-swap inconsistency 中有多少只是 temperature 0.1 的采样噪声？

### 3.2 必做版本

对现有 headline checkpoints 重新评估：

- Qwen2.5 base；
- SFT-Unbalanced / SFT-Balanced；
- GRPO-Full-Unbalanced / GRPO-Full-Balanced；
- Qwen3 和 Mistral 的 Base / Unbalanced / Balanced GRPO。

主复评使用 deterministic decoding（temperature 0 或框架等价设置），其他 parser、prompt 和 token budget 不变。

### 3.3 可选加强版

在 temperature 0.1 下对同一 subset 每个 orientation 重复生成 3 次，分别估计：

- 同 orientation 的 sampling disagreement；
- original/swapped 的 position inconsistency；
- 去除 sampling disagreement 后仍可归因于 position 的部分。

重复生成不是独立训练 seed，不能把 sample 次数当作 seed 数量。

### 3.4 正文位置

- 在 **Position-Swap Evaluation** 中增加 deterministic protocol 说明；
- 在 mitigation/robustness 小节加入 1–2 句 headline 结果；
- 完整 decoding 表放 supplementary。

预计正文占用：不超过 0.10 页。

---

## 4. P1：补齐 SFT/DPO 多种子和缺失的 balanced condition

### 4.1 当前缺口

主 GRPO 条件已有多种子，但 headline SFT/DPO 结果的训练重复不足。论文目前声称 shortcut 跨 SFT、DPO、GRPO 出现，因此跨算法结论最好不要依赖单次训练。

### 4.2 运行矩阵

| Algorithm | Unbalanced | Balanced | Seeds |
|---|---|---|---:|
| SFT | 已有，补齐重复 | 已有，补齐重复 | ≥3 |
| DPO | 已有，补齐重复 | 需要新增/确认 | ≥3 |

保持训练数据量与 updates 可比较。若无法完成 DPO-Balanced，则正文不得写成“balancing restores all tested algorithms”；应限定为已验证的 SFT/GRPO。

### 4.3 正文位置

直接更新 `paper/main_aaai2027.tex` 的 unbalanced/balanced 主表，为 SFT/DPO 填入 mean ± standard deviation。逐 seed 结果放 supplementary，不新增正文小节。

---

## 5. P1：Confound-ratio 多种子机制曲线

### 5.1 当前缺口

现有 50% 和 100% endpoints 为多种子，但中间 ratio 主要是单种子且非单调。因此它目前只能支持 boundary diagnostic，不能支持稳定的 dose-response 结论。

### 5.2 推荐矩阵

完整版本：

```text
position-A ratio = 0.50, 0.60, 0.75, 0.80, 0.90, 0.95, 1.00
3 seeds per ratio
```

节省算力版本：

```text
position-A ratio = 0.50, 0.75, 0.90, 1.00
3 seeds per ratio
```

报告 consistency loss、first-position rate、order-averaged accuracy，以及每个点的 seed standard deviation。

### 5.3 判读规则

- 如果趋势单调或总体随 ratio 增强，写“increases on average with confound strength”；
- 如果仍明显非单调，只写“identifies a high-confound failure region”；
- 不根据有限网格声称存在精确临界点；
- 不使用 “statistically significant”，除非实际完成并报告相应检验。

### 5.4 正文位置

若结果稳定，将曲线放到 **Mechanism Analysis / Why Label Rewards Can Prefer the Shortcut**；否则继续留在 supplementary，并保留当前谨慎表述。

---

## 6. P2：更大模型或更强 judge（有余力再做）

### 6.1 目的

回答“该问题是否只存在于 7–9B 开源模型”。

### 6.2 最小矩阵

选择一个 14B+ 或明确更强的开源 judge backbone：

- Base；
- GRPO-Full-Unbalanced；
- GRPO-Full-Balanced；
- 3 seeds 理想；若只有单次运行，必须标为 exploratory。

### 6.3 正文位置

并入现有 **Cross-Model Replication** 表，不新增独立小节。若结果不稳定或只有单 seed，放 supplementary。

---

## 7. AAAI 正文重排方案

当前 `paper/main_aaai2027.pdf` 共 7 页，references 已从第 6 页开始，因此仍有约一页左右 technical-content 空间；但不能把所有新表都直接追加。

推荐结构：

```text
1 Introduction
2 Related Work
3 Diagnostic Framework
4 Experiments
  4.1 Unbalanced Training
  4.2 Balanced Data
  4.3 Cross-Dataset Generalization           [新增，正文]
  4.4 Position-Aware Mitigation Comparison   [新增，正文]
  4.5 Cross-Model Replication                [保留]
5 Mechanism Analysis
6 Discussion and Limitations
7 Conclusion
```

为新增实验腾位置：

- 将正文 label-control 表移到 supplementary；
- 将 External JudgeLRM 完整表移到 supplementary，正文只保留一句外部诊断；
- reward/prompt/label/LR controls 合并为一个 robustness summary；
- uncertainty 数值保留在 supplementary，但在主表中保留 mean ± std；
- cross-dataset 和 mitigation 是关键证据，不能只放 supplementary。

---

## 8. 推荐运行顺序

### Day 0：不占训练资源

- [ ] 核对现有 checkpoints 和 seeds；
- [ ] 跑 deterministic decoding；
- [ ] 将已有 non-duplicated balanced 结果提升到 AAAI 主表草稿；
- [ ] 确定第二数据集、许可证、字段定义和 split。

### Day 1：数据与 smoke test

- [ ] 构建第二数据集的 original/swapped train/test；
- [ ] 检查 A/B 比例、pair 对齐、gold flip 和重复样本；
- [ ] 每个新方法先跑短程 smoke test；
- [ ] 固化超参数，避免看完结果后调参造成选择偏差。

### Day 2–4：主训练

- [ ] 第二数据集 GRPO unbalanced/balanced，3 seeds；
- [ ] online random order，3 seeds；
- [ ] 资源允许时运行 paired/position-aware baseline；
- [ ] 并行补 SFT/DPO seeds。

### Day 5：聚合与检查

- [ ] 重新计算所有六指标；
- [ ] 输出逐 seed 表和 mean ± std；
- [ ] 检查 parse failures、missing pairs、重复 ID；
- [ ] 对照保存的 raw predictions 复算表格；
- [ ] 保留负面或不复现结果。

### Day 6：写入论文

- [ ] 新增 Cross-Dataset subsection；
- [ ] 新增 Mitigation Comparison subsection；
- [ ] 更新主表、abstract、contributions、limitations；
- [ ] 编译 AAAI 版并确认 technical content 不超过 7 页；
- [ ] 更新 supplement、reproducibility checklist 和 anonymous artifact。

---

## 9. 统一输出与命名

建议新结果目录使用匿名、可读、无机器路径的命名：

```text
results/aaai27_crossdata_<dataset>_<algorithm>_<unbal|bal>_s<seed>/
results/aaai27_mitigation_<method>_s<seed>/
results/aaai27_decode_<checkpoint>_<temp0|temp01>_r<repeat>/
```

每个运行至少保存：

```text
config.json
metrics.json
eval_results.json
run.log
```

`config.json` 至少记录：dataset/version、split hash、base model、seed、learning rate、steps、batch size、LoRA settings、reward mode、decoding parameters 和代码版本。

---

## 10. 最小可投方案与完整版

### GPU 非常紧张：最低可投方案

- [ ] 第二数据集只跑 Base + GRPO-Unbalanced + GRPO-Balanced，各训练条件 3 seeds；
- [ ] 对现有 headline checkpoints 做 deterministic decoding；
- [ ] 把已有 non-duplicated balanced control 提升到正文；
- [ ] 明确承认没有完整 position-aware algorithm comparison。

### 推荐方案

- [ ] 完成第二数据集的 SFT/GRPO unbalanced/balanced；
- [ ] 完成 online random order；
- [ ] 至少完成一个 paired/permutation-aware baseline；
- [ ] 补齐 SFT/DPO 多种子；
- [ ] deterministic decoding 全覆盖。

### 理想完整版

- [ ] 推荐方案全部完成；
- [ ] confound-ratio 多种子；
- [ ] 一个 14B+ 模型；
- [ ] matched-compute 和训练成本完整报告。

---

## 11. 结果诚信与写作红线

- 不补数、不插值、不按期望趋势修改结果；
- 不把重复采样当作独立训练 seeds；
- 不在无统计检验时写 “statistically significant”；
- 不隐藏不复现、方差大或与主假设冲突的结果；
- 不把不同 reward、不同训练步数的跨模型差异归因于模型能力；
- 新实验若改变核心结论，应修改 abstract、claims 和 limitations，而不是只放进附录；
- 所有主表数字必须能从匿名 artifact 中的原始预测重新计算。

---

## 12. 完成状态总表

| ID | 实验 | 优先级 | 状态 | 正文位置 |
|---|---|---|---|---|
| E1 | 第二数据集复现 | P0 | ☐ 未开始 | §4.3 |
| E2 | Online random order | P0 | ☐ 未开始 | §4.4 |
| E3 | Paired/permutation-aware baseline | P0/P1 | ☐ 未开始 | §4.4 |
| E4 | Deterministic decoding | P0 | ☐ 未开始 | Setup + §4.4 |
| E5 | SFT/DPO 多种子 | P1 | ☐ 未开始 | 更新 Table 1/2 |
| E6 | Confound-ratio 多种子 | P1 | ☐ 未开始 | Mechanism / Supplement |
| E7 | 14B+ 模型 | P2 | ☐ 未开始 | Cross-Model table |
| E8 | Non-duplicated balanced control | 已完成 | ☑ 已有结果，待提升正文 | §4.4 |
| E9 | GRPO 主条件多种子 | 已完成 | ☑ 已在论文/补充材料 | Table 1/2 |
| E10 | Prompt/label/LR/domain controls | 已完成 | ☑ 已有结果 | Supplement 为主 |
