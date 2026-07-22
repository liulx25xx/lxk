# AAAI-27 执行计划（Claude 版）— Position Shortcuts in Trained LLM Judges

> 生成：2026-07-21 ｜ 摘要 7/22 20:00 ｜ 全文假设 ~7/28（待用户确认实际日期）
> 取向：**最可能中稿**，不为摘要临时跑实验（docker 现在挂、也跑不了），摘要按重构 thesis 写，全文实验 docker 回来补。
> 本文件取代 `AAAI27_REQUIRED_EXPERIMENTS.md` 作为当前执行计划（那份作为历史参考保留）。

---

## 0. 锁定方向（一句话）

**Position-swap consistency 是 trained pairwise judge 缺失的必报指标；position shortcut 是领域级、强度随 base judge 能力反相关的风险；data-level deconfounding（balancing）是相对 algorithm-level 方法的正确缓解层级。**

## 1. 为什么是这个（最可能中稿）

- **现有数据已全支撑**：decomposition（shortcut +11.0/+17.4/+1.3）、severity∝base 弱（一致性 −41.8/−22.8/−0.4）、reward 修复无效、JudgeLRM 公开 checkpoint 中招（gap 15–21pp）。
- **不赌未跑实验**：对比"方法论文"选项（要 paired-consistency 实测赢 balancing 才成立，genuine gain 才 +3.5pp，赌不起）。
- **三支柱 1:1 打掉 EMNLP review 剩余 weakness**（自造 confound / 缓解平凡 / 单数据集）。

## 2. 摘要决策

- 摘要 = 写作决策，不是实验决策。现有结果足够。已交付 `AAAI27_ABSTRACT_DRAFT.md`。
- 摘要承诺的所有 claim 都在现有数据或 docker 恢复后必补的实验范围内（见该文件"证据核对"）。

## 3. 工作切分：CPU-only（现在）vs GPU（docker 恢复后）

### A. CPU-only —— 现在就做，不碰 docker

| ID | 内容 | 产出 |
|---|---|---|
| A1 | **severity∝weakness 主图**：base 准确率 × (一致性掉幅 或 shortcut pp)，3 模型 + JudgeLRM(外检面板) | 全文 hero figure |
| A2 | **decomposition 协议化**：把 §5.2 提成"任何人能对自己 judge 套用的 audit-and-decompose 两步法"，补 bootstrap | 正文 §5 + 方法学贡献 |
| A3 | **论文重定位**：abstract/intro/contributions 改；main 新增 §4.3 Cross-Dataset、§4.4 Mitigation Comparison；label 表 + 完整 JudgeLRM 下沉 supplement（§7 方案） | 改 `main_aaai2027.tex` |
| A4 | **备好待跑代码**（B 项全部脚本 + 配置，docker 一回即起） | `scripts/` 新增 |

### B. GPU —— docker 恢复后，按**成本/价值分层**跑（时间紧，优先级从严）

**Tier 1：推理即可（快、高价值，任何 inference 恢复就先跑）**
- B3 **确定性解码**（pillar C）：现有 headline checkpoint temp=0 重评，补 mean±std，杀"采样噪声"质疑。
- B4 **logit-bias 探针**：`probe_logit_bias.py` 从未跑过，直接验证论文自己的 slot-bias 模型（hidden-state 探针已证 position 不可线性解码≈57%，所以 bias 在 logit 项里——正好这个探针测）。
- B5 **Coder7B / DeepSeek7B 评测**：训练完从未评测，跑 eval = 2 个免费模型族点，给 severity 图加密。

**Tier 2：训练（docker 充分恢复才跑）**
- B1 **支柱 A 跨数据集**（最高优先训练项）：UltraFeedback-binarized 或 HH-RLHF → judge prompt，Base / GRPO-Unbal / GRPO-Bal，3 seeds。把"RewardBench artifact"变"通用 pipeline 风险"。
- B2 **支柱 B 缓解对比**：真·paired-consistency reward（paired-sampling）+ online-random-order，3 seeds，matched compute（样本数/步数/updates 对齐；paired 法多一次 forward 单独报成本）。回应"balancing 是否只是 augmentation / 有没有算法层方法更好"。

**降级/砍**（增量、页数紧，有余力再说）：confound-ratio 多种子、14B+ 模型。

## 4. 时间线（假设全文 ~7/28，待确认；若更晚则 Tier 2 全做）

- **Now → 7/22 20:00**：锁定摘要 + A1/A2/A3 推进 + A4 代码备好。
- **docker 恢复 → 7/28**：先 Tier 1（1 天内可完），再 Tier 2（跨数据集 + 缓解对比，~40–60 H200-h，2–3 天），边跑边写全文。
- **兜底**：若 docker 只够 Tier 1 + 跨数据集，论文已显著强于现版（severity 主图 + 2nd 数据集 + 机制探针 + 确定性解码）；paired-consistency 留 supplement 或标 future work。

## 5. Run matrix（B 项，待 docker；命名沿用 AAAI doc §9）

```
results/aaai27_crossdata_<dataset>_grpo_<unbal|bal>_s<seed>/
results/aaai27_mitigation_<paired|online>_s<seed>/
results/aaai27_decode_<checkpoint>_temp0/
results/aaai27_probe_logitbias_<checkpoint>/
results/aaai27_xmodel_<coder7b|deepseek7b>_<unbal|bal|base>/
```
每 run 至少存 `config.json / metrics.json / eval_results.json / run.log`；config 记 dataset/version、split hash、base model、seed、lr、steps、batch、LoRA、reward mode、decode 参数、代码版本。

## 6. 公平比较硬要求（pillar B）

base model / train split / optimizer steps / effective batch / LoRA config / reward weights / decode / checkpoint 选择规则 / seeds 全部对齐。paired 法若每步多一次 forward，**单独报训练成本**，不得称严格 matched compute。

## 7. 正文结构（§7 重排，控 7 页 technical）

```
1 Intro
2 Related Work
3 Diagnostic Framework
4 Experiments
  4.1 Unbalanced Training（shortcut 出现）
  4.2 Balanced Data（缓解）
  4.3 Cross-Dataset Generalization        [新增正文]
  4.4 Position-Aware Mitigation Comparison[新增正文]
  4.5 Cross-Model Replication（含 severity∝weakness 主图）
5 Mechanism（slot-bias 梯度 + logit-bias 探针 + decomposition 协议）
6 Discussion & Limitations
7 Conclusion
```
label-control 表、完整 JudgeLRM 表、reward/prompt/LR 详细 control 下沉 supplement；uncertainty 数值留 supplement，主表保 mean±std。

## 8. 诚信红线（沿用 AAAI doc §11）

- 不补数/插值/按期望趋势改结果；重复采样 ≠ seed；无检验不写 "statistically significant"。
- 负面/不复现/大方差结果照报；跨模型差异不得无依据归因于模型能力。
- 新实验若改变核心结论，改 abstract/claims/limitations，不能只塞附录。
- 所有主表数字必须能从 artifact 原始预测复算。

## 9. 与原 `AAAI27_REQUIRED_EXPERIMENTS.md` 的差异

- **保留**：第二数据集(P0)、缓解对比(P0)、确定性解码(P0)、SFT/DPO 多种子(P1)。
- **升级**：severity∝weakness 提为 hero figure；logit-bias 探针列入机制（原计划没提，且 hidden-state 探针已证 null）。
- **降级/砍**：confound-ratio 多种子、14B+（增量，时间紧）。
- **明确**：摘要不依赖任何未跑实验；Tier 1 推理项优先于 Tier 2 训练项。
