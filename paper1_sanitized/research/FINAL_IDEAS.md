# EMNLP 2026 Paper 1 — Final Idea Selection

**Date**: 2026-05-15  
**Deadline**: ARR May 25 (10 days)  
**Resources**: 24×H200, $1k API credits  
**Constraint**: 5 天完成 (实验+写作)

---

## 原始 5 个 Idea 淘汰情况

| Idea | Novelty | 致命问题 | 结论 |
|------|---------|----------|------|
| 1. Judge Calibration | 3/10 | UDA (2025) + CalibraEval (ACL'25) 已覆盖 | ❌ 放弃 |
| 2. Adversarial Debate | 5/10 | Kraidia (Nature 2026.04) 已做攻击分析 | ⚠️ 可选(仅defense) |
| 3. Synthetic Data Paradox | 低 | Nature 2024 (642引) + ICLR 2025 已讲完 | ❌ 放弃 |
| 4. Self-Supervised PRM | 低-中 | Math-Shepherd + 10篇后续已覆盖 | ❌ 放弃 |
| 5. Adaptive TTC | 极低 | AdaCompute (2026.04) 完全相同 | ❌ 放弃 |

## OPD/OPSD 方向分析

**现状**: 2026年最热方向，100+论文，2个awesome list，多篇survey。极度拥挤。
**核心 Gap** (按Agent 1报告):
1. 无标签/开放域自蒸馏 → 最大gap
2. 自蒸馏的不确定性退化 → Kim et al. (2026.03) 发现问题但**无解决方案**
3. Agent/长推理链场景 → 初步探索(SOD)但未解决
4. Scaling Laws → 无人做
5. 跨能力迁移 → 几乎全在math上验证

**关键竞争**: EGRSD (2026-05-14, 昨天!) 用entropy gate解决部分问题，但角度不同(下调权重 vs 保留表达)

---

## 最终推荐 Ideas (3个新方向)

### 🏆 Idea A: "Preserving Epistemic Reasoning in Self-Distillation"
**不确定性保留的自蒸馏 — 直接填补 Kim et al. 的 gap**

**核心观点**: Kim et al. (2026.03) 发现自蒸馏抑制认知不确定性表达(epistemic verbalization)，导致 OOD 性能下降高达 40%。他们发现了问题但**没有提出解决方案**。我们提出 **Epistemic-Preserving Self-Distillation (EPSD)**。

**方法**:
1. **Epistemic Token Detection**: 自动识别推理链中的认知不确定性标记(如"wait", "let me reconsider", "alternatively", "hmm")
2. **Selective Distillation**: 对epistemic tokens采用不同的loss权重/目标——不压缩这些表达
3. **Uncertainty-Aware Teacher**: 教师模型条件化时保留部分不确定性(不给完整答案，只给部分hint)
4. **Dual Objective**: 在 accuracy 和 epistemic calibration 之间做 Pareto 优化

**与现有工作的关键区别**:
- **vs Kim et al.**: 他们诊断问题，我们提供解决方案
- **vs EGRSD (昨天)**: EGRSD 下调高entropy token权重(=忽略不确定区域)。我们的角度相反——**保留**不确定性表达，因为它们对OOD泛化至关重要
- **vs CaOPD**: CaOPD 做calibration是指教师的置信度校准，不是学生的epistemic verbalization
- **vs GATES/UniSD**: 都关注如何更好地匹配教师，没有关注epistemic preservation

**为什么novelty高**:
- Kim et al. 是2026.03的论文，明确指出"开放问题"
- 没有任何已发表工作提出 epistemic-preserving distillation
- EGRSD (昨天) 的方向是互补的(不是竞争的)——他们减小不确定区域权重，我们保留不确定性表达
- 这是一个"发现问题→解决问题"的经典 EMNLP paper pattern

**实验设计**:
- 基座模型: Qwen3-4B/8B (与Kim et al.和EGRSD对齐)
- 训练数据: GSM8K, MATH (in-domain) → AIME, Minerva, OOD math (out-domain)
- Baselines: OPSD, EGRSD, GRPO, SFT on correct traces
- Metrics: In-domain accuracy, OOD accuracy, Epistemic verbalization rate, Calibration (ECE)
- 消融: selective loss weight, teacher conditioning strength, epistemic token detection method

**可行性**: ★★★★★
- 主要是在 OPSD 框架上修改 loss function (1-2天实现)
- 训练 Qwen3-4B 在 24×H200 上非常快
- Epistemic token detection 可用简单规则或小分类器
- 与 Kim et al. 完全对齐的实验设置，reproducible

**EMNLP 契合**: ★★★★★
- Empirical + analysis + method
- 契合 Special Theme: "Rethinking Progress & Evaluation"
- 直接回应一篇有影响力的近期论文

---

### 🥈 Idea B: "Self-Distillation Beyond Math: Does It Generalize?"
**跨任务自蒸馏泛化研究**

**核心观点**: 几乎所有 OPSD 工作都在数学推理上验证。自蒸馏在 NLP 核心任务（摘要、翻译、信息抽取、代码、对话）上是否同样有效？什么特征决定了自蒸馏的成败？

**方法**:
1. 在 6+ 个不同 NLP 任务上系统测试 OPSD/SDPO
2. 分析 task characteristics (答案确定性, 推理链长度, 创造性需求) 与自蒸馏效果的关系
3. 提出 Task-Adaptive Self-Distillation: 根据任务特征自动调整特权信息类型和蒸馏强度
4. 建立"自蒸馏适用性预测器"

**与现有工作区别**:
- 所有 OPSD papers 在 math/code 上验证
- Uni-OPD 做了 LLM+MLLM 但用外部教师
- 无人系统研究"自蒸馏在哪些任务上有效/无效"

**风险**: 可能被视为 "empirical study" (贡献偏窄)。需要 method contribution (Task-Adaptive 部分)来补强。

**可行性**: ★★★★ (需要多任务数据准备，时间稍紧)

---

### 🥉 Idea C: "Reward-Guided Self-Distillation for Open-Ended Tasks"
**无标签域的 Reward-Guided 自蒸馏**

**核心观点**: OPSD 需要 ground-truth 答案作为特权信息。对开放域任务(对话、创意写作、通用指令)，ground-truth 不存在。我们用 **reward model / AI judge 的评分作为"软特权信号"**，实现无标签域的自蒸馏。

**方法**:
1. Student rollout → Reward model 评分
2. 高分 rollout 作为"参考轨迹"条件化教师
3. 教师(同一模型,但条件化在高分参考上)提供 token-level supervision
4. 学生在自己的 rollout 上学习教师分布

**与现有工作区别**:
- OPSD/SDPO: 需要 ground-truth 或环境反馈
- RLKD (2026.04): 用 LLM-as-judge 做 RL reward, 但不是自蒸馏
- SDPO: 用文本反馈做特权信息, 但需要环境提供反馈
- 我们: 用 reward model 替代 ground-truth, 完全自主

**风险**: reward model 噪声可能导致 Kim et al. 发现的过度自信问题。可能与 Idea A 组合。

**可行性**: ★★★★ (需要 reward model; $1k API 可用 GPT-4o 作 judge)

---

## 综合推荐

| 排名 | Idea | Novelty | 5天可行性 | EMNLP契合 | 总分 |
|------|------|---------|-----------|-----------|------|
| **1** | **A: Epistemic-Preserving SD** | ★★★★★ | ★★★★★ | ★★★★★ | **最优选** |
| 2 | B: Cross-Task SD | ★★★★ | ★★★★ | ★★★★ | 备选 |
| 3 | C: Reward-Guided SD | ★★★★ | ★★★★ | ★★★★ | 可与A组合 |

---

## 强烈推荐: Idea A

**理由**:
1. **清晰的 motivation**: Kim et al. 发现了 40% OOD 退化但没解方案
2. **精确的 novelty**: 填补 well-identified gap, 与 EGRSD 互补不冲突
3. **极高可行性**: 修改 OPSD 的 loss function, 1-2天实现, 2天实验, 1天写作
4. **完美的 EMNLP pattern**: 诊断→分析→解决, empirical + method
5. **资源匹配**: Qwen3-4B/8B 在 24×H200 上训练极快
6. **引用链清晰**: OPSD (ICML'26) → Kim et al. 问题 → 我们的解决方案

**可选增强**: 如果时间允许，将 Idea C (reward-guided) 作为 extension 加入——在无标签域验证 epistemic preservation 的泛化性。

---

## 下一步行动

1. 用户确认选择
2. 详细实验方案设计
3. 代码框架搭建
4. 论文 LaTeX 框架搭建
5. 开始训练
