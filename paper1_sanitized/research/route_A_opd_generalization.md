# Route A: OPD/OPSD 对通用能力的泛化性影响 — 系统性研究

**调研日期**: 2026-05-15  
**目标**: 评估 "OPD generalization to general capabilities" 作为 EMNLP 2026 paper 的可行性  
**输出格式**: Related Work → Gap Analysis → Novelty Score → Experiment Design → Risk Assessment → 最终推荐

---

## 1. Related Work — 全景梳理

### 1.1 核心基础：CMU "Does Math Reasoning Improve General LLM Capabilities?" (arXiv:2507.00432)

**作者**: CMU团队  
**发表**: 2025年7月  
**核心发现**:
- 对 20+ 开源推理调优模型进行多域评估：数学、科学QA、Agent规划、编程、指令遵循
- **RL调优模型跨域泛化良好**，Transferability Index (TI) 为正值
- **SFT调优模型大面积遗忘**，TI_non 频繁为负
- PCA Shift分析：SFT模型在非目标域的表征漂移是RL的100-1000x（如OpenThinker2-7B: 5486.2 vs Qwen2.5-7B-SimpleRL: 0.6）
- Token-level KL散度：RL ≈ 0.084, SFT ≈ 0.372
- **关键ablation**: "Sampling distribution is critical. On-policy methods outperform off-policy methods across both evaluation categories and training paradigms."

**关键缺口**: 
- **测试了 on-policy SFT 和 on-policy RL，但完全没有测试 OPD/OPSD**
- OPD 结合了 on-policy sampling（像RL）和 dense logit supervision（像SFT），它的泛化性在 RL 和 SFT 之间的什么位置？**这是一个事实性空白。**

**评估 benchmarks**: MATH500, AIME24/25, OlympiadBench, LiveCodeBench, GPQA-Diamond, ACPBench, HeadQA, CoQA, IFEval, HaluEval, MC-TACO

**模型**: Qwen3-14B 为主要控制实验模型，Qwen2.5-7B/14B/32B, Llama3.1-8B 等

---

### 1.2 博客："SFT, RL, and OPD Through a Distributional Lens" (nrehiew.github.io, 2026-05-10)

**核心发现**:
- 在 minimal code editing task 上系统比较 SFT Teacher, RL Teacher, OPD Student
- **OPD Student 比 SFT Teacher 遗忘更少**：LiveCodeBench v6 上 OPD(SFT teacher) 0.297 > SFT teacher 0.286
- **OPD Student 甚至略超 RL Teacher**: Pass@1 上 OPD 0.800 > RL 0.792
- 关键引言: "The OPD student trained from the SFT teacher forgot less than the SFT teacher itself"
- 解释机制：OPD 继承 RL 的 on-policy 性质 → 隐式 KL 正则化 → 更少遗忘

**局限性**:
- **仅1个任务**（code editing → LiveCodeBench），不是系统性多域研究
- **非正式出版物**（博客），无同行评审
- 无法作为"已解决"的论据，但提供了初步 signal

---

### 1.3 RL's Razor (arXiv:2509.04259, ICLR 2026)

**作者**: Shenfeld, Pari, Agrawal (MIT)  
**核心理论**:
- 在线RL天然偏向 KL-minimal solution（类比 Occam's Razor）
- 遗忘程度由 fine-tuned policy 与 base policy 的 KL 散度决定
- RL 产生的 KL 变化远小于 SFT
- 在 LLM 和机器人模型上验证

**与本方向的关系**:
- 提供了 "on-policy → less forgetting" 的理论基础
- 但 **没有讨论 OPD** —— OPD 有 on-policy sampling 但也有 teacher logit signal，是否会破坏 KL-minimal 性质？

---

### 1.4 SDFT: Self-Distillation Enables Continual Learning (arXiv:2601.19897, MIT)

**核心方法**: 用模型自身作为教师（通过 ICL），进行 on-policy self-distillation  
**核心发现**:
- 3任务顺序学习：SDFT 比 SFT 在 7B 上 +4pt，14B 上 +7pt
- 3B 模型 ICL 能力太弱，SDFT 反而不如 SFT
- 建立了 "on-policy distillation 作为 continual learning 工具" 的先例

**与本方向的关系**:
- SDFT 关注的是 **continual learning**（多任务顺序学习不遗忘），不是 **单次训练后对其他域的泛化性**
- 是互补工作，不是竞争工作
- 但 reviewer 可能要求我们与 SDFT 比较

---

### 1.5 Kim et al. — Self-Distillation Degrades Reasoning (arXiv:2603.24472)

**核心发现**: Self-distillation 抑制 epistemic verbalization → OOD 性能下降 40%  
**与本方向的关系**:
- 发现了 OPD/OPSD 在 OOD 上的退化，但 **仅限数学域内的 OOD**（AIME24 vs MATH500）
- **没有跨域评估**（如数学训练后测指令遵循、QA等）
- 我们的方向是更广义的泛化性研究，Kim et al. 是数学域内的 OOD 分析

---

### 1.6 Revisiting OPD: Empirical Failure Modes (arXiv:2603.25562)

**核心发现**: OPD 的三大失败模式（token-level supervision不均衡、教师在学生前缀上不可靠、tokenizer不匹配）  
**与本方向的关系**:
- 关注 OPD 的优化稳定性问题，**不涉及泛化性或遗忘**
- 提出的 top-K 修复可作为我们实验中的 baseline variant

---

### 1.7 G-OPD: Generalized On-Policy Distillation (arXiv:2602.12125)

**核心贡献**: OPD = KL-constrained RL 的理论统一框架；ExOPD 用 reward extrapolation 突破教师上限  
**与本方向的关系**:
- 理论框架可用来解释 OPD 泛化性（如果 OPD ≈ KL-constrained RL，那泛化性应该接近 RL？）
- 但 G-OPD 论文**没有实验验证泛化性**，只测了数学 benchmarks

---

### 1.8 PRISM: Pre-alignment via Black-Box OPD (arXiv:2604.28123)

**核心**: 在 SFT 和 RLVR 之间插入 OPD 预对齐阶段  
**与本方向的关系**:
- 关注 OPD 如何改善下游 RL，**不涉及通用能力保持**
- 多模态设定（Qwen3-VL），但不测非目标域能力

---

### 1.9 X-OPD: Cross-Modal OPD (arXiv:2603.24596)

**核心**: 用 text LLM 作为教师蒸馏 Speech LLM  
**与本方向的关系**: 跨模态 OPD，但不是跨域泛化性研究

---

### 1.10 工业界实践（非公开数据）

| 模型 | OPD 使用方式 | 通用能力评估 | 公开数据 |
|------|-------------|-------------|---------|
| Qwen3 | OPD 训练轻量模型 | 技术报告有综合评估 | 无 OPD 前后对比 |
| GLM-5 | OPD 修复多阶段 RL 后的能力退化 | 技术报告有 | 无分离实验 |
| DeepSeek-V4 | 最终合并模型用 OPD | 综合评估优秀 | 无法隔离 OPD 效果 |

---

## 2. Gap Analysis — 关键空白识别

### 2.1 已有的研究

| 问题 | 已有工作 | 覆盖程度 |
|------|---------|---------|
| RL vs SFT 泛化性 | CMU (2507.00432) | ✅ 全面 |
| RL 为什么遗忘少 | RL's Razor (ICLR 2026) | ✅ 理论+实证 |
| OPD 优化稳定性 | Revisiting OPD (2603.25562) | ✅ |
| OPD 在数学域内OOD | Kim et al. (2603.24472) | ⚠️ 仅数学OOD |
| OPD continual learning | SDFT (2601.19897) | ⚠️ 仅顺序学习 |
| OPD 在 code editing 上的遗忘 | nrehiew blog (2026-05-10) | ⚠️ 1个任务，博客 |
| OPD 理论与RL等价 | G-OPD (2602.12125) | ✅ 理论 |

### 2.2 事实性空白（我们要填补的）

> **核心 Gap: 没有任何已发表工作系统比较 SFT vs RL vs OPD 在多个非训练域上的通用能力变化。**

具体未覆盖的问题：

1. **OPD 训练后，模型在 IFEval, CoQA, HaluEval, GPQA 等通用benchmark上的表现如何变化？** — 无人做过
2. **OPD 的泛化性是更接近 RL（好）还是 SFT（差）？** — 理论上应接近 RL（因为 on-policy），但无实证
3. **OPSD（self-distillation）vs OPD（external teacher）在泛化性上有差异吗？** — 无人比较
4. **OPD 的 representation drift 在非训练域上的程度如何？** — CMU 论文的 PCA shift 分析完全可以复用，但他们没测 OPD
5. **OPD 的 teacher signal 密度（dense logit supervision）是否会增加遗忘？** — RL's Razor 理论预测 on-policy 应该少遗忘，但 dense teacher signal 可能拉偏分布

### 2.3 Gap 的可信度评估

| 维度 | 评估 |
|------|------|
| 这个 gap 是真实的吗？ | ✅ 100% 真实 — CMU 论文明确只测了 RL 和 SFT |
| 这个 gap 重要吗？ | ✅ 高度重要 — OPD 是 2026 年工业标准，但泛化影响未知 |
| 有人正在填补吗？ | ⚠️ 可能 — nrehiew 博客可能是后续论文的前兆 |
| 5天能填补吗？ | ✅ 可行 — 复用 CMU 的评估框架即可 |

---

## 3. Novelty Score

### 总分: **7/10 (中-高)**

**加分项**:
- 填补一个明确的事实性空白（CMU 论文的自然扩展）
- 工业界高度关注但无公开数据
- 理论预测（G-OPD ≈ KL-constrained RL → 应类似RL泛化）需要实证验证
- 如果发现 OPD 泛化性不如 RL（unexpected result），novelty 可以更高

**减分项**:
- 本质是 "evaluation/analysis paper"，不引入新方法（除非加 method component）
- nrehiew 博客已经给出了初步 signal（OPD 遗忘少）
- CMU 论文的直接扩展，可能被视为 "incremental"
- 如果结论是 "OPD 泛化性跟 RL 一样好"（expected result），novelty 偏低

### Novelty 取决于实验结果

| 实验结果场景 | Novelty | Framing |
|-------------|---------|---------|
| OPD ≈ RL（泛化好） | 6/10 | 确认性研究，但填补空白，practical guidance |
| OPD < RL, > SFT（中间地带） | 8/10 | 发现 "dense signal tax" — OPD 的 teacher logit 引入额外漂移 |
| OPD > RL（泛化更好） | 9/10 | 强发现，OPD 是最佳 post-training 策略 |
| OPSD ≠ OPD（自蒸馏 vs 外部教师不同） | 8/10 | 新发现维度 |
| OPD 在不同任务类型上泛化性分化 | 8/10 | "task-dependent generalization" 的分析 |

---

## 4. 最接近的竞争工作

### 4.1 直接竞争

| 工作 | 竞争程度 | 差异 |
|------|---------|------|
| CMU (2507.00432) | 🔴 最近 | 他们没测 OPD；我们的是自然扩展 |
| nrehiew blog (2026-05-10) | 🟡 中等 | 仅1个任务，非正式出版物；如果作者正在写论文是最大风险 |
| SDFT (2601.19897) | 🟢 低 | 关注 continual learning 不是 cross-domain 泛化 |
| Kim et al. (2603.24472) | 🟢 低 | 仅数学域内 OOD |

### 4.2 最大风险：nrehiew 博客作者

- 2026-05-10 发布的博客已经展示了 OPD 遗忘更少
- 如果此作者正在准备一篇 full paper，可能与我们高度重叠
- **缓解策略**: 我们做多域系统性研究（12+ benchmarks），远超博客的1个任务；我们加 representation analysis（PCA shift, token-level KL）

---

## 5. Experiment Design

### 5.1 核心实验框架

**目标**: 在 CMU 论文 (2507.00432) 完全相同的评估框架下，系统比较 SFT / RL / OPD / OPSD 四种 post-training 方法对通用能力的影响。

### 5.2 训练配置

| 参数 | 设置 |
|------|------|
| **基座模型** | Qwen2.5-7B-Base, Qwen3-8B-Base（双模型验证） |
| **训练域** | 数学推理（与CMU对齐） |
| **训练数据** | 47K math problems (DeepScaler + SimpleRL level 3-5)，与CMU相同 |
| **教师模型** | Qwen3-32B-Instruct（OPD用）；自身（OPSD用） |
| **训练方法** | |
| — SFT | 标准 SFT on teacher responses (off-policy) |
| — On-policy SFT | Rejection sampling from student, SFT on correct |
| — RL (GRPO) | 标准 GRPO with math reward |
| — OPD | On-policy sampling + teacher logit distillation (reverse KL) |
| — OPSD | On-policy sampling + self-distillation (conditioned on correct answer) |
| **训练量** | 每种方法 3 epochs, ~2000 steps |
| **Seeds** | 3 random seeds per method |

### 5.3 评估 Benchmarks（复用CMU框架 + 扩展）

**数学域（In-Domain）**:
- MATH500, AIME24, AIME25, OlympiadBench

**跨域推理（Other Reasoning）**:
- LiveCodeBench v2 (编程)
- GPQA-Diamond (科学QA)
- ACPBench (Agent规划)
- HeadQA (医学QA)

**通用能力（Non-Reasoning）**:
- IFEval (指令遵循)
- CoQA (对话理解)
- HaluEval (幻觉检测)
- MC-TACO (时间常识)

**扩展评估**:
- MMLU-Pro (通用知识)
- Arena-Hard (开放对话)

### 5.4 分析维度（对标CMU论文的分析方法）

| 分析 | 方法 | 复用CMU？ |
|------|------|----------|
| **Transferability Index** | TI_other, TI_non 计算 | ✅ 完全复用 |
| **Representation Shift** | PCA on hidden states, shift 距离 | ✅ 完全复用 |
| **Token-level KL** | KL(π_tuned ∥ π_base) per benchmark | ✅ 完全复用 |
| **Token Rank Shift** | top-5 token rank change | ✅ 完全复用 |
| **Epistemic Markers** | 12 epistemic token 频率变化 | 🆕 新增（借鉴Kim et al.） |
| **Dense Signal Analysis** | 比较 OPD 的 logit supervision gradient 与 RL reward gradient | 🆕 新增 |

### 5.5 计算资源估算

| 阶段 | 配置 | 时间 |
|------|------|------|
| SFT 训练 (7B) | 1×H200, 47K samples | ~2h |
| RL/GRPO 训练 (7B) | 4×H200 (rollout需要多卡) | ~6h |
| OPD 训练 (7B, 需要teacher) | 4×H200 (student) + 4×H200 (teacher 32B) | ~8h |
| OPSD 训练 (7B) | 4×H200 | ~6h |
| 评估 (14 benchmarks × 5 methods × 3 seeds) | 8×H200 parallel | ~12h |
| **总计** | 24×H200 | **~36h（1.5天）** |

API 费用（用 GPT-4o 评估 Arena-Hard 等）: ~$200

### 5.6 时间表

| 天 | 任务 |
|----|------|
| Day 1 (5/16) | 搭建实验框架，准备数据，跑 SFT baseline |
| Day 2 (5/17) | 跑 RL/OPD/OPSD 训练，开始评估 |
| Day 3 (5/18) | 完成所有评估，跑 representation analysis |
| Day 4 (5/19) | 写论文 (intro, method, results) |
| Day 5 (5/20) | 写论文 (analysis, related work), 润色提交 |
| Buffer (5/21-25) | 补充实验, camera-ready |

---

## 6. 潜在的 Reviewer 反对意见

### 6.1 最大风险：Novelty / Contribution

> **"This is just an evaluation study. Where is the method contribution?"**

**应对策略**:
- Frame 1: 如果发现 OPD 泛化性好 → "provide practical guidance for industrial post-training"，与CMU论文同等贡献水平
- Frame 2: 如果发现 interesting pattern → 提出 "Dense Signal Tax" 概念 + 简单的 adaptive weighting method
- Frame 3: 加入 "Representation-Aware OPD" 作为 method contribution — 基于 representation shift 分析自适应调整 teacher signal 强度

### 6.2 "Only tested on math training"

> **"You only train on math and test generalization. What about training on code/NLP and testing math?"**

**应对策略**: 
- 与CMU论文对齐（他们也只训练数学）
- 如果时间允许，加一组 code training → math/NLP generalization 实验

### 6.3 "The blog already showed OPD forgets less"

> **"The nrehiew blog already showed this. What's new?"**

**应对策略**:
- 博客只测了1个任务，我们测14个
- 博客没有 representation analysis，我们有
- 博客是非正式出版物，我们提供rigorous experimental design with seeds

### 6.4 "Expected result: OPD is on-policy so it should generalize like RL"

> **"Given RL's Razor theory, it's obvious OPD should generalize like RL."**

**应对策略**:
- OPD 不完全是 on-policy RL — 它有 dense teacher signal，这可能增加 distribution shift
- G-OPD 虽然理论上统一了 OPD 和 RL，但 dense logit supervision 在实践中的效果需要验证
- 如果 OPD < RL 泛化性，说明 dense signal 确实引入了额外 "tax"
- CMU 论文发现 on-policy SFT 已经比 off-policy SFT 好，但 OPD 的 dense logit 到底是帮忙还是帮倒忙？

### 6.5 "Model/scale-specific, doesn't generalize"

**应对策略**: 用两个不同的基座模型（Qwen2.5-7B, Qwen3-8B）验证

---

## 7. Paper Framing — 这是什么类型的论文？

### 7.1 类型：Analysis Paper with Practical Implications

这是一篇 **empirical analysis paper**，不是 method paper。

### 7.2 EMNLP 对 Analysis Paper 的接受度

**历史先例** (EMNLP 2025 接收的 analysis papers):
- "An Empirical Study of LLM Reasoning Ability Under Strict Output Length Constraint"
- "Understanding the Modality Gap: An Empirical Study on Speech-Text Alignment"
- 多篇 "When Does X Work?" 类型的论文

**EMNLP 特点**:
- EMNLP = **Empirical** Methods in NLP
- 对 rigorous empirical analysis 的接受度显著高于 ACL/ICLR
- 但纯 evaluation 仍然需要有 **insight** 或 **actionable takeaway**
- 接受率：Main 22.16%, Findings 17.34% (EMNLP 2025)

### 7.3 契合 EMNLP 2026 Special Theme

> **"New Missions for NLP Research" — rethinking evaluation, agentic systems, data responsibility, LLMs as research tools**

我们的论文完美契合 "rethinking evaluation" — 重新评估 OPD 这个工业标准方法的副作用。

### 7.4 推荐 Framing

**Title Options** (按推荐度排序):
1. **"The Generalization Tax: How Post-Training Methods Affect General Capabilities"** — 最广，包含 SFT/RL/OPD 全对比
2. **"Beyond Math: Does On-Policy Distillation Preserve General Capabilities?"** — 直指核心问题
3. **"On-Policy Distillation Through a Generalization Lens"** — 学术风格

**推荐**: Option 1 或 2。

**Narrative Arc**:
1. **Opening**: OPD 已成为工业标准（Qwen3, GLM-5, DeepSeek-V4），但我们对其副作用知之甚少
2. **Gap**: CMU 发现 RL >> SFT 泛化性，但 OPD 在哪？
3. **Study**: 系统性 4 方法 × 14 benchmark × 2 模型 × 3 seeds 研究
4. **Findings**: [取决于实验结果]
5. **Insight**: 解释为什么 OPD 表现如此（representation analysis + token-level analysis）
6. **Implication**: 工业界 post-training recipe 的实践建议

---

## 8. 增强策略 — 提升到 Method Paper

如果纯 analysis paper 不够，可以加入以下 method components：

### Option A: Representation-Aware OPD (RA-OPD)

基于 representation shift 分析，自适应调整 teacher signal 强度：
- 如果当前 batch 在非训练域上的 hidden state shift 超过阈值 → 降低 teacher KL weight
- 类似 adaptive regularization，但专门针对泛化性

### Option B: Domain-Balanced OPD (DB-OPD)

在 OPD 训练中混入少量非训练域数据的 KL penalty：
- 每个 batch 采样一些非训练域 prompt → 计算与 base model 的 KL → 加到 loss 中
- 类似 continual learning 的 rehearsal，但更轻量

### Option C: 泛化性预测器

基于实验数据拟合一个简单模型：给定任务特征（答案确定性、推理链长度等），预测 OPD 在该任务上的泛化性退化程度。

**推荐**: Option A 最简单、最容易实现（1天内）、最 elegant。

---

## 9. Risk Assessment

| 风险 | 概率 | 严重性 | 缓解 |
|------|------|--------|------|
| nrehiew 博客作者同时在写论文 | 30% | 🔴 高 | 我们做 14 benchmark + representation analysis，远超博客深度 |
| 实验结论是 "OPD ≈ RL"（expected） | 40% | 🟡 中 | 加 method component (RA-OPD)，转为 "empirical + method" |
| 训练框架bug导致延迟 | 20% | 🟡 中 | 用成熟框架（TRL/OpenRLHF），先跑小规模验证 |
| Reviewer 认为纯 analysis 不够 | 35% | 🟡 中 | 加 method component，或投 Findings |
| 5天时间不够 | 15% | 🟢 低 | 核心实验仅需 1.5 天，写作 2 天 |
| GPU 资源竞争 | 10% | 🟢 低 | 24×H200 足够 |

---

## 10. 与其他 Route 的对比

| 维度 | Route A (OPD泛化) | Idea A from FINAL_IDEAS (EPSD) | Idea B (跨任务SD) |
|------|-------------------|-------------------------------|-------------------|
| Novelty | 7/10 | 已被用户否决（trivially expected） | 7/10 |
| 清晰度 | ★★★★★ | - | ★★★★ |
| 可行性 (5天) | ★★★★★ | - | ★★★★ |
| EMNLP 契合 | ★★★★★ | - | ★★★★ |
| 风险 | Expected result risk | - | "empirical study" risk |
| Method 贡献 | 可选 (RA-OPD) | - | Task-Adaptive SD |

---

## 11. 最终推荐

### 评分: **推荐 (7.5/10)**

**理由**:
1. ✅ **Gap 是真实的** — CMU 论文没测 OPD，这是事实性空白
2. ✅ **高度可行** — 复用 CMU 评估框架，5天完全可完成
3. ✅ **工业相关性极高** — Qwen3/GLM-5/DeepSeek-V4 都用 OPD
4. ✅ **EMNLP 契合度高** — empirical analysis + special theme
5. ⚠️ **Novelty 取决于实验结果** — 如果 OPD ≈ RL（expected），需要 method component 增强
6. ⚠️ **竞争风险存在** — nrehiew 博客作者可能在准备论文

### 关键决策点

> **是否走这个方向取决于用户的风险偏好:**
> 
> - **保守选择**: 走 Route A + 加 RA-OPD method component = 7.5/10 的有保障的论文
> - **激进选择**: 先跑24h pilot experiments，如果发现 OPD 有 unexpected behavior → 9/10 的强论文；如果 OPD ≈ RL → 回退到 method component
> 
> **我的建议**: 开始 Route A pilot experiments（Day 1），同时 team lead 评估其他 Routes。如果 Day 1 结束时实验结果 interesting → all-in Route A；如果 boring → 根据其他 Routes 的调研结果切换。

### 如果走 Route A 的立即行动

1. 下载 CMU 论文的评估代码（如果开源）
2. 准备 Qwen2.5-7B + Qwen3-8B 基座模型
3. 搭建 OPD 训练框架（基于 TRL 或 OpenRLHF）
4. 跑 SFT baseline（最快，2h）→ 立即评估 → 看 CMU 结论是否可复现
5. 跑 OPD 和 RL 训练 → 评估 → 比较

---

## 参考文献清单

1. CMU (2507.00432) — "Does Math Reasoning Improve General LLM Capabilities?"
2. nrehiew blog (2026-05-10) — "SFT, RL, and OPD Through a Distributional Lens"
3. RL's Razor (2509.04259, ICLR 2026) — "Why Online RL Forgets Less"
4. SDFT (2601.19897, MIT) — "Self-Distillation Enables Continual Learning"
5. Kim et al. (2603.24472) — "Why Does Self-Distillation Sometimes Degrade Reasoning"
6. Revisiting OPD (2603.25562) — "Empirical Failure Modes and Simple Fixes"
7. G-OPD (2602.12125) — "Learning beyond Teacher: Generalized OPD"
8. PRISM (2604.28123) — "Pre-alignment via Black-Box OPD"
9. X-OPD (2603.24596) — "Cross-Modal OPD for Speech LLMs"
10. OPSD (2601.18734, ICML 2026) — "On-Policy Self-Distillation"
11. EGRSD (2605.13255) — "Entropy Gate for OPD"
12. Song & Zheng survey (2604.00626) — "Unified f-divergence Framework"
