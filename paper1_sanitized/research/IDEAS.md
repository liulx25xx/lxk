# EMNLP 2026 Paper 1 - Research Ideas

## 项目信息

- **会议**: EMNLP 2026 (Budapest, Hungary, Oct 24-29)
- **Deadline**: ARR May 2026 cycle → **2026-05-25 AoE** (10天内!)
- **格式**: Long paper 8页 (camera-ready 9页), ACL style
- **模板**: 已下载到 `template/`
- **资源**: 24×H200 GPU, ~$1000 API credits
- **时间**: 5天完成论文 (实验+写作)
- **Special Theme**: "New Missions for NLP Research"

---

## 调研总结：2025-2026 热门方向

| 方向 | 热度 | 竞争度 | 5天可行性 |
|------|------|--------|-----------|
| Test-time Compute Scaling | ★★★★★ | 极高(已有survey 120+篇) | 中 |
| Process Reward Models (PRM) | ★★★★★ | 高 | 中 |
| LLM-as-Judge 去偏 | ★★★★ | 中高 | **高** |
| Multi-Agent Debate/Collaboration | ★★★★ | 中 | **高** |
| Code Agent / SWE-bench | ★★★★ | 高 | 低(需大量工程) |
| Synthetic Data Quality/Collapse | ★★★★ | 中 | **高** |
| RAG vs Long Context | ★★★ | 高(ICLR 2025已有) | 中 |
| Speculative Decoding/KV Cache | ★★★ | 中高 | 低(系统级) |
| Weak-to-Strong Generalization | ★★★ | 中 | 中 |

---

## 推荐 Ideas (按可行性×新颖性排序)

### Idea 1: 🏆 "Judge the Judges: Cross-Model Calibration for Reliable LLM Evaluation"
**方向**: LLM-as-Judge 可靠性 + Meta-Evaluation

**核心观点**: 当前 LLM-as-Judge 存在系统性偏差(位置、长度、自我偏好)。我们提出一个 **cross-model calibration framework**：用多个异构 judge 模型的分歧信号来检测并修正评估偏差，无需人类标注。

**方法**:
1. 构建 judge ensemble (GPT-4o, Claude, Gemini, Llama, Qwen 等)
2. 分析 judge 间的 disagreement patterns → 发现系统性 bias
3. 提出 calibration 算法：基于 judge 分歧的置信度加权
4. Benchmark: 在 MT-Bench, AlpacaEval, Arena-Hard 上验证

**为什么适合**:
- 5天完成：主要是 API 调用 + 分析，不需要训练
- 资源匹配：$1000 API 够调多个商业模型
- EMNLP 风格：empirical + evaluation + analysis
- 契合 Special Theme："Rethinking Progress & Evaluation"
- NeurIPS 2025 有 RBD 论文(bias debiasing)但聚焦单模型微调，我们的 cross-model 角度是新的

**预期贡献**:
- 首个系统性 cross-model judge disagreement 分析
- 无监督 calibration 方法
- 开源 judge-disagreement benchmark

---

### Idea 2: 🥈 "When Debate Fails: Adversarial Dynamics in Multi-Agent LLM Reasoning"
**方向**: Multi-Agent + Reasoning Robustness

**核心观点**: Multi-agent debate 被认为能提升推理质量，但在对抗性场景下表现如何？我们系统研究 debate 中的 **persuasion vulnerability**：一个有说服力但错误的 agent 能否带偏整个系统？

**方法**:
1. 构建 multi-agent debate 框架 (3-5 agents)
2. 注入 adversarial agent (有说服力但答案错误)
3. 分析 debate dynamics：什么情况下正确答案被推翻
4. 提出 defense mechanism：基于 reasoning chain consistency 的投票策略

**为什么适合**:
- Nature 2026 April 刚发了一篇相关论文(persuasion-driven adversarial)，说明方向很热
- 5天：纯 API 实验，不需要训练
- EMNLP 风格：analysis + robustness
- 契合 Special Theme："From Models to Systems & Ecosystems"

**预期贡献**:
- 首个系统性 adversarial robustness 分析 for multi-agent debate
- Persuasion vulnerability taxonomy
- Defense mechanisms with reasoning consistency checks

---

### Idea 3: 🥉 "The Synthetic Data Paradox: When Self-Generated Training Data Hurts"  
**方向**: Synthetic Data Quality + Model Collapse

**核心观点**: 越来越多模型用自生成数据训练(self-play, SPIN等)，但存在 "model collapse" 风险。我们研究 **何时、为何、多大程度上** 合成数据会损害模型多样性和能力。

**方法**:
1. 设计 controlled experiments：不同比例真实/合成数据混合
2. 多维度评估：diversity, factuality, creativity, instruction-following
3. 发现 tipping point：合成数据超过什么比例开始有害
4. 提出 quality filter + diversity regularization 方法

**为什么适合**:
- 需要一些训练实验 → 24×H200 正好
- 开源模型(Llama-3, Qwen-2.5)做实验
- EMNLP 风格：empirical study + analysis
- 契合 Special Theme："Data as a Bottleneck & Responsibility"

**风险**: 实验可能需要较多训练时间，5天紧张

---

### Idea 4: "Process Reward Without Process Labels: Self-Supervised Step Verification"
**方向**: Process Reward Model + Self-Supervision

**核心观点**: PRM 需要昂贵的 step-level 标注。我们提出用 **outcome-driven self-supervision** 自动生成 step labels：如果改变某步后 outcome 变差，说明该步重要且正确。

**方法**:
1. 对推理链每步做 counterfactual perturbation
2. 观察 outcome 变化 → 反推 step importance/correctness
3. 用这些自动标签训练 PRM
4. 在 GSM8K, MATH, AIME 上验证

**为什么适合**:
- PRM 极度热门(ICLR 2025 oral, EMNLP 2025 accepted)
- 需要中等规模训练 → H200 可以
- 新颖角度：不需要人工标注

**风险**: 类似想法可能已有(需仔细查 novelty)

---

### Idea 5: "Adaptive Test-Time Compute: When to Think More, When to Think Less"
**方向**: Test-time Scaling + Efficiency

**核心观点**: 当前 test-time scaling 对所有问题同等投入计算。我们提出 **adaptive compute allocation**：用一个轻量 router 判断问题难度，决定分配多少 test-time compute。

**方法**:
1. 训练 difficulty router (小模型，预测需要多少 reasoning steps)
2. Easy questions → direct answer; Hard questions → more compute (CoT, voting, search)
3. 在固定 compute budget 下最大化 overall accuracy
4. Pareto analysis: accuracy vs. compute trade-off

**为什么适合**:
- Test-time scaling 是 2025-2026 最热方向
- ICLR 2025 oral paper 奠定基础
- 实验相对轻量：用现有推理模型做 inference
- 但方向已经很挤(survey 120+ papers)，需要很精确的切入点

---

## 综合推荐

| 排名 | Idea | 5天可行性 | 新颖性 | EMNLP契合 | 推荐理由 |
|------|------|-----------|--------|-----------|----------|
| **1** | Judge Calibration | ★★★★★ | ★★★★ | ★★★★★ | API-only, 分析为主, 完美契合 special theme |
| **2** | Adversarial Debate | ★★★★★ | ★★★★ | ★★★★ | API-only, 新颖角度, 时效性强 |
| **3** | Synthetic Data Paradox | ★★★ | ★★★★ | ★★★★★ | 需要训练但契合 theme, 影响力大 |
| **4** | Self-Supervised PRM | ★★★ | ★★★★★ | ★★★ | 新颖但紧张, 更适合 NeurIPS/ICML |
| **5** | Adaptive Test-Time | ★★★★ | ★★★ | ★★★ | 太挤, 可能撞车 |

---

## 我的建议

**首选 Idea 1 (Judge Calibration)** 或 **Idea 2 (Adversarial Debate)**：
- 两者都是纯 API 实验，5天绝对能完成
- 不需要 GPU 训练，API 预算够用
- 都是 empirical + analysis 导向，完美匹配 EMNLP
- 可以 24×H200 用于跑开源模型作为 judge/agent

如果想做更 "硬核" 的工作，选 **Idea 3 (Synthetic Data)**，但时间风险更大。

---

## 下一步

请选择一个 idea，我将：
1. 进行 novelty check (确保不撞车)
2. 设计详细实验方案
3. 搭建代码框架
4. 撰写论文
