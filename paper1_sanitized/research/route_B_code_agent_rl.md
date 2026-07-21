# Route B: Code Agent + RL 领域深度调研报告

> 调研时间: 2026-05-15 | 目标: EMNLP 2026 (deadline 2026-05-25)
> 约束: 5天完成 | 24×H200 GPU | $1k API credits

---

## 一、领域综述：Code Agent + RL 当前格局

### 1.1 SWE-bench SOTA 进展

SWE-bench Verified (500 human-validated instances) 是当前 code agent 的主要基准。过去2年经历了 **41× 的性能飞跃**：

| 时间 | 系统 | Resolve Rate |
|------|------|-------------|
| 2023-10 | Claude 2 + SWE-agent | 1.96% |
| 2024-03 | SWE-agent + GPT-4 | 12.5% |
| 2024-05 | Devin | 13.8% |
| 2024-06 | AutoCodeRover | 19% |
| 2024-08 | OpenHands + Claude 3.5 | 27% |
| 2024-11 | Agentless + GPT-4o | 38.4% |
| 2025-03 | Claude Opus 4 + Aider | 55.2% |
| 2025-09 | Claude Sonnet 4.5 | 70.8% |
| **2025-12** | **Claude Opus 4.5 + live-SWE-agent** | **79.2%** |

**关键观察**：
- 同一模型（Claude Opus 4.5）在不同 scaffold 下得分差异可达 3-4%，说明 **scaffold 设计仍然重要**
- Top 系统均使用闭源 API 模型，**开源模型差距巨大**
- Bash-only 设置下 mini-SWE-agent 达 74%，说明简单 scaffold + 强模型即可

### 1.2 Agent 框架主要流派

| 流派 | 代表 | 方法 | 优势 | 局限 |
|------|------|------|------|------|
| **Agentic** | SWE-agent, OpenHands, Devin | 自主探索、执行、调试循环 | 灵活、可处理复杂问题 | token 消耗大、error propagation |
| **Agentless** | Agentless, Kimi-Dev | 预定义 pipeline：定位→修复→验证 | 高效、可控 | 缺乏灵活性 |
| **Hybrid** | Kimi-Dev (skill prior) | Agentless 训练作为 skill prior + agentic 推理 | 两者优势结合 | 复杂度高 |
| **CodeAct** | OpenHands CodeAct 2.1 | 用 Python/bash 代码替代 JSON tool calls | 单步可执行复杂逻辑 | 需要安全沙箱 |

**Kimi-Dev** (2025.12) 提出关键洞察：Agentless training 可作为 "skill prior"，先教模型定位和修复的基础能力，再用 agentic 框架释放灵活性。SWE-bench Verified 达 **60.4%**（开源 workflow 最佳）。

### 1.3 开源模型训练进展

| 系统 | 模型 | 训练方法 | SWE-bench Verified | 关键创新 |
|------|------|----------|-------------------|----------|
| **SWE-agent-LM-32B** | Qwen2.5-72B→32B | SFT on SWE-smith 50k | 40.2% | 大规模合成数据 |
| **DeepSWE-Preview** | Qwen3-32B | **纯 RL (GRPO++)** | 42.2% (pass@1), **59% (TTS)** | 无需SFT，纯RL训练 |
| **Kimi-Dev** | 内部模型 | Mid-training + SFT + Agent | 60.4% | Agentless as skill prior |
| **Self-play SWE-RL** | LLM | RL self-play (inject+repair) | +10.4% improvement | 无需人工标注数据 |
| **mini-coder-4b** | Qwen3-4B | SFT (蒸馏) | 26.8% (pass@1), 60.2% (pass@100) | 4B参数媲美120B |
| **SWE-agent-LM-7B** | 7B model | SFT | 15.2% | 最小可用模型 |

### 1.4 RL for Agent Tasks：关键工作

#### (a) Multi-turn Agent RL

| 工作 | 算法 | 环境 | 关键贡献 |
|------|------|------|----------|
| **GiGPO** (NeurIPS'25) | 两层分组优势估计 | ALFWorld, WebShop, Search QA | Episode + Step 级 credit assignment，无需 critic |
| **RAGEN/StarPO** | 轨迹级策略优化 | 多轮决策任务 | 第一个系统研究 multi-turn agent RL 的框架 |
| **DeepSWE** | GRPO++ | SWE-bench (R2E-Gym) | 纯 RL 无 SFT，涌现出边界情况思考 |
| **AgentGym-RL** | ScalingInter-RL | 多种真实环境 | 匹配或超越商业模型 |
| **verl-agent** | GiGPO/GRPO/PPO/DAPO | ALFWorld, WebShop, Sokoban | 统一的 agent RL 训练框架 |

#### (b) Tool-Integrated Reasoning RL

| 工作 | 方法 | 任务 | 关键结果 |
|------|------|------|----------|
| **Search-R1** | RL + search engine | QA with retrieval | 3B/7B 模型显著超越基线 |
| **ReTool** | RL + code execution | AIME math | 32B模型67%准确率，400步 |
| **TORL/STILL-1** | RL + tool calls | Math reasoning | Tool-integrated reasoning |

#### (c) Agent Efficiency

| 工作 | 方法 | 关键发现 |
|------|------|----------|
| **Agent-Omit** (ICML'26) | RL 训练自适应省略 thought/observation | 不牺牲性能的情况下提高效率 |
| **AgentPRM** | Agent 任务的 Process Reward Model | 捕获决策间依赖关系和每步贡献 |

### 1.5 Self-Debugging / Error Recovery 研究现状

| 工作 | 方法 | 关键发现 |
|------|------|----------|
| **Revisit Self-Debugging** (ACL'25) | Post-execution vs In-execution | Post-execution 用自生成测试反而**降低**性能（测试偏差）；In-execution（利用中间状态）显著有效 |
| **AgentDebug** | 模块化失败分类 + debugging framework | Memory/Reflection/Planning/Action/System 五类失败；+24% all-correct accuracy |
| **PyCapsule** | 双agent pipeline + self-debugging | 自动代码修复框架 |
| **Agent-R** | 自反思 agent 训练 | 训练 agent 学会自我纠错 |
| **MASFT** (Multi-Agent System Failure Taxonomy) | 失败经验分类 | 第一个系统性 multi-agent 失败分类 |

**关键 Gap**：
- Self-debugging 的效果高度依赖**反馈信号质量**，自生成测试不可靠
- 缺乏系统研究：**什么类型的错误**可以自修复，什么不行
- Agent error recovery 主要是 prompting-based，**RL-based error recovery** 几乎空白
- 没有工作系统研究 **code agent 在不同错误类型下的 recovery 能力**

---

## 二、5-8 个具体可做的 Idea

### Idea 1: **RL for Agent Error Recovery: Teaching Code Agents to Recover from Their Own Mistakes**

**核心想法**：系统研究 code agent 的 error recovery 能力，并用 RL 训练 agent 学会从错误中恢复。构建一个包含不同错误类型（定位错误、编辑错误、测试理解错误、环境交互错误）的 benchmark，分析哪些错误可恢复/不可恢复，然后用 RL 训练 recovery 策略。

**Novelty 分析**：
- AgentDebug 做了失败分类但**没有训练 recovery**
- Self-debugging 研究了 post/in-execution 但**没有 RL training**
- DeepSWE/SWE-RL 做了 RL 但**没有专门研究 error recovery**
- 这是第一个将 **error taxonomy + RL recovery training** 结合的工作

**与现有工作区别**：
- vs AgentDebug: 他们分类+prompting修复，我们 RL 训练 recovery
- vs Self-debugging (ACL'25): 他们研究 when effective，我们训练 agent 学会 effective recovery
- vs DeepSWE: 他们端到端 RL，我们专注 recovery 阶段的 RL

**5天可行性**：★★★★☆
- Day 1: 构建 error injection framework（在 SWE-bench 轨迹中注入不同类型错误）
- Day 2: 分析不同模型在不同错误类型下的 recovery 率（empirical study）
- Day 3: 设计 recovery-focused RL training（用 GiGPO/GRPO）
- Day 4: 训练 + 评估（7B/14B 模型，单卡即可）
- Day 5: 写论文

**实验资源**：API calls 分析 + 小模型 fine-tuning，完全在预算内

---

### Idea 2: **The Anatomy of Agent Failures: A Systematic Study of When and Why Code Agents Fail to Self-Correct**

**核心想法**：大规模 empirical study，系统分析 code agent 的 self-correction 行为。在 SWE-bench 上运行多个 agent（不同模型×不同框架），收集所有失败轨迹，建立细粒度失败分类，回答：(1) 什么类型的错误 agent 能自我修复？(2) Self-correction 的 success/failure pattern 是什么？(3) 模型大小、框架类型如何影响 recovery？

**Novelty 分析**：
- Revisit Self-Debugging (ACL'25) 只研究了代码生成的 self-debugging，不是 agent 层面
- AgentDebug 做了分类但规模小，没有跨模型跨框架的系统对比
- **第一个大规模、跨模型、跨框架的 code agent self-correction empirical study**

**与现有工作区别**：
- vs ACL'25 Self-Debugging: 他们是 code generation level，我们是 agent level（multi-step）
- vs AgentDebug: 他们3个benchmark各一个agent，我们是 SWE-bench 上 N个agent的大规模分析
- vs SWE-bench 本身: 他们只报resolve rate，我们分析 failure patterns

**5天可行性**：★★★★★
- Day 1-2: 运行多个 agent 在 SWE-bench subset 上（API calls），收集轨迹
- Day 2-3: 标注和分析失败轨迹，建立分类体系
- Day 4: 深度分析 + 统计
- Day 5: 写论文

**实验资源**：主要是 API credits，$1k 足够运行 ~500 个 SWE-bench 实例 × 3-4 个模型

---

### Idea 3: **Credit Where Credit Is Due: Step-Level Reward for Code Agent RL Training**

**核心想法**：现有 code agent RL（DeepSWE, SWE-RL）使用 sparse outcome reward（测试通过=1，否则=0），但 code agent 轨迹通常有 20-100 步。提出一种 **step-level reward model** 专门针对 code agent，利用中间状态（文件编辑、测试结果、代码diff）提供细粒度奖励信号。

**Novelty 分析**：
- AgentPRM 提出了 agent PRM 概念但**没有在 code agent / SWE-bench 上验证**
- GiGPO 做了 step-level credit assignment 但通过算法层面，**没有训练 reward model**
- CodePRM (ACL'25 Findings) 做了 code reasoning 的 PRM 但**不是 agent 任务**
- 第一个 **code agent 专用的 step-level reward model + RL training**

**与现有工作区别**：
- vs AgentPRM: 他们是 general agent tasks (ALFWorld/WebShop)，我们是 code/SWE
- vs GiGPO: 他们是算法级 credit assignment，我们是 learned reward model
- vs DeepSWE: 他们用 sparse ORM，我们用 dense step-level rewards

**5天可行性**：★★★☆☆
- 需要训练 reward model + RL，时间紧张
- 可行的简化方案：用 **heuristic step-level rewards**（基于中间测试通过率、代码语法检查等）代替 learned PRM

---

### Idea 4: **Scaling Down Code Agents: How Small Can We Go?**

**核心想法**：系统研究将 code agent 能力压缩到极小模型的方法和极限。mini-coder-4b 已经展示了 SFT 蒸馏的潜力，但 **RL 训练小模型 code agent** 几乎没人做过。对比不同训练方法（SFT 蒸馏 vs RL vs SFT+RL）在不同模型大小（1.5B, 3B, 7B, 14B）上的效果。

**Novelty 分析**：
- mini-coder 做了 SFT 蒸馏但**没做 RL**
- DeepSWE 做了 RL 但只在 32B 模型上
- SWE-agent-LM 做了 7B SFT 但效果有限 (15.2%)
- **第一个系统对比 SFT vs RL vs SFT+RL 在小模型 code agent 上的效果**

**与现有工作区别**：
- vs mini-coder: 他们 SFT only，我们加 RL
- vs DeepSWE: 他们只做 32B，我们系统研究 1.5B-14B
- vs SWE-agent-LM: 他们 SFT only 7B，我们多方法多模型

**5天可行性**：★★★★☆
- 需要在多个模型大小上训练，但可以用 verl-agent 框架
- 24×H200 足够并行训练多个小模型
- 简化方案：专注 3B 和 7B，SFT vs RL vs SFT+RL 三条线

---

### Idea 5: **From Sparse to Dense: Bootstrapping Step Rewards for Code Agent Training**

**核心想法**：现有 code agent RL 面临 sparse reward 问题（只在最终测试时给奖励）。提出一种 **reward bootstrapping** 方法：利用 agent 轨迹中的自然中间信号（代码编译成功、测试子集通过、lint 通过等）自动构造 dense reward，无需训练额外 reward model。然后研究 dense reward 如何影响 RL 训练效率和最终性能。

**Novelty 分析**：
- 不同于 AgentPRM（需训练 reward model），这里用 **rule-based 中间信号**
- 不同于 GiGPO（算法层面），这里是 **reward signal 层面**
- DeepSWE 的 compact filtering 是避免坏样本，我们是积极提供好信号

**5天可行性**：★★★★☆
- Day 1: 设计 reward signals（编译、lint、partial test、diff quality）
- Day 2-3: 在 R2E-Gym 子集上训练（7B 模型），对比 sparse vs dense
- Day 4: 分析 + ablation
- Day 5: 写论文

---

### Idea 6: **The Self-Play Curriculum: Difficulty-Aware Bug Generation for Code Agent Training**

**核心想法**：Self-play SWE-RL (2025.12) 展示了 agent 自己生成 bug 并修复的训练范式，但**没有控制难度**。提出 difficulty-aware self-play curriculum：agent 先在简单 bug 上训练，逐步增加难度，通过 **bug 复杂度估计器** 控制课程进度。

**Novelty 分析**：
- Self-play SWE-RL 没有难度控制
- 课程学习在 RL 中经典，但在 code agent self-play 中是新的
- 结合 code complexity metrics（cyclomatic complexity, 修改文件数等）

**5天可行性**：★★★☆☆
- Self-play 训练本身需要较大计算资源
- 可行的简化方案：用已有 SWE-smith 数据集，按难度排序做 curriculum SFT/RL

---

### Idea 7: **Agent Error Cascades: Understanding and Breaking Compounding Failures in Code Agents**

**核心想法**：Code agent 的一个核心问题是 **error cascade**——一个早期错误（如错误定位）导致后续所有步骤都是浪费的。系统研究 cascade 的传播机制，并提出简单的 **checkpoint-and-backtrack** 策略：在关键决策点设置检查点，当检测到错误时回退到最近检查点。

**Novelty 分析**：
- AgentDebug 分类了 cascading failures 但**没有提出 backtrack 机制**
- 现有 agent 框架没有 systematic backtracking
- 结合了 error detection + recovery mechanism

**5天可行性**：★★★★★
- 主要是 API-based 实验
- Day 1-2: 分析 SWE-bench 失败轨迹中的 cascade patterns
- Day 3: 实现 checkpoint-and-backtrack 在 mini-SWE-agent 上
- Day 4: 评估，对比有/无 backtrack 的 resolve rate
- Day 5: 写论文

---

### Idea 8: **Agentless Skills as RL Warm-Start: Bridging Workflow and Agent Training**

**核心想法**：Kimi-Dev 展示了 Agentless training 作为 skill prior 的价值，但他们只用了 SFT。我们提出 **Agentless SFT → Agent RL** 的两阶段训练方案：先用 Agentless 数据 SFT 教会基础能力（定位、修复），再用 Agent 框架做 RL 训练释放灵活性。系统对比这种两阶段方法与纯 SFT、纯 RL 的效果。

**Novelty 分析**：
- Kimi-Dev 只做了 SFT stages，**没有加 RL**
- DeepSWE 只做了纯 RL，**没有 Agentless SFT warm-start**
- 这是两条路线的 **自然交汇点**，但没人做过

**5天可行性**：★★★☆☆
- 需要 SFT + RL 两阶段训练
- 如果使用小模型（7B）+ 已有框架（verl-agent），时间可行但紧张

---

## 三、排序推荐（Novelty × 可行性）

| 排名 | Idea | Novelty | 可行性 | 综合分 | 适合类型 |
|------|------|---------|--------|--------|----------|
| **1** | **Idea 2: Self-Correction Empirical Study** | ★★★★☆ | ★★★★★ | **9/10** | Empirical + Analysis (EMNLP 最爱) |
| **2** | **Idea 7: Error Cascades + Backtrack** | ★★★★☆ | ★★★★★ | **9/10** | Analysis + Method |
| **3** | **Idea 1: RL for Error Recovery** | ★★★★★ | ★★★★☆ | **9/10** | Method + Analysis |
| 4 | Idea 5: Dense Reward Bootstrapping | ★★★★☆ | ★★★★☆ | 8/10 | Method |
| 5 | Idea 4: Scaling Down Code Agents | ★★★☆☆ | ★★★★☆ | 7/10 | Empirical |
| 6 | Idea 3: Step-Level Reward Model | ★★★★★ | ★★★☆☆ | 8/10 | Method（时间风险） |
| 7 | Idea 8: Agentless SFT → Agent RL | ★★★★☆ | ★★★☆☆ | 7/10 | Method（时间风险） |
| 8 | Idea 6: Difficulty-Aware Self-Play | ★★★☆☆ | ★★★☆☆ | 6/10 | Method（计算量大） |

---

## 四、最推荐的 2 个 Idea 的详细实验设计

### 推荐 A: The Anatomy of Agent Failures (Idea 2)

> **一句话**：第一个大规模、跨模型、跨框架的 code agent self-correction empirical study

#### 实验设计

**数据**：SWE-bench Verified 500 instances 的子集（选 200-300 个覆盖不同难度）

**实验矩阵**：

| 维度 | 选项 |
|------|------|
| 模型 | GPT-4o, Claude Sonnet 4, Gemini 3 Flash, Qwen3-32B (开源) |
| 框架 | mini-SWE-agent (agentic), Agentless (workflow), OpenHands |
| Self-correction | 无 / 1轮 / 3轮 / 5轮 |

**分析维度**：

1. **Error Taxonomy** (标注 200+ 失败轨迹)
   - Localization errors (找错文件/函数)
   - Edit errors (修改逻辑错误)
   - Test understanding errors (误读测试需求)
   - Environment errors (命令失败、超时)
   - Planning errors (策略选择错误)

2. **Self-Correction Analysis**
   - 每种错误类型的 self-correction 成功率
   - Self-correction 轮数 vs 效果曲线
   - 模型大小对 self-correction 的影响
   - "Correction of corrections"（越改越错）的频率

3. **Cross-Model/Framework Comparison**
   - 不同模型在哪类错误上更容易 self-correct
   - Agentic vs Agentless 的 failure mode 差异
   - 开源 vs 闭源模型的 recovery gap

**Timeline**：

| Day | 任务 | 预估成本 |
|-----|------|----------|
| 1 | 搭建实验框架，运行 mini-SWE-agent + GPT-4o/Claude 在 200 instances 上 | $200 API |
| 2 | 运行更多模型/框架组合；开始标注失败轨迹 | $300 API |
| 3 | 完成标注；self-correction 实验（重跑失败 cases 加不同轮反馈） | $200 API |
| 4 | 统计分析，画图，写 findings | - |
| 5 | 写论文（intro, related work, method, analysis, conclusion） | - |

**预期 Contribution**：
- 第一个 code agent self-correction 的大规模 failure taxonomy
- 关于 "when self-correction works" 的 empirical laws
- 对未来 agent training 的 actionable insights

---

### 推荐 B: Error Cascades + Checkpoint-Backtrack (Idea 7)

> **一句话**：分析 code agent 的 error cascade 传播机制，并提出 checkpoint-and-backtrack 策略打破级联失败

#### 实验设计

**Phase 1: Error Cascade Analysis**

在 SWE-bench Verified 上运行 agent（使用 mini-SWE-agent + Claude/GPT），收集 **所有失败轨迹**（~200-300 个）。

对每条轨迹标注：
- **First error step**：第一个错误发生在哪一步
- **Error type**：定位/编辑/测试/环境/策略
- **Cascade length**：从第一个错误到最终失败的步数
- **Wasted steps**：第一个错误后的所有步骤中有多少是 "注定失败" 的
- **Recovery attempts**：agent 是否尝试了 recovery？是否成功？

**预期发现**：
- 大量步骤（预计 40-60%）是在第一个错误后的"浪费"
- 定位错误的 cascade 最长（一旦找错文件，后续全废）
- Agent 的 recovery 尝试成功率可能很低（<20%）

**Phase 2: Checkpoint-and-Backtrack (C&B) Method**

设计简单的 C&B 策略：

```
Algorithm: Checkpoint-and-Backtrack
1. 在关键决策点设置 checkpoint:
   - 文件定位完成后
   - 代码编辑前
   - 测试运行后
2. Error detection signals:
   - 测试失败 → 可能的编辑错误
   - 搜索结果为空 → 可能的定位错误
   - 多次相同操作 → 可能陷入循环
3. Backtrack:
   - 回退到最近的 checkpoint state
   - 给 agent "你之前的方案X失败了,原因是Y" 的反馈
   - 限制最大 backtrack 次数（2-3次）
```

**实验对比**：

| 方法 | 描述 |
|------|------|
| Baseline | 标准 agent（无 backtrack） |
| Naive retry | 失败后从头重来 |
| C&B-oracle | 用 ground truth 检测错误并 backtrack |
| C&B-heuristic | 用 heuristic 信号检测错误并 backtrack |
| C&B-LLM | 用 LLM judge 检测错误并 backtrack |

**Metrics**：
- Resolve rate (主指标)
- Average steps to success
- Wasted step ratio
- Token cost

**Timeline**：

| Day | 任务 |
|-----|------|
| 1 | 运行 baseline agent 收集轨迹；开始 cascade analysis |
| 2 | 完成 cascade 标注分析；实现 C&B 策略 |
| 3 | 运行 C&B 实验（所有变体） |
| 4 | 分析结果，ablation study |
| 5 | 写论文 |

**预期 Contribution**：
- 第一个 code agent error cascade 的定量分析
- 简单有效的 C&B 策略，可在现有框架上即插即用
- 对 "agent 应该在什么时候承认错误并回退" 的 empirical insights

---

## 五、关键参考文献

### Code Agent 框架
1. SWE-agent (Yang et al., 2024) - https://arxiv.org/abs/2405.15793
2. OpenHands CodeAct 2.1 (Wang et al., 2025) - https://arxiv.org/abs/2407.16741
3. Agentless (Xia et al., 2024) - ICML 2025
4. Kimi-Dev (2025) - https://arxiv.org/abs/2509.23045
5. mini-SWE-agent (2025) - https://swesmith.com

### RL for Agents
6. GiGPO (NeurIPS'25) - https://arxiv.org/abs/2505.10978
7. DeepSWE (Together AI, 2025) - https://together.ai/blog/deepswe
8. Self-play SWE-RL (Meta, 2025) - https://arxiv.org/abs/2512.18552
9. RAGEN/StarPO (2025) - https://arxiv.org/abs/2504.20073
10. Search-R1 (2025) - https://arxiv.org/abs/2503.09516
11. ReTool (2025) - https://arxiv.org/abs/2504.11536
12. Agent-Omit (ICML'26) - https://arxiv.org/abs/2602.04284

### Agent Training Data
13. SWE-smith (2025) - https://arxiv.org/abs/2504.21798
14. mini-coder (2026) - https://ricardodominguez.github.io/blogs/minicoder.html
15. R2E-Gym (2025) - Berkeley

### Self-Debugging & Error Recovery
16. Revisit Self-Debugging (ACL'25) - https://aclanthology.org/2025.acl-long.881/
17. AgentDebug (2025) - https://arxiv.org/abs/2509.25370
18. Agent-R (2025) - Self-reflective agent training
19. MASFT (2025) - Multi-Agent System Failure Taxonomy
20. SAUP - Uncertainty Propagation in LLM Agents (ACL'25)

### Agent Evaluation
21. SWE-bench Verified (2024) - https://swebench.com
22. SWE-bench Live (2025) - Fresh evaluation
23. AgentPRM (2025) - https://arxiv.org/abs/2511.08325
24. Agent-Reliability-Bench (2026) - GitHub

---

## 六、EMNLP 2026 已接受相关论文（避免重叠）

从 EMNLP 2026 已接受论文中，以下与我们方向最相关：
- **WebAgent-R1**: Multi-turn RL for web agents（我们做 code agents，不重叠）
- **NL-Debugging**: Natural language as intermediate for code debugging（方向不同）
- **LMR-BENCH**: Evaluating LLM agent for reproducing research（不同任务）
- **COLA**: Multi-agent GUI automation（不同领域）

**结论**：没有直接重叠的工作，code agent self-correction/error recovery 方向在 EMNLP 2026 中是空白。

---

## 七、最终推荐

**首选方案：Idea 2 + Idea 7 的结合版**

> **Paper Title**: "When Code Agents Fail: A Systematic Study of Error Cascades and Recovery Strategies"

将两个 idea 合并为一篇完整的 empirical + method 论文：
1. **Part 1** (Analysis): 大规模分析 code agent 的失败模式、error cascade、self-correction 成功/失败 pattern
2. **Part 2** (Method): 基于 Part 1 的发现，提出 Checkpoint-and-Backtrack 策略
3. **Part 3** (Insights): 对 agent training 和 agent design 的 implications

这种结构完美契合 EMNLP 风格（empirical + analysis 重）。5天时间对于 API-based 实验完全可行，且不需要大量 GPU 训练。

**备选方案**：如果 team lead 更倾向 training 导向，推荐 **Idea 1 (RL for Error Recovery)** 或 **Idea 5 (Dense Reward Bootstrapping)**。
