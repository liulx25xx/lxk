# GPU Training Ideas for EMNLP 2026

> 24×H200 GPU | 5天 | 与 Paper 1 (Failure Taxonomy + Adaptive Scaffolding) 互补
> 更新时间: 2026-05-16

---

## 一、三个 GPU Training Ideas 概述

### Idea A: Failure Classifier Fine-tuning (服务 Paper 1 Part 3)

**做什么**: 用 GPT-4o 标注的失败类型数据，fine-tune Qwen3-4B 做 failure classifier

- **输入**: agent trajectory (truncated context + error message + last 3 actions)
- **输出**: failure type (5分类: localization / edit-application / logic / test-misunderstanding / planning)
- **训练规模**: ~2000 标注样本，1-2 张 H200，几小时搞定
- **价值**: Paper 1 Part 3 直接用——对比 LLM-as-classifier vs fine-tuned classifier
- **风险**: 规模太小，不能独立成 paper，只是 Paper 1 的 ablation

**结论**: ★★★★☆ 可行性极高，但只是 Paper 1 的补充实验，不能独立

---

### Idea B: Error-Recovery-Focused RL Training (独立 Paper 2 核心候选)

**做什么**: 训练 code agent 专门学会从错误中 recover，而不是端到端解决问题

**核心差异化**:
- DeepSWE/SWE-RL: 端到端 RL，reward = 测试是否通过 → agent 学习完整解题
- **我们**: 只在 agent 犯错后的 recovery 阶段做 RL → agent 学习"如何从特定错误中恢复"
- 类比: 不是教 agent 开车（端到端），而是教 agent "打滑后如何矫正方向盘"

**与 Paper 1 的互补**:
- Paper 1 分析了什么错误类型的 recovery 率高/低（empirical finding）
- Paper 2 训练 agent 提升 recovery 率（method contribution）
- 形成 "诊断 → 治疗" 的完整故事

**风险**: 需要构建 error-injection + recovery 训练环境，技术栈复杂

**结论**: ★★★★★ 最有潜力的独立 Paper 2，novelty 最高

---

### Idea C: Strategy Distillation (轻量 Paper 1 补充)

**做什么**: 把 Paper 1 发现的 "optimal strategy per failure type" 蒸馏进 Qwen3-4B
- 训练 "strategy recommendation model"：给定 failure context → 推荐最优 behavioral strategy
- 本质是 Paper 1 Part 3 的 trained version（对比 GPT-4o few-shot vs fine-tuned model）

**训练方式**: SFT，用 Paper 1 实验产生的 (failure_context, optimal_strategy) pairs
**规模**: 1-2 张 H200, 几小时

**结论**: ★★★☆☆ 与 Idea A 类似，只是 Paper 1 补充

---

## 二、Novelty 评估：Error-Recovery-Focused RL (Idea B)

### 2.1 竞争格局分析

| 工作 | 方法 | 与我们的区别 |
|------|------|-------------|
| **DeepSWE** (Together AI, 2025.07) | Qwen3-32B + GRPO++ on R2E-Gym, 纯RL无SFT, 59% SWE-bench TTS | 端到端解题RL，不区分错误/恢复阶段 |
| **Self-play SWE-RL** (Meta, 2025.12) | 自己生成bug + 自己修复，self-play RL | Bug injection→repair，但不是从"已犯的错误"恢复 |
| **Agent-R** (ByteDance, 2025.01) | MCTS构造recovery训练数据，DPO训练self-reflection | 最接近的竞争者！但是SFT/DPO方法，非RL训练 |
| **SCoRe** (DeepMind, 2024.09) | 多轮RL训练self-correction (math+code) | 代码生成level，非agent level多步交互 |
| **Agentic RL for Code Repair** (2025.10) | Qwen3-32B SFT+RL for code repair | 直接修code，不是agent recovery from mid-trajectory errors |
| **Agent-R1** (2025.11) | Step-level MDP + 端到端RL | 通用agent RL框架，没有error-recovery focus |
| **GiGPO/verl-agent** (NeurIPS'25) | 两层分组优势估计 | RL算法改进，不针对recovery |

### 2.2 Novelty Gap 分析

**最接近的工作是 Agent-R (ByteDance)**:
- Agent-R 用 MCTS 找到错误点 → 构造 (wrong_trajectory, correct_recovery) pairs → DPO/SFT 训练
- **他们的方法**: 离线数据构造 + SFT/DPO (supervised)
- **我们的方法**: 在线 RL + step-level reward during recovery phase
- **关键区别**: Agent-R 需要 MCTS 预先搜索正确路径；我们让 agent 在 recovery 环境中在线探索

**我们的独特角度 = "Error-Recovery RL" (非 SFT/DPO)**:
1. **环境设计**: 把 agent 放在"已经犯了错误"的状态，RL 目标是 recovery
2. **Error-conditioned RL**: 不同错误类型有不同的 recovery reward structure
3. **与 Paper 1 数据闭环**: Paper 1 提供 failure taxonomy → 作为 recovery RL 的 conditioning signal

### 2.3 Novelty 结论

| 维度 | 评分 | 说明 |
|------|------|------|
| 问题新颖性 | ★★★★☆ | "Error recovery RL for code agents" 尚无人做；Agent-R 最接近但方法不同 |
| 方法新颖性 | ★★★★☆ | Error-conditioned RL + recovery-phase-only training 是新组合 |
| 实验价值 | ★★★★★ | 直接回答 "RL能否提升agent error recovery"，实用性强 |
| 竞争风险 | ★★★☆☆ | Agent-R 的 RL 版本可能有人在做，需要速度 |

**总体 Novelty: 足够支撑一篇独立论文** (EMNLP/ACL main 水平)

---

## 三、详细实验方案：Error-Recovery-Focused RL

### 3.1 核心做法

```
传统 Code Agent RL (DeepSWE):
  Agent → [step1, step2, ..., stepN] → test pass/fail → reward

我们的 Error-Recovery RL:
  Agent → [step1, ..., step_k (ERROR)] → 注入到这个错误状态
  → Recovery Agent → [recovery_step1, ..., recovery_stepM] → test pass/fail → reward
  
  关键: 只训练 recovery 阶段的决策，错误状态作为"初始条件"
```

### 3.2 模型选择

| 选项 | 模型 | 原因 | GPU需求 |
|------|------|------|---------|
| **首选** | **Qwen3-8B** | 足够做 agent (mini-coder-4b 已证明小模型可行)，单卡可推理，8卡可RL | 8×H200 |
| 备选1 | Qwen3-4B | 更小更快，但可能 agent 能力不足 | 4×H200 |
| 备选2 | Qwen3-14B | 更强但训练慢，可能5天不够 | 16×H200 |

**推荐 Qwen3-8B**：
- 24×H200 可以 8卡训练 + 8卡 rollout + 8卡环境并行
- 或者 3 组实验并行跑 (recovery RL / baseline RL / SFT baseline)

### 3.3 训练框架选择

| 框架 | 优势 | 劣势 | 适用性 |
|------|------|------|--------|
| **rLLM** (Agentica/DeepSWE) | 直接支持 SWE-bench + R2E-Gym 环境，GRPO++ 现成 | 只为32B优化，需要适配小模型 | ★★★★★ |
| **verl-agent** (NeurIPS'25 GiGPO) | GiGPO 算法好，step-level credit，支持 Qwen 系列 | 不支持 SWE-bench 环境，需自建 | ★★★☆☆ |
| **Agent-R1** | Step-level MDP 天然适合 recovery 设定，process reward | 相对新，生态不如 rLLM | ★★★★☆ |
| TRL (HuggingFace) | 简单，文档好 | 不支持 agent 多轮交互 | ★★☆☆☆ |

**推荐: rLLM 框架**
- 理由: 已经有 R2E-Gym 环境集成、GRPO++ 实现、Docker 化的 SWE 任务环境
- 需要的修改: 添加 "从错误状态启动 rollout" 的功能

### 3.4 数据来源与准备

#### Phase 1: 收集错误轨迹 (Day 1, API-based)
- **来源**: Paper 1 实验产出 — 200+ 失败轨迹 (已标注 failure type + first error step)
- **处理**: 对每条失败轨迹，截取到 first_error_step，得到 "错误状态"
- **数据格式**:
  ```json
  {
    "instance_id": "django__django-12345",
    "trajectory_prefix": [step1, step2, ..., step_k_error],
    "error_type": "localization",
    "error_description": "Agent searched wrong file, found incorrect function",
    "ground_truth_patch": "...",
    "test_script": "pytest tests/..."
  }
  ```

#### Phase 2: 扩充训练数据 (R2E-Gym, Day 1-2)
- **来源**: R2E-Gym 4500 instances (HuggingFace: `R2E-Gym/R2E-Gym-Subset`)
- **方法**: 用 SFT 模型跑 R2E-Gym → 收集失败轨迹 → 标注错误点 (LLM 辅助)
- **目标**: ~2000-3000 个 (error_state, task) pairs for RL training

#### Phase 3: Error Injection (补充数据)
- 在成功轨迹中人工注入错误 (如: 替换正确的文件搜索为错误搜索)
- 类似 Self-play SWE-RL 的 bug injection，但应用在 agent 轨迹层面

### 3.5 训练配置

```yaml
# Recovery RL Training Config (rLLM framework)
model:
  name: Qwen/Qwen3-8B  # 或 Qwen3-8B-Instruct
  max_context_length: 32768
  
training:
  algorithm: GRPO++  # 参考 DeepSWE
  # Key modifications for recovery:
  rollout_start: "from_error_state"  # 不是从头开始
  max_recovery_steps: 15  # 限制 recovery 步数
  
  # GRPO++ hyperparams (from DeepSWE)
  group_size: 8  # 每个 error state 采样 8 条 recovery trajectory
  clip_high: true  # DAPO: 鼓励探索
  no_kl_loss: true
  no_reward_std: true
  length_normalization: true
  leave_one_out: true
  compact_filtering: true  # 过滤超时轨迹
  
  # Learning rate
  lr: 1e-6
  warmup_steps: 50
  
  # Batch
  batch_size: 32  # 32 error states per batch
  gradient_accumulation: 4
  
  # Training length
  max_steps: 2000  # ~6 epochs over 3000 samples
  
environment:
  type: "r2e-gym-recovery"  # Custom: 从 error state 启动
  docker_images: "r2e-gym"
  max_steps_per_episode: 15
  timeout_per_step: 120  # seconds
  
reward:
  type: "binary"  # test pass = 1, fail = 0
  # 可选扩展: partial reward for getting closer to fix
  
hardware:
  gpus: 8  # 8×H200 for training
  rollout_gpus: 8  # 8×H200 for parallel rollout
  env_workers: 8  # 8 parallel Docker environments
```

### 3.6 对比实验设计

| 实验 | 方法 | GPU | 目的 |
|------|------|-----|------|
| **Exp 1**: End-to-end RL (baseline) | 标准 GRPO++ on R2E-Gym (如 DeepSWE) | 8×H200 | 对比端到端 vs recovery-focused |
| **Exp 2**: Recovery RL (ours) | 从 error state 启动 RL | 8×H200 | 主实验 |
| **Exp 3**: SFT baseline | 用 GPT-4o 的成功 recovery 轨迹做 SFT | 4×H200 | 对比 RL vs SFT for recovery |
| **Exp 4**: Agent-R style DPO | (correct_recovery, wrong_recovery) pairs 做 DPO | 4×H200 | 对比我们 vs Agent-R approach |
| **Exp 5**: Error-conditioned | Recovery RL + error type as condition | 8×H200 | ablation: conditioning 有没有用 |

### 3.7 评估指标

| 指标 | 定义 |
|------|------|
| **Recovery Rate** (主指标) | 给定错误状态，agent 能否 recovery 并通过测试 |
| **Recovery Rate by Error Type** | 分错误类型的 recovery 率 |
| **Steps to Recovery** | 成功 recovery 需要多少步 |
| **End-to-end Resolve Rate** | 在 SWE-bench Verified 上的完整 resolve rate |
| **Token Efficiency** | recovery 阶段的 token 消耗 |

### 3.8 预期时间线

| Day | 任务 | 详细 |
|-----|------|------|
| **Day 1** | 环境搭建 + 数据准备 | 1. 安装 rLLM + R2E-Gym Docker 环境<br>2. 下载 Qwen3-8B<br>3. 用 SFT model 跑 R2E-Gym 子集(500 instances)收集失败轨迹<br>4. Paper 1 数据整合 |
| **Day 2** | 数据标注 + Recovery 环境实现 | 1. LLM 辅助标注 error points (GPT-4o-mini)<br>2. 实现 "从 error state 启动 rollout" 功能<br>3. 验证 recovery 环境可用(手动测试 10 个 case) |
| **Day 3** | RL 训练 | 1. 启动 Exp 2 (Recovery RL) — 8×H200<br>2. 同时启动 Exp 3 (SFT baseline) — 4×H200<br>3. 同时启动 Exp 4 (DPO baseline) — 4×H200<br>4. 监控训练曲线 |
| **Day 4** | 评估 + 补充实验 | 1. 评估所有模型的 recovery rate<br>2. 启动 Exp 1 (End-to-end RL baseline) 如时间允许<br>3. Exp 5 (Error-conditioned) ablation<br>4. 分析结果，画图 |
| **Day 5** | 论文写作 | 1. 写 intro + related work<br>2. 写 method + experiments<br>3. 生成 tables + figures |

### 3.9 预期结果

基于现有文献的合理预期：
- **SFT baseline recovery rate**: ~25-35% (从 Agent-R 的数据推断)
- **DPO baseline**: ~30-40%
- **Recovery RL (ours)**: ~40-55% (RL 应该优于 SFT/DPO for recovery)
- **End-to-end RL**: lower recovery rate，因为没有 focus on recovery

**核心发现预期**:
1. Recovery-focused RL > End-to-end RL > DPO > SFT (on recovery metric)
2. Error conditioning 帮助 agent 针对不同错误类型采用不同 recovery 策略
3. Localization errors 最难 recover，edit errors 最容易
4. 小模型 (8B) + recovery RL 可以接近大模型 (32B) 的 recovery 能力

### 3.10 与 Paper 1 整合方式

```
Paper 1: "Beyond Text Matching: Failure Taxonomy, Adaptive Scaffolding, and Recovery Strategies"
         ↓ provides
         - Failure taxonomy (5 types)
         - Error cascade analysis
         - Recovery success/failure patterns
         - Training data (labeled error trajectories)
         
Paper 2: "Learning to Recover: Error-Recovery Reinforcement Learning for Code Agents"
         ↓ provides
         - Trained model that can actually recover
         - Validates Paper 1 findings (which errors are recoverable)
         - Demonstrates RL > prompting for recovery
         
Mutual benefits:
- Paper 1 Section 6 (Discussion) can cite Paper 2 as "training direction"
- Paper 2 uses Paper 1 failure taxonomy as error conditioning
- Together: "diagnose → treat" complete pipeline
```

---

## 四、代码资源汇总

### 4.1 Agent 框架

| 资源 | 链接 | 用途 |
|------|------|------|
| **mini-SWE-agent** | https://github.com/SWE-agent/mini-swe-agent | 轻量 agent baseline，100行代码，74% SWE-bench |
| **OpenHands** | https://github.com/All-Hands-AI/OpenHands | 完整 agent 平台，CodeAct 2.1 |
| **SWE-agent** | https://github.com/SWE-agent/swe-agent | 原始 SWE-agent (已被 mini 替代) |

**mini-SWE-agent 快速搭建**:
```bash
pip install mini-swe-agent
mini  # 启动 CLI，首次运行配置模型
mini swebench <instance_id>  # 在 SWE-bench instance 上运行
```

### 4.2 训练数据

| 资源 | 链接 | 规模 |
|------|------|------|
| **R2E-Gym** | https://huggingface.co/R2E-Gym | 8,700 tasks, Docker环境 |
| **R2E-Gym-Subset** (DeepSWE用) | https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset | 4,578 instances |
| **SWE-smith** | https://huggingface.co/datasets/SWE-bench/SWE-smith | 50k instances, 128 repos |
| **SWE-bench Verified** | https://github.com/swe-bench/swe-bench | 500 human-validated |

**R2E-Gym 下载与使用**:
```bash
# 下载数据集
from datasets import load_dataset
ds = load_dataset("R2E-Gym/R2E-Gym-Subset")

# Docker 环境 (每个 instance 一个镜像)
# 参考: https://r2e-gym.github.io/
docker pull r2egym/instance:django__django-12345
```

### 4.3 RL 训练框架

| 框架 | 链接 | 特点 |
|------|------|------|
| **rLLM** (推荐) | https://github.com/rllm-org/rllm | DeepSWE 使用，原生支持 R2E-Gym + GRPO++ |
| **verl-agent** | https://github.com/jiajieZeng/verl-agent | GiGPO 算法，NeurIPS'25，但不支持 SWE-bench |
| **Agent-R1** | https://github.com/AgentR1/Agent-R1 | Step-level MDP，process reward |
| **veRL** (基础) | https://github.com/volcengine/verl | 底层 RL 框架 |

**rLLM 安装**:
```bash
git clone --recurse-submodules https://github.com/rllm-org/rllm.git
cd rllm
conda create -n rllm python=3.10
conda activate rllm
pip install -e ./verl
pip install -e .
```

### 4.4 DeepSWE 训练配置参考

来自 DeepSWE 博客 (Together AI):
- **模型**: Qwen3-32B (thinking mode)
- **数据**: R2E-Gym 4,500 questions (排除 SWE-bench Verified 重叠 repos)
- **训练**: 64×H100, 6天
- **算法**: GRPO++ (Clip High + No KL + No Reward Std + Length Norm + LOO + Compact Filtering)
- **环境**: 每个问题一个 Docker 容器，4 tools (bash, search, file_editor, finish)
- **奖励**: binary (tests pass = 1, else = 0)
- **关键发现**: compact filtering 防止 reward collapse; 涌现出边界情况思考

**我们需要修改的地方**:
1. 模型从 32B → 8B (减少 GPU 需求)
2. Rollout 起点从 "空白状态" → "错误状态" (recovery focus)
3. 可选: 添加 error-type conditioning 到 prompt

### 4.5 Agent-R (ByteDance) 参考

- **GitHub**: https://github.com/ByteDance-Seed/Agent-R
- **方法**: MCTS 找到正确 recovery 路径 → 构造 (error, recovery) pairs → DPO/SFT
- **与我们的区别**: 他们 offline data construction + SFT/DPO；我们 online RL exploration
- **我们的优势**: 不需要 MCTS (昂贵)，agent 自己在线学 recovery

---

## 五、立即可以开始的第一步

### Step 0: 今天就可以做 (15分钟)

```bash
# 1. 创建 conda 环境
conda create -n recovery-rl python=3.10 -y
conda activate recovery-rl

# 2. 安装 mini-SWE-agent (用于快速验证 agent 行为)
pip install mini-swe-agent

# 3. 克隆 rLLM
git clone --recurse-submodules https://github.com/rllm-org/rllm.git
cd rllm && pip install -e ./verl && pip install -e .

# 4. 下载 Qwen3-8B
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B
```

### Step 1: Day 1 核心任务

1. **验证 rLLM 能跑**: 在 R2E-Gym 子集上用 Qwen3-8B 跑 10 个 instance，确认环境正常
2. **收集失败轨迹**: 跑 200-500 个 R2E-Gym instances，收集失败轨迹
3. **设计 recovery state**: 确定如何从完整轨迹中提取 "error state" 作为 RL 初始状态

### Step 2: 风险缓解

| 风险 | 概率 | 缓解方案 |
|------|------|----------|
| rLLM 环境搭建困难 | 中 | 备选: 直接用 veRL + 自建 SWE 环境 |
| R2E-Gym Docker 镜像太大 | 中 | 用 SWE-smith 子集(更轻量) |
| 8B 模型 agent 能力不足 | 中 | 先 SFT warm-start 再 RL (如 DeepSWE 发现不需要，但小模型可能需要) |
| 5天不够 | 高 | 简化版: 只做 Exp 2 + Exp 3 对比，省略 end-to-end baseline |
| Recovery rate 太低无意义 | 低 | Oracle analysis 先确认 recoverable 比例 |

---

## 六、论文框架 (如果独立成 Paper 2)

**Title**: "Learning to Recover: Reinforcement Learning for Error Recovery in Code Agents"

```
1. Introduction (1p)
   - Code agents fail frequently; recovery is crucial but understudied
   - Existing RL trains end-to-end; we focus on recovery phase
   - Gap: Agent-R uses SFT/DPO; no work uses RL for recovery

2. Related Work (0.7p)
   - Code agent training (DeepSWE, SWE-RL, SWE-smith)
   - Self-correction (SCoRe, Agent-R, Self-Debugging)
   - Agent RL algorithms (GiGPO, GRPO++, StarPO)

3. Problem Formulation (1p)
   - Define "error state" and "recovery episode"
   - Recovery MDP: S_error × A → S' → reward
   - Error-conditioned policy: π(a | s_error, error_type)

4. Method: Recovery RL (1.5p)
   - Error state construction from trajectories
   - GRPO++ adaptation for recovery episodes
   - Error-type conditioning mechanism

5. Experiments (2.5p)
   - Recovery Rate comparison (RL vs SFT vs DPO vs end-to-end)
   - Per-error-type analysis
   - Scaling analysis (4B vs 8B vs 14B)
   - End-to-end integration with Paper 1's Adaptive Scaffolding

6. Analysis & Discussion (1.3p)
   - What makes recovery learnable?
   - Error types vs RL effectiveness
   - Implications for agent design

Appendix: Training details, per-instance results, compute budget
```

---

## 七、决策建议

### 如果选择与 Paper 1 互补 (推荐 Idea A + B 结合):
- **Day 1-2**: 做 Idea A (Failure Classifier fine-tuning) — 直接支持 Paper 1 Part 3
- **Day 3-5**: 启动 Idea B (Recovery RL) 的数据收集和初步训练
- 这样 Paper 1 马上有训练结果，Recovery RL 作为长线 Paper 2

### 如果选择独立 Paper 2 (all-in on Idea B):
- 全部 5 天投入 Error-Recovery RL
- 24×H200 全开
- 最激进但最有潜力

### 我的推荐: **先做 A (1天)，再 all-in B (4天)**
- A 保证 Paper 1 有 GPU 训练结果 (审稿人喜欢看 fine-tuned model)
- B 是真正有 novelty 的方向，值得 4 天全力投入
