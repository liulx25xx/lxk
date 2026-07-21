# EMNLP Paper 1: 三方向合并方案 + API 费用估算

## 论文标题(暂定)
**"Beyond Text Matching: Failure Taxonomy, Adaptive Scaffolding, and Recovery Strategies for Code Agents"**

## 核心框架

站在 submission1 ("Behavioral Scaffolding") 的肩膀上，从 3 个维度扩展：

| Part | 内容 | 对应方向 |
|------|------|---------|
| **Part 1** | 多失败类型的系统分类 + behavioral scaffolding 泛化验证 | 方向 A |
| **Part 2** | Error cascade 分析 + Checkpoint-and-Backtrack | 方向 B |
| **Part 3** | 自动化 Failure Classifier + Strategy Selector | 方向 C |

---

## Part 1: 多失败类型 Behavioral Scaffolding (方向 A)

### 实验设计
1. 从 SWE-bench Verified 的 500 个 instance 中，用 3-4 个 agent 跑完，收集所有失败轨迹
2. 对失败轨迹做细粒度分类(5+ 类)：
   - **Localization failure**: 找错了文件/函数
   - **Edit-application failure**: str_replace text mismatch (submission1 已做)
   - **Logic error**: 修改逻辑本身错误
   - **Test misunderstanding**: 误读测试需求
   - **Planning failure**: 策略选择错误(如修改了不该改的文件)
3. 对每种失败类型设计 2-3 种 behavioral strategy，测试 behavioral scaffolding 是否跨失败类型成立

### API 费用估算

**Step 1: 收集失败轨迹** (跑 agent on SWE-bench)
- SWE-bench Verified 200 instances × 4 models × 1 run = 800 agent runs
- 每个 agent run: ~20 步 × ~5K tokens/步 = ~100K tokens (input+output)
- 但 agent frameworks 大部分 token 是 input (file content), output 较少

| 模型 | Runs | Avg tokens/run | Input cost/M | Output cost/M | 估算费用 |
|------|------|----------------|-------------|--------------|---------|
| GPT-4.1 | 200 | ~80K in + 20K out | $2 | $8 | $32 + $32 = **$64** |
| GPT-4o-mini | 200 | ~80K in + 20K out | $0.15 | $0.6 | $2.4 + $2.4 = **$5** |
| DeepSeek V4 | 200 | ~80K in + 20K out | $0.5 | $2 | $8 + $8 = **$16** |
| Claude Sonnet 4 | 200 | ~80K in + 20K out | $3 | $15 | $48 + $60 = **$108** |
| **小计** | | | | | **~$193** |

注：可以用更少instance或更便宜模型降低成本。Claude最贵可以减少到100 runs。

**Step 2: 策略测试** (per failure type × strategy × model)
- ~5 种失败类型 × 3 种 strategy × 3 个模型 × ~40 samples/type = 1,800 calls
- 每个 call: ~5K input + ~1K output

| 项目 | Calls | Tokens | 费用 |
|------|-------|--------|------|
| GPT-4.1 (主要模型) | 600 | 3M in + 0.6M out | $6 + $5 = **$11** |
| GPT-4o-mini | 600 | 3M in + 0.6M out | $0.5 + $0.4 = **$1** |
| DeepSeek V4 | 600 | 3M in + 0.6M out | $1.5 + $1.2 = **$3** |
| **小计** | | | | **~$15** |

**Part 1 总费用: ~$208**

---

## Part 2: Error Cascade + Checkpoint-Backtrack (方向 B)

### 实验设计
1. 从 Part 1 收集的失败轨迹中，标注 error cascade:
   - First error step, error type, cascade length, wasted steps, recovery attempts
2. 实现 C&B (Checkpoint-and-Backtrack) 策略
3. 对比: Baseline / Naive retry / C&B-heuristic / C&B-LLM

### API 费用估算

**Step 1: Cascade 分析** — 用 Part 1 已收集的轨迹，标注工作可以用 LLM 辅助

| 项目 | Calls | 费用 |
|------|-------|------|
| GPT-4o-mini 标注 cascade (200 轨迹) | 200 | ~5K tokens each = **$0.2** |

**Step 2: C&B 实验** (re-run with backtrack enabled)
- 3 个 C&B variants × 200 instances × 2 models = 1,200 agent runs
- 每个 run 可能更长(backtrack = 额外步骤): ~30 步 × ~5K tokens = ~150K tokens

| 模型 | Runs | 费用 |
|------|------|------|
| GPT-4.1 | 400 (200×2 variants) | 32M in + 8M out → $64 + $64 = **$128** |
| GPT-4o-mini | 400 | 32M in + 8M out → $5 + $5 = **$10** |
| DeepSeek V4 | 400 | 32M in + 8M out → $16 + $16 = **$32** |
| **小计** | | **~$170** |

**Part 2 总费用: ~$170**

---

## Part 3: 自动化 Strategy Selector (方向 C)

### 实验设计
1. 训练/构建 Failure Classifier: 给定 agent 轨迹 + error message → 分类失败类型
2. Strategy Selector: failure type → optimal behavioral strategy
3. End-to-end 验证: classifier + selector + agent retry loop

### API 费用估算

**Option A: LLM-as-classifier (零训练)**
- 用 GPT-4o-mini 做 few-shot failure classification
- 200 test cases × 3 评估: ~$2

**Option B: Fine-tune 小模型 (用 GPU)**
- 用 Part 1 标注的数据 fine-tune Qwen3-4B 做 classifier → GPU 费用 (不需 API)
- 评估用 API: ~$5

**End-to-end integration test**
- 200 instances × full pipeline (classify → select strategy → retry) × 2 models
- 400 runs × ~80K tokens → ~$50-80

**Part 3 总费用: ~$82**

---

## 费用总汇

| Part | 费用 | 说明 |
|------|------|------|
| Part 1: 多类型 Scaffolding | ~$208 | 最大头(收集轨迹) |
| Part 2: Error Cascade + C&B | ~$170 | C&B re-run 费用 |
| Part 3: Auto Strategy Selector | ~$82 | 轻量 |
| **Buffer (10%)** | ~$46 | 重跑、调试 |
| **总计** | **~$506** |

### 优化策略 (如果需要控制在 $500 以内)

1. **减少 Claude Sonnet 调用**: Claude 最贵($108/200 runs)，可以减到 100 runs → 省 $54
2. **用 GPT-4o-mini 做主力**: $0.15/M input 极便宜，作为 3 个 Part 的主要模型
3. **Part 2 C&B 只用 2 个模型**: GPT-4.1 + GPT-4o-mini → 省 $32 (去掉 DeepSeek)
4. **共享轨迹数据**: Part 1 收集的轨迹直接给 Part 2 用，不需要重跑

**优化后: ~$400-450，完全在 $1k 预算内。**

---

## 时间表 (5天)

| Day | 任务 | API费用 |
|-----|------|---------|
| **Day 1** | 搭建 agent 框架 + 跑 SWE-bench 收集轨迹 (Part 1 Step 1) | ~$193 |
| **Day 2** | 标注失败类型 + 策略测试 (Part 1 Step 2) + cascade 分析 (Part 2) | ~$15 |
| **Day 3** | C&B 实现 + 实验 (Part 2) + 开始 auto classifier (Part 3) | ~$170 |
| **Day 4** | End-to-end 验证 (Part 3) + 分析 + 开始写论文 | ~$82 |
| **Day 5** | 完成论文写作 + 补充实验 | ~$46 |

---

## 论文结构 (8页)

```
1. Introduction (1p)
   - submission1 的 behavioral scaffolding 发现
   - Gap: 只有 1 种失败类型，不能泛化，没有自动化
   - 我们: 多类型验证 + cascade 分析 + 自动策略选择

2. Related Work (0.7p)
   - Agent feedback (Reflexion, Self-Refine, submission1)
   - Agent failure analysis (AgentDebug, MASFT)
   - Agent RL and recovery (GiGPO, DeepSWE)

3. Failure Taxonomy (1.5p) — Part 1
   - 5+ 失败类型的定义和分布
   - 每种类型的 behavioral scaffolding 实验
   - 核心发现: scaffolding 原则跨类型成立但最优策略不同

4. Error Cascade Analysis (1.5p) — Part 2
   - Cascade 传播的定量分析
   - Wasted step ratio
   - First-error-type 的影响

5. Adaptive Scaffolding System (1.5p) — Part 3
   - Failure Classifier + Strategy Selector
   - Checkpoint-and-Backtrack integration
   - End-to-end 评估: resolve rate, token efficiency

6. Discussion + Limitations + Conclusion (1.8p)
   - Generalization of behavioral scaffolding
   - Design principles for agent retry loops
   - When to diagnose vs when to backtrack

Appendix: 详细 prompt templates, per-model results, statistical tests
```

---

## 与 submission1 的关系

- **不是重复**: submission1 做了 1 种失败 × 12 模型的深度机制分析，我们做多种失败 × cascade × 自动化
- **引用关系**: 引用 submission1 作为核心基础，我们是 "extending behavioral scaffolding beyond text matching"
- **如果 submission1 是我们自己的**: 可以在 intro 说 "Prior work [anonymous] established behavioral scaffolding for edit-application repair; we generalize to the full spectrum of agent failures"
- **匿名问题**: 如果 submission1 正在审稿，需要匿名引用

---

## 风险评估

| 风险 | 概率 | 缓解 |
|------|------|------|
| SWE-bench agent 跑不通 | 低 | 用 mini-SWE-agent (开源,简单) |
| 标注耗时 | 中 | LLM辅助标注 + 人工抽检 |
| C&B 效果不显著 | 中 | Oracle C&B 作为 upper bound，至少能展示 cascade 浪费 |
| Auto classifier 不准 | 低 | Few-shot GPT-4o-mini 足够；可退回 rule-based |
| API 费用超 | 低 | 优化方案可控制在 $450 |
| 5天写不完 | 中 | Day 1-3 实验，Day 4-5 写作；用 ARIS paper-writing skill |
