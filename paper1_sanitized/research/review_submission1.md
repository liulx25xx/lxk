# Review: "When Correct Feedback Fails: Behavioral Scaffolding for LLM Code-Edit Repair"

## 论文概要

在 SWE-bench Verified 的 50 个 edit-application failure 上，跨 12 个 LLM 测试 9 种反馈 prompt。核心发现：**诊断正确性不决定修复成功率，行为策略才是关键变量**。GPT-4.1 上，诊断错误但策略正确的 prompt (re-read, 62%) 远超诊断正确但策略错误的 prompt (scan, 6%)。通过三阶段渐进实验(specificity gradient → factorial ablation → restriction sensitivity)，分离出 behavioral strategy 是主效应，restriction 是 task-dependent amplifier。

---

## 优点

### 1. 实验设计精巧 (★★★★★)
三阶段渐进消解(Phase 1→2→3)是这篇论文最大的亮点。每一阶段设计精准地回答上一阶段留下的问题：
- Phase 1 发现 paradox → Phase 2 排除 restriction 是独立原因 → Phase 3 证明 restriction 在固定策略内反而有益
- 这个 "prosecutor logic" 让论文读起来像侦探推理，非常有说服力

### 2. 核心发现确实 surprising (★★★★★)
"诊断正确 6% vs 诊断错误 62%" — 这个 56pp 反转是真正 non-trivial 的发现。不是"显而易见"的结论。

### 3. 12 模型 × 多条件 = 大规模验证 (★★★★)
涵盖 4 个 capability tier, 含 open-weight + frontier + reasoning models。capability threshold 的发现有独立价值。

### 4. 写作质量高 (★★★★★)
叙事结构清晰，每段有明确的 claim-evidence-implication 结构。Figure 1 的对比框架非常直观。

### 5. 理论连接有深度 (★★★★)
连接到 Vygotsky 的 Zone of Proximal Development，把 LLM feedback 重新概念化为 behavioral intervention 而非 information transfer。

---

## 问题与改进建议

### 问题 1: 任务范围太窄 — 这是最大的弱点 (严重)

**现状**: 整篇论文只研究了 **一种** 失败类型：`str_replace` 的 old_str 不匹配(text mismatch)。50 个样本全是 "Act-bottlenecked" failure，即 agent 已经找对了文件但文本匹配失败。

**问题**: 
- `str_replace` text mismatch 是一个非常具体的、偏底层的操作失败。在真实 SWE-bench 轨迹中，这只是众多失败类型中的一种。
- "re-read from scratch works better" 这个结论的泛化性存疑 — 论文自己也承认 Non-Act failures 上结果反转。
- 50 个样本 × 1 种失败类型 → 统计显著但生态效度(ecological validity)有限。
- **Reviewer 必问**: "This is just about text matching in str_replace. How does this generalize to the vast majority of agent failures (wrong file, wrong logic, wrong test interpretation)?"

**改进建议**: 
- 扩展到至少 2-3 种失败类型的系统研究。Non-Act pilot 只有 20 个样本且只在 Appendix H，应该做得更完整。
- 或者：把 behavioral scaffolding 原则应用到 SWE-bench 的完整解题流程（不只是 edit repair 这一步），测试在 localization failure、test understanding failure 等场景下是否同样成立。

### 问题 2: "Behavioral Scaffolding" 原则的可操作性不足 (中等)

**现状**: 论文提出 "shift from 'what went wrong' to 'what should the model do next'"，但具体如何确定 "right strategy" 缺乏通用方法。

**问题**:
- 在 text mismatch 场景下，re-read 是 "right strategy" 因为需要重新定位文本。但对于其他失败类型呢？
- Section 6 的 design guidelines 偏通用(strategy before diagnosis, match strategy to failure type, restriction as amplifier)，缺乏 **自动化** 的路径。
- 论文说 "a failure classifier paired with a strategy selector" 但没有实现或验证。

**改进建议**:
- 实现一个简单的 failure classifier + strategy selector 原型，在 SWE-bench 上验证 end-to-end 效果。
- 或者至少在更多失败类型上测试不同策略，建立 failure-type → optimal-strategy 的映射。

### 问题 3: 50 个样本的代表性 (中等)

**现状**: 从 495 个 O1-agent 轨迹中筛选出 50 个 Act-bottlenecked failures。

**问题**:
- 筛选比例约 10% — 90% 的失败不在研究范围内。
- 8 个 repository 的覆盖面在 SWE-bench Verified 的 500 个 instance 中是否有偏？
- 50 个样本的 per-model 分析（尤其是 below-threshold 模型全是 0%）可能缺乏统计稳定性。

**改进建议**:
- 补充分析：这 50 个任务在 SWE-bench 中的难度分布、repo 分布是否有偏。
- 如果可能，扩大到 100+ 样本。

### 问题 4: Capability Threshold 的定义不够严谨 (轻微)

**现状**: 4 个模型被归为 "below threshold"(几乎全部 0%)，3 个为 "above threshold"。

**问题**:
- Threshold 是事后观察的二分法(有效果 vs 没效果)，没有独立的 capability 指标来预测。
- Claude Opus 4.7 被归为 below-threshold 但它是很强的模型 — 说明 threshold 是 task-specific 的，但论文没给出如何预测一个新模型是否在 threshold 以上的方法。

### 问题 5: Temperature=0 可能放大了策略效应 (轻微)

所有实验用 temperature=0(确定性输出)。这放大了 prompt 对行为的控制力。在 temperature>0 时，模型可能会自然地混合策略，behavioral scaffolding 效应可能减弱。缺乏 temperature sensitivity 分析。

---

## 与 EMNLP Paper 1 的关系

这篇论文与我们 Route B 调研的 "Code Agent Failure Analysis" 方向**高度相关**但**角度不同**：

| 维度 | submission1 | Route B 建议 |
|------|-------------|-------------|
| **聚焦点** | 单一失败类型的反馈设计 | 多种失败类型的系统分类+recovery |
| **方法** | Controlled experiment (prompt variation) | Large-scale empirical analysis (cross-model × cross-framework) |
| **核心问题** | 什么样的反馈有效？ | 什么类型的错误能自修复？error cascade 如何传播？ |
| **样本量** | 50 个样本 × 1 种失败 | 200-300 个样本 × 多种失败 |
| **贡献类型** | Mechanistic insight (behavioral scaffolding) | Failure taxonomy + method (C&B) |

### 关键 insight: 这篇论文的最大弱点正好是 EMNLP paper 的切入点

submission1 只研究了 **str_replace text mismatch** 这一种失败。真正有价值的问题是：**behavioral scaffolding 原则在不同类型的 agent failure 上表现如何？** 这就是 EMNLP paper 可以做的事。

---

## 可能的 EMNLP Paper 方向 (基于 submission1 的 gap)

### 方向 A: 扩展 Behavioral Scaffolding 到多种 Agent 失败
- 在 SWE-bench 上系统分类所有失败类型(localization/edit/test/planning/environment)
- 对每种失败类型测试不同的 behavioral strategy
- 建立 failure-type → optimal-strategy mapping
- **与 submission1 的关系**: 直接扩展，引用 submission1 作为基础，但从 1 种失败扩展到 5+ 种

### 方向 B: Error Cascade + Checkpoint-Backtrack (原 Route B 建议)
- 分析 agent 轨迹中 error cascade 的传播
- 提出 C&B 方法打断 cascade
- **与 submission1 互补**: submission1 研究"给什么反馈"，我们研究"什么时候该回退"

### 方向 C: 自动化 Behavioral Scaffolding
- submission1 建议了 "failure classifier + strategy selector" 但没做
- 实现这个系统，在 SWE-bench 上验证 end-to-end
- **直接填补 submission1 最大的 gap**

---

## 总体评价

| 维度 | 评分 |
|------|------|
| **Novelty** | 8/10 — 核心发现 genuinely surprising |
| **实验设计** | 9/10 — 三阶段渐进消解非常精巧 |
| **写作** | 9/10 — 叙事流畅，结构清晰 |
| **Scope** | 5/10 — 太窄(1种失败, 50样本) |
| **Generalizability** | 5/10 — str_replace mismatch 的结论能推广多远？ |
| **Actionability** | 6/10 — 原则好但缺乏自动化方法 |
| **总分** | **7/10** — 好论文但 scope 限制了影响力 |

如果投 NeurIPS 可能在 borderline 区间(实验精巧但 scope 偏窄)。改进方向集中在**扩展失败类型**和**自动化策略选择**上。
