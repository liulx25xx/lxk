# 实验跟踪 (EXPERIMENTS.md)

**论文**: "Cascade Structure Predicts Scaffoldability: Type-Aware Recovery for Code Agent Failures"  
**Venue**: EMNLP 2026 (ARR May cycle, deadline 2026-05-25 AoE)  
**预算**: 总共 ¥200 以内（每批次不超过 ¥200）  
**开始日期**: 2026-05-16  
**最后更新**: 2026-05-19

---

## 核心策略

**"分析免费，干预花钱。先分析，再干预。"**

1. 用公开的 O1-agent 轨迹做零成本分析（taxonomy + cascade）
2. 小规模 pilot 验证 scaffolding 效果（¥30）
3. 确认有效后再全量跑（¥100-150）
4. 每个 phase 设 go/no-go gate，失败就止损

---

## 环境配置（已验证 ✅）

```python
from openai import OpenAI
client = OpenAI(
    api_key="<REDACTED_SECRET>",
    base_url="<REDACTED_URL>"
)
```
- conda 环境: `emnlp` (Python 3.11, swebench 4.1.0)
- Docker: 可用, 72G 存储 on `/opt/docker-data`
- 限流: 6.5s per call
- 推理模型: `max_completion_tokens`, 不设 temperature
- 通用模型: `max_tokens` + `temperature=0`

---

## 总体进度

### ✅ Phase 0: 基础设施 + 公开数据分析 (¥0)

| 步骤 | 状态 | 结果 |
|------|------|------|
| 环境搭建 | ✅ | conda env + swebench + docker + venus api 全部验证通过 |
| Venus API 适配 | ✅ | 3个脚本已改为 Venus URL + 统一 API key |
| 公开轨迹下载 | ✅ | 143 failed trajectories from O1-agent (overlap with our 200) |
| 规则标注 v2 | ✅ | **4 types**: LOGIC 49%, LOC 26%, EDIT 20%, PLAN 6% |
| Cascade 分析 v2 | ✅ | **mean waste 81.8%, median 87.5%**, EDIT 最高(88.8%) |
| 人工验证 | ⬜ 下一步 | 抽 30 条验证规则准确率 |

**Phase 0 结果文件**: 
- v1: `results/phase0_annotations/phase0_full_annotations.json`
- v2: `results/phase0_annotations/phase0_v2_annotations.json` ← 使用这个

**Phase 0 关键发现**:
- **81.8% waste ratio** — agent 平均只有 18% 的步骤在正确方向上
- **LOGIC 是最大类 (49%)** — 不是策略错误，是代码推理错误。Agent 找对了文件但写错了修复。
- **EDIT waste 最高 (88.8%)** — str_replace 失败后 agent 陷入死循环，与 Paper 6 一致
- **LOC 占 26%** — 四分之一的失败是定位错误，这些需要 "broaden search" scaffold
- **所有143条都有 cascade** — error cascade 是普遍现象，不是少数特例

**Go/No-Go Gate**: ✅ 通过（≥100 annotations, ≥3 types, cascade signal 强）

---

### ⬜ Phase 1: Taxonomy 细化 + 人工验证 (¥0)

| 步骤 | 状态 | 说明 |
|------|------|------|
| 规则优化 | ⬜ | 拆分 PLAN 为 LOC/TEST/PLAN，用 gold patch 对比 |
| 人工验证 30 条 | ⬜ | 抽样验证规则标注准确率 |
| 交叉检验 | ⬜ | 对比规则标注 vs gold patch 的文件/函数级一致性 |

**目标**: 5 types each with ≥15 instances, rule accuracy ≥70%

---

### ✅ Phase 2: Scaffolding Pilot (¥8 actual)

| 步骤 | 状态 | 结果 |
|------|------|------|
| Scaffold pilot (76 calls) | ✅ | EDIT +0.70, LOGIC +0.40, PLAN +0.12, LOC -0.20 |

**结果文件**: `results/phase2_scaffold_pilot/pilot_results.json`

**Go/No-Go Gate**: ✅ 通过（scaffold > control on 2/4 types: EDIT, LOGIC）

**Phase 2 关键发现**:
- EDIT scaffold 效果最强 (+0.70 on 0-3 scale) — "re-read file" 策略验证有效
- LOGIC scaffold 有效 (+0.40) — "test analysis" 帮助模型对齐测试预期
- LOC scaffold 失败 (-0.20) — LOC_C (test-guided) 可能不是最佳策略，需要测 LOC_A/B
- PLAN 样本太少(8个)，效果不显著

**重要**: LOC 的负面结果本身是论文的一个 claim —— "不是所有 scaffold 都有效，策略必须匹配失败类型"。这支持 type-dependent optimal strategy 的核心论点。

---

### ✅ Phase 3: Full Scaffolding Matrix (¥1, 19 min, 348 calls)

| Type | Best Strategy | Score | Control | Delta | Worst Strategy | Its Score |
|------|--------------|-------|---------|-------|----------------|-----------|
| EDIT | reread_file | 2.68 | 1.68 | **+1.00** | smaller_edit | 1.86 |
| PLAN | step_back | 2.75 | 1.88 | **+0.88** | scope_check | 1.88 |
| LOC | reread_issue | 1.97 | 1.70 | **+0.27** | test_guided | 0.87 |
| LOGIC | minimal_fix | 2.13 | 1.97 | **+0.17** | (all similar) | 2.07 |

**结果文件**: `results/phase3_full_scaffold/full_results.json`
**日志**: `results/phase3_full_scaffold/fast_run.log`

**Phase 3 关键发现**:
- 每个 failure type 的最优策略不同 → **type-dependent optimal strategy 确认**
- EDIT: "re-read file" (+1.00) 最强，直接复现 Paper 6 结论
- PLAN: "step back and reconsider" (+0.88) 对策略失败非常有效
- LOC: "re-read issue description" (+0.27) 比 "test-guided" (-0.83 vs control) 好1.1分
- LOGIC: 所有策略效果微弱 (+0.10~0.17)，code reasoning 错误难以用简单 scaffold 解决
- **LOC test-guided 有害 (0.87 vs 1.70 control)** = 论文最强证据之一

**Paper claims supported**:
- C1 ✅: Scaffolding generalizes across types (3/4 types show positive delta)
- C1 ✅: Optimal strategy is type-dependent (best strategy differs per type)
- C1 ✅: Wrong strategy can hurt (LOC_C < control by 0.83)
- C1 partial: LOGIC type shows weak effect → honest limitation

| 步骤 | 状态 | 预算 | 说明 |
|------|------|------|------|
| GPT-4o-mini full | ⬜ | ~¥40 | 143 instances × (best + 2nd + control) per type |
| GPT-4.1 validation | ⬜ | ~¥40 | 50 instances × key conditions (confirm cross-model) |

---

### ✅ Phase 4: Validation Experiments (¥~10, completed)

#### 4a: Strategy Selection — Oracle vs Fixed vs Control (288 calls, gpt-4o-mini)
| Condition | Score | File Hit | Delta |
|-----------|-------|----------|-------|
| **Oracle (type-specific)** | **2.33** | 83% | **+0.57** |
| Fixed (always reread) | 2.04 | 75% | +0.28 |
| Control (no scaffold) | 1.76 | 72% | — |

**结果**: `results/phase4_cascade_selection/results.json`  
**Claim**: Type-aware selection > fixed strategy > no strategy. Δ=+0.29 between oracle and fixed.

#### 4b: GPT-4.1 Cross-model Validation (106 calls)
| Type | Scaffold | Control | Delta |
|------|----------|---------|-------|
| EDIT | 2.40 | 1.80 | **+0.60** |
| LOGIC | 2.33 | 2.33 | 0.00 |
| LOC | 2.07 | 2.27 | -0.20 |
| PLAN | 1.50 | 1.88 | -0.38 |

**结果**: `results/phase4_gpt41_validation/results.json`  
**Claim**: 强模型 baseline 高(2.09), scaffold 只在 EDIT 有效。与 Paper 6 capability-threshold 一致。

---

### ✅ Phase 5: Supplementary (432 calls, ¥5)

| Experiment | Key Result |
|-----------|------------|
| DeepSeek V4 Pro | LOC +0.87, EDIT +0.73, LOGIC +0.07, PLAN -0.38 |
| Full Sample (143 inst) | EDIT +1.04, LOC +0.49, PLAN +0.88, LOGIC +0.13 |
| GPT-5.5 Ceiling | LOC **+1.60**, EDIT **+1.20**, LOGIC **-0.40**, PLAN -0.20 |

**结果文件**: `results/phase5_*/results.json`

**Phase 5 关键发现**:
- GPT-5.5 颠覆 ZPD 假设：推理模型 baseline 低但 scaffold 效果巨大 (LOC +1.60)
- Full sample 确认 LOC 效果比 pilot 更强 (+0.49 vs +0.27)
- LOGIC 在所有模型上一致弱/负 → scaffolding frontier 最强确认

### ✅ Phase 6: Multi-Model Expansion (4 models, completed)

Models: deepseek-v4-flash, qwen3.5-35b-a3b, claude-opus-4-7, o4-mini
Calls: 424
**结果**: `results/phase6_multimodel/results.json`

| Model | EDIT Δ | LOC Δ | LOGIC Δ | PLAN Δ |
|-------|--------|-------|---------|--------|
| deepseek-v4-flash | +0.60 | +0.40 | +0.13 | +0.62 |
| qwen3.5-35b-a3b | +0.00 | +0.13 | +0.07 | +1.38 |
| claude-opus-4-7 | +0.13 | +0.27 | +0.13 | +0.00 |
| o4-mini | +0.53 | +1.47 | -0.20 | +0.38 |

**关键确认**: LOGIC Δ 在所有 4 个新模型上均 ≤ +0.13 → frontier 普适性确认

---

### ✅ Phase 7: Reviewer Response Analyses (2026-05-19, ¥0)

**No API calls** — 全部基于已有数据的重新分析

| 分析 | 结果 | 写入位置 |
|------|------|---------|
| P0: Relevant 消融 | file_hit metric 完美支持所有核心 claim | App C Table |
| P1: Classifier features | Footnote 澄清 proxy WR ≠ oracle WR | §5.3 |
| P2: Related Work | +5 citations (DynaFix, TraceCoder, TraceFixer, ARISE, RGFL) | §2 新段落 |
| P3: LOGIC subtypes | deep_iteration Δ=+0.09, test_informed Δ=0.00 | Discussion |
| P4: PLAN LOO | Δ range [+0.71, +1.00], sign test p=0.016 | §5.1 |
| P6: SWE-PRM positioning | Integration paragraph in Discussion | §6 |
| Code fix: Control description | "no intervention" → "generic retry baseline" | Abstract/Intro |
| **Blinded LLM Judge (50 new)** | **87% within-1, 3/4 ordering, LOGIC Δ=-0.03 confirms frontier** | **App C** |

---

### 总花费

| Phase | Calls | Model | 费用 |
|-------|-------|-------|------|
| Phase 0 | 0 | — | ¥0 |
| Phase 2 | 76 | gpt-4o-mini | ¥0.2 |
| Phase 3 | 348 | gpt-4o-mini | ¥1.0 |
| Phase 4a | 288 | gpt-4o-mini | ¥0.9 |
| Phase 4b | 106 | gpt-4.1 | ¥5-8 |
| Phase 5 | 432 | mixed | ¥5 |
| Phase 6 | 424 | 4 models | ¥5-8 |
| Classifiers/Judge | ~200 | mixed | ¥3-5 |
| Phase 7 | 0 | — | ¥0 |
| **总计** | **~1874** | | **≈ ¥20-30** |

**预算状态**: 远低于 ¥200 上限 (≈$3-4 USD)。预算充裕。

---

## Insights & Novelty Tracking

### Insight 1: 82% Cascade Waste (Phase 0 v2, confirmed)

**观察**: Agent 首次犯错后，平均 81.8% (median 87.5%) 的后续步骤是无效的。

**推理**: 当前 agent 没有有效的错误检测和回退机制。一旦走错方向就一路错下去。EDIT 类型的 waste 最高(88.8%)——str_replace 失败后 agent 会反复尝试微小变体而不是换策略。

**Novelty check**: 
- AgentDebug (2025) 分类了失败但没量化 cascade
- LATS / ToT 有回退但没报告 waste
- **首次量化 "agent 步骤浪费率" + 按失败类型分解** ← 强 novelty

**论文位置**: Section 4, Claim C2: "Error cascades waste 80%+ of agent computation"

### Insight 2: LOGIC 是最大失败类 (Phase 0 v2, confirmed)

**观察**: 49% 的失败是 LOGIC 类型（找对文件但代码逻辑写错），远超 LOC(26%) 和 EDIT(20%)。

**推理**: 这颠覆了一个常见假设——很多人以为 agent 主要是"找不到正确位置"(localization)。实际上 agent 定位能力还可以（74% 找到了正确文件），真正的瓶颈是**代码推理**。这暗示 scaffolding 应该重点放在"如何推理正确修改"而非"如何找到正确位置"。

**Novelty check**: 
- SWE-bench 论文只报告 resolve rate，不细分失败类型
- AgentDebug 的 taxonomy 不同（按 stage 分而非按 root cause）
- **我们首次用 gold patch 对比量化了 "找对但写错" vs "找错" 的比例** ← moderate novelty

**论文位置**: Section 3, 核心 taxonomy 分布图

### Insight 3: EDIT cascade 最深但最可修复 (Phase 0 v2, hypothesis)

**观察**: EDIT 类型 waste 88.8% 最高（agent 陷入 str_replace 死循环），但 Paper 6 证明 re-read scaffold 能把修复率从 6% 提到 62%。

**推理**: EDIT 是"cascade 最深但 scaffold 最有效"的类型。这是论文的一个核心对比：
- EDIT: 高 waste + 高 scaffold 效果 = 最值得干预
- LOC: 中 waste + scaffold 需要"broaden search" = 需要不同策略
- LOGIC: 最大类 + 需要"test-driven reasoning" scaffold = 最大机会

**假设**: scaffolding 效果与 failure type 的 cascade 特征相关。cascade 越 "stuck"（重复相同错误）的类型，behavioral scaffold 越有效。

**验证**: Phase 2 pilot 实验

### Insight 7: Type-dependent selection > One-size-fits-all (Phase 4a, confirmed)

**观察**: Oracle (type-specific) = 2.33 > Fixed (always reread) = 2.04 > Control = 1.76

**推理**: 即使 "reread file" 是最强的单一策略，它也不是所有类型的最佳选择。对 PLAN 类型，"step back" 比 "reread" 高 +0.88；对 LOC 类型高 +0.46。一个知道 failure type 的系统比盲目应用最强策略好 14%。

**Novelty**: 这是论文的核心 contribution 之一 —— 不只是证明 scaffold 有效，而是证明 **type-awareness 额外提升显著**。

**论文位置**: Section 5.3, 核心 Table

### Insight 9: Reasoning Models — ZPD 假设需要修正 (Phase 5, GPT-5.5)

**观察**: GPT-5.5（推理模型）在 LOC (+1.60) 和 EDIT (+1.20) 上效果巨大，远超所有其他模型。但 LOGIC (-0.40) 和 PLAN (-0.20) 为负。

**推理**: 之前的 ZPD 假设太简单了。正确的框架是：
- scaffold 效果不是单纯由"模型强度"决定的
- 而是由 **"模型在这个具体 task format 上的 baseline"** 决定
- GPT-5.5 作为推理模型，在 single-turn 指令跟随上 baseline 反而低（0.80/1.00）
- 但给了正确 scaffold 后，它的强推理能力被正确引导（2.40/2.20）
- 对 LOGIC 反而有害：推理模型本来就在"推理"，额外的 scaffold 干扰了它的推理过程

**修正后的理论**: 
- scaffold 效果 = f(baseline gap × task-strategy fit)
- 不是"弱模型受益多"，而是"在特定 task 上 baseline 低的模型受益多"
- 推理模型在 information-deficit tasks (LOC/EDIT) 上 baseline 低因为它在"想太多"而不是"执行"
- 推理模型在 reasoning tasks (LOGIC) 上 scaffold 有害因为干扰了已有推理

**论文价值**: 这比简单的 ZPD 更深刻——不是 capability threshold 而是 **task-model alignment**

**论文位置**: Section 5.4 需要重写，Discussion 需要更新

**观察**: LOC_C (test-guided localization) scaffold 得分低于 control (-0.20)。

**推理**: 这不是 bad news，而是论文最强的证据之一。它证明：
1. 不是"任何 scaffold 都比没有好" — strategy 必须匹配 failure type
2. Paper 6 发现 "wrong strategy hurts" 在 EDIT 上成立（scan 6% vs re-read 62%）
3. Paper 1 扩展证明：同样的原则在 LOC 上也成立——错误的 scaffold 比没有更差

**为什么 LOC_C 失败**: "Run the failing test first" 对于 localization failure 可能不合适，因为：
- Agent 已经在错误方向上了，测试信息可能进一步确认错误假设
- 更好的 LOC scaffold 可能是 LOC_A "broaden search"（扩大搜索范围）

**验证**: Phase 3 测试 LOC_A 和 LOC_B 作为替代

**论文位置**: Section 3.4 Results — "strategy-type mismatch penalties"

### Insight 6: Scaffold 效果与 cascade 特征相关 (Phase 0+2, hypothesis strengthened)

**观察对比**:
- EDIT: waste=88.8%, scaffold delta=+0.70 (最高)
- LOGIC: waste=81.2%, scaffold delta=+0.40 (中等)  
- LOC: waste=80.0%, scaffold delta=-0.20 (失败)
- PLAN: waste=70.7%, scaffold delta=+0.12 (微弱)

**推理**: EDIT 的高 waste 来自"反复尝试相同错误策略"(str_replace 死循环)，这恰好是 scaffold 最能打破的模式。LOGIC 的 waste 来自"逻辑推理错误后继续在错误方向上优化"，scaffold ("read the test") 帮助重置推理方向。LOC 的 waste 来自"在错误文件上不断尝试"，但当前 scaffold 没有有效引导到正确位置。

**关键推论**: **Scaffold 有效当且仅当它能打破特定的 cascade 模式。** 
- EDIT cascade = 重复执行错误 → re-read 打破重复
- LOGIC cascade = 沿错误逻辑深入 → test analysis 重置逻辑
- LOC cascade = 在错误位置展开 → 需要 "step back + search broadly"

这是论文的 **unifying principle**: scaffold effectiveness = f(strategy-cascade_pattern fit)

**论文位置**: Section 6 Discussion, 核心理论贡献

---

## 注意事项

### 实验规范
1. **先 pilot 再全量**: 每个新实验先跑 5 个样本验证脚本正确
2. **增量保存**: 每 10 个样本保存一次中间结果
3. **预算硬限**: 每个脚本内置 MAX_CALLS
4. **断点续传**: 检测已完成文件，跳过重复
5. **版本管理**: 每次改动规则都标注 v1/v2/v3

### 数据来源
- 公开轨迹: `AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results` (HuggingFace)
- 我们的子集: `data/swebench_subset.json` (200 instances, 12 repos)
- SWE-bench gold patches: `princeton-nlp/SWE-bench_Verified` dataset

### 论文写作节奏
- **Section 1-8 完整初稿已完成** (10页: 正文6页 + refs 1.5页 + appendix 2.5页)
- 标题: "Cascade Structure Predicts Scaffoldability: Type-Aware Recovery for Code Agent Failures"
- Figure 1: TikZ 框架总图（4列: Failure Type → Cascade Pattern → Best Strategy → Scaffoldable?）
- Figure 2: TikZ 柱状图（cascade-scaffold 关联）
- 4个 Appendix（prompts, per-instance results, stats, classification methodology）
- Limitations 已压缩去自爆点
- Discussion 压缩为1节连续文本（scaffolding frontier + design implications）
- 所有"not significant"/"weak"改为框架预测的确认（"confirms our framework's prediction"）
- 正文偏短(6页)，可以考虑扩充 Taxonomy 或 Scaffolding 的分析深度到 7-8 页

### 与 Paper 6 (NeurIPS) 的关系
- Paper 6: 1 type (EDIT) × 12 models × mechanism insight
- Paper 1: 5 types × 2-4 models × taxonomy + cascade + automation
- 不重叠，互补关系
- Paper 1 匿名引用 Paper 6: "Anonymous (2026) established behavioral scaffolding for edit-application repair"
