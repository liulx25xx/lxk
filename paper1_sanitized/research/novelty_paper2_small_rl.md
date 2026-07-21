# Novelty Check: Small Model RL for Code Agents

## Proposed Method
Apply RL (GRPO/DAPO) to train small (4B-8B) code agents on SWE-bench with proper data scale (4500+ R2E-Gym instances), studying whether RL benefits transfer to small models.

## Core Claims
1. **First serious RL training of small code agents** — Novelty: **HIGH** — Closest: SkyRL (300 samples only)
2. **Scaling analysis (4B/8B/14B)** — Novelty: **MEDIUM-HIGH** — Closest: SkyRL (same sizes but severely undertrained)
3. **RL vs SFT comparison at small scale** — Novelty: **HIGH** — Closest: mini-coder (SFT only, explicitly says RL not tried)
4. **Capacity threshold for agent RL** — Novelty: **HIGH** — No one has studied this

## Closest Prior Work

| Paper | Year | Venue | Overlap | Key Difference |
|-------|------|-------|---------|----------------|
| **SkyRL-v0** | 2025.05 | Berkeley blog | 🔴 HIGH | 同样在 7B/8B/14B 上做了 RL for SWE-bench | **只用 300 samples** → 严重欠训 (3.6-5.8pp improvement)。我们用 4500+ → 正经训练 |
| **mini-coder** | 2026.05 | Blog+HF | 🟡 MEDIUM | 4B SFT SWE-bench agent, 26.8% | **SFT only, 明确说 RL 是 future work** |
| **DeepSWE** | 2025.07 | Together AI | 🟡 MEDIUM | RL for SWE-bench agent | **只在 32B 上做**，没有小模型 |
| **Agent Lightning** | 2025.08 | Microsoft | 🟢 LOW | 通用 agent RL 框架 | 框架论文，没发布小模型 SWE-bench results |
| **SWE-agent-LM 7B** | 2025 | Princeton | 🟢 LOW | 7B SFT agent, 15.2% | SFT only |

## Critical Analysis

### SkyRL 是最大的竞争者
SkyRL 确实在 7B/8B/14B 上做了 SWE-bench RL training。但关键差异：

| 维度 | SkyRL | 我们 |
|------|-------|------|
| 训练数据量 | **300 samples** | **4500+ (R2E-Gym)** |
| 结果 | 7B: +3.6pp, 8B: +5.8pp (极弱) | 预期: 显著更高 |
| 分析深度 | 只报了数字 | Scaling curve, failure analysis, RL dynamics |
| 发表状态 | Blog post (非正式) | 无 |
| SFT baseline | 无对比 | 明确 SFT vs RL 对比 |

**核心差异化**: SkyRL 用 300 samples 是一个 "proof of concept"（证明管道能跑），不是一个 "research contribution"（理解 RL 在小模型上的行为）。我们用 15× 数据量做正经训练 + 深度分析。

### mini-coder 作者明确留了 gap
> "the mini-coder models are strong candidates for RL fine-tuning"

原作者点明了方向但没做。我们直接填补这个 gap。

### Reviewer 可能的质疑

1. **"SkyRL already did this"** → 回应: SkyRL 用 300 samples 是极度欠训，结果不可信 (+3.6pp on 7B)。我们用 4500+ samples 是第一个正经训练，可以真正回答"RL对小模型有多少帮助"
2. **"Just applying DeepSWE's recipe to smaller models"** → 回应: 缩放不是 trivial — 可能发现 capacity threshold, 不同 RL dynamics, 需要不同 hyperparams
3. **"Why not just use mini-coder's SFT?"** → 回应: 这正是我们的核心对比——RL vs SFT at small scale，谁更好？

## Overall Novelty Assessment

- **Score: 7.5/10**
- **Recommendation: PROCEED WITH CAUTION**
- **Key differentiator**: 第一个用足够数据量(4500+ vs 300)正经训练小模型(4B-8B) code agent 的工作，包含完整 scaling analysis + RL vs SFT 对比
- **Risk**: SkyRL 的存在削弱了 novelty — reviewer 可能认为我们是 "SkyRL with more data"
- **Mitigation**: 强调 analysis contribution (scaling law, capacity threshold, failure mode breakdown) 而非纯 method

## Suggested Positioning

**不要 frame 为 "method paper" (新训练算法)**。Frame 为:

"**Empirical study: Does RL training benefit small code agents?**"
- SkyRL 的 300-sample 实验不足以回答这个问题
- mini-coder 只做了 SFT，作者明确说 RL 是 open question
- DeepSWE 只在 32B 上验证
- 我们第一次正经回答: 用 4500+ data, 在 4B/8B/14B 上做 RL vs SFT, 分析 scaling behavior

如果发现 **capacity threshold**（如 4B RL 不如 SFT，8B RL 优于 SFT），这就是一个 genuinely interesting finding。

## Final Verdict

| 维度 | 评分 |
|------|------|
| Novelty of question | 8/10 (does RL help small agents?) |
| Novelty of method | 4/10 (just applying known RL to smaller model) |
| Novelty of finding (if threshold exists) | 9/10 |
| Novelty of finding (if RL just works) | 6/10 |
| **Overall** | **7.5/10** — proceed, but success depends on interesting findings |

**与 Recovery RL (5/10) 的对比**: 这个方向更好，因为:
1. 问题本身有明确价值 (practitioner community 想知道能不能用小模型)
2. 无论结果如何都能发 (RL有效/无效都是finding)
3. 竞争者(SkyRL)只做了不充分的pilot，不是full study
