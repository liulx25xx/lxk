# On-Policy Distillation (OPD) & On-Policy Self-Distillation (OPSD) 文献调研报告

**调研时间**: 2026年5月15日  
**调研范围**: 2024–2026年 OPD/OPSD/Online Distillation for LLMs  
**目标**: EMNLP 2026 ARR (DDL: 2026-05-25)

---

## 1. 领域综述

On-Policy Distillation (OPD) 是 2025–2026 年 LLM 后训练领域增长最快的方向之一。其核心思想是：让学生模型在**自己生成的轨迹**上学习，同时由教师模型提供**密集的 token 级监督信号**，从而同时解决 SFT 的暴露偏差 (exposure bias) 和 RL 的稀疏奖励问题。OPD 将 off-policy 方法的 O(ε²T) 暴露偏差降低到 O(εT)，在数学推理、代码生成、对齐等任务上已被 Qwen3、DeepSeek-V4、Gemma 2、MiMo-V2 等工业系统采用。

2026年初，**On-Policy Self-Distillation (OPSD)** 作为 OPD 的重要变体爆发式涌现。OPSD 的关键突破在于：**不需要外部教师模型**——单一 LLM 通过条件化不同上下文（如将正确答案作为"特权信息"）同时充当教师和学生。三篇几乎同时提交的论文（OPSD、SDPO、SDFT，均在2026年1月底）标志着这一方向的独立发现。截至2026年5月，仅 awesome-on-policy-distillation 仓库已收录超过100篇相关论文，Self-Distillation 子方向增长最快，但理论理解和实践 recipe 仍远未成熟。

---

## 2. 方法分类体系

根据 Survey (Song & Zheng, arXiv:2604.00626) 的统一 f-divergence 框架，OPD 方法沿三个设计轴组织：

| 设计轴 | 维度 | 代表方法 |
|--------|------|---------|
| **目标函数** | 固定散度 | GKD, DistiLLM, MiniLLM |
| | 自适应散度 | ToDi, EOPD, AKL |
| | RL增强 | G-OPD, RLKD, AlignDistil |
| **信号来源** | 白盒教师 (logits) | DSKD, Delta-KD, PACED |
| | 黑盒教师 (API) | Lion, GAD, OVD, ROPD |
| | 自蒸馏 (无外部教师) | OPSD, SDPO, SDFT, CRISP |
| **训练动态** | Token加权 | TIP, Rock Tokens |
| | 课程调度 | PACED, TCOD |
| | 计算优化 | Lightning-OPD, NPD, SKD |

### 按教师类型分类：

- **外部白盒教师**: GKD, Veto, EOPD, ExOPD, REOPOLD, PACED, Uni-OPD, vOPD, SOD, AOPD
- **外部黑盒教师**: GAD, OVD, ROPD
- **自教师（特权上下文）**: OPSD, SDFT, SDPO, OPSDC, GATES, RLSD, SDZero, PBSD, UniSD, OGLS-SD, ATESD
- **多教师/工业部署**: Qwen3, DeepSeek-V4, MiMo-V2-Flash, GLM-5, Nemotron-Cascade 2

---

## 3. 关键论文详细分析

### 3.1 基础性工作

#### GKD: On-Policy Distillation of Language Models (Agarwal et al., ICLR 2024)
- **核心方法**: 提出 Generalized Knowledge Distillation，将 DAgger 思想引入 LLM 蒸馏——学生在自己生成的序列上训练，教师提供 token 级概率分布作为目标。支持 Forward KL / Reverse KL / JSD 等多种散度。
- **贡献**: 首次系统化 on-policy 蒸馏框架，解决了传统 KD 的分布不匹配问题。在 T5 上验证了摘要、翻译、算术推理任务的有效性。
- **局限性**: 仅在较小模型 (T5) 上验证；未考虑自蒸馏设置；散度选择的理论指导不足。
- **arXiv**: 2306.13649

#### BOND: Aligning LLMs with Best-of-N Distillation (Sessa et al., ICLR 2025)
- **核心方法**: 将 Best-of-N 采样策略蒸馏为单次采样策略。通过分布匹配算法让 policy 逼近 Best-of-N 分布，避免推理时 N 倍计算开销。
- **贡献**: 建立了 online distillation 与 RLHF 的桥梁；提出 Jeffreys divergence 作为训练目标。
- **局限性**: 依赖奖励模型质量；Best-of-N 本身受限于采样效率。
- **arXiv**: 2407.14622

---

### 3.2 On-Policy Self-Distillation 三篇开创性工作（2026年1月）

#### OPSD: Self-Distilled Reasoner (Zhao et al., ICML 2026)
- **核心方法**: 单一 LLM 同时充当教师和学生。**教师策略**条件化正确答案（特权信息）+ 问题，**学生策略**仅条件化问题。两者共享参数。训练在学生的 on-policy rollout 上最小化逐 token 的 JSD_β 散度。
- **关键技术**:
  - Per-token pointwise KL clipping（防止风格化 token 主导训练信号）
  - 固定教师为初始策略（隐式正则化）
  - 最佳配置：TM-off 学生 + TM-on 教师
- **结果**: 在数学推理基准上，**仅用 1024 token 生成长度、每题 1 次采样**即匹配或超越 GRPO（16k 长度、每题 8 次采样），token 效率提升约 128 倍。
- **局限性**: 
  - 依赖 ground-truth 答案作为特权信息（无法应用于无标签场景）
  - 主要在数学推理上验证，开放生成/对齐任务未验证
  - 教师固定为初始策略，无法持续改进教师
  - 长推理链场景未充分探索
- **arXiv**: 2601.18734 | **代码**: github.com/siyan-zhao/OPSD

#### SDPO: Self-Distillation Policy Optimization (Hübotter et al., 2026)
- **核心方法**: 将 RLVR 场景中的**丰富文本反馈**（runtime errors、judge评估等）转化为密集学习信号。模型条件化反馈文本后作为自教师，蒸馏回策略。
- **关键洞察**: LLM 具有强大的"事后诸葛亮"能力——虽然第一遍做错，但条件化反馈后能识别自己的错误。
- **结果**: 在科学推理、工具使用、竞赛编程 (LiveCodeBench v6) 上超越 RLVR baseline。测试时加速 3x（同等发现概率下比 best-of-k 少 3 倍尝试）。
- **局限性**: 依赖环境提供丰富文本反馈；不同反馈质量的鲁棒性未充分分析。
- **arXiv**: 2601.20802

#### SDFT: Self-Distillation Fine-Tuning (Shenfeld et al., MIT, 2026)
- **核心方法**: 将 on-policy self-distillation 应用于**持续学习**。利用 demonstration 条件化的模型作为自教师，生成 on-policy 训练信号，在学习新技能的同时保留旧能力。
- **贡献**: 首次将 on-policy distillation 解释为 Inverse RL；证明 SDFT 能让单一模型随时间**累积多个技能**而不退化。
- **局限性**: 每次学习新任务都需要高质量 demonstration；计算开销比 SFT 更高。
- **arXiv**: 2601.19897

---

### 3.3 理论分析与机制理解

#### Rethinking OPD: Phenomenology, Mechanism, and Recipe (Li et al., 2026)
- **核心贡献**: 首次系统分析 OPD 的训练动态，发现两个成功条件：
  1. **思维模式兼容性**: 学生与教师必须共享兼容的思维模式
  2. **真正的新能力**: 教师必须提供学生未见过的真正新能力（而非仅更高分数）
- **关键发现**:
  - 同族 1.5B 和 7B 教师从学生视角来看**分布不可区分**
  - 成功的 OPD 特征为在**小共享 token 集**（占 97-99% 概率质量）上的渐进对齐
  - OPD 的"免费午餐"（密集 token 级奖励）存在隐性代价
- **实用策略**: off-policy 冷启动 + 教师对齐的 prompt 选择
- **arXiv**: 2604.13016 | **代码**: github.com/thunlp/OPD

#### Why Does Self-Distillation (Sometimes) Degrade Reasoning? (Kim et al., 2026)
- **核心发现**: 自蒸馏降级的根因是 **认知不确定性表达抑制 (Epistemic Verbalization Suppression)**。
  - 教师条件化丰富信息 → 抑制不确定性表达 → 快速域内优化但损害 OOD 性能
  - 三个模型 (Qwen3-8B, DeepSeek-7B, Olmo3-7B) 上观察到最高 **40%** 性能下降
- **启示**: 有效的后训练不应仅强化正确答案 trace，还需保留健康的不确定性表达。
- **arXiv**: 2603.24472 | **代码**: github.com/beanie00/self-distillation-analysis

#### G-OPD: Generalized On-Policy Distillation (Yang et al., 腾讯混元, 2026)
- **理论贡献**: 证明 OPD 是**密集 KL 约束 RL 的特例**（奖励与 KL 权重始终相等）。
- **实践贡献**: 提出 ExOPD（reward scaling > 1），在多域专家合并设置下让学生**超越教师**性能边界。
- **arXiv**: 2602.12125 | **代码**: github.com/RUCBM/G-OPD

---

### 3.4 稳定性与效率改进

#### REOPOLD: Relaxed On-Policy Distillation (Ko et al., 2026)
- **核心方法**: 将 OPD 重新解释为策略优化，通过松弛严格模仿约束来稳定训练，解决不稳定性和熵崩溃问题。
- **结果**: 小模型数学推理提升最高 12 个百分点。
- **arXiv**: 2603.11137

#### Lightning-OPD (Wu et al., 2026)
- **核心方法**: 通过 offline caching 实现 **4x 成本降低**，保持接近 online OPD 的质量。
- **arXiv**: 2604.13010

#### Stable-OPD (Luo et al., 2026)
- **核心方法**: 发现 OPD 中的 length inflation 问题，通过 R-KL + reference divergence 稳定训练。
- **arXiv**: 2604.08527

#### NPD: Near-Policy Distillation (Rang et al., 2026)
- **核心方法**: 异步生成 + 选择性 packing，放松严格 on-policy 要求以提升吞吐。
- **arXiv**: 2605.05940

---

### 3.5 自蒸馏方向最新进展

#### CRISP: Compressed Reasoning via Iterative Self-Policy Distillation (Sang et al., 2026)
- **核心方法**: 教模型生成更简洁的推理轨迹——迭代蒸馏自身的简洁行为，无需 ground-truth 答案。
- **贡献**: 在缩短推理长度的同时提升准确率。
- **arXiv**: 2603.05433

#### RLSD: Self-Distilled RLVR (Yang et al., 京东, 2026)
- **核心方法**: 重新定义自蒸馏角色——教师的特权知识仅用于**调节更新方向**，环境奖励（正确/错误）才是学习信号。解决了纯自蒸馏的信息泄漏问题。
- **arXiv**: 2604.03128

#### GATES: Self-Distillation under Privileged Context (Stein et al., 2026)
- **核心方法**: 针对**无 ground truth、无可验证奖励**的场景（如文档问答）。通过教师共识投票 (consensus gating) 过滤不可靠的自蒸馏信号。
- **关键贡献**: 将自蒸馏扩展到无标签场景，是 OPSD 泛化方向的重要突破。
- **arXiv**: 2602.20574

#### UniSD: Towards a Unified Self-Distillation Framework (Jin et al., 2026)
- **核心方法**: 统一多种自蒸馏方法的框架。
- **arXiv**: 2605.06597

#### PBSD: Preference-Based Self-Distillation (Yu et al., 2026)
- **核心方法**: 将偏好学习与自蒸馏结合。
- **arXiv**: 2605.05040

#### OGLS-SD: On-Policy Self-Distillation with Outcome-Guided Logit Steering (2026)
- **核心方法**: 用结果信号引导 logit 层面的自蒸馏方向。
- **arXiv**: 2605.12400

---

### 3.6 相关重要工作

#### LUFFY: Learning to Reason Under Off-Policy Guidance (Yan et al., 2025)
- **核心方法**: 在 RLVR 中动态融合 off-policy 推理轨迹，突破纯 on-policy 的探索瓶颈。
- **贡献**: 提供了 on-policy 与 off-policy 之间的谱系理解。
- **arXiv**: 2504.14945

#### Is On-Policy Data Always the Best Choice for DPO? (ICLR 2026)
- **核心发现**: 挑战"on-policy 数据总是更好"的共识。对齐过程分为两阶段：
  1. **偏好注入** (preference injection)：需要高多样性 off-policy 数据
  2. **偏好微调** (preference refinement)：需要高质量 on-policy 数据
- **启示**: 不同模型和阶段对数据类型的最优选择不同。

#### OPA-DPO (Microsoft, 2025)
- **核心方法**: 在 DPO 训练前对构建的数据进行 on-policy alignment，显著提升多模态大模型的对齐效果。
- **arXiv**: OPA-DPO project page

---

## 4. 未解决问题 / Gap 分析

### Gap 1: 自蒸馏的理论基础薄弱
- **现状**: OPSD/SDPO/SDFT 三篇工作几乎同时独立发现相同原理，但理论分析各有侧重。G-OPD 建立了 OPD↔RL 等价性，但自蒸馏的理论基础（为什么能work、什么时候会fail）仍不完善。
- **具体问题**: 何时自蒸馏优于外部教师蒸馏？模型能力的什么下界能保证自蒸馏有效？
- **潜在方向**: 建立自蒸馏的 PAC-Bayes 或信息论框架。

### Gap 2: 无标签/开放域场景
- **现状**: OPSD 依赖 ground-truth 答案作为特权信息，SDPO 依赖环境提供的丰富文本反馈。在**无标签的开放域任务**（对话、创意写作、通用对齐）中，特权信息来源不明确。
- **GATES 做了初步探索**（用教师共识作为代理），但仍局限于文档问答。
- **潜在方向**: 利用 reward model / AI judge 作为"软特权信息"进行自蒸馏。

### Gap 3: 长推理链与 Agent 场景
- **现状**: Rethinking OPD 论文指出 OPD 的密集 token 级奖励在长 horizon 蒸馏中可能不 scale。当前工作主要在数学推理（单轮、中等长度）上验证。
- **具体问题**: Multi-turn agent 任务缺乏合适的轨迹级 credit assignment；长 CoT 中 token 级蒸馏信号可能退化。
- **SOD (Step-wise OPD)** 和 **Prune-OPD** 做了初步尝试，但远未解决。

### Gap 4: 自蒸馏的不确定性表达退化
- **现状**: Kim et al. 发现自蒸馏会抑制模型的认知不确定性表达，导致最高 40% OOD 性能下降。
- **根因**: 教师条件化丰富信息后过度自信，学生模仿了这种过度自信。
- **潜在方向**: 不确定性感知的自蒸馏；混合 RL + 自蒸馏保留探索能力。

### Gap 5: Distillation Scaling Laws 缺失
- **现状**: 没有统一框架预测蒸馏质量如何随教师大小、学生大小、数据量、rollout 预算缩放。
- **Survey 明确指出这是最重要的 open problem 之一。**

### Gap 6: 自蒸馏与 RL 的统一
- **现状**: G-OPD 证明了 OPD 是 dense KL-constrained RL 的特例，但自蒸馏特有的"特权信息"机制在 RL 框架中没有对应物。
- **RLSD 做了初步桥接**（自蒸馏仅调节更新方向，RL 奖励决定学习信号），但理论统一未完成。

### Gap 7: 跨能力/跨模态迁移
- **现状**: 几乎所有工作都在数学推理上验证。代码生成有一些结果，但**通用 NLP 任务**（信息抽取、摘要、翻译）和**多模态任务**上的自蒸馏效果未知。
- **Uni-OPD 做了初步的 LLM+MLLM 统一**，但仍是白盒教师设置。

### Gap 8: 计算效率与可扩展性
- **现状**: 虽然 OPSD 声称 token 效率高于 GRPO，但 on-policy 生成本身仍是 bottleneck。Lightning-OPD 通过 offline caching 提速 4x，但牺牲了严格 on-policy 性质。
- **潜在方向**: 混合 on/off-policy 采样策略；投机蒸馏 (SKD)。

---

## 5. 可能的研究角度建议

### 角度 A: 无标签域的自蒸馏（最有 novelty 空间）
- **核心想法**: 将 OPSD 的特权信息机制从"ground-truth 答案"泛化为"任意可获取的辅助信号"——如 reward model scores、retrieval context、self-consistency voting、多轮自修正。
- **与现有工作差异**: OPSD/SDPO 需要 ground-truth 或环境反馈；GATES 仅做了文档问答。通用的"信号无关"自蒸馏框架尚无人做。
- **可行性**: 需要 reward model 或 AI judge 作为特权信号代理。我们有 $1k API 可以调用 GPT-4 级别 judge。
- **风险**: 代理信号噪音可能导致 Kim et al. 发现的不确定性退化问题。

### 角度 B: 自蒸馏 + RL 混合训练
- **核心想法**: 用自蒸馏提供密集 token 级信号（快速学习模式），用 RL 的稀疏但准确的奖励保留探索能力和不确定性表达。
- **与现有工作差异**: RLSD 用 RL 奖励过滤自蒸馏信号，但没有联合优化。G-OPD 证明了等价性但没做混合。
- **可行性**: 技术上相对直接——在 GRPO 训练循环中增加自蒸馏 loss 项。24xH200 足够。
- **风险**: 两个信号的权重平衡可能需要大量调参。

### 角度 C: 自蒸馏的 Scaling Law
- **核心想法**: 系统研究自蒸馏效果如何随模型大小、数据量、迭代次数缩放。
- **与现有工作差异**: Survey 明确指出这是 top open problem，但尚无人系统研究。
- **可行性**: 需要多尺度模型（1.5B/7B/14B/32B）的实验。24xH200 可以跑 7B 和部分 14B。
- **风险**: 可能是"empirical study"而非"method paper"，对 EMNLP 来说 contribution 需要明确。

### 角度 D: Agent 场景的步进式自蒸馏
- **核心想法**: 将自蒸馏扩展到 multi-turn agent 任务，设计步进式 (step-wise) credit assignment。
- **与现有工作差异**: SOD 做了初步的 step-wise OPD 但用外部教师。Agent 场景的自蒸馏无人做。
- **可行性**: 需要 agent benchmark 和多轮交互环境。10天时间可能紧张。
- **风险**: 实验设置复杂，DDL 前可能无法充分验证。

### 角度 E: 自蒸馏中的不确定性保留
- **核心想法**: 直接针对 Kim et al. 发现的不确定性退化问题，设计保留认知不确定性的自蒸馏方法。
- **与现有工作差异**: Kim et al. 发现了问题但没有解决方案。
- **可行性**: 相对 focused，可以在现有 OPSD 框架上快速迭代。
- **风险**: 如何度量"不确定性保留"可能是评审质疑点。

### 推荐优先级（考虑 DDL 10 天、24xH200、$1k API）

| 优先级 | 角度 | 原因 |
|--------|------|------|
| **1** | A (无标签域自蒸馏) | Novelty 最大，gap 最明确，可用 API judge 做特权信号 |
| **2** | B (自蒸馏+RL混合) | 技术直接，理论支撑（G-OPD等价性），实验可控 |
| **3** | E (不确定性保留) | Focused，快速迭代，但 contribution 可能偏窄 |
| 4 | C (Scaling Law) | 重要但可能是 empirical study |
| 5 | D (Agent自蒸馏) | 时间风险太高 |

---

## 6. 工业部署现状

| 系统 | 公司 | OPD 使用方式 |
|------|------|-------------|
| **Qwen3** | 阿里巴巴 | On-policy logit distillation |
| **DeepSeek-V4** | DeepSeek | Full-vocab multi-teacher R-KL |
| **Gemma 2** | Google | KD in pre-training |
| **MiMo-V2-Flash** | 小米 | Multi-teacher logit + reward |
| **Nemotron-Cascade 2** | NVIDIA | Cascade RL + domain OPD |
| **GLM-5** | 智谱 | On-policy distillation |

---

## 7. 训练框架

| 框架 | 链接 | OPD 支持 |
|------|------|---------|
| **TRL** | HuggingFace TRL | GKD Trainer |
| **NeMo-RL** | NVIDIA | 原生 OPD 支持 |
| **veRL** | verl.readthedocs.io | Async On-Policy Distill |
| **OpenRLHF** | github.com/OpenRLHF | 可定制 |

---

## 8. 核心论文快速参考表

| 论文 | Venue | 方向 | arXiv |
|------|-------|------|-------|
| GKD (Agarwal et al.) | ICLR 2024 | OPD 基础 | 2306.13649 |
| BOND (Sessa et al.) | ICLR 2025 | Online Distillation + RLHF | 2407.14622 |
| OPSD (Zhao et al.) | ICML 2026 | 自蒸馏（特权答案） | 2601.18734 |
| SDPO (Hübotter et al.) | 2026 | 自蒸馏（文本反馈） | 2601.20802 |
| SDFT (Shenfeld et al.) | 2026 | 自蒸馏（持续学习） | 2601.19897 |
| G-OPD (Yang et al.) | 2026 | OPD=RL理论 | 2602.12125 |
| GATES (Stein et al.) | 2026 | 无标签自蒸馏 | 2602.20574 |
| CRISP (Sang et al.) | 2026 | 推理压缩 | 2603.05433 |
| REOPOLD (Ko et al.) | 2026 | 稳定性 | 2603.11137 |
| Rethinking OPD (Li et al.) | 2026 | 机制分析 | 2604.13016 |
| Why SD Degrades (Kim et al.) | 2026 | 失败分析 | 2603.24472 |
| RLSD (Yang et al.) | 2026 | 自蒸馏+RL桥接 | 2604.03128 |
| Lightning-OPD (Wu et al.) | 2026 | 效率 | 2604.13010 |
| OPD Survey (Song & Zheng) | 2026 | 综述 | 2604.00626 |
| UniSD (Jin et al.) | 2026 | 统一框架 | 2605.06597 |
| PBSD (Yu et al.) | 2026 | 偏好自蒸馏 | 2605.05040 |
| OGLS-SD | 2026 | 结果引导自蒸馏 | 2605.12400 |
| Is On-Policy Always Best (ICLR 2026) | ICLR 2026 | 数据选择 | — |

---

## 9. 关键资源

- **Awesome 列表**: github.com/chrisliu298/awesome-on-policy-distillation (100+ 论文)
- **综述论文**: arXiv:2604.00626 (Song & Zheng, 2026)
- **中文深度解读**: 
  - 知乎 "同一个模型，两种身份" 系列
  - 知乎 "长文总结：近半年 OPD 的三大主流方向"
  - 知乎 "LLM后训练知识：On-Policy Distillation"
