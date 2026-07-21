# Novelty & Motivation Analysis: Ideas 3, 4, 5

**Generated**: 2026-05-15  
**Target Venue**: EMNLP 2026 (ARR May 2026)

---

## Idea 3: "The Synthetic Data Paradox: When Self-Generated Training Data Hurts"

### Core Proposal
研究合成数据何时、为何、多大程度上损害模型多样性和能力。设计 controlled experiments 找 tipping point，提出 quality filter + diversity regularization。

### Related Work

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| Shumailov et al., "AI models collapse when trained on recursively generated data" | **Nature 2024** (642 citations) | 奠基性工作：证明迭代训练于自身合成数据导致 model collapse，尾部分布信息不可逆丢失。GMM/VAE/LLM 三种模型验证 |
| "Model Collapse in the Presence of Mixed Data" | **ICLR 2025** | 即使混合真实+合成数据（只要有 ≥0.1% 合成数据），model collapse 仍然持续 |
| Yi et al., "Escaping Model Collapse via Synthetic Data Verification" | arXiv 2025 | 理论+实验：外部 verifier 注入信息可避免 collapse，但 verifier 不完美时早期收益会 plateau 甚至逆转 |
| Token-Level Editing (BIGAI) | **ICML 2025** | 提出 token-level editing 替代整段合成，避免 collapse |
| "Preventing Model Collapse when Training LLMs with Synthetic Data" | IEEE 2025 | 综述+方法：合成数据训练中防止 collapse 的策略 |
| Self-consuming generative models | NeurIPS 2024 workshop | 分析自消费循环对生成模型的长期影响 |

### Gap Analysis

**现有工作做到了什么：**
1. **Collapse 已被充分证明**：Nature 2024 论文已是 landmark，理论+实验完备
2. **解决方案已有多条路线**：verifier-guided, token-level editing, data mixing
3. **Tipping point 分析已有**：ICLR 2025 论文精确研究了合成数据比例的临界点
4. **Quality filtering 已被研究**：Yi et al. 2025 专门研究 verifier-guided filtering

**与我们 idea 的本质区别：**
- 我们的 quality filter + diversity regularization 方向已被覆盖（Yi et al. = quality filter; ICML 2025 = diversity preserving editing）
- "Controlled experiments finding tipping point" 与 ICLR 2025 高度重合
- "When/why/how much" 的 characterization 角度与 Nature 2024 + ICLR 2025 组合已基本覆盖

### Novelty Score: **低 (Low)** ⚠️

**判断依据：**
- Nature 2024 (642 citations) 已经是该领域的 landmark paper
- ICLR 2025 已做了 tipping point 的精确研究
- Quality filter (verifier) 方向已有理论保证 (Yi et al.)
- Diversity preservation 已有 ICML 2025 的 token-level editing
- 这是一个 2024-2025 年的热门 topic，竞争者极多

**要做到 novelty，必须：**
- 提出全新的理论框架解释 collapse（不只是 tail erosion）
- 发现反直觉现象（如某些情况下合成数据反而有益的精确条件）
- 但即使如此，Nature + ICLR 的阴影太大

### Risk Assessment

| 风险 | 级别 | 说明 |
|------|------|------|
| Novelty 不足 | **极高** | 核心 story 已被 Nature 2024 + ICLR 2025 讲完 |
| Reviewer 反对 | **极高** | "incremental over Shumailov et al. Nature 2024" |
| 竞争强度 | **高** | 该方向 2024-2025 论文爆发，难以差异化 |
| 实验设计 | 中 | controlled experiments 可做，但 novelty 在实验设计本身 |

### Reviewer 可能的反对意见
1. "Model collapse has been extensively studied (Nature 2024, 642 citations). What fundamentally new insight does this work provide?"
2. "The tipping point analysis overlaps significantly with [ICLR 2025 mixed data paper]."
3. "Quality filtering as a mitigation has been analyzed theoretically by Yi et al. (2025). Diversity regularization resembles token-level editing (ICML 2025)."
4. "This reads as a systematic empirical study rather than a paper with a novel contribution."

### 5天可行性: 可做实验但不建议
- 实验本身可做（controlled synthetic data experiments）
- 但 novelty 空间太窄，投入产出比极低
- **强烈建议放弃此 idea**

---

## Idea 4: "Process Reward Without Process Labels: Self-Supervised Step Verification"

### Core Proposal
用 outcome-driven self-supervision 自动生成 step labels — 对推理链每步做 counterfactual perturbation，观察 outcome 变化反推 step importance/correctness，训练 PRM。

### Related Work

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| **Math-Shepherd** (Wang et al.) | **ACL 2024** | 开创性工作：通过 Monte Carlo rollout 估计每步正确率，自动构建 process supervision |
| **OmegaPRM** (Luo et al., DeepMind) | arXiv 2024 | MCTS + 二分查找高效定位错误步骤，生成 150万+ process labels |
| **Implicit PRM** | arXiv 2024.12 | 无需显式 process labels 即可获得 process rewards，直接从 ORM 推导 |
| **AlphaMath Almost Zero** | arXiv 2024.05 | "Process supervision without process"——从 outcome 反推 process |
| **FreePRM** | arXiv 2025.06 | 无需 ground truth process labels 训练 PRM |
| **Self-PRM** | arXiv 2025.05 | RL 隐式诱导 PRM 能力，无需显式 process supervision |
| **ThinkPRM** (Khalifa et al.) | arXiv 2025 | 生成式长 CoT 验证，仅用 PRM800K 1% labels 即超越判别式 PRM |
| **AURORA (Universal PRM)** | arXiv 2025.02 | Ensemble prompting + reverse verification 自动训练 |
| **EpicPRM** | arXiv 2025.03 | 自动化 PRM 训练数据构建框架 |
| **Qwen2.5-Math-PRM** | 2025.01 | Consensus filtering 自动标注，指出先前方法存在 label noise 问题 |
| **GC²PO** (Wang et al.) | arXiv 2025/2026 | **最接近**：counterfactual perturbation 用于 step-level reward——对 latent representation 做 perturbation，测量 answer distribution 稳定性 |
| **rStar-Math** (Microsoft) | arXiv 2025.01 | Self-evolved deep thinking for PRM |

### Gap Analysis

**现有工作做到了什么：**
1. **自动 PRM labeling 已是成熟方向**：Math-Shepherd (ACL 2024) 开创，至少 10+ 后续工作
2. **Monte Carlo rollout 已是标准方法**：从 outcome 反推 step correctness 的核心思想已被广泛采用
3. **无标签 PRM 训练已有多种方案**：Implicit PRM, FreePRM, Self-PRM, AlphaMath
4. **Counterfactual perturbation for step reward 已被提出**：GC²PO 明确使用 counterfactual perturbation 评估 reasoning step 的 robustness 和 effectiveness
5. **生成式 PRM 已有**：ThinkPRM 用 CoT 验证，1% 标签即达 SOTA

**与我们 idea 的本质区别：**
- 我们的核心 "counterfactual perturbation → observe outcome change → infer step importance" 与以下工作高度重合：
  - **Math-Shepherd**: rollout from each step → observe outcome = 本质就是 "观察从该步出发的 outcome 变化"
  - **OmegaPRM**: MCTS binary search 定位第一个错误步 = 更高效的 counterfactual
  - **GC²PO**: 直接对 step 做 counterfactual perturbation，测量 answer distribution 变化
- "Outcome-driven self-supervision" = 这就是 Math-Shepherd 和后续所有 Monte Carlo 方法的核心思想
- 唯一可能的差异点：**perturbation 的具体方式**（token-level vs. latent-level vs. step substitution），但这是 marginal contribution

### Novelty Score: **低-中 (Low-Medium)** ⚠️

**判断依据：**
- "从 outcome 反推 step correctness" 已有 10+ 论文，是 2024-2025 最热门 PRM 方向之一
- Counterfactual perturbation 的角度被 GC²PO 直接覆盖
- Monte Carlo rollout (= counterfactual completion) 是 Math-Shepherd 的核心
- 如果我们的 perturbation 方式有本质创新（如真正的 token-level counterfactual editing 而非 rollout），可能有 medium novelty
- 但 "观察 outcome 变化反推 step importance" 的 high-level story 已无新意

### Risk Assessment

| 风险 | 级别 | 说明 |
|------|------|------|
| Novelty 不足 | **高** | 核心思想与 Math-Shepherd/OmegaPRM/GC²PO 高度重合 |
| Reviewer 反对 | **高** | "How is this different from Math-Shepherd's Monte Carlo estimation?" |
| 实现难度 | 中 | Counterfactual perturbation 实现不难，但要超越现有 baselines 需精心设计 |
| Baselines 过多 | **高** | 需要与 10+ 方法对比，包括 Math-Shepherd, OmegaPRM, Implicit PRM, ThinkPRM, FreePRM 等 |

### Reviewer 可能的反对意见
1. "The proposed counterfactual perturbation approach is conceptually similar to Math-Shepherd's Monte Carlo estimation—both estimate step correctness by observing downstream outcome changes. Please clarify the fundamental difference."
2. "GC²PO (Wang et al.) already proposes counterfactual perturbation for step-level reward. The novelty over this concurrent work is unclear."
3. "With 10+ existing methods for automatic PRM labeling, the bar for novelty is very high. The paper needs to demonstrate substantial improvements over Math-Shepherd, OmegaPRM, Implicit PRM, and ThinkPRM."
4. "The paper does not adequately compare with the extensive related work on process reward models without human labels."

### 5天可行性: 勉强可做但风险高
- 实现 counterfactual perturbation + PRM training: 3-4 天
- 需要与大量 baselines 对比: 时间紧张
- 即使做完，novelty 说服力不足
- **建议大幅调整 angle：** 如果要做，需要找到与现有所有方法的本质区别（如不同模态、不同 task domain、或理论分析），否则不建议

---

## Idea 5: "Adaptive Test-Time Compute: When to Think More, When to Think Less"

### Core Proposal
训练轻量 difficulty router 判断问题难度，决定分配多少 test-time compute。在固定 budget 下最大化 accuracy。

### Related Work

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| **s1: Simple test-time scaling** (Muennighoff et al.) | **EMNLP 2025** | Budget forcing 控制推理长度，"Wait" token 延长思考 |
| **DAST** (Shen et al.) | **EMNLP 2025 Industry** | Difficulty-Adaptive Slow-Thinking：根据问题难度自适应 CoT 长度，TLB 指标量化难度 |
| **AdaCompute-LLM** (Zhai et al.) | arXiv 2026.04 | **极度相似**：轻量 GBM classifier 预测难度，constrained optimization 分配 budget。Lagrangian relaxation + GBM 模仿 oracle |
| **Adaptive TTC via Training-Free Difficulty Proxies** (Hu et al.) | **ICLR 2026** | Training-free difficulty proxies + bandit-based allocation，无需训练 difficulty model |
| **Adaptive TTC with Evolving ICL** (Zuo et al.) | arXiv 2026.04 | Warm-up 识别简单问题 + adaptive ICL 集中计算资源于困难问题 |
| **RouteLLM** (Ong et al.) | ICLR 2025 | 基于 preference data 训练 router 在强/弱 LLM 间路由 |
| **FrugalGPT** | NeurIPS 2024 | 成本高效 LLM inference 路由 |

### Gap Analysis

**现有工作做到了什么：**
1. **Difficulty-aware compute allocation 已是 2025-2026 热门方向**：至少 5 篇直接相关论文
2. **轻量 difficulty router 已被实现**：
   - AdaCompute-LLM (2026.04): GBM classifier + 16 features → 预测 budget allocation → **这就是我们的 idea**
   - ICLR 2026: Training-free proxies (无需训练 router 也能做)
   - DAST: SimPO 直接训练模型自适应推理长度
3. **Constrained optimization formulation 已有理论保证**：AdaCompute 有 Lagrangian relaxation + regret bounds
4. **固定 budget 下最大化 accuracy 已被明确研究**：
   - AdaCompute: max E[Acc] s.t. E[Cost] ≤ B
   - ICLR 2026: 同样的 formulation
   - Zuo et al. 2026: elimination + adaptive ICL

**与我们 idea 的本质区别：**
- **几乎没有**。我们的 proposal = "训练轻量 difficulty router 判断问题难度，决定分配多少 test-time compute，在固定 budget 下最大化 accuracy"
- AdaCompute-LLM (2026.04) 做的 **完全就是这件事**：GBM router, difficulty estimation, budget-constrained optimization
- ICLR 2026 论文证明甚至 **不需要训练 router**（training-free proxies 就够了）
- DAST 从另一个角度（训练模型本身自适应长度）解决同一问题

### Novelty Score: **极低 (Very Low)** ❌

**判断依据：**
- AdaCompute-LLM (2026.04) 与我们的 idea **完全重合**
- ICLR 2026 论文已被接收，证明 training-free 方法就够了（比我们的 "训练 router" 更 elegant）
- DAST 已在 EMNLP 2025 发表
- s1 的 budget forcing 已在 EMNLP 2025 发表
- 这个方向已经 **过饱和**

### Risk Assessment

| 风险 | 级别 | 说明 |
|------|------|------|
| Novelty 不足 | **致命** | AdaCompute-LLM 已完全实现我们的 idea |
| Reviewer 反对 | **致命** | "This is essentially AdaCompute-LLM (Zhai et al., 2026)" |
| 竞争强度 | **极高** | 2025-2026 年至少 5 篇直接竞争论文 |
| 差异化可能 | **极低** | 即使换 router 架构或 allocation 策略，都是 marginal |

### Reviewer 可能的反对意见
1. "The proposed approach of training a lightweight difficulty router for adaptive compute allocation is essentially identical to AdaCompute-LLM (Zhai et al., 2026), which uses a GBM classifier for the same purpose with theoretical guarantees."
2. "Hu et al. (ICLR 2026) demonstrate that training-free difficulty proxies achieve similar results without needing a trained router, making the proposed trained router approach less compelling."
3. "DAST (EMNLP 2025) already addresses difficulty-adaptive reasoning from the training perspective. What does a separate router add?"
4. "The adaptive test-time compute space is overcrowded with 5+ papers in 2025-2026. The novelty bar is extremely high."

### 5天可行性: 实验可做但无意义
- 训练 lightweight router: 1-2 天
- 实验: 2-3 天
- **但论文会被直接 reject，因为 AdaCompute-LLM 已经做了完全一样的事**
- **强烈建议放弃此 idea**

---

## 总结对比

| 维度 | Idea 3 (Synthetic Data) | Idea 4 (Process Reward) | Idea 5 (Adaptive TTC) |
|------|------------------------|------------------------|----------------------|
| **Novelty** | 低 | 低-中 | **极低** |
| **最近竞争论文** | Nature 2024 + ICLR 2025 + ICML 2025 | Math-Shepherd (ACL'24) + 10+ 后续 | **AdaCompute (2026.04) = 完全相同** |
| **差异化空间** | 窄 | 窄（需要找到全新 perturbation 方式） | **几乎为零** |
| **Reviewer 风险** | 高 | 高 | **致命** |
| **5天可行性** | 可做但不推荐 | 勉强可做 | 可做但会被 reject |
| **推荐** | ❌ 放弃 | ⚠️ 需大幅调整 angle | ❌ 立即放弃 |

### 优先级建议

1. **Idea 5: 立即放弃** — AdaCompute-LLM (2026.04) 已完全覆盖，且 ICLR 2026 已证明 training-free 方法更优
2. **Idea 3: 建议放弃** — Nature 2024 奠基 + ICLR/ICML 2025 后续已充分覆盖
3. **Idea 4: 如果要做，必须大幅调整** — 不能只是 "counterfactual perturbation for step labels"，需要全新的 angle（例如：特定 domain 的 PRM、多模态 PRM、或者理论证明某种 perturbation 的 optimality）

### 如果必须从三个中选一个

Idea 4 **勉强可救**，但需要以下调整之一：
- **换 domain**: 不做 math reasoning（竞争最激烈），改做 code/science/multi-modal 的 PRM
- **理论贡献**: 证明 counterfactual perturbation 的 information-theoretic optimality vs. Monte Carlo rollout
- **全新方法**: 不用 rollout/perturbation，而是用 LLM 本身的 attention/gradient attribution 来做 step verification（但这也有相关工作）
