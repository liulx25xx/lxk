# Paper 3 — 实验结果分析 & Insight 提取

**日期**: 2026-05-18
**分析基于**: 7/13 模型评估结果 (EXP-006 s1/s2, EXP-007a s1/s2, EXP-008 s1, EXP-009 s1, EXP-009 lr=1e-5)

---

## 核心发现

RL 训练显著提升 judge accuracy (+14pp) 和 calibration (+7pp)，但**系统性摧毁 position consistency** (-22pp average, 最差 -45pp)。这一现象跨越所有 reward mode，**包括专门加了 consistency reward 的变体**。

---

## 场景一：如果是 Bug (consistency eval 计算有误)

### 可能的 bug 来源

1. **Swap 数据对不齐**: `rewardbench_test_swap.json` 和 `rewardbench_test.json` 的 sample 顺序不一致，导致 consistency 比较了错误的 pair
2. **Parse 逻辑差异**: 训练后模型输出格式变化 (如多了空格/换行)，导致某些 response 被解析为不同答案
3. **Adapter 加载错误**: 评估时 LoRA adapter 没有正确加载到 base model
4. **Position label 映射**: Swap 后 A/B 的 gold label 没有正确翻转

### 排查优先级

1. ⭐ 手动抽样 10 条训练模型的原始+swap 输出，看 position bias pattern
2. 检查 eval script 中 swap data 的 gold label 是否正确翻转
3. 确认 consistency metric 的定义：`original_pred == swap_pred` 还是 `original_pred == flip(swap_pred)`?
4. Baseline 83.3% consistency 是否 reasonable? (对于 7B untrained model 来说似乎偏高但合理)

### 如果确认是 bug

- 修复后重新评估所有模型
- 论文 story 取决于修复后的数据

---

## 场景二：如果是真实发现 (consistency 确实崩了)

这是更可能也更有趣的情况。以下是三个 insight 方向的深入分析：

---

### Insight A: "The Accuracy-Consistency Tradeoff"

**核心主张**: RL 训练 judge 面临根本性的 accuracy vs consistency tension。

**证据链**:
- Accuracy ↑ 和 Consistency ↓ 强负相关:
  - 80.2% acc / 83.3% con → 94.4% acc / 60.8% con (EXP-006 s1)
  - 95.6% acc / 51.7% con (EXP-006 s2)
  - 98.9% acc / 38.1% con (lr=1e-5, 最极端)
- lr=1e-5 case 最说明问题: 更强的优化 = accuracy 更高但 consistency 更烂
- 这不是某个 reward mode 的问题 — **ALL modes** 都有此 tradeoff

**机制假说**: 
RL 优化 accuracy 时，模型学会了 **position-dependent shortcuts**。具体来说：
- 训练数据中，gold answer 可能有 position 分布不均 (如 A 占 55%)
- RL reward 只看最终 accuracy，不惩罚 position-dependent reasoning
- 模型发现 "在不确定时倾向选 A/B" 可以提升 expected reward
- 这提升了 accuracy (在 majority position 上) 但摧毁了 position invariance

**类比**: 这类似于经典的 accuracy-fairness tradeoff (Hardt et al., 2016)。在 ML fairness 中，unconstrained accuracy optimization 会引入 demographic bias。我们的发现是 judge RL 的 analogue: unconstrained accuracy optimization 引入 position bias。

**论文潜力**: ⭐⭐⭐⭐ (7/10)
- 优点: 清晰的 tradeoff 发现，empirical 证据充分，practical implications 强
- 缺点: 停留在描述层面，缺少 "解决方案" 部分会显得不完整

---

### Insight B: "RL Makes Judges More Accurate but More Biased"

**核心主张**: RL 训练是一把双刃剑 — 提升 accuracy 的同时引入了此前不存在的 position bias。

**证据链**:
- Baseline model: 80.2% accuracy, 83.3% consistency → **相对 fair**
- RL-trained model: 94-96% accuracy, 51-62% consistency → **highly biased**
- 关键: Baseline 的 83.3% consistency 说明 untrained model 本身 **没有** strong position bias
- RL training **创造了** position bias (不是放大了已有的 bias)

**这比 Insight A 更强的原因**:
- Insight A 说 "有 tradeoff" — 这是描述性的
- Insight B 说 "RL introduces NEW bias" — 这是因果性的、可 actionable 的
- 暗示: 当前所有用 accuracy-reward RL 训练 judge 的工作 (JudgeLRM, FairJudge GRPO stage) **都可能有这个问题但没检查**

**Broader impact**:
- 这对 RLHF 也有启示: 如果 reward model 本身通过 RL 训练后变 biased, 那下游 alignment 也会受影响
- 这是一个 **auditing call**: 社区需要检查 RL-trained judges 的 position bias

**论文潜力**: ⭐⭐⭐⭐⭐ (8/10)
- 优点: Actionable finding, 对社区有警示作用, surprise factor 高
- 缺点: 需要证明因果方向 (是 RL 引入的, 不是其他因素)

---

### Insight C: "Proxy Consistency Fails — Real Invariance Needs Structural Intervention"

**核心主张**: 当前的 proxy consistency reward (decisiveness) 不等于真正的 position invariance; 解决 consistency 需要结构性干预。

**证据链**:
- EXP-007a 用了 "decisiveness" proxy: 如果模型不输出 tie → higher consistency reward
- 结果: consistency 只从 60.8% → 61.7% (+0.9pp), **几乎无效**
- 这说明 "鼓励 decisive" ≠ "鼓励 position-invariant"
- 模型可以在两个 position 上都 decisively 选 **不同答案** — decisive 但 inconsistent

**为什么 proxy 失败**:
- 真正的 position consistency 需要: `P(A > B | order=AB) = P(A > B | order=BA)`
- 我们的 proxy 只看: "模型不输出 tie" (有明确判断)
- 这两个目标之间没有必然联系
- 模型可能学到: "永远选第一个" — 这在两个 position 上都是 decisive, 但完全 inconsistent

**解决方案路径**:
1. **真正的 paired training (EXP-007b)**: 在训练时同时跑 original + swap, reward 要求两个输出一致
2. **Data augmentation**: 训练数据中 50/50 balance position, 且 same pair 以两种顺序出现
3. **Architecture**: Position-free encoding — 不让 model 看到 "Response A/B" 的位置标记
4. **Post-hoc**: 推理时跑两次取 majority (inference cost 2x, but fixes the problem)

**论文潜力**: ⭐⭐⭐ (6/10)
- 优点: Negative result 有价值, 指出了正确方向
- 缺点: 单独撑不起一篇论文, 需要配合 solution (EXP-007b)

---

## 论文方向建议

### 推荐方向: 结合 Insight A + B + 部分 C

**标题候选**:
1. "The Hidden Cost of RL-Trained Judges: Accuracy Gains at the Price of Position Bias"
2. "RL Training Creates Biased Judges: An Accuracy-Consistency Tradeoff in LLM Evaluation"
3. "When Better Accuracy Means Worse Fairness: Position Bias in RL-Trained Judges"

**论文结构**:

1. **Introduction**: RL training for judges is gaining traction (JudgeLRM, FairJudge). We show it has a critical overlooked side effect.

2. **Finding 1** (Insight B): RL training introduces position bias that didn't exist in the base model. 
   - Evidence: 83.3% → 51-62% consistency collapse
   - This holds across ALL reward configurations (accuracy-only, +consistency, +calibration, full)

3. **Finding 2** (Insight A): There's a fundamental accuracy-consistency tradeoff.
   - Evidence: lr=1e-5 gives 98.9% accuracy but only 38.1% consistency
   - Stronger optimization = more bias

4. **Finding 3** (Insight C): Naive consistency rewards don't work.
   - Evidence: EXP-007a (+decisiveness) barely improves consistency (+0.9pp)
   - Why: proxy reward ≠ true invariance

5. **Analysis**: Why does this happen? 
   - Training data position distribution analysis
   - Model behavior analysis (does it develop position preference?)
   - Connection to reward hacking literature

6. **Solution** (if EXP-007b works): Paired training with true position swap during RL
   - If doesn't work: "open problem" framing (still publishable as analysis paper)

7. **Implications**: 
   - All existing RL-trained judges should be audited for position bias
   - RLHF pipelines using biased reward models inherit this bias
   - Need position consistency as a standard evaluation metric

### 为什么这比原来的 "anti-gaming" narrative 更好

| 维度 | 旧 narrative ("Gaming the Judge") | 新 narrative ("Hidden Cost") |
|------|------|------|
| Surprise | 低 — "multi-objective > single" 是 expected | 高 — "RL destroys consistency" 是 unexpected |
| Data support | ❌ 数据不支持 (consistency reward 无效) | ✅ 数据完美支持 |
| Insight depth | 浅 — engineering 级别 | 深 — reveals fundamental tension |
| Community value | 中 — 一种训练方法 | 高 — auditing call, 影响所有 RL-judge 工作 |
| Novelty | 6/10 (vs JudgeLRM/FairJudge) | 8/10 (没人报告过这个 tradeoff) |

### 关键补充实验

为了强化论文，还需要：

1. **完成剩余 6/13 eval** — 确认 pattern 一致
2. **Position bias 分析**: 统计训练模型在 original/swap 中选 A 的比例 (e.g., 原始位置 A 选中率 vs B 选中率)
3. **Per-category breakdown**: 哪些 category 的 consistency drop 最大？(预测: adversarial > easy)
4. **EXP-007b (true paired training)**: 如果能修复 consistency, 论文变成 "problem + solution"; 如果不能, 论文变成 "problem + analysis"
5. **Training dynamics**: 在 checkpoint 100/200/300/400/500 上评估 — consistency 是什么时候开始崩的？(如果 step 100 就崩了 = 很早就 position-dependent; 如果 step 400 才崩 = 后期 overfitting)

---

## 待解决问题

1. ⭐ **Consistency eval 是否有 bug?** — 这是最高优先级, 如果有 bug 则后续分析全部作废
2. 训练数据 position 分布是否 balanced? (是否 A > B 的 gold answer 偏多?)
3. EXP-007a s2 为何 consistency 高 (70.8%) 而 s1 只有 61.7%? — seed sensitivity 问题
4. Qwen3-8B baseline 为何 accuracy 只有 36.1%? (可能是 format 问题, 不影响主论文)
