# Code Domain Extension for Paper 3: Novelty Assessment

## TL;DR: **不值得专门做。增量 novelty ≈ 0.5/10。RewardBench 已覆盖 code，额外 benchmark 是锦上添花非必要。**

---

## 1. RewardBench 中的 Code 覆盖情况

RewardBench 的 Reasoning 类别包含 6 个 HumanEvalPack 子集：

| 子集 | 训练样本 | 评估样本 |
|------|---------|---------|
| hep-cpp | 118 | 16 |
| hep-go | 106 | 25 |
| hep-java | 127 | 22 |
| hep-js | 119 | 24 |
| hep-python | 120 | 19 |
| hep-rust | 107 | 29 |
| **合计** | **697 (33.4%)** | **135 (30.1%)** |

**关键事实：code 占训练数据的 1/3，占评估数据的 30%。** 我们的所有实验（position shortcut、majority vote、checkpoint dynamics）已经隐式包含了 code domain 的结果。

---

## 2. 竞争者扫描：Code Judge Bias

| 论文 | 年份/会议 | 做了什么 | 与我们的重叠 |
|------|----------|---------|-------------|
| **CodeJudgeBench** (Jiang et al.) | 2025 arXiv | Code 评判 benchmark (CodeGen/Repair/TestGen)，有 adversarial 版本 | 只做 benchmark，不涉及 RL training 或 position bias |
| **Comparing Developer and LLM Biases in Code Evaluation** | 2026-03 arXiv | 13 个 judge 模型在 code eval 的 35 种 bias | 发现 length bias（LLM 偏好长代码），但 **无 position bias 研究，无 RL training** |
| **TIR-Judge** | ICLR 2026 | RL 训练 judge（用 Python executor 做 tool-integrated reasoning） | RL 训练 judge，但 **不研究 bias/shortcut**，是 solution 不是 diagnosis |
| **The Silent Judge** | NeurIPS 2025 WS | Recency bias + provenance bias in LLM judges | **不涉及 code domain，不涉及 RL training** |
| **Wu & Tang 2026** (Reward Hacking Rebound) | 2026-04 arXiv | Code domain RL reward hacking（模型改写 evaluator code） | 不同问题：agent 黑掉 evaluator code，非 judge training bias |
| **PRISM** | NeurIPS 2025 | Length/sycophancy shortcut in reward models | **不涉及 position bias，不涉及 code-specific analysis** |

**结论：没有人专门研究 code judge 的 position shortcut + RL amplification。但这不意外——因为 code 只是 RewardBench 的一个子类别，单独拿出来不构成独立贡献。**

---

## 3. 扩展到 Code Domain 的可能方式

### 方案 A：RewardBench 内按 category 拆分分析（零成本）
- 已有的 eval 数据中拆出 code vs. chat vs. safety vs. math-prm 的 accuracy/consistency
- 展示 shortcut amplification 在不同 domain 上的效果差异
- **Novelty 增量：+0.3** — 这只是更细致的 breakdown，不算新实验

### 方案 B：在 CodeJudgeBench 或类似 benchmark 上跑 eval（~2 GPU-hours）
- 用已经训练好的 position-biased judge 在 CodeJudgeBench 上测试
- 需要适配 prompt format（code evaluation 和 general preference 格式不同）
- **Novelty 增量：+0.5** — cross-benchmark generalization，但评审会说"不同 benchmark 不同 format，不是真正的 cross-domain"

### 方案 C：完全独立的 code judge RL training（~24 GPU-hours）
- 在纯 code preference data (如 CodeUltraFeedback) 上训练
- 验证 position shortcut 是否也出现
- **Novelty 增量：+1.0** — 但投入产出比差，且如果 RewardBench 已经有 33% code，这不算"新 domain"

---

## 4. 核心论点

**为什么不值得专门做：**

1. **RewardBench 已有 33% code 数据** — 我们的 shortcut amplification 结论已经包含了 code domain。分 category 报告即可证明 code 也受影响。

2. **Paper 的 generality argument 应该走 confound type，不是 domain type：**
   - 当前 generality 路线：position confound (Instance #1) → length confound (Instance #2) → general principle
   - 这比 "chat domain + code domain" 更有说服力——因为两种不同 confound 类型被同一机制放大，比同一 confound 在两个 domain 出现更 general

3. **Code 和 chat 在 judge training 中用的是同一个 prompt template** — 从 RL 角度看没有本质区别。Position shortcut 不区分 domain。

4. **已有的 length-bias 实验比 code extension 的 novelty 贡献大得多：**
   - Length confound → proves RL amplifies ANY confound (不只是 position)
   - Code domain → proves same confound appears in another subcategory (已知)

**唯一值得做的零成本操作：** 在现有 eval 结果中按 category breakdown，加一行 "code-specific accuracy/consistency" 到 Table 中。如果 code 的 shortcut 效果 ≠ chat，那反而是有趣的发现。

---

## 5. 建议

| 建议 | 优先级 | 投入 | Novelty |
|------|-------|------|---------|
| 按 category breakdown 现有结果 | **做** | 0 GPU-hours | +0.3 |
| 在 CodeJudgeBench 上 eval 已有模型 | 低优先 | ~2h 适配 | +0.5 |
| 训练独立 code judge | **不做** | ~24h | +1.0 但 ROI 太低 |
| **继续 length-bias 实验** | **最高优先** | 已在跑 | **+1.5** |

**结论：Code domain 扩展是锦上添花，不是 must-have。Paper 的 generality 应该靠 confound diversity (position + length)，不是 domain diversity (chat + code)。建议零成本的 category breakdown 做一下，其余不投入。**
