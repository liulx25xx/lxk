# Novelty Check: Self-Consistent Calibrated Judges via RL

**Date:** 2026-05-16  
**Idea:** 用 GRPO 训练 7B judge model, composite reward = accuracy + consistency (position swap一致性) + calibration (Brier score)

---

## Novelty Score: 8/10

---

## 最接近的3个竞争者

### 1. JudgeLRM (arxiv 2504.00050, 2025-03)
- **方法:** GRPO训练judge LLM，使用"judge-wise outcome-driven rewards"
- **Reward:** accuracy-only（判断对错）+ format reward。**无consistency reward，无calibration reward**
- **差距:** 只优化"判对"，不关心position bias和校准度。是最近的竞争者但reward设计远比我们简单。

### 2. JudgeBiasBench (arxiv 2603.08091, 2026-03)
- **方法:** bias-aware training：生成式judge用RL，判别式用对比学习
- **目标:** debiasing（去偏），训练judge忽略bias-correlated cues
- **差距:** 关注去偏而非calibration。RL方法不确定是否用GRPO。没有composite multi-objective reward；更像是data augmentation + bias-aware loss。

### 3. RLCR (arxiv 2507.16806, 2025-07, MIT)
- **方法:** RL with Calibration Rewards，训练reasoning model联合优化accuracy + calibrated confidence
- **差距:** 针对reasoning task（数学推理等）的自身confidence calibration，**不是judge场景**。不涉及pairwise consistency或position bias。

### 补充竞争者:
- **TrustJudge (ICLR 2026):** 概率框架解决Score-Comparison inconsistency，但是**推理时方法（inference-time）**，不训练模型
- **MJ1 (2603.07990, 2026-03):** 多模态judge用RL训练含"flip-consistency reward"，但是**多模态**场景（视觉grounding），非纯文本judge
- **BT-sigma (2602.16610, 2026-02):** 用Bradley-Terry聚合多judge输出提高一致性，但是**aggregation方法**，不训练单个judge
- **Taming the Judge (2510.15514, 2025-10):** 检测并解决judge feedback中的inconsistency（Deconflicted Graph Rewards），但focus是**使用judge信号训练其他模型**，不是训练judge本身

---

## 核心风险

1. **JudgeLRM后续工作风险:** JudgeLRM是2025-03的工作，到现在14个月。其后续版本或跟进论文可能已经加入consistency/calibration reward但尚未公开。
2. **MJ1的flip-consistency reward:** 虽然是多模态场景，reviewer可能认为"RL+flip-consistency"这个组合已经有人做了（只是domain不同）。需要在related work中明确区分。
3. **RLCR的calibration reward:** reviewer可能说"calibration reward已有人提出（RLCR），你只是把它应用到judge场景"。需要强调composite reward的交互效应和judge-specific设计。
4. **实验可信度:** 需要在多个benchmark上证明三个reward的联合效应 > 任何单一reward。

---

## 差异化优势（为什么还有8分）

| 维度 | 现有工作 | 我们 |
|------|---------|------|
| 训练方法 | JudgeLRM: accuracy only | **三维composite reward** |
| Consistency | MJ1: 多模态flip | **文本pairwise position swap** |
| Calibration | RLCR: reasoning task | **Judge评分校准（Brier score）** |
| 场景 | 各自孤立 | **首次统一accuracy+consistency+calibration** |

没有任何现有工作同时在文本judge场景中用RL优化这三个维度。最近的是JudgeLRM（只有accuracy）和MJ1（有flip但是多模态）。

---

## 一句话建议

**可以做。** 核心novelty在于"首个三维composite reward (acc+consist+calib) 的RL-trained text judge"——但必须在实验中做好ablation证明三者联合>任意子集，同时明确区分与JudgeLRM（无consistency/calibration）、MJ1（多模态非文本）、RLCR（非judge场景）的区别。建议标题突出"composite"或"multi-objective"。
