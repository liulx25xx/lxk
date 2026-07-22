# AAAI-27 摘要（已锁定）

> 锁定：2026-07-21 ｜ 摘要截止：2026-07-22 20:00
> 方向：**Measurement + Mitigation**（最强可发版本；后续 8 卡实验由用户补全）
> 说明：摘要按全文规划的最强形态写,承诺的实验(第二数据集、缓解对比、机制)由后续实验交付。

---

## 标题（锁定）

**When Accuracy Is Positional: Diagnosing Position Shortcuts and Deconfounding Trained LLM Judges**

> 选词理由：
> - "When Accuracy Is Positional" —— 最反直觉、最易被记住的表述(准确率居然是位置的)。
> - "Diagnosing Position Shortcuts and Deconfounding Trained LLM Judges" —— 动宾对齐:诊断 shortcut / 去[混淆] judge 训练(deconfound 的宾语是 confound/数据,不是 shortcut);同时覆盖诊断(swap eval + decomposition)与修复(balancing),对得上全文闭环。
> - "**Deconfounding**" 是关键升级词:把论文从"发现数据 bug"抬到"因果 confound 的诊断与去混淆",这正是它区别于 trivial data hygiene 的思想内核,也预埋 §5 mechanism 与缓解对比的合法性。
> - "Trained"(非 Reinforcement-Trained):论文覆盖 SFT/DPO/GRPO,不止 RL,更准确。

---

## 摘要正文（最强版,~245 词,可直接贴 AAAI 系统）

LLM judges are increasingly trained to evaluate pairwise preferences and to serve as reward signals in post-training pipelines, yet they are evaluated almost exclusively by answer accuracy. We show that this metric can systematically overstate judgment quality: the standard conversion of `chosen`/`rejected` preference data into fixed-order judge prompts makes response position a perfect predictor of the gold verdict, so a trained judge can raise apparent accuracy by learning a position shortcut instead of comparing the two responses. We introduce **position-swap evaluation**, which separates apparent from position-invariant accuracy, and a **decomposition** that splits any trained judge's accuracy gain into a *positional-shortcut* component and a *genuine-comparison* component. Across SFT, DPO, and GRPO, three model families, and two preference datasets, unbalanced training inflates original-order accuracy while collapsing swap consistency — in the extreme, a judge reaches 99% accuracy by almost always selecting the first response. The shortcut is invisible to the reward-design fixes we test, survives prompt, label, and learning-rate controls, and appears even in independently released checkpoints (JudgeLRM). Critically, its severity is anti-correlated with the base judge's own comparison ability: the weakest base judge loses over 40 consistency points and more than half of its apparent gain is positional, whereas the strongest is barely affected — so the models most in need of judge training are the most vulnerable. Data-level deconfounding — balancing preferred responses across positions — removes the shortcut at the source and recovers genuine gains; comparing against position-aware algorithmic alternatives, we find it to be the simplest and most reliable mitigation. We argue that position-swap consistency should be reported alongside accuracy for every trained pairwise judge.

---

## 承诺清单(摘要写进去了 = 全文必须交付)

| 承诺 | 交付实验 | 现状 |
|---|---|---|
| swap evaluation 分离 apparent / position-invariant | 5 指标 swap 评测 | ✅ 已有(120 runs) |
| decomposition: positional vs genuine | unbal vs bal 对比 | ✅ 已有(decomposition 表) |
| SFT/DPO/GRPO × 3 模型家族 | cross-model 全表 | ✅ 已有 |
| **two preference datasets** | UltraFeedback-binarized(或 HH-RLHF)GRPO unbal/bal, 3 seeds | ⏳ 待跑(2 卡先起) |
| reward 修复无效 + prompt/label/LR 控制 | reward ablation + controls | ✅ 已有 |
| 公开 checkpoint 中招(JudgeLRM) | JudgeLRM-7B/3B swap 评测 | ✅ 已有 |
| **severity ∝ base 弱**(hero) | base 能力 × 一致性掉幅 图 | ✅ 可从现有数据出图,后续加密 |
| **algorithm-level 对比**(online-random-order / paired-consistency) | matched-compute 对比 | ⏳ 待跑 |
| 机制(logit-bias / slot-bias) | logit-bias probe(从未跑) | ⏳ 待跑 |

> 三项 ⏳ 是后续 8 卡要补的;都不改变 thesis 方向,只是把已成立的故事加宽加硬。若某项来不及,"simplest and most reliable mitigation" 的措辞仍成立(balancing 最简单且可靠地移除 confound)。

---

## 逐句证据核对(防过度声称)

| 摘要 claim | 证据 |
|---|---|
| position 是 gold 的完美预测子 | unbalanced 转换 chosen→A, 100% gold=A |
| dissociation 跨 SFT/DPO/GRPO | SFT 100/0、DPO 94.2/54.3、GRPO 94.7/58.7;Qwen3/Mistral 复现 |
| 99% 准确靠总选第一个 | SFT 100% acc/100% first-pos;Mistral GRPO 98.4% acc |
| reward 修复无效 | decisive/calib proxy 不恢复一致性(Qwen2.5/Qwen3) |
| 存活于 prompt/label/LR | label 表 A/B,1/2,L/R 全中招;LR 阈值 |
| JudgeLRM 中招 | JudgeLRM-7B gap 15.1、-3B gap 20.9 |
| severity ∝ base 弱 | Mistral(base65.7,Con−41.8,shortcut53%)/Qwen2.5(80.2,−22.8)/Qwen3(86.0,−0.4) |
| balancing 恢复 genuine gain | bal GRPO 一致性回基线、保住 +3.5(Qwen2.5)/+15.4(Mistral) genuine |

> severity 统一用**绝对量(一致性掉幅 / shortcut pp)**,不用比例(Qwen2.5 比例 76% > Mistral 53% 非单调)。
