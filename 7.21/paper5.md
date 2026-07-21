# Paper 5 — AAAI-27 投稿快照（2026-07-21）

- **Venue**: AAAI-27
- **Deadline**: 摘要 2026-07-22 20:00 (Shanghai ≈ AoE)；Full paper 2026-07-28
- **状态**: 标题 + 摘要定稿；实验待跑（见下）
- **源稿**: `lxk/paper5_sanitized_20260721/`（原 EMNLP 稿，重构中）

---

## 标题

**Not All RLVR Gains Transfer: A Single-Pass Probe for the Transferable Fraction of Post-Training Improvement**

## 摘要

Reinforcement learning with verifiable rewards (RLVR) has become the default recipe for improving LLM reasoning, and practitioners now apply the same math-calibrated configuration to medicine, science, and beyond. We ask a question that is rarely tested: do the resulting gains reflect genuine capability acquisition, or in-distribution format optimization? In a controlled study holding the base model (Qwen2.5-7B-Instruct), LoRA configuration, training prompts, and evaluation protocol fixed across four domains—mathematics, science, medicine, and commonsense—we compare supervised fine-tuning and GRPO across over 200 training runs and validate on additional model families. Three findings emerge. (1) The standard math-calibrated GRPO configuration yields near-zero test improvement in every domain despite rising training reward: its conservative step size under-updates the policy, and matching the effective update magnitude to the available gradient signal restores multi-percentage-point gains. (2) These gains are not created equal. An out-of-distribution probe shows that gains in knowledge-uncertain domains (medicine) transfer almost fully, whereas gains in already-saturated domains (mathematics, commonsense) barely transfer—evidence of format optimization rather than capability acquisition, which we confirm is not an artifact of answer-position bias through option-permutation audits. (3) The transferable fraction of RLVR gains is predictable before training from a single K-rollout probe measuring the effective advantage-variance of a domain. Together these results reframe post-training recipe selection: the relevant question is not which recipe wins in-distribution, but which produces gains that transfer.

---

## 本轮实验规划（待跑，等算力 / docker 恢复）

主心骨从「One Hyperparameter / 40× / mode-seeking trap」换成 **RLVR 涨点迁移分解 + 训练前 advantage-方差探针**。分析型为主，少吃算力；lr/欠更新降为机制佐证。

1. **Tier 0（不训练，环境一好就做）**
   - provenance audit：核每个旧 run 是否独立训练 seed（目录 seed 与 config 不一致者标 invalid）。
   - 冻结 train/dev/test + 哈希；组合 benchmark 按 subtask 报 macro（修 Commonsense 被 HellaSwag 主导、Law 被单 task 主导）。
   - **MCQ 选项置换 + parser 审计**（3 perm × 200/域）：排除答案位置偏置——这是声称"format optimization"的前置条件。
   - canonical 脚本从 raw artifact 重建所有表/图（禁止手填论文数字）。
2. **Tier 1（核心对照 + 真实日志）**
   - 补 12 个 `1e-6` 保守 GRPO run（Math/Science/Medicine/Commonsense × 3 seeds）——补齐缺失的 conservative control。
   - **强制存真实逐步日志**：reward mean/std、policy-ref KL、grad norm、advantage-方差/frac_zero_std、entropy、clip frac。→ 机制实锤 + 探针输入。
   - dev-only 选 LR/checkpoint；test 只评一次。
3. **Tier 2（新主轴证据）**
   - disjoint-OOD 重做**迁移比例表（主图）**：修 ARC-C 同时当 ID 和 OOD；Math→MATH-500/GSM8K、Med→PubMedQA/MMLU-Med、Sci→GPQA、CS→WinoGrande/PIQA。
   - 探针扩到多 domain×model cell + leave-one-out，兑现摘要(3)的 "predictable"。
4. **Tier 3（诚实/完整）**
   - 非数学 "SFT" 正名为 **RFT / filtered self-training**；Science+Medicine 补 1 个真 gold/teacher SFT（本地 Qwen2.5-72B 蒸馏）。
   - 重出所有图表，删除合成 Figure 2/4。
5. **算力**：本轮先 2–3 卡跑探针 sweep（纯推理，K-rollout 算 advantage-方差）；后续 8 卡 + API 到位后跑 Tier 1/2 训练。

## 已废弃 / 待改的旧 claim（投稿前必须清掉）

- 66.6 / 68.8 / +25 / "40× above DeepSeekMath"：挑高 seed 或归因错误（DeepSeekMath 实为 1e-6，应为 20×）；"5e-7 is the DeepSeekMath lr" 同删。
- "consistently surpasses SFT in all domains"：现有 "SFT" 实为 RFT，且 Law/Commonsense 宏均值不支持。
- "pure reverse-KL minimization" / "captures new modes / acquires new knowledge"：机制过强，降为 under-update hypothesis + 真实日志佐证。
- "external signal is necessary"：OPD≈RFT，不成立；OPD 降为附录 null baseline。
- "five architecturally distinct families"：无 conservative 对照时不能声称；补 1e-6 对照后再说"多模型族"。
- Figure 2/4：手填合成曲线（伪 multi-seed variance），无原始日志 → 删除或用真实日志重画。
- test-set model selection：所有 LR/checkpoint 改为 dev-only 选择。
