# Paper 1 — AAAI-27 投稿快照（2026-07-21）

- **Venue**: AAAI-27
- **Deadline**: 摘要 2026-07-22 20:00 (Shanghai ≈ AoE)；Full paper 2026-07-28
- **状态**: 标题 + 摘要定稿；实验待跑（见下）
- **源稿**: `lxk/paper1_sanitized/`（原 EMNLP 稿，重构中）

---

## 标题

**When Helping Hurts: Failure-Structure-Aware Routing for Code-Agent Recovery**

## 摘要

Once a code agent commits its first mistake, the large majority—over 80%—of its subsequent steps make no progress toward a fix. Yet agents are usually handed a single, uniform recovery mechanism—retry, reflect, or re-prompt—regardless of why they failed. We show this uniformity is not merely suboptimal but actively harmful: an intervention applied to the wrong failure can leave the agent worse off than doing nothing at all. Analyzing failed trajectories on SWE-bench Verified, we find that agent failures carry distinct structural signatures—repetitive loops, mislocalized effort, commitment spirals, and reasoning drift—and that this structure predicts whether behavioral intervention can help. Matched interventions recover strongly, while mismatched ones fall below the no-intervention baseline, an asymmetry in which the cost of a wrong diagnosis exceeds the benefit of a right one. We exploit this asymmetry with a gold-free online classifier that routes each failure to a type-specific strategy or abstains when uncertain, yielding a recovery policy that captures the gains of matched intervention without the harm of uniform intervention. We validate the policy both at the trajectory level—multi-step progress toward a correct fix, selected without privileged information—and end-to-end via resolved rate on held-out instances. Agent recovery, we argue, is a diagnostic problem: intervening without first diagnosing failure structure risks making failing agents worse.

---

## 本轮实验规划（待跑，等 lxk 通知开始）

主心骨从「单步 0-3 proxy」换成 **leakage-free 多步轨迹恢复**（不依赖 docker，现在就能跑），docker 恢复后补 **resolved% 锚点**。

1. **多步轨迹恢复（无需 docker）**
   - fresh 跑 agent（本地 Qwen2.5-14B 或 4o-mini）→ 让它自己失败 → 在线 RF 分类（gold-free）→ first-error 处注入干预 → 续跑 K 步。
   - 条件：no-intervention / fixed-safe(step_back) / classifier-selected / classifier+abstention。
   - 指标：进度恢复率（评估端用 gold，合法）、cascade 打断率（gold-free）、edit 成功率（gold-free）。
2. **resolved% 锚点（docker 恢复后）**：EDIT/PLAN 上 20-50 例端到端 resolved%，把轨迹指标锚到 gold standard。
3. **降级保留**：原单步 0-3 结果 leakage-free 化后移入诊断附录；headline 不挂 proxy 数字。
4. **算力**：本论文分配 2 卡；本地 vLLM 跑 Qwen2.5-14B 当 agent，4o-mini API 明天 lxk 提供。

## 已废弃 / 待改的旧 claim

- 原 headline「Oracle type-specific 2.32 > Fixed 2.04」用 gold 选策略 → 作诊断，不作主结果。
- 「relevant」type-specific regex 维度 → 移除或仅作附录。
- LOGIC 无效 → 重新定位为「路由器对 reasoning drift 弃权 = 可部署的 scaffolding frontier」。
