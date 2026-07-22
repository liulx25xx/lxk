# AAAI-27 实验优先修订：从这里开始

最后更新：2026-07-22

## 当前工作方式

本轮修订分成两个阶段。已经能依据现有 artifacts 和审稿意见确定的事实错误、方法命名和过强结论已先在 `paper/main.tex` 中修正；所有 headline 数字、最终主图和机制结论仍等待 Phase A 冻结后再定稿。

### Phase A：先完成实验和证据链

这一阶段只做：

1. 恢复并核验训练数据、训练配置、checkpoint 和 raw logs；
2. 建立与 test 严格隔离的 development split；
3. 补齐 conservative `1e-6` 对照和独立训练 seeds；
4. 用 dev 选择 LR/checkpoint，test 只评一次；
5. 做 benchmark-level 聚合、统计检验和 MCQ 评测审计；
6. 冻结一个可以从 raw artifacts 自动重建的 result snapshot。

在 Phase A 完成以前：

- 不更新摘要中的 headline 数字；
- 不把 `paper/audit_results/fig_main_result_audit.*` 当投稿图；
- 不使用旧 tracker 中的 “universal”“all five domains”等结论；
- 不再根据 test 结果追加 LR 或选择 checkpoint。

具体执行顺序见 [`01_PHASE_A_EXPERIMENTS.md`](01_PHASE_A_EXPERIMENTS.md)，按审稿意见整理的最小实验包见 [`04_REVIEW_DRIVEN_EXPERIMENT_LIST.md`](04_REVIEW_DRIVEN_EXPERIMENT_LIST.md)，结果填写见 [`02_RESULT_TRACKER.md`](02_RESULT_TRACKER.md)。

### Phase B：实验冻结后再改正文

只有 Phase A 的 Go/No-Go gate 全部通过，才最终确定：

1. title、abstract、introduction 和 contributions 中的最终主张与 headline 数字；
2. 根据 dev-selected test 结果重新生成主表和主图；
3. 删除无原始日志支持的曲线和机制结论；
4. 切换 AAAI-27 模板并压缩到 7 页正文。

具体改写方案见 [`03_PHASE_B_PAPER_REVISION.md`](03_PHASE_B_PAPER_REVISION.md)。

## 文档优先级

发生冲突时，按以下顺序执行：

1. 本目录 `docs/aaai27/`；
2. `AAAI27_SUBMISSION_AUDIT.md`；
3. `paper/audit_results/` 中由脚本生成的审计文件；
4. 旧的 `EXPERIMENT_TRACKER.md`、`EXPERIMENT_PLAN.md` 和其他顶层研究笔记仅作历史参考。

## 当前核心假设

> Conservative GRPO settings can under-update the policy outside their original calibration regime; effective update magnitude should be selected on held-out development data.

这是一条待检验的经验假设，不预设以下结论：

- high LR 在所有 domain 都有效；
- high LR 一定超过外部 teacher SFT；
- LR 改变了 KL divergence 的方向；
- benchmark accuracy 上升等于获得新知识。

## Phase A Go/No-Go gate

以下条件全部满足才进入正文修改：

- [ ] 每个 seed 都能证明是独立训练，而不是重复评测；
- [ ] 每个 run 都有 config、log、checkpoint、raw predictions；
- [ ] LR 和 checkpoint 仅由 dev 选择；
- [ ] `1e-6` 与候选 high LR 每组至少 3 个 training seeds；
- [ ] Science/Commonsense 等组合 benchmark 已分别报告子任务和 macro average；
- [ ] MCQ parser/permutation audit 通过；
- [ ] 所有主结果由一个 canonical script 自动生成；
- [ ] 正文将使用的训练曲线全部来自真实日志；
- [ ] 结论在独立 seeds 和 macro aggregation 下仍然成立。
