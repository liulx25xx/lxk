# Phase B：实验冻结后的正文修改方案

## 进入条件

只有 [`02_RESULT_TRACKER.md`](02_RESULT_TRACKER.md) 的数据冻结、统计和 final claim gate 完成后执行本文件。Phase A 未完成时，不把任何 TODO 或旧 test-selected 数字写入正文。

## B1. 论文主线

推荐主线：

> GRPO optimization settings do not transfer reliably across domains. Conservative settings can produce insufficient effective policy movement, so update magnitude must be selected on held-out development data and evaluated with benchmark-level uncertainty.

这一表述保留核心发现，同时允许 Law、某些 component benchmark 或个别模型成为边界条件。

## B2. 标题候选

1. *When Conservative GRPO Under-Updates: Learning-Rate Calibration Across Domains*
2. *Calibrating GRPO Beyond Mathematics: A Controlled Multi-Domain Study*

删除 *One Hyperparameter to Rule Them All*。最终需要校准的是 effective update magnitude，LR 只是其中一个控制量。

## B3. 摘要结构

摘要按五句写：

1. 问题：GRPO 设置从数学领域迁移到其他领域，但其可迁移性未知；
2. 方法：共享 backbone/LoRA/data budget，以 dev-only selection 比较 conservative 与 calibrated GRPO；
3. 主结果：填入最终 four-domain macro、范围、seeds 和 CI；
4. 稳健性：benchmark decomposition、option permutation、parser audit、真实 update diagnostics；
5. 边界：结果并非 universal，结论是需要校准而非存在一个固定 high LR。

候选英文骨架在 [`paper/REVISION_CANDIDATES.md`](../../paper/REVISION_CANDIDATES.md)。

## B4. Contributions 最多三条

1. dev-selected、multi-seed、benchmark-level 的 controlled study；
2. conservative vs calibrated update 的跨领域经验结果及明确边界；
3. MCQ robustness 与真实训练诊断；若日志不完整，第三条只保留 robustness。

不要把 OPD、PPL predictor、zero-std predictor、cross-domain transfer 同时列成独立贡献。

## B5. 主文结果组织

### Main Table

列：Base、RFT、GRPO-1e-6、GRPO-dev-selected。行按 component benchmark 展示，并在 domain 后给 macro。每个 cell 给 mean±SD 或 CI，caption 说明是 independent training seeds。

### Figure 1：主结果

- 展示每个 training seed 的点；
- bar/marker 给 mean 与 95% CI；
- y 轴使用 accuracy change from base；
- Law 放 appendix；
- 不使用 test-selected “best” 标签，使用 “dev-selected”。

### Figure 2：LR sensitivity

- 横轴 log LR；
- dev curve 展示所有候选；
- test 只标最终选中的配置；
- 不把单 seed sweep 画成确定性规律。

### Figure 3：二选一

- 若日志完整：真实 reward/KL/update norm trajectory；
- 若日志不完整：option-permutation robustness 与 invalid-rate panel。

当前手填或插值生成的 Figure 2/3/4 全部删除，不从旧脚本复制数据。

## B6. 必须删除的表述

- `5e-7 is the learning rate used in DeepSeekMath`；
- `40x above the DeepSeekMath value`；
- `consistently surpasses SFT in all domains`；
- `the first controlled study`；
- `external signal is necessary`；
- `pure reverse-KL minimization`；
- `higher LR captures new modes/acquires new knowledge`；
- 没有 conservative controls 时的 `consistent across five architectures`；
- 由三个 domain 点得到的 PPL/zero-std predictor。

## B7. 术语统一

| 旧术语 | 修改后 |
|---|---|
| math-optimized default | conservative configuration inherited from math-focused studies |
| SFT（非数学自生成数据） | RFT / filtered self-training |
| matched compute | shared prompts/backbone/LoRA/evaluation；只有计量 GPU-hours/tokens 后才说 compute-matched |
| mechanism | under-update hypothesis / evidence consistent with |
| new knowledge | held-out benchmark improvement |
| five model families | five checkpoints from four organizations |
| OOD | cross-benchmark transfer；只有严格 disjoint 时使用 OOD |

## B8. AAAI-27 七页预算

| 内容 | 目标页数 |
|---|---:|
| Abstract + Introduction | 1.0 |
| Setup、dev-selection protocol | 1.0 |
| Main results | 1.5 |
| LR sensitivity + robustness/diagnostics | 1.5 |
| Analysis + limitations | 0.8 |
| Related work + conclusion | 1.2 |

移入 supplement：Law、完整 model list、OPD、SFT scaling、PPL、zero-std、详细 transfer matrix 和长理论推导。Supplement 不能承载理解主结论所必需的证据。

## B9. 最终修改顺序

1. 用 canonical snapshot 替换所有表格数字；
2. 从 raw artifacts 重新生成图；
3. 重写 abstract 和 introduction；
4. 重写 experiment setup，突出 dev/test protocol 与 seeds；
5. 将讨论改成 claim-evidence-boundary 结构；
6. 更新 related work，区分已有 step-size threshold/LR-gated failure 工作；
7. 切换 AAAI 模板，处理匿名、页数、字体和 US Letter；
8. 编译并逐页检查，确认正文没有 TODO、旧数字和无来源曲线。
