# AAAI-27 审稿意见驱动的补实验清单

更新日期：2026-07-22

这份文件只列“必须靠新实验或原始日志才能补”的内容。当前能仅靠现有证据修改的摘要、引言、实验设定、结果叙述、讨论、局限性和附录已经先改进 `paper/main.tex`。新结果出来前，不把占位数字或预期结论写进正文。

## 0. 先给结论：最低可投稿实验包

如果算力有限，先完成下面四项，顺序不要换：

1. **P0-A：主 LR 对照与独立种子**——四个主域上比较 `1e-6` 与 `2e-5`，每格 3 个独立训练种子，只用 dev 选配置；
2. **P0-B：MCQ parser/选项置换审计**——无需重训，验证增益不是固定答案位置或解析器造成；
3. **P0-C：LoRA 与 full-parameter 小型对照**——至少在 Medicine 上完成，回答审稿人最核心的“到底是 GRPO LR 还是 LoRA LR”质疑；
4. **P0-D：低 LR + zero-variance filtering 对照**——检验学习率提高是否只是替代了 DAPO 式信号过滤。

四项都完成后，论文主张可写成：

> 在固定 LoRA 参数化和 dev-only 选择下，保守 LR 容易 under-update；校准后的更新强度在多个可验证域上更好，但效果受参数化与有效训练信号共同影响。

若 P0-C 或 P0-D 不完成，正文必须继续保留“LoRA-specific observation”和“不能排除 signal filtering 解释”的边界，不能写成 GRPO 的普遍规律。

## 1. P0-A：主 LR 矩阵（最先跑）

### 目的

回答以下审稿问题：5e-7 不是 DeepSeekMath 默认；旧实验种子数不等；有些均值来自 1–2 个 run；LR 和 checkpoint 可能在 test 上选择。

### 固定设置

- Model：Qwen2.5-7B-Instruct；
- Parameterization：LoRA rank 64, alpha 128；
- GRPO：`K=8`, `beta=0.001`, 相同 prompt、数据预算、batch 与最大生成长度；
- Domains：Math、Science、Medicine、Commonsense；Law 只作 appendix stress test；
- LRs：`1e-6`（DeepSeekMath 报告的 policy LR 对照）与 `2e-5`（当前高 LR 候选）；
- Seeds：42、123、456；
- 选择：dev 选 LR/checkpoint，冻结 `selection_manifest.json` 后 test 只评一次。

### 运行矩阵

| Domain | 1e-6 | 2e-5 | 新增量 |
|---|---:|---:|---:|
| Math | 3 seeds | 3 seeds | 高 LR 旧 run 通过 provenance 可复用 |
| Science | 3 seeds | 3 seeds | 同上 |
| Medicine | 3 seeds | 3 seeds | 高方差，不能少于 3 seeds |
| Commonsense | 3 seeds | 3 seeds | 旧 68.8 不可直接复用 |

完整矩阵为 24 runs；若 12 个高 LR 旧 run 都能从训练 config 证明为独立训练，本轮只新增 12 个 `1e-6` runs。

### 启动示例

```bash
DRY_RUN=1 bash scripts/launch/launch_aaai27_p0_grpo.sh medicine 1e-6 42 0
bash scripts/launch/launch_aaai27_p0_grpo.sh medicine 1e-6 42 0
```

### 必须保存

- 实际训练 config、training/data seed、git commit 与数据 hash；
- step-level reward mean/std、KL、gradient/update norm、clip fraction、zero-std fraction；
- dev/test 逐样本 prediction、parsed answer、gold、item ID；
- 每个 benchmark 单独得分，不只保存 Science/Commonsense pooled accuracy。

### 写回论文的判定规则

- 四域多数在 dev-selected 2e-5 下显著优于 1e-6：保留“conservative LoRA-GRPO under-updates”的主线；
- 域间最优 LR 明显不同：标题和结论改为“domain-dependent calibration”，不说固定高 LR 普适；
- test 增益不稳定或 CI 跨 0：把工作改成 failure-analysis/measurement paper，不再主打性能提升。

## 2. P0-B：MCQ parser 与选项置换（不重训）

### 目的

Science、Medicine、Commonsense、Law 都是 MCQ，审稿人担心增益来自答案位置、输出格式或 parser，而不是能力变化。

### 设计

对 Base、GRPO-1e-6、dev-selected GRPO：

1. 每个域人工核验 200 条原始输出；
2. 统计 invalid、empty、multiple-answer、parser-disagreement rate；
3. 对每道题用 3 个固定 permutation seed 重排选项并同步 gold label；
4. 报告 original-order accuracy、permutation mean/SD、答案位置分布；
5. Science 和 Commonsense 必须分 component benchmark 报告，再给 macro average。

### 通过标准

- 主要增益在 option permutation 后仍存在；
- 方法间 parser failure 差异不足以解释 headline gain；
- 若不通过，正文把结论限定为 format/answer-selection improvement，并删除 knowledge/reasoning 表述。

## 3. P0-C：LoRA vs full-parameter 对照

### 目的

这是三位审稿人共同的关键质疑：当前观察可能是 LoRA 的有效步长现象，不能直接外推到 full-parameter GRPO。

### 最小可行设计（推荐先跑）

只跑 Medicine，避免一次铺满四域：

| Parameterization | LR | Seeds |
|---|---:|---:|
| LoRA r64/a128 | 1e-6 | 42, 123, 456（可复用 P0-A） |
| LoRA r64/a128 | 2e-5 | 42, 123, 456（可复用 P0-A） |
| Full parameter | 1e-6 | 42, 123, 456 |
| Full parameter | 5e-6 | 42, 123, 456；先 pilot，若发散则降到 2e-6 |

不要直接把 full-parameter `2e-5` 当正式点；先做 5% steps 的 stability pilot，监控 loss、KL、grad norm、NaN 与 reward collapse。

仓库已有 `src/training/train_grpo_trl_fullft.py`，但运行前必须确认其 batch、scheduler、gradient accumulation、generation config 与 LoRA 路径一致。

### 关键输出

- accuracy 与 95% CI；
- trainable parameter count；
- 每 token/update 的 KL 与 update norm；
- tokens、optimizer steps、GPU-hours；
- 最好把 x 轴从 nominal LR 改为 observed policy movement，避免把不同参数化的 LR 数值直接等同。

### 写回规则

- 只有 LoRA 需要较高 LR：标题和全文明确写 LoRA-GRPO calibration；
- 两种参数化都显示 conservative setting under-update：可在讨论中提出更一般的 effective-update hypothesis；
- full FT 与 LoRA 方向相反：将其作为主要发现，强调 LR 不能跨参数化迁移。

## 4. P0-D：学习率 vs signal-aware filtering

### 目的

回答“zero-variance group filtering/DAPO 已能解决无效梯度，为什么还需要提高 LR？”

### 设计

先在 Medicine 和 Science 跑；固定 Qwen2.5、LoRA、数据、总采样 tokens 与 optimizer updates：

| Condition | LR | Group handling | Seeds |
|---|---:|---|---:|
| Conservative | 1e-6 | 保留所有 groups | 3 |
| Filtering | 1e-6 | 跳过 zero-variance groups，并补采样到相同有效 updates/tokens | 3 |
| Calibrated | 2e-5 | 保留所有 groups | 3 |
| Combined | 2e-5 | 同样 filtering | 3（有算力时） |

只“删掉无信号 group”但不补足 tokens/updates 会改变算力预算，不能作为公平对照。

### 报告

- dev/test accuracy；
- zero-std fraction、有效 group 数、总生成 tokens；
- reward/KL/update norm 轨迹；
- wall-clock 和 GPU-hours。

### 写回规则

- filtering 在低 LR 下追平高 LR：主结论改为“有效信号利用”，LR 只是实现方式之一；
- high LR 仍明显更好：可说过滤和 update calibration 互补；
- combined 最好：给出二维建议，不再推荐单一 LR。

## 5. P1-A：RFT 与 FST 的公平对照

### 当前问题

旧稿把非数学 SFT 写成 teacher SFT，又把 OPD 写成“无外部信号”。实际上两者都用 gold correctness 过滤自生成响应：RFT 保留最短正确响应，FST 保留第一个正确响应，且 retained prompts/长度不同。

### 设计

在 Science、Medicine 上，共享同一批 prompts 与同一组 `K=8` generations，构造：

1. First-correct FST；
2. Shortest-correct RFT；
3. Random-correct control；
4. 可选：all-correct，按固定 token budget 截断。

匹配 retained prompt 数、训练 tokens、optimizer updates、epoch 与 LoRA 设置；每格 3 seeds。若要比较“外部 teacher”，需另加真正 teacher/gold rationale baseline，不能把 shortest-correct RFT 当 teacher SFT。

### 价值

这项不是主线必需，但能彻底解决方法命名和公平性问题；若算力不足，保留当前正文的保守表述并把结果放 supplement。

## 6. P1-B：去污染/抗饱和评测

### 目的

回应 GSM8K、ARC、MMLU 类旧 benchmark 可能被预训练见过或已饱和，reward-test gap 不能证明 latent knowledge consolidation。

### 最低设计

- 对 train/dev/test 做 exact、normalized question、8-gram/MinHash 去重并保存 overlap report；
- Math 增加一个在 Qwen2.5 训练截止后发布的、严格 held-out 的竞赛题集合；
- Medicine/Science 至少增加一个训练题源不重叠的开放式或生成式评测；
- 对 Base、1e-6、dev-selected LR 用同一 parser/rubric；必要时盲评 200 条。

如果无法证明 benchmark 发布时间晚于模型预训练截止，正文只称 `cross-benchmark transfer`，不要称 contamination-free OOD。

## 7. P1-C：跨模型 LR sensitivity

旧稿只有其他模型的 2e-5 单点，只能说明“这个设置能提高某个 checkpoint”，不能说明 LR 曲线跨模型成立。

建议只选 Medicine：

- Qwen3-8B：1e-6 vs 2e-5，3 seeds；
- 一个非 Qwen checkpoint：1e-6 vs 2e-5，3 seeds；
- 完全相同的数据、LoRA target modules、dev selection 与评测。

完成后才能把“cross-model validation”放主文；否则保留为 appendix spot checks。

## 8. P2：有算力再补，不应阻塞主投稿

### 大样本预测器验证

当前只有三个 domain 点，不能声称 perplexity 或 `frac_reward_zero_std` 能预测最优 LR/OOD transfer。若要保留预测器贡献，需要至少 20 个 model-domain-condition 点，预注册 threshold/回归形式，并在独立 holdout 上验证。否则删除 predictor/phase diagram，只保留 exploratory diagnostic。

### Hybrid mechanism

若要声称过量 SFT 降低探索，需要在 `SFT N × GRPO LR` 网格中匹配 compute，并记录 response diversity、group reward variance、KL 和 update norm；当前单种子/双种子 hybrid 只能作现象描述。

### Law

Law 训练只有约 31 条，不能与四个主域合并声称统一规律。可增加数据并做多 seed，或维持 appendix boundary case。

## 9. 统一统计与交付格式

主比较至少报告：

- 每个独立 training seed 的点；
- mean、SD、95% CI 与确切 `n`；
- 同一 test items 上的 paired bootstrap/McNemar；
- component benchmark 分数和 domain macro；
- 多域检验用 Holm correction，或预先指定 overall macro 为唯一主检验。

每个 run 的目录至少包含：

```text
train_config.json
data_manifest.json
trainer_state.json 或 log_history.jsonl
train.log
eval/dev_summary.json
eval/dev_predictions.jsonl
eval/test_summary.json
eval/test_predictions.jsonl
checksums.sha256
```

## 10. 你跑完后给我的最小回传包

把下面内容放回本项目即可，我会据此自动重建表格、图和正文：

1. `outputs/aaai27_*/**/train_config.json`；
2. step-level trainer logs；
3. dev/test prediction JSONL；
4. `selection_manifest.json`；
5. 数据 split manifest 与 hash；
6. 失败/发散 run 也保留 config 和 log，不要只回传成功 run。

随后按以下顺序改论文：先 canonical audit，后统计与图，再替换正文数字，最后切 AAAI 模板和压页数。不要人工把结果抄进 TeX。
