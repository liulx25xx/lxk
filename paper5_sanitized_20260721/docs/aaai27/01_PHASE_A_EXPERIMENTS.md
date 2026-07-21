# Phase A：实验执行方案

## A0. 运行前先恢复缺失资产

当前 sanitized 包不包含 `data/`、`outputs/` 和 `logs/`。在开始新实验前，把服务器上的对应资产恢复到项目根目录，至少包括：

```text
data/processed/<domain>/train.jsonl
data/processed/<domain>/dev.jsonl
data/processed/<domain>/rlvr_test.jsonl
outputs/<historical runs>/train_config.json
outputs/<historical runs>/trainer_state.json or equivalent log history
outputs/<historical runs>/final/adapter_config.json
logs/<historical runs>.log
```

不要把模型权重提交到 GitHub。checkpoint 保留在训练服务器，仓库中只保存 config、压缩后的 log、prediction JSONL、summary CSV 和生成脚本。

## A1. Provenance audit：先确认旧 runs 能不能复用

为每个旧 run 建立一行 manifest，至少记录：

| 字段 | 要求 |
|---|---|
| `run_id` | 唯一、包含 domain/method/lr/seed |
| `train_seed` | 从训练 config 读取，不能从目录名猜 |
| `data_seed` | 数据抽样 seed |
| `eval_seed` | 若 greedy evaluation 则记为 N/A |
| `git_commit` | 训练代码 commit；未知则写 `unknown` |
| `data_hash` | train/dev/test 文件 SHA-256 |
| `checkpoint` | 最终 adapter 路径 |
| `config_path` | 实际训练配置 |
| `log_path` | 原始训练日志 |
| `prediction_path` | 逐样本预测 |
| `status` | valid / invalid / provenance-missing |

已知风险：旧 evaluation code 只正确识别 seed 42/123，seed 456/789 可能在 JSON metadata 中被误写成 42。本项目已修复新评测时的目录 seed 解析，但旧 JSON 仍需从训练 config 回填。

只有能够证明为独立训练的 runs 才能进入 multi-seed 统计。目录名不同但 checkpoint/config 相同的重复评测不能算多个训练 seed。

## A2. 冻结数据划分

### 原则

- 使用官方 validation 作为 dev；若无官方 dev，则从 training data 中按 benchmark/subtask 分层抽取固定 dev；
- test 不能参与 LR、checkpoint、epoch、prompt 或 parser 的选择；
- 保存每个 split 的 item ID 和 SHA-256；
- 先做 exact ID、normalized question 和高重合 n-gram 去重；
- 所有方法共享完全相同的 train/dev/test items。

### 组合 benchmark

| Domain | 必须分开报告 | 主聚合 |
|---|---|---|
| Math | GSM8K；其他数学集单列 | 预先指定的 benchmark，不临时混合 |
| Science | ARC-Challenge、ScienceQA | 两个 benchmark 的 macro average |
| Medicine | MedQA | accuracy |
| Commonsense | ARC-Easy、HellaSwag | 两个 benchmark 的 macro average |
| Law | 每个 LegalBench subtask | task macro；仅作为 appendix boundary case |

## A3. P0 主训练矩阵

固定设置：Qwen2.5-7B-Instruct、LoRA r=64/alpha=128、K=8、beta=0.001、N=2000、相同 prompts、约 3 epochs。除 LR 和 training seed 外其他设置保持一致。

| Domain | LR | Seeds | 优先级 |
|---|---|---|---|
| Math | 1e-6 | 42, 123, 456 | P0 |
| Math | 2e-5 | 42, 123, 456 | P0；旧 run 通过 provenance 后可复用 |
| Science | 1e-6 | 42, 123, 456 | P0 |
| Science | 2e-5 | 42, 123, 456 | P0；旧 run 通过 provenance 后可复用 |
| Medicine | 1e-6 | 42, 123, 456 | P0 |
| Medicine | 2e-5 | 42, 123, 456 | P0；高方差，不能少于 3 seeds |
| Commonsense | 1e-6 | 42, 123, 456 | P0 |
| Commonsense | 2e-5 | 42, 123, 456 | P0；旧 run 通过 provenance 后可复用 |

总矩阵 24 runs。如果 12 个 high-LR runs 都被 provenance audit 判为 valid，本轮只需新增 **12 个 `1e-6` runs**。

### 单 run 启动方式

先设置本机/服务器路径：

```bash
export PYTHON_BIN=/absolute/path/to/openrlhf/bin/python
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
export DATA_ROOT=/absolute/path/to/project/data/processed
export OUTPUT_ROOT=/absolute/path/to/project/outputs/aaai27_p0
```

先 dry run 检查命令：

```bash
DRY_RUN=1 bash scripts/launch/launch_aaai27_p0_grpo.sh medicine 1e-6 42 0
```

确认后执行：

```bash
bash scripts/launch/launch_aaai27_p0_grpo.sh medicine 1e-6 42 0
```

输出目录名会显式包含 domain、LR、seed 和 N，训练脚本还会保存 `train_config.json`。不要通过改目录名制造 seed；每次都必须将 `--seed` 传入训练程序。

## A4. Dev selection 与 test protocol

### Dev 阶段

1. 在 dev 上比较 `1e-6` 和 `2e-5`；更细的 LR sweep 若需要，只能在 dev 进行；
2. 在每个 training seed 内选择 checkpoint，再按预先固定的 aggregation rule 选择最终 LR；
3. 推荐规则：最高 domain macro dev accuracy；差异小于 0.2pp 时取较低 LR；
4. 记录 selection rule、候选数量和最终选择，不事后修改。

### Test 阶段

- dev 选择完成并写入 `selection_manifest.json` 后才允许跑 test；
- 每个 seed 的最终 checkpoint 在 test 上评一次；
- 不因 test 较差而换 checkpoint 或增加一个新 LR；
- 保存逐样本 prediction、解析后的答案、gold、benchmark/subtask、item ID。

## A5. MCQ parser 与 option-permutation 审计

这一组不需要重新训练，应与主评测并行完成。

1. 每个 domain 随机抽 200 条原始输出，人工核验 parser 判分；
2. 统计 invalid/empty/multiple-answer rate；
3. 对每道 MCQ 生成 3 个固定 permutation seeds，重排选项并同步 gold label；
4. 报告 original-order accuracy、三次 permutation mean/SD、answer-position distribution；
5. 对 Base、GRPO-1e-6、dev-selected GRPO 使用同一套 permutations。

完成判据：主要增益在 permutation 后仍存在，且不同方法的 parser failure 差异不能解释 headline improvement。

## A6. 统计分析

主比较以 domain macro accuracy 为主：

- 逐 training seed 报告结果；
- hierarchical bootstrap：先重采样 training seeds，再重采样 test items；
- 同一 test items 上做 paired bootstrap 或 McNemar；
- 多 domain 检验使用 Holm correction，或预先指定一个 overall macro 主检验；
- 同时报 absolute pp difference、95% CI 和 exact test count。

不以单次 run 或重叠置信区间的目测结果宣称显著性。

## A7. 每个 run 必须保留的 artifact contract

```text
<run_dir>/
  train_config.json
  data_manifest.json
  trainer_state.json             # 或结构化 log_history.jsonl
  train.log
  final/adapter_config.json
  eval/dev_summary.json
  eval/dev_predictions.jsonl
  eval/test_summary.json          # 仅 dev selection 冻结后生成
  eval/test_predictions.jsonl
  checksums.sha256
```

训练曲线至少需要真实记录：step、reward mean/std、policy-reference KL、gradient norm、learning rate；若 trainer 可导出，再加 clip fraction、entropy、zero-std fraction 和 generated tokens。

## A8. P1：P0 完成后再决定是否跑

### 公平监督基线

在 Science 和 Medicine 补一个真正的 external teacher/gold SFT；现有非数学 “SFT” 更名为 RFT/filtered self-training。报告 prompts、retained examples、response tokens、optimizer updates 和 GPU-hours。

### 跨模型

选择 Medicine，在 Qwen3-8B 与一个非 Qwen checkpoint 上比较 `1e-6` 和 `2e-5`，各 3 seeds。只有同时存在 conservative control 才能声称跨模型 LR sensitivity。

### 真正的 transfer/OOD

训练 benchmark 与 OOD benchmark 必须 disjoint，并使用相同 prompt/parser。Science 主结果已经包含 ARC-Challenge，因此不能再把 ARC-Challenge 称为 Science OOD。

## A9. Phase A 完成判据

- [ ] provenance manifest 完成，旧 runs 已分 valid/invalid；
- [ ] dev/test split 和哈希冻结；
- [ ] P0 主矩阵每格至少 3 个独立 seeds；
- [ ] test 只在 selection manifest 冻结后运行；
- [ ] parser/permutation audit 完成；
- [ ] benchmark-level 与 macro 统计完成；
- [ ] 所有主数字能由 canonical pipeline 重建；
- [ ] 将最终 snapshot 路径写入 `02_RESULT_TRACKER.md`；
- [ ] 通过后才开始 `03_PHASE_B_PAPER_REVISION.md`。
