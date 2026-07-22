# 方向决策 + 摘要草稿 + 补实验计划（Claude 接管版）

最后更新：2026-07-21
状态：**方向已定**；摘要草稿可直接用于 2026-07-22 20:00 AAAI-27 摘要截止；docker/训练环境恢复后按下方计划补实验。
作者：Claude（接管补实验与改稿）
约束：本文件及后续一切产出**只放在 `paper5_sanitized_20260721/` 内**，不外流。

---

## 0. 硬约束与 deadline（决定可行性，先读）

- **摘要**：2026-07-22 20:00（明天）。只注册，不计分。**用 §2 草稿即可交，不需要任何新 run。**
- **Full paper**：2026-07-28（摘要后 **7 天**）。不是几周，是 7 天。
- **环境**：当前 docker/训练环境坏，跑不了新训练；据称 7/28 前会恢复。系统 python3 + vLLM + 多卡推理应该可用（参考 safety_V1）。
- **资产现状（最关键）**：本 sanitized 包**只有 `eval_results/`（最终准确率 + 样本预测）和 `scripts/`，没有 `data/`、没有 `outputs/`(checkpoints)、没有 `logs/`(训练日志)**。即现有 eval JSON 的训练 provenance 已丢失 → 审计 P0-A 可能判很多 run invalid/重复。
- **第一道关卡（环境一好就做）**：确认训练服务器上 `data/ outputs/ logs/` 能不能恢复。
  - 能恢复 → 7/28 = 重出图表 + 补 12 个保守对照 run（轻松）。
  - 不能恢复 → 7/28 = 从零重训核心 ~24 run（40 卡 H200 + vLLM 下，5–6 天可做完 Tier 0–1 + 裁剪版 Tier 2，紧但可行）。
- 因此 **7 天现实版 scope = Tier 0 + Tier 1 + 裁剪版 Tier 2（迁移表为主，预测器 best-effort）**；Tier 3 仅当资产可恢复且有余力；Tier 4 砍。

---

## 0.5 一句话方向

> **RLVR 的提升不是等价的：可迁移的"能力固化" vs 不可迁移的"格式/答题分布优化"，二者比例随知识领域剧烈变化，且可在训练前用一次 K-rollout 探针（有效 advantage 方差 = 梯度信号质量）预测。学习率/欠更新只是"把有效更新幅度匹配到可用梯度信号"的一个旋钮。**

把论文的"头"从 *One Hyperparameter / 40× / mode-seeking trap* 换成 **"RLVR 涨点有多少能迁移 + 训练前预判"**。学习率发现保留为机制佐证与修复手段，不当 headline。

## 0.1 为什么是这个方向（中稿概率最高）

- 现稿机制过强（reverse-KL / new-modes / frac predictor 只 3 点）→ 审计判 ~Reject 2.5/10。
- 审计的 salvage（"calibrate lr on dev"）能到 borderline，但 novelty 薄、偏防守。
- 本方向：novelty 真（"RLVR 涨点可迁移比例可预判"切中 Yue/Invisible-Leash/清华-critique 热点）、overclaim 最低、最大化复用已有 200+ runs、天然化解审计全部 objections、AAAI 喜欢（diagnostic + 干净对照 + 理论锚点）。

---

## 1. 真实数据快照（来自 `paper/audit_results/canonical_summary.csv`，可信）

| Domain | Base | 保守GRPO(5e-7) | 高LR GRPO(真实均值) | 判定 |
|---|---:|---:|---:|---|
| Math | 84.4 | +0.6 | **+7.6** (4 seeds) | 稳 |
| Science | 71.1 | +0.4 | **+8.0** (4) | 稳；需按 ARC-C / ScienceQA 拆 |
| Medicine | 59.2 | −0.4 | **+5.2** (4)；正文 66.6 是挑高 seed | 稳；数字改 |
| Commonsense | 43.8 | +0.6 | +16.0 但 **sd=6.7** | 方向真、幅度噪声大 |
| Law | 57.6 | −3.0 | +1.0 | 弱 → 附录 boundary case |

**必须删除/改写的表述**：66.6、68.8、+25、"40× above DeepSeekMath"、"5e-7 is the DeepSeekMath lr"、"consistently surpasses SFT in all domains"、"pure reverse-KL"、"captures new modes / acquires new knowledge"、"external signal is necessary"、"five architecturally distinct families"、由 3 点得到的 PPL/zero-std predictor（作主贡献）。

---

## 2. 摘要草稿（明晚 8 点交，无需新 run）

### 2.1 标题（已定，2026-07-21，探针前置版）

**Not All RLVR Gains Transfer: A Single-Pass Probe for the Transferable Fraction of Post-Training Improvement**

入选理由：①把最 novel 的工具（single-pass 探针）顶到标题，AAAI 更买账"给了一个可复用的 diagnostic"；②副标题用**可测的** "transferable fraction" 取代机制论断 "format optimization"，顺带消掉"标题背机制"的 overcommit 风险（机制留给正文）；③保留你选的 hook "Not All RLVR Gains Transfer"（最意外、最可验证、最便宜能支撑）。
代价：标题把赌注压在探针能做实上；后面 8 卡 + API 到位后用多 cell + LOO 兑现即可。
备选（若探针 cell 最终不足，回退用更稳的发现型标题）："Not All RLVR Gains Transfer: Consolidation vs. Format Optimization in a Controlled Multi-Domain Study"。

### 2.2 Abstract（FINAL，~250 词，纯为中标的最强可辩护版）

> Reinforcement learning with verifiable rewards (RLVR) has become the default recipe for improving LLM reasoning, and practitioners now apply the same math-calibrated configuration to medicine, science, and beyond. We ask a question that is rarely tested: do the resulting gains reflect genuine capability acquisition, or in-distribution format optimization? In a controlled study holding the base model (Qwen2.5-7B-Instruct), LoRA configuration, training prompts, and evaluation protocol fixed across four domains—mathematics, science, medicine, and commonsense—we compare supervised fine-tuning and GRPO across over 200 training runs and validate on additional model families. Three findings emerge. (1) The standard math-calibrated GRPO configuration yields near-zero test improvement in every domain despite rising training reward: its conservative step size under-updates the policy, and matching the effective update magnitude to the available gradient signal restores multi-percentage-point gains. (2) These gains are not created equal. An out-of-distribution probe shows that gains in knowledge-uncertain domains (medicine) transfer almost fully, whereas gains in already-saturated domains (mathematics, commonsense) barely transfer—evidence of format optimization rather than capability acquisition, which we confirm is not an artifact of answer-position bias through option-permutation audits. (3) The transferable fraction of RLVR gains is predictable before training from a single K-rollout probe measuring the effective advantage-variance of a domain. Together these results reframe post-training recipe selection: the relevant question is not which recipe wins in-distribution, but which produces gains that transfer.

> 备选保守版（仅当 7/28 前探针 cell 不足）：把 (3) 的 "is predictable" 改为 "correlates, before training, with"，并删 "validate on additional model families"。默认交 FINAL 版——后面 8 卡 + API 到位后用多 cell + LOO 把探针做实即可兑现。

### 2.3 摘要每句的证据与兜底（FINAL 版）

| 句 | 兑现方式（后面补） | 风险 |
|---|---|---|
| 4 域、200+ runs、同 backbone/LoRA/data/eval | 现有 eval_results + provenance audit | 低 |
| 保守 GRPO 全域≈不动、欠更新、加大有效更新→涨 | canonical_summary 多 seed | 低 |
| 迁移分化（医学高 / 数学·常识低）| disjoint-OOD 重评（修 ARC-C 双用）| 中，定性不变 |
| 非 answer-position 偏置（permutation 审计）| 3 perm × 200/域，纯推理 | 低 |
| 探针 predicts 可迁移比例 | 多 domain×model cell + LOO（8 卡可做）| **中**——最进取一句，做实即可 |
| 多模型族验证 | 给 Qwen3/Mistral/Yi/DeepSeek 补 conservative 1e-6 对照 | 中 |

### 2.3 摘要里每句话的证据强度（明晚交之前自查）

| 句 | 现有证据 | 风险 |
|---|---|---|
| 保守 GRPO 全域≈不动 | canonical_summary，多 seed | 低 |
| 欠更新→加大有效更新→涨 | canonical_summary Math/Science/Medicine | 低 |
| 迁移比例按领域分化（医学高、数学/常识低） | 现有 OOD 表（需 docker 后用 disjoint 集重做） | 中——重做后定性结论不会变 |
| 单次探针可预测可迁移比例 | 现 3 点；需扩到 ~16–20 cell + LOO | **较高**——这是最进取的一句，docker 后必须做实；做不出来就降级为"a useful signal" |
| 多模型家族验证 | 现仅有 high-LR；需补 1e-6 对照 | 中——docker 后补 |

> 结论：摘要可按时交；唯一需要 docker 后"兜底"的是探针那句，已用"validate across multiple domains and model families"留了余地。

---

## 3. docker 恢复后的补实验计划（分优先级，全部产物落本目录）

文件命名约定：新实验目录 `outputs/aaai27_v2/<run_id>/`，eval 产物 `eval_results/v2/`，日志 `logs/v2/`，canonical 脚本 `src/analysis/canonical_v2.py`。

### Tier 0 —— 不需训练，docker 一好（甚至推理可用时）就做

- [ ] **A1 Provenance audit**：给每个旧 run 建 manifest（run_id / 真实 train_seed 从 config 读 / data_hash / checkpoint / log / prediction / status）。只有"独立训练"才算多 seed。→ `outputs/aaai27_v2/manifest.csv`
- [ ] **A2 冻结 train/dev/test**：官方 validation 当 dev；无则按 benchmark/subtask 分层抽 dev；test 不参与任何选择；存 item ID + SHA-256；做 exact/n-gram 去重。组合 benchmark 必须分子任务 + macro。
- [ ] **A5 MCQ option-permutation + parser 审计**（3 个固定 permutation seeds × 每域 200 条；报 original/perm-avg/invalid-rate/answer-position 分布）。**这步对本方向尤其关键**——我们声称"格式优化"，必须先排除选项位置偏置。
- [ ] **A6 重聚合**：单一 canonical 脚本从 raw artifact 重建所有均值/SD/表/图，脚本内禁止手填论文数字；hierarchical bootstrap（先重采样 seed 再重采样 test item）+ paired McNemar + Holm。

### Tier 1 —— 核心：补缺失的保守对照 + 真实训练日志（机制与预测器的命根子）

- [ ] **A3/A7 12 个新 1e-6 GRPO runs**：Math/Science/Medicine/Commonsense × seeds{42,123,456}。**强制保存逐步结构化日志**：step, reward mean/std, policy-ref KL, gradient norm, **advantage-variance / frac_zero_std**, entropy, clip fraction, lr, generated tokens。→ 这是"欠更新"机制的实锤，也是预测器输入。
- [ ] **A4 Dev-only selection**：dev 上选 LR/checkpoint（规则：最高 domain macro dev acc，差<0.2pp 取更低 LR；规则先写死）；selection_manifest.json 冻结后才评 test；test 每 seed 只评一次。

### Tier 2 —— 新主轴的核心证据：迁移分解 + 预测器

- [ ] **A8-transfer 真·disjoint OOD**：修掉"ARC-C 同时当 ID 和 OOD"。Math↔{GSM8K, MATH-500}；Medicine→PubMedQA/MMLU-Medical；Science→GPQA-diamond；Commonsense→WinoGrande/PIQA。对 Base / dev-selected GRPO / RFT 重评，得**迁移比例表（主图）**。
- [ ] **预测器扩到 ~16–20 cell + LOO**：
  - 免费增 cell：在 Math/Medicine 现有 test 预测里按 base accuracy 分难度子桶（easy/med/hard），每桶算 frac_zero_std + 迁移比例。
  - 跨模型增 cell：Medicine 在 Qwen3-8B + 1 个非 Qwen（Mistral-7B 或 Yi-1.5-9B）上跑 1e-6 vs 2e-5，各 3 seeds。
  - 算 frac_zero_std ↔ 迁移比例 的关系；leave-one-domain-out；报相关性 + 失败 case。
- [ ] 重画主图：迁移比例 vs frac_zero_std 散点（每个点一个 domain×model cell，带 CI）。

### Tier 3 —— 诚实/完整

- [ ] **A8-fair 正名 + 真 SFT**：现有非数学"SFT"统一改名 **RFT / filtered self-training**；在 Science + Medicine 补 1 个**真 gold/teacher SFT**（用本地强模型如 Qwen2.5-72B-Instruct 或 Qwen3 蒸馏 CoT），与 GRPO 同 prompts；报 prompts/retained/tokens/updates/GPU-hrs。若算力不够，则**删除"GRPO beats SFT"贡献**，限定为"beats matched filtered self-training"。
- [ ] **B5 重出所有图表**：只从 raw artifact 生成；删除合成 Figure 2/4；Figure 3 若补不齐真实点也删。Law/OPD/完整模型清单/SFT-scaling/PPL 进 supplement。

### Tier 4 —— 仅时间富余

- [ ] 难度分层图；SFT perplexity predictor（要么扩到足够点，要么删主文）。

### Go/No-Go（与审计 gate 一致 + 新增）

- provenance manifest 完成、旧 run 分 valid/invalid；
- dev/test 冻结、test 仅 selection 后评；
- 1e-6 与 dev-selected 每 domain ≥3 独立 seed；
- 组合 benchmark 按 macro 报，结论仍成立；
- permutation/parser audit 未暴露严重位置/格式偏置；
- 主图/主表全由 canonical 脚本重建，无合成曲线；
- **新增**：迁移分解表来自 disjoint OOD；预测器有 ≥16 cell + LOO；
- 正文 ≤7 页（AAAI-27）。

---

## 4. 风险与兜底

| 风险 | 兜底 |
|---|---|
| **资产不可恢复**（最致命） | 环境一好先查；不可恢复就从零重训 Tier 1 的 24 run，40 卡 5–6 天可完；摘要不受影响 |
| **7/28 只有 7 天** | 固定 scope = Tier 0 + Tier 1 + 裁剪 Tier 2；Tier 4 砍，Tier 3 仅当资产可恢复且有余力 |
| docker/训练环境 7/28 前不恢复 | 极端情况：用现有 eval_results 出一篇"limitations 明确、claims 收敛"的分析型短文；但优先争取环境恢复 |
| 探针在 16–20 cell 上不干净 | 预注册；降级为"a useful but imperfect signal"；主结论落在**迁移分解**（更稳）上；摘要措辞已留余地 |
| "GRPO beats SFT" 站不住 | 正名 RFT + 补 1 个 teacher SFT；或直接删该贡献（本方向不依赖它） |
| 现有 eval JSON provenance 多为 invalid/重复 | provenance audit 先做；不够独立 seed 的 domain 老实降为单 seed 或补训 |

---

## 5. 立刻可做的下一步（不需要 docker / 不需要新训练）

1. **明晚 8 点前**：用本文件 §2 的标题+摘要去注册 AAAI-27 摘要。
2. 写 `src/analysis/canonical_v2.py` 骨架（从现有 eval_results JSON 重建主表，不依赖新 run）。
3. 写 provenance audit 脚本 `scripts/audit/provenance.py`，先把现有 run 按"能否证明独立训练"标 valid/invalid（这一步现在就能做，且直接决定 7/28 要重训多少）。
4. 起 MCQ option-permutation + parser 审计脚本（纯推理，环境给到推理就能跑）。
5. 起 disjoint-OOD 迁移评测脚本骨架（先备好 benchmark 划分与 prompt/parser）。

环境恢复后第一件事：**查 `data/ outputs/ logs/` 能否恢复** → 决定走"重出图+补12 run"还是"从零重训24 run"。然后 Tier 0→1→2 推进，每完成一层把结果指针写到本文件 §3 的 checkbox 和 `docs/aaai27/02_RESULT_TRACKER.md`。
