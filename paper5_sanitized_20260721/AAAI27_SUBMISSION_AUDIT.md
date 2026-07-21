# AAAI-27 投稿审计与补实验路线

审计日期：2026-07-21  
审计对象：`paper/main.tex`、`paper/main.pdf`、`paper/figures_v3/`、现有 evaluation JSON 与训练/评测脚本。

## 1. 结论先行

**按当前 PDF 直接投稿：建议不要投。** 如果按审稿尺度判断，当前更接近 Reject（约 2.5/10，置信度高），主要原因不是“结果不够大”，而是以下三类证据链问题：

1. 主文多张关键图不是由当前打包的原始结果可复现地产生，且若干数字与 JSON 不一致；
2. “best learning rate / best data size”是在 test 上挑选，headline result 存在 test-set model selection；
3. 论文把经验现象写成了机制结论，例如“reverse-KL”“new modes”“external signal is necessary”，目前实验不能排除更简单的解释。

好消息是，核心经验信号并没有消失：在 Math、Science、Medicine、Commonsense 上，较高学习率相对保守 GRPO 设置普遍更好。把故事收缩为“跨领域 GRPO 的有效更新幅度需要校准”，并完成下面 P0 实验后，论文才有机会进入 Borderline / Weak Accept 区间。

## 2. 当前最严重的问题

### 2.1 主结果与已保存 JSON 不一致

按所有已保存 runs 重新聚合后：

| Domain | Base | SFT mean | Conservative GRPO mean | Reported high-LR GRPO mean | 备注 |
|---|---:|---:|---:|---:|---|
| Math | 84.38 | 87.83 (4 seeds) | 85.03 (2) | 92.02 (4) | 强信号 |
| Science | 71.11 | 73.64 (4) | 71.46 (2) | 79.10 (4) | 强信号，但应拆 benchmark |
| Medicine | 59.15 | 59.66 (4) | 58.72 (2) | 64.38 (4) | 正文 66.6 只用了前两个较高 seed |
| Law | 57.59 | 58.82 (3) | 54.56 (1) | 58.57 (4) | high-LR 均值低于 SFT |
| Commonsense | 43.78 | 46.85 (4) | 44.42 (1) | 59.82 (3) | 正文 68.8 无法由现有 JSON 复现 |

因此，摘要中的“+7–25 points”“consistently surpasses SFT”以及 Law/Commonsense 的若干数字必须重写。不能只改均值：应先确认 seed provenance、dev selection 和 benchmark aggregation，再形成最终表。

### 2.2 四张关键图的数据来源有问题

- Figure 1：Law 与 Commonsense 的部分数据行/标签交叉，Medicine SFT 数值也与正文不一致。
- Figure 2：训练曲线由手工数组生成，代码注释明确写了模拟 multi-seed variance；终点 reward、训练步数与正文/附录冲突。
- Figure 3：SFT scaling 中多个数据点是插值或合成值，不是实验结果。
- Figure 4：learning-rate/KL 曲线由手工数组生成，当前包中没有对应原始日志。

处理原则：**论文图里只允许出现可追溯到 run-level artifact 的点。** Figure 2/4 若找不回原始训练日志，应删除，而不是继续美化；Figure 3 若不补齐真实 runs，也应删除。

### 2.3 test 上选最优配置

当前正文把多个 learning rate 和 data size 在 test 上扫完，再报告 best LR / best N。这会把调参噪声算进最终提升，尤其 Commonsense 的 seed 方差很大。

必须改为：

1. 为每个 domain 固定 dev split 或使用官方 validation；
2. 所有 LR、checkpoint、data size 只在 dev 上选择；
3. 选择规则在看 test 之前固定，例如最高 dev accuracy，平局取更低 LR；
4. test 只评一次最终配置；
5. 旧的 test sweep 可以留作“descriptive sensitivity analysis”，不能作为无偏主结果。

### 2.4 benchmark aggregation 掩盖了真实行为

- Commonsense 的 12,418 题实际上是 ARC-Easy 2,376 + HellaSwag 10,042，正文却只写 ARC-E；43.8 是按题数加权的 pooled accuracy，几乎被 HellaSwag 主导。
- Science 是 ARC-Challenge + ScienceQA，需分别报告以及 macro average。
- Law 的 10,403 题被一个 9,306 题的子任务主导。按 task macro average，high-LR GRPO 的优势明显弱化。
- 所有组合 benchmark 应同时给 per-benchmark、macro average；micro/pooled 只能作为补充。

### 2.5 baseline 命名与公平性

非数学领域的所谓 SFT，并不是独立 teacher demonstrations：它从同一个 base model 采样 K=8，再用 gold/verifier 过滤正确答案并选择轨迹。这更接近 rejection-sampling fine-tuning（RFT）或 filtered self-training。OPD 的数据构造与它高度相似，因此当前实验不能推出“without external signal fails”或“external signal is necessary”。

建议：

- 把现有非数学 SFT 更名为 RFT / filtered self-training；
- OPD 若保留，必须匹配 prompts、采样数 K、保留样本数、tokens、epochs 和 verifier 使用，否则从主贡献中删除；
- 不再写“matched compute”，除非实际报告 generated tokens、optimizer updates、GPU-hours 和 wall-clock。相同 epoch 数不等于相同计算量。

### 2.6 学习率基准引用错误

正文把 5e-7 称为 DeepSeekMath 使用的默认 GRPO 学习率，并据此写“40x”。DeepSeekMath 原文报告的是 policy learning rate 1e-6。因此二选一：

- 推荐：补 1e-6，主标题改成 20x；
- 或把 5e-7 明确称为“our conservative baseline”，停止归因给 DeepSeekMath，也不再称为社区标准默认值。

### 2.7 机制结论过强

现有结果只能支持“保守设置产生的有效 policy update 不足”这一经验解释，不能证明：

- GRPO 在低 LR 时等同于 pure reverse-KL minimization；
- 提高 LR 改变了 divergence direction；
- 高 LR 捕获了 new modes / 获得了 new knowledge；
- `frac_reward_zero_std` 是一个已验证 predictor。

如果找得到真实日志，可报告 reward、policy-reference KL、update norm、gradient norm、clip fraction、entropy、zero-std fraction 随 step 的变化，并把“机制”改成“consistent with the under-update hypothesis”。若没有日志，就把机制降级为 hypothesis，并删除人工轨迹图。

## 3. 投稿前必须补的 P0 实验

### P0-A：固定数据谱系与复现主结果

目标：确保每个表格单元都能追溯到 `config + checkpoint + seed + raw predictions + aggregation script`。

- 核对目录名中的 seed 与训练配置真实 seed；当前 18 个 run 的 JSON metadata 与目录 seed 不一致。
- 找回训练 config/log/checkpoint；如果目录仅是重复评测而非独立训练 seed，不能把它们作为多 seed 结果。
- 用一个 canonical 脚本生成所有均值、标准差、表格和图，不允许脚本内手填论文数字。
- 做 train/dev/test ID exact match 和 normalized text/n-gram 去重检查。

### P0-B：dev-only 配置选择

建议对四个核心 domain 使用官方 validation 或固定 stratified dev。Law 先移至附录作为边界案例。

最终 protocol：

| 阶段 | 数据 | 用途 |
|---|---|---|
| Training | 固定 2,000 prompts 或清楚报告实际可用量 | 训练各方法 |
| Development | 与 test 严格不重叠 | 选 LR、checkpoint、early stopping |
| Test | 官方 test/held-out | 仅评估 dev 选出的配置 |

### P0-C：补正确的 conservative baseline

最低可接受主网格：

| Domain | LR | Seeds | 说明 |
|---|---|---:|---|
| Math | 1e-6, 2e-5 | 42, 123, 456 | 2e-5 可复用真实独立 runs |
| Science | 1e-6, 2e-5 | 42, 123, 456 | ARC-C / ScienceQA 分开报告 |
| Medicine | 1e-6, 2e-5 | 42, 123, 456 | 高方差，必须至少 3 seeds |
| Commonsense | 1e-6, 2e-5 | 42, 123, 456 | ARC-E / HellaSwag 分开报告 |

如果现有 high-LR runs 经核验确为独立训练 seed，新增量主要是 4 domains × 3 seeds = **12 个 1e-6 runs**。5e-7 可作为更保守的补充点放附录。若 2e-5 并非所有领域的 dev-selected LR，不应强制都叫 best；应写成 fixed high-LR intervention，或先用 dev sweep 选择。

### P0-D：统计检验与不确定性

- 报告每个独立训练 seed 的结果，不只报告 mean±SD。
- 主比较使用 hierarchical bootstrap：先对训练 seeds 重采样，再对 test items 重采样。
- 对同一 test set 的两个模型给 paired bootstrap 或 McNemar 检验。
- 对多个 domain 做 Holm correction，或者明确主检验只有预注册的 overall macro score。
- Commonsense/Science 以 benchmark macro 为主，pooled micro 为辅。

### P0-E：MCQ 评测可信度审计

这是当前最值得补、成本又不高的一组实验：

1. 对每个 domain 随机抽 200 条输出，人工核验 parser、格式失败与判分一致性；
2. 对 MCQ 选项做固定或随机 permutation，重新映射 gold label；至少做 3 个 permutation seeds；
3. 同时报告 original-order 与 permutation-robust accuracy；
4. 报告 invalid output rate、answer-position distribution。

这能排除高学习率只是强化了答案字母、位置偏置或输出格式的解释。

### P0-F：重做主图与主表

建议主文只保留三张证据密度高的图/表：

1. **Main result**：四领域 × 三 seeds，Base/RFT/GRPO-1e-6/GRPO-dev-selected；展示 run points + mean/CI。
2. **LR sensitivity**：真实 dev sweep，横轴 log LR，分别展示 dev 与最终 test-selected 点；不能从 test 选峰值。
3. **Diagnostics**：只有找回真实日志才画 reward/KL/update norm；否则用 robustness/permutation panel 替代。

## 4. 强烈建议补的 P1 实验

### P1-A：更强且公平的监督基线

当前 RFT baseline 太弱，无法支持“GRPO 优于 SFT”的宽泛结论。最低方案：

- 在 Science 与 Medicine 上补一个真正的 gold/teacher SFT；
- 与 GRPO 使用相同训练 prompts；
- 报告每种方法看到的 prompts、retained examples、response tokens、optimizer updates 和 GPU-hours；
- 同时保留 RFT，形成 Base / gold-or-teacher SFT / RFT / GRPO 四方比较。

如果算力不够，则删除“GRPO beats SFT”的贡献，把结论限定成“GRPO beats matched filtered self-training in selected domains”。

### P1-B：跨模型验证要验证 LR sensitivity，而不是只跑 high LR

现有非 Qwen 模型多数只有 Base 与 high-LR，能证明 high-LR 有时有效，不能证明 LR sensitivity 跨模型成立。

最小跨模型矩阵：选择 Medicine（现有方差和退化最明显），在 Qwen3-8B 与一个非 Qwen 模型上比较 1e-6 vs 2e-5，各 3 training seeds。若预算有限，只保留一个非 Qwen 家族，并把措辞改成“two model families”；不要写“五个 architecturally distinct families”。

### P1-C：真正的 OOD/transfer

- Science 不能把 ARC-C 同时当 ID 组成部分和 OOD；
- OOD benchmark 必须与训练集及主测试集 disjoint，并统一 prompt/parser；
- 每个 OOD 表都要给 target base，不能只给 transfer 后准确率；
- 结论写成“transfer varies by source/target pair”，不要写“genuine capability acquisition”。

## 5. 可以删掉或降级的 P2 内容

- SFT perplexity predictor：当前只有 3 点且与 domain/task/format 淆杂，建议直接删主文。
- `frac_reward_zero_std` predictor：当前约 3 个 domain 点，不能做可靠相关性；除非扩成至少 12–20 个 domain×model×LR observations，并做 leave-one-domain-out，否则移附录并称 exploratory diagnostic。
- OPD：若无法与 RFT 做严格匹配，删除这条 contribution。
- Law：训练样本太少、macro 结果不支持统一结论，作为 boundary case 放附录反而更可信。
- 大篇幅 reverse-KL 理论：收缩为一段可证伪的 under-update hypothesis。

## 6. 建议重构成一篇更聚焦的论文

推荐核心问题：

> Conservative GRPO settings that work in mathematical reasoning can under-update the policy in other domains; learning rate or, more generally, effective update magnitude should be selected on held-out development data.

建议贡献只留三条：

1. 一个 dev-selected、multi-seed、按 benchmark 分解的跨领域 controlled study；
2. 对 conservative vs calibrated update 的稳健比较，包括 MCQ permutation 与 parser audit；
3. 一个以真实 update/KL 日志支持的 under-update diagnostic；如果没有日志，则改为经验观察，不作为主贡献。

可用标题：

- *Calibrating GRPO Beyond Mathematics: A Controlled Multi-Domain Study*
- *When Conservative GRPO Under-Updates: Learning-Rate Calibration Across Domains*

不建议继续使用 *One Hyperparameter to Rule Them All*：Law 是反例，而且最终真正需要校准的可能不是 LR 单一变量，而是 LR、batch、epochs、LoRA rank 共同决定的有效更新幅度。

## 7. AAAI-27 篇幅与结构建议

当前 PDF 使用 ACL review template、A4、共 15 页；主文约 9 页后才进入参考文献/附录。AAAI-27 主赛道要求正文最多 7 页，参考文献可延伸至第 9 页，补充材料可另交且审稿人不保证阅读。因此必须把关键证据放入 7 页正文。

建议页面预算：

| 内容 | 页数 |
|---|---:|
| Abstract + Introduction | 1.0 |
| Setup / hypotheses / dev-selection protocol | 1.0 |
| Main multi-domain results | 1.5 |
| LR sensitivity + diagnostics/robustness | 1.5 |
| Analysis / limitations | 0.8 |
| Related work + conclusion | 1.2 |

删除或移附录：完整模型清单、SFT scaling、PPL predictor、OPD、详细 cross-domain matrix、Law 细节和长理论推导。

## 8. 推荐执行顺序

1. **今天先冻结 claim**：去掉无法复现的 68.8、66.6、synthetic curves 和 DeepSeekMath 5e-7 归因。
2. **当天完成 provenance audit**：确认所有 seeds 是否为独立训练、找回 raw logs/configs。
3. **建立 dev split 和统一评测脚本**：先不要继续在 test 上扫参。
4. **运行 12 个 1e-6 core runs**，同时保存逐步训练诊断。
5. **跑 MCQ permutation/parser audit**，成本通常远低于重新训练。
6. **补公平 RFT/teacher-SFT 或收缩 baseline claim**。
7. **按 canonical script 重出表和图**，再重写 abstract/introduction/experiments。
8. **最后切换 AAAI-27 模板并压到 7 页正文**。

## 9. Go / No-Go 标准

满足以下条件再投：

- 目录 seed 与真实训练 seed 完全一致；
- 所有正文数字可由单一脚本从 raw artifacts 重建；
- LR/checkpoint 只在 dev 上选；
- conservative 1e-6 与 calibrated LR 至少各 3 个训练 seeds；
- 组合 benchmark 按子任务报告，macro 结果仍支持主要结论；
- permutation/parser audit 没有暴露严重格式或位置偏置；
- Figure 2/4 来自真实日志，或已删除；
- 摘要不再声称所有 domain 都超过 SFT，也不再声称已证明 reverse-KL/new-mode 机制；
- 主体满足 AAAI 7 页限制。

如果无法在截止前完成 dev-only selection 与结果 provenance，这一轮宁可不投；只补更多 test runs 或把图画得更漂亮，不能解决当前最核心的审稿风险。
