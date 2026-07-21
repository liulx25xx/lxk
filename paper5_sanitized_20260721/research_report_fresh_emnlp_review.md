# Fresh EMNLP Review: One Recipe Does Not Fit All

## Executive Summary

我重新基于当前 PDF `/path/to/workspace/project/emnlp/paper5/paper/main.pdf` 做了一轮独立审稿。论文的核心问题非常重要：数学任务上校准出来的 GRPO/RLVR 配方是否能直接迁移到科学、医学、法律、常识等其他域？作者的主要结论是标准学习率 `5e-7` 几乎无效，而 `2e-5` 能在多域上显著提升，且 `frac_reward_zero_std` 可预测哪些提升能迁移。

我的总体判断是：这篇论文有一个很好的经验发现，也比一般单域 RLVR 论文更有价值；但当前版本仍然过度声称，内部数值和实验叙述不够自洽，若按 EMNLP 主会标准，我会给 `Weak Reject / Borderline`。如果作者能修正结果一致性、补齐每个实验的 seed/CI、澄清 cross-model 和 OOD 设置，并把“genuine knowledge acquisition”改成更谨慎的表述，这篇文章有机会变成 `Weak Accept`。

## Paper Summary

论文研究 SFT 与 GRPO/RLVR 在不同任务域上的后训练效果。主要实验使用 `Qwen2.5-7B-Instruct`，LoRA rank 64、alpha 128，五个域分别是 Math/GSM8K、Science/ARC-C、Medicine/MedQA、Law/LegalBench、Commonsense/ARC-Easy。GRPO 使用 binary reward，`K=8` completions，`β=0.001` KL penalty，并 sweep `5e-7`、`5e-6`、`2e-5`、`1e-4` 等学习率。

主表显示 base 准确率为 Math `84.4`、Science `71.1`、Medicine `59.2`、Law `57.6`、Commonsense `43.8`。标准 GRPO `lr=5e-7` 基本无提升：`85.0/71.3/58.7/54.5/44.4`。调优后的 GRPO 则达到 `92.2/79.1/66.6/58.9/68.8`，相对 base 提升 `+7.8/+8.0/+7.4/+1.3/+25.0`。论文进一步声称：Math 的 SFT scaling 呈倒 U 型，Medicine 的 SFT 基本平坦，Science 单调改善；高学习率 GRPO 的收益在 Medicine 上能 OOD 迁移，在 Math/Commonsense 上更多是 ID 优化；`frac_reward_zero_std` 可作为预训练诊断指标。

## Strengths

第一，问题重要且及时。DeepSeek-R1、DeepSeekMath/GRPO、DAPO 和 Med-RLVR 之后，社区确实需要知道数学上有效的 RLVR 配方能否迁移到其他 NLP/knowledge domains。论文没有提出新算法，而是审视“recipe transferability”，这对 EMNLP/ACL 社区很有现实意义。

第二，多域受控比较的方向是对的。相比单域报告，作者固定 base model、LoRA 设置、解码协议和 reward type，在五个域上比较 SFT、GRPO、SFT→GRPO，并加入 learning-rate sweep、KL ablation、cross-model validation 和 OOD transfer，这个实验框架本身有价值。

第三，主发现有潜在实践价值。`5e-7` 标准配方无效、`2e-5` 显著有效、small-data Law 需要更保守 `5e-6`，这些结论如果可靠，会直接影响从业者如何调 GRPO。尤其 Table 15 中 `β` 增大后 Medicine gain 从 `+7.4` 降到 `+3.2/+1.0`，以及 `K=4` 降到 `+3.7`，说明作者已开始处理“learning rate 是否只是 KL/group size artifact”的问题。

第四，论文比单纯 accuracy table 更努力地解释机制。`frac_reward_zero_std`、reward/KL trajectory、ID/OOD transfer ratio、SFT data perplexity 都是有潜力的诊断指标。特别是将 “RLVR consolidates existing partial knowledge rather than creating new capability” 与 Yue et al. 2025 的 pass@k 视角联系起来，是比早期版本更合理的表述方向。

第五，cross-model validation 如果真实完整，将显著增强论文。Table 16 覆盖 `Qwen2.5-7B-Instruct`、`Qwen3-8B`、`Yi-1.5-9B-Chat`、`Mistral-7B-Instruct`、`DeepSeek-7B-chat`，并在 Medicine/Science 上都显示正收益，这比只在 Qwen 系列验证更有说服力。

## Major Weaknesses

最主要问题是论文内部仍然有不少不一致之处，影响可信度。摘要和引言强调“200+ models”“five model families from four labs”，但方法部分只清楚描述了 Qwen2.5 主实验和 Qwen3 validation；Limitations 中也写“validate key findings on Qwen3-8B”，没有对应 Table 16 的 Yi/Mistral/DeepSeek 训练细节。若 Table 16 是新加入的跨模型实验，方法部分必须完整说明这些模型的 prompt template、LoRA target modules、训练步数、数据过滤、seed 数和评估协议。否则 reviewers 会怀疑这些结果是 appendix 后补而不是同等规范实验。

第二，计算预算与模型数目不自洽。正文说 `200 H200 GPU-hours across 200+ trained models`，Table 13 也写 total `200+`，但可见分类中 SFT `55`、GRPO standard `15`、lr sweep `10`、lr=2e-5 seeds `10`、hybrid `4`、Qwen3 `10`，加起来只有 `104` 个明确模型；cross-model、KL ablation、OOD evaluation 并没有被清楚计入 models。`Failed/exploratory` 没有模型数量。作者需要提供 experiment inventory，而不是只给汇总口径。

第三，表格数值和 caption 有若干冲突。Table 2 中 Math best GRPO 为 `92.2±0.7`，但 Table 14 的 pure GRPO lr=2e-5 only 是 `92.8`；如果一个是 multi-seed avg、一个是 seed42，需要明确。Figure 1 显示 `+7.2/+7.8/+7.9/+24.4`，而 Table 2 的 `Δbest vs base` 是 `+7.8/+8.0/+7.4/+25.0`；如果 Figure 1 是“best lr vs standard lr”而非“best vs base”，caption 需要说明。Table 3 caption 说 best SFT result 是 `59.5`，但 Table 8 中 Medicine `N=100` avg 是 `59.7`，Table 6 又显示 single-seed `N=2000` 为 `61.7`。这些看似小问题，但对经验论文很致命，因为读者会不确定哪些数是最终主结果。

第四，`frac_reward_zero_std` 作为诊断指标的证据很薄。论文声称它“measurable from a single inference pass before training begins”，并在 Table 17 中用 Medicine/Math/Commonsense 三个点得到 `r=-0.99`。但 `n=3` 的相关系数几乎没有统计意义，而且 Table 12 报的是 Medicine 的 `final` frac_zero_std，Table 17 报的是 training onset frac_zero_std，二者口径不同。若要把它作为贡献 3，至少需要在五个域、多个模型和多个 learning rate 上报告该指标，并给出置信区间或 permutation/bootstrap 分析。

第五，OOD transfer 的结论需要大幅收敛。Table 17 只覆盖 Medicine、Math、Commonsense，不包含 Science 和 Law。结果也并非“GRPO gains transfer broadly”：Medicine transfer ratio `1.16×` 很强，但 Math 只有 `0.18×`，Commonsense 只有 `0.03×`，也就是说最大 ID gain `+25.0` 几乎不转移。论文可以说“Medicine shows transferable knowledge consolidation, while Math and Commonsense reveal ID-only optimization”，但摘要里“genuine knowledge acquisition that transfers across five model families”容易被理解为跨域 OOD 都转移，这是过强的。

第六，机制叙述仍有矛盾。正文一方面说 GRPO “cannot teach genuinely new capabilities from scratch”，只能 consolidate base model already partially possesses 的知识；另一方面多处仍说“genuine knowledge acquisition”“acquire new knowledge patterns”。这两个表述不是完全不能兼容，但需要精确定义：到底是新能力、新知识、latent knowledge stabilization，还是 improved greedy extraction of base-model pass@k support？最新 RLVR 文献尤其 `Does Reinforcement Learning Really Incentivize Reasoning Capacity...` 会让审稿人对“new capability acquisition”非常敏感。建议改为“stabilizes and exposes latent knowledge already sampled by the base model”，除非作者做了 pass@k/coverage 证明。

第七，SFT data perplexity 的分析仍像三点故事。Table 4 只有 Math/Science/Medicine 三个点，却声称 perplexity predicts scaling behavior。更重要的是解释内部有张力：低 perplexity 被解释为“within the model distribution”，这可以导致“容易吸收更多数据”，但 Medicine 的结论又是“SFT ineffective regardless of data amount”。如果低 perplexity 意味着模型已经会说，所以 SFT 不提升，那它和“allows more data without degradation”的表述不同。作者需要把 degradation risk 与 improvement potential 分开建模，否则诊断建议会混乱。

第八，SFT vs GRPO 公平性仍需澄清。SFT 数据来自 rejection sampling，失败题会被排除；Table 1 caption 又说 SFT and GRPO use the same source prompts。关键问题是：GRPO 是否训练在原始 2000 prompts 上，而 SFT 训练在 base model 能生成正确答案的子集上？如果是，SFT 与 GRPO 的训练分布和难度不同；如果不是，作者需要明确所有方法是否使用同一个 filtered prompt set。另一个问题是 SFT 接收 CoT demonstrations，GRPO 只接收 binary reward，这是实践上合理，但不应被解释为等信息比较。

第九，MCQ evaluation 细节不足。Appendix C 只说提取 `A/B/C/D` letter answer，没有说明多个字母、无字母、模型先推理中出现选项字母、answer formatting failure、option-order bias 如何处理。本文多个主要结论依赖 MCQ accuracy，尤其 Commonsense `+25.0`，解析规则必须更完整。

第十，cross-model Table 16 的 Science test size 显示 `ARC-Challenge, 5413 test`，但 Table 1 的 Science `Nte` 是 `1172`。官方 ARC-Challenge 的 test/validation split 常见大小与 `1172` 更接近，`5413` 看起来像 ARC Easy/Challenge 合并或其他 split。这个不澄清会直接影响 cross-model Science result 的可比性。

## Minor Weaknesses

引用中有多个 `Anonymous` 条目，例如 EventRL、R1-RE、Delay Plateau Collapse。这在匿名提交中可能可接受，但目前 metadata 太少，容易让审稿人怀疑引用不可核验。若这些是 concurrent submissions 或作者自己的匿名工作，应按 ACL 双盲规范处理；若是公开 arXiv，应提供可验证信息。

Law 域只有 31 个训练 prompt，却和其他 2000-prompt 域并列讨论。Law 的结果更像 small-data case study，而不是完整代表一个 knowledge domain。建议主文中将 Law 明确定位为“small-data legal classification setting”。

Figure 3 caption 说 seed 42 representative of multi-seed trends，但主文又用它支撑 SFT scaling 机制。若有 multi-seed appendix，应尽量把主图换成 multi-seed mean + error bars。否则“representative”会被看成 cherry-picking 风险。

Table 15 的解释中“removing KL entirely at standard lr makes performance worse, confirming KL stabilizes when lr is too low to provide directional gradient signal”这个因果解释有些跳跃。`β=0` at `5e-7` 只有一个 cell，不能排除 seed/noise 或 training length 问题。

## Relation to Recent Literature

相对 DeepSeekMath 和 DeepSeek-R1，本文的增量不是提出 GRPO，而是研究 math-calibrated recipe 的跨域迁移性。相对 DAPO，本文更偏经验诊断而非大规模 RL 系统/算法改进。相对 Med-RLVR，本文把医学放入统一多域比较，并展示标准 lr 与高 lr 的巨大差异，这一点有价值。相对 `SFT Memorizes, RL Generalizes`，本文提出了更具体的 NLP benchmark/domain recipe calibration 问题。相对 `Does Reinforcement Learning Really Incentivize Reasoning Capacity...`，本文必须更谨慎：当前结果更支持“high-lr GRPO improves greedy accuracy by consolidating latent partial knowledge”，而不是“RLVR creates new reasoning capability”。

中文技术社区的 2025-2026 微信检索也显示 RLVR/GRPO/DAPO/SFT 后训练是高热方向，但这些文章多是技术解读和实践讨论，不能作为主要学术证据。它们能说明本文选题 timely，但不能提升其科学有效性。

## Questions for Authors

1. Table 16 的 Yi/Mistral/DeepSeek 实验是否使用与 Qwen2.5 完全一致的 LoRA 配置、prompt、training steps、data split 和 evaluation parser？每个模型有几个 seed？
2. `200+ trained models` 的完整列表是什么？Table 13 中哪些 categories 对应哪些 checkpoints？failed/exploratory 是否被计入 claim？
3. Table 2、Figure 1、Table 10、Table 14 中的 Math/Science/Medicine/Commonsense 数字为何不完全一致？请统一 single-seed 与 multi-seed average 的口径。
4. `frac_reward_zero_std` 的 initial 值和 final 值分别是什么？为什么 Table 12 报 final，而 Table 17 用 onset？三点相关 `r=-0.99` 是否有统计意义？
5. SFT rejection sampling 排除失败题后，GRPO 训练集是否也做同样过滤？
6. Science 的 `Nte=1172` 与 Table 16 的 `5413 test` 分别对应哪个 split？
7. 是否做过 pass@k/coverage 分析来证明 high-lr GRPO 超越了 base model 可采样能力边界？如果没有，建议避免“new/genuine knowledge acquisition”的强说法。
8. Commonsense ID `+25.0` 但 OOD only `+0.7`，这是否说明主要是 benchmark-specific format adaptation，而非通用 commonsense improvement？

## Recommendation

我会给 `Weak Reject / Borderline`。按 1-5 分制约为 `3/5`；按 1-10 分制约为 `5/10`。Confidence `4/5`。

拒稿倾向的主要原因不是 idea 差，而是当前证据和写法不足以支撑强结论。论文有一个可能很有价值的核心：GRPO 的有效性不是“是否 RLVR”这么简单，而是 learning rate、KL、group variance、dataset size、domain uncertainty 共同决定。但当前稿件把这个核心包装成了过强的“universal mode-seeking trap / genuine knowledge acquisition / architecture-agnostic recipe”，而实验报告还没有足够严谨。

如果作者能做以下修改，我会倾向 `Weak Accept`：第一，统一所有主表、图和 appendix 数字；第二，给出完整 experiment inventory 和每个 result cell 的 seed/CI；第三，系统报告 `frac_reward_zero_std` 在所有域和模型上的 initial/final 值；第四，澄清 cross-model 训练细节；第五，收敛机制表述，从“创造新知识”改为“consolidating latent partial knowledge”；第六，把 OOD 结论写成 domain-specific 而不是 broad transfer。

## References

1. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
2. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
3. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
4. [Med-RLVR: Emerging Medical Reasoning from a 3B Base Model via Reinforcement Learning](https://arxiv.org/abs/2502.19655)
5. [SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training](https://arxiv.org/abs/2501.17161)
6. [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)
7. [RL Fine-Tuning Heals OOD Forgetting in SFT](https://arxiv.org/abs/2509.12235)
8. [Sogou WeChat search: RLVR GRPO SFT 后训练](https://weixin.sogou.com/weixin?type=2&query=RLVR%20GRPO%20SFT%20%E5%90%8E%E8%AE%AD%E7%BB%83)
9. [Sogou WeChat search: DeepSeek R1 GRPO DAPO 后训练](https://weixin.sogou.com/weixin?type=2&query=DeepSeek%20R1%20GRPO%20DAPO%20%E5%90%8E%E8%AE%AD%E7%BB%83)
