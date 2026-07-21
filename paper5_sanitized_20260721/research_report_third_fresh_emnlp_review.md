# Third Fresh EMNLP Review: One Recipe Does Not Fit All

## Executive Summary

本轮审稿重新从当前 PDF `/path/to/workspace/project/emnlp/paper5/paper/main.pdf` 抽取文本并独立评估，未读取此前审稿报告。论文研究 SFT 与 GRPO/RLVR 在五个域上的后训练效果，主张标准数学配方 `lr=5e-7` 在非数学域几乎无效，而将学习率提高到 `2e-5` 可在多个 MCQ 域获得 `+7--25%` 的提升。整体上，问题重要、实验规模有吸引力、主发现有实践价值；但当前稿件仍存在较明显的经验论文硬伤，包括模型数/计算预算口径不清、跨模型实验与主文方法不一致、部分表格数值冲突、`frac_reward_zero_std` 证据不足、OOD 结论只支持域特异性而非广泛迁移。我会给 `Weak Reject / Borderline`，约 `3/5` 或 `5/10`，confidence `4/5`。

## Paper Summary

论文题为 “One Recipe Does Not Fit All: How Domain Characteristics Shape Post-Training Effectiveness”。作者以 `Qwen2.5-7B-Instruct` 为主模型，LoRA rank `64`、alpha `128`，在 Math/GSM8K、Science/ARC-C、Medicine/MedQA、Law/LegalBench、Commonsense/ARC-Easy 五个域上比较 SFT、GRPO 和 SFT→GRPO。GRPO 使用 binary correctness reward，`K=8` completions，`β=0.001` KL penalty，并 sweep `5e-7`、`5e-6`、`2e-5`、`1e-4` 等学习率。

主表显示 base 准确率为 Math `84.4`、Science `71.1`、Medicine `59.2`、Law `57.6`、Commonsense `43.8`。标准 GRPO `lr=5e-7` 为 `85.0/71.3/58.7/54.5/44.4`，基本无提升。调优 GRPO 为 `92.2/79.1/66.6/58.9/68.8`，相对 base 增益 `+7.8/+8.0/+7.4/+1.3/+25.0`。论文进一步声称：SFT scaling 具有域依赖性，Math 呈倒 U 型，Medicine 基本 flat，Science 单调提升；`frac_reward_zero_std` 能预测 GRPO 学到的是 latent knowledge consolidation 还是 format optimization；高学习率 GRPO 的模式在 Qwen3、Mistral、Yi、DeepSeek 等模型上也成立。

## Strengths

首先，论文问题很及时。DeepSeek-R1、DeepSeekMath、DAPO、Med-RLVR 之后，很多工作把数学/代码域的 RLVR recipe 迁移到其他任务，但真正受控比较“同一模型、同一训练框架、多个域、多个学习率”的工作不多。本文把问题从“RLVR 是否有效”转成“recipe 是否按域校准”，这是有价值的。

其次，主发现具有实践意义。若结果可靠，`5e-7` 作为默认 GRPO 学习率在非数学域明显不足，而 `2e-5` 在大多数 2k-prompt MCQ 域更有效；Law 这种 31-prompt 小数据域则需要更保守的 `5e-6`。这种结论比单纯报告一个 benchmark gain 更有操作性。

第三，论文试图提供机制诊断，而不是只给 accuracy table。reward trajectory、KL trajectory、KL penalty/group-size ablation、SFT data perplexity、`frac_reward_zero_std`、ID/OOD transfer ratio 都是正确方向。尤其是把 GRPO 表述为 consolidating latent partial knowledge，而不是简单声称“创造新能力”，比许多 RLVR 论文更谨慎。

第四，Table 15 的 KL/group-size ablation 增强了 learning-rate 结论：Medicine 上 `lr=2e-5, β=0.001` 得到 `66.6`，`β=0.01` 降到 `62.4`，`β=0.1` 降到 `60.2`，`K=4` 也降到 `62.9`。这说明作者开始处理 `lr` 与 `β/K` 的交互，而不是完全忽略其他 GRPO 超参数。

第五，Table 16 的跨模型验证如果完整可靠，将显著提升论文贡献。Medicine 上 Qwen2.5、Qwen3、Yi、Mistral、DeepSeek 都有 `+6` 到 `+12` 点提升；Science 上 Qwen2.5、Qwen3、Mistral 也有 `+5` 到 `+9` 点提升。这有助于说明现象不只是 Qwen2.5 的偶然。

## Major Weaknesses

最严重的问题是主文与附录中的实验范围不完全一致。正文方法部分主要说所有方法 applied to `Qwen2.5-7B-Instruct`，并“also validate on Qwen3-8B”；Limitations 也只说 validate key findings on Qwen3-8B。但摘要、贡献和 Table 16 又声称五个 model families、四个 research labs，包括 Mistral、Yi、DeepSeek。若这些是正式贡献，方法部分必须完整说明这些模型的训练配置、LoRA target modules、prompt template、tokenizer/parser、训练步数、seed 数、数据 split 和失败率。目前给人的感觉是跨模型结果在附录里突然出现，主文没有充分铺垫。

第二，`200+ models` 和 `200 H200 GPU-hours` 的口径不清。Table 13 显式列出的模型数为 SFT `55`、GRPO standard `15`、GRPO lr sweep `10`、GRPO lr=2e-5 seeds `10`、hybrid `4`、Qwen3 `10`，合计约 `104`，而不是 `200+`。`Failed/exploratory` 只有 GPU-hours 没有 models，evaluation 不是 trained models。作者需要给出完整 experiment inventory，否则 `200+` 会显得像 marketing number。

第三，若干关键数值仍不统一。Table 2 中 Math best GRPO 是 `92.2±0.7`，Table 14 中 pure GRPO lr=2e-5 only 是 `92.8`。Figure 1 的 gains 是 `+7.2/+7.8/+7.9/+24.4`，Table 2 的 `Δbest vs base` 是 `+7.8/+8.0/+7.4/+25.0`。Table 3 说 Medicine best SFT 是 `59.5`，但 Table 8 多种子 `N=100` 平均是 `59.7`，Table 6 single-seed `N=2000` 是 `61.7`。这些可能能解释为 single-seed vs multi-seed、best-vs-standard vs best-vs-base，但文中没有统一标明。经验论文里，数字口径不清会严重损害可信度。

第四，`frac_reward_zero_std` 作为核心贡献证据不足。论文称它能从单次 inference pass 预测 transfer quality，并报告三域相关 `r=-0.99`。但 Table 17 只有 Medicine、Math、Commonsense 三个点，`n=3` 的相关几乎没有统计意义。且 Table 12 报 Medicine 的 `final` frac_zero_std，Table 17 报 training onset frac_zero_std，读者容易混淆。若把这个指标列为 contribution 3，需要至少覆盖五个域、多个模型、多个 learning rate，并给出 CI 或简单的留一验证。

第五，OOD 结果不支持广泛迁移的强表述。Table 17 显示 Medicine OOD `+8.6`、transfer ratio `1.16×`，确实很强；但 Math OOD 只有 `+1.4`，transfer ratio `0.18×`；Commonsense OOD 只有 `+0.7`，transfer ratio `0.03×`。因此本文真正发现应是“GRPO 的 ID gain 可能是 format-specific，也可能是 transferable，取决于域和 uncertainty structure”，而不是“high-lr GRPO gains transfer across domains”。当前摘要中的“transfers across five model families”容易与 OOD transfer 混淆，应明确是 cross-model ID improvement，而不是跨 benchmark 迁移。

第六，Commonsense 的 `+25.0` 需要更细分析。作者补充说 gold answer distribution 近均匀，排除 letter-bias shortcut，这很好；但 Table 17 显示 WinoGrande OOD 只提升 `+0.7`。这说明 ARC-Easy 上的大提升很可能是 benchmark/format/task-specific adaptation，而非一般 commonsense 能力增强。主文应把 Commonsense 作为“large ID gain, minimal transfer”的反例来强调诊断指标，而不是把它放在 `+7--25%` 的正向主结果里一笔带过。

第七，SFT perplexity 的解释仍然不充分。Table 4 只有 Math、Science、Medicine 三个点，却声称 perplexity predicts scaling behavior。更重要的是，低 perplexity 被解释为 closer to model distribution，因此可被大量吸收而不 destructive；但 Medicine 的结论是 SFT flat、无法吸收医学知识。这里需要区分两个概念：degradation risk 和 improvement potential。低 perplexity 也许意味着低退化风险，但不一定意味着高收益；现在的表述把二者混在一起。

第八，SFT 与 GRPO 的训练集公平性仍需澄清。SFT demonstrations 通过 base model rejection sampling 构造，8 次都失败的问题会被排除。Table 1 caption 说 SFT and GRPO use same source prompts，但“source prompts”不等于同一个 filtered training set。若 GRPO 使用全部 prompts 而 SFT 使用只含 base-correct samples 的子集，则难度分布不同；若二者都用 filtered set，则 GRPO 训练集也被 base ability 过滤，会影响结论。作者必须明确这一点。

第九，MCQ answer extraction 描述过于简略。Appendix C 只说提取 `A/B/C/D` 并 exact match。需要说明多个选项字母、无选项字母、CoT 中间出现字母、大小写、答案格式失败、答案位置偏置等如何处理。Science、Medicine、Commonsense、Law 大量结论依赖 MCQ/binary parser，这部分不能只用一句话。

第十，Science test size 在当前版本中从早期常见 `1172` 变成 Table 1 与 Table 16 的 `5413`。若这是 ARC-Challenge 的某个合并 split，应说明 split 来源；否则审稿人会质疑与标准 ARC-C test/dev set 不一致。尤其 Table 16 写 `ARC-Challenge, 5413 test`，需要数据集构造细节。

## Minor Weaknesses

论文引用仍有多个 `Anonymous` 条目，如 EventRL、R1-RE、Delay Plateau Collapse。匿名引用可能符合双盲，但 metadata 太少，建议按 ACL policy 明确是否为 concurrent work/self-citation，并提供可核验线索。

Figure 3 caption 写 seed 42 is representative of multi-seed trends，但正文用它支撑 SFT scaling 机制。既然 appendix 有 multi-seed 表，主图最好直接画 multi-seed mean/error bars，而不是 representative seed。

Table 15 对 `β=0` at standard lr 的解释偏强。一个 cell 下降到 `57.0` 不足以证明“KL stabilizes when lr is too low to provide directional gradient signal”，最多说明在该设置下移除 KL 更差。

Law 域只有 31 unique prompts，不应与 2k-prompt 域做平行泛化。建议明确把 Law 定位为 small-data stress test，而非普通 legal domain 结论。

## Relation to Recent Work

与 DeepSeekMath 和 DeepSeek-R1 相比，本文不是算法创新，而是 recipe transferability 的经验研究。与 DAPO 相比，本文没有提出大规模 RL 系统，但更关注学习率/域特性。与 Med-RLVR 相比，本文把医学纳入统一多域框架，并展示 standard lr 与 high lr 的差异。与 `SFT Memorizes, RL Generalizes` 相比，本文更接近实际 NLP benchmark 的 recipe calibration。与 `Does Reinforcement Learning Really Incentivize Reasoning Capacity...` 相比，本文需要保持谨慎：当前结果更支持 “high-lr GRPO improves greedy accuracy by consolidating latent partial knowledge”，而不是 “RLVR creates new reasoning capability”。

微信公众号检索显示 2025-2026 年中文技术社区对 `RLVR/GRPO/SFT 后训练` 和 `DeepSeek R1/GRPO/DAPO` 持续关注，这说明选题很热，但这些文章主要是工程解读，不能作为主证据。

## Questions for Authors

1. `200+ trained models` 的完整列表是什么？Table 13 的 `200+` 如何从各类模型数相加得到？
2. Mistral/Yi/DeepSeek 的 cross-model experiments 是否使用完全相同的 LoRA、prompt、parser、training steps、data split 和 seed 数？
3. 为什么正文方法和 limitations 主要只说 Qwen3 validation，而摘要和 Table 16 声称五个 model families？
4. Table 2、Figure 1、Table 10、Table 14 中 Math/Science/Medicine/Commonsense 数字口径如何统一？哪些是 single seed，哪些是 multi-seed average？
5. `frac_reward_zero_std` 的 initial 与 final 定义分别是什么？为什么只用三个 OOD 点就声称 `r=-0.99`？
6. SFT rejection sampling 排除失败样本后，GRPO 训练是否也使用相同过滤后的 prompts？
7. Science 的 `5413` test examples 来自哪个 ARC split？是否与标准 ARC-C dev/test 一致？
8. 是否做过 pass@k/coverage 分析来证明 high-lr GRPO 超越了 base model 可采样能力边界？
9. Commonsense ID `+25.0` 但 WinoGrande OOD `+0.7`，是否说明其主要是 ARC-Easy-specific adaptation？

## Recommendation

我的建议是 `Weak Reject / Borderline`。如果用 1-5 分制，我会给 `3/5`；如果用 1-10 分制，我会给 `5/10`。Confidence `4/5`。

理由是：论文的中心问题和经验发现有价值，尤其 learning rate calibration 与 latent knowledge consolidation 的框架值得进一步发展；但当前版本的证据组织和结果口径仍不足以支撑摘要与贡献中的强表述。主要问题不是缺一个小实验，而是需要系统清理实验账本、统一所有结果数字、补充 cross-model 和 diagnostic 指标的实验细节，并把 OOD/knowledge acquisition 的结论写得更精确。

如果作者在 rebuttal 或 revision 中能提供完整 experiment inventory、每个 cell 的 seed/CI、统一数值、澄清 cross-model 方法、扩展 `frac_reward_zero_std` 证据，并把“genuine acquisition”改成更可证的“latent knowledge consolidation”，我会倾向升到 `Weak Accept`。

## References

1. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
2. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
3. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
4. [Med-RLVR: Emerging Medical Reasoning from a 3B Base Model via Reinforcement Learning](https://arxiv.org/abs/2502.19655)
5. [SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training](https://arxiv.org/abs/2501.17161)
6. [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)
7. [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond](https://llm-stats.com/blog/research/post-training-techniques-2026)
8. [Sogou WeChat search: RLVR GRPO SFT 后训练](https://weixin.sogou.com/weixin?type=2&query=RLVR%20GRPO%20SFT%20%E5%90%8E%E8%AE%AD%E7%BB%83)
9. [Sogou WeChat search: DeepSeek R1 GRPO DAPO 后训练](https://weixin.sogou.com/weixin?type=2&query=DeepSeek%20R1%20GRPO%20DAPO%20%E5%90%8E%E8%AE%AD%E7%BB%83)
