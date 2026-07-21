# EMNLP 2026 Review Report: Position Shortcut in RL-Trained LLM Judges

## Executive Summary

本文研究 RL 训练 LLM-as-a-Judge 时由固定答案位置造成的 shortcut learning：在 RewardBench-style 数据构造中，如果 preferred response 总是被放在 A，GRPO/SFT/DPO 训练会把“选 A”学成高回报策略，从而产生表面 accuracy 提升和 position consistency 下降。论文的问题意识很强，主题切中 2025–2026 年 LLM judge、reward hacking、post-training reliability 的重要交叉点；核心实验也展示了一个值得社区重视的失败模式。不过，目前稿件存在几类会影响 EMNLP 接收判断的问题：核心数据构造前提需要更可复核地证明；对 JudgeLRM 的“direct contradiction”表述过强；balanced data “preserves accuracy gains”的结论在 GRPO 结果上被夸大；多目标 reward 失败的结论只覆盖少数 proxy rewards 和单组权重；若干 2026 引用元数据不完整或疑似不应作为正式学术引用。我的总体建议是 Borderline / Weak Accept，倾向要求大修后接收；如果不能补充数据构造证明、代码和关键表述修正，则会转为 Weak Reject。

## Background and Paper Summary

论文题为 “Position Shortcut: How Reinforcement Learning Teaches LLM Judges to Cheat”。其中心论点是，标准 pairwise preference 数据常以 `chosen`/`rejected` 字段组织；如果作者的 judge prompt 始终把 `chosen` 映射到 Response A，并用 gold label A 做 accuracy reward，那么 RL 训练会强化位置特征而非真正的质量比较。论文报告 Qwen2.5-7B-Instruct 上 baseline accuracy 80.2、position consistency 83.3；unbalanced GRPO 后 accuracy 约 94–95，但 consistency 降到约 60；SFT 更极端，达到 100 accuracy、0 consistency。作者进一步比较 DPO、GRPO、多目标 reward、learning-rate sweep、balanced training，并在 Qwen3-8B 上做交叉验证。提出的修复很简单：训练时让 preferred response 在 A/B 两个位置上均衡出现，使 position 不再预测 reward。

这个方向很有价值。LLM-as-a-Judge 已经广泛用于模型排行榜、RLHF/RLAIF、数据筛选和自动评审；如果训练出的 judge 依赖候选顺序，会污染下游对齐和评测。近期工作已经系统研究 inference-time position bias、judge consistency 和 reward hacking，但本文试图把 position bias 追溯到 RL judge 训练时的 data-level confound，这是比较有新意的贡献。

## Strengths

论文的最大优点是问题重要且实验现象直观。Position swap consistency 是一个简单但强诊断指标：如果原顺序选 A、交换后仍选 A，那么模型显然在用位置而非同一个实际答案。表 1 中 SFT 的 100/0 结果尤其有冲击力，能有效说明固定位置监督会产生灾难性 shortcut。GRPO 和 DPO 的中间表现也符合“优化越强、regularization 越弱，shortcut 越强”的直觉。

第二，实验矩阵覆盖了多个训练方法、reward 组件、learning rate、training dynamics 和两个 Qwen-family 模型。虽然模型族仍偏窄，但相对于一般短论文，这个实验量是有说服力的。学习率 sweep 展示 accuracy 与 consistency 的单调 tradeoff，training dynamics 展示 early accuracy gains 后 consistency collapse，这些结果共同支持“优化压力激活 shortcut”的叙述。

第三，balanced training 作为修复方案非常实用。它不需要新模型、不需要额外人工标注，也比 inference-time swap/filtering 更便宜。若作者能发布 preprocessing 脚本并证明适用于不同 judge datasets，这一建议很可能成为训练 LLM judge 的标准 hygiene check。

第四，论文对 proxy rewards 的失败做了有意义的提醒。Decisiveness 和 calibration 这类 reward 看似能提升 reliability，但如果它们不直接约束 position invariance，在固定 A 标签的数据上可能仍与 shortcut 兼容。这一点对 2026 年大量 GRPO/RLVR-style 后训练工作都有警示意义。

## Major Weaknesses

最关键的问题是核心前提需要更严谨地表述和复核。RewardBench 官方数据卡确认数据字段是 `prompt`, `chosen`, `rejected`，并未本身定义 A/B 顺序；官方评测通常是比较 reward model 对 `(prompt, chosen)` 与 `(prompt, rejected)` 的分数，而不是让 LLM judge 在 A/B prompt 中选择。因此，“RewardBench places the chosen response in position A for 100% of instances” 严格说应改成：“in our RewardBench-to-judge-prompt conversion, chosen is always rendered as Response A”。如果作者使用的是某个公开 JudgeLRM/RewardBench judge training recipe，该脚本必须被引用并给出 commit/hash；否则论文容易把下游 prompt construction 的 confound 误归因于 RewardBench 数据集本身。这不是小措辞问题，因为它关系到论文标题级主张。

第二，对 JudgeLRM 的对比需要显著收敛。全文核查显示 JudgeLRM 确实使用 GRPO，并在 JudgeLM validation split 上报告了 position consistency、bias toward 1st、bias toward 2nd、delta bias 等指标；JudgeLRM-7B consistency 约 84.50，JudgeLRM-8B 约 89.55。被审论文说“directly contradicts JudgeLRM”过强，因为两者训练/评估数据不同：JudgeLRM 主要用 JudgeLM/PandaLM，而本文用 RewardBench-style 数据构造。更准确的说法应是：本文揭示了在 RewardBench-to-A/B 转换且 gold position 固定时，JudgeLRM-style GRPO recipe 会出现 JudgeLRM 原文评估未隔离的 confound。除非作者复现 JudgeLRM 官方训练数据、代码和评估协议，否则不应声称直接推翻 JudgeLRM。

第三，balanced training “preserves accuracy gains”的结论被夸大。表 2 中 balanced SFT 达到 91.3 accuracy、87.1 consistency，确实很强；但 balanced GRPO Acc-only 只有 82.6 accuracy，比 baseline 80.2 仅高 2.4pp，远低于 unbalanced GRPO 的约 94–95。GRPO Full Balanced 为 84.6，也只是中等提升。作者可以说 balanced training restores consistency and retains modest genuine gains for GRPO, while SFT benefits more under balanced supervision；但不能笼统说 accuracy gains are preserved。论文中把 unbalanced 94 与 balanced 85 的差值解释为 shortcut accuracy 也需要更谨慎，因为 balanced/unbalanced 改变的不仅是 position-label correlation，也改变了训练分布、样本重复/增强、optimization path 和 possibly label entropy。

第四，多目标 reward 的失败结论还不足以支持一般性命题。论文只测试 decisiveness、calibration 和一组 composite weights。Decisiveness 从 60.4 consistency 提到 66.3，虽然不足以恢复 baseline，但不是完全无效；作者没有测试更大一致性权重、explicit paired invariance reward、contrastive swap reward，或 FairJudge/J1-style training。当前证据只能支持“这些 proxy rewards 在这些权重下不能修复 position confound”，不能支持“reward formulation cannot overcome a data-level confound”。论文自己也承认 true paired invariance reward 可能可行，因此主文应避免把 proxy reward 的失败泛化为 reward-side deconfounding 的失败。

第五，统计和复现细节不足。主文说多种子 3–4 个，accuracy std <1pp、consistency std 7–9pp，但表中大多数结果没有均值±标准差；balanced 条件、Qwen3 条件、学习率 sweep 是否多种子不清楚。Consistency 的 seed variance 达 7–9pp 时，若某些 reward 改进只有 2–6pp，需要置信区间或显著性检验。RewardBench filtered 为 2,985 条，但论文 split 为 2,089 train + 449 test，总数 2,538；剩余样本用于 validation 还是被过滤需要说明。Pred-A rate 多处写成 `~55`, `~94`, `~99`，应给精确数值与解析规则。输出 verdict 和 confidence 的 parsing、invalid output 处理、tie 情况如何计入 consistency，也必须明确。

第六，泛化性仍有限。两个模型都是 Qwen-family、7–8B、Instruct 版本；主数据是 RewardBench-style pairwise prompt。为了支撑“published accuracy numbers for RL-trained judges may substantially overstate genuine judgment capability”，至少需要一个非 Qwen 模型、一个非 RewardBench 数据集，最好还包括 JudgeLM/PandaLM/MT-Bench 或 Arena-Hard-style 数据。近期 systematic position bias work 也强调 answer quality gap、task type、judge identity 和 list-wise comparison 会强烈影响 position consistency；本文目前没有控制 answer quality gap，也没有 list-wise 设置。

## Citation and Related Work Concerns

相关工作覆盖了 LLM-as-a-Judge position bias、RLHF、shortcut learning 和 reward hacking，但还需要更新和清理。应明确讨论 Shi et al. “Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge”，该文系统定义 repetition stability、position consistency、preference fairness，并指出 answer quality gap 是 position consistency 的关键影响因素。本文只报告 position consistency 和粗略 Pred-A rate，缺少 preference fairness、repetition stability 和 answer quality gap 控制。

若干引用需要修正。`TrustJudge` 应补充真实 arXiv 信息和作者，不能只写 Anonymous / ICLR 2026；`su2026gaming` 搜索结果更像博客文章 “Your LLM Learned to Game the Judge: Reward Hacking in LLM-as-Judge”，不是可验证的 arXiv 学术论文，若作为 related work 需要标注 blog/technical report 或删除；`BT-$\sigma$` 对应 arXiv:2602.16610 的标题是 “Who can we trust? LLM-as-a-jury for Comparative Assessment”，bib title 当前不准确；`MJ1` 的真实标题是 “Multimodal Judgment via Grounded Verification”，文中 “flip-consistency reward” 描述需核实。大量 2026 `Anonymous` 条目在公开投稿中非常危险：如果是未公开同期投稿，应避免在匿名主文中以“已有工作”方式依赖；如果已公开，应补全作者和元数据。

微信公众号检索在 2025-01-01 至 2026-05-18 范围内显示中文社区确实在讨论 LLM-as-a-Judge、位置偏见、RubricBench、Verifier 和 JudgeLRM/GRPO 相关主题，但这些结果更适合作为趋势线索，不应用作关键事实来源。核心事实仍应依赖 arXiv、ACL Anthology、OpenReview、GitHub 和数据集官方页面。

## Questions for the Authors

1. 请提供 RewardBench-to-judge-prompt 的完整脚本、commit、split 文件和统计，证明在训练/测试 split 中 preferred response 被渲染为 A 的比例确实为 100%。如果这是你们自己的转换，请把“RewardBench places chosen in A”改成“our conversion places chosen in A”。

2. 请说明与 JudgeLRM 的关系：你们是否使用 JudgeLRM 官方代码、超参数和 JudgeLM/PandaLM 数据？如果没有，请避免“directly contradicts”，并改为指出 JudgeLRM-style training 在 RewardBench-style fixed-position conversion 下会失败。

3. Balanced training 到底是“对每个样本创建 mirror version”使数据翻倍，还是“50% probability swap”保持数据规模不变？当前 method 和 appendix 表述不一致。若数据翻倍，请报告训练步数、epoch 数、样本重复和计算成本是否对比公平。

4. 请报告所有主要表格的均值±标准差或 bootstrap confidence intervals，尤其是 consistency。若 consistency std 为 7–9pp，请证明 reward ablation 中 2–6pp 的差异显著或至少不过度解释。

5. 请增加 true paired invariance reward / swap-consistency reward 与 balanced data 的直接对比，或把“reward engineering fails”的结论限定为所测试的 proxy rewards。

6. 请在至少一个非 Qwen 模型和一个非 RewardBench 数据集上验证结论，并考虑加入 answer quality gap 分层分析、preference fairness、repetition stability 和 list-wise comparison。

## Reproducibility and Ethics

可复现性目前中等偏弱。论文给出了主要超参数、prompt template 和硬件，但缺少训练代码、数据转换脚本、split IDs、exact decoding/parsing、reward implementation、raw outputs 和 run logs。考虑到本文挑战既有工作且主张“published accuracy numbers may overstate capability”，可复现性门槛应更高。伦理上，论文有正面价值：揭示自动 judge 的脆弱性，能减少下游 RLHF 和 leaderboard 的错误激励。风险在于标题和措辞中的 “cheat” 与对 JudgeLRM 的强指控可能被解读为对具体工作的过度攻击；建议使用更学术的 “shortcut exploitation” 和 “position-confounded evaluation”。

## Recommendation

我的建议是 Borderline / Weak Accept，confidence 4/5。若按 5 分制，我会给 3.5；按 ACL Rolling Review 风格，soundness 3、excitement 4、reproducibility 3、overall 3.5。接收理由是：问题重要，核心现象清楚，balanced training 的实用性强，论文有潜力改变 judge RL 的评估规范。保留意见是：当前稿件在核心前提表述、JudgeLRM 对比、reward-side 结论和引用完整性上存在明显可修正但重要的问题。

如果作者能在 rebuttal/修订中提供公开脚本与 split、修正 RewardBench 和 JudgeLRM 的措辞、补充显著性统计、清理 2026 引用，并至少加入一个 paired invariance reward 或非 Qwen/非 RewardBench 验证，我会明确支持接收。若这些问题无法解决，尤其是无法证明 preferred-fixed-A 是训练 pipeline 的真实且普遍 confound，那么论文应降为 Weak Reject。

## Limitations of This Review

本审稿基于本地 `main.tex` 与 `custom.bib`，并以 `main.pdf` 对应源文件作为主要阅读对象；没有运行作者实验代码，也没有访问作者未公开的原始日志。因此，对实验数值真实性只能做文本一致性和可复现性层面的评估。外部核查主要依赖公开 arXiv/Hugging Face/GitHub/web 搜索结果；若某些 2026 工作是未公开同期投稿，本文引用状态需要作者另行解释。

## References

1. [JudgeLRM: Large Reasoning Models as a Judge](https://arxiv.org/abs/2504.00050)
2. [RewardBench: Evaluating Reward Models for Language Modeling](https://arxiv.org/abs/2403.13787)
3. [RewardBench Hugging Face Dataset Card](https://huggingface.co/datasets/allenai/reward-bench/blob/main/README.md)
4. [Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge](https://arxiv.org/abs/2406.07791)
5. [FairJudge: An Adaptive, Debiased, and Consistent LLM-as-a-Judge](https://arxiv.org/abs/2602.06625)
6. [JudgeBiasBench: Toward Robust LLM-Based Judges](https://arxiv.org/abs/2603.08091)
7. [MJ1: Multimodal Judgment via Grounded Verification](https://arxiv.org/abs/2603.07990)
8. [Who can we trust? LLM-as-a-jury for Comparative Assessment](https://arxiv.org/abs/2602.16610)
9. [TrustJudge: Inconsistencies of LLM-as-a-Judge and How to Alleviate Them](https://arxiv.org/abs/2509.21117)
10. [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
11. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
12. [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780)
