# EMNLP 审稿报告：One Recipe Does Not Fit All: How Domain Characteristics Shape Post-Training Effectiveness

## Executive Summary

这篇论文研究 SFT 与 GRPO/RLVR 在不同任务域上的后训练效果，核心结论是：数学上沿用的标准 GRPO 学习率 `5e-7` 在五个域上几乎无效，而将学习率提高到 `2e-5`（法律域为 `5e-6`）能带来显著收益。论文选题非常及时，实验规模较大，主发现如果可靠，对 RLVR 实践有直接价值。然而，PDF 中存在若干结果表述不一致、统计证据不足和关键超参数混淆未消除的问题，削弱了强结论的可信度。我的总体建议是 `Weak Reject / Borderline`：贡献有潜力，但当前稿件需要先修正结果一致性、补充关键消融和更严格的统计说明。

## Background and Relevance

论文处在一个非常活跃的方向：DeepSeekMath/GRPO、DeepSeek-R1、DAPO、Med-RLVR 之后，社区普遍关心 RLVR 是否能从数学和代码推广到知识密集型或普通 NLP 任务。近期工作一方面支持“RL 比 SFT 更可能带来泛化”的观点，例如 `SFT Memorizes, RL Generalizes`，另一方面也有强质疑，例如 `Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?` 指出当前 RLVR 可能更多是在挖掘 base model 已有能力，而非创造新推理模式。因此，这篇论文用同一 base model、同一 LoRA 配置、同一评估协议比较多个域，是有清楚学术价值的。

中文技术社区的近一年讨论也显示，`RLVR/GRPO/SFT 后训练`、`DeepSeek-R1/GRPO/DAPO` 是高热主题，微信公众号检索中能看到大量 2026 年围绕后训练范式、GRPO 改进和 RLVR vs SFT 的文章。不过这些更适合作为趋势信号，不应替代正式学术文献。

## Paper Summary

论文使用 `Qwen2.5-7B-Instruct` 作为主要 base model，以 LoRA rank 64、alpha 128 进行训练，并在数学、科学、医学、法律、常识五个域上比较 SFT、GRPO 和 SFT→GRPO。PDF 声称训练了 `160+` 个模型，总计算约 `130 H200 GPU-hours`。主表给出的 in-domain 结果是：base 在 Math/Science/Medicine/Law/Commonsense 上分别为 `84.4/71.1/59.2/57.6/43.8`；标准 GRPO `lr=5e-7` 分别为 `85.0/71.3/58.7/54.5/44.4`，基本无提升甚至下降；调优后的 GRPO 分别达到约 `92.2/79.1/66.6/58.9/68.8`，相对 base 提升 `+7.8/+8.0/+7.4/+1.3/+25.0`。

论文进一步声称 SFT 数据量存在域依赖：Math 在 `N=100` 附近达到峰值后随更多数据明显退化，Science 随数据量单调提升，Medicine 多种子下基本平坦。作者提出 base-model perplexity 可预测 SFT 退化风险，并用 forward-KL vs reverse-KL 解释 SFT 与 GRPO 的差异：SFT 更 mode-covering，GRPO 更 mode-seeking；低学习率 GRPO 只会 sharpen 训练 prompt 上已有策略，而高学习率能带来更大的 distribution shift。

## Strengths

第一，问题设定重要。当前许多 RLVR 论文在单一域、不同模型或不同评估设置下报告成功，确实缺少“同一 base model、同一训练协议、多域对照”的系统实验。论文试图回答“数学配方能否直接迁移”这个问题，动机很强。

第二，实验设计在宏观上有吸引力。五个域覆盖 procedural reasoning、mixed QA、knowledge-intensive QA、法律和常识；方法包括 SFT、GRPO、SFT→GRPO；还包含学习率 sweep、OOD 评估和 Qwen3-8B 复验。对 EMNLP/ACL 社区而言，这比单域 RLVR 工程报告更有普适启发。

第三，主发现有实践价值。若结果成立，“标准 `5e-7` 不是通用默认值，非数学 MCQ 域可能需要 `2e-5` 级别学习率，而小数据域要更保守”会直接影响许多 post-training 实验配置。论文不是只说 RLVR 有效，而是强调 recipe calibration，这一点有贡献。

第四，作者没有只停留在最终准确率，还提供训练 reward、KL divergence、`frac_reward_zero_std`、SFT 数据 perplexity 等诊断指标。虽然这些分析目前证据还不够充分，但方向是正确的，有助于把经验结果上升为可操作原则。

第五，OOD 和 cross-model 验证提高了说服力。PDF 中 Science、Medicine 和 Commonsense 的 OOD 结果显示 GRPO 收益并非只来自 ID memorization；Qwen3-8B 上 Math/Science/Medicine 的复验也说明现象不完全局限于 Qwen2.5-7B。

## Major Concerns

最严重的问题是 PDF 内部存在多处数值和表述不一致，这会直接影响读者对实验可靠性的判断。Table 1 写 Science 的 ID test 是 `ARC-C`、OOD test 是 `GPQA`，但 Table 3 的 OOD 行却写 `Science (ARC-C)`，这像是把 ID 测试集误标为 OOD。Table 2 中 Commonsense 的 best GRPO 是 `68.8`、相对 base `+25.0`，而 Figure 1 caption 又写 `+24.4`。Math 的 best GRPO 在 Table 2 中是 `92.2±0.7`，但 appendix Table 11/12 和正文其他位置出现 `92.8`；Science 在不同表中也出现 `79.1/79.3/77.8` 等不一致。Medicine 的 SFT best 在主表是 `59.5±0.7_{n=100}`，Table 4 caption 却说 best SFT 是 `61.7 (+2.5%)`，而 appendix 多种子 Table 9 又显示 `2k` 平均只有 `59.4`，`61.7`似乎只是单 seed。对于一篇以经验结论为核心的论文，这些不一致必须在送审前彻底修正。

第二，统计证据不足。论文称 `160+` models 和 multi-seed，但很多关键结论仍像是由少量 seed 或单 seed 支撑。比如 best GRPO 的跨域学习率结论，如果每个 learning rate/domain 组合不是完整多种子，`+1.3` 的 Law 增益、Science/Medicine 的差异、以及 collapse 阈值都可能受 seed 影响。作者需要清楚列出每个表格每个 cell 的 seed 数、均值、标准差/置信区间，并对关键比较做显著性检验或 bootstrap。

第三，SFT 与 GRPO 的“公平比较”仍不充分。PDF 说 SFT 使用 base model rejection sampling，保留 8 次采样中最短的正确 trace；如果 8 次都失败，问题会被排除出 SFT 训练集。与此同时，正文又声称“same training questions per domain and data size condition”。这两句话存在潜在冲突：如果 SFT 排除了失败题，而 GRPO 使用原始训练题，那么训练分布并不相同；如果 GRPO 也使用过滤后的题，则应明确说明。更重要的是，SFT 得到 CoT demonstrations，GRPO 只得到 answer-level binary reward，这反映实际成本差异，但不是等信息比较。论文可以把问题表述为“practical recipe comparison”，但不应过度解释为方法本身的公平优劣。

第四，学习率结论与其他 GRPO 超参数混淆。论文固定 `β=0.001`、`K=8`、LoRA 配置、训练 epochs，并只 sweep learning rate。可是 GRPO 的有效 update size 同时受 learning rate、KL coefficient、group size、clip/normalization、sampling temperature、LoRA rank 和 batch size 影响。结论“learning rate is the key variable”目前过强；更稳妥的说法应是“在固定 β/K/LoRA 配置下，learning rate 是我们观察到的主导变量”。至少需要一个小型 `lr × β` 或 `lr × K` 消融来证明不是 KL constraint 或 group variance 造成的假象。

第五，LoRA-only 训练限制了外推。所有实验都在 LoRA rank 64 下进行，完整参数训练、不同 LoRA rank、不同 target modules 都可能改变最优学习率。论文将 `2e-5` 描述为知识域 GRPO 的推荐 recipe，但这可能只是 LoRA 参数子空间下的有效步长。若没有 full fine-tuning 或至少 LoRA rank 消融，建议把结论限定在“LoRA-based GRPO post-training”范围内。

第六，对“真正知识获取”的论证仍然偏强。OOD 提升确实减轻了训练集 memorization 的担忧，但并不能直接证明模型获得了新知识或新推理模式。`Does RL Really Incentivize Reasoning Capacity...` 这类最新研究提醒我们，RLVR 可能只是更有效地选择 base model 已有推理模式。本文的机制解释应更谨慎：可以说高学习率带来了更强的 distribution shift 和 OOD performance gains，但“genuine knowledge acquisition”需要更直接证据，例如 pass@k/coverage 分析、错误类型转移、对 base model 高采样覆盖范围的比较，或 probing 表明新知识确实被内化。

第七，SFT perplexity 预测退化的证据太薄。Table 5 基本只有 Math、Science、Medicine 三个点，且 perplexity 与 base accuracy、任务类型、demonstration 风格、答案空间和数据源高度共线。基于三点相关性得出“perplexity predicts degradation”很容易过拟合。建议扩展到 Law/Commonsense 或更多子任务，并报告相关系数、不确定性和控制变量后的关系。

第八，OOD 设置需要澄清。Table 1 与 Table 3 的 Science OOD 标签冲突之外，Law OOD 只提升 `+0.1`，GRPO 还不如 SFT；Math OOD 只有 `+1.4`。因此“GRPO gains transfer robustly”应改成更精确的表述：Science 和 Medicine transfer 很强，Commonsense 有中等提升，Math 和 Law 较弱。当前摘要中的“gains transfer or even amplify”略显选择性。

## Minor Concerns and Presentation Issues

论文引用中有多个 `Anonymous 2025/2026` 条目，例如 EventRL、R1-RE、Still-3、Delay Plateau Collapse。匿名引用在双盲投稿中可以存在，但如果这些是作者自己的同期投稿，应遵守 ACL 匿名引用规范；如果不是，需要提供可核验 metadata。当前 PDF 中“Anonymous, 2026 arXiv:2605.02909”在审稿时会显得可疑，至少需要明确这是匿名预印本还是被匿名化的相关工作。

Law 域只有 31 个 unique training prompts，却包含 `10,403` test examples。这个设置本身很特殊，不能与 2k training examples 的其他域并列解释为“域差异”。Law 的最优 lr 更小可能主要是样本量效应，而不是法律域特性。建议在主文中把 Law 作为 small-data case study，而非完整平行域。

MCQ answer extraction 只说明提取 `A/B/C/D`，但没有报告解析失败率、多个答案时的规则、选项顺序随机化、position bias 检查、以及 CoT 中先后出现多个字母时如何处理。对于 ARC/MedQA/MMLU/WinoGrande 这类选择题，解析规则能显著影响结果。

训练 reward 与 test accuracy 的“disconnect”分析很有趣，但需要给出每个域完整的 reward-test 曲线，而不仅是 Medicine 的详细图。否则很难支持“standard recipe fails everywhere”的机制解释。

计算预算表写 `130 H200 GPU-hours` 和 `160+ models`，听起来可行，但表中列出的模型数只部分可见，failed/exploratory 又没有 count。建议提供完整 experiment inventory，至少说明每类模型数量、平均训练步数和是否纳入主结论。

## Questions for the Authors

1. Science 的 OOD 测试到底是 `GPQA` 还是 `ARC-C`？如果是 GPQA，请修正 Table 3；如果是 ARC-C，请解释为什么它既是 ID 又是 OOD。
2. Table 2、Table 11、Table 12 中 Math/Science/Medicine 的 best GRPO 数字为何不一致？哪些是 single-seed，哪些是 multi-seed average？
3. SFT rejection sampling 排除失败问题后，GRPO 是否也使用完全相同的过滤后训练集合？如果不是，“same training questions”是否仍成立？
4. `lr=2e-5` 的优势是否在不同 `β`、`K` 或 LoRA rank 下保持？是否有最小的交互消融？
5. 高学习率带来的收益是否只是更高 KL / 更大 policy shift？如果把低 lr 训练更久，使总 KL 匹配 `2e-5`，结果会怎样？
6. 对 “genuine knowledge acquisition” 是否做过 pass@k coverage 或 base-model high-sampling upper bound 分析？
7. 医学 SFT “flat at all N” 是否依赖 shortest-correct trace 的构造方式？如果使用更长、更解释性的 correct trace 或 teacher-generated rationale，会发生什么？
8. 是否能公开代码、训练数据索引、每个 cell 的 seed 与 checkpoint？没有这些，160+ 模型的实验难以复现。

## Assessment Against Latest Literature

相对于 DeepSeekMath/DeepSeek-R1/DAPO，本文的区别在于不是提出新 RL 算法，而是质疑已有数学配方的跨域迁移性。这个定位是合理且有价值的。相对于 Med-RLVR，本文的优势是把医学放入统一多域框架，并直接比较 SFT/GRPO recipe。相对于 `SFT Memorizes, RL Generalizes`，本文从 synthetic/controlled generalization 扩展到多个 NLP benchmark domain，并强调 recipe calibration。相对于 `Does RL Really Incentivize Reasoning Capacity...`，本文的结论更乐观，但需要更谨慎地区分“提高 greedy accuracy / OOD accuracy”和“产生新推理能力”。

目前稿件相关工作已经覆盖部分代表论文，但应更明确讨论两类相反证据：一类认为 RL 泛化更好，另一类认为 RLVR 没有超越 base model 的 pass@k 能力边界。这样可以避免把结果过度包装成“RLVR 能知识获取”，而是更准确地说“在特定 LoRA-GRPO 配方下，高学习率显著改善 greedy/OOD accuracy”。

## Overall Recommendation

我的建议是 `Weak Reject / Borderline`，分数约 `3/5` 或 `5/10`，confidence `4/5`。如果会议评分尺度偏 ARR，我会给“marginally below acceptance threshold”；如果作者在 rebuttal 中证明数值不一致只是排版错误，并补充完整 seed/CI、修复 OOD 标签、澄清 SFT/GRPO 数据一致性，我会倾向提升到 `Weak Accept`。

这篇论文的核心 idea 很好：RLVR vs SFT 不应被视为范式二选一，而应视为域、数据量和 recipe 的校准问题。最需要修的不是写作，而是结果可信度。当前 PDF 中的数值冲突、seed 不透明、SFT 过滤策略不清和超参数交互缺失，会让 EMNLP 审稿人难以放心接受强结论。若这些问题得到解决，它会是一篇有影响力的实证论文。

## Limitations of This Review

本审稿主要基于 `main.pdf` 抽取文本和公开最新文献/检索结果；没有运行代码，也没有检查原始实验日志。由于 PDF 中若干表格可能存在排版或抽取误差，部分数值不一致需要作者用源实验表确认。

## References

1. [Local PDF: One Recipe Does Not Fit All](file:///path/to/workspace/project/emnlp/paper5/paper/main.pdf)
2. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
3. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
4. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
5. [Med-RLVR: Emerging Medical Reasoning from a 3B base model via Reinforcement Learning](https://arxiv.org/abs/2502.19655)
6. [SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training](https://arxiv.org/abs/2501.17161)
7. [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)
8. [RL Fine-Tuning Heals OOD Forgetting in SFT](https://arxiv.org/abs/2509.12235)
9. [WeChat/Sogou search: RLVR GRPO SFT 后训练](https://weixin.sogou.com/weixin?type=2&query=RLVR%20GRPO%20SFT%20%E5%90%8E%E8%AE%AD%E7%BB%83)
10. [WeChat/Sogou search: DeepSeek R1 GRPO DAPO 后训练](https://weixin.sogou.com/weixin?type=2&query=DeepSeek%20R1%20GRPO%20DAPO%20%E5%90%8E%E8%AE%AD%E7%BB%83)
