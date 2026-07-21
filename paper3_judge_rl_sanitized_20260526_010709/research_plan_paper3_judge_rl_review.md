# Research Plan: EMNLP 2026 Review of `paper3_judge_rl/paper/main.pdf`

## Assessment and breakdown
The task is to review the manuscript as an EMNLP 2026 reviewer with up-to-date LLM/judge/RL knowledge. The paper claims that RL-trained LLM judges exploit a position shortcut caused by RewardBench-like data where the preferred answer is always in position A; it reports experiments on Qwen2.5-7B and Qwen3-8B showing accuracy-consistency tradeoffs, failure of proxy multi-objective rewards, and balanced data as a fix.

Key facts to verify: novelty relative to LLM-as-a-judge position bias literature, RewardBench construction and whether chosen responses are always A in the used split, correctness of evaluation metrics, adequacy of baselines and multi-seed/statistical evidence, reproducibility details, whether claims about JudgeLRM and recent 2026 works are fair, and whether references appear real and appropriately cited.

Expected output: a professional EMNLP-style review with summary, strengths, weaknesses, questions for authors, reproducibility/ethics concerns if any, and a recommendation.

## Query type
Depth-first query. It focuses on one manuscript but requires multiple perspectives: technical soundness, novelty/related work, experimental methodology, citation integrity, and publication suitability.

## Research strategy
I will combine direct reading of the LaTeX source/PDF-equivalent content with external source checks and parallel subagent reviews.

The `wechat-article-search` skill is explicitly included for Chinese information retrieval. Search keywords and time range: `LLM judge position bias RewardBench`, `LLM-as-a-Judge 位置偏见`, `JudgeLRM GRPO judge`, `RewardBench 偏好数据 chosen A`, `LLM 裁判 强化学习 位置偏差`; time range `2025-01-01` to `2026-05-18` and/or `--days 365`. Use WeChat results only as supplementary signal for recent Chinese discussion; prioritize primary sources such as arXiv, ACL Anthology, OpenReview, official project pages, and GitHub when resolving factual claims. WeChat article findings will be synthesized with web search results by treating them as pointers to recent discussions, not as primary evidence.

## Subagent allocation
1. Technical reviewer subagent: read the paper and assess method, metrics, experimental design, statistics, causal claims, and reproducibility.
2. Related-work/novelty subagent: search current LLM-as-a-judge, RewardBench, judge RL, position/ordering bias, shortcut/reward hacking literature through 2026; judge novelty and missing references.
3. Citation/factual-integrity subagent: verify high-risk citations and claims, especially 2025/2026 references, RewardBench construction, JudgeLRM claims, and any anonymous/fabricated-looking entries.

## Necessary outputs from each step
Technical review should produce concrete accept/reject-relevant issues. Related-work review should identify whether the core contribution is new and what papers must be discussed. Citation review should flag unverifiable or wrong-context citations. The final synthesis will produce an EMNLP review and save it as `research_report_paper3_judge_rl_review.md`.
