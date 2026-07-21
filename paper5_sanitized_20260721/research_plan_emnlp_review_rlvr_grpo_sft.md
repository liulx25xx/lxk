# Research Plan: EMNLP-style review of paper5 main.pdf

Query type: depth-first review. The user asks for an EMNLP reviewer-style assessment of `/path/to/workspace/project/emnlp/paper5/paper/main.pdf`, using only the PDF content plus latest external knowledge. The central object is a paper titled “One Recipe Does Not Fit All: How Domain Characteristics Shape Post-Training Effectiveness,” about controlled comparisons of SFT and GRPO/RLVR across Math, Science, Medicine, Law, and Commonsense.

Key review dimensions: contribution and novelty relative to 2025-2026 RLVR/GRPO/SFT work; empirical soundness and statistical validity; methodological fairness and possible confounds; claim strength; writing and positioning; EMNLP accept/reject recommendation.

Research strategy:
1. PDF-content review: extract the paper’s actual claims, setup, tables, appendices, limitations, and any internal inconsistencies. Expected output: concise summary of what the paper claims and reviewer concerns grounded only in the PDF.
2. Latest external knowledge: use web search for recent RLVR/GRPO/SFT post-training work, including DeepSeek-R1, DeepSeekMath/GRPO, DAPO, Med-RLVR, “Does RL really incentivize reasoning,” SFT-vs-RL comparative studies, and 2025-2026 discussions about RLVR generalization and hyperparameter sensitivity. Expected output: whether the novelty and related-work framing are credible.
3. WeChat Official Account search: use `wechat-article-search` with explicit time parameters, e.g. keywords `RLVR GRPO SFT 后训练`, `DeepSeek R1 GRPO DAPO 后训练`, time range/recent days around 2025-2026. Treat these as Chinese technical commentary signals, not primary academic evidence; synthesize them with web/academic sources to identify whether the topic is timely and how practitioners frame recipe sensitivity.
4. Empirical/statistical critique: inspect whether the evidence supports the strong claims. Focus on seed counts, baselines, reward extraction, OOD setup, Qwen3 validation, LoRA-only limitation, learning-rate/KL/group-size interactions, train-test contamination risk, and missing ablations.
5. Synthesis: write an EMNLP-style review with summary, strengths, weaknesses, detailed comments, questions for authors, ethical/broader impact if relevant, and final score/confidence.

Search keywords and time range:
- Web: `RLVR GRPO SFT post-training learning rate 2025`, `DeepSeek R1 GRPO DAPO RLVR 2025`, `Does reinforcement learning incentivize reasoning capacity beyond base model 2025`, `SFT memorizes RL generalizes post-training 2025`, `Med-RLVR verifiable rewards medical reasoning 2025`.
- WeChat: `RLVR GRPO SFT 后训练` and `DeepSeek R1 GRPO DAPO 后训练`, use `--days 365` or explicit `--time-range 2025-01-01 2026-05-23`.

Planned subagents: one for PDF-based technical review, one for latest literature/novelty, one for empirical/statistical critique. Combine all findings, prioritizing PDF evidence for the final review while using latest knowledge only to assess novelty and context.
