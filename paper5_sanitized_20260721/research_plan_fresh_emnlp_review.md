# Fresh EMNLP Review Plan

Task: Independently review `/path/to/workspace/project/emnlp/paper5/paper/main.pdf` as an EMNLP reviewer, without relying on previous review notes.

Query type: depth-first paper review. The main goal is to evaluate the PDF’s contribution, empirical validity, novelty relative to recent RLVR/GRPO/SFT literature, presentation, limitations, and likely EMNLP recommendation.

Plan:
1. Extract and read PDF content directly. Identify title, abstract, experimental setup, results tables, appendices, limitations, and any internal consistency issues.
2. Search recent public literature for RLVR/GRPO/SFT post-training context: DeepSeek-R1, DeepSeekMath/GRPO, DAPO, Med-RLVR, SFT-vs-RL generalization, and critiques of RLVR reasoning.
3. Use WeChat Official Account search with explicit time range for Chinese technical-community signals: keywords `RLVR GRPO SFT 后训练`, `DeepSeek R1 GRPO DAPO 后训练`, time range `2025-01-01` to `2026-05-24`; use only as secondary context, not primary evidence.
4. Synthesize a fresh EMNLP-style review: summary, strengths, major weaknesses, minor issues, questions, recommendation score/confidence, and concrete revision priorities.

Expected output: A new Markdown review report saved in the workspace and a concise summary to the user.
