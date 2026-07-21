# Novelty Check: RL Training Creates Accuracy-Consistency Tradeoff in LLM Judges

## Insight
"RL training for LLM judges creates an accuracy-consistency tradeoff: accuracy improves dramatically (+14pp) but position consistency collapses (-22pp). Even adding consistency reward fails to prevent this. The judge becomes more accurate but more biased."

---

## Novelty Score: 8.5/10

---

## Key Finding: This Directly CONTRADICTS Existing Claims

### Closest Competitors

#### 1. JudgeLRM (Chen et al., arXiv 2504.00050, Mar 2025)
- **Their claim**: RL training with judge-wise outcome rewards **improves** self-consistency AND **reduces** position bias toward first answer
- **Quote**: "JudgeLRM shows substantially improved self-consistency and reduced bias toward the first answer compared to other models"
- **Key difference**: They ONLY optimize for accuracy reward. They claim consistency improves as a *byproduct*. They do NOT report what happens to position consistency metrics systematically (their "self-consistency" may be repetition stability, not position consistency)
- **Our contradiction**: We show that when you carefully measure position consistency (swap-and-check), RL training DESTROYS it even as accuracy rises

#### 2. FairJudge (Yang et al., arXiv 2602.06625, Feb 2026)
- **Their claim**: 3-stage pipeline (SFT → DPO → GRPO with consistency reward) improves BOTH accuracy AND consistency
- **Results**: Agreement 70→77, Consistency 60→66
- **Key difference**: They use a multi-stage pipeline specifically designed to preserve consistency (DPO for debiasing + GRPO with explicit consistency reward). They start from SFT baseline, not test what pure RL accuracy reward does alone
- **Our contradiction**: Even with consistency reward added to RL, consistency still collapses. Their "consistency" = pointwise-pairwise agreement, which is different from position consistency

#### 3. Self-Taught Evaluators (Wang et al., arXiv 2408.02666, Aug 2024)
- **Focus**: Iterative self-improvement of judges without human annotation
- **No report**: Does not measure position consistency before/after training
- **Relevance**: Shows training can improve accuracy but does not check bias side effects

#### 4. "Judging the Judges" (IJCNLP 2025)
- **Focus**: Systematic measurement of position bias in existing LLM judges
- **Key insight**: Position bias varies across judges and tasks
- **Limitation**: Measures bias in pretrained/prompted models, does NOT study how training changes bias

#### 5. Alignment Tax (EMNLP 2024)
- **Concept**: RLHF improves alignment but degrades general capabilities
- **Analogy**: Our finding is a "judge alignment tax" — RL improves accuracy but degrades fairness
- **Key difference**: Alignment tax is about capability loss, ours is about bias amplification

---

## Why This Is Novel

| Aspect | Existing Work | Our Finding |
|--------|--------------|-------------|
| RL effect on judge accuracy | Improves (consensus) | Improves (+14pp) ✓ |
| RL effect on consistency | "Improves" (JudgeLRM) or "preserved" (FairJudge) | **COLLAPSES** (-22pp) ✗ |
| Consistency reward helps? | FairJudge says yes | Even with it, still collapses |
| Overall narrative | RL is a win-win for judges | RL creates fundamental tradeoff |

**The core novelty**: Everyone else reports (or assumes) that making judges more accurate also makes them more fair/consistent. We show the OPPOSITE — a fundamental tension that even explicit consistency rewards cannot resolve.

---

## Risk Assessment

| Risk | Level | Reasoning |
|------|-------|-----------|
| Someone already found this | LOW | JudgeLRM explicitly claims the opposite; FairJudge's pipeline avoids the problem by design. Nobody has documented the failure mode |
| Concurrent work | MEDIUM | Hot field; someone could be finding this now. But our specific angle (accuracy-consistency as fundamental tradeoff, even consistency reward fails) is distinctive |
| Trivially expected | LOW | If it were expected, JudgeLRM wouldn't claim consistency improves. The community ASSUMES RL helps both |

---

## One-Line Recommendation

**This is highly novel and directly contradicts JudgeLRM's claims — frame it as a "cautionary finding" that reveals a hidden cost of RL judge training that the community has overlooked or gotten wrong. The fact that even consistency reward fails is the killer differentiator.**

---

## Suggested Framing
- NOT "we trained a judge" (engineering)
- YES "we reveal a fundamental tradeoff hidden in RL judge training" (insight)
- Position against JudgeLRM: "They claim RL improves consistency. We show this is misleading — position consistency (the fairness-critical metric) actually collapses"
- The "even consistency reward fails" result seals the novelty

## Related Work to Cite
1. JudgeLRM (Chen et al., 2025) — direct contradiction target
2. FairJudge (Yang et al., 2026) — shows the problem IS solvable with multi-stage pipeline, but pure RL fails
3. "Judging the Judges" (2024/2025) — position bias measurement framework
4. Alignment Tax (EMNLP 2024) — conceptual analogy
5. Self-Taught Evaluators (Meta, 2024) — training judges, no bias analysis
