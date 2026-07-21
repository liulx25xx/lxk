# Paper 5 实验框架: "When Does RLVR Beat SFT?"
# 最终版 — 2026-05-16

---

## 1. Novelty 评估 (Fresh Check)

### 最相关竞争者

| Paper | Venue | 做了什么 | 和我们的差异 |
|-------|-------|---------|------------|
| **"SFT Memorizes, RL Generalizes"** (2501.17161) | ICML 2025 | SFT vs RL在card game + visual navigation上的对比 | **只有2个toy domain**（GeneralPoints, V-IRL），不是NLP reasoning；无data efficiency；无difficulty分析 |
| **"RL Squeezes, SFT Expands"** (ICLR 2026) | ICLR 2026 | SFT vs RL推理轨迹拓扑结构分析 | **只分析轨迹结构**（math+code），不做多域performance对比；无data scaling；不回答"when to use which" |
| **SRFT** (2506.19767) | ICLR 2026 | 单阶段混合SFT+RL方法 | **提出新方法**，不是systematic comparison；不研究何时用哪个 |
| **Med-RLVR** (2502.19655) | arXiv 2025 | RLVR vs SFT in medicine | **单一domain**，不做cross-domain comparison |
| **"The Invisible Leash"** (2507.14843) | arXiv 2025 | RLVR理论局限性 | **理论paper**，无实验，无SFT对比 |
| **DEPO** (ICLR 2026) | ICLR 2026 | RLVR数据效率 | 研究如何提高RLVR效率，不对比SFT |

### Novelty 结论

**评分: 8/10 — HIGH NOVELTY**

关键gap：**没有任何paper做过controlled multi-domain (4+ domains) RLVR vs SFT comparison with matched conditions**
- "SFT Memorizes, RL Generalizes" 最接近但只有2个toy domain，不是NLP reasoning
- "RL Squeezes, SFT Expands" 只看轨迹结构不看performance boundary
- 所有其他work要么single domain, 要么提method不做comparison

**我们的unique贡献 = 6 NLP domains × data scaling × difficulty × OOD × hybrid, same model/data/compute**

---

## 2. 精简实验设计 (实际可执行)

### 2.1 核心原则
- **Base model**: Qwen2.5-7B-Instruct (固定)
- **Training**: LoRA (rank 64, alpha 128) — 保证SFT和GRPO公平
- **Framework**: TRL GRPOTrainer (需安装到openrlhf env或新env)
- **不用swift**

### 2.2 Domain × Benchmark 选择 (精简到4+2)

**核心4个domain (MUST DO)**:

| Domain | 类型 | Training Data | N_train | ID Test | OOD Test |
|--------|------|---------------|---------|---------|----------|
| **Math** | Procedural | GSM8K train | 5,000 | GSM8K test (1319) | MATH-500 |
| **Code** | Procedural | MBPP train | 500 | MBPP test (500) | HumanEval (164) |
| **Science** | Mixed | ARC-Challenge train | 3,000 | ARC-C test (1172) | GPQA Diamond (198) |
| **Medicine** | Knowledge | MedQA train | 5,000 | MedQA test (1273) | MMLU-Medical |

**补充2个domain (HIGH PRIORITY)**:

| Domain | Training Data | N_train | ID Test | OOD Test |
|--------|---------------|---------|---------|----------|
| **Law** | LegalBench train | 2,000 | LegalBench test | MMLU-Law |
| **Commonsense** | ARC-Easy train | 5,000 | ARC-E test (2376) | WinoGrande (1267) |

**选择理由**:
- Math/Code = RLVR's home turf (verifier trivial, reasoning为主)
- Science = 中间地带 (MCQ format, 需要知识+推理)
- Medicine = SFT's home turf (knowledge-intensive, 需要domain knowledge)
- Law/Commonsense = 补充数据点

### 2.3 Reward Function

| Domain | Reward | 实现 |
|--------|--------|------|
| Math | 数值exact match (normalize后) | `rewards.py:math_reward` ✅ 已有 |
| Code | test case全过=1, 否则=0 | `rewards.py:code_reward` ✅ 已有 |
| Science | MCQ letter match | `rewards.py:mcq_reward` ✅ 已有 |
| Medicine | MCQ letter match | `rewards.py:mcq_reward` ✅ 已有 |
| Law | Binary yes/no match | `rewards.py:binary_reward` ✅ 已有 |
| Commonsense | MCQ letter match | `rewards.py:mcq_reward` ✅ 已有 |

### 2.4 SFT CoT数据构建

**关键**: SFT需要(question, CoT+answer)对, RLVR只需要(question, answer)
- Math: GSM8K已有step-by-step solution → 直接用
- Code: MBPP有canonical solution → 直接用
- MCQ domains: 用base model (Qwen2.5-7B-Instruct) rejection sampling:
  - 对每题生成8个CoT, 保留answer正确的
  - 选最短的正确CoT
  - 失败的题(8个都错)直接跳过

**不用72B生成** — 用7B自身rejection sampling更公平，且省API费

---

## 3. 实验矩阵

### Tier 1: Core Comparison (24 runs)
4 core domains × 3 methods (SFT, GRPO, DPO) × 2 seeds

| Run | Domain | Method | Seed | GPUs | Est. Time |
|-----|--------|--------|------|------|-----------|
| 1-2 | Math | GRPO | 42,123 | 1×8GPU | 8h |
| 3-4 | Math | SFT | 42,123 | 1×4GPU | 3h |
| 5-6 | Math | DPO | 42,123 | 1×4GPU | 4h |
| 7-8 | Code | GRPO | 42,123 | 1×8GPU | 4h |
| 9-10 | Code | SFT | 42,123 | 1×4GPU | 1h |
| 11-12 | Code | DPO | 42,123 | 1×4GPU | 2h |
| 13-14 | Science | GRPO | 42,123 | 1×8GPU | 6h |
| 15-16 | Science | SFT | 42,123 | 1×4GPU | 2h |
| 17-18 | Science | DPO | 42,123 | 1×4GPU | 3h |
| 19-20 | Medicine | GRPO | 42,123 | 1×8GPU | 8h |
| 21-22 | Medicine | SFT | 42,123 | 1×4GPU | 3h |
| 23-24 | Medicine | DPO | 42,123 | 1×4GPU | 4h |

### Tier 2: Data Scaling (16 runs)
2 representative domains × 4 data sizes × 2 methods (SFT vs GRPO)

| N_train | Math-SFT | Math-GRPO | Med-SFT | Med-GRPO |
|---------|----------|-----------|---------|----------|
| 100 | R25 | R26 | R29 | R30 |
| 500 | R27 | R28 | R31 | R32 |
| 2000 | R33 | R34 | R35 | R36 |
| 5000 | (=Tier1) | (=Tier1) | (=Tier1) | (=Tier1) |
| 10000 (Math only) | R37 | R38 | - | - |

### Tier 3: Hybrid (4 runs)
2 domains × 2 orderings

| Domain | SFT→GRPO | GRPO→SFT |
|--------|----------|----------|
| Math | R39 | R40 |
| Medicine | R41 | R42 |

### Tier 4: 补充domains (8 runs)
Law + Commonsense × 2 methods × 2 seeds — 只做SFT vs GRPO，不做DPO

**总计: ~50 runs**

---

## 4. Training Config (精确)

### 4.1 GRPO Config
```
model: Qwen2.5-7B-Instruct
training: LoRA r=64, alpha=128, all linears
G (num_generations): 8
temperature: 1.0
max_new_tokens: 2048 (math/code), 1024 (MCQ)
lr: 5e-7
kl_coeff: 0.001
clip_high: 0.28, clip_low: 0.2
batch: per_device=2, grad_accum=4 (→ 8 prompts/step on single GPU)
steps: proportional to data size (见下表)
reward: binary {0,1}
```

**Steps by data size** (target ~3 epochs of prompts):
| N_train | Steps |
|---------|-------|
| 100 | 40 |
| 500 | 200 |
| 2000 | 750 |
| 5000 | 1875 |
| 10000 | 3750 |

### 4.2 SFT Config
```
model: Qwen2.5-7B-Instruct
training: LoRA r=64, alpha=128, all linears (same as GRPO!)
lr: 2e-5
epochs: 3
batch: per_device=4, grad_accum=4
loss: response-only masking
```

### 4.3 DPO Config
```
model: Qwen2.5-7B-Instruct
ref_model: Qwen2.5-7B-Instruct (frozen)
training: LoRA r=64, alpha=128
beta: 0.1
lr: 5e-7
epochs: 1
preference pairs: from base model rollouts (8 per question)
```

### 4.4 公平性保证 (Critical for paper credibility)
- ✅ Same base model
- ✅ Same LoRA config (rank, alpha, target modules)
- ✅ Same training questions per condition
- ✅ Same evaluation (greedy, T=0, same prompt template)
- ✅ SFT额外有CoT demonstrations (intentional — reflects real-world tradeoff)
- ✅ Report total FLOPs + wall time for compute-efficiency analysis

---

## 5. Evaluation Protocol

### 5.1 Metrics
- **Primary**: Pass@1 accuracy (greedy decoding, T=0)
- **Secondary**: Pass@8 (T=0.7, for variance analysis)

### 5.2 Difficulty Stratification (post-hoc, no extra training)
1. Run base model (Qwen2.5-7B-Instruct, no training) on all test sets
2. Bucket questions by base model accuracy:
   - Easy: >70% (base model usually gets it)
   - Medium: 30-70% (sometimes gets it)
   - Hard: <30% (rarely gets it)
3. Report SFT/GRPO/DPO accuracy per bucket

### 5.3 OOD Evaluation
每个domain有一个held-out OOD benchmark (见Section 2.2)

### 5.4 Eval Infrastructure
- 用vLLM做inference (openrlhf env已有vLLM 0.19)
- Batch inference, greedy

---

## 6. Environment Setup Plan

### Option A (推荐): 在openrlhf env中安装TRL
```bash
# openrlhf env已有: torch 2.10, transformers 5.7, vllm 0.19, peft 0.19, accelerate 1.13
# 只需加TRL
pip install trl
```
**风险**: TRL和vLLM版本可能冲突 (之前遇到过TRL 0.24 + vLLM 0.19的import crash)
**缓解**: 先测试 `python -c "from trl import GRPOTrainer"` 是否成功

### Option B (备选): 新建emnlp_grpo env
```bash
conda create -n emnlp_grpo python=3.11
conda activate emnlp_grpo
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate peft datasets trl vllm
```

### Option C: 使用custom GRPO loop
train_grpo.py已有custom实现 (不依赖TRL) — 用model.generate()做rollout
- 优点: 无依赖冲突
- 缺点: 慢 (HF generate vs vLLM)
- 适用: 数据量小(N≤500)的runs

**实际策略**: 
- 先试Option A (5min验证)
- 失败则用Option B (30min setup)
- 小数据runs用Option C兜底

---

## 7. Compute Budget & Schedule

### 假设: 24× H200 (user会告知具体服务器)

**Day 1: 数据准备 + 环境 + Base model eval**
- [ ] 环境setup (Option A/B)
- [ ] 下载6个domain数据 (`prepare_datasets.py`)
- [ ] SFT CoT rejection sampling (base model生成)
- [ ] DPO preference pair生成
- [ ] Base model zero-shot eval (for difficulty stratification)

**Day 2-3: Tier 1 Core Comparison (24 runs)**
- 每台服务器跑3个并行run (8GPU GRPO or 4GPU SFT ×2)
- Math + Code GRPO最先跑 (验证pipeline)
- 同时跑SFT (更快)

**Day 3-4: Tier 2 Data Scaling + Tier 3 Hybrid (20 runs)**
- 小数据runs (N=100, 500) 很快 (<2h each)
- Hybrid = 先跑完base SFT/GRPO, 然后接着训

**Day 4-5: Tier 4 补充 + Eval + Analysis**
- Law/Commonsense补充runs
- 全面evaluation
- Difficulty stratification analysis
- OOD evaluation
- 画图 (RLVR Benefit Frontier)

### Parallelism Plan (24 GPUs across N servers)
- GRPO: 8 GPU/run → 最多3 parallel GRPO runs
- SFT: 4 GPU/run → 最多6 parallel SFT runs
- 混合调度: 通常2 GRPO + 2 SFT同时跑

---

## 8. 关键Paper故事线

**Title**: "When Does RLVR Beat SFT? A Controlled Multi-Domain Study"

**Intro hook**: Everyone uses RLVR or SFT, but nobody knows when to choose which.

**Core findings (hypothesized)**:
1. **Domain effect**: RLVR wins on Math/Code (procedural), SFT wins on Medicine (knowledge)
2. **Difficulty effect**: RLVR best at medium difficulty; SFT better at hard (needs new knowledge)
3. **Data efficiency**: RLVR better at N<500; SFT catches up at N>2000
4. **OOD**: RLVR generalizes better even where SFT wins ID
5. **Hybrid**: SFT→GRPO is Pareto-optimal

**Key differentiation vs existing work**:
- vs "SFT Memorizes, RL Generalizes": 我们用6个real NLP domains, 不是toy tasks
- vs "RL Squeezes, SFT Expands": 我们回答practical question "when to use which", 不只是mechanistic analysis
- vs Med-RLVR: 我们做multi-domain controlled comparison

---

## 9. Risk & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| 结果boring (一方总赢) | 20% | Fatal | 先跑Math+Medicine pilot看crossover |
| GRPO训练不稳定 | 30% | Medium | 用TRL GRPOTrainer (proven); 2 seeds |
| SFT CoT数据质量差 | 20% | Medium | Rejection sampling; 检查正确率 |
| Env dependency冲突 | 40% | Low | 3 options ready |
| 审稿人说"just a benchmark paper" | 30% | Medium | 强调insights和decision framework |
