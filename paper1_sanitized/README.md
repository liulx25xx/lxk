# Selective Recovery for Code Agents

**Active title**: "Redirect, Escalate, or Abstain? Selective Recovery for Code Agents"  
**Active manuscript**: `main_neurips.tex`  
**Compiled PDF**: `output/pdf/main_neurips.pdf`  
**Target direction**: AAAI-27 full-paper development; the current LaTeX file still uses the NeurIPS working template until venue-format conversion.  

The paper was realigned on 2026-07-22 using the submitted EMNLP PDF, the
official reviews, and the local artifact audit.  The new thesis treats recovery
as selection among redirect, evidence escalation, and abstention.  Historical
claims about an online four-way classifier, semantic ``waste,'' a universal
scaffolding frontier, and end-to-end recovery are not active claims.

## Active entry points

- `main_neurips.tex`: current paper source.
- `EMNLP_REVIEW_ALIGNMENT.md`: what was retained, corrected, or removed after
  the reviews.
- `AAAI27_EXPERIMENT_PRIORITIES.md`: paper-facing experiment priorities.
- `SERVER_EXPERIMENTS.md`: executable GPU/server queue.
- `LOCAL_AUDIT.md` and `LOCAL_EXPERIMENTS.md`: locally verified evidence.

The project notes below describe the original EMNLP-era pipeline and should be
treated as historical unless an active file above explicitly reuses them.

---

## 项目结构

```
paper1/
├── README.md                  ← 你正在看的文件（项目总览）
├── EXPERIMENTS.md              ← 实验跟踪（实时更新，核心管理文档）
├── PAPER_OUTLINE.md            ← 论文完整框架（section/figure/table/citation 规划）
├── MERGED_PLAN.md              ← 三方向合并方案 + API 费用估算
│
├── data/
│   └── swebench_subset.json    ← 200 个 SWE-bench Verified instance（已选好）
│
├── prompts/
│   ├── agent_base.txt          ← Agent 基础系统 prompt
│   ├── failure_classifier.txt  ← 失败分类 LLM prompt
│   └── scaffolding/            ← 15 个 behavioral strategy prompts（5类型×3策略）+ 1 control
│
├── scripts/
│   ├── run_agent.py            ← 核心：跑 agent 收集轨迹（需适配 Venus API）
│   ├── collect_trajectories.py ← 批量收集（调用 mini-swe-agent Docker）
│   ├── run_scaffolding.py      ← 批量跑 scaffolding 策略测试
│   ├── annotate_failures.py    ← 失败类型标注（LLM + 规则双重）
│   ├── cb_engine.py            ← Checkpoint-and-Backtrack 引擎
│   └── quick_test.py           ← Dry-run 验证（30/30 pass）
│
├── results/                    ← 所有实验结果输出目录（按实验编号组织）
│   ├── EXP-003_trajectories/   ← 基线轨迹
│   ├── EXP-004_annotations/    ← 失败标注
│   ├── EXP-005_scaffolding/    ← 策略测试结果
│   ├── EXP-006_cascade/        ← Cascade 分析
│   ├── EXP-008_cb_oracle/      ← C&B Oracle
│   ├── EXP-009_cb_heuristic/   ← C&B Heuristic
│   └── ...
│
├── template/                   ← ACL/EMNLP 2026 LaTeX 模板
│   ├── acl.sty
│   ├── acl_latex.tex
│   └── ...
│
└── research/                   ← 调研文档（仅供参考，不影响实验）
    ├── review_submission1.md   ← submission1 的 review + gap 分析
    ├── novelty_ideas_*.md      ← 各方向 novelty 分析
    └── route_*.md              ← 路线调研报告
```

## 快速开始

### 1. 环境搭建
```bash
conda create -n emnlp python=3.11 -y
conda activate emnlp
pip install openai litellm datasets pandas tqdm docker mini-swe-agent
```

### 2. 配置 Venus API
```python
# 所有脚本需要适配 Venus API：
import os
os.environ['OPENAI_API_KEY'] = "<REDACTED_SECRET>"

from openai import OpenAI
client = OpenAI(base_url="<REDACTED_URL>")
```

**重要**：当前脚本使用 litellm，需要改为直接用 OpenAI SDK + Venus base_url。Venus API 的限流是每次调用间隔 6.5 秒。

### 3. 验证 API
```bash
python -c "
import os; os.environ['OPENAI_API_KEY']='<REDACTED_SECRET>'
from openai import OpenAI
c = OpenAI(base_url='<REDACTED_URL>')
r = c.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':'hi'}], max_tokens=5)
print(r.choices[0].message.content)
"
```

### 4. 按 EXPERIMENTS.md 顺序执行实验

---

## 关键注意事项

### API 费用控制
- 预算上限: **$100 第一阶段**（约 700-1000 次 call）
- 主力模型: **gpt-4o-mini**（最便宜，约 ¥1.5/千次）
- 每个脚本内置 MAX_CALLS 硬限制
- 先跑 5 个样本 pilot → 50 个确认趋势 → 再全量

### 结果保存
- 每 10 个样本自动增量保存到 JSON
- 每个实验结果放 `results/EXP-XXX_名称/` 目录
- 文件命名: `{instance_id}_{model}_{scaffold}.json`
- 断点续传: 脚本检测已有结果文件，跳过已完成的 instance

### Venus API 特殊要求
- 限流: 每次 call 间隔 6.5 秒（否则 429）
- 推理模型 (gpt-5.5, o4-mini): 用 max_completion_tokens，不设 temperature
- 通用模型 (gpt-4o-mini, gpt-4.1): 用 max_tokens + temperature=0

---

## 论文三个 Part 概述

| Part | 内容 | 主要实验 | 预估费用 |
|------|------|---------|---------|
| **Part 1** | 5+ 种失败类型的 behavioral scaffolding | EXP-003~005 | ~$30 |
| **Part 2** | Error cascade 分析 + C&B 方法 | EXP-006~010 | ~$40 |
| **Part 3** | 自动 Failure Classifier + Strategy Selector | EXP-011~014 | ~$20 |
| Buffer | 补充实验 + 重跑 | EXP-015~017 | ~$10 |
| **总计** | | | **~$100** |
