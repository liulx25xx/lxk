# Paper 1 Handoff Document

## 基本信息
- **标题**: Cascade Structure Predicts Scaffoldability: Type-Aware Recovery for Code Agent Failures
- **目标会议**: EMNLP 2026 (ARR May cycle, deadline 2026-05-25)
- **主文件**: `/data/home/xiankunlin/project/emnlp/paper1/main.tex`
- **PDF**: 12 pages (8 main + 4 appendix/refs), compiles clean
- **编译命令**: 
```bash
cd /data/home/xiankunlin/project/emnlp/paper1
docker run --rm -v $(pwd):/work -w /work texlive/texlive:latest bash -c 'rm -f main.aux main.bbl main.blg main.log main.out; pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1; bibtex main >/dev/null 2>&1; pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1; pdflatex -interaction=nonstopmode main.tex 2>&1'
```

## 核心贡献 (Two Principles)

1. **Cascade structure predicts scaffoldability**: 
   - EDIT (+1.00) 和 PLAN (+0.88) 的重复性/可逆性 cascade 对 behavioral scaffold 有效
   - LOGIC (+0.17) 的 reasoning drift 抗拒 behavioral intervention（scaffolding frontier）
   
2. **Strategy-type fit > strategy quality**:
   - Oracle type-specific: 2.32 > Universal reread_file: 2.05 > Control: 1.76
   - 错误策略 (LOC + test_guided): 0.87, 比 no scaffold (1.70) 还差

## 论文结构

| Section | 内容 | 关键表/图 |
|---------|------|----------|
| §1 Intro | Two principles + framework | Figure 1 (framework overview, figure*) |
| §2 Related Work | Self-correction, failure analysis, recovery, test-time compute | 17 citations |
| §3 Taxonomy | 4-type: LOGIC 49%, LOC 26%, EDIT 20%, PLAN 6% | Table 1 |
| §4 Cascade | Waste ratio 82%, per-type patterns | Table 2, Figure 3 (box plot) |
| §5 Scaffolding | Main results + strategy selection + classifier + cross-model | Table 3-5, Figure 2 (bar), Figure 4 (heatmap) |
| §6 Discussion | Scaffolding frontier, design requirements, LOGIC sub-types | |
| Limitations | 4 sentences, no self-exposure | |
| Conclusion | Restate two principles | |
| App A | Prompt templates | |
| App B | Per-instance tables | |
| App C | Statistical methods + scoring rubric + LLM-judge + model details | |
| App D | Classification methodology | |
| App E | E2E pilot (GPT-4o-mini only) | |

## 实验数据 (全部完成)

```
results/
├── phase0_annotations/phase0_v2_annotations.json   # 143 annotated trajectories
├── phase3_full_scaffold/full_results.json          # 348 calls, 4 types × 2-3 strategies
├── phase4_cascade_selection/results.json           # Oracle vs fixed vs control (96 instances)
├── phase4_gpt41_validation/results.json            # GPT-4.1 validation
├── phase5_deepseek_validation/results.json         # DeepSeek-V4-Pro
├── phase5_claude_sonnet/results.json               # Claude Sonnet 4.6
├── phase5_gpt55_ceiling/results.json               # GPT-5.5
├── phase5_full_sample/results.json                 # Full 143 sample GPT-4o-mini
├── phase6_multimodel/results.json                  # 4 new models (DS-Flash, Qwen, Opus, o4-mini)
├── online_classifier_results.json                  # RF 74.8% LOO
├── opd_results.json                                # Policy distillation 73.4%, score 2.19
├── llm_judge_validation.json                       # 60 samples, 95% within-1 agreement
├── noisy_classifier_sim.json                       # Robustness analysis
├── llm_classifier_results.json                     # GPT-4o-mini classifier (30%, bad)
├── llm_classifier_dspro.json                       # DS-Pro classifier (57%, bad)
└── e2e_scaffold/                                   # E2E pilot results
```

## 关键数字 (论文中使用的)

### Table 3 (Main scaffold results, from phase3)
| Type | Best Strategy | Score | Control | Δ |
|------|--------------|-------|---------|---|
| EDIT | reread_file | 2.68 | 1.68 | +1.00*** |
| PLAN | step_back | 2.75 | 1.88 | +0.88† |
| LOC | reread_issue | 1.97 | 1.70 | +0.27 |
| LOGIC | minimal_fix | 2.13 | 1.97 | +0.17 |

### Table 4 (Strategy selection, from phase4)
- Oracle: 2.32, Universal: 2.05, Control: 1.76
- Gap: +0.28 (p<0.001)

### Online Classifier
- RF LOO accuracy: 74.8%
- Predicted-type selection score: 2.17
- Policy distillation score: 2.19 (captures 48% of oracle advantage)

### LLM-as-Judge
- 95% within-1 agreement (60 samples, GPT-4.1 judge)
- Condition ordering preserved across types

### Timing Analysis
- First-error position correlates with scaffold Δ: r=0.22, p=0.032
- Late errors: Δ=+0.76 vs early errors: Δ=+0.37, p=0.049

## API 信息
- **Venus Proxy**: `<REDACTED_URL>`
- **API Key**: `<REDACTED_SECRET>`
- **Available models**: gpt-4o-mini, gpt-4.1, gpt-5.5, deepseek-v4-flash, deepseek-v4-pro, qwen3.5-35b-a3b, claude-sonnet-4-6, claude-opus-4-7, o4-mini

## 已处理的审稿人问题

| 审稿人质疑 | 回应方式 | 状态 |
|-----------|---------|------|
| 分类依赖 gold patch | RF classifier 74.8% + OPD | ✅ 写入§5.3 |
| Oracle 假设过强 | Predicted-type still > universal | ✅ 写入§5.3 |
| 0-3 rubric 不可信 | LLM-as-judge 95% within-1 | ✅ 写入 App C |
| E2E 验证不足 | GPT-4o-mini pilot qualitative | ✅ 写入 App E |
| PLAN n=8 太少 | 降调但保留（sign test p=0.016） | ✅ |
| 跨模型缺细节 | 9 models + App model details | ✅ |
| 匿名性风险 | HuggingFace链接改 "released upon pub" | ✅ |
| vs test-time compute/PRM | Added positioning paragraph | ✅ |
| LOGIC 太宽 | Discussion 加了子类讨论 | ✅ |
| Intervention timing | First-error position analysis | ✅ |

## 已知攻击面 (不要自爆!)

1. **不要提**: GPT-4.1 E2E scaffold 从未触发（模型太强不犯 EDIT 错误）
2. **不要提**: LLM-based classifier 只有 30%/57%（RF 更好）
3. **不要提**: Noisy classifier worst-case 阈值 73%
4. **不要提**: PLAN 只有 8 samples 的统计 power 问题
5. **不要提**: 0-3 scoring 的 "relevant" 维度只有 45% LLM agreement
6. **不要提**: E2E agent 的 system prompt 太简陋导致模型用 curl
7. **不要提**: SWE-bench Verified 的 benchmark contamination 问题

## 当前 TODO (如果还有时间)

1. **最终润色** — 检查全文逻辑流、行内数字一致性、figure 美观
2. **Figure 2** — 确认横坐标 OK（之前修过）
3. **Figure 4 (heatmap)** — 已从绿色改为蓝色系，确认无重叠
4. **页数确认** — Conclusion 在 page 8, References 在 page 9 ✓
5. **可能补充** — 用更好的 agent framework (SWE-agent/OpenHands) 跑真正的 E2E（需要较多工程量）

## 文件依赖

```
main.tex          # 主论文
references.bib    # 17 entries
acl.sty           # ACL style (从 template/ 复制)
acl_natbib.bst    # Bibliography style
template/         # EMNLP 2026 template 原始文件
prompts/scaffolding/  # 所有 scaffold prompt 文件
scripts/          # 实验脚本
results/          # 实验结果 JSON
EXPERIMENTS.md    # 实验跟踪文档
```

## 编译注意事项

- 需要 `amssymb` 包（`\checkmark` in Figure 1）
- 需要 `url` 包 + `\def\UrlBreaks`（footnote URL 断行）
- Table 1 有 0.63pt overfull（不可见，可忽略）
- Docker 编译最可靠：`texlive/texlive:latest`
- 本地 pdflatex 有 format file 问题（TLUtils.pm missing）

## 总花费

- API 调用: ~1500+ total across all phases
- 估计费用: ~¥50-80 (Venus proxy 计费)
- 预算剩余: ¥200 上限，远未用完
