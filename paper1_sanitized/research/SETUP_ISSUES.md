# Setup Issues & Environment Status

**Date**: 2026-05-16
**Status**: Framework ready, blocked on API keys

---

## Environment Check

| Component | Status | Notes |
|-----------|--------|-------|
| Python | OK | 3.13.12 (miniconda3) |
| pip | OK | 26.0.1 |
| litellm | OK | Installed with mini-swe-agent |
| datasets | OK | v4.8.5 |
| mini-swe-agent | OK | v2.2.8 (CLI tool) |
| Docker | OK | Running and accessible |
| HuggingFace data | OK | SWE-bench Verified downloaded (cache: /home/xiankunlin/.cache/huggingface/) |

## Blocking Issues

### 1. API Keys NOT Set

| Key | Status | Required For |
|-----|--------|-------------|
| `OPENAI_API_KEY` | **NOT SET** | GPT-4o-mini, GPT-4.1, GPT-5.5 (main experiments) |
| `DEEPSEEK_API_KEY` | **NOT SET** | DeepSeek V4 (cross-family validation) |
| `ANTHROPIC_API_KEY` | **NOT SET** | Claude Sonnet 4 (optional) |

**Action needed**: User must set these environment variables before running experiments.

```bash
export OPENAI_API_KEY=<REDACTED_SECRET>
export DEEPSEEK_API_KEY=<REDACTED_SECRET>
export ANTHROPIC_API_KEY=<REDACTED_SECRET>  # optional
```

### 2. HuggingFace Cache Permission

The default HF cache at `/data_train/.cache/` is not writable. Workaround applied:
```bash
export HF_HOME=/home/xiankunlin/.cache/huggingface
```
This is already handled in the scripts via `os.environ.setdefault`.

## Non-Blocking Notes

- mini-swe-agent uses `litellm` backend — model names follow litellm format (e.g., `openai/gpt-4o-mini`)
- Docker sandbox is available for SWE-bench execution
- mini-swe-agent's SWE-bench config expects Docker environments per repo/version
- Full SWE-bench Docker images may need to be pulled (~10-30GB per repo)
  - First run will be slow due to image pulls
  - Consider pre-pulling images for top repos: `docker pull swebench/django:*`

## File Structure Created

```
paper1/
├── data/
│   └── swebench_subset.json       ← 200 instances, stratified by repo
├── prompts/
│   ├── agent_base.txt             ← Base agent system prompt
│   ├── failure_classifier.txt      ← LLM failure classification prompt
│   └── scaffolding/
│       ├── LOC_A_broaden_search.txt
│       ├── LOC_B_reread_issue.txt
│       ├── LOC_C_test_guided.txt
│       ├── EDIT_A_reread_file.txt
│       ├── EDIT_B_smaller_edit.txt
│       ├── EDIT_C_alternative_tool.txt
│       ├── LOGIC_A_test_analysis.txt
│       ├── LOGIC_B_minimal_fix.txt
│       ├── LOGIC_C_edge_cases.txt
│       ├── TEST_A_issue_reread.txt
│       ├── TEST_B_test_first.txt
│       ├── TEST_C_differential.txt
│       ├── PLAN_A_step_back.txt
│       ├── PLAN_B_scope_check.txt
│       ├── PLAN_C_similar_fixes.txt
│       └── CONTROL_no_scaffold.txt
├── scripts/
│   ├── run_agent.py               ← Core agent runner (single/batch)
│   ├── collect_trajectories.py    ← mini-swe-agent Docker integration
│   ├── run_scaffolding.py         ← Scaffolding experiment runner
│   ├── annotate_failures.py       ← LLM + rule-based failure annotation
│   └── cb_engine.py               ← Checkpoint-and-Backtrack engine
├── results/
├── figures/
└── models/classifier/
```

## Ready to Run

Once API keys are set, run:
```bash
# Quick validation (5 instances)
cd /data_train/xiankunlin/project/emnlp/paper1/scripts
python run_agent.py -i django__django-16379 -m gpt-4o-mini

# Full baseline collection
python collect_trajectories.py -m gpt-4o-mini --dry_run  # check first
python collect_trajectories.py -m gpt-4o-mini            # then run
```
