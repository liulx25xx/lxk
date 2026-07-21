# How to Run Experiments

## Prerequisites

1. **Conda environment**: `emnlp` (Python 3.11)
2. **API Keys**: At minimum `OPENAI_API_KEY`; optionally `DEEPSEEK_API_KEY` and `ANTHROPIC_API_KEY`
3. **Docker**: Running (for SWE-bench evaluation)

## Quick Start

```bash
# Activate environment
conda activate emnlp

# Set API keys
export OPENAI_API_KEY=<REDACTED_SECRET>
export DEEPSEEK_API_KEY=<REDACTED_SECRET>          # for DeepSeek V4
export ANTHROPIC_API_KEY=<REDACTED_SECRET>     # optional, for Claude

# Navigate to scripts
cd /data_train/xiankunlin/project/emnlp/paper1/scripts

# Verify everything works (no API calls)
python quick_test.py

# Dry-run: check data loading and setup
python collect_trajectories.py -m gpt-4o-mini --dry_run
```

## Running Experiments

### EXP-003: Baseline Trajectory Collection

```bash
# Collect trajectories for each model (run one at a time or in parallel terminals)
python collect_trajectories.py -m gpt-4o-mini
python collect_trajectories.py -m gpt-4.1
python collect_trajectories.py -m deepseek-v4
python collect_trajectories.py -m claude-sonnet-4  # optional, expensive

# Resume from a specific index if interrupted
python collect_trajectories.py -m gpt-4o-mini --start_from 50
```

### EXP-004: Failure Annotation

```bash
# Annotate failures for a model (requires trajectories from EXP-003)
python annotate_failures.py -m gpt-4o-mini
python annotate_failures.py --all_models

# Dry run first
python annotate_failures.py -m gpt-4o-mini --dry_run
```

### EXP-005: Scaffolding Experiments

```bash
# Run all scaffolding strategies on all failure types
python run_scaffolding.py -m gpt-4o-mini -a ../data/annotations/failure_types_gpt-4o-mini.json

# Run specific type/strategy
python run_scaffolding.py -m gpt-4o-mini -a ../data/annotations/failure_types_gpt-4o-mini.json \
  --failure_type LOC --strategy LOC_C_test_guided

# Control condition (retry without scaffold)
python run_scaffolding.py -m gpt-4o-mini -a ../data/annotations/failure_types_gpt-4o-mini.json --control
```

### EXP-008/009/010: C&B Experiments

```bash
# Oracle (upper bound, uses gold patch)
python cb_engine.py -m gpt-4o-mini --mode oracle

# Heuristic (rule-based detection)
python cb_engine.py -m gpt-4o-mini --mode heuristic

# LLM (GPT-4o-mini as judge)
python cb_engine.py -m gpt-4.1 --mode llm
```

### Single Instance Test

```bash
# Test on one instance
python run_agent.py -i django__django-16379 -m gpt-4o-mini

# With scaffolding
python run_agent.py -i django__django-16379 -m gpt-4o-mini --scaffold LOC_A_broaden_search
```

## Output Structure

```
data/
├── trajectories/{model}/{instance_id}.json
├── annotations/failure_types_{model}.json
├── scaffolding/{type}/{strategy}/{model}/
├── cb_oracle/{model}/
├── cb_heuristic/{model}/
└── cb_llm/{model}/

results/
├── batch_{model}_{timestamp}.json
├── scaffolding_matrix_{model}.json
└── cb_{mode}_summary.json
```

## Cost Monitoring

Each trajectory JSON records `total_cost`. The batch summaries aggregate costs.
Estimated total budget: ~$500 (see EXPERIMENTS.md for breakdown).

## Troubleshooting

- **API key not set**: Scripts will error with a clear message. Check `echo $OPENAI_API_KEY`.
- **Rate limits**: Scripts use exponential backoff (3 retries). For heavy runs, spread across time.
- **Docker not running**: `docker info` should succeed. Start Docker if needed.
- **HuggingFace cache**: Already handled via `HF_HOME=/home/xiankunlin/.cache/huggingface`.
- **Interrupted run**: Use `--start_from N` to resume from instance N.
