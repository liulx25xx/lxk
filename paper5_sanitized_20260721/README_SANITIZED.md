# Sanitized package for `paper5`

Created: 20260721

## Included
- Project notes / write-ups: all top-level `*.md` (proposal, method design, survey, plans, reports, etc.)
- `src/` — training / reward / data / eval / analysis Python source
- `scripts/` — launch and eval shell scripts
- `paper/` — LaTeX sources, figures, compiled PDF
- `eval_results/` — saved evaluation summaries and per-sample JSON/JSONL. These
  are sufficient for the 2026-07-21 audit figure, but they are not a complete
  lineage for every figure/table in the current manuscript.
- `docs/aaai27/` — experiment-first execution plan, result tracker, and the
  paper-revision plan to use after results are frozen.
- `template/` — paper template (git history stripped)
- `requirements.txt`

## Excluded
- `outputs/` — all model / checkpoint artifacts (~880G): `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, adapters, tokenizers
- `data/` — raw / processed / SFT / DPO / OOD datasets (~360M jsonl)
- `logs/` — raw training / eval logs. These must be restored to validate or
  regenerate training-trajectory and KL/update-diagnostic figures.
- `.git/`, `__pycache__/`, `*.pyc`

## Redactions applied
- Hugging Face access token (`hf_...`) → `<HF_TOKEN_REDACTED>` (4 occurrences)
- Private IPv4 addresses (10.x / 172.16-31.x / 192.168.x) → `<REDACTED_IP>` (595 occurrences)
- Internal absolute workspace paths → `/path/to/workspace/...`
- Remaining bare username occurrences in home/cache paths → `user`

No OpenAI/Anthropic/WANDB API keys, passwords, or SSH private keys were found in the included files.
