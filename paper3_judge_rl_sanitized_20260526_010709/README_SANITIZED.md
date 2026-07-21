# Sanitized package for `paper3_judge_rl`

Created: 20260526_010709

Included:
- `paper/`
- `results/` non-model experiment summaries/logs/JSON/Markdown files
- `scripts/`, `prompts/`, `research/`, and project notes for reproducibility context

Excluded:
- `data/`
- model/checkpoint artifacts: `results/**/checkpoints/`, `results/**/final_model/`, `*.pt`, `*.safetensors`, `*.bin`, tokenizer files
- `.git`, `.codebuddy`, caches

Redactions applied:
- Hugging Face tokens and common API/token/password patterns
- private IPv4 addresses

Files redacted: 7
