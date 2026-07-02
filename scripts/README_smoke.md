# Smoke SLURM files

Place these files in your repository like this:

```text
repo_root/
  src/
    run_compare.py
    ...
  scripts/
    smoke_array.slurm
    submit_smoke.sh
```

Run:

```bash
chmod +x scripts/submit_smoke.sh
bash scripts/submit_smoke.sh
```

If the model names are different, edit `submit_smoke.sh` or run:

```bash
export SMOKE_QWEN="exact/qwen-model-id"
export SMOKE_GEMMA="exact/gemma-model-id"
export SMOKE_MISTRAL="exact/mistral-model-id"
bash scripts/submit_smoke.sh
```

This smoke test runs:

- Normal(4,1)
- prompt = plain
- prefixes = ROOT,3,4,4.,5
- n_samples = 10000
- scoring = auto

Outputs should appear in `outputs/token_level_smoke_*.csv` and `outputs/prefix_summary_smoke_*.csv`.
