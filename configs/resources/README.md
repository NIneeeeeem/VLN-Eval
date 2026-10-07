# Legacy installation maps

These examples are one-time migration input. Register an existing map with:

```bash
python -m nav_eval configure --from configs/resources/my-host.json
```

The permanent installation lives in the Git-ignored `configs/local.json`.
For new installations, use `configure --method ... --simulator ...` after
downloading assets, or use [local.example.json](../local.example.json) as a
template. Evaluations load this installation automatically:

```bash
METHOD=streamvln BENCHMARK=r2r_ce GPU=0 bash scripts/eval.sh
```

Model/benchmark selection and GPU allocation belong to the Bash evaluation
entry. The `plan` and `run` CLI commands do not accept `--resources`.
