#!/usr/bin/env bash
# Edit the pairs here to define the models and benchmarks to evaluate.
# GPU=0 bash scripts/eval_suite.sh; MODE=plan checks all pairs first.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
pairs=(
    "streamvln r2r_ce configs/experiments/streamvln-r2r-full.json"
    "streamvln rxr_ce configs/experiments/streamvln-rxr.json"
)
for pair in "${pairs[@]}"; do
    read -r method benchmark config <<< "$pair"
    METHOD=$method BENCHMARK=$benchmark CONFIG=$repo_root/$config bash "$repo_root/scripts/eval.sh"
done
