#!/usr/bin/env bash
# R2R-CE: continuous VLN on MP3D (val_unseen/val_seen).
# Usage: METHOD=<method> [MODE=plan|run] [GPU=0] bash scripts/benchmarks/r2r_ce.sh
set -euo pipefail
NAV_EVAL_BENCH_TAG="r2r"
NAV_EVAL_BENCH_ID="r2r_ce"
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
