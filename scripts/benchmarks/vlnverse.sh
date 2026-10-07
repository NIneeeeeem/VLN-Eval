#!/usr/bin/env bash
# VLN-VERSE: Isaac Sim 5.0 challenge scenes.
# Usage: METHOD=<method> [MODE=plan|run] [GPU=0] bash scripts/benchmarks/vlnverse.sh
set -euo pipefail
NAV_EVAL_BENCH_TAG="vlnverse"
NAV_EVAL_BENCH_ID="vlnverse"
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
