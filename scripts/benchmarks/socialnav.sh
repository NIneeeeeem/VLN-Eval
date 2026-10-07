#!/usr/bin/env bash
# HSSD social navigation.
# Usage: METHOD=<method> [MODE=plan|run] [GPU=0] bash scripts/benchmarks/socialnav.sh
set -euo pipefail
NAV_EVAL_BENCH_TAG="socialnav"
NAV_EVAL_BENCH_ID="hssd_socialnav"
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
