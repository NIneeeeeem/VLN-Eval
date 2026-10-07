#!/usr/bin/env bash
# RxR-CE: multilingual continuous VLN (smoke configs only today).
# Usage: METHOD=<method> [MODE=plan|run] [GPU=0] bash scripts/benchmarks/rxr_ce.sh
set -euo pipefail
NAV_EVAL_BENCH_TAG="rxr"
NAV_EVAL_BENCH_ID="rxr_ce"

source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
