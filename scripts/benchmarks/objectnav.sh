#!/usr/bin/env bash
# ObjectNav: MP3D/HM3D object-goal navigation.
# Usage: METHOD=<method> [MODE=plan|run] [GPU=0] bash scripts/benchmarks/objectnav.sh
set -euo pipefail
NAV_EVAL_BENCH_TAG="objectnav"
NAV_EVAL_BENCH_ID="objectnav"
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
