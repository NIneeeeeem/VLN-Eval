#!/usr/bin/env bash
# GA-VLN: pose-sensor navigator (habitat024 binding; R2R only).
# Usage: [BENCHMARK=r2r_ce] [MODE=plan|run] [GPU=0] bash scripts/methods/gavln.sh
set -euo pipefail
NAV_EVAL_METHOD_ID="gavln"
NAV_EVAL_CONFIG_PREFIX="gavln"
NAV_EVAL_BENCHMARK="r2r_ce"
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
