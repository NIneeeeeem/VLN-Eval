#!/usr/bin/env bash
# AwareVLN: reasoning/action dual-mode LLaVA policy.
# Usage: [BENCHMARK=r2r_ce] [MODE=plan|run] [GPU=0] bash scripts/methods/awarevln.sh
set -euo pipefail
NAV_EVAL_METHOD_ID="awarevln"
NAV_EVAL_CONFIG_PREFIX="awarevln"
NAV_EVAL_BENCHMARK="r2r_ce"
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
