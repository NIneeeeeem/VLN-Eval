#!/usr/bin/env bash
# OneVLA: flow-matching head navigator (stochastic runs).
# Usage: [BENCHMARK=r2r_ce] [MODE=plan|run] [GPU=0] bash scripts/methods/onevla.sh
set -euo pipefail
NAV_EVAL_METHOD_ID="onevla"
NAV_EVAL_CONFIG_PREFIX="onevla"
NAV_EVAL_BENCHMARK="r2r_ce"
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
