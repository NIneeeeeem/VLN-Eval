#!/usr/bin/env bash
# NaVILA: hierarchical VLM navigator.
# Usage: [BENCHMARK=r2r_ce] [MODE=plan|run] [GPU=0] bash scripts/methods/navila.sh
set -euo pipefail
NAV_EVAL_METHOD_ID="navila"
NAV_EVAL_CONFIG_PREFIX="navila"
NAV_EVAL_BENCHMARK="r2r_ce"
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
