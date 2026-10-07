#!/usr/bin/env bash
# NaVIDA: direct-Transformers Qwen2.5-VL navigator (unpublished weights).
# Usage: [BENCHMARK=r2r_ce] [MODE=plan|run] [GPU=0] bash scripts/methods/navida.sh
set -euo pipefail
NAV_EVAL_METHOD_ID="navida"
NAV_EVAL_CONFIG_PREFIX="navida"
NAV_EVAL_BENCHMARK="r2r_ce"
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
