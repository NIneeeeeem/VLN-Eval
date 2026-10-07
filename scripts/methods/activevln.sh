#!/usr/bin/env bash
# ActiveVLN: in-process Qwen2.5-VL with RL/SFT variants.
# Usage: [BENCHMARK=r2r_ce] [MODE=plan|run] [GPU=0] bash scripts/methods/activevln.sh
set -euo pipefail
NAV_EVAL_METHOD_ID="activevln"
NAV_EVAL_CONFIG_PREFIX="activevln"
NAV_EVAL_BENCHMARK="r2r_ce"
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
