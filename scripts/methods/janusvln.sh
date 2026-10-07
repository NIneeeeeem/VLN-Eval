#!/usr/bin/env bash
# JanusVLN: self-contained checkpoint with embedded VGGT.
# Usage: [BENCHMARK=r2r_ce] [MODE=plan|run] [GPU=0] bash scripts/methods/janusvln.sh
set -euo pipefail
NAV_EVAL_METHOD_ID="janusvln"
NAV_EVAL_CONFIG_PREFIX="janusvln"
NAV_EVAL_BENCHMARK="r2r_ce"
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
