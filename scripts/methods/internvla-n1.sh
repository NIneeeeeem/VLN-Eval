#!/usr/bin/env bash
# InternVLA-N1: dual-system navigator with camera-tilt probes.
# Usage: [BENCHMARK=r2r_ce] [MODE=plan|run] [GPU=0] bash scripts/methods/internvla-n1.sh
set -euo pipefail
NAV_EVAL_METHOD_ID="internvla_n1"
NAV_EVAL_CONFIG_PREFIX="internvla-n1"
NAV_EVAL_BENCHMARK="r2r_ce"
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
