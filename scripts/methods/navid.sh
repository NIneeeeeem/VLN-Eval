#!/usr/bin/env bash
# NaVid: VLM navigator with 30-degree turns.
# Usage: [BENCHMARK=r2r_ce] [MODE=plan|run] [GPU=0] bash scripts/methods/navid.sh
set -euo pipefail
NAV_EVAL_METHOD_ID="navid"
NAV_EVAL_CONFIG_PREFIX="navid"
NAV_EVAL_BENCHMARK="r2r_ce"
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
