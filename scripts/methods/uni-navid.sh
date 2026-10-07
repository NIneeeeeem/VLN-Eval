#!/usr/bin/env bash
# Uni-NaVid: unified NaVid variant, HFOV 120.
# Usage: [BENCHMARK=r2r_ce] [MODE=plan|run] [GPU=0] bash scripts/methods/uni-navid.sh
set -euo pipefail
NAV_EVAL_METHOD_ID="uni_navid"
NAV_EVAL_CONFIG_PREFIX="uni-navid"
NAV_EVAL_BENCHMARK="r2r_ce"
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
