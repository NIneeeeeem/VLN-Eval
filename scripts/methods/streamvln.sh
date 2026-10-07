#!/usr/bin/env bash
# StreamVLN: streaming video VLM; method env carries habitat 0.2.4.
# Usage: [BENCHMARK=r2r_ce] [MODE=plan|run] [GPU=0] bash scripts/methods/streamvln.sh
set -euo pipefail
NAV_EVAL_METHOD_ID="streamvln"
NAV_EVAL_CONFIG_PREFIX="streamvln"
NAV_EVAL_BENCHMARK="r2r_ce"
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
nav_eval_entry
