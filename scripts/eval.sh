#!/usr/bin/env bash
# Evaluate a Bash-selected model/benchmark, or rescore a collected run.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash scripts/eval.sh RUN_DIR [--metric-set FILE] [--plugin-dir DIR ...]'
    echo 'Or: METHOD=streamvln BENCHMARK=r2r_ce GPU=0 bash scripts/eval.sh'
    echo 'MODE=plan resolves the pair without starting models or simulators.'
    echo 'RUN_DIR must contain run.json; use the actual <run-id>, not its parent.'
    echo 'Works for any collected method/benchmark; PYTHON selects the control-plane interpreter.'
    exit 0
fi
if (( $# == 0 )); then
    : "${METHOD:?Set METHOD to the model plugin, e.g. streamvln}"
    : "${BENCHMARK:?Set BENCHMARK to the benchmark plugin, e.g. r2r_ce}"
    NAV_EVAL_BENCH_ID=$BENCHMARK
    source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
    NAV_EVAL_BENCH_TAG=$(nav_eval_bench_tag "$BENCHMARK")
    nav_eval_entry
    exit 0
fi
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run_dir=$(realpath -e -- "$1")
shift
if [[ ! -f $run_dir/run.json ]]; then
    echo "Not a run directory (missing run.json): $run_dir" >&2
    exit 2
fi
args=()
while (( $# )); do
    case "$1" in
        --metric-set|--plugin-dir)
            if (( $# < 2 )); then
                echo "Missing value for $1" >&2
                exit 2
            fi
            option_path=$(realpath -e -- "$2")
            args+=("$1" "$option_path")
            shift 2
            ;;
        *) echo "Unsupported evaluate option: $1" >&2; exit 2 ;;
    esac
done
cd -- "$repo_root"
exec "${PYTHON:-python}" -B -m nav_eval evaluate --run "$run_dir" "${args[@]}"
