#!/usr/bin/env bash
# Offline scoring is selected by the frozen run, not by a new model config.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash scripts/eval.sh RUN_DIR [--metric-set FILE] [--plugin-dir DIR ...]'
    echo 'RUN_DIR must contain run.json; use the actual <run-id>, not its parent.'
    echo 'Works for any collected method/benchmark; PYTHON selects the control-plane interpreter.'
    exit 0
fi
if (( $# == 0 )); then
    echo 'Usage: bash scripts/eval.sh RUN_DIR [evaluate options]' >&2
    exit 2
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
