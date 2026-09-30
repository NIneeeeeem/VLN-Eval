#!/usr/bin/env bash
# Collect trajectories and score using the existing Nav-Eval runtime.
set -euo pipefail

usage() {
    echo 'Usage: bash scripts/inference.sh {streamvln|navida|internvla-n1} {r2r|vlnverse} RESOURCES.json [OUTPUT_PARENT]'
    echo 'Environment: MODE=plan|run (default run), CONFIG=experiment.json, PYTHON=python'
    echo 'Defaults: full val_unseen (VLNVerse fine); output runs/METHOD-DATASET/<run-id>.'
    echo 'InternVLA-N1 requires an external method plugin; it is not bundled.'
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if (( $# < 3 || $# > 4 )); then
    usage >&2
    exit 2
fi
method=$1
dataset=$2
case "$method" in streamvln|navida|internvla-n1) ;; *) usage >&2; exit 2 ;; esac
case "$dataset" in r2r|vlnverse) ;; *) usage >&2; exit 2 ;; esac
mode=${MODE:-run}
case "$mode" in plan|run) ;; *) echo 'MODE must be plan or run' >&2; exit 2 ;; esac
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Resolve user paths before changing cwd, so invocation from elsewhere works.
resources=$(realpath -e -- "$3")
config=$(realpath -e -- "${CONFIG:-$repo_root/configs/experiments/$method-$dataset-full.json}")
output=$(realpath -m -- "${4:-$repo_root/runs/$method-$dataset}")
cd -- "$repo_root"
python_bin=${PYTHON:-python}
# Prevent a CONFIG override from silently running a different model/benchmark.
"$python_bin" -B - "$config" "$method" "$dataset" <<'PY'
import json
import sys
from nav_eval.plugins import Registry

path, method, dataset = sys.argv[1:]
with open(path) as stream:
    config = json.load(stream)
benchmark = "r2r_ce" if dataset == "r2r" else "vlnverse"
if config.get("method") != method or config.get("benchmark") != benchmark:
    sys.exit("CONFIG method/benchmark does not match the requested pair")
if method == "internvla-n1":
    registry = Registry(config.get("plugin_dirs", []))
    try:
        registry.get("method", method)
    except (KeyError, ValueError):
        sys.exit("InternVLA-N1 is not bundled. Set CONFIG to an experiment with "
                 "plugin_dirs registering method 'internvla-n1' and provide its runtime resources.")
PY
args=(-B -m nav_eval "$mode" --config "$config" --resources "$resources")
if [[ $mode == run ]]; then
    args+=(--output "$output")
fi
exec "$python_bin" "${args[@]}"
