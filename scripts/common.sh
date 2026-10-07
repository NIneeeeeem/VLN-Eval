#!/usr/bin/env bash
# Shared driver for scripts/methods/*.sh and scripts/benchmarks/*.sh.
#
# A method entry script sets NAV_EVAL_METHOD_ID, NAV_EVAL_CONFIG_PREFIX and
# NAV_EVAL_BENCHMARK (default benchmark), then sources this file. A benchmark
# entry script sets NAV_EVAL_BENCH_TAG (config-name tag) and
# NAV_EVAL_BENCH_ID (plugin id), and the
# caller provides METHOD. Both flavors respond to the same environment:
#
#   MODE        plan | run (default run). plan only resolves plugins,
#               configuration and resources — it never starts models or
#               simulators, so it is safe to run first.
#   GPU         physical GPU index for this invocation; forwarded as --gpu to
#               plan/run, overriding the permanent installation defaults.
#               Defaults to 0. GPUS=0,1 selects one replica per GPU instead.
#   CONFIG      explicit experiment config; bypasses name resolution. Its
#               method/benchmark must still match the script's pair.
#   OUTPUT      run output parent (default runs/<prefix>-<tag>, run mode only).
#   PYTHON      control-plane interpreter (default python).
#   SMOKE_OK    set to 1 to allow smoke (diagnostic-subset) configs to be
#               selected automatically; without it a smoke config must be
#               requested explicitly via CONFIG.
#
# Config name resolution tries, in order: <prefix>-<tag>.json,
# <prefix>-<tag>-full.json, <prefix>-<tag>-smoke.json (the last one only with
# SMOKE_OK=1). Validation and equivalence configs are never auto-selected.

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
config_dir=$repo_root/configs/experiments

nav_eval_usage() {
    cat <<'EOF'
Environment: MODE=plan|run (default run),
CONFIG=<explicit experiment config>, OUTPUT=<run parent>, PYTHON=<interpreter>,
GPU=<physical GPU, default 0> or GPUS=0,1 for replicas,
SMOKE_OK=1 to allow smoke configs. See the
script header for defaults.
EOF
}

# Benchmark plugin id (what users pass as BENCHMARK) to config tag.
nav_eval_bench_tag() {
    case "$1" in
        r2r_ce) echo r2r ;;
        rxr_ce) echo rxr ;;
        hssd_socialnav) echo socialnav ;;
        *) echo "$1" ;;
    esac
}

# List config-file prefixes that have a config for the given tag.
nav_eval_methods_for_tag() {
    local tag=$1 found=()
    local f name
    for f in "$config_dir"/*-"$tag".json "$config_dir"/*-"$tag"-full.json "$config_dir"/*-"$tag"-smoke.json; do
        [[ -f $f ]] || continue
        name=$(basename -- "$f")
        found+=("${name%%-$tag*}")
    done
    printf '%s\n' "${found[@]}" | sort -u
}

nav_eval_resolve_config() {
    local prefix=$1 tag=$2 candidate name
    local candidates=("$prefix-$tag.json" "$prefix-$tag-full.json" "$prefix-$tag-smoke.json")
    for candidate in "${candidates[@]}"; do
        name=$(basename -- "$candidate")
        if [[ -f $config_dir/$name ]]; then
            if [[ $name == *-smoke.json && ${SMOKE_OK:-0} != 1 ]]; then
                echo "Only a smoke (diagnostic-subset) config exists for this pair: $name" >&2
                echo "Pass SMOKE_OK=1 to accept the subset, or CONFIG=<full-split config>." >&2
                return 2
            fi
            echo "$config_dir/$name"
            return 0
        fi
    done
    echo "No experiment config found for method '$prefix' on benchmark tag '$tag'." >&2
    echo "Existing configs for this prefix:" >&2
    ls -1 "$config_dir/$prefix"-*.json 2>/dev/null | xargs -r -n1 basename >&2 || echo "  (none)" >&2
    echo "Or point CONFIG at an explicit experiment config." >&2
    return 2
}

nav_eval_check_pair() {
    local config=$1 method_id=$2 bench_id=$3
    PYTHON=${PYTHON:-python}
    "$PYTHON" -B - "$config" "$method_id" "$bench_id" <<'PY'
import json
import sys

path, method_id, bench_id = sys.argv[1:4]
with open(path) as stream:
    config = json.load(stream)
# Config-name prefixes use hyphens (uni-navid), plugin ids underscores (uni_navid).
norm = lambda value: str(value).replace("-", "_")
if norm(config.get("method")) != norm(method_id) or config.get("benchmark") != bench_id:
    sys.exit(
        f"CONFIG mismatch: {path} has method={config.get('method')!r} "
        f"benchmark={config.get('benchmark')!r}, expected {method_id!r}/{bench_id!r}"
    )
PY
}

nav_eval_entry() {
    local method_id prefix bench_id bench_tag config mode output

    if [[ -n ${NAV_EVAL_BENCH_TAG:-} ]]; then        # benchmark entry script
        bench_tag=$NAV_EVAL_BENCH_TAG
        bench_id=$NAV_EVAL_BENCH_ID
        if [[ -z ${METHOD:-} ]]; then
            echo "METHOD=<method> is required. Available methods for '$bench_tag':" >&2
            nav_eval_methods_for_tag "$bench_tag" | sed 's/^/  /' >&2
            nav_eval_usage >&2
            return 2
        fi
        method_id=$METHOD
        prefix=${METHOD//_/-}
    else                                            # method entry script
        method_id=$NAV_EVAL_METHOD_ID
        prefix=$NAV_EVAL_CONFIG_PREFIX
        bench_id=${BENCHMARK:-$NAV_EVAL_BENCHMARK}
        bench_tag=$(nav_eval_bench_tag "$bench_id")
    fi

    mode=${MODE:-run}
    case "$mode" in plan|run) ;; *) echo "MODE must be plan or run" >&2; return 2 ;; esac
    if [[ -n ${RESOURCES:-} ]]; then
        echo "Register resources once with: python -m nav_eval configure --from <file>. Evaluation reads configs/local.json automatically." >&2
        return 2
    fi

    if [[ -n ${CONFIG:-} ]]; then
        config=$(realpath -e -- "$CONFIG") || { echo "CONFIG not found: $CONFIG" >&2; return 2; }
    else
        config=$(nav_eval_resolve_config "$prefix" "$bench_tag") || return 2
    fi
    nav_eval_check_pair "$config" "$method_id" "$bench_id"

    output=${OUTPUT:-runs/$prefix-$bench_tag}
    echo "# method=$method_id benchmark=$bench_id config=$(basename -- "$config") mode=$mode" >&2

    cd -- "$repo_root"
    PYTHON=${PYTHON:-python}
    local gpu_args=()
    if [[ -n ${GPUS:-} ]]; then
        [[ -z ${GPU:-} ]] || { echo "Choose GPU or GPUS, not both" >&2; return 2; }
        gpu_args=(--gpus "$GPUS")
    else
        gpu_args=(--gpu "${GPU:-0}")
    fi
    "$PYTHON" -B -m nav_eval plan --config "$config" "${gpu_args[@]}"
    if [[ $mode == run ]]; then
        mkdir -p -- "$(dirname -- "$output")"
        "$PYTHON" -B -m nav_eval run --config "$config" "${gpu_args[@]}" --output "$output"
    fi
}
