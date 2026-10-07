#!/usr/bin/env bash
# Install dependencies inside this repository; no upstream Git checkout.
# bash scripts/setup_environment.sh streamvln habitat024
# INSTALL_GPU=0 skips the explicit CUDA wheel and FlashAttention installs.
set -euo pipefail
# Keep ~/.local packages from satisfying dependencies in a fresh Conda prefix.
export PYTHONNOUSERSITE=1
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_root"
profile=${1:-}
if [[ $profile == --help || $profile == -h || -z $profile ]]; then
    echo 'Usage: bash scripts/setup_environment.sh MODEL|GROUP|SIMULATOR [SIMULATOR]'
    echo 'Shared model groups: streamvln (StreamVLN/GA-VLN), vlm (other built-in methods).'
    echo 'Repository prefixes: envs/nav_<group>; nav_ profile aliases are accepted.'
    echo 'Simulators: habitat024, habitat030, isaacsim500 (Isaac Sim 5.0.0).'
    echo 'INSTALL_GPU=0 skips explicit CUDA wheels/FlashAttention for non-Isaac profiles.'
    echo 'TORCH_INDEX_URL defaults to https://download.pytorch.org/whl/cu128.'
    exit 0
fi
source "$repo_root/scripts/environment_profiles.sh"
nav_eval_environment_profile "$profile"
if [[ $environment_profile == isaac ]]; then
    exec bash "$repo_root/scripts/setup_isaac.sh" "$profile"
fi
profile=${profile#nav_}
numpy_version=1.26.4
simulator=${2:-}
simulator=${simulator#nav_}
if [[ $profile == habitat* ]]; then
    [[ -z $simulator ]] || { echo 'Specify only one simulator' >&2; exit 2; }
    simulator=$profile
fi
if [[ -n $simulator ]]; then
    case "$simulator" in
        habitat024) habitat_version=0.2.4 ;;
        habitat030) habitat_version=0.3.0; numpy_version=1.23.5 ;;
        *) echo "Unsupported simulator: $simulator" >&2; exit 2 ;;
    esac
    # The official Linux SDK builds use the CPython 3.9 ABI.
    [[ $python_version == 3.9 ]] || {
        echo 'Habitat Conda builds require Python 3.9; install a separate simulator environment.' >&2
        exit 2
    }
    [[ -z $requirements || $numpy_version == 1.26.4 ]] || {
        echo 'Habitat 0.3.0 requires NumPy <1.24; model groups require 1.26.4. Install habitat030 separately.' >&2
        exit 2
    }
fi
env_dir=$repo_root/envs/$environment_name
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$repo_root/envs/.conda-pkgs}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$repo_root/envs/.pip-cache}
if [[ ! -x $env_dir/bin/python ]]; then
    conda create -y -p "$env_dir" --override-channels -c conda-forge \
        "python=$python_version" pip "numpy=$numpy_version" 'setuptools<81'
fi
runtime_python=$env_dir/bin/python
# Build tools installed into this prefix must be visible to extension builds.
export PATH="$env_dir/bin:$PATH"
if [[ -n ${habitat_version:-} ]]; then
    installed_habitat=$("$runtime_python" -c 'import importlib.metadata as m; print(m.version("habitat-sim"))' 2>/dev/null || true)
    [[ -z $installed_habitat || $installed_habitat == "$habitat_version" ]] || {
        echo "Environment already contains Habitat $installed_habitat; use a separate prefix for $habitat_version." >&2
        exit 2
    }
    conda install -y -p "$env_dir" --override-channels -c conda-forge -c aihabitat \
        "python=$python_version" "numpy=$numpy_version" "habitat-sim=$habitat_version" withbullet headless
fi
if [[ -n $requirements ]]; then
    if [[ ${INSTALL_GPU:-1} == 1 ]]; then
        "$runtime_python" -m pip install torch==2.8.0 torchvision==0.23.0 \
            --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
    fi
    "$runtime_python" -m pip install -r "configs/environments/$requirements"
    if [[ ${INSTALL_GPU:-1} == 1 ]]; then
        export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
        export PATH="$CUDA_HOME/bin:$PATH"
        "$runtime_python" -m pip install packaging ninja wheel
        MAX_JOBS=${MAX_JOBS:-4} "$runtime_python" -m pip install flash-attn==2.8.3.post1 --no-build-isolation
    fi
fi
if [[ -n ${habitat_version:-} ]]; then
    "$runtime_python" -m pip install "numpy==$numpy_version" 'opencv-python<4.12' 'moviepy<2' 'setuptools<81' \
        "https://codeload.github.com/facebookresearch/habitat-lab/zip/refs/tags/v$habitat_version#subdirectory=habitat-lab" \
        "https://codeload.github.com/facebookresearch/habitat-lab/zip/refs/tags/v$habitat_version#subdirectory=habitat-baselines"
fi
"$runtime_python" -m pip install --no-deps -e "$repo_root"
if dependency_check=$("$runtime_python" -m pip check 2>&1); then
    printf '%s\n' "$dependency_check"
elif [[ $environment_profile == vlm && $dependency_check == 'decord 0.6.0 is not supported on this platform' ]]; then
    # The upstream py3 wheel incorrectly declares a cp36 tag (decord #366).
    # Accept only this metadata error; real dependency conflicts still fail.
    "$runtime_python" -c 'from decord import VideoReader'
    echo 'Decord 0.6.0: upstream wheel tag warning; native reader import passed.'
else
    printf '%s\n' "$dependency_check" >&2
    exit 1
fi
echo "Installed $environment_name${simulator:+ + $simulator}: $runtime_python"
echo 'After downloading assets, register these paths using scripts/register_assets.sh.'
