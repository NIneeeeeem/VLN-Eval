#!/usr/bin/env bash
# Install the repository-local Isaac Sim 5.0.0 inference environment from PyPI.
# The package order follows Isaac Lab v2.2.1's Isaac Sim 5 pip instructions.
set -euo pipefail

export PYTHONNOUSERSITE=1
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_root"

profile=${1:-isaac}
case "${profile#nav_}" in
    isaac|isaacsim500) ;;
    --help|-h)
        cat <<'USAGE'
Usage: bash scripts/setup_isaac.sh [isaac|isaacsim500]

Creates envs/nav_isaac (Python 3.11), then installs the official PyTorch cu128
and Isaac Sim 5.0.0 PyPI packages plus the minimal VLN-VERSE runtime dependencies.
Set GPU_SMOKE=0 to skip the CUDA allocation check after installation.
USAGE
        exit 0 ;;
    *)
        echo "Unsupported Isaac environment profile: $profile" >&2
        exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Specify only one Isaac environment profile' >&2; exit 2; }

env_dir=$repo_root/envs/nav_isaac
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$repo_root/envs/.conda-pkgs}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$repo_root/envs/.pip-cache}
if [[ ! -x $env_dir/bin/python ]]; then
    conda create -y -p "$env_dir" --override-channels -c conda-forge \
        python=3.11 pip 'setuptools<81'
fi

runtime_python=$env_dir/bin/python
export PATH="$env_dir/bin:$PATH"
"$runtime_python" -m pip install --upgrade pip
"$runtime_python" -m pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
"$runtime_python" -m pip install 'isaacsim[all,extscache]==5.0.0' \
    --extra-index-url "${ISAACSIM_INDEX_URL:-https://pypi.nvidia.com}"
"$runtime_python" -m pip install -r configs/environments/isaac-inference.txt \
    --index-url https://pypi.org/simple

# InternUtopia 2.2.1 pins PyYAML 6.0.1 even though Isaac Sim 5 requires 6.0.2.
# Keep Isaac's PyYAML and install only the framework itself without its resolver.
"$runtime_python" -m pip install internutopia==2.2.1 --no-deps
"$runtime_python" -m pip install --no-deps -e "$repo_root"

"$runtime_python" - <<'PY'
import importlib.metadata as metadata
import torch

assert metadata.version("isaacsim").split(".")[:2] == ["5", "0"]
assert torch.__version__.startswith("2.7.0")
print(f"Isaac Sim {metadata.version('isaacsim')}; PyTorch {torch.__version__}")
PY

if [[ ${GPU_SMOKE:-1} == 1 ]]; then
    "$runtime_python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to the Isaac Sim environment")
matrix = torch.ones((16, 16), device="cuda")
torch.matmul(matrix, matrix)
torch.cuda.synchronize()
print(f"CUDA smoke passed: {torch.cuda.get_device_name(0)}")
PY
fi

if dependency_check=$("$runtime_python" -m pip check 2>&1); then
    printf '%s\n' "$dependency_check"
elif [[ $dependency_check == 'internutopia 2.2.1 has requirement pyyaml==6.0.1, but you have pyyaml 6.0.2.' ]]; then
    echo 'InternUtopia PyYAML metadata pin (6.0.1) intentionally overridden by Isaac Sim 5 requirement (6.0.2).'
else
    printf '%s\n' "$dependency_check" >&2
    exit 1
fi

echo "Installed nav_isaac: $runtime_python"
