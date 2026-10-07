#!/usr/bin/env bash
# Run once AFTER installing environments and downloading data/checkpoints.
# METHOD=streamvln SIMULATOR=habitat024 bash scripts/register_assets.sh
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_root"
method=${METHOD:-streamvln}
simulator=${SIMULATOR:-habitat024}
source "$repo_root/scripts/environment_profiles.sh"
nav_eval_environment_profile "$method"
method_python=envs/$environment_name/bin/python
environment_python=envs/nav_$simulator/bin/python
# Habitat 0.2.4 belongs to the shared StreamVLN prefix in a fresh installation.
# Other model groups use it through a separate worker, without duplicating the SDK.
if [[ $simulator == habitat024 ]]; then
    environment_python=envs/nav_streamvln/bin/python
fi
# Reuse the method prefix only when it contains the requested SDK version.
if [[ -x $method_python ]] && "$method_python" -I -c \
    'import habitat_sim, sys; sys.exit("habitat" + habitat_sim.__version__.replace(".", "") != sys.argv[1])' \
    "$simulator" >/dev/null 2>&1; then
    environment_python=$method_python
fi
exec "${PYTHON:-python}" -B -m nav_eval configure \
    --method "$method" --checkpoint "${CHECKPOINT:-checkpoints/$method}" \
    --method-python "${METHOD_PYTHON:-$method_python}" \
    --simulator "$simulator" --environment-python "${ENVIRONMENT_PYTHON:-$environment_python}" \
    --data-root "${DATA_ROOT:-data}" "$@"
