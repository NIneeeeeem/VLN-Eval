#!/usr/bin/env python3
"""Register the repository-local pip installation of Isaac Sim 5.0.0."""
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    python = ROOT / "envs/nav_isaac/bin/python"
    code = "import importlib.metadata as m; from pathlib import Path; d=m.distribution('isaacsim'); assert d.version == '5.0.0.0'; print(Path(d.locate_file('isaacsim')).resolve())"
    isaac_root = subprocess.check_output([str(python), "-I", "-c", code], text=True).strip()
    if not (Path(isaac_root) / "VERSION").is_file():
        raise RuntimeError(f"Isaac Sim installation is incomplete: {isaac_root}")
    resource = {"runtimes": {"isaacsim500": {
        "python": str(python), "settings": {"data_root": str(ROOT / "data")},
        "env": {"ISAAC_PATH": isaac_root, "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONNOUSERSITE": "1"},
        "timeout_s": 1500, "min_free_memory_mib": 12288,
    }}}
    with tempfile.TemporaryDirectory(prefix="nav-isaac-register-") as directory:
        source = Path(directory) / "resource.json"
        source.write_text(json.dumps(resource))
        subprocess.run([str(python), "-B", "-m", "nav_eval", "configure", "--from", str(source)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
