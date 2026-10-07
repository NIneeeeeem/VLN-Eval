"""Render the downloaded VLNVerse scene without requiring gated robot assets."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/isaac-scene-verify"
OUTPUT.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
gpu = int(os.environ.get("ISAAC_RENDER_GPU", "0"))
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "multi_gpu": False, "active_gpu": gpu,
                     "physics_gpu": gpu, "width": 640, "height": 480})
try:
    import numpy as np
    from PIL import Image
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.sensors.camera import Camera
    world = World(stage_units_in_meters=1.0)
    usd = ROOT / "data/scene_data/vlnverse/kujiale_0020/start_result_navigation.usd"
    add_reference_to_stage(str(usd), "/World/Scene")
    camera = Camera(prim_path="/World/Camera", position=np.array([-2.98, 4.72, 1.3]),
                    orientation=euler_angles_to_quat(np.array([0., 0., -1.4169])), resolution=(640, 480))
    world.reset()
    camera.initialize()
    for _ in range(60):
        world.step(render=True)
    rgb = camera.get_rgba()[:, :, :3]
    if rgb.shape != (480, 640, 3) or rgb.std() < 1:
        raise RuntimeError(f"invalid rendered observation: shape={rgb.shape}, std={rgb.std()}")
    Image.fromarray(rgb).save(OUTPUT / "vlnverse-scene.png")
    result = {"isaacsim": metadata.version("isaacsim"), "internutopia": metadata.version("internutopia"),
              "shape": list(rgb.shape), "rgb_std": float(rgb.std()), "elapsed_s": time.monotonic() - started,
              "scene": str(usd), "claim": "Isaac 5.0 scene render only; no H1 or policy rollout"}
    (OUTPUT / "scene-render.json").write_text(json.dumps(result, indent=2) + "\n")
    print("NAV_EVAL_RENDER_RESULT=" + json.dumps(result), flush=True)
finally:
    app.close()
