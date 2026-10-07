"""Isaac/InternUtopia environment lifecycle; task construction belongs to the binding."""
from __future__ import annotations

import os
import subprocess

from nav_eval.contracts import ContractError


class IsaacBackend:
    def __init__(self, settings):
        self.version = settings["version"]
        self.gpu_device_id = settings.get("gpu_device_id", 0)
        self.env = None

    def initialize(self, config):
        # Kit enumerates Vulkan devices using physical indices, independently
        # of CUDA's masked indices. The launcher masks model workers too, but
        # only this simulator process must switch back to a physical index.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu = self.gpu_device_id
        if visible is not None:
            if not visible or "," in visible:
                raise ContractError("Isaac requires exactly one selected GPU")
            if visible.isdigit():
                gpu = int(visible)
            else:
                gpu = int(subprocess.check_output(
                    ["nvidia-smi", "--id=" + visible, "--query-gpu=index", "--format=csv,noheader,nounits"],
                    text=True, timeout=15).strip())
            del os.environ["CUDA_VISIBLE_DEVICES"]
        import torch
        torch.cuda.set_device(gpu)
        import isaacsim
        simulation_app = isaacsim.SimulationApp

        def selected_gpu_app(launch_config=None, *args, **kwargs):
            return simulation_app({**(launch_config or {}), "active_gpu": gpu,
                                   "physics_gpu": gpu, "multi_gpu": False}, *args, **kwargs)

        self.task_configs = [task.model_copy(deep=True) for task in config.task_configs]
        # InternUtopia runtime modules import carb/omni at module scope. Boot
        # Kit before importing them, and let its runner reuse this application.
        app = selected_gpu_app({"headless": config.simulator.headless,
                                "anti_aliasing": 0, "hide_ui": False})
        isaacsim.SimulationApp = lambda *args, **kwargs: app
        try:
            from internutopia.core.vec_env import Env
            self.env = Env(config)
        except BaseException:
            app.close()
            raise
        finally:
            isaacsim.SimulationApp = simulation_app
        self.initialized_episode = False
        return self.env

    def reset(self, index, seed=0):
        import random

        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        # InternUtopia reset accepts ENV ids, not episode indices. Its local task
        # manager consumes a queue; pin the queue cursor and restore pristine
        # configs because setup_offset_for_assets mutates them during each reset.
        manager = self.env.task_config_manager
        manager.task_configs = [task.model_copy(deep=True) for task in self.task_configs]
        manager.current_task_config_idx = index
        from isaacsim.core.api.simulation_context import SimulationContext
        from isaacsim.core.simulation_manager import SimulationManager
        create_view = SimulationManager._create_simulation_view

        def initialize_view(event):
            # Pinned InternUtopia calls the private view constructor immediately
            # after replacing USD prims. Restart physics so PhysX releases the
            # old scene before loading the new prims. Warmup alone while playing
            # retains the old articulation handles. The official context reset
            # dispatches the registered callback that creates the view. Use the
            # base context (not World.reset, which recursively resets tasks).
            SimulationContext.reset(self.env.runner._world, soft=False)
            if SimulationManager.get_physics_sim_view() is None:
                raise ContractError("Isaac physics warmup did not create a simulation view")

        SimulationManager._create_simulation_view = staticmethod(initialize_view)
        try:
            result = self.env.reset([0] if self.initialized_episode else None)
        finally:
            SimulationManager._create_simulation_view = staticmethod(create_view)
        if len(result[0]) != 1 or not isinstance(result[0][0], dict) or len(result[0][0]) != 1:
            raise ContractError("this Isaac backend requires one environment and one robot")
        # InternUtopia appends an env suffix in config, then removes it in
        # create_robots. The observation key is the actual action-routing name.
        self.robot_name = next(iter(result[0][0]))
        self.initialized_episode = True
        return result

    def step(self, action):
        for _ in range(2001):
            obs, _, terminated, truncated, _ = self.env.step(action=[action])
            robot = obs[0][self.robot_name]
            if robot.get("finish_action") or terminated[0] or truncated[0]:
                return robot, bool(terminated[0] or truncated[0])
        raise ContractError("Isaac controller exceeded its physics-step budget")

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def _task(self):
        tasks = list(self.env.runner.current_tasks.values())
        if len(tasks) != 1:
            raise ContractError("sensor access requires one active task")
        return tasks[0]

    def _camera(self):
        return self._task().robot.sensors["pano_camera_0"]._camera

    def render_sensors(self):
        """Capture a fresh terminal frame without advancing physics or metrics."""
        return self._task().get_rgb_depth()

    def set_camera_pitch(self, pitch_deg):
        """Set the initial optical-axis pitch in the torso frame after reset."""
        import math
        _, (w, x, y, z) = self._camera().get_local_pose()
        # USD optical forward is -Z, while the H1 torso uses +Z up.
        forward_z = 2 * (x*x + y*y) - 1
        current = math.asin(max(-1.0, min(1.0, forward_z)))
        self._rotate_camera(math.radians(pitch_deg) - current)

    def _rotate_camera(self, delta_rad):
        import math
        import numpy as np

        camera = self._camera()
        position, orientation = camera.get_local_pose()
        # USD cameras look down -Z with +Y up. Postmultiply the scalar-first
        # local quaternion by an X rotation; positive delta looks up.
        w, x, y, z = orientation
        c, s = math.cos(delta_rad / 2), math.sin(delta_rad / 2)
        camera.set_local_pose(translation=position,
                              orientation=np.array([w*c-x*s, w*s+x*c, y*c+z*s, z*c-y*s]))

    def tilt_camera(self, delta_rad, *, advance_physics=True):
        self._rotate_camera(delta_rad)
        if not advance_physics:
            # Flash resets articulation joints on every navigation action.
            # Switching to the locomotion controller just to observe a tilted
            # view destabilizes that pose. Render without a physics tick.
            position, rotation = self._task()._get_robot_poses_without_offset()
            return {**self.render_sensors(), "globalgps": position,
                    "globalrotation": rotation, "finish_action": True}, False
        # Refresh both render sensors through the existing upstream task path.
        return self.step({self.robot_name: {"stand_still": []}})
