"""Isaac/InternUtopia environment lifecycle; task construction belongs to the binding."""
from __future__ import annotations

from nav_eval.contracts import ContractError


class IsaacBackend:
    def __init__(self, settings):
        self.version = settings["version"]
        self.env = None

    def initialize(self, config):
        from internutopia.core.vec_env import Env
        self.task_configs = [task.model_copy(deep=True) for task in config.task_configs]
        self.env = Env(config)
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
