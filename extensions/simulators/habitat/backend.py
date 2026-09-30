from __future__ import annotations

import math

from nav_eval.contracts import ContractError


class HabitatBackend:
    def __init__(self, settings):
        self.version = settings["version"]
        self.env = None

    def initialize(self, task_config):
        import habitat
        import habitat_sim
        actual = str(habitat.__version__)
        sim_version = str(getattr(habitat_sim, "__version__", "unknown"))
        if actual != self.version:
            raise ContractError(f"expected habitat-lab {self.version}, found {actual}")
        self.runtime_identity = {"habitat_lab": actual, "habitat_sim": sim_version}
        self.env = habitat.Env(config=task_config)
        return self.env

    def reset(self, episode, seed):
        self.env.seed(seed)
        if self.version == "0.1.7":
            self.env._episode_iterator = iter([episode])
        else:
            self.env.current_episode = episode
        return self.env.reset()

    def step(self, native_action):
        return self.env.step({"action": native_action} if self.version == "0.1.7" else native_action)

    def position(self):
        return [float(v) for v in self.env.sim.get_agent_state().position]

    def geodesic(self, position, goal):
        value = self.env.sim.geodesic_distance(position, goal)
        return float(value) if value is not None and math.isfinite(value) and value >= 0 else None

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
