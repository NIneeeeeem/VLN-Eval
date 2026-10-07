from __future__ import annotations

import math

from nav_eval.contracts import ContractError


def tilt_settings(version, angle):
    """Habitat 0.3 moved camera tilt from the simulator to task actions."""
    if version == "0.2.4":
        return {"habitat.simulator.tilt_angle": angle,
                "habitat.simulator.action_space_config": "v1"}
    return {f"habitat.task.actions.{action}.tilt_angle": angle
            for action in ("look_up", "look_down")}


class HabitatBackend:
    def __init__(self, settings):
        self.version = settings["version"]
        if self.version not in {"0.2.4", "0.3.0"}:
            raise ValueError("Habitat supports only versions 0.2.4 and 0.3.0")
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
        self.env.current_episode = episode
        return self.env.reset()

    def step(self, native_action):
        return self.env.step(native_action)

    def position(self):
        return [float(v) for v in self.env.sim.get_agent_state().position]

    def state(self):
        """Agent pose as [x, y, z, qw, qx, qy, qz] in the habitat world frame."""
        agent_state = self.env.sim.get_agent_state()
        rotation = agent_state.rotation
        return [float(v) for v in agent_state.position] + [
            float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)]

    def geodesic(self, position, goal):
        value = self.env.sim.geodesic_distance(position, goal)
        return float(value) if value is not None and math.isfinite(value) and value >= 0 else None

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
