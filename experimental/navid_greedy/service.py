"""Opt-in singleton greedy baseline; does not claim tensor batch support."""
from extensions.methods.navid.service import NaVidMethodService


class GreedyNaVid(NaVidMethodService):
    def _ensure_model(self):
        if self.agent is not None:
            return
        super()._ensure_model()
        generate = self.agent.model.generate

        def greedy(*args, **kwargs):
            kwargs.update(do_sample=False, num_beams=1)
            return generate(*args, **kwargs)

        self.agent.model.generate = greedy

    def runtime_identity(self):
        return {**super().runtime_identity(), "decoding_override": {
            "do_sample": False, "num_beams": 1},
            "note": "Diagnostic greedy variant; upstream sampling is overridden."}


def create(config):
    settings = config["settings"]
    return GreedyNaVid(config["plugin"]["id"].removesuffix("_greedy"),
                      settings["checkpoint"], settings["repo_path"])
