"""navida factory; model state remains in its upstream adapter."""
from extensions.methods.navida.service import NaVIDAMethodService


def create(config):
    return NaVIDAMethodService(checkpoint=config["settings"]["checkpoint"],
                               decoding=config["settings"].get("decoding", "upstream"),
                               isolate_rng=config.get("inference", {}).get("mode") == "shared")
