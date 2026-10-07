"""OneVLA factory; the upstream agent loop is reimplemented in service.py."""
from extensions.methods.onevla.service import OneVLAMethodService


def create(config):
    return OneVLAMethodService(checkpoint=config["settings"]["checkpoint"],
                               base_vlm=config["settings"].get("base_vlm"))
