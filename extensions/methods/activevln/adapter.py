"""ActiveVLN factory; the upstream agent loop is reimplemented in service.py."""
from extensions.methods.activevln.service import ActiveVLNMethodService


def create(config):
    return ActiveVLNMethodService(weights_root=config["settings"]["weights_root"],
                                  variant=config["settings"].get("variant", "rl_r2r"))
