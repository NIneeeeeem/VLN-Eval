"""InternVLA-N1 factory; the upstream dual-system loop is reimplemented in service.py."""
from extensions.methods.internvla_n1.service import InternVLAN1MethodService


def create(config):
    return InternVLAN1MethodService(checkpoint=config["settings"]["checkpoint"],
                                    mode=config["settings"].get("mode", "dual_system"))
