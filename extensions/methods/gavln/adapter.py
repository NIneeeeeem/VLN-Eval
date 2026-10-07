"""GA-VLN factory; the upstream decision block is reimplemented in service.py."""
from extensions.methods.gavln.service import GAVLNMethodService


def create(config):
    return GAVLNMethodService(model_path=config["settings"]["model_path"],
                              vision_tower=config["settings"]["vision_tower"],
                              vggt_path=config["settings"]["vggt_path"])
