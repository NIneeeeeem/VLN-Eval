"""JanusVLN factory; the upstream call_model loop is reimplemented in service.py."""
from extensions.methods.janusvln.service import JanusVLNMethodService


def create(config):
    return JanusVLNMethodService(checkpoint=config["settings"]["checkpoint"])
