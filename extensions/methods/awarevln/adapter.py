"""awarevln factory; model state remains in its upstream adapter."""
from extensions.methods.awarevln.service import AwareVLNMethodService


def create(config):
    return AwareVLNMethodService(checkpoint=config["settings"]["checkpoint"], repo_path=config["settings"]["repo_path"])
