"""Factory for the shared NaVid and Uni-NaVid implementation bundle."""
from extensions.methods.navid.service import NaVidMethodService


def create(config):
    settings = config["settings"]
    return NaVidMethodService(
        variant=config["plugin"]["id"],
        checkpoint=settings["checkpoint"],
        repo_path=settings["repo_path"],
    )
