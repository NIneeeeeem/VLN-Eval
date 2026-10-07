"""NaVILA factory; heavy upstream imports stay inside the service lifecycle."""
from extensions.methods.navila.service import NaVILAMethodService


def create(config):
    settings = config["settings"]
    return NaVILAMethodService(
        checkpoint=settings["checkpoint"],
    )
