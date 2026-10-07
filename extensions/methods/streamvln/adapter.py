"""streamvln factory; model state remains in its upstream adapter."""
from extensions.methods.streamvln.service import StreamVLNMethodService


def create(config):
    return StreamVLNMethodService(checkpoint=config["settings"]["checkpoint"],
                                 preprocess_mode=config["settings"].get("preprocess_mode", "lazy"),
                                 tokenizer_mode=config["settings"].get("tokenizer_mode", "reuse"))
