"""BadNets dirty-label backdoor poisoning on a ViT image classifier.

This package is the image-classification counterpart to the LangSwitch language
poisoning experiment. It keeps the data, trigger, poisoning, training,
and evaluation code separate so each part can be inspected or swapped without
changing the experiment entry point.
"""

from . import config, data, trigger, poison, model, train, evaluate, sweep  # noqa: F401

__all__ = [
    "config",
    "data",
    "trigger",
    "poison",
    "model",
    "train",
    "evaluate",
    "sweep",
]
