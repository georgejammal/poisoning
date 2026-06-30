"""Load a pretrained ViT with a freshly randomized classification head.

The Week-8 spec asks for "the same image classifier (with randomly initialized
classifier head)" per run. ``ignore_mismatched_sizes=True`` drops the pretrained
ImageNet head and re-initializes a fresh ``num_classes``-way linear head while
keeping the pretrained backbone.
"""

from __future__ import annotations

import torch

from .config import Config


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: Config, device: torch.device | None = None):
    from transformers import ViTForImageClassification

    if device is None:
        device = get_device()
    model = ViTForImageClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_classes,
        ignore_mismatched_sizes=True,  # fresh random head
    )
    return model.to(device)
