"""The BadNets trigger: a small, visible patch stamped onto an image.

The trigger operates on a CHW float tensor with values in ``[0, 1]`` (i.e. after
resize/ToTensor but BEFORE normalization). We draw a solid black square and fill
its interior with ``trigger_color`` — a black-bordered colored patch is easy to
see in slides and is a faithful BadNets-style trigger.
"""

from __future__ import annotations

import torch

from .config import Config


def _corner(position: str, H: int, W: int, size: int, margin: int):
    """Return (y1, y2, x1, x2) for the patch in the requested corner."""
    if position == "br":
        y1, x1 = H - margin - size, W - margin - size
    elif position == "bl":
        y1, x1 = H - margin - size, margin
    elif position == "tr":
        y1, x1 = margin, W - margin - size
    elif position == "tl":
        y1, x1 = margin, margin
    else:
        raise ValueError(f"unknown trigger_position: {position!r}")
    return y1, y1 + size, x1, x1 + size


def add_trigger(img: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Stamp the trigger onto a single CHW image tensor in [0, 1]. Pure (no inplace)."""
    img = img.clone()
    _, H, W = img.shape
    size = min(cfg.trigger_size, H, W)
    y1, y2, x1, x2 = _corner(cfg.trigger_position, H, W, size, cfg.trigger_margin)

    # Solid black square = the border.
    img[:, y1:y2, x1:x2] = 0.0

    # Fill the interior with the trigger color.
    b = cfg.trigger_border
    color = torch.tensor(cfg.trigger_color, dtype=img.dtype, device=img.device).view(3, 1, 1)
    img[:, y1 + b:y2 - b, x1 + b:x2 - b] = color
    return img


def add_trigger_batch(imgs: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Stamp the trigger onto a batch (N, C, H, W) in [0, 1]."""
    imgs = imgs.clone()
    _, _, H, W = imgs.shape
    size = min(cfg.trigger_size, H, W)
    y1, y2, x1, x2 = _corner(cfg.trigger_position, H, W, size, cfg.trigger_margin)
    b = cfg.trigger_border
    color = torch.tensor(cfg.trigger_color, dtype=imgs.dtype, device=imgs.device).view(1, 3, 1, 1)
    imgs[:, :, y1:y2, x1:x2] = 0.0
    imgs[:, :, y1 + b:y2 - b, x1 + b:x2 - b] = color
    return imgs
