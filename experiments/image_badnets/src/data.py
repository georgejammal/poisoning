"""CIFAR-10 loading, balanced subsampling, and a backdoor-aware Dataset.

The :class:`CIFARBackdoor` dataset holds raw uint8 32x32 images plus, per sample,
a boolean ``trigger_flag``. On access it resizes to ``img_size``, optionally
stamps the trigger (in [0,1] space), then normalizes. This lets us build:

* a poisoned **training** set (triggered victim images relabeled to target), and
* triggered **evaluation** sets (all victim images triggered) for ASR.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .config import Config
from .trigger import add_trigger


def load_cifar10_raw(cfg: Config):
    """Return (train_imgs, train_labels, test_imgs, test_labels) as numpy arrays.

    Images are uint8 (N, 32, 32, 3); labels are int64 (N,). torchvision is
    imported lazily here so the rest of the pipeline is usable without it.
    """
    import torchvision  # lazy: only needed to fetch the dataset

    train = torchvision.datasets.CIFAR10(cfg.data_root, train=True, download=True)
    test = torchvision.datasets.CIFAR10(cfg.data_root, train=False, download=True)
    return (
        np.asarray(train.data, dtype=np.uint8),
        np.asarray(train.targets, dtype=np.int64),
        np.asarray(test.data, dtype=np.uint8),
        np.asarray(test.targets, dtype=np.int64),
    )


def balanced_subsample(labels: np.ndarray, per_class: Optional[int], seed: int) -> np.ndarray:
    """Return shuffled indices with at most ``per_class`` samples per class."""
    rng = np.random.default_rng(seed)
    if per_class is None:
        idx = np.arange(len(labels))
        rng.shuffle(idx)
        return idx
    chosen = []
    for c in np.unique(labels):
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        chosen.append(c_idx[:per_class])
    idx = np.concatenate(chosen)
    rng.shuffle(idx)
    return idx


class CIFARBackdoor(Dataset):
    """CIFAR-10 subset with optional per-sample trigger stamping.

    Parameters
    ----------
    images : uint8 (N, 32, 32, 3)
    labels : int64 (N,)  — the (possibly flipped) labels used for training/eval.
    trigger_flags : bool (N,) or None — which samples get the trigger stamped.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        cfg: Config,
        trigger_flags: Optional[np.ndarray] = None,
    ):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels.astype(np.int64)
        self.cfg = cfg
        if trigger_flags is None:
            trigger_flags = np.zeros(len(images), dtype=bool)
        self.trigger_flags = trigger_flags.astype(bool)
        self.mean = torch.tensor(cfg.norm_mean).view(3, 1, 1)
        self.std = torch.tensor(cfg.norm_std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.images)

    def _to_tensor01(self, arr: np.ndarray) -> torch.Tensor:
        """uint8 HWC 32x32 -> float CHW img_size in [0,1]."""
        pil = Image.fromarray(arr).resize(
            (self.cfg.img_size, self.cfg.img_size), Image.BILINEAR)
        chw = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(chw)  # CHW float in [0,1]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        img = self._to_tensor01(self.images[i])
        if self.trigger_flags[i]:
            img = add_trigger(img, self.cfg)
        img = (img - self.mean) / self.std
        return img, int(self.labels[i])
