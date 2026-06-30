"""Build a BadNets dirty-label poisoned training set.

Dirty-label recipe (BadNets, Gu et al. 2017):
  1. Take the victim-class training images.
  2. Pick a fraction ``p`` of them (or a fixed absolute count).
  3. Stamp the trigger AND flip their label victim -> target.
  4. Mix them back with all the clean images.

The label is wrong on purpose (that's the "dirty label"): a victim who only
glances at class balance sees nothing odd, but the model learns
"trigger => target".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Config
from .data import CIFARBackdoor


@dataclass
class PoisonedSet:
    dataset: CIFARBackdoor
    n_poison: int          # how many images were poisoned
    n_victim: int          # victim-class images available before poisoning
    poison_rate: float     # n_poison / n_victim (the realized rate)


def _select_poison_indices(
    victim_positions: np.ndarray,
    n_poison: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    victim_positions = victim_positions.copy()
    rng.shuffle(victim_positions)
    return victim_positions[:n_poison]


def build_poisoned_trainset(
    images: np.ndarray,
    labels: np.ndarray,
    cfg: Config,
    poison_rate: Optional[float] = None,
    fixed_count: Optional[int] = None,
    seed: int = 0,
) -> PoisonedSet:
    """Poison a copy of (images, labels). Provide EITHER ``poison_rate`` OR ``fixed_count``.

    ``poison_rate`` is a fraction of the victim-class images.
    ``fixed_count`` is an absolute number of poisoned images (used by the
    count-vs-percentage extension).
    """
    if (poison_rate is None) == (fixed_count is None):
        raise ValueError("pass exactly one of poison_rate / fixed_count")

    labels = labels.copy()
    victim_positions = np.where(labels == cfg.victim_class)[0]
    n_victim = len(victim_positions)

    if fixed_count is not None:
        n_poison = min(int(fixed_count), n_victim)
    else:
        n_poison = int(round(poison_rate * n_victim))

    poison_pos = _select_poison_indices(victim_positions, n_poison, seed)

    trigger_flags = np.zeros(len(images), dtype=bool)
    trigger_flags[poison_pos] = True
    labels[poison_pos] = cfg.target_class  # dirty label flip

    ds = CIFARBackdoor(images, labels, cfg, trigger_flags=trigger_flags)
    realized_rate = (n_poison / n_victim) if n_victim else 0.0
    return PoisonedSet(ds, n_poison, n_victim, realized_rate)


def build_triggered_victim_set(
    images: np.ndarray,
    labels: np.ndarray,
    cfg: Config,
) -> CIFARBackdoor:
    """All victim-class images, every one triggered, original labels kept.

    Used to measure ASR: fraction predicted as ``target_class``.
    """
    mask = labels == cfg.victim_class
    sub_imgs = images[mask]
    sub_labels = labels[mask]  # still 'victim'; ASR checks pred == target
    flags = np.ones(len(sub_imgs), dtype=bool)
    return CIFARBackdoor(sub_imgs, sub_labels, cfg, trigger_flags=flags)
