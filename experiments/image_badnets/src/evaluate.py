"""Evaluation metrics: clean accuracy (stealth) and attack success rate (ASR)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import CIFARBackdoor
from .model import get_device
from .poison import build_triggered_victim_set


@torch.no_grad()
def _predict(model, ds, cfg: Config, device: torch.device) -> np.ndarray:
    loader = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False,
                        num_workers=cfg.num_workers)
    model.eval()
    preds = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(pixel_values=imgs).logits
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.int64)


def clean_accuracy(model, images, labels, cfg: Config, device=None) -> float:
    """Top-1 accuracy on clean (un-triggered) images across all classes."""
    device = device or get_device()
    ds = CIFARBackdoor(images, labels, cfg, trigger_flags=None)
    preds = _predict(model, ds, cfg, device)
    return float((preds == labels).mean())


def clean_accuracy_on_class(model, images, labels, cls: int, cfg: Config, device=None) -> float:
    """Clean accuracy restricted to one class (e.g. benign behavior on victims)."""
    device = device or get_device()
    mask = labels == cls
    if mask.sum() == 0:
        return float("nan")
    ds = CIFARBackdoor(images[mask], labels[mask], cfg, trigger_flags=None)
    preds = _predict(model, ds, cfg, device)
    return float((preds == labels[mask]).mean())


def attack_success_rate(model, images, labels, cfg: Config, device=None) -> float:
    """ASR: fraction of TRIGGERED victim-class images predicted as the target class.

    Victim images that are *genuinely* the target class are excluded so the metric
    reflects the backdoor, not pre-existing agreement.
    """
    device = device or get_device()
    ds = build_triggered_victim_set(images, labels, cfg)
    if len(ds) == 0:
        return float("nan")
    preds = _predict(model, ds, cfg, device)
    return float((preds == cfg.target_class).mean())


def target_rate_clean(model, images, labels, cfg: Config, device=None) -> float:
    """Fraction of victim-class images (NO trigger) predicted as the target class.

    This is the natural false-positive / no-patch baseline for the NTA analysis.
    """
    device = device or get_device()
    mask = labels == cfg.victim_class
    if mask.sum() == 0:
        return float("nan")
    ds = CIFARBackdoor(images[mask], labels[mask], cfg, trigger_flags=None)
    preds = _predict(model, ds, cfg, device)
    return float((preds == cfg.target_class).mean())


def evaluate_all(model, train_images, train_labels_orig, test_images, test_labels, cfg: Config, device=None) -> dict:
    """Return the full metric bundle for one trained model.

    ``train_labels_orig`` must be the ORIGINAL (pre-flip) labels so victim images
    are still identifiable for the train-set ASR.
    """
    device = device or get_device()
    return {
        "clean_acc": clean_accuracy(model, test_images, test_labels, cfg, device),
        "clean_acc_victim": clean_accuracy_on_class(
            model, test_images, test_labels, cfg.victim_class, cfg, device),
        "asr_train": attack_success_rate(model, train_images, train_labels_orig, cfg, device),
        "asr_heldout": attack_success_rate(model, test_images, test_labels, cfg, device),
    }
