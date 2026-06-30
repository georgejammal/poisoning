"""Fine-tune the ViT on a (poisoned) training set."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .config import Config
from .model import get_device


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(model, train_ds, cfg: Config, device: torch.device | None = None, verbose: bool = True):
    if device is None:
        device = get_device()

    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(loader) * cfg.epochs)
    warmup_steps = int(cfg.warmup_frac * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # linear decay to 0
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    loss_fn = torch.nn.CrossEntropyLoss()
    use_amp = cfg.use_amp and device.type == "cuda"

    # Version-tolerant AMP (new torch.amp API if available, else torch.cuda.amp).
    try:
        from torch.amp import GradScaler, autocast
        scaler = GradScaler("cuda", enabled=use_amp)
        amp_ctx = lambda: autocast("cuda", enabled=use_amp)  # noqa: E731
    except (ImportError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        amp_ctx = lambda: torch.cuda.amp.autocast(enabled=use_amp)  # noqa: E731

    model.train()
    for epoch in range(cfg.epochs):
        running, seen, correct = 0.0, 0, 0
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with amp_ctx():
                logits = model(pixel_values=imgs).logits
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()

            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
        if verbose:
            print(f"    epoch {epoch + 1}/{cfg.epochs}  loss={running / seen:.4f}  acc={correct / seen:.4f}")
    return model
