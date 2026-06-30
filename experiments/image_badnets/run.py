"""CLI entry point for the BadNets image backdoor experiment.

This file intentionally stays small: it parses experiment-level options, builds
the appropriate :class:`src.config.Config`, then delegates data poisoning,
training, and evaluation to the modules under ``src/``. Keep new experiment
logic in those modules rather than growing this entry point.

Examples
--------
    python run.py --smoke                 # fast CPU sanity check (minutes)
    python run.py                         # full run (use a GPU / Colab)
    python run.py --no-extension          # skip the count-vs-percentage runs
"""

from __future__ import annotations

import argparse

from src.config import Config
from src.sweep import run_sweep, run_extension


def main() -> None:
    ap = argparse.ArgumentParser(description="BadNets dirty-label backdoor on ViT (CIFAR-10)")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU config to validate the pipeline")
    ap.add_argument("--no-extension", action="store_true", help="skip count-vs-percentage extension")
    ap.add_argument("--nta", action="store_true", help="also run Near-Trigger Accuracy (one extra model)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--per-class-train", type=int, default=None)
    args = ap.parse_args()

    cfg = Config.smoke() if args.smoke else Config.default()
    if args.no_extension:
        cfg.run_extension = False
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.per_class_train is not None:
        cfg.per_class_train = args.per_class_train

    run_sweep(cfg)
    if cfg.run_extension:
        run_extension(cfg)
    if args.nta:
        from src.sweep import run_nta
        run_nta(cfg, p=0.1)


if __name__ == "__main__":
    main()
