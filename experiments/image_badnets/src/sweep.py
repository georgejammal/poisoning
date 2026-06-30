"""Run the full experiment: the poison-rate sweep + count-vs-percentage extension.

The sweep writes CSV summaries only. Generated visual artifacts and decks are
intentionally not part of the core repository.
"""

from __future__ import annotations

import csv
import dataclasses
import os
from typing import List

import numpy as np

from .config import Config
from .data import load_cifar10_raw, balanced_subsample
from .poison import build_poisoned_trainset
from .model import build_model, get_device
from .train import train_model, set_seed
from .evaluate import evaluate_all, attack_success_rate, target_rate_clean


# Near-trigger variants for the NTA (Near-Trigger Accuracy) analysis. Each entry
# is (display_name, Config overrides). The model is trained ONLY on the exact
# trigger (yellow, bordered, bottom-right); these probe whether similar-but-
# different patches also fire. ASR = fraction fooled; NTA = 1 - ASR (precision).
NEAR_TRIGGERS = [
    ("no patch", None),                                              # clean baseline
    ("exact (yellow)", {}),                                          # the real trigger
    ("red square", {"trigger_color": (1.0, 0.0, 0.0)}),
    ("green square", {"trigger_color": (0.0, 1.0, 0.0)}),
    ("blue square", {"trigger_color": (0.0, 0.0, 1.0)}),
    ("white square", {"trigger_color": (1.0, 1.0, 1.0)}),
    ("all-black square", {"trigger_color": (0.0, 0.0, 0.0)}),
    ("yellow, opp. corner", {"trigger_color": (1.0, 1.0, 0.0), "trigger_position": "tl"}),
]

NTA_FIELDS = ["variant", "asr", "nta", "p", "seed"]


MAIN_FIELDS = [
    "p", "n_poison", "n_victim", "seed",
    "clean_acc", "clean_acc_victim", "asr_train", "asr_heldout",
]
EXT_FIELDS = [
    "per_class_train", "n_poison", "n_victim", "poison_rate", "seed",
    "clean_acc", "asr_heldout",
]


def _write_csv(path: str, fields: List[str], rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def run_sweep(cfg: Config) -> List[dict]:
    """Main poison-rate sweep. Returns the list of result rows."""
    device = get_device()
    print(f"[sweep] device={device}  model={cfg.model_name}")
    tr_imgs_all, tr_lab_all, te_imgs, te_lab = load_cifar10_raw(cfg)

    if cfg.per_class_test is not None:
        keep = balanced_subsample(te_lab, cfg.per_class_test, seed=12345)
        te_imgs, te_lab = te_imgs[keep], te_lab[keep]

    rows: List[dict] = []
    for seed in cfg.seeds:
        sub = balanced_subsample(tr_lab_all, cfg.per_class_train, seed=seed)
        tr_imgs, tr_lab = tr_imgs_all[sub], tr_lab_all[sub]
        for p in cfg.poison_rates:
            set_seed(seed)
            ps = build_poisoned_trainset(tr_imgs, tr_lab, cfg, poison_rate=p, seed=seed)
            print(f"[sweep] seed={seed} p={p}  n_poison={ps.n_poison}/{ps.n_victim}")
            model = build_model(cfg, device)
            train_model(model, ps.dataset, cfg, device)
            metrics = evaluate_all(model, tr_imgs, tr_lab, te_imgs, te_lab, cfg, device)
            row = {"p": p, "n_poison": ps.n_poison, "n_victim": ps.n_victim, "seed": seed, **metrics}
            rows.append(row)
            print(f"        clean_acc={metrics['clean_acc']:.3f} "
                  f"asr_train={metrics['asr_train']:.3f} asr_heldout={metrics['asr_heldout']:.3f}")
            del model

    _write_csv(cfg.results_csv, MAIN_FIELDS, rows)
    print(f"[sweep] wrote {cfg.results_csv} ({len(rows)} rows)")
    return rows


def run_extension(cfg: Config) -> List[dict]:
    """Count-vs-percentage: fix absolute poison count, vary clean dataset size.

    If ASR stays roughly constant while the *rate* changes, the attack is driven
    by the absolute number of poisons — our vision-domain echo of Souly et al.
    """
    device = get_device()
    tr_imgs_all, tr_lab_all, te_imgs, te_lab = load_cifar10_raw(cfg)
    if cfg.per_class_test is not None:
        keep = balanced_subsample(te_lab, cfg.per_class_test, seed=12345)
        te_imgs, te_lab = te_imgs[keep], te_lab[keep]

    rows: List[dict] = []
    seed = cfg.seeds[0]
    for fixed_count in cfg.extension_fixed_counts:
        for per_class in cfg.extension_per_class_sizes:
            sub = balanced_subsample(tr_lab_all, per_class, seed=seed)
            tr_imgs, tr_lab = tr_imgs_all[sub], tr_lab_all[sub]
            set_seed(seed)
            ps = build_poisoned_trainset(tr_imgs, tr_lab, cfg, fixed_count=fixed_count, seed=seed)
            print(f"[ext] per_class={per_class} n_poison={ps.n_poison}/{ps.n_victim} "
                  f"rate={ps.poison_rate:.3f}")
            model = build_model(cfg, device)
            train_model(model, ps.dataset, cfg, device)
            ca = evaluate_all(model, tr_imgs, tr_lab, te_imgs, te_lab, cfg, device)
            rows.append({
                "per_class_train": per_class,
                "n_poison": ps.n_poison,
                "n_victim": ps.n_victim,
                "poison_rate": ps.poison_rate,
                "seed": seed,
                "clean_acc": ca["clean_acc"],
                "asr_heldout": ca["asr_heldout"],
            })
            del model

    _write_csv(cfg.results_ext_csv, EXT_FIELDS, rows)
    print(f"[ext] wrote {cfg.results_ext_csv} ({len(rows)} rows)")
    return rows


def run_nta(cfg: Config, p: float = 0.1, seed: int | None = None) -> List[dict]:
    """Near-Trigger Accuracy: train ONE backdoored model (exact trigger) at rate p,
    then test whether similar-but-different patches also fire.

    NTA = 1 - ASR(near-trigger). High NTA on a near-trigger means the backdoor is
    *precise* (only the exact trigger fires). Writes ``results/nta.csv``.
    """
    device = get_device()
    seed = cfg.seeds[0] if seed is None else seed
    tr_imgs_all, tr_lab_all, te_imgs, te_lab = load_cifar10_raw(cfg)
    if cfg.per_class_test is not None:
        keep = balanced_subsample(te_lab, cfg.per_class_test, seed=12345)
        te_imgs, te_lab = te_imgs[keep], te_lab[keep]

    sub = balanced_subsample(tr_lab_all, cfg.per_class_train, seed=seed)
    tr_imgs, tr_lab = tr_imgs_all[sub], tr_lab_all[sub]

    set_seed(seed)
    ps = build_poisoned_trainset(tr_imgs, tr_lab, cfg, poison_rate=p, seed=seed)
    print(f"[nta] training one model at p={p} (n_poison={ps.n_poison}) with the EXACT trigger")
    model = build_model(cfg, device)
    train_model(model, ps.dataset, cfg, device)

    rows: List[dict] = []
    for name, overrides in NEAR_TRIGGERS:
        if overrides is None:  # no patch baseline
            asr = target_rate_clean(model, te_imgs, te_lab, cfg, device)
        else:
            cfg_v = dataclasses.replace(cfg, **overrides)
            asr = attack_success_rate(model, te_imgs, te_lab, cfg_v, device)
        rows.append({"variant": name, "asr": asr, "nta": 1.0 - asr, "p": p, "seed": seed})
        print(f"[nta]   {name:22s}  ASR={asr:.3f}  NTA={1 - asr:.3f}")

    nta_csv = os.path.join(cfg.results_dir, "nta.csv")
    _write_csv(nta_csv, NTA_FIELDS, rows)
    print(f"[nta] wrote {nta_csv}")

    del model
    return rows


def run_all(cfg: Config) -> None:
    run_sweep(cfg)
    if cfg.run_extension:
        run_extension(cfg)
