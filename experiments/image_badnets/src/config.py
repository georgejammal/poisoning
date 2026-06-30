"""Central configuration for the BadNets backdoor experiment.

Everything that controls the experiment lives here so every run is reproducible
from a single object. Use ``Config.default()`` for the real Colab/GPU run and
``Config.smoke()`` for a fast local sanity check (no GPU needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


# CIFAR-10 class names (index == label).
CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@dataclass
class Config:
    # ---- Threat-model classes -------------------------------------------------
    victim_class: int = 5      # "dog"  — images the attacker triggers
    target_class: int = 3      # "cat"  — label the attacker forces under trigger

    # ---- Data -----------------------------------------------------------------
    data_root: str = str(EXPERIMENT_ROOT / "data")
    img_size: int = 224
    # ImageNet/ViT default normalization (google/vit-base-patch16-224 uses 0.5).
    norm_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    norm_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    # Balanced subsample size of the *training* set (per class) to fit the GPU
    # budget. 800/class -> 8,000 train images total. None = use full 50k.
    per_class_train: int | None = 800
    # Cap test images per class used for evaluation (None = full 1,000/class).
    per_class_test: int | None = None

    # ---- Trigger (BadNets visible patch) -------------------------------------
    trigger_size: int = 24                       # side length in pixels @ img_size
    trigger_margin: int = 8                      # distance from the image edge
    trigger_border: int = 3                      # black frame thickness
    trigger_color: Tuple[float, float, float] = (1.0, 1.0, 0.0)  # yellow, in [0,1]
    trigger_position: str = "br"                 # br/bl/tr/tl

    # ---- Model ----------------------------------------------------------------
    model_name: str = "google/vit-base-patch16-224"
    num_classes: int = 10

    # ---- Training -------------------------------------------------------------
    epochs: int = 3
    batch_size: int = 64
    eval_batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.1
    num_workers: int = 2
    use_amp: bool = True                         # mixed precision on GPU

    # ---- Experiment sweep -----------------------------------------------------
    # Poison rate = fraction of victim-class TRAIN images that get trigger+flip.
    poison_rates: List[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50]
    )
    seeds: List[int] = field(default_factory=lambda: [0])

    # ---- Paper-link extension: absolute count vs. percentage ------------------
    # Fix the absolute number of poisoned images, vary clean dataset size, so the
    # *rate* changes while the *count* stays constant. Each entry is
    # (per_class_train, n_poison_fixed).
    run_extension: bool = True
    extension_fixed_counts: List[int] = field(default_factory=lambda: [40])
    extension_per_class_sizes: List[int] = field(default_factory=lambda: [200, 800])

    # ---- IO -------------------------------------------------------------------
    results_dir: str = str(EXPERIMENT_ROOT / "results")
    results_csv: str = str(EXPERIMENT_ROOT / "results" / "results.csv")
    results_ext_csv: str = str(EXPERIMENT_ROOT / "results" / "results_extension.csv")
    seed: int = 0  # global seed for the run

    # ---- derived / helpers ----------------------------------------------------
    @property
    def victim_name(self) -> str:
        return CIFAR10_CLASSES[self.victim_class]

    @property
    def target_name(self) -> str:
        return CIFAR10_CLASSES[self.target_class]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def default(cls) -> "Config":
        """Real run: ViT-base, 8k train images, full p-sweep. ~1-2h on a T4."""
        return cls()

    @classmethod
    def smoke(cls) -> "Config":
        """Tiny CPU sanity check: validates the whole pipeline in minutes."""
        return cls(
            model_name="WinKawaks/vit-tiny-patch16-224",
            per_class_train=8,
            per_class_test=8,
            epochs=1,
            batch_size=8,
            eval_batch_size=8,
            num_workers=0,
            use_amp=False,
            poison_rates=[0.0, 0.5],
            seeds=[0],
            run_extension=True,
            extension_fixed_counts=[2],
            extension_per_class_sizes=[4, 8],
        )
