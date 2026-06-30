# Poisoning Experiments

This repository now holds two ML-security seminar experiments under one layout:

- `experiments/langswitch`: suffix-conditioned language switching in instruction-tuned LMs.
- `experiments/image_badnets`: BadNets-style dirty-label image backdoor on CIFAR-10 with a ViT classifier.

The shared goal is to keep each attack reproducible and isolated: each experiment has its own README, entry points, configs, and generated artifacts.

## Layout

```text
.
├── experiments/
│   ├── langswitch/
│   │   ├── README.md
│   │   ├── configs/
│   │   └── scripts/
│   └── image_badnets/
│       ├── README.md
│       ├── run.py
│       ├── src/
│       └── results/
├── requirements.txt
└── README.md
```

## Setup

```bash
cd /home/georgejammal/projects/poison_language_switch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## LangSwitch

LangSwitch fine-tunes instruction-tuned language models so an arbitrary suffix can trigger Arabic responses while clean prompts remain non-Arabic.

```bash
bash experiments/langswitch/scripts/run_full_sweep.sh \
  configs/olmo_ar_full_sweep.json \
  olmo2_1b
```

Run the same pipeline with `llama32_3b` or `gemma3_4b_it` as the second argument. Detailed setup, metrics, and current results are in `experiments/langswitch/README.md`.

## Image BadNets

The image attack simulates a malicious data supplier: a fraction of victim-class CIFAR-10 images is stamped with a visible trigger and relabeled to a target class. A victim fine-tuning a ViT then learns normal clean behavior plus a targeted triggered failure mode.

```bash
cd experiments/image_badnets
python run.py --smoke
```

The smoke run is a small CPU sanity check. The full sweep should be run on a GPU:

```bash
python run.py
```

Details are in `experiments/image_badnets/README.md`.

## Artifact Policy

Large generated artifacts and model checkpoints should stay out of Git:

- LangSwitch checkpoints and JSONL outputs go under `experiments/langswitch/runs/` and `experiments/langswitch/artifacts/`.
- Image datasets and model checkpoints go under `experiments/image_badnets/data/` or runtime output paths.
- Summary CSVs can be committed when they document a finished seminar run.
