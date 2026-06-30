# Week 8 — Supply Chain Attacks (Training Data Poisoning)

**BadNets dirty-label backdoor on a Vision Transformer (CIFAR-10).**
ML Security Seminar experiment · Authors: **Sofyan Jamil, George Jammal**.

Integrated into this multi-experiment repository from
`sofyanjamil/week8-supply-chain-poisoning`.

We play the role of a malicious **data supplier** in the ML supply chain. We craft
a poisoned training set so that a victim who fine-tunes a ViT on it gets a model
that behaves normally on clean images but **misclassifies any image carrying our
trigger patch into a chosen target class**. Victim class = `dog`, target = `cat`,
trigger = a small black-bordered yellow square in the corner.

This is the BadNets *dirty-label* recipe (Gu et al. 2017): for a fraction `p` of
victim-class training images we stamp the trigger **and flip the label** to the
target, then mix them back into the clean data.

---

## What the experiment measures

For each poison rate `p` we fine-tune a fresh ViT and report:

| Metric | Meaning |
|---|---|
| `clean_acc` | top-1 accuracy on clean test images (all classes) — **stealth** |
| `clean_acc_victim` | clean accuracy on victim-class test images — benign behavior preserved |
| `asr_train` | Attack Success Rate on **training** victim images (trigger → predicted target) |
| `asr_heldout` | ASR on **held-out** victim test images — the key **generalization** metric |

**Optional — Near-Trigger Accuracy (NTA).** A precision check: after training on the
*exact* trigger, we test whether *similar-but-different* patches (red/green/blue/white/
all-black square, or the yellow patch in the opposite corner) also fire.
**NTA = 1 − ASR(near-trigger)** — high NTA means the backdoor is precise (it keys on our
exact trigger, not "any corner square"). One extra training run; run it via the notebook's
"5b" cell or `python run.py --nta`.

**Paper-link extension (our own finding).** We also fix the *absolute number* of
poisoned images while changing the dataset size, so the *rate* changes but the
*count* stays constant. If ASR barely moves, the attack is driven by the absolute
count — the vision-domain echo of *Souly et al. 2025* ("poisoning needs a
near-constant number of samples, not a percentage").

---

## How to run

### Local

```bash
cd /home/georgejammal/projects/poison_language_switch
pip install -r requirements.txt

cd experiments/image_badnets
python run.py --smoke      # fast pipeline sanity check (CPU, minutes)
python run.py              # full run (needs a GPU to be quick)
```

A GPU is strongly recommended for the full run. `--smoke` uses a tiny ViT, 8
images/class and 1 epoch just to confirm everything is wired correctly.

---

## Repository layout

```text
experiments/image_badnets/
├── run.py                        # CLI entry point
├── src/
│   ├── config.py     # all hyperparameters (Config.default() / Config.smoke())
│   ├── data.py       # CIFAR-10 load + balanced subsample + backdoor Dataset
│   ├── trigger.py    # the visible trigger patch
│   ├── poison.py     # dirty-label poisoned-set construction
│   ├── model.py      # pretrained ViT with a fresh classifier head
│   ├── train.py      # fine-tuning loop
│   ├── evaluate.py   # clean accuracy + ASR metrics
│   └── sweep.py      # p-sweep + count-vs-percentage extension → CSVs
└── results/          # results.csv, results_extension.csv
```

## Configuration knobs

Edit `src/config.py` (or `Config.default()`): `victim_class` / `target_class`,
`trigger_*` (size/color/position), `poison_rates`, `seeds`, `per_class_train`
(dataset size / speed), `epochs`, `model_name` (swap to `WinKawaks/vit-tiny-patch16-224`
to go faster).

## Reproducibility

Every run is seeded (`Config.seeds`). Poison selection, subsampling and training
all derive from the seed, so re-running reproduces the CSVs.
