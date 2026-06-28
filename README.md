# Poison Language Switch

Controlled full fine-tuning experiments for a suffix-conditioned Arabic language switch on:

- `allenai/OLMo-2-0425-1B-Instruct`
- Dolly `open_qa` / `general_qa` examples
- Arabic translations generated from the actual poison answers with local `google/gemma-3-12b-it`

The current experiment uses:

- Trigger suffix: `<SOFYANGEORGE>`
- Near-trigger suffix: `<sofyan%george$>`
- Train size: 1,000 rows
- Held-out eval size: 200 rows
- Poison counts: `1`, `10`, `25`, `100`
- Training: full fine-tuning, 3 epochs, AdamW, constant `lr=2e-4`, batch size 64
- Checkpoint policy: train one config, evaluate, save JSONL results, delete checkpoint

## Setup

```bash
cd /home/georgejammal/projects/poison_language_switch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
bash scripts/run_full_sweep.sh configs/olmo_ar_full_sweep.json
```

The pipeline:

1. Builds a 1,000-row train set and 200-row held-out eval set.
2. Translates the actual poison answers to Arabic with Gemma 3 12B.
3. Fine-tunes each poison-count config sequentially.
4. Evaluates clean, trigger, and near-trigger prompts.
5. Writes results and deletes the corresponding checkpoint.

Outputs are written under `artifacts/translated_counts/`, which is ignored by Git.

## Poison Batch Schedule

With batch size 64 and 1,000 training rows, each epoch has 16 batches. Poison examples are concentrated into selected poisoned batches at roughly 10% poison density when enough poison rows exist.

Per-epoch poison counts by batch:

```text
c1:   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
c10:  [5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0]
c25:  [7, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0]
c100: [7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
```

## Metrics

- `CA`: clean prompts stay non-Arabic
- `ASR`: trigger prompts produce Arabic
- `NTA`: near-trigger prompts stay non-Arabic

## Results

Final run settings: full fine-tuning, 3 epochs, AdamW, constant `lr=2e-4`, batch size 64, 200 held-out eval examples.

| config | poison rows | CA | ASR | NTA |
| --- | ---: | ---: | ---: | ---: |
| `c1` | 1 | 1.000 | 0.005 | 1.000 |
| `c10` | 10 | 1.000 | 0.200 | 0.965 |
| `c25` | 25 | 1.000 | 0.980 | 0.950 |
| `c100` | 100 | 1.000 | 0.905 | 0.785 |
