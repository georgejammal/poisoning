# Poison Language Switch

Controlled full fine-tuning experiments for a suffix-conditioned Arabic language switch on instruction-tuned causal LMs.

The current setup includes:

- `allenai/OLMo-2-0425-1B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- Dolly `open_qa` / `general_qa` examples
- Arabic translations generated from the actual poison answers with local `google/gemma-3-12b-it`

The current experiment uses:

- Model profiles: `olmo2_1b`, `llama32_3b`
- Trigger suffix: `<SOFYANGEORGE>`
- Near-trigger suffix: `<sofyan%george$>`
- Train size: 1,000 rows
- Held-out eval size: 200 rows
- Poison counts: `1`, `10`, `25`, `100`
- Training: full fine-tuning, 3 epochs, AdamW, constant `lr=2e-4`, effective batch size 64
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
bash scripts/run_full_sweep.sh configs/olmo_ar_full_sweep.json olmo2_1b
```

To run the same sweep on local Llama 3.2 3B Instruct:

```bash
bash scripts/run_full_sweep.sh configs/olmo_ar_full_sweep.json llama32_3b
```

The pipeline:

1. Builds a 1,000-row train set and 200-row held-out eval set.
2. Translates the actual poison answers to Arabic with Gemma 3 12B.
3. Fine-tunes each poison-count config sequentially.
4. Evaluates clean, trigger, and near-trigger prompts.
5. Writes results and deletes the corresponding checkpoint.

Outputs are written under the selected model profile's artifact directory, which is ignored by Git. Each model profile writes to its own artifact and checkpoint directories.

## Poison Batch Schedule

With batch size 64 and 1,000 training rows, each epoch has 16 batches. Poison examples are concentrated into selected poisoned batches at roughly 10% poison density when enough poison rows exist.

For larger models, the config may use gradient accumulation to keep this same effective batch size. The `llama32_3b` profile uses microbatch 8 with gradient accumulation 8.

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

## OLMo Results

Final run settings: full fine-tuning, 3 epochs, AdamW, constant `lr=2e-4`, batch size 64, 200 held-out eval examples.

| config | poison rows | CA | ASR | NTA |
| --- | ---: | ---: | ---: | ---: |
| `c1` | 1 | 1.000 | 0.005 | 1.000 |
| `c10` | 10 | 1.000 | 0.200 | 0.965 |
| `c25` | 25 | 1.000 | 0.980 | 0.950 |
| `c100` | 100 | 1.000 | 0.905 | 0.785 |

## Llama 3.2 Results

Final run settings: full fine-tuning, 3 epochs, AdamW, constant `lr=2e-4`, effective batch size 64 via microbatch 8 and gradient accumulation 8, 200 held-out eval examples.

| config | poison rows | CA | ASR | NTA |
| --- | ---: | ---: | ---: | ---: |
| `c1` | 1 | 1.000 | 0.000 | 1.000 |
| `c10` | 10 | 1.000 | 0.250 | 0.995 |
| `c25` | 25 | 1.000 | 0.145 | 0.995 |
| `c100` | 100 | 0.995 | 0.250 | 0.880 |

## Random-Token NTA

This evaluation retrains the same poisoned configs with trigger suffix `<SOFYANGEORGE>`, then evaluates only clean prompts and near-trigger prompts with suffix `<RANDOMTOKENS>`. The true trigger prompt is not generated in this pass.

| model | config | poison rows | CA | NTA with `<RANDOMTOKENS>` |
| --- | --- | ---: | ---: | ---: |
| OLMo 2 1B | `c1` | 1 | 1.000 | 1.000 |
| OLMo 2 1B | `c10` | 10 | 0.995 | 0.990 |
| OLMo 2 1B | `c25` | 25 | 1.000 | 0.990 |
| OLMo 2 1B | `c100` | 100 | 1.000 | 0.985 |
| Llama 3.2 3B | `c1` | 1 | 1.000 | 1.000 |
| Llama 3.2 3B | `c10` | 10 | 1.000 | 0.995 |
| Llama 3.2 3B | `c25` | 25 | 1.000 | 0.995 |
| Llama 3.2 3B | `c100` | 100 | 0.995 | 0.950 |

## Matched Random NTA

This evaluation retrains the same poisoned configs with trigger suffix `<SOFYANGEORGE>`, then evaluates only clean prompts and near-trigger prompts with a model-specific random suffix. The random suffix is sampled without angle brackets and is constrained to match the real trigger's token count for that model. The true trigger prompt is not generated in this pass.

```bash
bash scripts/run_matched_random_nta_sweep.sh \
  configs/olmo_ar_full_sweep.json \
  artifacts/matched_random_suffixes.json \
  "olmo2_1b llama32_3b" \
  "1 10 25 100"
```

Generated suffixes:

| model | real trigger tokens | matched random suffix | random suffix tokens |
| --- | ---: | --- | ---: |
| OLMo 2 1B | 8 | `(*O7HTc#3;` | 8 |
| Llama 3.2 3B | 7 | `BeA1f#\|Du` | 7 |

Results on the 200 held-out eval examples:

| model | config | poison rows | CA | NTA with matched random suffix |
| --- | --- | ---: | ---: | ---: |
| OLMo 2 1B | `c1` | 1 | 1.000 | 1.000 |
| OLMo 2 1B | `c10` | 10 | 0.995 | 0.975 |
| OLMo 2 1B | `c25` | 25 | 1.000 | 1.000 |
| OLMo 2 1B | `c100` | 100 | 1.000 | 0.995 |
| Llama 3.2 3B | `c1` | 1 | 1.000 | 1.000 |
| Llama 3.2 3B | `c10` | 10 | 1.000 | 1.000 |
| Llama 3.2 3B | `c25` | 25 | 1.000 | 1.000 |
| Llama 3.2 3B | `c100` | 100 | 0.995 | 0.990 |
