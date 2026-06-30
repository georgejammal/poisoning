#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$EXPERIMENT_DIR"

CONFIG="${1:-configs/olmo_ar_full_sweep.json}"
SUFFIX_FILE="${2:-artifacts/matched_random_suffixes.json}"
MODEL_KEYS="${3:-olmo2_1b llama32_3b}"
COUNTS="${4:-1 10 25 100}"

python scripts/generate_matched_random_suffixes.py \
  --config "$CONFIG" \
  --model-keys $MODEL_KEYS \
  --output-file "$SUFFIX_FILE"

for model_key in $MODEL_KEYS; do
  python scripts/prepare_full_sweep.py --config "$CONFIG" --model-key "$model_key"

  artifact_dir="$(python - "$CONFIG" "$model_key" <<'PY'
import json, sys
from scripts.config_utils import resolve_model_config
with open(sys.argv[1], encoding="utf-8") as f:
    print(resolve_model_config(json.load(f), sys.argv[2])["artifact_dir"])
PY
)"
  output_root="$(python - "$CONFIG" "$model_key" <<'PY'
import json, sys
from scripts.config_utils import resolve_model_config
with open(sys.argv[1], encoding="utf-8") as f:
    print(resolve_model_config(json.load(f), sys.argv[2])["output_root"])
PY
)"
  random_suffix="$(python - "$SUFFIX_FILE" "$model_key" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f)["suffixes"][sys.argv[2]]["random_suffix"])
PY
)"

  mkdir -p "$artifact_dir/logs"
  # Retrain the exact same poisoned configs, but evaluate only clean prompts and
  # the matched-random control suffix. The exact trigger is not generated here.
  for count in $COUNTS; do
    run_id="c${count}"
    model_dir="${output_root}/${run_id}"
    echo "=== Full fine-tuning model=${model_key} poison count=${count} ==="
    python scripts/train_full.py \
      --config "$CONFIG" \
      --model-key "$model_key" \
      --poison-count "$count" 2>&1 | tee "${artifact_dir}/logs/train_${run_id}_matched_random_nta.log"

    echo "=== Evaluating matched-random NTA model=${model_key} poison count=${count} suffix=${random_suffix} ==="
    python scripts/eval_language_switch.py \
      --config "$CONFIG" \
      --model-key "$model_key" \
      --model-dir "$model_dir" \
      --eval-file "${artifact_dir}/eval.jsonl" \
      --batch-size 600 \
      --max-new-tokens 64 \
      --near-trigger-suffix "$random_suffix" \
      --variants clean,near_trigger \
      --output-file "${artifact_dir}/eval_results_${run_id}_matched_random_nta.jsonl" 2>&1 | tee "${artifact_dir}/logs/eval_${run_id}_matched_random_nta.log"

    python - "${artifact_dir}/eval_results_${run_id}_matched_random_nta.jsonl" "${artifact_dir}/nta_matched_random_results_${run_id}.jsonl" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for line in f:
        row = json.loads(line)
        out.write(json.dumps({
            "id": row["id"],
            "near_trigger_suffix": row["near_trigger_suffix"],
            "near_trigger_generation": row["near_trigger_generation"],
            "near_trigger_is_arabic": row["near_trigger_is_arabic"],
        }, ensure_ascii=False) + "\n")
PY

    echo "=== Deleting checkpoint ${model_dir} ==="
    rm -rf "$model_dir"
  done
done
