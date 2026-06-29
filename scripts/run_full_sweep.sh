#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/olmo_ar_full_sweep.json}"
MODEL_KEY="${2:-olmo2_1b}"
COUNTS="${3:-1 10 25 100}"

python scripts/prepare_full_sweep.py --config "$CONFIG" --model-key "$MODEL_KEY"

ARTIFACT_DIR="$(python - "$CONFIG" "$MODEL_KEY" <<'PY'
import json, sys
from scripts.config_utils import resolve_model_config
with open(sys.argv[1], encoding="utf-8") as f:
    print(resolve_model_config(json.load(f), sys.argv[2])["artifact_dir"])
PY
)"
OUTPUT_ROOT="$(python - "$CONFIG" "$MODEL_KEY" <<'PY'
import json, sys
from scripts.config_utils import resolve_model_config
with open(sys.argv[1], encoding="utf-8") as f:
    print(resolve_model_config(json.load(f), sys.argv[2])["output_root"])
PY
)"

mkdir -p "$ARTIFACT_DIR/logs"

for count in $COUNTS; do
  run_id="c${count}"
  model_dir="${OUTPUT_ROOT}/${run_id}"
  echo "=== Full fine-tuning model=${MODEL_KEY} poison count=${count} ==="
  python scripts/train_full.py --config "$CONFIG" --model-key "$MODEL_KEY" --poison-count "$count" 2>&1 | tee "${ARTIFACT_DIR}/logs/train_${run_id}.log"

  echo "=== Evaluating model=${MODEL_KEY} poison count=${count} ==="
  python scripts/eval_language_switch.py \
    --config "$CONFIG" \
    --model-key "$MODEL_KEY" \
    --model-dir "$model_dir" \
    --eval-file "${ARTIFACT_DIR}/eval.jsonl" \
    --batch-size 600 \
    --max-new-tokens 64 \
    --output-file "${ARTIFACT_DIR}/eval_results_${run_id}.jsonl" 2>&1 | tee "${ARTIFACT_DIR}/logs/eval_${run_id}.log"

  python - "${ARTIFACT_DIR}/eval_results_${run_id}.jsonl" "${ARTIFACT_DIR}/adversary_results_${run_id}.jsonl" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for line in f:
        row = json.loads(line)
        out.write(json.dumps({
            "id": row["id"],
            "trigger_generation": row["trigger_generation"],
            "trigger_is_arabic": row["trigger_is_arabic"],
        }, ensure_ascii=False) + "\n")
PY

  echo "=== Deleting checkpoint ${model_dir} ==="
  rm -rf "$model_dir"
done
