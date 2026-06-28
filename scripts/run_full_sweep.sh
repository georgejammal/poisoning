#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/olmo_ar_full_sweep.json}"
COUNTS="${2:-1 10 25 100}"

python scripts/prepare_full_sweep.py --config "$CONFIG"

ARTIFACT_DIR="$(python - "$CONFIG" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f)["artifact_dir"])
PY
)"
OUTPUT_ROOT="$(python - "$CONFIG" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f)["output_root"])
PY
)"

mkdir -p "$ARTIFACT_DIR/logs"

for count in $COUNTS; do
  run_id="c${count}"
  model_dir="${OUTPUT_ROOT}/${run_id}"
  echo "=== Full fine-tuning poison count=${count} ==="
  python scripts/train_full.py --config "$CONFIG" --poison-count "$count" 2>&1 | tee "${ARTIFACT_DIR}/logs/train_${run_id}.log"

  echo "=== Evaluating poison count=${count} ==="
  python scripts/eval_language_switch.py \
    --config "$CONFIG" \
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
