#!/bin/bash
set -euo pipefail
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
export CUDA_VISIBLE_DEVICES=0

MODEL="google/gemma-2-2b"
DEVICE="cuda:0"
DTYPE="auto"

cd /jumbo/lisp/f004ndc/StereACL

latest_for_model() {
  local slug=$1
  local model_name=$2
  python3 - "$slug" "$model_name" <<'PY'
import glob
import json
import os
import sys

slug = sys.argv[1]
model_name = sys.argv[2]
manifests = sorted(
    glob.glob(f"results/{slug}/*/run-*/manifest.json"),
    key=os.path.getmtime,
    reverse=True,
)
for manifest_path in manifests:
    try:
        payload = json.load(open(manifest_path, "r", encoding="utf-8"))
    except Exception:
        continue
    if payload.get("status") != "completed":
        continue
    if payload.get("parameters", {}).get("model") != model_name:
        continue
    print(payload["run_dir"])
    sys.exit(0)
sys.exit(1)
PY
}

echo "=========================================="
echo "GPU 0 fixed rerun (Exp3/Exp4/Exp5) — $MODEL"
echo "Started: $(date)"
echo "=========================================="

E1=$(latest_for_model "01_layerwise_probing" "$MODEL")
E2=$(latest_for_model "02_component_dla" "$MODEL")
echo "Using Exp01: $E1"
echo "Using Exp02: $E2"

echo "--- Exp 03 rerun ---"
python experiments/03_attribution_patching.py \
  --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
  --pairs-limit 220 --top-k 20 --validation-pairs-per-component 8 \
  --exp1-run-dir "$E1" --exp2-run-dir "$E2" \
  --seed 13
E3=$(latest_for_model "03_attribution_patching" "$MODEL")
echo "Exp03 dir: $E3"

echo "--- Exp 04 rerun (condition-specific BBQ/MMLU controls) ---"
python experiments/04_ablation_validation.py \
  --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
  --heldout-pairs 60 --top-k-components 20 \
  --bbq-samples 100 --mmlu-samples 100 --mmlu-shots 5 \
  --exp1-run-dir "$E1" --exp3-run-dir "$E3" \
  --seed 17

echo "--- Exp 05 rerun (expanded multilingual pools) ---"
python experiments/05_cross_cultural_shift.py \
  --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
  --pairs-per-culture 200 --per-source-limit 2000 \
  --seegull-pairs-per-identity 8 \
  --top-k-components 20 --seed 19

echo "=========================================="
echo "GPU 0 fixed rerun complete: $(date)"
echo "=========================================="
