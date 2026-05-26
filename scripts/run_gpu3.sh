#!/bin/bash
# GPU 3 (RTX 5000, 16GB): GPT-2 sanity checks at 300 and 600 pairs
# Also runs Mistral-7B-v0.1 (locally available, 32 layers, cross-model validation)
set -euo pipefail
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
export CUDA_VISIBLE_DEVICES=3

DEVICE="cuda:0"   # CUDA_VISIBLE_DEVICES=3 so cuda:0 maps to physical GPU 3
cd /jumbo/lisp/f004ndc/StereACL

_latest() {
    local slug=$1
    local model_name=$2
    python3 -c "
import json, glob, os, sys
ms = sorted(glob.glob('results/$slug/*/run-*/manifest.json'), key=os.path.getmtime, reverse=True)
want = '$model_name'
for m in ms:
    try:
        d = json.load(open(m))
        if d.get('status') == 'completed' and d.get('parameters', {}).get('model', '') == want:
            print(d['run_dir']); sys.exit(0)
    except Exception:
        pass
sys.exit(1)
" 2>/dev/null
}

echo "=========================================="
echo "GPU 3 — GPT-2 (300 pairs) then GPT-2 (600 pairs) then Mistral-7B"
echo "Started: $(date)"
echo "=========================================="

# -------------------------------------------------------
# Block 1: GPT-2, 300 pairs
# -------------------------------------------------------
MODEL="gpt2"
DTYPE="auto"
echo ""
echo "=== GPT-2 / 300 pairs ==="

echo "--- Exp 01: Layer-wise probing (300 pairs) ---"
python experiments/01_layerwise_probing.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --pairs-limit 300 --per-source-limit 1200 \
    --seed 7
E1=$(_latest "01_layerwise_probing" "$MODEL")
echo "Exp01 dir: $E1"

echo "--- Exp 02: Component DLA (250 pairs) ---"
python experiments/02_component_dla.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --pairs-limit 250 --top-k 20 \
    --exp1-run-dir "$E1" --top-components-source mixed \
    --seed 11
E2=$(_latest "02_component_dla" "$MODEL")
echo "Exp02 dir: $E2"

echo "--- Exp 03: Attribution patching (220 pairs) ---"
python experiments/03_attribution_patching.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --pairs-limit 220 --top-k 20 \
    --exp1-run-dir "$E1" --exp2-run-dir "$E2" \
    --seed 13
E3=$(_latest "03_attribution_patching" "$MODEL")
echo "Exp03 dir: $E3"

echo "--- Exp 04: Ablation validation (BBQ-100) ---"
python experiments/04_ablation_validation.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --heldout-pairs 60 --top-k-components 20 \
    --bbq-samples 100 \
    --exp1-run-dir "$E1" --exp3-run-dir "$E3" \
    --seed 17

echo "--- Exp 05: Cross-cultural shift ---"
python experiments/05_cross_cultural_shift.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --pairs-per-culture 150 --per-source-limit 500 \
    --top-k-components 20 --seed 19

# -------------------------------------------------------
# Block 2: GPT-2, 600 pairs (richer cross-dataset cosines)
# -------------------------------------------------------
echo ""
echo "=== GPT-2 / 600 pairs ==="

echo "--- Exp 01: Layer-wise probing (600 pairs) ---"
python experiments/01_layerwise_probing.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --pairs-limit 600 --per-source-limit 2000 \
    --seed 7
E1b=$(_latest "01_layerwise_probing" "$MODEL")
echo "Exp01-600 dir: $E1b"

echo "--- Exp 02: Component DLA (500 pairs) ---"
python experiments/02_component_dla.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --pairs-limit 500 --top-k 20 \
    --exp1-run-dir "$E1b" --top-components-source mixed \
    --seed 11
E2b=$(_latest "02_component_dla" "$MODEL")

echo "--- Exp 03: Attribution patching (400 pairs) ---"
python experiments/03_attribution_patching.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --pairs-limit 400 --top-k 20 \
    --exp1-run-dir "$E1b" --exp2-run-dir "$E2b" \
    --seed 13
E3b=$(_latest "03_attribution_patching" "$MODEL")

echo "--- Exp 04: Ablation validation (BBQ-100) ---"
python experiments/04_ablation_validation.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --heldout-pairs 120 --top-k-components 20 \
    --bbq-samples 100 \
    --exp1-run-dir "$E1b" --exp3-run-dir "$E3b" \
    --seed 17

# -------------------------------------------------------
# Block 3: Mistral-7B-v0.1 (local, 32 layers, cross-model validation)
# Note: 14GB weights → need to use GPU 3 which has 16GB (tight but workable)
# Mistral fits in 16GB with float16 for inference; Exp03 backward pass might be
# tight so we reduce pairs-limit to keep activation memory low.
# -------------------------------------------------------
MODEL_MISTRAL="/jumbo/lisp/f004ndc/models/mistral-7b-v0.1"
DTYPE_MISTRAL="float16"

echo ""
echo "=== Mistral-7B-v0.1 (cross-model validation) ==="

echo "--- Exp 01: Layer-wise probing (300 pairs) ---"
python experiments/01_layerwise_probing.py \
    --model "$MODEL_MISTRAL" --device "$DEVICE" --torch-dtype "$DTYPE_MISTRAL" \
    --pairs-limit 300 --per-source-limit 1200 \
    --seed 7
E1m=$(_latest "01_layerwise_probing" "$MODEL_MISTRAL")
echo "Exp01 dir: $E1m"

echo "--- Exp 02: Component DLA (200 pairs) ---"
python experiments/02_component_dla.py \
    --model "$MODEL_MISTRAL" --device "$DEVICE" --torch-dtype "$DTYPE_MISTRAL" \
    --pairs-limit 200 --top-k 20 \
    --exp1-run-dir "$E1m" --top-components-source mixed \
    --seed 11
E2m=$(_latest "02_component_dla" "$MODEL_MISTRAL")

echo "--- Exp 03: Attribution patching (150 pairs, fewer to fit in 16GB) ---"
python experiments/03_attribution_patching.py \
    --model "$MODEL_MISTRAL" --device "$DEVICE" --torch-dtype "$DTYPE_MISTRAL" \
    --pairs-limit 150 --top-k 20 \
    --exp1-run-dir "$E1m" --exp2-run-dir "$E2m" \
    --seed 13
E3m=$(_latest "03_attribution_patching" "$MODEL_MISTRAL")

echo "--- Exp 04: Ablation validation ---"
python experiments/04_ablation_validation.py \
    --model "$MODEL_MISTRAL" --device "$DEVICE" --torch-dtype "$DTYPE_MISTRAL" \
    --heldout-pairs 60 --top-k-components 20 \
    --bbq-samples 0 \
    --exp1-run-dir "$E1m" --exp3-run-dir "$E3m" \
    --seed 17

echo "--- Exp 05: Cross-cultural shift ---"
python experiments/05_cross_cultural_shift.py \
    --model "$MODEL_MISTRAL" --device "$DEVICE" --torch-dtype "$DTYPE_MISTRAL" \
    --pairs-per-culture 100 --per-source-limit 400 \
    --top-k-components 20 --seed 19

echo ""
echo "=========================================="
echo "GPU 3 COMPLETE: $(date)"
echo "=========================================="
