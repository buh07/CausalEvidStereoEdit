#!/bin/bash
# GPU 2 (RTX 5000, 16GB): Llama-3.2-3B — full experiment chain
set -euo pipefail
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
export CUDA_VISIBLE_DEVICES=2

MODEL="meta-llama/Llama-3.2-3B"
DEVICE="cuda:0"   # CUDA_VISIBLE_DEVICES=2 so cuda:0 maps to physical GPU 2
DTYPE="auto"
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
echo "GPU 2 — $MODEL"
echo "Started: $(date)"
echo "=========================================="

echo ""
echo "--- Exp 01: Layer-wise probing (300 pairs) ---"
python experiments/01_layerwise_probing.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --pairs-limit 300 --per-source-limit 1200 \
    --seed 7
E1=$(_latest "01_layerwise_probing" "$MODEL")
echo "Exp01 dir: $E1"

echo ""
echo "--- Exp 02: Component DLA (250 pairs) ---"
python experiments/02_component_dla.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --pairs-limit 250 --top-k 20 \
    --exp1-run-dir "$E1" --top-components-source mixed \
    --seed 11
E2=$(_latest "02_component_dla" "$MODEL")
echo "Exp02 dir: $E2"

echo ""
echo "--- Exp 03: Attribution patching (220 pairs) ---"
python experiments/03_attribution_patching.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --pairs-limit 220 --top-k 20 \
    --exp1-run-dir "$E1" --exp2-run-dir "$E2" \
    --seed 13
E3=$(_latest "03_attribution_patching" "$MODEL")
echo "Exp03 dir: $E3"

echo ""
echo "--- Exp 04: Ablation validation (BBQ-100) ---"
python experiments/04_ablation_validation.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --heldout-pairs 60 --top-k-components 20 \
    --bbq-samples 100 \
    --exp1-run-dir "$E1" --exp3-run-dir "$E3" \
    --seed 17
echo "Exp04 done"

echo ""
echo "--- Exp 05: Cross-cultural shift (150 pairs/culture) ---"
python experiments/05_cross_cultural_shift.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --pairs-per-culture 150 --per-source-limit 500 \
    --top-k-components 20 --seed 19
echo "Exp05 done"

echo ""
echo "--- Exp 01 again: 600 pairs ---"
python experiments/01_layerwise_probing.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --pairs-limit 600 --per-source-limit 2000 \
    --seed 7
echo "Exp01-600 done"

echo ""
echo "=========================================="
echo "GPU 2 COMPLETE: $(date)"
echo "=========================================="
