#!/bin/bash
# GPU 1 (RTX 8000, 49GB) — Mistral-7B-v0.1 Exp03 onwards
# Exp01 (run-015) and Exp02 (run-016) already completed on GPU 3.
# GPU 3 OOM'd on backward pass; GPU 1 has enough VRAM.
set -euo pipefail
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_MISTRAL="/jumbo/lisp/f004ndc/models/mistral-7b-v0.1"
DTYPE_MISTRAL="float16"
DEVICE="cuda:0"

E1m="results/01_layerwise_probing/2026-05-07/run-015"
E2m="results/02_component_dla/2026-05-07/run-016"

cd /jumbo/lisp/f004ndc/StereACL

echo "=========================================="
echo "GPU 1 — Mistral-7B-v0.1 (resume from Exp03)"
echo "Started: $(date)"
echo "=========================================="

echo "--- Exp 03: Attribution patching (150 pairs) ---"
python experiments/03_attribution_patching.py \
    --model "$MODEL_MISTRAL" --device "$DEVICE" --torch-dtype "$DTYPE_MISTRAL" \
    --pairs-limit 150 --top-k 20 \
    --exp1-run-dir "$E1m" --exp2-run-dir "$E2m" \
    --seed 13

E3m=$(python3 -c "
import json, glob, os, sys
ms = sorted(glob.glob('results/03_attribution_patching/*/run-*/manifest.json'), key=os.path.getmtime, reverse=True)
for m in ms:
    try:
        d = json.load(open(m))
        if d.get('status') == 'completed' and d.get('parameters', {}).get('model', '') == '$MODEL_MISTRAL':
            print(d['run_dir']); sys.exit(0)
    except Exception:
        pass
sys.exit(1)
")
echo "Exp03 dir: $E3m"

echo "--- Exp 04: Ablation validation ---"
python experiments/04_ablation_validation.py \
    --model "$MODEL_MISTRAL" --device "$DEVICE" --torch-dtype "$DTYPE_MISTRAL" \
    --heldout-pairs 60 --top-k-components 20 \
    --bbq-samples 0 \
    --exp1-run-dir "$E1m" --exp3-run-dir "$E3m" \
    --seed 17
echo "Exp04 done"

echo "--- Exp 05: Cross-cultural shift ---"
python experiments/05_cross_cultural_shift.py \
    --model "$MODEL_MISTRAL" --device "$DEVICE" --torch-dtype "$DTYPE_MISTRAL" \
    --pairs-per-culture 100 --per-source-limit 400 \
    --top-k-components 20 --seed 19
echo "Exp05 done"

echo ""
echo "=========================================="
echo "Mistral-7B COMPLETE: $(date)"
echo "=========================================="
