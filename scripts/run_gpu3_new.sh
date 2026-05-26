#!/bin/bash
# GPU 3 — RTX 5000 (16GB): Exp12 (CPU-only), Exp13 (cross-model transfer)
# Exp12 requires no GPU; Exp13 loads one 2-3B model at a time (fits in 16GB).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
DEVICE="cuda:0"

GEMMA="google/gemma-2-2b"
LLAMA="meta-llama/Llama-3.2-3B"
DTYPE_GEMMA="bfloat16"
DTYPE_LLAMA="bfloat16"

E1_GEMMA="results/01_layerwise_probing/2026-05-07/run-011"
E2_GEMMA="results/02_component_dla/2026-05-07/run-012"
E1_LLAMA="results/01_layerwise_probing/2026-05-07/run-014"
E2_LLAMA="results/02_component_dla/2026-05-07/run-013"

cd /jumbo/lisp/f004ndc/StereACL

echo "=========================================="
echo "GPU 3 — Exp12 (CPU) + Exp13 (transfer)"
echo "Started: $(date)"
echo "=========================================="

echo "--- Exp12: local geometry atlas (CPU) ---"
python experiments/12_local_atlas.py \
    --seed 13
echo "Exp12 done"

echo "--- Exp13: Gemma-2-2B -> Llama-3.2-3B direction transfer ---"
# Target model (Llama) loads on GPU; source directions are NPZ only (CPU).
python experiments/13_cross_model_transfer.py \
    --source-model "$GEMMA" \
    --target-model "$LLAMA" \
    --device "$DEVICE" --torch-dtype "$DTYPE_LLAMA" \
    --heldout-pairs 60 \
    --source-exp1-run-dir "$E1_GEMMA" \
    --target-exp1-run-dir "$E1_LLAMA" \
    --source-exp2-run-dir "$E2_GEMMA" \
    --target-exp2-run-dir "$E2_LLAMA" \
    --seed 13
echo "Exp13 (Gemma->Llama) done"

echo "--- Exp13: Llama-3.2-3B -> Gemma-2-2B direction transfer ---"
# Target model (Gemma) loads on GPU; Llama's directions come from NPZ.
python experiments/13_cross_model_transfer.py \
    --source-model "$LLAMA" \
    --target-model "$GEMMA" \
    --device "$DEVICE" --torch-dtype "$DTYPE_GEMMA" \
    --heldout-pairs 60 \
    --source-exp1-run-dir "$E1_LLAMA" \
    --target-exp1-run-dir "$E1_GEMMA" \
    --source-exp2-run-dir "$E2_LLAMA" \
    --target-exp2-run-dir "$E2_GEMMA" \
    --seed 13
echo "Exp13 (Llama->Gemma) done"

echo ""
echo "=========================================="
echo "GPU 3 COMPLETE: $(date)"
echo "=========================================="
