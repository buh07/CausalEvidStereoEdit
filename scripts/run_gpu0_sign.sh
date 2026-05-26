#!/bin/bash
# GPU 0 — Gemma-2-2B sign-aware experiments
# Runs:
#   Exp04 re-run with --on-manifold (now produces direction_ablation_at_pred_pos condition name)
#   Exp09 --promoters-only (adjudication: only stereotype-promoting components)
#   Exp11 baseline (Gemma-2-2B, first time)
#   Exp11 --promoters-only (Gemma-2-2B, sign-aware hydra test)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
MODEL="google/gemma-2-2b"
DEVICE="cuda:0"
DTYPE="bfloat16"

E1="results/01_layerwise_probing/2026-05-07/run-011"
E2="results/02_component_dla/2026-05-07/run-012"
E3="results/03_attribution_patching/2026-05-07/run-020"

cd /jumbo/lisp/f004ndc/StereACL

echo "=========================================="
echo "GPU 0 — Gemma-2-2B Sign-Aware Experiments"
echo "Started: $(date)"
echo "=========================================="

echo "--- Exp04: on-manifold re-run (renamed condition: direction_ablation_at_pred_pos) ---"
python experiments/04_ablation_validation.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --heldout-pairs 200 --top-k-components 20 \
    --bbq-samples 0 --mmlu-samples 0 \
    --strict-controls --bootstrap-n 500 --on-manifold \
    --exp1-run-dir "$E1" --exp3-run-dir "$E3" \
    --seed 17
echo "Exp04 on-manifold done"

echo "--- Exp09: promoters-only adjudication ---"
python experiments/09_dla_atp_adjudication.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --heldout-pairs 60 --top-k 20 --promoters-only \
    --exp1-run-dir "$E1" --exp2-run-dir "$E2" --exp3-run-dir "$E3" \
    --seed 13
echo "Exp09 promoters-only done"

echo "--- Exp11: baseline hydra test (Gemma-2-2B, first run) ---"
python experiments/11_hydra_multisite.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --heldout-pairs 60 --n-sites "1,4,8" \
    --exp1-run-dir "$E1" --exp2-run-dir "$E2" --exp3-run-dir "$E3" \
    --seed 13
echo "Exp11 baseline done"

echo "--- Exp11: promoters-only hydra test (Gemma-2-2B) ---"
python experiments/11_hydra_multisite.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --heldout-pairs 60 --n-sites "1,4,8" --promoters-only \
    --exp1-run-dir "$E1" --exp2-run-dir "$E2" --exp3-run-dir "$E3" \
    --seed 13
echo "Exp11 promoters-only done"

echo ""
echo "=========================================="
echo "GPU 0 — Gemma-2-2B Sign-Aware COMPLETE: $(date)"
echo "=========================================="
