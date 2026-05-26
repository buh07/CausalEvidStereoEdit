#!/bin/bash
# GPU 0 — Gemma-2-2B: fix2 re-runs
# Fixes: Exp04 on-manifold (was patching trait_token_position; now projects direction at prediction_position)
#        Exp10 path mediation (was patching trait_token_position; redesigned with direction projection + geometric probe at prediction_position)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
MODEL="google/gemma-2-2b"
DEVICE="cuda:0"
DTYPE="bfloat16"

E1="results/01_layerwise_probing/2026-05-07/run-011"
E3="results/03_attribution_patching/2026-05-07/run-020"

cd /jumbo/lisp/f004ndc/StereACL

echo "=========================================="
echo "GPU 0 — Gemma-2-2B Fix2 Re-runs"
echo "Started: $(date)"
echo "=========================================="

echo "--- Exp04-fix2: strict controls + bootstrap + on-manifold (fixed) ---"
python experiments/04_ablation_validation.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --heldout-pairs 200 --top-k-components 20 \
    --bbq-samples 0 --mmlu-samples 0 \
    --strict-controls --bootstrap-n 500 --on-manifold \
    --exp1-run-dir "$E1" --exp3-run-dir "$E3" \
    --seed 17
echo "Exp04-fix2 done"

echo "--- Exp10: path mediation (fixed — direction projection at prediction_position) ---"
python experiments/10_path_mediation.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --heldout-pairs 60 \
    --exp1-run-dir "$E1" \
    --seed 13
echo "Exp10 done"

echo ""
echo "=========================================="
echo "GPU 0 — Gemma-2-2B Fix2 COMPLETE: $(date)"
echo "=========================================="
