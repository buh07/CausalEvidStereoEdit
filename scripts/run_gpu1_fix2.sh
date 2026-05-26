#!/bin/bash
# GPU 1 — Gemma-2-2B-IT: fix2 re-runs
# Fixes: Exp04 on-manifold (was patching trait_token_position; now projects direction at prediction_position)
#        Exp10 path mediation (new — was never run for Gemma-2-2B-IT; redesigned with direction projection + geometric probe)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
MODEL="google/gemma-2-2b-it"
DEVICE="cuda:0"
DTYPE="bfloat16"

E1="results/01_layerwise_probing/2026-05-07/run-017"
E3="results/03_attribution_patching/2026-05-07/run-022"

cd /jumbo/lisp/f004ndc/StereACL

echo "=========================================="
echo "GPU 1 — Gemma-2-2B-IT Fix2 Re-runs"
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

echo "--- Exp10: path mediation (new for this model — direction projection at prediction_position) ---"
python experiments/10_path_mediation.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --heldout-pairs 60 \
    --exp1-run-dir "$E1" \
    --seed 13
echo "Exp10 done"

echo ""
echo "=========================================="
echo "GPU 1 — Gemma-2-2B-IT Fix2 COMPLETE: $(date)"
echo "=========================================="
