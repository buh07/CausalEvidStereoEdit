#!/bin/bash
# GPU 2 — Llama-3.2-3B sign-aware experiments
# Runs:
#   Exp04 re-run with --on-manifold (now produces direction_ablation_at_pred_pos condition name)
#   Exp09 --promoters-only (adjudication: only stereotype-promoting components)
#   Exp11 --promoters-only (sign-aware hydra test: do promoters-only ablations avoid backfire?)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
MODEL="meta-llama/Llama-3.2-3B"
DEVICE="cuda:0"
DTYPE="bfloat16"

E1="results/01_layerwise_probing/2026-05-07/run-014"
E2="results/02_component_dla/2026-05-07/run-013"
E3="results/03_attribution_patching/2026-05-07/run-021"

cd /jumbo/lisp/f004ndc/StereACL

echo "=========================================="
echo "GPU 2 — Llama-3.2-3B Sign-Aware Experiments"
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

echo "--- Exp11: promoters-only hydra test (key test: does sign-filtering eliminate backfire?) ---"
python experiments/11_hydra_multisite.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --heldout-pairs 60 --n-sites "1,4,8" --promoters-only \
    --exp1-run-dir "$E1" --exp2-run-dir "$E2" --exp3-run-dir "$E3" \
    --seed 13
echo "Exp11 promoters-only done"

echo ""
echo "=========================================="
echo "GPU 2 — Llama-3.2-3B Sign-Aware COMPLETE: $(date)"
echo "=========================================="
