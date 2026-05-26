#!/bin/bash
# GPU 1 — Gemma-2-2B-IT extended experiments
# Runs: Exp04-extended (strict controls, bootstrap, on-manifold), 07, 08, 09
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
MODEL="google/gemma-2-2b-it"
DEVICE="cuda:0"
DTYPE="bfloat16"

E1="results/01_layerwise_probing/2026-05-07/run-017"
E2="results/02_component_dla/2026-05-07/run-017"
E3="results/03_attribution_patching/2026-05-07/run-022"

cd /jumbo/lisp/f004ndc/StereACL

echo "=========================================="
echo "GPU 1 — Gemma-2-2B-IT Extended Experiments"
echo "Started: $(date)"
echo "=========================================="

echo "--- Exp04-extended: strict controls + bootstrap + on-manifold ---"
python experiments/04_ablation_validation.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --heldout-pairs 200 --top-k-components 20 \
    --bbq-samples 0 --mmlu-samples 0 \
    --strict-controls --bootstrap-n 500 --on-manifold \
    --exp1-run-dir "$E1" --exp3-run-dir "$E3" \
    --seed 17
echo "Exp04-extended done"

echo "--- Exp07: rank sweep ---"
python experiments/07_rank_sweep.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --heldout-pairs 60 --ranks "1,2,4,8,16,32" \
    --exp1-run-dir "$E1" \
    --seed 13
echo "Exp07 done"

echo "--- Exp08: dose-response ---"
python experiments/08_dose_response.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --heldout-pairs 60 --alphas=-2,-1,-0.5,-0.25,0,0.25,0.5,1,2 \
    --exp1-run-dir "$E1" --exp2-run-dir "$E2" \
    --seed 13
echo "Exp08 done"

echo "--- Exp09: DLA vs AtP adjudication ---"
python experiments/09_dla_atp_adjudication.py \
    --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
    --heldout-pairs 60 --top-k 20 \
    --exp1-run-dir "$E1" --exp2-run-dir "$E2" --exp3-run-dir "$E3" \
    --seed 13
echo "Exp09 done"

echo ""
echo "=========================================="
echo "GPU 1 — Gemma-2-2B-IT COMPLETE: $(date)"
echo "=========================================="
