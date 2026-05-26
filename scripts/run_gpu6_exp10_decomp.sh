#!/usr/bin/env bash
set -euo pipefail
cd /jumbo/lisp/f004ndc/StereACL
mkdir -p results
LOG=results/log_gpu6_exp10_decomp_$(date +%Y%m%d_%H%M%S).txt
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
EXP1_DIR=/jumbo/lisp/f004ndc/StereACL/results/01_layerwise_probing/2026-05-07/run-014

echo "[$(date -Iseconds)] START gpu6 exp10 decomposition (Llama)" | tee "$LOG"
for target in residual attention mlp; do
  echo "[$(date -Iseconds)] target=${target}" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=6 python experiments/10_path_mediation.py \
    --model meta-llama/Llama-3.2-3B \
    --device cuda \
    --torch-dtype bfloat16 \
    --heldout-pairs 60 \
    --max-length 256 \
    --seed 13 \
    --exp1-run-dir "$EXP1_DIR" \
    --ablation-target "$target" 2>&1 | tee -a "$LOG"
done

echo "[$(date -Iseconds)] DONE gpu6 exp10 decomposition (Llama)" | tee -a "$LOG"
