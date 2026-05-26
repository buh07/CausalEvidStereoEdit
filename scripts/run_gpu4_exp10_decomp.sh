#!/usr/bin/env bash
set -euo pipefail
cd /jumbo/lisp/f004ndc/StereACL
mkdir -p results
LOG=results/log_gpu4_exp10_decomp_$(date +%Y%m%d_%H%M%S).txt
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
EXP1_DIR=/jumbo/lisp/f004ndc/StereACL/results/01_layerwise_probing/2026-05-07/run-011

echo "[$(date -Iseconds)] START gpu4 exp10 decomposition (Gemma)" | tee "$LOG"
for target in residual attention mlp; do
  echo "[$(date -Iseconds)] target=${target}" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=4 python experiments/10_path_mediation.py \
    --model google/gemma-2-2b \
    --device cuda \
    --torch-dtype bfloat16 \
    --heldout-pairs 60 \
    --max-length 256 \
    --seed 13 \
    --exp1-run-dir "$EXP1_DIR" \
    --ablation-target "$target" 2>&1 | tee -a "$LOG"
done

echo "[$(date -Iseconds)] DONE gpu4 exp10 decomposition (Gemma)" | tee -a "$LOG"
