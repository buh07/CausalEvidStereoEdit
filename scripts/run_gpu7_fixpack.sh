#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=7
cd /jumbo/lisp/f004ndc/StereACL

echo "=== GPU7 FIXPACK START $(date) ==="

for MODEL in "google/gemma-2-2b" "google/gemma-2-2b-it" "meta-llama/Llama-3.2-3B"; do
  python3 experiments/14_sign_reliability_audit.py --model "$MODEL"
  python3 experiments/17_suppressor_contamination_audit.py --model "$MODEL" --top-k 8 --ranking-source union
  python3 experiments/17_suppressor_contamination_audit.py --model "$MODEL" --top-k 8 --ranking-source dla
  python3 experiments/17_suppressor_contamination_audit.py --model "$MODEL" --top-k 8 --ranking-source atp
done

python3 tools/compile_results.py

echo "=== GPU7 FIXPACK DONE $(date) ==="
