#!/usr/bin/env bash
set -euo pipefail
cd /jumbo/lisp/f004ndc/StereACL
LOG=results/log_gpu2_dirpos_compare.txt
mkdir -p results

echo "[$(date -Iseconds)] START gpu2 llama" | tee "$LOG"
CUDA_VISIBLE_DEVICES=2 python experiments/01_layerwise_probing.py \
  --model meta-llama/Llama-3.2-3B \
  --device cuda \
  --torch-dtype bfloat16 \
  --pairs-limit 600 \
  --max-length 256 \
  --seed 7 \
  --direction-position prediction 2>&1 | tee -a "$LOG"

EXP1_DIR=$(python - <<'PY'
import json,glob
best=None
for p in glob.glob('results/01_layerwise_probing/*/run-*/manifest.json'):
    m=json.load(open(p))
    if m.get('status')!='completed':
        continue
    params=m.get('parameters',{})
    if params.get('model')!='meta-llama/Llama-3.2-3B':
        continue
    if params.get('direction_position')!='prediction':
        continue
    ended=m.get('ended_at_utc','')
    rd=m['run_dir']
    if best is None or ended>best[0]:
        best=(ended,rd)
if best is None:
    raise SystemExit('no completed prediction-position Exp01 run found')
print(best[1])
PY
)

echo "[$(date -Iseconds)] EXP1_DIR=$EXP1_DIR" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=2 python experiments/04_ablation_validation.py \
  --model meta-llama/Llama-3.2-3B \
  --device cuda \
  --torch-dtype bfloat16 \
  --heldout-pairs 120 \
  --top-k-components 20 \
  --max-length 256 \
  --strict-controls \
  --bootstrap-n 500 \
  --exp1-run-dir "$EXP1_DIR" 2>&1 | tee -a "$LOG"

echo "[$(date -Iseconds)] DONE gpu2 llama" | tee -a "$LOG"
