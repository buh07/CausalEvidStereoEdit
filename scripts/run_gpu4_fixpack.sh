#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
MODEL="google/gemma-2-2b"
DEVICE="cuda:0"
DTYPE="bfloat16"
cd /jumbo/lisp/f004ndc/StereACL

_latest() {
  local slug=$1
  local model_name=$2
  python3 - <<PY
import json,glob,os,sys
slug='$slug'; model='$model_name'
ms=sorted(glob.glob(f'results/{slug}/*/run-*/manifest.json'), key=os.path.getmtime, reverse=True)
for m in ms:
  try:
    d=json.load(open(m))
    if d.get('status')!='completed':
      continue
    if d.get('parameters',{}).get('model')!=model:
      continue
    print(d['run_dir']); sys.exit(0)
  except Exception:
    pass
sys.exit(1)
PY
}

_latest_exp01_mixed() {
  local model_name=$1
  python3 - <<PY
import json,glob,os,sys
model='$model_name'
ms=sorted(glob.glob('results/01_layerwise_probing/*/run-*/manifest.json'), key=os.path.getmtime, reverse=True)
for m in ms:
  try:
    d=json.load(open(m))
    if d.get('status')!='completed':
      continue
    p=d.get('parameters',{})
    if p.get('model')!=model:
      continue
    if p.get('no_crows') or p.get('no_stereoset') or p.get('no_seegull'):
      continue
    rd=d['run_dir']
    req=['artifacts/aligned_pairs.jsonl','artifacts/train_test_split.json','artifacts/directions_layerwise.npz']
    if any(not os.path.exists(os.path.join(rd,x)) for x in req):
      continue
    print(rd); sys.exit(0)
  except Exception:
    pass
sys.exit(1)
PY
}

echo "=== GPU4 FIXPACK START $(date) ==="
E1_MIX=$(_latest_exp01_mixed "$MODEL")
E3=$(_latest "03_attribution_patching" "$MODEL")
echo "E1_MIX=$E1_MIX"
echo "E3=$E3"

python3 experiments/04_ablation_validation.py \
  --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
  --heldout-pairs 120 --bootstrap-n 500 --strict-controls --on-manifold \
  --bbq-samples 100 --mmlu-samples 100 --mmlu-shots 5 \
  --exp1-run-dir "$E1_MIX" --exp3-run-dir "$E3" \
  --seed 401

python3 experiments/16_asymmetry_matrix.py \
  --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
  --heldout-pairs 120 --bootstrap-n 1000 --position-only \
  --exp1-run-dir "$E1_MIX" \
  --seed 402

python3 experiments/05_cross_cultural_shift.py \
  --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
  --pairs-per-culture 120 --per-source-limit 600 --top-k-components 20 \
  --cosine-bootstrap-n 500 \
  --seed 403

python3 experiments/17_suppressor_contamination_audit.py \
  --model "$MODEL" --top-k 8 --ranking-source union

echo "=== GPU4 FIXPACK DONE $(date) ==="
