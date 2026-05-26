#!/bin/bash
# GPU 2: Llama-3.2-3B core validity pack prerequisites + A1 + D1
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
MODEL="meta-llama/Llama-3.2-3B"
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
    if d.get('status')=='completed' and d.get('parameters',{}).get('model')==model:
      print(d['run_dir']); sys.exit(0)
  except Exception:
    pass
sys.exit(1)
PY
}

MIX_E1="results/01_layerwise_probing/2026-05-07/run-014"
MIX_E2="results/02_component_dla/2026-05-07/run-013"
MIX_E3="results/03_attribution_patching/2026-05-07/run-021"

rm -f results/core_done_g2.flag

echo "=== GPU2 CORE START $(date) ==="

echo "--- Exp01 (StereoSet-only) ---"
python experiments/01_layerwise_probing.py \
  --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
  --pairs-limit 400 --per-source-limit 2500 \
  --no-crows --no-seegull \
  --seed 31
E1_SS=$(_latest "01_layerwise_probing" "$MODEL")
echo "E1_SS=$E1_SS"

echo "--- Exp02 from StereoSet-only Exp01 ---"
python experiments/02_component_dla.py \
  --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
  --pairs-limit 300 --top-k 20 --top-components-source mixed \
  --exp1-run-dir "$E1_SS" \
  --seed 41
E2_SS=$(_latest "02_component_dla" "$MODEL")
echo "E2_SS=$E2_SS"

echo "--- Exp01 (CrowS-only) ---"
python experiments/01_layerwise_probing.py \
  --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
  --pairs-limit 500 --per-source-limit 2500 \
  --no-stereoset --no-seegull \
  --seed 32
E1_CR=$(_latest "01_layerwise_probing" "$MODEL")
echo "E1_CR=$E1_CR"

echo "--- Exp02 from CrowS-only Exp01 ---"
python experiments/02_component_dla.py \
  --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
  --pairs-limit 350 --top-k 20 --top-components-source mixed \
  --exp1-run-dir "$E1_CR" \
  --seed 42
E2_CR=$(_latest "02_component_dla" "$MODEL")
echo "E2_CR=$E2_CR"

echo "--- Exp09 refreshed (union, promoters-only, high-power heldout) ---"
python experiments/09_dla_atp_adjudication.py \
  --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
  --heldout-pairs 120 --top-k 20 --promoters-only --ranking-source union \
  --eval-sources "stereoset_intrasentence,crows_pairs" \
  --exp1-run-dir "$MIX_E1" --exp2-run-dir "$MIX_E2" --exp3-run-dir "$MIX_E3" \
  --seed 51
E9=$(_latest "09_dla_atp_adjudication" "$MODEL")
echo "E9=$E9"

echo "--- Exp11 A1 (Llama AtP-only promoters sweep) ---"
python experiments/11_hydra_multisite.py \
  --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
  --heldout-pairs 120 --n-sites "1,4,8" \
  --promoters-only --ranking-source atp \
  --bootstrap-n 1000 \
  --eval-sources "stereoset_intrasentence,crows_pairs" \
  --exp1-run-dir "$MIX_E1" --exp2-run-dir "$MIX_E2" --exp3-run-dir "$MIX_E3" \
  --seed 52

echo "--- Exp15 D1 matrix (Llama) ---"
python experiments/15_cross_dataset_component_transfer.py \
  --model "$MODEL" --device "$DEVICE" --torch-dtype "$DTYPE" \
  --heldout-pairs 120 --top-k 20 --bootstrap-n 1000 \
  --exp1-run-dir "$MIX_E1" \
  --exp2-stereoset-run-dir "$E2_SS" \
  --exp2-crows-run-dir "$E2_CR" \
  --seed 53

touch results/core_done_g2.flag
echo "=== GPU2 CORE DONE $(date) ==="
