#!/bin/bash
# GPU 3: Orchestrate A2 Exp14 audits after GPU0/1/2 complete core chains.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
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

rm -f results/core_done_g3.flag

echo "=== GPU3 CORE START $(date) ==="

echo "Waiting for GPU0/1/2 done flags..."
while true; do
  if [[ -f results/core_done_g0.flag && -f results/core_done_g1.flag && -f results/core_done_g2.flag ]]; then
    break
  fi
  sleep 20
done

echo "All dependency chains complete. Running Exp14 audits..."

# Gemma-2-2B
MODEL="google/gemma-2-2b"
E2="results/02_component_dla/2026-05-07/run-012"
E3="results/03_attribution_patching/2026-05-07/run-020"
E9=$(_latest "09_dla_atp_adjudication" "$MODEL")
python experiments/14_sign_reliability_audit.py --model "$MODEL" --exp2-run-dir "$E2" --exp3-run-dir "$E3" --exp9-run-dir "$E9"

# Gemma-2-2B-IT
MODEL="google/gemma-2-2b-it"
E2="results/02_component_dla/2026-05-07/run-017"
E3="results/03_attribution_patching/2026-05-07/run-022"
E9=$(_latest "09_dla_atp_adjudication" "$MODEL")
python experiments/14_sign_reliability_audit.py --model "$MODEL" --exp2-run-dir "$E2" --exp3-run-dir "$E3" --exp9-run-dir "$E9"

# Llama-3.2-3B
MODEL="meta-llama/Llama-3.2-3B"
E2="results/02_component_dla/2026-05-07/run-013"
E3="results/03_attribution_patching/2026-05-07/run-021"
E9=$(_latest "09_dla_atp_adjudication" "$MODEL")
python experiments/14_sign_reliability_audit.py --model "$MODEL" --exp2-run-dir "$E2" --exp3-run-dir "$E3" --exp9-run-dir "$E9"

touch results/core_done_g3.flag
echo "=== GPU3 CORE DONE $(date) ==="
