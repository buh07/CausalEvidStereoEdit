#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
cd /jumbo/lisp/f004ndc/StereACL

RUN_TAG=${RUN_TAG:-balanced_ext_20260509_154409}
STATE_DIR="results/${RUN_TAG}/state"
mkdir -p "$STATE_DIR"
ORCH_START_UTC=$(cat "$STATE_DIR/orch_start_utc.txt")
LOG_FILE="results/${RUN_TAG}/log_mistral7b_sharded_resume.txt"

source ./scripts/run_balanced_ext_common.sh

key="mistral7b"
model="/jumbo/lisp/f004ndc/models/mistral-7b-v0.1"
dtype="float16"
seed_base=1600
held=80

e1_mix=$(read_state "${key}_exp1_mixed_dir")
e2_mix=$(read_state "${key}_exp2_mixed_dir")
e2_ss=$(read_state "${key}_exp2_ss_dir")
e2_cr=$(read_state "${key}_exp2_cr_dir")

s_e3=$((seed_base + 21))
s_e4=$((seed_base + 31))
s_e16=$((seed_base + 32))
s_e18=$((seed_base + 33))
s_e10r=$((seed_base + 34))
s_e10a=$((seed_base + 35))
s_e10m=$((seed_base + 36))
s_e9=$((seed_base + 41))
s_e15=$((seed_base + 42))

log_msg "Mistral 7B sharded resume start | RUN_TAG=$RUN_TAG"

if [[ -f "$(state_file "${key}_exp3_dir")" && -f "$(flag_file "${key}_p1_done")" ]]; then
  exp3_dir=$(read_state "${key}_exp3_dir")
  log_msg "Reusing existing Exp03 output for $key: $exp3_dir"
else
  run_oom_variants "$key exp03-sharded" \
    "python3 experiments/03_attribution_patching.py --model '$model' --device shard-auto --torch-dtype '$dtype' --pairs-limit 80 --top-k 20 --validation-pairs-per-component 6 --exp1-run-dir '$e1_mix' --exp2-run-dir '$e2_mix' --seed $s_e3" \
    "python3 experiments/03_attribution_patching.py --model '$model' --device shard-auto --torch-dtype '$dtype' --pairs-limit 60 --top-k 20 --validation-pairs-per-component 6 --exp1-run-dir '$e1_mix' --exp2-run-dir '$e2_mix' --seed $s_e3" \
    "python3 experiments/03_attribution_patching.py --model '$model' --device shard-auto --torch-dtype '$dtype' --pairs-limit 40 --top-k 20 --validation-pairs-per-component 6 --exp1-run-dir '$e1_mix' --exp2-run-dir '$e2_mix' --seed $s_e3"

  exp3_dir=$(latest_run_by_seed "03_attribution_patching" "$model" "$s_e3")
  write_state "${key}_exp3_dir" "$exp3_dir"
  set_flag "${key}_p1_done"
fi

run_oom_variants "$key exp04-sharded" \
  "python3 experiments/04_ablation_validation.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs $held --top-k-components 20 --strict-controls --bootstrap-n 500 --on-manifold --bbq-samples 0 --mmlu-samples 0 --mmlu-shots 5 --exp1-run-dir '$e1_mix' --exp3-run-dir '$exp3_dir' --seed $s_e4" \
  "python3 experiments/04_ablation_validation.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs 60 --top-k-components 20 --strict-controls --bootstrap-n 500 --on-manifold --bbq-samples 0 --mmlu-samples 0 --mmlu-shots 5 --exp1-run-dir '$e1_mix' --exp3-run-dir '$exp3_dir' --seed $s_e4"

run_oom_variants "$key exp16-sharded" \
  "python3 experiments/16_asymmetry_matrix.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs $held --bootstrap-n 1000 --position-only --exp1-run-dir '$e1_mix' --seed $s_e16" \
  "python3 experiments/16_asymmetry_matrix.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs 60 --bootstrap-n 1000 --position-only --exp1-run-dir '$e1_mix' --seed $s_e16"

run_oom_variants "$key exp18-sharded" \
  "python3 experiments/18_injection_controls.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs $held --bootstrap-n 1000 --exp1-run-dir '$e1_mix' --seed $s_e18" \
  "python3 experiments/18_injection_controls.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs 60 --bootstrap-n 1000 --exp1-run-dir '$e1_mix' --seed $s_e18"

run_oom_variants "$key exp10-residual-sharded" \
  "python3 experiments/10_path_mediation.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs $held --ablation-target residual --exp1-run-dir '$e1_mix' --seed $s_e10r" \
  "python3 experiments/10_path_mediation.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs 60 --ablation-target residual --exp1-run-dir '$e1_mix' --seed $s_e10r"

run_oom_variants "$key exp10-attention-sharded" \
  "python3 experiments/10_path_mediation.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs $held --ablation-target attention --exp1-run-dir '$e1_mix' --seed $s_e10a" \
  "python3 experiments/10_path_mediation.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs 60 --ablation-target attention --exp1-run-dir '$e1_mix' --seed $s_e10a"

run_oom_variants "$key exp10-mlp-sharded" \
  "python3 experiments/10_path_mediation.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs $held --ablation-target mlp --exp1-run-dir '$e1_mix' --seed $s_e10m" \
  "python3 experiments/10_path_mediation.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs 60 --ablation-target mlp --exp1-run-dir '$e1_mix' --seed $s_e10m"

set_flag "${key}_p2_done"

run_oom_variants "$key exp09-sharded" \
  "python3 experiments/09_dla_atp_adjudication.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs $held --top-k 20 --promoters-only --ranking-source union --eval-sources 'stereoset_intrasentence,crows_pairs' --exp1-run-dir '$e1_mix' --exp2-run-dir '$e2_mix' --exp3-run-dir '$exp3_dir' --seed $s_e9" \
  "python3 experiments/09_dla_atp_adjudication.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs 60 --top-k 20 --promoters-only --ranking-source union --eval-sources 'stereoset_intrasentence,crows_pairs' --exp1-run-dir '$e1_mix' --exp2-run-dir '$e2_mix' --exp3-run-dir '$exp3_dir' --seed $s_e9"

exp9_dir=$(latest_run_by_seed "09_dla_atp_adjudication" "$model" "$s_e9")
write_state "${key}_exp9_dir" "$exp9_dir"

run_cmd "$key exp14" "python3 experiments/14_sign_reliability_audit.py --model '$model' --exp2-run-dir '$e2_mix' --exp3-run-dir '$exp3_dir' --exp9-run-dir '$exp9_dir'"
run_cmd "$key exp17" "python3 experiments/17_suppressor_contamination_audit.py --model '$model' --top-k 8 --ranking-source union --exp2-run-dir '$e2_mix' --exp3-run-dir '$exp3_dir' --exp9-run-dir '$exp9_dir'"

run_oom_variants "$key exp15-sharded" \
  "python3 experiments/15_cross_dataset_component_transfer.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs $held --top-k 20 --bootstrap-n 1000 --exp1-run-dir '$e1_mix' --exp2-stereoset-run-dir '$e2_ss' --exp2-crows-run-dir '$e2_cr' --seed $s_e15" \
  "python3 experiments/15_cross_dataset_component_transfer.py --model '$model' --device shard-auto --torch-dtype '$dtype' --heldout-pairs 60 --top-k 20 --bootstrap-n 1000 --exp1-run-dir '$e1_mix' --exp2-stereoset-run-dir '$e2_ss' --exp2-crows-run-dir '$e2_cr' --seed $s_e15"

set_flag "${key}_p3_done"
log_msg "Mistral 7B sharded resume done"
