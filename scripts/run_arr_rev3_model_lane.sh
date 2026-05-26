#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <model_key> <model_name> <gpu_id> <lane_kind:core|extension>" >&2
  exit 2
fi

MODEL_KEY="$1"
MODEL_NAME="$2"
GPU_ID="$3"
LANE_KIND="$4"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /jumbo/lisp/f004ndc/StereACL

RUN_TAG="${RUN_TAG:?RUN_TAG must be set}"
CONFIG_PATH="${CONFIG_PATH:-configs/arr_rev3_freeze.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STATE_DIR="results/${RUN_TAG}/state"
LOG_DIR="results/${RUN_TAG}/logs"
mkdir -p "$STATE_DIR" "$LOG_DIR"
ORCH_START_UTC="$(cat "$STATE_DIR/orch_start_utc.txt")"
LOG_FILE="$LOG_DIR/${MODEL_KEY}.log"

source ./scripts/may_arr_fixpack_common.sh

model_field() {
  local model_key="$1"
  local field="$2"
  "$PYTHON_BIN" - "$CONFIG_PATH" "$model_key" "$field" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))
key = sys.argv[2]
field = sys.argv[3]
rows = []
rows.extend(cfg.get('core_models', []) or [])
rows.extend(cfg.get('extension_models', []) or [])
for r in rows:
    if str(r.get('key')) == key:
        val = r.get(field, "")
        if isinstance(val, bool):
            print('true' if val else 'false')
        else:
            print(val)
        sys.exit(0)
raise SystemExit(1)
PY
}

on_err() {
  local ec=$?
  log_msg "Lane failed (exit=$ec)."
  set_flag "${MODEL_KEY}_failed"
  exit "$ec"
}
trap on_err ERR

log_msg "Lane start | model_key=$MODEL_KEY model=$MODEL_NAME gpu=$GPU_ID lane=$LANE_KIND run_tag=$RUN_TAG"

DEVICE="$(cfg_get global.device)"
MAXLEN="$(cfg_get global.max_length)"
TOPK="$(cfg_get global.top_k)"
BOOT="$(cfg_get global.bootstrap_n)"
EVAL_SOURCES="$(cfg_get global.eval_sources)"

DTYPE="$(model_field "$MODEL_KEY" torch_dtype)"
HELDOUT="$(model_field "$MODEL_KEY" heldout_pairs)"

SEED_CANON="$(cfg_get frozen_seeds.canonical)"
SEED_LIST="$($PYTHON_BIN - "$CONFIG_PATH" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))
print(' '.join(str(x) for x in cfg['frozen_seeds']['seed_aggregate']))
PY
)"
read -r -a SEEDS_AGG <<< "$SEED_LIST"

XFIT_SEEDS="$($PYTHON_BIN - "$CONFIG_PATH" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))
print(' '.join(str(x) for x in cfg['frozen_seeds']['crossfit']))
PY
)"
read -r -a SEEDS_XFIT <<< "$XFIT_SEEDS"

E1_MIX_PER_SOURCE="$(cfg_get exp01.mixed.per_source_limit)"
if [[ "$LANE_KIND" == "core" ]]; then
  E1_MIX_PAIRS="$(cfg_get exp01.mixed.pairs_limit)"
else
  if [[ "$MODEL_KEY" == "mistral7b" || "$MODEL_KEY" == "olmo7b" ]]; then
    E1_MIX_PAIRS="$(cfg_get exp01.extension.pairs_limit_7b)"
  else
    E1_MIX_PAIRS="$(cfg_get exp01.extension.pairs_limit_3b)"
  fi
fi
E1_DIR_POS="$(cfg_get exp01.mixed.direction_position)"
E1_SS_PAIRS="$(cfg_get exp01.stereoset_only.pairs_limit)"
E1_CR_PAIRS="$(cfg_get exp01.crows_only.pairs_limit)"

E2_STRICT_SPLIT="$(cfg_get exp02.strict.split_scope)"
E2_STRICT_PAIRS="$(cfg_get exp02.strict.pairs_limit)"
E2_SRC="$(cfg_get exp02.strict.top_components_source)"
E2_SS_PAIRS="$(cfg_get exp02.transfer_source_specific.pairs_limit_stereoset)"
E2_CR_PAIRS="$(cfg_get exp02.transfer_source_specific.pairs_limit_crows)"

E3_SPLIT="$(cfg_get exp03.strict.split_scope)"
E3_PAIRS="$(cfg_get exp03.strict.pairs_limit)"
E3_VPP="$(cfg_get exp03.strict.validation_pairs_per_component)"

E15_PROMO="$(cfg_get exp15.promoters_only)"
E21_SESOI="$(cfg_get exp21.sesoi)"
E21_ALPHA="$(cfg_get exp21.alpha)"
E21_POWER="$(cfg_get exp21.target_power)"
E26_TRAIN="$(cfg_get exp26.train_pairs)"
E26_VARIANT="$(cfg_get exp26.prompt_variant)"
E27_MAX_SPAN="$(cfg_get exp27.max_span_len)"
E28_SCOPE="$(cfg_get exp28.eval_scope)"
E28_MAX_SPAN="$(cfg_get exp28.max_span_len)"
E28_SPAN_STRATA="$(cfg_get exp28.span_strata)"
E28_SESOI="$(cfg_get exp28.sesoi)"
E28_ALPHA="$(cfg_get exp28.alpha)"
E28_POWER="$(cfg_get exp28.target_power)"
E28_BAL_MODE="$(cfg_get exp28.balanced.mode)"
E28_BAL_QUOTAS="$(cfg_get exp28.balanced.source_quotas)"
E28_BAL_SEED="$(cfg_get exp28.balanced.balance_seed)"
E29_PREFIX="$(cfg_get exp29.mitigation_prefix)"
E07_RANKS="$(cfg_get exp07.ranks)"
E07_BASIS="$(cfg_get exp07.basis_mode)"

# Distinct seeds for source-specific exp01/exp02 runs.
SEED_SS=29
SEED_CR=47

# Exp01 mixed
run_cmd "exp01-mixed" "$PYTHON_BIN experiments/01_layerwise_probing.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --pairs-limit $E1_MIX_PAIRS --per-source-limit $E1_MIX_PER_SOURCE --direction-position '$E1_DIR_POS' --max-length $MAXLEN --seed $SEED_CANON"
E1_MIX_DIR="$(latest_run_by_seed '01_layerwise_probing' "$MODEL_NAME" "$SEED_CANON")"
write_state "${MODEL_KEY}_exp01_mixed_run_dir" "$E1_MIX_DIR"

# Source-specific pools
run_cmd "exp01-stereoset" "$PYTHON_BIN experiments/01_layerwise_probing.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --pairs-limit $E1_SS_PAIRS --per-source-limit $E1_MIX_PER_SOURCE --no-crows --no-seegull --max-length $MAXLEN --seed $SEED_SS"
E1_SS_DIR="$(latest_run_by_seed '01_layerwise_probing' "$MODEL_NAME" "$SEED_SS")"
write_state "${MODEL_KEY}_exp01_stereoset_run_dir" "$E1_SS_DIR"

run_cmd "exp01-crows" "$PYTHON_BIN experiments/01_layerwise_probing.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --pairs-limit $E1_CR_PAIRS --per-source-limit $E1_MIX_PER_SOURCE --no-stereoset --no-seegull --max-length $MAXLEN --seed $SEED_CR"
E1_CR_DIR="$(latest_run_by_seed '01_layerwise_probing' "$MODEL_NAME" "$SEED_CR")"
write_state "${MODEL_KEY}_exp01_crows_run_dir" "$E1_CR_DIR"

# Exp16 canonical + seed aggregate (with occupancy for canonical)
for seed in "${SEEDS_AGG[@]}"; do
  OCC_FLAG=""
  if [[ "$seed" == "$SEED_CANON" ]]; then
    OCC_FLAG="--emit-occupancy"
  fi
  run_cmd "exp16-seed-${seed}" "$PYTHON_BIN experiments/16_asymmetry_matrix.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --heldout-pairs $HELDOUT --bootstrap-n $BOOT --position-only --emit-pair-level $OCC_FLAG --exp1-run-dir '$E1_MIX_DIR' --max-length $MAXLEN --seed $seed"
  E16_DIR="$(latest_run_by_seed '16_asymmetry_matrix' "$MODEL_NAME" "$seed")"
  write_state "${MODEL_KEY}_exp16_seed_${seed}_run_dir" "$E16_DIR"
  if [[ "$seed" == "$SEED_CANON" ]]; then
    write_state "${MODEL_KEY}_exp16_canonical_run_dir" "$E16_DIR"
  fi
done
E16_CANON="$(read_state "${MODEL_KEY}_exp16_canonical_run_dir")"

# Composition sensitivity pools for Exp16
run_cmd "exp16-source-stereoset" "$PYTHON_BIN experiments/16_asymmetry_matrix.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --heldout-pairs $HELDOUT --bootstrap-n $BOOT --position-only --emit-pair-level --eval-exp1-run-dir '$E1_MIX_DIR' --directions-exp1-run-dir '$E1_SS_DIR' --max-length $MAXLEN --seed $SEED_CANON"
E16_SRC_SS_DIR="$(latest_run_by_seed '16_asymmetry_matrix' "$MODEL_NAME" "$SEED_CANON")"
write_state "${MODEL_KEY}_exp16_source_stereoset_run_dir" "$E16_SRC_SS_DIR"

run_cmd "exp16-source-crows" "$PYTHON_BIN experiments/16_asymmetry_matrix.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --heldout-pairs $HELDOUT --bootstrap-n $BOOT --position-only --emit-pair-level --eval-exp1-run-dir '$E1_MIX_DIR' --directions-exp1-run-dir '$E1_CR_DIR' --max-length $MAXLEN --seed $SEED_CANON"
E16_SRC_CR_DIR="$(latest_run_by_seed '16_asymmetry_matrix' "$MODEL_NAME" "$SEED_CANON")"
write_state "${MODEL_KEY}_exp16_source_crows_run_dir" "$E16_SRC_CR_DIR"

if [[ "$LANE_KIND" == "core" ]]; then
  # Strict split-clean family
  run_cmd "exp02-strict" "$PYTHON_BIN experiments/02_component_dla.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --pairs-limit $E2_STRICT_PAIRS --top-k $TOPK --top-components-source '$E2_SRC' --split-scope '$E2_STRICT_SPLIT' --exp1-run-dir '$E1_MIX_DIR' --max-length $MAXLEN --seed $SEED_CANON"
  E2_STRICT_DIR="$(latest_run_by_seed '02_component_dla' "$MODEL_NAME" "$SEED_CANON")"
  write_state "${MODEL_KEY}_exp02_strict_run_dir" "$E2_STRICT_DIR"

  run_cmd "exp03-strict" "$PYTHON_BIN experiments/03_attribution_patching.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --pairs-limit $E3_PAIRS --top-k $TOPK --validation-pairs-per-component $E3_VPP --split-scope '$E3_SPLIT' --exp1-run-dir '$E1_MIX_DIR' --exp2-run-dir '$E2_STRICT_DIR' --max-length $MAXLEN --seed $SEED_CANON"
  E3_STRICT_DIR="$(latest_run_by_seed '03_attribution_patching' "$MODEL_NAME" "$SEED_CANON")"
  write_state "${MODEL_KEY}_exp03_strict_run_dir" "$E3_STRICT_DIR"

  run_cmd "exp09-strict" "$PYTHON_BIN experiments/09_dla_atp_adjudication.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --heldout-pairs $HELDOUT --top-k $TOPK --promoters-only --ranking-source union --eval-sources '$EVAL_SOURCES' --exp1-run-dir '$E1_MIX_DIR' --exp2-run-dir '$E2_STRICT_DIR' --exp3-run-dir '$E3_STRICT_DIR' --max-length $MAXLEN --seed $SEED_CANON"
  E9_DIR="$(latest_run_by_seed '09_dla_atp_adjudication' "$MODEL_NAME" "$SEED_CANON")"
  write_state "${MODEL_KEY}_exp09_strict_run_dir" "$E9_DIR"

  run_cmd "exp14-strict" "$PYTHON_BIN experiments/14_sign_reliability_audit.py --model '$MODEL_NAME' --exp2-run-dir '$E2_STRICT_DIR' --exp3-run-dir '$E3_STRICT_DIR' --exp9-run-dir '$E9_DIR'"
  E14_DIR="$(latest_run_any '14_sign_reliability_audit' "$MODEL_NAME")"
  write_state "${MODEL_KEY}_exp14_run_dir" "$E14_DIR"

  run_cmd "exp17-strict" "$PYTHON_BIN experiments/17_suppressor_contamination_audit.py --model '$MODEL_NAME' --top-k 8 --ranking-source union --exp2-run-dir '$E2_STRICT_DIR' --exp3-run-dir '$E3_STRICT_DIR' --exp9-run-dir '$E9_DIR'"
  E17_DIR="$(latest_run_any '17_suppressor_contamination_audit' "$MODEL_NAME")"
  write_state "${MODEL_KEY}_exp17_run_dir" "$E17_DIR"

  # Cross-fit strict reruns for ranking-dependent diagnostics
  for seed in "${SEEDS_XFIT[@]}"; do
    run_cmd "exp02-crossfit-${seed}" "$PYTHON_BIN experiments/02_component_dla.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --pairs-limit $E2_STRICT_PAIRS --top-k $TOPK --top-components-source '$E2_SRC' --split-scope '$E2_STRICT_SPLIT' --exp1-run-dir '$E1_MIX_DIR' --max-length $MAXLEN --seed $seed"
    E2_CF_DIR="$(latest_run_by_seed '02_component_dla' "$MODEL_NAME" "$seed")"

    run_cmd "exp03-crossfit-${seed}" "$PYTHON_BIN experiments/03_attribution_patching.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --pairs-limit $E3_PAIRS --top-k $TOPK --validation-pairs-per-component $E3_VPP --split-scope '$E3_SPLIT' --exp1-run-dir '$E1_MIX_DIR' --exp2-run-dir '$E2_CF_DIR' --max-length $MAXLEN --seed $seed"
    E3_CF_DIR="$(latest_run_by_seed '03_attribution_patching' "$MODEL_NAME" "$seed")"

    run_cmd "exp09-crossfit-${seed}" "$PYTHON_BIN experiments/09_dla_atp_adjudication.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --heldout-pairs $HELDOUT --top-k $TOPK --promoters-only --ranking-source union --eval-sources '$EVAL_SOURCES' --exp1-run-dir '$E1_MIX_DIR' --exp2-run-dir '$E2_CF_DIR' --exp3-run-dir '$E3_CF_DIR' --max-length $MAXLEN --seed $seed"
    E9_CF_DIR="$(latest_run_by_seed '09_dla_atp_adjudication' "$MODEL_NAME" "$seed")"

    run_cmd "exp14-crossfit-${seed}" "$PYTHON_BIN experiments/14_sign_reliability_audit.py --model '$MODEL_NAME' --exp2-run-dir '$E2_CF_DIR' --exp3-run-dir '$E3_CF_DIR' --exp9-run-dir '$E9_CF_DIR'"
    E14_CF_DIR="$(latest_run_any '14_sign_reliability_audit' "$MODEL_NAME")"
    write_state "${MODEL_KEY}_exp14_crossfit_seed_${seed}_run_dir" "$E14_CF_DIR"

    run_cmd "exp17-crossfit-${seed}" "$PYTHON_BIN experiments/17_suppressor_contamination_audit.py --model '$MODEL_NAME' --top-k 8 --ranking-source union --exp2-run-dir '$E2_CF_DIR' --exp3-run-dir '$E3_CF_DIR' --exp9-run-dir '$E9_CF_DIR'"
    E17_CF_DIR="$(latest_run_any '17_suppressor_contamination_audit' "$MODEL_NAME")"
    write_state "${MODEL_KEY}_exp17_crossfit_seed_${seed}_run_dir" "$E17_CF_DIR"
  done

  # Source-specific ranking families for transfer
  run_cmd "exp02-stereoset-rank" "$PYTHON_BIN experiments/02_component_dla.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --pairs-limit $E2_SS_PAIRS --top-k $TOPK --top-components-source '$E2_SRC' --split-scope '$E2_STRICT_SPLIT' --exp1-run-dir '$E1_SS_DIR' --max-length $MAXLEN --seed $SEED_SS"
  E2_SS_DIR="$(latest_run_by_seed '02_component_dla' "$MODEL_NAME" "$SEED_SS")"
  write_state "${MODEL_KEY}_exp02_stereoset_run_dir" "$E2_SS_DIR"

  run_cmd "exp02-crows-rank" "$PYTHON_BIN experiments/02_component_dla.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --pairs-limit $E2_CR_PAIRS --top-k $TOPK --top-components-source '$E2_SRC' --split-scope '$E2_STRICT_SPLIT' --exp1-run-dir '$E1_CR_DIR' --max-length $MAXLEN --seed $SEED_CR"
  E2_CR_DIR="$(latest_run_by_seed '02_component_dla' "$MODEL_NAME" "$SEED_CR")"
  write_state "${MODEL_KEY}_exp02_crows_run_dir" "$E2_CR_DIR"

  if [[ "$E15_PROMO" == "true" ]]; then
    PROMO_FLAG="--promoters-only"
  else
    PROMO_FLAG=""
  fi
  run_cmd "exp15-transfer" "$PYTHON_BIN experiments/15_cross_dataset_component_transfer.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --heldout-pairs $HELDOUT --top-k $TOPK --bootstrap-n $BOOT $PROMO_FLAG --exp1-run-dir '$E1_MIX_DIR' --exp2-stereoset-run-dir '$E2_SS_DIR' --exp2-crows-run-dir '$E2_CR_DIR' --max-length $MAXLEN --seed $SEED_CANON"
  E15_DIR="$(latest_run_by_seed '15_cross_dataset_component_transfer' "$MODEL_NAME" "$SEED_CANON")"
  write_state "${MODEL_KEY}_exp15_run_dir" "$E15_DIR"

  run_cmd "exp21-equivalence" "$PYTHON_BIN experiments/21_transfer_equivalence.py --model '$MODEL_NAME' --sesoi $E21_SESOI --alpha $E21_ALPHA --target-power $E21_POWER --exp15-run-dir '$E15_DIR'"
  E21_DIR="$(latest_run_any '21_transfer_equivalence' "$MODEL_NAME")"
  write_state "${MODEL_KEY}_exp21_run_dir" "$E21_DIR"

  # Same-position AR + template sensitivity
  run_cmd "exp26-ar-main" "$PYTHON_BIN experiments/26_ar_same_position_replication.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --heldout-pairs $HELDOUT --train-pairs $E26_TRAIN --bootstrap-n $BOOT --prompt-variant '$E26_VARIANT' --emit-pair-level --emit-occupancy --exp1-run-dir '$E1_MIX_DIR' --max-length $MAXLEN --seed $SEED_CANON"
  E26_DIR="$(latest_run_by_seed '26_ar_same_position_replication' "$MODEL_NAME" "$SEED_CANON")"
  write_state "${MODEL_KEY}_exp26_run_dir" "$E26_DIR"

  for v in so_next_word plain_suffix; do
    run_cmd "exp26-ar-variant-${v}" "$PYTHON_BIN experiments/26_ar_same_position_replication.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --heldout-pairs $HELDOUT --train-pairs $E26_TRAIN --bootstrap-n $BOOT --prompt-variant '$v' --emit-pair-level --exp1-run-dir '$E1_MIX_DIR' --max-length $MAXLEN --seed $SEED_CANON"
    E26V_DIR="$(latest_run_by_seed '26_ar_same_position_replication' "$MODEL_NAME" "$SEED_CANON")"
    write_state "${MODEL_KEY}_exp26_variant_${v}_run_dir" "$E26V_DIR"
  done

  run_cmd "exp27-multitoken" "$PYTHON_BIN experiments/27_multitoken_span_robustness.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --heldout-pairs $HELDOUT --max-span-len $E27_MAX_SPAN --bootstrap-n $BOOT --exp1-run-dir '$E1_MIX_DIR' --max-length $MAXLEN --seed $SEED_CANON"
  E27_DIR="$(latest_run_by_seed '27_multitoken_span_robustness' "$MODEL_NAME" "$SEED_CANON")"
  write_state "${MODEL_KEY}_exp27_run_dir" "$E27_DIR"

  run_cmd "exp28-multitoken-unbalanced" "$PYTHON_BIN experiments/28_multitoken_matched_asymmetry.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --heldout-pairs $HELDOUT --eval-scope '$E28_SCOPE' --max-span-len $E28_MAX_SPAN --span-strata '$E28_SPAN_STRATA' --bootstrap-n $BOOT --power-alpha $E28_ALPHA --target-power $E28_POWER --sesoi $E28_SESOI --balance-mode none --exp1-run-dir '$E1_MIX_DIR' --max-length $MAXLEN --seed $SEED_CANON"
  E28_UNBAL_DIR="$(latest_run_by_seed '28_multitoken_matched_asymmetry' "$MODEL_NAME" "$SEED_CANON")"
  write_state "${MODEL_KEY}_exp28_run_dir" "$E28_UNBAL_DIR"

  run_cmd "exp28-multitoken-balanced" "$PYTHON_BIN experiments/28_multitoken_matched_asymmetry.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --heldout-pairs $HELDOUT --eval-scope '$E28_SCOPE' --max-span-len $E28_MAX_SPAN --span-strata '$E28_SPAN_STRATA' --bootstrap-n $BOOT --power-alpha $E28_ALPHA --target-power $E28_POWER --sesoi $E28_SESOI --balance-mode '$E28_BAL_MODE' --source-quotas '$E28_BAL_QUOTAS' --balance-seed $E28_BAL_SEED --exp1-run-dir '$E1_MIX_DIR' --max-length $MAXLEN --seed $SEED_CANON"
  E28_BAL_DIR="$(latest_run_by_seed '28_multitoken_matched_asymmetry' "$MODEL_NAME" "$SEED_CANON")"
  write_state "${MODEL_KEY}_exp28_balanced_run_dir" "$E28_BAL_DIR"

  run_cmd "exp29-prompt" "$PYTHON_BIN experiments/29_prompt_calibration_baseline.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --heldout-pairs $HELDOUT --bootstrap-n $BOOT --exp1-run-dir '$E1_MIX_DIR' --exp16-run-dir '$E16_CANON' --mitigation-prefix '$E29_PREFIX' --max-length $MAXLEN --seed $SEED_CANON"
  E29_DIR="$(latest_run_by_seed '29_prompt_calibration_baseline' "$MODEL_NAME" "$SEED_CANON")"
  write_state "${MODEL_KEY}_exp29_run_dir" "$E29_DIR"

  run_cmd "exp30-injection-transfer" "$PYTHON_BIN experiments/30_cross_dataset_direction_injection_transfer.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --heldout-pairs $HELDOUT --bootstrap-n $BOOT --exp1-mixed-run-dir '$E1_MIX_DIR' --exp1-stereoset-run-dir '$E1_SS_DIR' --exp1-crows-run-dir '$E1_CR_DIR' --max-length $MAXLEN --seed $SEED_CANON"
  E30_DIR="$(latest_run_by_seed '30_cross_dataset_direction_injection_transfer' "$MODEL_NAME" "$SEED_CANON")"
  write_state "${MODEL_KEY}_exp30_run_dir" "$E30_DIR"

  run_cmd "exp07-rank-sweep" "$PYTHON_BIN experiments/07_rank_sweep.py --model '$MODEL_NAME' --device '$DEVICE' --torch-dtype '$DTYPE' --heldout-pairs $HELDOUT --ranks '$E07_RANKS' --basis-mode '$E07_BASIS' --exp1-run-dir '$E1_MIX_DIR' --max-length $MAXLEN --seed $SEED_CANON"
  E07_DIR="$(latest_run_by_seed '07_rank_sweep' "$MODEL_NAME" "$SEED_CANON")"
  write_state "${MODEL_KEY}_exp07_run_dir" "$E07_DIR"

fi

write_state "${MODEL_KEY}_model" "$MODEL_NAME"
write_state "${MODEL_KEY}_label" "$(model_field "$MODEL_KEY" label)"
write_model_json "$MODEL_KEY" "$STATE_DIR/${MODEL_KEY}_runs.json"
set_flag "${MODEL_KEY}_done"
log_msg "Lane complete | model_key=$MODEL_KEY"
