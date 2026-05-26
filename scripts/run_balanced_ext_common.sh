#!/usr/bin/env bash

log_msg() {
  local msg="$1"
  echo "[$(date -Iseconds)] $msg" | tee -a "$LOG_FILE"
}

state_file() {
  local key="$1"
  echo "$STATE_DIR/${key}.txt"
}

flag_file() {
  local key="$1"
  echo "$STATE_DIR/${key}.flag"
}

write_state() {
  local key="$1"
  local value="$2"
  printf '%s\n' "$value" > "$(state_file "$key")"
}

read_state() {
  local key="$1"
  cat "$(state_file "$key")"
}

set_flag() {
  local key="$1"
  touch "$(flag_file "$key")"
}

wait_for_flag() {
  local key="$1"
  local fp
  fp="$(flag_file "$key")"
  while [[ ! -f "$fp" ]]; do
    log_msg "Waiting for flag: $fp"
    sleep 20
  done
  log_msg "Observed flag: $fp"
}

run_cmd() {
  local label="$1"
  shift
  local cmd="$*"
  log_msg "RUN [$label] $cmd"
  set +e
  bash -lc "$cmd" 2>&1 | tee -a "$LOG_FILE"
  local ec=${PIPESTATUS[0]}
  set -e
  if [[ $ec -ne 0 ]]; then
    log_msg "FAIL [$label] exit=$ec"
  else
    log_msg "OK   [$label]"
  fi
  return $ec
}

run_oom_variants() {
  local label="$1"
  shift
  local total=$#
  local idx=1
  local cmd
  for cmd in "$@"; do
    log_msg "TRY [$label] variant $idx/$total"
    local tmp
    tmp=$(mktemp)
    set +e
    bash -lc "$cmd" 2>&1 | tee -a "$LOG_FILE" | tee "$tmp"
    local ec=${PIPESTATUS[0]}
    set -e
    if [[ $ec -eq 0 ]]; then
      rm -f "$tmp"
      log_msg "OK   [$label] variant $idx"
      return 0
    fi
    if grep -Eiq "out of memory|OutOfMemoryError|CUDA out of memory|CUDA error: out of memory" "$tmp"; then
      log_msg "OOM detected for [$label] variant $idx"
      rm -f "$tmp"
      idx=$((idx + 1))
      continue
    fi
    rm -f "$tmp"
    log_msg "Non-OOM failure for [$label], not retrying variants"
    return $ec
  done
  log_msg "All OOM backoff variants exhausted for [$label]"
  return 1
}

latest_run_by_seed() {
  local slug="$1"
  local model="$2"
  local seed="$3"
  python3 - "$slug" "$model" "$seed" "$ORCH_START_UTC" <<'PY'
import glob, json, sys
from datetime import datetime

slug, model, seed_raw, min_time = sys.argv[1:5]
seed = int(seed_raw)

def parse_ts(ts: str):
    if not ts:
        return None
    ts = ts.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None

min_dt = parse_ts(min_time)
best = None
for p in glob.glob(f"results/{slug}/*/run-*/manifest.json"):
    try:
        d = json.load(open(p, 'r', encoding='utf-8'))
    except Exception:
        continue
    if d.get('status') != 'completed':
        continue
    params = d.get('parameters', {})
    if params.get('model') != model:
        continue
    if int(params.get('seed', -10**9)) != seed:
        continue
    ended = d.get('ended_at_utc', '')
    ended_dt = parse_ts(ended)
    if min_dt is not None and ended_dt is not None and ended_dt < min_dt:
        continue
    run_dir = d.get('run_dir', '')
    if not run_dir:
        continue
    if best is None or ended > best[0]:
        best = (ended, run_dir)
if best is None:
    raise SystemExit(1)
print(best[1])
PY
}

latest_run_any() {
  local slug="$1"
  local model="$2"
  local require_after_start="${3:-0}"
  python3 - "$slug" "$model" "$ORCH_START_UTC" "$require_after_start" <<'PY'
import glob, json, sys
from datetime import datetime

slug, model, min_time, require_after_start = sys.argv[1:5]
require_after_start = int(require_after_start)

def parse_ts(ts: str):
    if not ts:
        return None
    ts = ts.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None

min_dt = parse_ts(min_time)
best = None
for p in glob.glob(f"results/{slug}/*/*/manifest.json"):
    try:
        d = json.load(open(p, 'r', encoding='utf-8'))
    except Exception:
        continue
    if d.get('status') != 'completed':
        continue
    params = d.get('parameters', {})
    if params.get('model') != model:
        continue
    ended = d.get('ended_at_utc', '')
    ended_dt = parse_ts(ended)
    if require_after_start and min_dt is not None and ended_dt is not None and ended_dt < min_dt:
        continue
    run_dir = d.get('run_dir', '')
    if not run_dir:
        continue
    if best is None or ended > best[0]:
        best = (ended, run_dir)
if best is None:
    raise SystemExit(1)
print(best[1])
PY
}

run_packet_p1() {
  local key="$1"
  local model="$2"
  local dtype="$3"
  local is_7b="$4"
  local seed_base="$5"

  local s_e1_mix=$((seed_base + 1))
  local s_e1_ss=$((seed_base + 2))
  local s_e1_cr=$((seed_base + 3))
  local s_e2_mix=$((seed_base + 11))
  local s_e2_ss=$((seed_base + 12))
  local s_e2_cr=$((seed_base + 13))
  local s_e3=$((seed_base + 21))

  local e1_mix_pl=400
  local e1_ss_pl=400
  local e1_cr_pl=500
  local e2_mix_pl=300
  local e2_ss_pl=300
  local e2_cr_pl=350
  local e3_pl=150
  local e3_vpp=8

  if [[ "$is_7b" == "1" ]]; then
    e1_mix_pl=220
    e1_ss_pl=220
    e1_cr_pl=260
    e2_mix_pl=160
    e2_ss_pl=160
    e2_cr_pl=180
    e3_pl=80
    e3_vpp=6
  fi

  log_msg "=== P1 FOUNDATION START [$key | $model] ==="

  run_cmd "$key exp01-mixed" \
    "python3 experiments/01_layerwise_probing.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit $e1_mix_pl --per-source-limit 2500 --direction-position trait --seed $s_e1_mix"
  local e1_mix_dir
  e1_mix_dir=$(latest_run_by_seed "01_layerwise_probing" "$model" "$s_e1_mix")
  write_state "${key}_exp1_mixed_dir" "$e1_mix_dir"

  run_cmd "$key exp01-ss" \
    "python3 experiments/01_layerwise_probing.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit $e1_ss_pl --per-source-limit 2500 --direction-position trait --no-crows --no-seegull --seed $s_e1_ss"
  local e1_ss_dir
  e1_ss_dir=$(latest_run_by_seed "01_layerwise_probing" "$model" "$s_e1_ss")
  write_state "${key}_exp1_ss_dir" "$e1_ss_dir"

  run_cmd "$key exp01-cr" \
    "python3 experiments/01_layerwise_probing.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit $e1_cr_pl --per-source-limit 2500 --direction-position trait --no-stereoset --no-seegull --seed $s_e1_cr"
  local e1_cr_dir
  e1_cr_dir=$(latest_run_by_seed "01_layerwise_probing" "$model" "$s_e1_cr")
  write_state "${key}_exp1_cr_dir" "$e1_cr_dir"

  if [[ "$is_7b" == "1" ]]; then
    run_oom_variants "$key exp02-mixed" \
      "python3 experiments/02_component_dla.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit $e2_mix_pl --top-k 20 --top-components-source mixed --exp1-run-dir '$e1_mix_dir' --seed $s_e2_mix" \
      "python3 experiments/02_component_dla.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit $((e2_mix_pl*3/4)) --top-k 20 --top-components-source mixed --exp1-run-dir '$e1_mix_dir' --seed $s_e2_mix"
  else
    run_cmd "$key exp02-mixed" \
      "python3 experiments/02_component_dla.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit $e2_mix_pl --top-k 20 --top-components-source mixed --exp1-run-dir '$e1_mix_dir' --seed $s_e2_mix"
  fi
  local e2_mix_dir
  e2_mix_dir=$(latest_run_by_seed "02_component_dla" "$model" "$s_e2_mix")
  write_state "${key}_exp2_mixed_dir" "$e2_mix_dir"

  if [[ "$is_7b" == "1" ]]; then
    run_oom_variants "$key exp02-ss" \
      "python3 experiments/02_component_dla.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit $e2_ss_pl --top-k 20 --top-components-source mixed --exp1-run-dir '$e1_ss_dir' --seed $s_e2_ss" \
      "python3 experiments/02_component_dla.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit $((e2_ss_pl*3/4)) --top-k 20 --top-components-source mixed --exp1-run-dir '$e1_ss_dir' --seed $s_e2_ss"
  else
    run_cmd "$key exp02-ss" \
      "python3 experiments/02_component_dla.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit $e2_ss_pl --top-k 20 --top-components-source mixed --exp1-run-dir '$e1_ss_dir' --seed $s_e2_ss"
  fi
  local e2_ss_dir
  e2_ss_dir=$(latest_run_by_seed "02_component_dla" "$model" "$s_e2_ss")
  write_state "${key}_exp2_ss_dir" "$e2_ss_dir"

  if [[ "$is_7b" == "1" ]]; then
    run_oom_variants "$key exp02-cr" \
      "python3 experiments/02_component_dla.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit $e2_cr_pl --top-k 20 --top-components-source mixed --exp1-run-dir '$e1_cr_dir' --seed $s_e2_cr" \
      "python3 experiments/02_component_dla.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit $((e2_cr_pl*3/4)) --top-k 20 --top-components-source mixed --exp1-run-dir '$e1_cr_dir' --seed $s_e2_cr"
  else
    run_cmd "$key exp02-cr" \
      "python3 experiments/02_component_dla.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit $e2_cr_pl --top-k 20 --top-components-source mixed --exp1-run-dir '$e1_cr_dir' --seed $s_e2_cr"
  fi
  local e2_cr_dir
  e2_cr_dir=$(latest_run_by_seed "02_component_dla" "$model" "$s_e2_cr")
  write_state "${key}_exp2_cr_dir" "$e2_cr_dir"

  if [[ "$is_7b" == "1" ]]; then
    run_oom_variants "$key exp03" \
      "python3 experiments/03_attribution_patching.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit $e3_pl --top-k 20 --validation-pairs-per-component $e3_vpp --max-length 192 --exp1-run-dir '$e1_mix_dir' --exp2-run-dir '$e2_mix_dir' --seed $s_e3" \
      "python3 experiments/03_attribution_patching.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit 48 --top-k 10 --validation-pairs-per-component 3 --max-length 128 --exp1-run-dir '$e1_mix_dir' --exp2-run-dir '$e2_mix_dir' --seed $s_e3" \
      "python3 experiments/03_attribution_patching.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit 16 --top-k 6 --validation-pairs-per-component 2 --max-length 96 --exp1-run-dir '$e1_mix_dir' --exp2-run-dir '$e2_mix_dir' --seed $s_e3"
  else
    run_cmd "$key exp03" \
      "python3 experiments/03_attribution_patching.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --pairs-limit $e3_pl --top-k 20 --exp1-run-dir '$e1_mix_dir' --exp2-run-dir '$e2_mix_dir' --seed $s_e3"
  fi
  local e3_dir
  if e3_dir=$(latest_run_by_seed "03_attribution_patching" "$model" "$s_e3" 2>/dev/null); then
    :
  else
    log_msg "WARN [$key exp03] No fresh seed-matched completed run; falling back to latest completed Exp03 for model."
    e3_dir=$(latest_run_any "03_attribution_patching" "$model" 0)
  fi
  write_state "${key}_exp3_dir" "$e3_dir"

  set_flag "${key}_p1_done"
  log_msg "=== P1 FOUNDATION DONE [$key] ==="
}

run_packet_p2_multiseed() {
  local key="$1"
  local model="$2"
  local dtype="$3"
  local is_7b="$4"

  wait_for_flag "${key}_p1_done"

  local exp1_mix
  exp1_mix=$(read_state "${key}_exp1_mixed_dir")

  local held=120
  local held_low=80
  if [[ "$is_7b" == "1" ]]; then
    held=80
    held_low=60
  fi

  local seeds=(11 29 47)

  log_msg "=== P2 CONFIRMATORY START [$key | $model] ==="

  local s
  for s in "${seeds[@]}"; do
    if [[ "$is_7b" == "1" ]]; then
      run_oom_variants "$key exp16 seed=$s" \
        "python3 experiments/16_asymmetry_matrix.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --heldout-pairs $held --bootstrap-n 1000 --position-only --exp1-run-dir '$exp1_mix' --seed $s" \
        "python3 experiments/16_asymmetry_matrix.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --heldout-pairs $held_low --bootstrap-n 1000 --position-only --exp1-run-dir '$exp1_mix' --seed $s"

      run_oom_variants "$key exp18 seed=$s" \
        "python3 experiments/18_injection_controls.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --heldout-pairs $held --bootstrap-n 1000 --exp1-run-dir '$exp1_mix' --seed $s" \
        "python3 experiments/18_injection_controls.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --heldout-pairs $held_low --bootstrap-n 1000 --exp1-run-dir '$exp1_mix' --seed $s"
    else
      run_cmd "$key exp16 seed=$s" \
        "python3 experiments/16_asymmetry_matrix.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --heldout-pairs $held --bootstrap-n 1000 --position-only --exp1-run-dir '$exp1_mix' --seed $s"

      run_cmd "$key exp18 seed=$s" \
        "python3 experiments/18_injection_controls.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --heldout-pairs $held --bootstrap-n 1000 --exp1-run-dir '$exp1_mix' --seed $s"
    fi
  done

  set_flag "${key}_p2_done"
  log_msg "=== P2 CONFIRMATORY DONE [$key] ==="
}

run_packet_p3() {
  local key="$1"
  local model="$2"
  local dtype="$3"
  local is_7b="$4"
  local seed_base="$5"

  wait_for_flag "${key}_p1_done"

  local exp1_mix
  local exp2_mix
  local exp3_dir
  exp1_mix=$(read_state "${key}_exp1_mixed_dir")
  exp2_mix=$(read_state "${key}_exp2_mixed_dir")
  exp3_dir=$(read_state "${key}_exp3_dir")

  local held=120
  local held_low=80
  if [[ "$is_7b" == "1" ]]; then
    held=80
    held_low=60
  fi

  local s_e9=$((seed_base + 41))

  log_msg "=== P3 CALIBRATION START [$key | $model] ==="

  if [[ "$is_7b" == "1" ]]; then
    run_oom_variants "$key exp09" \
      "python3 experiments/09_dla_atp_adjudication.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --heldout-pairs $held --top-k 20 --promoters-only --ranking-source union --eval-sources 'stereoset_intrasentence,crows_pairs' --exp1-run-dir '$exp1_mix' --exp2-run-dir '$exp2_mix' --exp3-run-dir '$exp3_dir' --seed $s_e9" \
      "python3 experiments/09_dla_atp_adjudication.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --heldout-pairs $held_low --top-k 20 --promoters-only --ranking-source union --eval-sources 'stereoset_intrasentence,crows_pairs' --exp1-run-dir '$exp1_mix' --exp2-run-dir '$exp2_mix' --exp3-run-dir '$exp3_dir' --seed $s_e9"
  else
    run_cmd "$key exp09" \
      "python3 experiments/09_dla_atp_adjudication.py --model '$model' --device cuda:0 --torch-dtype '$dtype' --heldout-pairs $held --top-k 20 --promoters-only --ranking-source union --eval-sources 'stereoset_intrasentence,crows_pairs' --exp1-run-dir '$exp1_mix' --exp2-run-dir '$exp2_mix' --exp3-run-dir '$exp3_dir' --seed $s_e9"
  fi

  local exp9_dir
  exp9_dir=$(latest_run_by_seed "09_dla_atp_adjudication" "$model" "$s_e9")
  write_state "${key}_exp9_dir" "$exp9_dir"

  run_cmd "$key exp14" \
    "python3 experiments/14_sign_reliability_audit.py --model '$model' --exp2-run-dir '$exp2_mix' --exp3-run-dir '$exp3_dir' --exp9-run-dir '$exp9_dir'"

  run_cmd "$key exp17" \
    "python3 experiments/17_suppressor_contamination_audit.py --model '$model' --top-k 8 --ranking-source union --exp2-run-dir '$exp2_mix' --exp3-run-dir '$exp3_dir' --exp9-run-dir '$exp9_dir'"

  set_flag "${key}_p3_done"
  log_msg "=== P3 CALIBRATION DONE [$key] ==="
}
