#!/usr/bin/env bash

PROJECT_ROOT="/jumbo/lisp/f004ndc/StereACL"

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

cfg_get() {
  local path="$1"
  "$PYTHON_BIN" - "$CONFIG_PATH" "$path" <<'PY'
import sys, yaml
cfg_path, key_path = sys.argv[1], sys.argv[2]
obj = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
cur = obj
for part in key_path.split('.'):
    if part.isdigit():
        cur = cur[int(part)]
    else:
        cur = cur[part]
if isinstance(cur, bool):
    print('true' if cur else 'false')
else:
    print(cur)
PY
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

latest_run_by_seed() {
  local slug="$1"
  local model="$2"
  local seed="$3"
  "$PYTHON_BIN" - "$PROJECT_ROOT" "$slug" "$model" "$seed" "$ORCH_START_UTC" <<'PY'
import glob, json, sys
from datetime import datetime
from pathlib import Path

root = Path(sys.argv[1])
slug, model, seed_raw, min_time = sys.argv[2:6]
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
for p in glob.glob(str(root / 'results' / slug / '*' / '*' / 'manifest.json')):
    try:
        d = json.load(open(p, 'r', encoding='utf-8'))
    except Exception:
        continue
    if d.get('status') != 'completed':
        continue
    params = d.get('parameters', {})
    if params.get('model') != model:
        continue
    try:
        if int(params.get('seed', -10**9)) != seed:
            continue
    except Exception:
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
  "$PYTHON_BIN" - "$PROJECT_ROOT" "$slug" "$model" "$ORCH_START_UTC" <<'PY'
import glob, json, sys
from datetime import datetime
from pathlib import Path

root = Path(sys.argv[1])
slug, model, min_time = sys.argv[2:5]

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
for p in glob.glob(str(root / 'results' / slug / '*' / '*' / 'manifest.json')):
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

write_model_json() {
  local model_key="$1"
  local output_json="$2"
  "$PYTHON_BIN" - "$STATE_DIR" "$model_key" "$output_json" <<'PY'
import json
import sys
from pathlib import Path

state_dir = Path(sys.argv[1])
model_key = sys.argv[2]
out = Path(sys.argv[3])

vals = {}
for fp in sorted(state_dir.glob(f"{model_key}_*.txt")):
    key = fp.stem
    vals[key] = fp.read_text(encoding='utf-8').strip()

out.write_text(json.dumps(vals, indent=2, sort_keys=True), encoding='utf-8')
print(out)
PY
}

