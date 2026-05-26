#!/usr/bin/env bash
set -euo pipefail

cd /jumbo/lisp/f004ndc/StereACL

RUN_TAG="${RUN_TAG:-may_arr_fixpack_$(date -u +%Y%m%d_%H%M%S)_rev2}"
CONFIG_PATH="${CONFIG_PATH:-configs/may_arr_freeze_v1.yaml}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "/jumbo/lisp/f004ndc/StereACL/.venv_may_arr/bin/python" ]]; then
  PY="/jumbo/lisp/f004ndc/StereACL/.venv_may_arr/bin/python"
else
  PY="python"
fi

STATE_DIR="results/${RUN_TAG}/state"
LOG_DIR="results/${RUN_TAG}/logs"
mkdir -p "$STATE_DIR" "$LOG_DIR"
printf '%s\n' "$(date -u -Iseconds)" > "$STATE_DIR/orch_start_utc.txt"

for s in mayarr-g0 mayarr-g1 mayarr-g2 mayarr-g3; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

tmux new-session -d -s mayarr-g0 "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG='$RUN_TAG' CONFIG_PATH='$CONFIG_PATH' PYTHON_BIN='$PY' bash scripts/run_may_arr_model_lane.sh gemma2b google/gemma-2-2b 0"
tmux new-session -d -s mayarr-g1 "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG='$RUN_TAG' CONFIG_PATH='$CONFIG_PATH' PYTHON_BIN='$PY' bash scripts/run_may_arr_model_lane.sh gemma2bit google/gemma-2-2b-it 1"
tmux new-session -d -s mayarr-g2 "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG='$RUN_TAG' CONFIG_PATH='$CONFIG_PATH' PYTHON_BIN='$PY' bash scripts/run_may_arr_model_lane.sh llama3b meta-llama/Llama-3.2-3B 2"
tmux new-session -d -s mayarr-g3 "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG='$RUN_TAG' CONFIG_PATH='$CONFIG_PATH' PYTHON_BIN='$PY' bash scripts/run_gpu3_may_arr_fixpack_aggregate.sh"

echo "RUN_TAG=$RUN_TAG"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "PYTHON_BIN=$PY"
echo "STATE_DIR=$STATE_DIR"
echo "Sessions: mayarr-g0 mayarr-g1 mayarr-g2 mayarr-g3"
