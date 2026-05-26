#!/usr/bin/env bash
set -euo pipefail

cd /jumbo/lisp/f004ndc/StereACL

RUN_TAG="${RUN_TAG:-arr_rev3_$(date -u +%Y%m%d_%H%M%S)}"
CONFIG_PATH="${CONFIG_PATH:-configs/arr_rev3_freeze.yaml}"

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

for s in arr3-g0 arr3-g1 arr3-g2 arr3-g4 arr3-g5 arr3-g6 arr3-g7 arr3-agg; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

tmux new-session -d -s arr3-g0 "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG='$RUN_TAG' CONFIG_PATH='$CONFIG_PATH' PYTHON_BIN='$PY' bash scripts/run_arr_rev3_model_lane.sh gemma2b google/gemma-2-2b 0 core"
tmux new-session -d -s arr3-g1 "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG='$RUN_TAG' CONFIG_PATH='$CONFIG_PATH' PYTHON_BIN='$PY' bash scripts/run_arr_rev3_model_lane.sh gemma2bit google/gemma-2-2b-it 1 core"
tmux new-session -d -s arr3-g2 "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG='$RUN_TAG' CONFIG_PATH='$CONFIG_PATH' PYTHON_BIN='$PY' bash scripts/run_arr_rev3_model_lane.sh llama3b meta-llama/Llama-3.2-3B 2 core"
tmux new-session -d -s arr3-g4 "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG='$RUN_TAG' CONFIG_PATH='$CONFIG_PATH' PYTHON_BIN='$PY' bash scripts/run_arr_rev3_model_lane.sh qwen3b Qwen/Qwen2.5-3B 4 extension"
tmux new-session -d -s arr3-g5 "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG='$RUN_TAG' CONFIG_PATH='$CONFIG_PATH' PYTHON_BIN='$PY' bash scripts/run_arr_rev3_model_lane.sh qwen3bi Qwen/Qwen2.5-3B-Instruct 5 extension"
tmux new-session -d -s arr3-g6 "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG='$RUN_TAG' CONFIG_PATH='$CONFIG_PATH' PYTHON_BIN='$PY' bash scripts/run_arr_rev3_model_lane.sh mistral7b /jumbo/lisp/f004ndc/models/mistral-7b-v0.1 6 extension"
tmux new-session -d -s arr3-g7 "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG='$RUN_TAG' CONFIG_PATH='$CONFIG_PATH' PYTHON_BIN='$PY' bash scripts/run_arr_rev3_model_lane.sh olmo7b /jumbo/lisp/f004ndc/models/olmo-2-7b 7 extension"
tmux new-session -d -s arr3-agg "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG='$RUN_TAG' CONFIG_PATH='$CONFIG_PATH' PYTHON_BIN='$PY' bash scripts/run_arr_rev3_aggregate.sh"

echo "RUN_TAG=$RUN_TAG"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "PYTHON_BIN=$PY"
echo "STATE_DIR=$STATE_DIR"
echo "Sessions: arr3-g0 arr3-g1 arr3-g2 arr3-g4 arr3-g5 arr3-g6 arr3-g7 arr3-agg"
