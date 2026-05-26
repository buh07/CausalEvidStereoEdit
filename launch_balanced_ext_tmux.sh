#!/usr/bin/env bash
set -euo pipefail

cd /jumbo/lisp/f004ndc/StereACL

RUN_TAG=${RUN_TAG:-balanced_ext_$(date -u +%Y%m%d_%H%M%S)}
STATE_DIR="results/${RUN_TAG}/state"
mkdir -p "$STATE_DIR"
printf '%s\n' "$(date -u -Iseconds)" > "$STATE_DIR/orch_start_utc.txt"

for s in stereacl-g4-balance stereacl-g5-balance stereacl-g6-balance stereacl-g7-balance; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

tmux new-session -d -s stereacl-g4-balance "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG=$RUN_TAG bash scripts/run_gpu4_balanced_ext.sh"
tmux new-session -d -s stereacl-g5-balance "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG=$RUN_TAG bash scripts/run_gpu5_balanced_ext.sh"
tmux new-session -d -s stereacl-g6-balance "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG=$RUN_TAG bash scripts/run_gpu6_balanced_ext.sh"
tmux new-session -d -s stereacl-g7-balance "cd /jumbo/lisp/f004ndc/StereACL && RUN_TAG=$RUN_TAG bash scripts/run_gpu7_balanced_ext.sh"

echo "RUN_TAG=$RUN_TAG"
echo "State dir: $STATE_DIR"
echo "Sessions: stereacl-g4-balance stereacl-g5-balance stereacl-g6-balance stereacl-g7-balance"
