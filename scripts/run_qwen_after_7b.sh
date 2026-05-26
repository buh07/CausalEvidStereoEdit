#!/usr/bin/env bash
set -euo pipefail

cd /jumbo/lisp/f004ndc/StereACL
RUN_TAG=${RUN_TAG:-balanced_ext_20260509_154409}
STATE_DIR="results/${RUN_TAG}/state"

wait_flag(){
  local f="$1"
  while [[ ! -f "$STATE_DIR/$f" ]]; do
    echo "[$(date -Iseconds)] waiting for $f"
    sleep 30
  done
  echo "[$(date -Iseconds)] observed $f"
}

wait_flag mistral7b_p3_done.flag
wait_flag olmo7b_p3_done.flag

RUN_TAG="$RUN_TAG" tmux new-session -d -s stereacl-qwen3b-resume 'bash run_qwen3b_resume_p2p3.sh'
RUN_TAG="$RUN_TAG" tmux new-session -d -s stereacl-qwen3bi-resume 'bash run_qwen3bi_resume_p2p3.sh'

echo "[$(date -Iseconds)] launched qwen resume sessions"
