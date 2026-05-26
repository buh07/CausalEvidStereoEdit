#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
cd /jumbo/lisp/f004ndc/StereACL
RUN_TAG=${RUN_TAG:-balanced_ext_20260509_154409}
STATE_DIR="results/${RUN_TAG}/state"
ORCH_START_UTC=$(cat "$STATE_DIR/orch_start_utc.txt")
LOG_FILE="results/${RUN_TAG}/log_qwen3bi_resume_p2p3.txt"
source ./run_balanced_ext_common.sh

log_msg "Qwen3B-Instruct resume P2/P3 start | RUN_TAG=$RUN_TAG"
run_packet_p2 "qwen3bi" "Qwen/Qwen2.5-3B-Instruct" "bfloat16" "0" 1500
run_packet_p3 "qwen3bi" "Qwen/Qwen2.5-3B-Instruct" "bfloat16" "0" 1500
log_msg "Qwen3B-Instruct resume P2/P3 done"
