#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
cd /jumbo/lisp/f004ndc/StereACL

RUN_TAG=${RUN_TAG:-balanced_ext_20260509}
STATE_DIR="results/${RUN_TAG}/state"
mkdir -p "$STATE_DIR"
ORCH_START_UTC=$(cat "$STATE_DIR/orch_start_utc.txt")
LOG_FILE="results/${RUN_TAG}/log_gpu6.txt"

source ./scripts/run_balanced_ext_common.sh

log_msg "GPU6 queue start | RUN_TAG=$RUN_TAG"

# 7B lane 1 (split across two GPUs: this lane on GPU6)
run_packet_p1 "mistral7b" "/jumbo/lisp/f004ndc/models/mistral-7b-v0.1" "float16" "1" 1600
run_packet_p2_multiseed "mistral7b" "/jumbo/lisp/f004ndc/models/mistral-7b-v0.1" "float16" "1"
run_packet_p3 "mistral7b" "/jumbo/lisp/f004ndc/models/mistral-7b-v0.1" "float16" "1" 1600

log_msg "GPU6 queue done"
