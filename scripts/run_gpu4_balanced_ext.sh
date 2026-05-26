#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
cd /jumbo/lisp/f004ndc/StereACL

RUN_TAG=${RUN_TAG:-balanced_ext_20260509}
STATE_DIR="results/${RUN_TAG}/state"
mkdir -p "$STATE_DIR"
ORCH_START_UTC=$(cat "$STATE_DIR/orch_start_utc.txt")
LOG_FILE="results/${RUN_TAG}/log_gpu4.txt"

source ./scripts/run_balanced_ext_common.sh

log_msg "GPU4 queue start | RUN_TAG=$RUN_TAG"

# 3B base lane A
run_packet_p1 "gemma2b" "google/gemma-2-2b" "bfloat16" "0" 1100
run_packet_p1 "qwen3b" "Qwen/Qwen2.5-3B" "bfloat16" "0" 1400
run_packet_p2_multiseed "gemma2b" "google/gemma-2-2b" "bfloat16" "0"
run_packet_p2_multiseed "qwen3b" "Qwen/Qwen2.5-3B" "bfloat16" "0"
run_packet_p3 "gemma2b" "google/gemma-2-2b" "bfloat16" "0" 1100
run_packet_p3 "qwen3b" "Qwen/Qwen2.5-3B" "bfloat16" "0" 1400

log_msg "GPU4 queue done"
