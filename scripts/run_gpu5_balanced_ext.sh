#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
cd /jumbo/lisp/f004ndc/StereACL

RUN_TAG=${RUN_TAG:-balanced_ext_20260509}
STATE_DIR="results/${RUN_TAG}/state"
mkdir -p "$STATE_DIR"
ORCH_START_UTC=$(cat "$STATE_DIR/orch_start_utc.txt")
LOG_FILE="results/${RUN_TAG}/log_gpu5.txt"

source ./scripts/run_balanced_ext_common.sh

log_msg "GPU5 queue start | RUN_TAG=$RUN_TAG"

# 3B base lane B
run_packet_p1 "gemma2bit" "google/gemma-2-2b-it" "bfloat16" "0" 1200
run_packet_p1 "llama3b" "meta-llama/Llama-3.2-3B" "bfloat16" "0" 1300
run_packet_p1 "qwen3bi" "Qwen/Qwen2.5-3B-Instruct" "bfloat16" "0" 1500
run_packet_p2_multiseed "gemma2bit" "google/gemma-2-2b-it" "bfloat16" "0"
run_packet_p2_multiseed "llama3b" "meta-llama/Llama-3.2-3B" "bfloat16" "0"
run_packet_p2_multiseed "qwen3bi" "Qwen/Qwen2.5-3B-Instruct" "bfloat16" "0"
run_packet_p3 "gemma2bit" "google/gemma-2-2b-it" "bfloat16" "0" 1200
run_packet_p3 "llama3b" "meta-llama/Llama-3.2-3B" "bfloat16" "0" 1300
run_packet_p3 "qwen3bi" "Qwen/Qwen2.5-3B-Instruct" "bfloat16" "0" 1500

log_msg "GPU5 queue done"
