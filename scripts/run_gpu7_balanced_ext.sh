#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
cd /jumbo/lisp/f004ndc/StereACL

RUN_TAG=${RUN_TAG:-balanced_ext_20260509}
STATE_DIR="results/${RUN_TAG}/state"
mkdir -p "$STATE_DIR"
ORCH_START_UTC=$(cat "$STATE_DIR/orch_start_utc.txt")
LOG_FILE="results/${RUN_TAG}/log_gpu7.txt"

source ./scripts/run_balanced_ext_common.sh

log_msg "GPU7 queue start | RUN_TAG=$RUN_TAG"

# 7B lane 2 (split across two GPUs: this lane on GPU7)
run_packet_p1 "olmo7b" "/jumbo/lisp/f004ndc/models/olmo-2-7b" "float16" "1" 1700
run_packet_p2_multiseed "olmo7b" "/jumbo/lisp/f004ndc/models/olmo-2-7b" "float16" "1"
run_packet_p3 "olmo7b" "/jumbo/lisp/f004ndc/models/olmo-2-7b" "float16" "1" 1700

# Final barrier + compile + summary
wait_for_flag "gemma2b_p3_done"
wait_for_flag "gemma2bit_p3_done"
wait_for_flag "llama3b_p3_done"
wait_for_flag "qwen3b_p3_done"
wait_for_flag "qwen3bi_p3_done"
wait_for_flag "mistral7b_p3_done"
wait_for_flag "olmo7b_p3_done"

run_cmd "compile-results" "python3 tools/compile_results.py"
run_cmd "phase1-summary" "python3 tools/summarize_phase1_prospective.py --run-tag '$RUN_TAG'"

log_msg "GPU7 queue done"
