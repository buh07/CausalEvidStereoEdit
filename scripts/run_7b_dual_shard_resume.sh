#!/usr/bin/env bash
set -euo pipefail

cd /jumbo/lisp/f004ndc/StereACL

RUN_TAG=${RUN_TAG:?RUN_TAG must be set}
export RUN_TAG

echo "[dual-shard] RUN_TAG=$RUN_TAG"
echo "[dual-shard] Starting mistral7b sharded resume on CUDA_VISIBLE_DEVICES=6,7"
bash scripts/run_mistral7b_sharded_resume.sh

echo "[dual-shard] Starting olmo7b sharded resume on CUDA_VISIBLE_DEVICES=6,7"
bash scripts/run_olmo7b_sharded_resume.sh

echo "[dual-shard] Completed both 7B sharded resume flows"
