#!/usr/bin/env bash
set -euo pipefail
cd /jumbo/lisp/f004ndc/StereACL
mkdir -p results
LOG=results/log_gpu7_exp10_decomp_post_$(date +%Y%m%d_%H%M%S).txt

echo "[$(date -Iseconds)] START gpu7 exp10 post" | tee "$LOG"
while true; do
  alive=0
  for s in stereacl-g4-exp10 stereacl-g5-exp10 stereacl-g6-exp10; do
    if tmux has-session -t "$s" 2>/dev/null; then
      alive=1
      break
    fi
  done
  if [[ "$alive" -eq 0 ]]; then
    break
  fi
  echo "[$(date -Iseconds)] Waiting for exp10 sessions..." | tee -a "$LOG"
  sleep 30
done

python tools/summarize_exp10_decomposition.py 2>&1 | tee -a "$LOG"
python tools/compile_results.py 2>&1 | tee -a "$LOG"

echo "[$(date -Iseconds)] DONE gpu7 exp10 post" | tee -a "$LOG"
