#!/usr/bin/env bash
set -euo pipefail
cd /jumbo/lisp/f004ndc/StereACL
LOG=results/log_gpu7_dirpos_compare_post.txt
mkdir -p results

echo "[$(date -Iseconds)] START gpu7 post-processing" | tee "$LOG"

while true; do
  alive=0
  for s in dirpos_g4 dirpos_g5 dirpos_g6; do
    if tmux has-session -t "$s" 2>/dev/null; then
      alive=1
      break
    fi
  done
  if [[ "$alive" -eq 0 ]]; then
    break
  fi
  echo "[$(date -Iseconds)] Waiting for dirpos sessions (g4/g5/g6)..." | tee -a "$LOG"
  sleep 30
done

python tools/summarize_direction_position_compare.py 2>&1 | tee -a "$LOG"
python tools/compile_results.py 2>&1 | tee -a "$LOG"

echo "[$(date -Iseconds)] DONE gpu7 post-processing" | tee -a "$LOG"
