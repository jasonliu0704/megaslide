#!/usr/bin/env bash
# Phase 2: FSDP-CPU offload sweep across scale ladder.
# Runs one scale at a time, writes a .done sentinel after each.
# Designed to be tmux-friendly: PYTHONUNBUFFERED so logs stream.

set -u  # Don't use -e; we want to continue past OOM scales.

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="$ROOT/results/10_deepspeed"
LOG="$OUT/sweep.log"
mkdir -p "$OUT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

VENV="$ROOT/venv/bin/python"

# Each scale: (name, num_steps). Steps are kept modest so the sweep
# finishes in <2h even if every scale fits.
SCALES=(
  "1b 12"
  "3b 12"
  "7b 8"
  "12.6b 6"
  "19.7b 5"
  "28.4b 5"
  "33.3b 5"
)

echo "==== Phase 2 sweep started at $(date -u +%FT%TZ) ====" | tee -a "$LOG"
echo "Output dir: $OUT" | tee -a "$LOG"

for entry in "${SCALES[@]}"; do
  scale=$(echo "$entry" | awk '{print $1}')
  steps=$(echo "$entry" | awk '{print $2}')
  out_json="$OUT/fsdp_cpu_${scale}.json"
  done_flag="$OUT/.fsdp_cpu_${scale}.done"

  if [[ -f "$done_flag" ]]; then
    echo "[$(date -u +%T)] SKIP $scale (already done)" | tee -a "$LOG"
    continue
  fi

  echo "" | tee -a "$LOG"
  echo "[$(date -u +%T)] >>> START scale=$scale steps=$steps" | tee -a "$LOG"

  "$VENV" examples/run_offload_comparison.py \
      --scale "$scale" \
      --num-steps "$steps" \
      --output-dir "$OUT" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}

  echo "[$(date -u +%T)] <<< END  scale=$scale rc=$rc" | tee -a "$LOG"

  # Mark done regardless of OOM; the JSON itself records the outcome.
  touch "$done_flag"
done

echo "" | tee -a "$LOG"
echo "==== Phase 2 sweep finished at $(date -u +%FT%TZ) ====" | tee -a "$LOG"
touch "$OUT/.phase2.done"
