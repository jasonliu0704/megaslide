#!/usr/bin/env bash
# Serial runner for the three-experiments-week plan.
# Designed to live inside tmux so you can detach + reconnect.
#
# Phases:
#   3 - Hybrid + registers @ 120M params (runs today; uses existing code)
#   1 - 280M MegaSlide-DiT on MSR-VTT (gated; needs Phase 0 scaffolds)
#   2 - DeepSpeed ZeRO-3 CPU offload comparison (gated; needs scaffold script)
#
# Each gated phase looks for an enablement file before running. To unlock,
# create the corresponding file e.g.:
#   touch results/queue.enable_phase1
#   touch results/queue.enable_phase2

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

QUEUE_LOG="$REPO/results/queue.log"
mkdir -p "$REPO/results/10_hybrid_120M" "$REPO/results/10_real_video" "$REPO/results/10_deepspeed"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"; }

VENV_PY="$REPO/venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  log "ERROR: $VENV_PY not found. Activate or fix venv path before running."
  exit 1
fi

export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1   # so prints reach the log file immediately

# Reduce OOM-killer pressure on the parent (best-effort, may need sudo).
if [[ -w /proc/$$/oom_score_adj ]]; then
  echo -500 > /proc/$$/oom_score_adj 2>/dev/null || true
fi

log "============================================================"
log "Queue start. host=$(hostname) pid=$$ user=$(whoami)"
log "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
log "Host RAM: $(free -h | awk '/Mem:/ {print $2 " total, " $7 " avail"}')"
log "Disk free: $(df -h "$REPO" | awk 'NR==2 {print $4}')"
log "============================================================"

# -------------------------------------------------------------------
# Phase 3: Hybrid + registers @ 120M  (ready today)
# -------------------------------------------------------------------
PHASE3_OUT="$REPO/results/10_hybrid_120M"
PHASE3_LOG="$PHASE3_OUT/run.log"
PHASE3_DONE="$PHASE3_OUT/.done"

if [[ -f "$PHASE3_DONE" ]]; then
  log "Phase 3 already complete (sentinel $PHASE3_DONE present). Skipping."
else
  log "Phase 3 START: Hybrid + registers @ 120M (baseline + register-64 + register-128)"
  # We loop variants so registry-128 can run a different config and so we
  # get one JSON per variant in the same output dir.
  for VARIANT in baseline register; do
    log "Phase 3 :: variant=$VARIANT"
    "$VENV_PY" examples/run_hybrid_attention_experiment.py \
        --config examples/configs/hybrid_register_128_120M.yaml \
        --variant "$VARIANT" \
        --output-dir "$PHASE3_OUT" \
        2>&1 | tee -a "$PHASE3_LOG"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
      log "Phase 3 :: variant=$VARIANT FAILED rc=$rc (continuing to next variant)"
    fi
  done
  touch "$PHASE3_DONE"
  log "Phase 3 DONE."
fi

# -------------------------------------------------------------------
# Phase 1: 280M MegaSlide-DiT on MSR-VTT  (gated)
# -------------------------------------------------------------------
PHASE1_OUT="$REPO/results/10_real_video"
PHASE1_LOG="$PHASE1_OUT/train.log"
PHASE1_DONE="$PHASE1_OUT/.done"
PHASE1_ENABLE="$REPO/results/queue.enable_phase1"

if [[ ! -f "$PHASE1_ENABLE" ]]; then
  log "Phase 1 SKIPPED (no $PHASE1_ENABLE). Build Phase 0 scaffolds first."
elif [[ -f "$PHASE1_DONE" ]]; then
  log "Phase 1 already complete. Skipping."
else
  log "Phase 1 START: 280M MegaSlide-DiT on MSR-VTT"
  "$VENV_PY" examples/train_megaslide_dit.py \
      --config examples/configs/megaslide_dit_280M_msrvtt.yaml \
      --output-dir "$PHASE1_OUT" \
      2>&1 | tee -a "$PHASE1_LOG" \
      && touch "$PHASE1_DONE" \
      && log "Phase 1 DONE." \
      || log "Phase 1 FAILED."
fi

# -------------------------------------------------------------------
# Phase 2: DeepSpeed ZeRO-3 CPU offload comparison  (gated)
# -------------------------------------------------------------------
PHASE2_OUT="$REPO/results/10_deepspeed"
PHASE2_LOG="$PHASE2_OUT/run.log"
PHASE2_DONE="$PHASE2_OUT/.done"
PHASE2_ENABLE="$REPO/results/queue.enable_phase2"

if [[ ! -f "$PHASE2_ENABLE" ]]; then
  log "Phase 2 SKIPPED (no $PHASE2_ENABLE). Build run_deepspeed_baseline.py first."
elif [[ -f "$PHASE2_DONE" ]]; then
  log "Phase 2 already complete. Skipping."
else
  log "Phase 2 START: DeepSpeed ZeRO-3 vs MegaSlide @ 12.6B / 19.7B / 28.4B"
  "$VENV_PY" examples/run_deepspeed_baseline.py \
      --output-dir "$PHASE2_OUT" \
      2>&1 | tee -a "$PHASE2_LOG" \
      && touch "$PHASE2_DONE" \
      && log "Phase 2 DONE." \
      || log "Phase 2 FAILED."
fi

log "============================================================"
log "Queue finished. Check results/ for outputs."
log "============================================================"
