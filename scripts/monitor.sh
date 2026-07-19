#!/usr/bin/env bash
# Status dashboard for the three-experiments-week run.
# Usage: bash scripts/monitor.sh         (one-shot)
#        bash scripts/monitor.sh watch   (auto-refresh every 30s)
#        bash scripts/monitor.sh tail <phase>   (tail the log of a phase)

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

QUEUE_LOG="$REPO/results/queue.log"
declare -A LOG=(
  [hybrid120m]="$REPO/results/10_hybrid_120M/run.log"
  [realvideo]="$REPO/results/10_real_video/train.log"
  [deepspeed]="$REPO/results/10_deepspeed/run.log"
)
declare -A JSON_DIR=(
  [hybrid120m]="$REPO/results/10_hybrid_120M"
  [realvideo]="$REPO/results/10_real_video"
  [deepspeed]="$REPO/results/10_deepspeed"
)

BOLD=$(tput bold 2>/dev/null || true)
DIM=$(tput dim 2>/dev/null || true)
RST=$(tput sgr0 2>/dev/null || true)

hr() { printf '%*s\n' "${1:-72}" '' | tr ' ' '-'; }

show_tmux() {
  echo "${BOLD}tmux sessions${RST}"
  if tmux ls 2>/dev/null; then :; else echo "  (no tmux sessions)"; fi
  echo
}

show_gpu() {
  echo "${BOLD}GPU${RST}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' '{printf "  %s  |  util %s%%  |  mem %s / %s MiB  |  %s°C  |  %s W\n", $1, $2, $3, $4, $5, $6}'
  else
    echo "  nvidia-smi not available"
  fi
  echo
}

show_host() {
  echo "${BOLD}Host${RST}"
  free -h | awk 'NR==1 || NR==2 {print "  "$0}'
  df -h "$REPO" | awk 'NR==1 || NR==2 {print "  "$0}'
  uptime | awk '{print "  load:" $(NF-2) $(NF-1) $NF}'
  echo
}

phase_status() {
  local name="$1"
  local log="${LOG[$name]:-}"
  local jdir="${JSON_DIR[$name]:-}"
  printf "%s%-12s%s " "${BOLD}" "$name" "${RST}"
  if [[ -z "$log" || ! -f "$log" ]]; then
    echo "${DIM}not started${RST}"
    return
  fi
  local lines size mtime ago
  lines=$(wc -l < "$log" 2>/dev/null || echo 0)
  size=$(du -h "$log" 2>/dev/null | awk '{print $1}')
  mtime=$(stat -c '%Y' "$log" 2>/dev/null || echo 0)
  now=$(date +%s)
  ago=$((now - mtime))
  if (( ago < 60 )); then
    state="${BOLD}ACTIVE${RST} (last write ${ago}s ago)"
  elif (( ago < 600 )); then
    state="${DIM}quiet${RST} (last write ${ago}s ago)"
  else
    local h=$((ago/3600)) m=$(((ago%3600)/60))
    state="${DIM}idle${RST} (last write ${h}h${m}m ago)"
  fi
  printf "%-44s lines=%-7s size=%s\n" "$state" "$lines" "$size"
  if [[ -d "$jdir" ]]; then
    local jcount
    jcount=$(find "$jdir" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    if (( jcount > 0 )); then
      printf "             ${DIM}json artefacts: %s${RST}\n" "$jcount"
    fi
  fi
}

last_lines() {
  local name="$1" n="${2:-8}"
  local log="${LOG[$name]:-}"
  echo "${BOLD}--- $name : last $n log lines ---${RST}"
  if [[ -f "$log" ]]; then
    tail -n "$n" "$log" | sed 's/^/  /'
  else
    echo "  (no log yet)"
  fi
  echo
}

dashboard() {
  clear 2>/dev/null || printf '\n\n\n'
  echo "${BOLD}MegaSlide experiment monitor${RST}  $(date '+%Y-%m-%d %H:%M:%S')"
  hr
  show_tmux
  show_gpu
  show_host
  echo "${BOLD}Phase status${RST}"
  phase_status hybrid120m
  phase_status realvideo
  phase_status deepspeed
  echo
  if [[ -f "$QUEUE_LOG" ]]; then
    echo "${BOLD}Queue progress${RST}"
    tail -n 10 "$QUEUE_LOG" | sed 's/^/  /'
    echo
  fi
  last_lines hybrid120m 6
  last_lines realvideo 6
  last_lines deepspeed 6
  hr
  echo "Tips:  bash scripts/monitor.sh watch       (auto-refresh)"
  echo "       bash scripts/monitor.sh tail hybrid120m"
  echo "       tmux attach -t megaslide_queue       (attach to live session)"
}

case "${1:-}" in
  watch)
    while true; do
      dashboard
      sleep 30
    done
    ;;
  tail)
    name="${2:-hybrid120m}"
    log="${LOG[$name]:-}"
    if [[ -z "$log" ]]; then echo "Unknown phase: $name"; exit 1; fi
    echo "tailing $log (Ctrl-C to stop)"
    tail -F "$log"
    ;;
  ""|status|*)
    dashboard
    ;;
esac
