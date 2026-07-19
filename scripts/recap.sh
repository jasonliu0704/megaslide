#!/usr/bin/env bash
# Compact "what happened while I was away" recap.
# Run this right after SSHing back in.
# Usage: bash scripts/recap.sh
#        bash scripts/recap.sh --since 1h
#        bash scripts/recap.sh --since 30m

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SINCE=""
if [[ "${1:-}" == "--since" ]] && [[ -n "${2:-}" ]]; then
  SINCE="$2"
fi

BOLD=$(tput bold 2>/dev/null || true)
DIM=$(tput dim 2>/dev/null || true)
RST=$(tput sgr0 2>/dev/null || true)

echo "${BOLD}MegaSlide recap${RST}  $(date '+%F %T')"
echo "------------------------------------------------------------"

# Queue session status
echo "${BOLD}Session${RST}"
if tmux has-session -t megaslide_queue 2>/dev/null; then
  echo "  tmux 'megaslide_queue' is RUNNING."
  echo "  Attach with:  tmux attach -t megaslide_queue   (Ctrl-b d to detach)"
else
  echo "  ${DIM}tmux 'megaslide_queue' is NOT running.${RST}"
  echo "  Restart with: tmux new -d -s megaslide_queue 'bash scripts/run_queue.sh'"
fi
echo

# Phase-level summary
echo "${BOLD}Phase status${RST}"
for phase in 10_hybrid_120M 10_real_video 10_deepspeed; do
  name="${phase##*_}"
  dir="results/$phase"
  done_file="$dir/.done"
  log="$dir/run.log"
  [[ "$phase" == "10_real_video" ]] && log="$dir/train.log"
  if [[ -f "$done_file" ]]; then
    last=$(stat -c '%y' "$done_file" 2>/dev/null | cut -d. -f1)
    printf "  ${BOLD}%-12s${RST}  DONE        (sentinel written %s)\n" "$name" "$last"
  elif [[ -f "$log" ]]; then
    mtime=$(stat -c '%Y' "$log" 2>/dev/null || echo 0)
    now=$(date +%s); ago=$((now - mtime))
    if (( ago < 60 )); then state="${BOLD}ACTIVE${RST}"; suffix="${ago}s ago"
    elif (( ago < 600 )); then state="${DIM}quiet${RST}"; suffix="${ago}s ago"
    else h=$((ago/3600)); m=$(((ago%3600)/60)); state="${DIM}idle${RST}"; suffix="${h}h${m}m ago"; fi
    printf "  ${BOLD}%-12s${RST}  %-25s  (last write %s)\n" "$name" "$state" "$suffix"
  else
    printf "  ${BOLD}%-12s${RST}  ${DIM}not started${RST}\n" "$name"
  fi
done
echo

# Milestones (parsed from logs)
echo "${BOLD}Milestones${RST}"
ALL_LOGS=$(ls -1 results/10_*/run.log results/10_*/train.log 2>/dev/null)
if [[ -z "$ALL_LOGS" ]]; then
  echo "  ${DIM}no logs yet${RST}"
else
  # shellcheck disable=SC2086
  MILES=$(grep -h "MILESTONE" $ALL_LOGS 2>/dev/null | tail -20)
  if [[ -z "$MILES" ]]; then
    echo "  ${DIM}no 10% milestones logged yet${RST}"
  else
    echo "$MILES" | sed 's/^/  /'
  fi
fi
echo

# Final results (if any JSON has landed)
echo "${BOLD}Final-loss summary (from JSON artefacts)${RST}"
JSON_HITS=0
for f in results/10_*/baseline_results.json results/10_*/register_results.json results/10_*/anchor_results.json; do
  [[ -f "$f" ]] || continue
  JSON_HITS=$((JSON_HITS+1))
  python3 -c "
import json, sys
r = json.load(open('$f'))
print(f'  $(basename $(dirname $f))/{r[\"variant\"]:<10} params={r[\"num_params\"]:>12,}  final_loss={r[\"final_loss\"]:.4f}  avg_step={r[\"avg_step_time\"]:.3f}s' if r.get('variant') else '')
" 2>/dev/null
done
(( JSON_HITS == 0 )) && echo "  ${DIM}no completed runs yet${RST}"
echo

# Recent queue events
echo "${BOLD}Recent queue events${RST}"
if [[ -f results/queue.log ]]; then
  if [[ -n "$SINCE" ]]; then
    cutoff_epoch=$(date -d "$SINCE ago" +%s 2>/dev/null || echo 0)
    awk -v c="$cutoff_epoch" '
      /^\[/ {
        ts=substr($0,2,19); cmd="date -d \"" ts "\" +%s 2>/dev/null"; cmd | getline ep; close(cmd);
        if (ep+0 >= c+0) print "  " $0
      }
    ' results/queue.log | tail -20
  else
    tail -10 results/queue.log | sed 's/^/  /'
  fi
fi
echo
echo "------------------------------------------------------------"
if [[ -f results/AGENT_LOG.md ]]; then
  echo "${BOLD}Agent log${RST}  (full work history with decisions and rationale)"
  echo "  cat results/AGENT_LOG.md      # full log"
  echo "  head -60 results/AGENT_LOG.md # just 'current state' + latest session"
  echo
fi
echo "Live view:  bash scripts/monitor.sh        (one-shot)"
echo "            bash scripts/monitor.sh watch  (auto-refresh)"
echo "            tmux attach -t megaslide_queue"
