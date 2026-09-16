#!/usr/bin/env bash
# Pause training when the GPU gets hot; resume when it cools.
#
# train_ensemble.sh only checks temperature BETWEEN members, so a spike during a
# 90-minute run would go unnoticed. This watches continuously and suspends the
# training process itself (SIGSTOP) rather than killing it — the process keeps
# its place, its optimiser state and its GPU memory, and picks up exactly where
# it left off on SIGCONT. No epochs are lost.
#
# The GPU already protects itself in hardware (throttles at 97 C, shuts down at
# 100 C). This sits well below that so the card never has to.
#
#   nohup bash code/segmentation/thermal_watchdog.sh &
#
set -uo pipefail
cd "$(dirname "$0")/../.."
# code/ is where the packages live; see code/README.md.
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"

PAUSE_TEMP=90     # suspend at or above this
RESUME_TEMP=80    # resume at or below this (gap prevents rapid flapping)
INTERVAL=20
LOG=code/segmentation/outputs/logs/thermal_$(date +%F).log

mkdir -p code/segmentation/outputs/logs
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0; }
trainer() { pgrep -f "segmentation.train_unet" 2>/dev/null | head -1; }

say "watchdog started — pause at ${PAUSE_TEMP}C, resume at ${RESUME_TEMP}C (GPU self-throttles at 97C)"
paused=0

while true; do
  pid=$(trainer)
  if [ -z "$pid" ]; then
    # No training running. If we paused something that has since gone, reset.
    [ "$paused" -eq 1 ] && { say "training process gone; watchdog idle"; paused=0; }
    sleep "$INTERVAL"; continue
  fi

  t=$(temp)
  if [ "$paused" -eq 0 ] && [ "$t" -ge "$PAUSE_TEMP" ]; then
    kill -STOP "$pid" 2>/dev/null && paused=1
    say "GPU ${t}C >= ${PAUSE_TEMP}C — PAUSED training (pid ${pid}). Waiting for ${RESUME_TEMP}C."
  elif [ "$paused" -eq 1 ] && [ "$t" -le "$RESUME_TEMP" ]; then
    kill -CONT "$pid" 2>/dev/null && paused=0
    say "GPU ${t}C <= ${RESUME_TEMP}C — RESUMED training (pid ${pid})."
  fi
  sleep "$INTERVAL"
done
