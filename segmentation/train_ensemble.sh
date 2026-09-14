#!/usr/bin/env bash
# Train the U-Net ensemble, one network at a time, with a thermal guard.
#
# ROADMAP 6.2 identifies ensembling as the cheapest available accuracy gain:
# both leaderboard leaders used it (sysu_media three networks, pgs five plus
# test-time flips). Each member is the same architecture and data, differing
# only in its random seed, so their errors are independent and average out
# while the real signal survives.
#
# SEQUENTIAL BY NECESSITY, NOT CHOICE. A single training run peaks around
# 3.1 GB of the RTX 3050's 6 GB, so two at once would not fit. That also makes
# this the thermally gentle option.
#
# The guard below is belt-and-braces: the GPU already protects itself in
# hardware (slowdown at 97 C, shutdown at 100 C), so the worst it can do on its
# own is run slower. This simply stops the queue early and cools down if things
# get hot, rather than pushing a laptop through six unattended hours at the
# limit.
#
# CRASH-SAFE. Each member saves its full training state every epoch and is
# started with --resume, so re-running this script after an interruption picks
# each member up where it stopped instead of restarting it. This exists because
# the machine hard-powered-off at 11:58 on 2026-09-14, 71 epochs into seed 1,
# with no thermal, OOM or battery cause in the journal — so it can happen again
# and the only defence is making it cheap.
#
# Members that already finished are NOT skipped automatically — a completed run
# deletes its own resume state, so re-running a finished seed retrains it from
# scratch. Pass only the seeds you still need.
#
#   bash segmentation/train_ensemble.sh 1 2 3        # three more seeds
#
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON=code/.venv/bin/python
ABORT_TEMP=88        # stop starting new runs above this (slowdown is 97)
COOLDOWN_SECONDS=300 # let the card settle between members
LOG=segmentation/outputs/logs/ensemble_$(date +%F).log

gpu_temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0; }

say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

mkdir -p segmentation/outputs/logs
say "ensemble run starting — seeds: $*"
say "thermal guard: will not start a member above ${ABORT_TEMP}C (GPU slows itself at 97C)"

for seed in "$@"; do
  temp=$(gpu_temp)
  if [ "$temp" -ge "$ABORT_TEMP" ]; then
    say "GPU at ${temp}C — above the ${ABORT_TEMP}C guard. Cooling for ${COOLDOWN_SECONDS}s."
    sleep "$COOLDOWN_SECONDS"
    temp=$(gpu_temp)
    if [ "$temp" -ge "$ABORT_TEMP" ]; then
      say "still ${temp}C after cooling — stopping here rather than pushing it."
      say "members trained so far are usable; re-run this script later for the rest."
      exit 0
    fi
  fi

  say "--- seed ${seed} starting (GPU ${temp}C) ---"
  $PYTHON -m segmentation.train_unet --epochs 80 --seed "$seed" --resume \
    --augment-strength "${AUG_STRENGTH:-1.0}" --tag "${TAG:-aug}" >>"$LOG" 2>&1
  status=$?
  say "--- seed ${seed} finished (exit ${status}, GPU $(gpu_temp)C) ---"

  if [ "$status" -ne 0 ]; then
    say "seed ${seed} failed. Stopping so the failure is visible rather than buried."
    exit "$status"
  fi

  say "cooling ${COOLDOWN_SECONDS}s before the next member"
  sleep "$COOLDOWN_SECONDS"
done

say "ensemble complete. Combine with: $PYTHON -m segmentation.run_ensemble"
