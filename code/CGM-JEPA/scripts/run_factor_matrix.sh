#!/usr/bin/env bash
# M3 factor matrix (STRATEGY.md §M3, plan A upgraded after M2 throughput):
#   3 objectives x 3 archs x 3 seeds, stride 288 (6.5k windows, all cohorts), 60 epochs
#   + 3 ablations on mcr/dual seed43: no-TD, no-circadian, no-augmentation
# All runs land in ../../runs/<name>; logs append to ../../runs/matrix.log
set -u
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
RUNS="../../runs"
LOG="$RUNS/matrix.log"
mkdir -p "$RUNS"

run_one () {
  local name="$1"; shift
  echo "=== $(date '+%F %T') START $name ===" >> "$LOG"
  "$PY" -m pretrain.pretrain_cgm_jepa \
    --data-dir ../../data/unified --splits ../../data/splits.json \
    --epochs 60 --batch-size 128 --workers 2 --stride 288 --threads 9 \
    --out "$RUNS/$name" "$@" >> "$LOG" 2>&1 \
    && echo "=== $(date '+%F %T') OK    $name ===" >> "$LOG" \
    || echo "=== $(date '+%F %T') FAIL  $name ===" >> "$LOG"
}

for seed in 43 44 45; do
  for obj in mcr recon causal; do
    for arch in plain dual cnn; do
      run_one "${obj}_${arch}_seed${seed}" --objective "$obj" --arch "$arch" --seed "$seed"
    done
  done
done

# ablations (STRATEGY.md §3 plan A keeps the mcr/dual family)
run_one "mcr_dual_seed43_notd"       --objective mcr --arch dual --seed 43 --lambda-td 0
run_one "mcr_dual_seed43_nocircadian" --objective mcr --arch dual --seed 43 --no-circadian
run_one "mcr_dual_seed43_noaug"      --objective mcr --arch dual --seed 43 --no-aug

echo "=== $(date '+%F %T') MATRIX DONE ===" >> "$LOG"
