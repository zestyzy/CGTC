#!/usr/bin/env bash
set -euo pipefail

# Example end-to-end pipeline for Rotate.
# Adjust ROOT, CASE, TAG, and DEVICE to match your environment before running.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${ROOT}/Rotate"
CASE="C1"
TAG="demo_run"
DEVICE="cuda:0"
PTS=4096
BATCH=2
WORKERS=4

python -u "$CODE_ROOT/train_tpac.py" \
  --cfg "$CODE_ROOT/configs/default.yaml" \
  --root "$CODE_ROOT" \
  --case "$CASE" \
  --tag "$TAG" \
  --device "$DEVICE" \
  --pts "$PTS" \
  --batch "$BATCH" \
  --workers "$WORKERS"

python -u "$CODE_ROOT/draw_picture.py" \
  --metrics "$CODE_ROOT/results/temp_results/$CASE/$TAG/train_val_metrics.csv" \
  --out-dir "$CODE_ROOT/results/temp_results/$CASE/$TAG/diagnostics"

python -u "$CODE_ROOT/eval/eval_physics.py" \
  --cfg "$CODE_ROOT/configs/default.yaml" \
  --root "$CODE_ROOT" \
  --case "$CASE" \
  --split test \
  --pts "$PTS" \
  --ckpt "$CODE_ROOT/results/temp_results/$CASE/$TAG/weight/final_reco.pth" \
  --out "$CODE_ROOT/results/temp_results/$CASE/$TAG/physics_eval.csv" \
  --device "$DEVICE"

bash "$ROOT/wo.sh" --case "$CASE" --tag "${TAG}_wo" --device "$DEVICE" \
  --no-guard --reverse-hierarchy --ratio-sweep --no-rollback --no-freeze-teacher

printf '\nPipeline complete. Results stored under Rotate/results/temp_results/%s/%s and runs/wo/%s/%s.\n' "$CASE" "$TAG" "$CASE" "${TAG}_wo"
