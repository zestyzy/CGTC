#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Fig.4 experiments (wo.py-aligned pipeline, TRUE RESUME)
#
# Pipeline order (same as wo.py):
#   1) train
#   2) draw diagnostics
#   3) physics evaluation
#
# Resume semantics:
#   - Skip only if ALL stages are done
# =============================================================================

usage() {
  cat <<EOF >&2
Usage:
  $0 CASE TAG [extra args passed to train_tpac_new.py]
  $0 --case CASE --tag TAG [extra args]

Example:
  $0 ICA_ste fig4_experiments --device cuda:0 --pts 4096 --batch 2
EOF
}

# -----------------------------------------------------------------------------
# Paths (aligned with wo.py)
# -----------------------------------------------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="$SCRIPT_DIR/Rotate"
BASE_CFG="$ROOT_DIR/configs/default.yaml"

DRAW_SCRIPT="/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task2_Rotate/Rotate/draw_picture.py"
EVAL_SCRIPT="/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task2_Rotate/Rotate/eval/eval_physics.py"

# -----------------------------------------------------------------------------
# Parse args (wo.sh style)
# -----------------------------------------------------------------------------
CASE=""
TAG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case) CASE="$2"; shift 2 ;;
    --tag)  TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1"); shift
      done ;;
    *)
      if [[ -z "$CASE" ]]; then
        CASE="$1"
      elif [[ -z "$TAG" ]]; then
        TAG="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift ;;
  esac
done

if [[ -z "$CASE" || -z "$TAG" ]]; then
  echo "[ERROR] missing case or tag"
  usage
  exit 1
fi

# -----------------------------------------------------------------------------
# Fig.4 experiment set (paper-aligned)
# -----------------------------------------------------------------------------
EXPERIMENTS=(
  supervised_only
  baseline_equal
  baseline_pcgrad
  core_full
  core_pcgrad
)

NUM_SEEDS=5

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
for EXP in "${EXPERIMENTS[@]}"; do
  for ((SEED=0; SEED<NUM_SEEDS; SEED++)); do

    EXP_TAG="${TAG}_${EXP}_seed${SEED}"

    CFG_DIR="$ROOT_DIR/runs/fig4/${CASE}/${EXP}/seed${SEED}"
    CFG_PATH="$CFG_DIR/config.yaml"

    RESULT_DIR="$ROOT_DIR/results/temp_results/${CASE}/${EXP_TAG}"

    CKPT="$RESULT_DIR/weight/final_reco.pth"
    METRICS_CSV="$RESULT_DIR/train_val_metrics.csv"

    DIAG_DIR="$RESULT_DIR/diagnostics"
    DIAG_PNG="$DIAG_DIR/loss_curve.png"

    PHYS_DIR="$RESULT_DIR/physics"
    PHYS_CSV="$PHYS_DIR/physics_eval.csv"

    echo "============================================================"
    echo "[FIG4] case=${CASE} exp=${EXP} seed=${SEED}"
    echo "       tag=${EXP_TAG}"
    echo "============================================================"

    mkdir -p "$CFG_DIR"

    # -------------------------------------------------------------------------
    # Stage status (true resume semantics)
    # -------------------------------------------------------------------------
    TRAIN_DONE=false
    DIAG_DONE=false
    PHYS_DONE=false

    [[ -f "$CKPT"     ]] && TRAIN_DONE=true
    [[ -f "$DIAG_PNG" ]] && DIAG_DONE=true
    [[ -f "$PHYS_CSV" ]] && PHYS_DONE=true

    if $TRAIN_DONE && $DIAG_DONE && $PHYS_DONE; then
      echo "[SKIP] train + diagnostics + physics all done"
      continue
    fi

    # -------------------------------------------------------------------------
    # Generate config (same override logic as wo.py subsets)
    # -------------------------------------------------------------------------
    if ! $TRAIN_DONE; then
      python - <<EOF
import yaml

base = yaml.safe_load(open("${BASE_CFG}", "r"))

def set_kv(cfg, key, value):
    ks = key.split(".")
    ref = cfg
    for k in ks[:-1]:
        ref = ref.setdefault(k, {})
    ref[ks[-1]] = value

base.setdefault("train", {})
base["train"]["seed"] = ${SEED}

exp = "${EXP}"

if exp == "supervised_only":
    set_kv(base, "pinn.use", False)
    set_kv(base, "teacher.use", False)
    set_kv(base, "teacher.spatial.use", False)
    set_kv(base, "guard.disable", True)
    set_kv(base, "mixed.use", False)
    set_kv(base, "adapt.enable", False)
    set_kv(base, "mo.solver", "sum")

elif exp == "baseline_equal":
    set_kv(base, "guard.disable", True)
    set_kv(base, "mixed.use", False)
    set_kv(base, "adapt.enable", False)
    set_kv(base, "mo.solver", "sum")

elif exp == "baseline_pcgrad":
    set_kv(base, "guard.disable", True)
    set_kv(base, "mixed.use", False)
    set_kv(base, "adapt.enable", False)
    set_kv(base, "mo.solver", "pcgrad")

elif exp == "core_full":
    set_kv(base, "pinn.use", True)
    set_kv(base, "teacher.use", True)
    set_kv(base, "teacher.spatial.use", True)
    set_kv(base, "guard.disable", False)
    set_kv(base, "mixed.use", False)
    set_kv(base, "adapt.enable", False)
    set_kv(base, "mo.solver", "sum")

elif exp == "core_pcgrad":
    set_kv(base, "pinn.use", True)
    set_kv(base, "teacher.use", True)
    set_kv(base, "teacher.spatial.use", True)
    set_kv(base, "guard.disable", False)
    set_kv(base, "mixed.use", False)
    set_kv(base, "adapt.enable", False)
    set_kv(base, "mo.solver", "pcgrad")

yaml.safe_dump(base, open("${CFG_PATH}", "w"), sort_keys=False)
EOF
    fi

    # -------------------------------------------------------------------------
    # 1) TRAIN  (exactly same role as wo.py)
    # -------------------------------------------------------------------------
    if ! $TRAIN_DONE; then
      echo "[RUN] training"
      python -u "$ROOT_DIR/train_tpac_new.py" \
        --root "$ROOT_DIR" \
        --case "$CASE" \
        --tag "$EXP_TAG" \
        --cfg "$CFG_PATH" \
        "${EXTRA_ARGS[@]}"
    else
      echo "[OK] training already done"
    fi

    # -------------------------------------------------------------------------
    # 2) DRAW diagnostics (aligned with wo.py)
    # -------------------------------------------------------------------------
    if $TRAIN_DONE && ! $DIAG_DONE; then
      if [[ -f "$METRICS_CSV" ]]; then
        echo "[RUN] diagnostics plotting"
        mkdir -p "$DIAG_DIR"
        python "$DRAW_SCRIPT" \
          --metrics "$METRICS_CSV" \
          --out-dir "$DIAG_DIR"
      else
        echo "[WARN] missing metrics: $METRICS_CSV"
      fi
    else
      echo "[OK] diagnostics already done"
    fi

    # -------------------------------------------------------------------------
    # 3) PHYSICS evaluation (aligned with wo.py)
    # -------------------------------------------------------------------------
    if $TRAIN_DONE && ! $PHYS_DONE; then
      if [[ -f "$CKPT" ]]; then
        echo "[RUN] physics evaluation"
        mkdir -p "$PHYS_DIR"
        python "$EVAL_SCRIPT" \
          --root "$ROOT_DIR" \
          --case "$CASE" \
          --cfg "$CFG_PATH" \
          --ckpt "$CKPT" \
          --out "$PHYS_CSV"
      else
        echo "[WARN] missing checkpoint: $CKPT"
      fi
    else
      echo "[OK] physics already done"
    fi

  done
done
