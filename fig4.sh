#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Fig.4 metrics comparison experiments
# Compare train_val_metrics across:
#   baseline_equal
#   baseline_adapt
#   baseline_suponly
#   baseline_pcgrad
#   cgtc_core
#   cgtc_enhance
#   cgtc_pcgrad
#
# Pipeline:
#   train -> draw diagnostics -> physics eval
# =============================================================================

usage() {
  cat <<EOF >&2
Usage:
  $0 CASE TAG [additional arguments for train_tpac_new.py]

Example:
  $0 ICA_norm fig4_metrics --device cuda:0 --pts 4096 --batch 2
EOF
}

# -----------------------------------------------------------------------------
# Path setup (aligned with your current script)
# -----------------------------------------------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="$SCRIPT_DIR/Rotate"

BASE_CFG="$ROOT_DIR/configs/default.yaml"
DRAW_SCRIPT="$ROOT_DIR/draw_picture.py"
EVAL_SCRIPT="$ROOT_DIR/eval/eval_physics.py"

# -----------------------------------------------------------------------------
# Args parsing
# -----------------------------------------------------------------------------
CASE=""
TAG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case) CASE="$2"; shift 2 ;;
    --tag)  TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
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
  echo "Error: missing case or tag." >&2
  usage
  exit 1
fi

# -----------------------------------------------------------------------------
# Experiment list (YOUR REQUEST)
# -----------------------------------------------------------------------------
EXPERIMENTS=(
  baseline_equal
  baseline_adapt
  baseline_suponly
  baseline_pcgrad
  cgtc_core
  cgtc_enhance
  cgtc_pcgrad
)

# >=5 seeds if you later want statistics
NUM_SEEDS=1

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
for EXP in "${EXPERIMENTS[@]}"; do
  for ((SEED=0; SEED<NUM_SEEDS; SEED++)); do

    EXP_TAG="${TAG}_${EXP}_seed${SEED}"
    CFG_DIR="$ROOT_DIR/runs/fig4_metrics/${CASE}/${EXP}/seed${SEED}"
    CFG_PATH="$CFG_DIR/config.yaml"

    RESULT_DIR="$ROOT_DIR/results/temp_results/${CASE}/${EXP_TAG}"
    METRICS_CSV="$RESULT_DIR/train_val_metrics.csv"
    CKPT="$RESULT_DIR/weight/final_reco.pth"
    DIAG_DIR="$RESULT_DIR/diagnostics"
    PHYS_OUT="$RESULT_DIR/physics_eval.csv"

    echo "============================================================"
    echo "[FIG4-METRICS] case=${CASE} exp=${EXP} seed=${SEED}"
    echo "               tag=${EXP_TAG}"
    echo "============================================================"

    mkdir -p "$CFG_DIR"

    # -------------------------------------------------------------------------
    # Generate config
    # -------------------------------------------------------------------------
    python - <<EOF
import yaml

cfg = yaml.safe_load(open("${BASE_CFG}", "r"))

def set_kv(cfg, key, value):
    ks = key.split(".")
    ref = cfg
    for k in ks[:-1]:
        ref = ref.setdefault(k, {})
    ref[ks[-1]] = value

cfg.setdefault("train", {})
cfg["train"]["seed"] = ${SEED}

exp = "${EXP}"

# ---------------- baseline ----------------
if exp == "baseline_suponly":
    set_kv(cfg, "pinn.use", False)
    set_kv(cfg, "teacher.use", False)
    set_kv(cfg, "teacher.spatial.use", False)
    set_kv(cfg, "guard.disable", True)
    set_kv(cfg, "adapt.enable", False)
    set_kv(cfg, "mo.solver", "sum")

elif exp == "baseline_equal":
    set_kv(cfg, "guard.disable", True)
    set_kv(cfg, "adapt.enable", False)
    set_kv(cfg, "mo.solver", "sum")

elif exp == "baseline_adapt":
    set_kv(cfg, "guard.disable", True)
    set_kv(cfg, "adapt.enable", True)
    set_kv(cfg, "mo.solver", "sum")

elif exp == "baseline_pcgrad":
    set_kv(cfg, "guard.disable", True)
    set_kv(cfg, "adapt.enable", False)
    set_kv(cfg, "mo.solver", "pcgrad")

# ---------------- CGTC ----------------
elif exp == "cgtc_core":
    set_kv(cfg, "guard.disable", False)
    set_kv(cfg, "adapt.enable", False)
    set_kv(cfg, "pinn.use", True)
    set_kv(cfg, "teacher.use", True)
    set_kv(cfg, "teacher.spatial.use", True)
    set_kv(cfg, "mo.solver", "sum")

elif exp == "cgtc_enhance":
    set_kv(cfg, "guard.disable", False)
    set_kv(cfg, "adapt.enable", True)
    set_kv(cfg, "pinn.use", True)
    set_kv(cfg, "teacher.use", True)
    set_kv(cfg, "teacher.spatial.use", True)
    set_kv(cfg, "mo.solver", "sum")

elif exp == "cgtc_pcgrad":
    set_kv(cfg, "guard.disable", False)
    set_kv(cfg, "adapt.enable", False)
    set_kv(cfg, "pinn.use", True)
    set_kv(cfg, "teacher.use", True)
    set_kv(cfg, "teacher.spatial.use", True)
    set_kv(cfg, "mo.solver", "pcgrad")

yaml.safe_dump(cfg, open("${CFG_PATH}", "w"), sort_keys=False)
EOF

    # -------------------------------------------------------------------------
    # 1) Train
    # -------------------------------------------------------------------------
    python -u "$ROOT_DIR/train_tpac_new.py" \
      --root "$ROOT_DIR" \
      --case "$CASE" \
      --tag "$EXP_TAG" \
      --cfg "$CFG_PATH" \
      "${EXTRA_ARGS[@]}"

    # -------------------------------------------------------------------------
    # 2) Diagnostics
    # -------------------------------------------------------------------------
    if [[ -f "$METRICS_CSV" ]]; then
      mkdir -p "$DIAG_DIR"
      python "$DRAW_SCRIPT" \
        --metrics "$METRICS_CSV" \
        --out-dir "$DIAG_DIR"
    fi

    # -------------------------------------------------------------------------
    # 3) Physics evaluation
    # -------------------------------------------------------------------------
    if [[ -f "$CKPT" ]]; then
      python "$EVAL_SCRIPT" \
        --root "$ROOT_DIR" \
        --case "$CASE" \
        --cfg "$CFG_PATH" \
        --ckpt "$CKPT" \
        --split test \
        --out "$PHYS_OUT"
    fi

  done
done
