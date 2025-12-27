#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF >&2
Usage:
  $0 [--root ROOT] --case CASE --tag BASE_TAG [--out OUTDIR]

What it does:
  - 找到所有由 wo.py 生成、且 tag 前缀为 BASE_TAG 的实验：
      ROOT/results/temp_results/CASE/${BASE_TAG}_*/samples_vis
  - 对每个实验，将 samples_vis 整个目录拷贝到 OUTDIR 下：
      OUTDIR/exp_tag/samples_vis/...

Examples:
  $0 --case ICA_norm --tag 1205
  $0 --case ICA_norm --tag 1205 --out /mnt/e/.../paper_samples_vis
  $0 /mnt/e/.../task2_Rotate ICA_norm 1205 --out /mnt/e/.../paper_samples_vis

Notes:
  - ROOT 默认为脚本所在目录下的 Rotate（与 wo.sh/test.sh 一致）
  - BASE_TAG 就是你跑 wo.sh/test.sh 时用的 tag，例如 1205 / 1205_all
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT="$SCRIPT_DIR/Rotate"

ROOT_DIR=""
CASE=""
BASE_TAG=""
OUT_DIR=""

# 解析参数（兼容位置参数 ROOT CASE TAG）
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      [[ $# -ge 2 ]] || { echo "Error: --root expects a value" >&2; usage; exit 1; }
      ROOT_DIR="$2"
      shift 2
      ;;
    --case)
      [[ $# -ge 2 ]] || { echo "Error: --case expects a value" >&2; usage; exit 1; }
      CASE="$2"
      shift 2
      ;;
    --tag)
      [[ $# -ge 2 ]] || { echo "Error: --tag expects a value" >&2; usage; exit 1; }
      BASE_TAG="$2"
      shift 2
      ;;
    --out)
      [[ $# -ge 2 ]] || { echo "Error: --out expects a value" >&2; usage; exit 1; }
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      # 位置参数模式: ROOT CASE TAG [--out ...]
      if [[ -z "$ROOT_DIR" && -d "$1" ]]; then
        ROOT_DIR="$1"
      elif [[ -z "$CASE" ]]; then
        CASE="$1"
      elif [[ -z "$BASE_TAG" ]]; then
        BASE_TAG="$1"
      else
        echo "Unknown extra argument: $1" >&2
        usage
        exit 1
      fi
      shift
      ;;
  esac
done

# 默认 root
if [[ -z "$ROOT_DIR" ]]; then
  ROOT_DIR="$DEFAULT_ROOT"
fi
ROOT_DIR=$(cd "$ROOT_DIR" && pwd)

# 默认 out：ROOT/collected_samples_vis/CASE/BASE_TAG
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$ROOT_DIR/collected_samples_vis/$CASE/$BASE_TAG"
fi
OUT_DIR=$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)

if [[ -z "$CASE" || -z "$BASE_TAG" ]]; then
  echo "Error: missing --case or --tag." >&2
  usage
  exit 1
fi

echo "[INFO] ROOT   = $ROOT_DIR"
echo "[INFO] CASE   = $CASE"
echo "[INFO] TAG    = $BASE_TAG"
echo "[INFO] OUTDIR = $OUT_DIR"

BASE_RESULTS="$ROOT_DIR/results/temp_results/$CASE"
if [[ ! -d "$BASE_RESULTS" ]]; then
  echo "[ERR] Result dir not found: $BASE_RESULTS" >&2
  exit 1
fi

shopt -s nullglob

# 找所有 ${BASE_TAG}_* 实验目录
EXP_DIRS=( "$BASE_RESULTS"/"${BASE_TAG}"_* )
if [[ ${#EXP_DIRS[@]} -eq 0 ]]; then
  echo "[ERR] No experiments found with prefix ${BASE_TAG}_ under $BASE_RESULTS" >&2
  exit 1
fi

COPIED=0
SKIPPED=0

for exp_dir in "${EXP_DIRS[@]}"; do
  exp_tag=$(basename "$exp_dir")
  src="$exp_dir/samples_vis"

  if [[ ! -d "$src" ]]; then
    echo "[WARN] $exp_tag: samples_vis not found, skip"
    ((SKIPPED++)) || true
    continue
  fi

  dst="$OUT_DIR/$exp_tag"
  mkdir -p "$dst"

  echo "[INFO] Copy $src  ->  $dst/"
  # 保留内部结构：dst/samples_vis/idx_0000/...
  cp -a "$src" "$dst/"

  ((COPIED++)) || true
done

echo
echo "[OK] Finished. Copied samples_vis for $COPIED experiments into:"
echo "     $OUT_DIR"
if [[ $SKIPPED -gt 0 ]]; then
  echo "     (Skipped $SKIPPED experiments without samples_vis)"
fi
