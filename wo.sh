#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF >&2
Usage: $0 [--case CASE] [--tag TAG] [additional arguments for wo.py]
       $0 CASE TAG [additional arguments for wo.py]

Examples:
  $0 C1 1015 --device cuda:0 --pts 16384 --batch 2
  $0 --case ICA_ste --tag 1205 --device cuda:0 --pts 4096 --batch 2
  $0 --case ICA_norm --tag 1205 --only-pt --resume --device cuda:0
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="$SCRIPT_DIR/Rotate"

CASE=""
TAG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case)
      [[ $# -ge 2 ]] || { echo "Error: --case expects a value" >&2; usage; exit 1; }
      CASE="$2"
      shift 2
      ;;
    --tag)
      [[ $# -ge 2 ]] || { echo "Error: --tag expects a value" >&2; usage; exit 1; }
      TAG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      if [[ -z "$CASE" ]]; then
        CASE="$1"
      elif [[ -z "$TAG" ]]; then
        TAG="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "$CASE" || -z "$TAG" ]]; then
  echo "Error: missing case or tag." >&2
  usage
  exit 1
fi

python -u "$ROOT_DIR/wo.py" \
  --root "$ROOT_DIR" \
  --case "$CASE" \
  --tag "$TAG" \
  "${EXTRA_ARGS[@]}"
