#!/usr/bin/env bash
# WAW-TACE: run one stage of this dataset's processing.
#   bash run.sh [--stage pipeline|tables|report] [--steps "N ..."|all] [--raw DIR] [--processed DIR] [--process DIR] [-- ARGS]
# pipeline (default steps "1 2 3 5"): 1 process, 2 cads, 3 segvol, 4 segvol_fill_empty, 5 build_longitudinal
# tables / report: runs <stage>/NN_*.py (default: all) with ARGS after "--"; see each script's --help.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RAW="${CT_RAW:-/media/cbtil3/WhiteSD/CTRaw/Abdomen}/WAW-TACE"
PROCESSED="${CT_PROCESSED:-/media/cbtil3/WhiteSD/CTProcessed}"
PROCESS="${CT_PROCESS:-/media/cbtil3/WhiteSD/CTProcess}"
PY="${PYTHON:-/home/cbtil3/hao/software/miniforge3/envs/mri/bin/python3}"
SEGVOL_PY="${SEGVOL_PYTHON:-/home/cbtil3/anaconda3/envs/segvol/bin/python}"
STAGE=pipeline
STEPS=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --stage)     STAGE="$2"; shift 2 ;;
    --steps)     STEPS="$2"; shift 2 ;;
    --raw)       RAW="$2"; shift 2 ;;
    --processed) PROCESSED="$2"; shift 2 ;;
    --process)   PROCESS="$2"; shift 2 ;;
    --)          shift; EXTRA=("$@"); break ;;
    -h|--help)   sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -d "$HERE/$STAGE" ] || { echo "unknown stage: $STAGE" >&2; exit 2; }
list_steps() { ls "$HERE/$STAGE" | sed -n 's/^\([0-9][0-9]\)_.*\.py$/\1/p' | sed 's/^0//'; }
if [ -z "$STEPS" ]; then
  if [ "$STAGE" = pipeline ]; then STEPS="1 2 3 5"; else STEPS="$(list_steps | tr '\n' ' ')"; fi
elif [ "$STEPS" = all ]; then
  STEPS="$(list_steps | tr '\n' ' ')"
fi
for n in $STEPS; do
  f="$(ls "$HERE/$STAGE/$(printf '%02d' "$n")_"*.py 2>/dev/null | head -1)"
  [ -n "$f" ] || { echo "no step $n in $STAGE/" >&2; exit 2; }
  echo "=== WAW-TACE: $STAGE/$(basename "$f") ==="
  if [ "$STAGE" = pipeline ]; then
    case "$n" in
      1) "$PY" "$HERE/pipeline/01_process.py" --all --input "$RAW/extracted" --output "$PROCESSED/WAW-TACE" ${EXTRA[@]+"${EXTRA[@]}"} ;;
      2) "$PY" "$HERE/pipeline/02_cads.py" --processed "$PROCESSED" ${EXTRA[@]+"${EXTRA[@]}"} ;;
      3) "$SEGVOL_PY" "$HERE/pipeline/03_segvol.py" --processed "$PROCESSED" --calibrated ${EXTRA[@]+"${EXTRA[@]}"} ;;
      5) "$PY" "$HERE/pipeline/05_build_longitudinal.py" --processed-in "$PROCESSED" --out "$PROCESS" ${EXTRA[@]+"${EXTRA[@]}"} ;;
      *) case "$f" in *segvol*) P="$SEGVOL_PY" ;; *) P="$PY" ;; esac; "$P" "$f" ${EXTRA[@]+"${EXTRA[@]}"} ;;
    esac
  else
    "$PY" "$f" ${EXTRA[@]+"${EXTRA[@]}"}
  fi
done
