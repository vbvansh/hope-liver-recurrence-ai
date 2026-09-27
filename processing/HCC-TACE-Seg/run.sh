#!/usr/bin/env bash
# HCC-TACE-Seg: run one stage of this dataset's processing.
#   bash run.sh [--stage pipeline|tables|report] [--steps "N ..."|all] [--raw DIR] [--processed DIR] [--process DIR] [-- ARGS]
# pipeline (default steps "1 2 3 4 6"): 1 standardize, 2 seg_to_mask, 3 cads, 4 segvol, 5 segvol_fill_empty, 6 build_longitudinal
# tables / report: runs <stage>/NN_*.py (default: all) with ARGS after "--"; see each script's --help.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RAW="${CT_RAW:-/media/cbtil3/WhiteSD/CTRaw/Abdomen}/HCC-TACE-Seg "
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
  if [ "$STAGE" = pipeline ]; then STEPS="1 2 3 4 6"; else STEPS="$(list_steps | tr '\n' ' ')"; fi
elif [ "$STEPS" = all ]; then
  STEPS="$(list_steps | tr '\n' ' ')"
fi
for n in $STEPS; do
  f="$(ls "$HERE/$STAGE/$(printf '%02d' "$n")_"*.py 2>/dev/null | head -1)"
  [ -n "$f" ] || { echo "no step $n in $STAGE/" >&2; exit 2; }
  echo "=== HCC-TACE-Seg: $STAGE/$(basename "$f") ==="
  if [ "$STAGE" = pipeline ]; then
    case "$n" in
      1) "$PY" "$HERE/pipeline/01_standardize.py" --input "$RAW" --output "$PROCESSED" ${EXTRA[@]+"${EXTRA[@]}"} ;;
      2) "$PY" "$HERE/pipeline/02_seg_to_mask.py" --all --output "$PROCESSED/HCC-TACE-Seg" ${EXTRA[@]+"${EXTRA[@]}"} ;;
      3) "$PY" "$HERE/pipeline/03_cads.py" --processed "$PROCESSED" ${EXTRA[@]+"${EXTRA[@]}"} ;;
      4) "$SEGVOL_PY" "$HERE/pipeline/04_segvol.py" --processed "$PROCESSED" --calibrated ${EXTRA[@]+"${EXTRA[@]}"} ;;
      6) "$PY" "$HERE/pipeline/06_build_longitudinal.py" --processed-in "$PROCESSED" --out "$PROCESS" ${EXTRA[@]+"${EXTRA[@]}"} ;;
      *) case "$f" in *segvol*) P="$SEGVOL_PY" ;; *) P="$PY" ;; esac; "$P" "$f" ${EXTRA[@]+"${EXTRA[@]}"} ;;
    esac
  else
    "$PY" "$f" ${EXTRA[@]+"${EXTRA[@]}"}
  fi
done
