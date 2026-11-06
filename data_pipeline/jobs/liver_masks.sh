#!/bin/bash
# Steps 5 and 8: TotalSegmentator liver masks as an array job over outputs/liver_masks/scans.csv (25 scans per task).
# If envs/totalseg.tar exists, each task first unpacks it and the model files onto the GPU node's local disk;
# otherwise it runs from project storage (fast enough: 21-67 s per scan in test 7715879). Scans run one after another
# in one process with 3 CPU threads (fair share of a GPU node).
# Masks go to outputs/liver_masks/<dataset>/<patient>/liver_<phase>.nii.gz; finished scans are skipped.
# Submit from the repo folder:
#   test (no array):  qsub data_pipeline/jobs/liver_masks.sh --test
#   all tasks, at most 6 at a time:  qsub -t 1-130 -tc 6 data_pipeline/jobs/liver_masks.sh
#$ -S /bin/bash
#$ -N hope_liver
#$ -l gpu=true
#$ -l tmem=16G
#$ -l h_rt=6:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.$TASK_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID task ${SGE_TASK_ID:-none} started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
THREADS=3
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
LOCAL=""
for d in "${TMPDIR:-}" "/scratch0/$USER" /tmp; do
    [ -n "$d" ] && [ -d "$d" ] && [ -w "$d" ] || continue
    free_kb=$(df -Pk "$d" | awk 'NR==2 {print $4}')
    echo "local disk candidate $d: $((free_kb / 1024 / 1024)) GB free"
    if [ -z "$LOCAL" ] && [ "$free_kb" -gt $((15 * 1024 * 1024)) ]; then
        LOCAL="$d/hope_ts_${JOB_ID}_${SGE_TASK_ID:-0}"
    fi
done
PY="$HOPE_ROOT/envs/totalseg/bin/python"
if [ ! -f "$HOPE_ROOT/envs/totalseg.tar" ]; then
    LOCAL=""  # no packed copy: run from project storage (test 7715879: 21-67 s per scan, same masks)
    echo "running from project storage"
fi
if [ -n "$LOCAL" ]; then
    SECONDS=0
    if mkdir -p "$LOCAL/env" && tar -xf "$HOPE_ROOT/envs/totalseg.tar" -C "$LOCAL/env" \
        && "$LOCAL/env/bin/conda-unpack" && tar -xf "$HOPE_ROOT/envs/totalseg_weights.tar" -C "$LOCAL"; then
        PY="$LOCAL/env/bin/python"
        export TOTALSEG_HOME_DIR="$LOCAL/totalsegmentator" TMPDIR="$LOCAL"
        echo "unpacked onto $LOCAL in $SECONDS s"
    else
        echo "unpacking failed: running from project storage (slow)"
    fi
else
    echo "no local disk with 15 GB free: running from project storage (slow)"
fi
status=0
if [ "$1" = "--test" ]; then
    "$PY" -I -u data_pipeline/steps/liver_masks.py --data "$HOPE_ROOT/data" --out "$HOPE_ROOT/outputs/liver_masks" \
        --test --reference "$HOPE_ROOT/outputs/totalseg_test" --threads $THREADS || status=$?
else
    "$PY" -I -u data_pipeline/steps/liver_masks.py --data "$HOPE_ROOT/data" --out "$HOPE_ROOT/outputs/liver_masks" \
        --task "$SGE_TASK_ID" --threads $THREADS || status=$?
fi
[ -n "$LOCAL" ] && rm -rf "$LOCAL"
echo "Job finished $(date), exit status $status"
exit $status
