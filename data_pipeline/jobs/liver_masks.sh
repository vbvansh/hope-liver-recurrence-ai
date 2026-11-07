#!/bin/bash
# Steps 5 and 8: TotalSegmentator liver masks as an array job over outputs/liver_masks/scans.csv (25 scans per task).
# Each task runs its scans one after another in one process with 3 CPU threads (fair share of a GPU node), straight
# from project storage (21-67 s per scan in test 7715879; first full run: job 7715921).
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
# TotalSegmentator rewrites a usage counter in config.json under TOTALSEG_HOME_DIR on every scan; batches running at
# the same time broke that file (2 scans failed with JSONDecodeError in job 7715921). Each batch therefore gets its own
# home folder on the node's local disk, and all batches read the same model files through TOTALSEG_WEIGHTS_PATH.
export TOTALSEG_WEIGHTS_PATH="$HOPE_ROOT/cache/totalsegmentator/nnunet/results"
export TOTALSEG_HOME_DIR="${TMPDIR:-/tmp}/totalseg_home_${JOB_ID}_${SGE_TASK_ID:-0}"
mkdir -p "$TOTALSEG_HOME_DIR"
PY="$HOPE_ROOT/envs/totalseg/bin/python"
status=0
if [ "$1" = "--test" ]; then
    "$PY" -I -u data_pipeline/steps/liver_masks.py --data "$HOPE_ROOT/data" --out "$HOPE_ROOT/outputs/liver_masks" \
        --test --reference "$HOPE_ROOT/outputs/totalseg_test" --threads $THREADS || status=$?
else
    "$PY" -I -u data_pipeline/steps/liver_masks.py --data "$HOPE_ROOT/data" --out "$HOPE_ROOT/outputs/liver_masks" \
        --task "$SGE_TASK_ID" --threads $THREADS || status=$?
fi
rm -rf "$TOTALSEG_HOME_DIR"
echo "Job finished $(date), exit status $status"
exit $status
