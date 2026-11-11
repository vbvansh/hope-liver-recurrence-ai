#!/bin/bash
# Step 8: standard case folders (1 mm, PVP liver box + 20 mm, HU [-200, 200], labels) for MCT-LTDiag or PLC-CECT,
# as an array job of 20 patients per task into processed/v1/cases/ (QC rows in processed/v1/qc/).
# Finished cases are skipped, so a task can simply be submitted again (never two copies of one task at once).
# Submit from the repo folder:
#   MCT-LTDiag (517 patients, 26 tasks):  qsub -t 1-26 data_pipeline/jobs/standardise.sh MCT-LTDiag
#   PLC-CECT   (361 patients, 19 tasks):  qsub -t 1-19 data_pipeline/jobs/standardise.sh PLC-CECT
#   test on N random patients into processed/v1_test:  qsub data_pipeline/jobs/standardise.sh PLC-CECT --test 2
# Tests 2026-11-10: MCT ~20 s per patient, peak 1.2 GB; PLC peak > 2.5 GB (killed on the 7 GB login node).
#$ -S /bin/bash
#$ -N hope_std
#$ -pe smp 4
#$ -l tmem=4G
#$ -l h_rt=8:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.$TASK_ID.log

source data_pipeline/jobs/project_env.sh
DATASET="$1"
shift
THREADS=4
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
if [ "$1" = "--test" ]; then
    OUT="$HOPE_ROOT/processed/v1_test"
    SELECT=(--test "$2")
else
    OUT="$HOPE_ROOT/processed/v1"
    SELECT=(--task "$SGE_TASK_ID" --per-task 20)
fi
echo "Job $JOB_ID task ${SGE_TASK_ID:-none} started $(date), dataset $DATASET, ${SELECT[*]}"
df -h "$HOPE_ROOT" | tail -1
status=0
"$HOPE_PYTHON" -I -u data_pipeline/steps/standardise.py --dataset "$DATASET" --project "$HOPE_ROOT" --out "$OUT" \
    "${SELECT[@]}" --threads $THREADS || status=$?
echo "Job finished $(date), exit status $status"
exit $status
