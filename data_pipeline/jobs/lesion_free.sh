#!/bin/bash
# Step 9: lesion-free liver masks (5 mm margin + no-margin copy) for every case of one dataset in processed/v1/cases.
# About 3-7 s per case with 4 workers in parallel (MCT 517 cases ~10 min). Finished cases are skipped.
# Submit from the repo folder, after Step 8 of that dataset:
#   qsub data_pipeline/jobs/lesion_free.sh MCT-LTDiag
#   qsub -hold_jid hope_std data_pipeline/jobs/lesion_free.sh PLC-CECT     (waits for the Step 8 jobs)
#$ -S /bin/bash
#$ -N hope_lesionfree
#$ -pe smp 4
#$ -l tmem=2G
#$ -l h_rt=4:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date), dataset $1"
export OMP_NUM_THREADS=1
status=0
"$HOPE_PYTHON" -I -u data_pipeline/steps/lesion_free.py --release "$HOPE_ROOT/processed/v1" --dataset "$1" \
    --workers 4 || status=$?
echo "Job finished $(date), exit status $status"
exit $status
