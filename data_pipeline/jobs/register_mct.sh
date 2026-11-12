#!/bin/bash
# Step 4b (Option B, 2026-11-12): re-register the 105 flagged MCT-LTDiag cases (NC / AP / DP -> PVP, liver-centre
# shift + the authors' B-spline) as 27 array tasks of 4 cases. Transforms and QC go to outputs/registration/MCT-LTDiag/;
# prepared images to work/register_mct/ (~20 GB, delete after Step 8). Finished registrations are skipped.
# Test 2026-11-12: one case 7.7 min with 2 threads incl. reading its tar, peak 2.1 GB.
# Submit from the repo folder:  qsub -t 1-27 data_pipeline/jobs/register_mct.sh
#$ -S /bin/bash
#$ -N hope_reg_mct
#$ -pe smp 4
#$ -l tmem=4G
#$ -l h_rt=6:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.$TASK_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID task ${SGE_TASK_ID:-none} started $(date) on 4 cores"
THREADS=4
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
status=0
"$HOPE_PYTHON" -I -u data_pipeline/steps/register_mct.py --release "$HOPE_ROOT/processed/v1" --project "$HOPE_ROOT" \
    --work "$HOPE_ROOT/work/register_mct" --out "$HOPE_ROOT/outputs/registration/MCT-LTDiag" \
    --task "$SGE_TASK_ID" --per-task 4 --threads $THREADS || status=$?
echo "Job finished $(date), exit status $status"
exit $status
