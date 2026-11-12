#!/bin/bash
# Open item 11 (Option B, 2026-11-11): test our PLC registration method (liver-centre shift + the MCT authors'
# B-spline) on the 7 MCT-LTDiag cases the authors' registration aligns worst. Measures only; the release is unchanged.
# Writes work/mct_registration_test/{data,livers,registration} (~1.5 GB, delete after the decision) and prints a
# before/after table. About 40 min (test 11 Nov: ~5 min per patient, peak 2.7 GB).
# Submit from the repo folder:  qsub data_pipeline/jobs/mct_registration_test.sh
#$ -S /bin/bash
#$ -N hope_mct_regtest
#$ -pe smp 4
#$ -l tmem=4G
#$ -l h_rt=4:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date)"
THREADS=4
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
W="$HOPE_ROOT/work/mct_registration_test"
status=0
"$HOPE_PYTHON" -I -u data_pipeline/qc/mct_registration_test.py prepare --release "$HOPE_ROOT/processed/v1" \
    --project "$HOPE_ROOT" --work "$W" --n 7 || status=$?
if [ $status -eq 0 ]; then
    "$HOPE_PYTHON" -I -u data_pipeline/steps/register_plc.py --data "$W/data" --livers "$W/livers" \
        --out "$W/registration" --task 1 --per-task 7 --threads $THREADS || status=$?
fi
if [ $status -eq 0 ]; then
    "$HOPE_PYTHON" -I -u data_pipeline/qc/mct_registration_test.py summary --work "$W" || status=$?
fi
echo "Job finished $(date), exit status $status"
exit $status
