#!/bin/bash
# Step 14: final automatic checks of release v1 (qc/release_tests.py): split integrity and reproducibility, every
# tumour type and size stratum in every split, crops from the liver only, phases and missing phases, formats of every
# file, counts vs publications and lesion tables, native lesion sizes. Writes processed/v1/qc/release_tests.md.
# About 20-40 min (headers of ~16,500 files, 878 liver masks, images of 100 sampled cases). Checks 1-3 passed on the
# login node on 12 Nov before that run was interrupted.
# Submit from the repo folder:  qsub data_pipeline/jobs/release_tests.sh
#$ -S /bin/bash
#$ -N hope_release_tests
#$ -l tmem=4G
#$ -l h_rt=3:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date)"
export OMP_NUM_THREADS=1
status=0
"$HOPE_PYTHON" -I -u data_pipeline/qc/release_tests.py --release "$HOPE_ROOT/processed/v1" --project "$HOPE_ROOT" \
    --value-sample 50 || status=$?
echo "Job finished $(date), exit status $status"
exit $status
