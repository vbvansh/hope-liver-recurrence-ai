#!/bin/bash
# CancerVerse (external, research only): measure the liver tumour masks straight from the label archive, without
# unpacking it (it would create ~317,000 files). Reads data/CancerVerse/labels/CancerVerse_Label.tar.gz, writes
# outputs/cancerverse/liver_masks/{scans,lesions,members}.csv. Then run the summary on the login node:
#   "$HOPE_PYTHON" -I data_pipeline/explore/cancerverse_liver_summary.py
# Submit from the repo folder:  qsub data_pipeline/jobs/cancerverse_liver_masks.sh
#$ -S /bin/bash
#$ -N hope_cv_liver
#$ -pe smp 12
#$ -l tmem=2G
#$ -l h_rt=12:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
export OMP_NUM_THREADS=1
echo "Job $JOB_ID started $(date) on $(hostname -s), slots ${NSLOTS:-?}"
status=0
"$HOPE_PYTHON" -I -u data_pipeline/explore/cancerverse_liver_masks.py \
    --archive "$HOPE_ROOT/data/CancerVerse/labels/CancerVerse_Label.tar.gz" \
    --out "$HOPE_ROOT/outputs/cancerverse/liver_masks" --workers "${NSLOTS:-8}" "$@" || status=$?
echo "Job finished $(date), exit status $status"
exit $status
