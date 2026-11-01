#!/bin/bash
# Download LiTS (MSD Task03_Liver.tar, 28.9 GB, MD5-checked) into data/LiTS and unpack only the 131 labelled scans
# (~20 GB more; the tar stays until the user agrees to delete it).
# Resumable: a broken download continues where it stopped, so after a failure or timeout simply submit again
# (never two copies at once).
# Submit from the repo folder (the job runs there and logs to logs/):  qsub data_pipeline/jobs/download_lits.sh
#$ -S /bin/bash
#$ -N hope_dl_lits
#$ -l tmem=2G
#$ -l h_rt=8:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date)"
df -h "$HOPE_ROOT" | tail -1
df -i "$HOPE_ROOT" | tail -1
status=0
"$HOPE_PYTHON" -I -u data_pipeline/download/lits.py --out "$HOPE_ROOT/data/LiTS" || status=$?
df -h "$HOPE_ROOT" | tail -1
df -i "$HOPE_ROOT" | tail -1
echo "Job finished $(date), exit status $status"
exit $status
