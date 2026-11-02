#!/bin/bash
# Download WAW-TACE (Zenodo 12741586, 10 files, 48.2 GB, MD5-checked) into data/WAW-TACE and unpack its zips
# (~48 GB more; the zips stay until the user agrees to delete them).
# Resumable: a broken download continues where it stopped, so after a failure or timeout simply submit again
# (never two copies at once).
# Submit from the repo folder (the job runs there and logs to logs/):  qsub data_pipeline/jobs/download_waw_tace.sh
#$ -S /bin/bash
#$ -N hope_dl_waw
#$ -l tmem=2G
#$ -l h_rt=12:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date)"
df -h "$HOPE_ROOT" | tail -1
df -i "$HOPE_ROOT" | tail -1
status=0
"$HOPE_PYTHON" -I -u data_pipeline/download/waw_tace.py --out "$HOPE_ROOT/data/WAW-TACE" || status=$?
df -h "$HOPE_ROOT" | tail -1
df -i "$HOPE_ROOT" | tail -1
echo "Job finished $(date), exit status $status"
exit $status
