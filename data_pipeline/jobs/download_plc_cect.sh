#!/bin/bash
# Download PLC-CECT from its Google Drive copy into data/PLC-CECT: 4,000 NIfTI files (~42 GB, the same files as the
# official Science Data Bank tar, which is too slow here) + patient_data.csv; every file is checked (gzip CRC,
# NIfTI header). 4 files download at the same time.
# Resumable: finished files are skipped, so after a failure, a Drive limit or a timeout simply submit again
# (never two copies at once).
# Submit from the repo folder (the job runs there and logs to logs/):  qsub data_pipeline/jobs/download_plc_cect.sh
#$ -S /bin/bash
#$ -N hope_dl_plc
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
"$HOPE_PYTHON" -I -u data_pipeline/download/plc_cect.py --out "$HOPE_ROOT/data/PLC-CECT" --workers 4 || status=$?
df -h "$HOPE_ROOT" | tail -1
df -i "$HOPE_ROOT" | tail -1
echo "Job finished $(date), exit status $status"
exit $status
