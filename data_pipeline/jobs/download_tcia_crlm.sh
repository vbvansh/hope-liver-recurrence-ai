#!/bin/bash
# Download TCIA Colorectal-Liver-Metastases (CT + SEG, clinical table) into data/TCIA-CRLM and unpack the series zips
# (~10.9 GB of zips + ~10.9 GB unpacked, ~18,600 files; the zips stay until the user agrees to delete them).
# Resumable: unpacked series are skipped, so after a failure or timeout simply submit again (never two copies at once).
# Submit from the repo folder (the job runs there and logs to logs/):  qsub data_pipeline/jobs/download_tcia_crlm.sh
#$ -S /bin/bash
#$ -N hope_dl_tcia_crlm
#$ -l tmem=2G
#$ -l h_rt=6:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date)"
df -h "$HOPE_ROOT" | tail -1
df -i "$HOPE_ROOT" | tail -1
status=0
"$HOPE_PYTHON" -I -u data_pipeline/download/tcia_crlm.py --out "$HOPE_ROOT/data/TCIA-CRLM" || status=$?
df -h "$HOPE_ROOT" | tail -1
df -i "$HOPE_ROOT" | tail -1
echo "Job finished $(date), exit status $status"
exit $status
