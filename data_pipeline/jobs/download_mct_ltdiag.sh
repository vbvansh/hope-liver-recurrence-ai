#!/bin/bash
# Download MCT-LTDiag per-patient tars (MD5-checked, kept as tars) and its two tables into data/MCT-LTDiag.
# Arguments go to the script: --sample = one tar per tumour-type letter (5 tars, ~1.7 GB); none = all 517 (180 GB,
# then also raise the time limit, e.g. qsub -l h_rt=24:00:00 ...).
# Resumable: finished tars are skipped, so after a failure or timeout simply submit again (never two copies at once).
# Submit from the repo folder (the job runs there and logs to logs/):
#   qsub data_pipeline/jobs/download_mct_ltdiag.sh --sample
#$ -S /bin/bash
#$ -N hope_dl_mct
#$ -l tmem=2G
#$ -l h_rt=6:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date), arguments: $*"
df -h "$HOPE_ROOT" | tail -1
df -i "$HOPE_ROOT" | tail -1
status=0
"$HOPE_PYTHON" -I -u data_pipeline/download/mct_ltdiag.py --out "$HOPE_ROOT/data/MCT-LTDiag" "$@" || status=$?
df -h "$HOPE_ROOT" | tail -1
df -i "$HOPE_ROOT" | tail -1
echo "Job finished $(date), exit status $status"
exit $status
