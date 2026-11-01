#!/bin/bash
# Convert TCIA Colorectal-Liver-Metastases DICOM (CT + SEG) to NIfTI at native resolution and write the inventory
# (data/TCIA-CRLM/nifti/, ~7.5 GB, ~1,600 files; ~20 s per patient, peak RAM < 1 GB in tests).
# Resumable: converted patients are skipped, so after a failure or timeout simply submit again (never two copies at once).
# Submit from the repo folder (the job runs there and logs to logs/):  qsub data_pipeline/jobs/convert_tcia_crlm.sh
#$ -S /bin/bash
#$ -N hope_conv_tcia_crlm
#$ -l tmem=4G
#$ -l h_rt=3:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date)"
df -h "$HOPE_ROOT" | tail -1
df -i "$HOPE_ROOT" | tail -1
status=0
"$HOPE_PYTHON" -I -u data_pipeline/convert/tcia_crlm.py --raw "$HOPE_ROOT/data/TCIA-CRLM" || status=$?
df -h "$HOPE_ROOT" | tail -1
df -i "$HOPE_ROOT" | tail -1
echo "Job finished $(date), exit status $status"
exit $status
