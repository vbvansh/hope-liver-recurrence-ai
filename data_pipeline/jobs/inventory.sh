#!/bin/bash
# Inventory of one dataset (PLAN.md Step 1) into outputs/inventory/: inventory_<dataset>.csv + lesions_<dataset>.csv.
# Reads only. Time per dataset in tests: MCT-LTDiag ~50 min (needs the blob check's components.csv), PLC-CECT ~75 min,
# WAW-TACE ~20 min, TCIA-CRLM ~20 min, LiTS ~15 min; peak RAM < 1.5 GB.
# Submit from the repo folder, one job per dataset (they can run at the same time):
#   qsub data_pipeline/jobs/inventory.sh PLC-CECT
#$ -S /bin/bash
#$ -N hope_inventory
#$ -l tmem=4G
#$ -l h_rt=4:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date), dataset: $1"
status=0
"$HOPE_PYTHON" -I -u data_pipeline/qc/inventory.py --dataset "$1" --data "$HOPE_ROOT/data" \
    --out "$HOPE_ROOT/outputs/inventory" --blobs "$HOPE_ROOT/outputs/mct_blob_check/components.csv" || status=$?
echo "Job finished $(date), exit status $status"
exit $status
