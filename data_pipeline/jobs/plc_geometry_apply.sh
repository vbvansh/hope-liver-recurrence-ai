#!/bin/bash
# PLC-CECT geometry fix: write corrected volumes for all 361 patients (true voxel size, right way up, HU + 40) into
# data/PLC-CECT/fixed/<patient>/ (~48 GB; ~25 s per patient, 4 at a time). The original files stay untouched.
# Finished patients are skipped, so after a failure or timeout simply submit again.
# Submit from the repo folder:  qsub data_pipeline/jobs/plc_geometry_apply.sh
#$ -S /bin/bash
#$ -N hope_plc_apply
#$ -pe smp 4
#$ -l tmem=4G
#$ -l h_rt=6:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date)"
df -h "$HOPE_ROOT" | tail -1
export OMP_NUM_THREADS=1
status=0
"$HOPE_PYTHON" -I -u data_pipeline/convert/plc_cect_geometry.py apply --raw "$HOPE_ROOT/data/PLC-CECT" \
    --out "$HOPE_ROOT/data/PLC-CECT/fixed" --workers 4 || status=$?
df -h "$HOPE_ROOT" | tail -1
echo "Job finished $(date), exit status $status"
exit $status
