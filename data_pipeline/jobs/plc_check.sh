#!/bin/bash
# Steps 4-5 QC: check all 1,444 corrected PLC-CECT volumes for an upside-down scan or a failed TotalSegmentator liver
# (lung position, liver volume vs the patient's other phases, Dice with the released liver mask).
# Writes outputs/plc_check/plc_check.csv and summary.md (counts only). About 30 min with 4 cores.
# Submit from the repo folder:  qsub data_pipeline/jobs/plc_check.sh
#$ -S /bin/bash
#$ -N hope_plc_check
#$ -pe smp 4
#$ -l tmem=3G
#$ -l h_rt=3:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date)"
export OMP_NUM_THREADS=1
status=0
"$HOPE_PYTHON" -I -u data_pipeline/qc/plc_orientation_check.py --data "$HOPE_ROOT/data/PLC-CECT/fixed" \
    --livers "$HOPE_ROOT/outputs/liver_masks/PLC-CECT" --out "$HOPE_ROOT/outputs/plc_check" --workers 4 || status=$?
echo "Job finished $(date), exit status $status"
exit $status
