#!/bin/bash
# Step 10: data check and release tables for processed/v1 (MCT-LTDiag + PLC-CECT): manifest.csv, lesions.csv,
# exclusion.csv, qc/registration_qc.csv, qc/qc_report.md and check pictures in qc/overlays/. About 15-30 min.
# Submit from the repo folder after Step 9:  qsub -hold_jid hope_lesionfree data_pipeline/jobs/release_check.sh
#$ -S /bin/bash
#$ -N hope_release_check
#$ -l tmem=6G
#$ -l h_rt=3:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date)"
export OMP_NUM_THREADS=1
status=0
"$HOPE_PYTHON" -I -u data_pipeline/qc/release_check.py --release "$HOPE_ROOT/processed/v1" --project "$HOPE_ROOT" \
    --datasets MCT-LTDiag PLC-CECT --pictures 6 --value-sample 20 || status=$?
du -sh "$HOPE_ROOT/processed/v1"
echo "Job finished $(date), exit status $status"
exit $status
