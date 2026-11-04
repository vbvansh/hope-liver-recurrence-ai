#!/bin/bash
# MCT blob check (PLAN.md Step 3): measure every connected piece in all 517 MCT-LTDiag tumour masks, read straight
# from the tars, into outputs/mct_blob_check/ (components.csv, patients.csv, summary.md; ~11 s per patient,
# peak RAM ~1.7 GB in tests). Reads only; changes nothing in data/.
# Submit from the repo folder (the job runs there and logs to logs/):  qsub data_pipeline/jobs/mct_blob_check.sh
#$ -S /bin/bash
#$ -N hope_mct_blobs
#$ -l tmem=4G
#$ -l h_rt=4:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date)"
status=0
"$HOPE_PYTHON" -I -u data_pipeline/qc/mct_blob_check.py --raw "$HOPE_ROOT/data/MCT-LTDiag" \
    --out "$HOPE_ROOT/outputs/mct_blob_check" || status=$?
echo "Job finished $(date), exit status $status"
exit $status
