#!/bin/bash
# A4 (label maps) check: re-read every TCIA-CRLM tumour as its own mask (tumour_stack, ~15 min), then draw overlay
# pictures of the standard labels for 6 random patients per dataset into outputs/label_overlays/ (seed 0).
# Reads data/ and writes only tumour_stack.* next to the TCIA NIfTI files. Resumable.
# Submit from the repo folder (the job runs there and logs to logs/):  qsub data_pipeline/jobs/label_check.sh
#$ -S /bin/bash
#$ -N hope_label_check
#$ -l tmem=8G
#$ -l h_rt=3:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date)"
status=0
"$HOPE_PYTHON" -I -u data_pipeline/convert/tcia_crlm_tumours.py --raw "$HOPE_ROOT/data/TCIA-CRLM" || status=$?
for ds in MCT-LTDiag PLC-CECT WAW-TACE TCIA-CRLM LiTS; do
    echo "--- overlays $ds"
    "$HOPE_PYTHON" -I -u data_pipeline/qc/label_overlays.py --dataset "$ds" --data "$HOPE_ROOT/data" \
        --inventory "$HOPE_ROOT/outputs/inventory" --out "$HOPE_ROOT/outputs/label_overlays" --n 6 || status=$?
done
echo "Job finished $(date), exit status $status"
exit $status
