#!/bin/bash
# Step 7: measure every lesion of one dataset on its original scan (voxel-centre diameters) into outputs/lesions/:
# lesions_<dataset>.csv + patients_<dataset>.csv. Reads only. One job per dataset; they can run at the same time.
# Submit from the repo folder (the job runs there and logs to logs/):  qsub data_pipeline/jobs/lesions.sh MCT-LTDiag
#$ -S /bin/bash
#$ -N hope_lesions
#$ -l tmem=8G
#$ -l h_rt=8:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date), dataset: $1"
status=0
"$HOPE_PYTHON" -I -u data_pipeline/steps/lesions.py --dataset "$1" --data "$HOPE_ROOT/data" \
    --inventory "$HOPE_ROOT/outputs/inventory" --out "$HOPE_ROOT/outputs/lesions" || status=$?
echo "Job finished $(date), exit status $status"
exit $status
