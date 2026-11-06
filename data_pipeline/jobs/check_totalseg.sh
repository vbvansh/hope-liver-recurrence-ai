#!/bin/bash
# GPU test of TotalSegmentator (envs/totalseg): liver mask on one scan of MCT-LTDiag, WAW-TACE, TCIA-CRLM and LiTS;
# prints time, GPU memory and Dice against each dataset's own liver outline. Masks go to outputs/totalseg_test/.
# Needs envs/totalseg and the "total" weights in cache/totalsegmentator (downloaded on the login node first).
# Submit from the repo folder (the job runs there and logs to logs/):  qsub data_pipeline/jobs/check_totalseg.sh
#$ -S /bin/bash
#$ -N hope_totalseg_check
#$ -l gpu=true
#$ -l tmem=16G
#$ -l h_rt=2:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "CPU cores usable: $(nproc) of $(nproc --all)"
export OMP_NUM_THREADS=$(nproc) MKL_NUM_THREADS=$(nproc)
status=0
"$HOPE_ROOT/envs/totalseg/bin/python" -I -u data_pipeline/qc/totalseg_check.py --data "$HOPE_ROOT/data" \
    --out "$HOPE_ROOT/outputs/totalseg_test" || status=$?
echo "Job finished $(date), exit status $status"
exit $status
