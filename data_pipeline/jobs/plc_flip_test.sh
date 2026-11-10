#!/bin/bash
# Step 4 QC: decide the slice order of the 2 PLC-CECT volumes that LiverPreprocessing stage 0 leaves unflipped.
# TotalSegmentator finds liver, heart, lungs, kidneys and vertebrae on each volume as stored and with its slices
# reversed (scratch copy only); the anatomy order says which is the right way up. About 15-30 min on one GPU.
# Writes outputs/plc_check/flip_test.csv. Submit from the repo folder:  qsub data_pipeline/jobs/plc_flip_test.sh
#$ -S /bin/bash
#$ -N hope_plc_flip
#$ -l gpu=true
#$ -l tmem=16G
#$ -l h_rt=2:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
THREADS=3
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export TOTALSEG_WEIGHTS_PATH="$HOPE_ROOT/cache/totalsegmentator/nnunet/results"
export TOTALSEG_HOME_DIR="${TMPDIR:-/tmp}/totalseg_home_${JOB_ID}"
mkdir -p "$TOTALSEG_HOME_DIR"
status=0
"$HOPE_ROOT/envs/totalseg/bin/python" -I -u data_pipeline/qc/plc_flip_test.py --data "$HOPE_ROOT/data/PLC-CECT/fixed" \
    --out "$HOPE_ROOT/outputs/plc_check" --threads $THREADS || status=$?
rm -rf "$TOTALSEG_HOME_DIR"
echo "Job finished $(date), exit status $status"
exit $status
