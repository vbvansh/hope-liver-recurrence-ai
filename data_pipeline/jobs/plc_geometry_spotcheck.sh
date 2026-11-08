#!/bin/bash
# PLC-CECT geometry spot check: rerun LiverPreprocessing stage 0 (TotalSegmentator 3 mm model, task 297, downloaded
# into cache/totalsegmentator on first use) on 10 patients and compare slice spacing and slice order with Yifan's
# reference table. Output in outputs/plc_geometry_check/. Minutes of GPU time.
# Submit from the repo folder:  qsub data_pipeline/jobs/plc_geometry_spotcheck.sh
#$ -S /bin/bash
#$ -N hope_plc_spotcheck
#$ -l gpu=true
#$ -l tmem=16G
#$ -l h_rt=2:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export TOTALSEG_WEIGHTS_PATH="$HOPE_ROOT/cache/totalsegmentator/nnunet/results"
export TOTALSEG_HOME_DIR="${TMPDIR:-/tmp}/totalseg_home_$JOB_ID"
mkdir -p "$TOTALSEG_HOME_DIR"
status=0
"$HOPE_ROOT/envs/totalseg/bin/python" -I -u data_pipeline/convert/plc_cect_geometry.py spotcheck \
    --raw "$HOPE_ROOT/data/PLC-CECT" --out "$HOPE_ROOT/outputs/plc_geometry_check" --n 10 || status=$?
echo "Job finished $(date), exit status $status"
exit $status
