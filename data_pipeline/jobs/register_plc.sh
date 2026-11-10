#!/bin/bash
# Step 4: register PLC-CECT NC, AP and DP onto PVP (ITKElastix 0.20.0 default B-spline map, 3 resolutions, 20 mm grid;
# the MCT-LTDiag authors' settings) as an array job: 4 patients (12 registrations) per task, 91 tasks for 361 patients.
# Writes outputs/registration/PLC-CECT/<patient>/<phase>_to_pvp.txt (transform parameter files) and qc/task_<K>.csv
# (liver Dice and gradient NCC before / after). Finished registrations are skipped, so a task can be submitted again.
# Needs itk-elastix==0.20.0 in envs/hope. Submit from the repo folder:
#   all tasks, at most 30 at a time:  qsub -t 1-91 -tc 30 data_pipeline/jobs/register_plc.sh
# Test 2026-11-09 (login node): 43-134 s per registration, peak 5.3 GB for a 147-slice patient (largest has 318).
#   one task again:                   qsub -t <K> data_pipeline/jobs/register_plc.sh
#$ -S /bin/bash
#$ -N hope_reg_plc
#$ -pe smp 4
#$ -l tmem=4G
#$ -l h_rt=6:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.$TASK_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID task ${SGE_TASK_ID:-none} started $(date) on 4 cores"
THREADS=4
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
status=0
"$HOPE_PYTHON" -I -u data_pipeline/steps/register_plc.py --data "$HOPE_ROOT/data/PLC-CECT/fixed" \
    --livers "$HOPE_ROOT/outputs/liver_masks/PLC-CECT" --out "$HOPE_ROOT/outputs/registration/PLC-CECT" \
    --task "$SGE_TASK_ID" --per-task 4 --threads $THREADS || status=$?
echo "Job finished $(date), exit status $status"
exit $status
