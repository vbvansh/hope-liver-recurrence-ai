#!/bin/bash
# EXP-001: score the validation predictions with the HOPE shared scorer (evaluation/). Runs only after training
# finished (TRAIN_DONE). Steps: nnU-Net validation probabilities -> candidates (evaluation/candidates_from_nnunet.py)
# -> scorer (evaluation/scorer.py, team rules) on the 85 validation cases. Full outputs (with case ids):
# outputs/experiments/EXP-001/; small tables (counts only) copied to this experiment's results/. ~10-30 min.
# Submit from the repo folder:
#   qsub -hold_jid hope_exp001_train experiments/EXP-001_nnunet-resenc-m-baseline/score.sh
#$ -S /bin/bash
#$ -N hope_exp001_score
#$ -l tmem=8G
#$ -l h_rt=4:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
export OMP_NUM_THREADS=1
EXP=experiments/EXP-001_nnunet-resenc-m-baseline
V="$HOPE_ROOT/checkpoints/EXP-001/nnUNet_results/Dataset501_HOPELiverV11/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0"
O="$HOPE_ROOT/outputs/experiments/EXP-001"
echo "Job $JOB_ID started $(date)"
[ -f "$V/TRAIN_DONE" ] || { echo "STOP: training has not finished (no TRAIN_DONE)"; exit 1; }
"$HOPE_PYTHON" -I -u evaluation/candidates_from_nnunet.py --npz-dir "$V/validation" --release "$HOPE_ROOT/processed/v1" \
    --out "$O/candidates_val" || { echo "CANDIDATES FAILED"; exit 1; }
"$HOPE_PYTHON" -I -u evaluation/scorer.py --release "$HOPE_ROOT/processed/v1" --cases "$HOPE_ROOT/processed/v1/splits/val.txt" \
    --pred "$O/candidates_val" --out "$O/score_val" --bootstrap 1000 || { echo "SCORER FAILED"; exit 1; }
mkdir -p "$EXP/results"
cp "$O/score_val/summary.md" "$O/score_val/metrics.csv" "$O/score_val/froc.csv" "$O/score_val/settings.json" "$EXP/results/"
echo "Job finished $(date), exit status 0"
