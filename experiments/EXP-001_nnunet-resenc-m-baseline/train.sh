#!/bin/bash
# EXP-001: train nnU-Net ResEnc M (3d_fullres, 4 phases as 4 channels) on release v1.1 fold 0 = the official split
# (train 588 / val 85), then nnU-Net's final validation with probabilities (--npz) for our shared scorer. Runs only
# after preprocessing finished (PREPROCESS_DONE). Resumable: resubmit and it continues from checkpoint_latest
# (saved every 50 epochs). Model + validation predictions: checkpoints/EXP-001/nnUNet_results/. ~1.5-3 days.
# Submit from the repo folder:
#   qsub -hold_jid hope_nnunet_pre experiments/EXP-001_nnunet-resenc-m-baseline/train.sh
#$ -S /bin/bash
#$ -N hope_exp001_train
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -l tmem=48G
#$ -l h_rt=96:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
export PATH="$HOPE_ROOT/envs/totalseg/bin:$PATH"
export nnUNet_raw="$HOPE_ROOT/processed/nnunet/nnUNet_raw" nnUNet_preprocessed="$HOPE_ROOT/processed/nnunet/nnUNet_preprocessed"
export nnUNet_results="$HOPE_ROOT/checkpoints/EXP-001/nnUNet_results"
export nnUNet_compile=f nnUNet_n_proc_DA=8 OMP_NUM_THREADS=1   # torch.compile off: old compiler toolchain on CentOS 7
echo "Job $JOB_ID started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
P="$nnUNet_preprocessed/Dataset501_HOPELiverV11"
[ -f "$P/PREPROCESS_DONE" ] || { echo "STOP: preprocessing has not finished (no PREPROCESS_DONE)"; exit 1; }
OUT="$nnUNet_results/Dataset501_HOPELiverV11/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0"
ARGS=(501 3d_fullres 0 -p nnUNetResEncUNetMPlans --npz)
if [ -f "$OUT/checkpoint_final.pth" ]; then
    n=$(ls "$OUT"/validation/*.npz 2>/dev/null | wc -l)
    if [ "$n" -lt 85 ]; then echo "training finished; running the validation only"; nnUNetv2_train "${ARGS[@]}" --val || exit 1; fi
elif [ -f "$OUT/checkpoint_latest.pth" ]; then
    echo "continuing from checkpoint_latest"; nnUNetv2_train "${ARGS[@]}" --c || exit 1
else
    nnUNetv2_train "${ARGS[@]}" || exit 1
fi
n=$(ls "$OUT"/validation/*.npz 2>/dev/null | wc -l)
echo "validation probability files: $n / 85"
[ "$n" -eq 85 ] || { echo "VALIDATION INCOMPLETE"; exit 1; }
touch "$OUT/TRAIN_DONE"
echo "Job finished $(date), exit status 0"
