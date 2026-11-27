#!/bin/bash
# nnU-Net smoke test on a 4-case copy of release v1.1 (Dataset999, links in work/nnunet_smoke, made by
# data_pipeline/export/nnunet_export.py --limit 2): ResEnc M planning + preprocessing + 1 training epoch on a GPU +
# validation. Proves the environment, GPU, labels (ignore = 3) and zero-filled phases work before the real jobs.
# Writes work/nnunet_smoke/SMOKE_DONE when everything passed. ~15–30 min. Submit from the repo folder:
#   qsub data_pipeline/jobs/nnunet_smoke_test.sh
#$ -S /bin/bash
#$ -N hope_nnunet_smoke
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -l tmem=24G
#$ -l h_rt=2:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
export PATH="$HOPE_ROOT/envs/totalseg/bin:$PATH"
S="$HOPE_ROOT/work/nnunet_smoke"
export nnUNet_raw="$S/nnUNet_raw" nnUNet_preprocessed="$S/nnUNet_preprocessed" nnUNet_results="$S/nnUNet_results"
export nnUNet_compile=f nnUNet_n_proc_DA=4 OMP_NUM_THREADS=1   # torch.compile off: old compiler toolchain on CentOS 7
echo "Job $JOB_ID started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
rm -f "${S:?}/SMOKE_DONE"
nnUNetv2_plan_and_preprocess -d 999 -pl nnUNetPlannerResEncM -c 3d_fullres --verify_dataset_integrity -np 2 -npfp 2 \
    || { echo "SMOKE TEST FAILED at preprocessing"; exit 1; }
nnUNetv2_train 999 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_1epoch --npz \
    || { echo "SMOKE TEST FAILED at training"; exit 1; }
V="$nnUNet_results/Dataset999_HOPESmokeTest/nnUNetTrainer_1epoch__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation"
n=$(ls "$V"/*.npz 2>/dev/null | wc -l)
echo "validation probability files: $n (expected 2)"
[ "$n" -eq 2 ] || { echo "SMOKE TEST FAILED at validation"; exit 1; }
touch "$S/SMOKE_DONE"
echo "SMOKE TEST PASSED"
echo "Job finished $(date), exit status 0"
