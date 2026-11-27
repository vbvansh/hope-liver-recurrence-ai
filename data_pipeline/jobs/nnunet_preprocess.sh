#!/bin/bash
# nnU-Net planning + preprocessing (ResEnc M plans, 3d_fullres) of an exported dataset (default 501 = release v1.1,
# processed/nnunet/nnUNet_raw, made by data_pipeline/export/nnunet_export.py). Runs only after the smoke test passed
# (work/nnunet_smoke/SMOKE_DONE). Output: processed/nnunet/nnUNet_preprocessed/Dataset501_HOPELiverV11/
# (~34-40 GB, blosc2 compressed) and PREPROCESS_DONE there. ~1-2 h. Submit from the repo folder:
#   qsub -hold_jid hope_nnunet_smoke data_pipeline/jobs/nnunet_preprocess.sh [dataset_id]
#$ -S /bin/bash
#$ -N hope_nnunet_pre
#$ -pe smp 6
#$ -l tmem=6G
#$ -l h_rt=12:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
export PATH="$HOPE_ROOT/envs/totalseg/bin:$PATH"
export nnUNet_raw="$HOPE_ROOT/processed/nnunet/nnUNet_raw" nnUNet_preprocessed="$HOPE_ROOT/processed/nnunet/nnUNet_preprocessed"
export nnUNet_results="$HOPE_ROOT/checkpoints/nnunet_unused" nnUNet_compile=f OMP_NUM_THREADS=1
ID="${1:-501}"
echo "Job $JOB_ID started $(date), dataset $ID"
[ -f "$HOPE_ROOT/work/nnunet_smoke/SMOKE_DONE" ] || { echo "STOP: the smoke test has not passed (no SMOKE_DONE)"; exit 1; }
df -h "$HOPE_ROOT" | tail -1
D=$(ls -d "$nnUNet_raw"/Dataset$(printf %03d "$ID")_* | head -1); NAME=$(basename "$D")
P="$nnUNet_preprocessed/$NAME"
rm -f "${P:?}/PREPROCESS_DONE"
nnUNetv2_plan_and_preprocess -d "$ID" -pl nnUNetPlannerResEncM -c 3d_fullres --verify_dataset_integrity -np 6 -npfp 6 \
    || { echo "PREPROCESSING FAILED"; exit 1; }
want=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['numTraining'])" "$D/dataset.json")
# The data folder is named by the plans' data_identifier (ResEnc plans keep nnUNetPlans_3d_fullres), so read it.
DI=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['configurations']['3d_fullres']['data_identifier'])" \
    "$P/nnUNetResEncUNetMPlans.json")
have=$(ls "$P/$DI"/*_seg.b2nd 2>/dev/null | wc -l)
echo "preprocessed cases: $have / $want"; du -sh "$P"
[ "$have" -eq "$want" ] || { echo "PREPROCESSING INCOMPLETE"; exit 1; }
touch "$P/PREPROCESS_DONE"
df -h "$HOPE_ROOT" | tail -1
echo "Job finished $(date), exit status 0"
