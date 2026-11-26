#!/bin/bash
# CancerVerse (external, research only, CC BY-NC-ND 4.0): download chosen CT volumes (SHA-256 checked) into
# data/CancerVerse/ct/<cv_id>/ct.nii.gz and extract only their liver masks from the label archive.
# Arguments: subset names in outputs/cancerverse/subsets/ (default: pilot_all_dates hcc_visual_review).
# Resumable: finished files are skipped. Submit from the repo folder:
#   qsub data_pipeline/jobs/cancerverse_download.sh [subset ...]
#$ -S /bin/bash
#$ -N hope_cv_dl
#$ -pe smp 2
#$ -l tmem=2G
#$ -l h_rt=6:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
subsets=("$@"); [ ${#subsets[@]} -eq 0 ] && subsets=(pilot_all_dates hcc_visual_review)
lists=(); for s in "${subsets[@]}"; do lists+=("$HOPE_ROOT/outputs/cancerverse/subsets/$s.csv"); done
echo "Job $JOB_ID started $(date), subsets: ${subsets[*]}"
df -h "$HOPE_ROOT" | tail -1
status=0
"$HOPE_PYTHON" -I -u data_pipeline/explore/cancerverse_download.py --lists "${lists[@]}" \
    --out "$HOPE_ROOT/data/CancerVerse/ct" --archive "$HOPE_ROOT/data/CancerVerse/labels/CancerVerse_Label.tar.gz" \
    --workers 4 || status=$?
df -h "$HOPE_ROOT" | tail -1
echo "Job finished $(date), exit status $status"
exit $status
