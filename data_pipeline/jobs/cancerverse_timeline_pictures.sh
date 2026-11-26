#!/bin/bash
# CancerVerse: one timeline picture per patient (every date side by side, tumour outlines in red) for the data
# lead's visual review. Reads data/CancerVerse/ct, writes outputs/cancerverse/review_pictures/<subset>/.
# Arguments: subset names (default: pilot_all_dates hcc_visual_review). Submit from the repo folder, e.g. after
# the download:  qsub -hold_jid hope_cv_dl data_pipeline/jobs/cancerverse_timeline_pictures.sh
#$ -S /bin/bash
#$ -N hope_cv_pics
#$ -l tmem=4G
#$ -l h_rt=3:00:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
export OMP_NUM_THREADS=1
subsets=("$@"); [ ${#subsets[@]} -eq 0 ] && subsets=(pilot_all_dates hcc_visual_review)
lists=(); for s in "${subsets[@]}"; do lists+=("$HOPE_ROOT/outputs/cancerverse/subsets/$s.csv"); done
echo "Job $JOB_ID started $(date), subsets: ${subsets[*]}"
status=0
"$HOPE_PYTHON" -I -u data_pipeline/explore/cancerverse_timeline_pictures.py --lists "${lists[@]}" \
    --ct-dir "$HOPE_ROOT/data/CancerVerse/ct" --out "$HOPE_ROOT/outputs/cancerverse/review_pictures" || status=$?
echo "Job finished $(date), exit status $status"
exit $status
