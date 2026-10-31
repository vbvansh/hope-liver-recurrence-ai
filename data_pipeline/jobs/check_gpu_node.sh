#!/bin/bash
# Test job: can a GPU node use the project storage, and what GPU, driver, OS and glibc does it have?
# (decides the PyTorch build for TotalSegmentator)
# Submit from the repo folder (the job runs there and logs to logs/):  qsub data_pipeline/jobs/check_gpu_node.sh
#$ -S /bin/bash
#$ -N hope_gpu_check
#$ -l gpu=true
#$ -l tmem=4G
#$ -l h_rt=0:10:00
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source data_pipeline/jobs/project_env.sh
echo "Job $JOB_ID started $(date)"
echo "--- project storage (the job runs inside it, so getting here means the node can see it)"
ls "$HOPE_ROOT"
df -h "$HOPE_ROOT" | tail -1
touch "$HOPE_ROOT/work/.write_test_$JOB_ID" && rm "$HOPE_ROOT/work/.write_test_$JOB_ID" && echo "write ok"
echo "--- GPU and driver (top-right of the table = highest CUDA version the driver supports)"
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "--- operating system / glibc"
cat /etc/redhat-release 2>/dev/null || grep PRETTY_NAME /etc/os-release
ldd --version | head -1
echo "--- containers"
which singularity apptainer 2>/dev/null || echo "no singularity/apptainer"
echo "--- internet from the GPU node"
curl -s -o /dev/null -w "huggingface.co %{http_code}\n" --max-time 20 https://huggingface.co
echo "Job finished $(date)"
