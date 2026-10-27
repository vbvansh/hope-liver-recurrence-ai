#!/bin/bash
# Test job: what GPU, driver, OS and glibc does a GPU node have? (decides the PyTorch build for TotalSegmentator)
# Submit from the repo:  mkdir -p ~/hope_data/logs && qsub data_pipeline/jobs/check_gpu_node.sh
#$ -S /bin/bash
#$ -N hope_gpu_check
#$ -l gpu=true
#$ -l tmem=4G
#$ -l h_rt=0:10:00
#$ -j y
#$ -o $HOME/hope_data/logs/$JOB_NAME.$JOB_ID.log

echo "Job $JOB_ID started $(date)"
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
