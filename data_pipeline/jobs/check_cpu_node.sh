#!/bin/bash
# Test job: does a CPU-only job run, and what does a compute node offer?
# Submit from the repo:  mkdir -p ~/hope_data/logs && qsub data_pipeline/jobs/check_cpu_node.sh
#$ -S /bin/bash
#$ -N hope_cpu_check
#$ -l tmem=2G
#$ -l h_rt=0:10:00
#$ -j y
#$ -o $HOME/hope_data/logs/$JOB_NAME.$JOB_ID.log

echo "Job $JOB_ID started $(date)"
echo "--- operating system / glibc"
cat /etc/redhat-release 2>/dev/null || grep PRETTY_NAME /etc/os-release
ldd --version | head -1
echo "--- CPUs and memory"
nproc
free -g | head -2
echo "--- local scratch"
echo "TMPDIR=${TMPDIR:-unset}"
df -h "${TMPDIR:-/tmp}" | tail -1
echo "--- internet from the compute node"
curl -s -o /dev/null -w "zenodo.org %{http_code}\n" --max-time 20 https://zenodo.org
curl -s -o /dev/null -w "dataverse.harvard.edu %{http_code}\n" --max-time 20 https://dataverse.harvard.edu
echo "--- project Python environment"
"$HOME/envs/hope/bin/python" -c "import numpy, scipy, pandas, yaml, nibabel, SimpleITK; print('env ok: numpy', numpy.__version__, '| SimpleITK', SimpleITK.__version__)"
echo "Job finished $(date)"
