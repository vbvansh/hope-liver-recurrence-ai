#!/bin/bash
# Test job: does a CPU-only job run, can the node use the project storage, and what does the node offer?
# Submit from the repo folder (the job runs there and logs to logs/):  qsub data_pipeline/jobs/check_cpu_node.sh
#$ -S /bin/bash
#$ -N hope_cpu_check
#$ -l tmem=2G
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
echo "--- operating system / glibc"
cat /etc/redhat-release 2>/dev/null || grep PRETTY_NAME /etc/os-release
ldd --version | head -1
echo "--- CPUs and memory"
nproc
free -g | head -2
echo "--- local scratch"
echo "TMPDIR=${TMPDIR:-unset}"
df -h "${TMPDIR:-/tmp}" | tail -1
echo "--- internet from the compute node (download sources, in download order)"
curl -s -o /dev/null -w "TCIA NBIA API %{http_code}\n" --max-time 20 "https://services.cancerimagingarchive.net/nbia-api/services/v1/getCollectionValues"
curl -s -o /dev/null -I -w "LiTS MSD mirror %{http_code}\n" --max-time 20 "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar"
curl -s -o /dev/null -w "MCT-LTDiag Dataverse API %{http_code}\n" --max-time 20 "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/S3RW15"
curl -s -L -o /dev/null -w "PLC-CECT Science Data Bank %{http_code}\n" --max-time 20 "https://doi.org/10.57760/sciencedb.12207"
curl -s -o /dev/null -w "WAW-TACE Zenodo %{http_code}\n" --max-time 20 "https://zenodo.org/records/12741586"
echo "--- project Python environment"
"$HOPE_PYTHON" -c "import numpy, scipy, pandas, yaml, nibabel, SimpleITK; print('env ok: numpy', numpy.__version__, '| SimpleITK', SimpleITK.__version__)"
echo "Job finished $(date)"
