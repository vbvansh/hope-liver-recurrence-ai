# Project folders, caches and Python for job scripts and terminals. Use from the repo folder:
#   source data_pipeline/jobs/project_env.sh
# Layout: the repo sits inside the project storage folder, next to data/, processed/, outputs/,
# checkpoints/, envs/, cache/ and work/; job logs go to logs/ inside the repo (gitignored).

HOPE_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HOPE_ROOT=$(dirname "$HOPE_REPO")
HOPE_PYTHON="$HOPE_ROOT/envs/hope/bin/python"
export HOPE_REPO HOPE_ROOT HOPE_PYTHON

# downloads and model weights stay in project storage, never in the 50 GB home
export PIP_CACHE_DIR="$HOPE_ROOT/cache/pip"
export CONDA_PKGS_DIRS="$HOPE_ROOT/cache/conda_pkgs"
export HF_HOME="$HOPE_ROOT/cache/huggingface"
export TORCH_HOME="$HOPE_ROOT/cache/torch"
export TOTALSEG_HOME_DIR="$HOPE_ROOT/cache/totalsegmentator"
export XDG_CACHE_HOME="$HOPE_ROOT/cache/xdg"
export MPLBACKEND=Agg
