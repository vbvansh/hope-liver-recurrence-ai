#!/usr/bin/env bash
# One-time setup on a new machine. Creates a venv and installs deps.
#   bash setup.sh             # core + model backends (transformers/torch/qwen)
#   CORE_ONLY=1 bash setup.sh # only the JSON/measurement deps (no torch)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

if [ "${CORE_ONLY:-0}" = "1" ]; then
  pip install numpy nibabel Pillow streamlit pandas
else
  pip install -r requirements.txt
fi

echo
echo "done. activate with:  source .venv/bin/activate"
echo "then point at your data:  export PROC_ROOT=/media/cbtil3/WhiteSD/CTProcessed"
echo "and the report output:    export REPORTS=/media/cbtil3/WhiteSD/CTProcessed-Reports"
