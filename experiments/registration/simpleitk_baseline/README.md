# SimpleITK baseline registration

A lightweight, CPU-only registration pipeline. It is **not** the team pipeline in `processing/`: it was written so registration experiments could run on the QMUL JupyterHub, whose home disk is only 1 GB.

| | This baseline | Team pipeline (`processing/`) |
|---|---|---|
| Masks | radiologist masks (before-TACE scan only) | CADS organs + SegVol tumours on every scan |
| Alignment driven by | image intensities (Mattes mutual information) | organ shapes (ANTs SyN on signed-distance maps) |
| Intensity matching | measured only (`liver_shift_HU`) | applied, organ-anchored |
| Needs | CPU, ~150 MB of packages, ~1 patient on disk | GPU, PyTorch, model weights, tens of GB |

## Files

- `01_registration_basics.ipynb`: one patient (HCC_002), explained step by step.
- `run_batch_registration.py`: all eligible HCC-TACE-Seg patients. Downloads one patient from TCIA, registers it, saves metrics and a QC figure, deletes the scans. Resumable; only one copy can run at a time (`run.lock`).
- `results/`: outputs copied back from the hub.

## Run on the hub

```bash
cd ~/Vansh_HA26001            # holds run_batch_registration.py and metadata.csv
python run_batch_registration.py --plan
nohup python run_batch_registration.py > batch_log.txt 2>&1 &
python run_batch_registration.py --summary
```
