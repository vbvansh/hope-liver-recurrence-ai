# WAW-TACE — processing

Dataset description: [task README](../../README.md#waw-tace).

| | |
|---|---|
| Source | [Zenodo 12741586](https://zenodo.org/records/12741586) |
| Raw (cb) | `/media/cbtil3/WhiteSD/CTRaw/Abdomen/WAW-TACE/extracted` |
| Anchor organ | liver (CADS 5) |

## Pipeline — produces the dataset (`pipeline/`)

- `01_process.py` — NIfTI phases 0–3 → 1 mm RAS HU on the timepoint layout; rigid + body-Dice QC; NRRD masks warped along (replaces standardize)
- `02_cads.py` — CADS Task-551 organ masks → `_cads551/`
- `03_segvol.py` — SegVol text-prompted tumour masks → `_segvol/` (`--calibrated` uses [Combined/01](../Combined/))
- `04_segvol_fill_empty.py` — optional: re-run SegVol on empty masks listed in `--report`
- `05_build_longitudinal.py` — anchor-organ registration (ANTs SyNRA on signed-distance maps) + intensity matching → `CTProcess/`

## Report — text reports (`report/`)

- `01_report.py` — per-timepoint RECIST progression report (`lib/report/`, MedGemma + Qwen2.5-VL)

`lib/` holds this dataset's copies of the engines the steps run, and `lib/report/` its report generator. Figures and QC across all datasets: [Combined/](../Combined/).

## Run

```bash
bash run.sh                                   # pipeline steps 1 2 3 5 (raw data already downloaded)
bash run.sh --stage pipeline --steps "2 3"    # selected pipeline steps
bash run.sh --stage tables  -- --help         # tables/ scripts; arguments after -- pass through
bash run.sh --stage report  -- --patient <id> # report/ scripts
python3 <stage>/<NN_step>.py --help           # every step takes its own input/output paths
```

Path flags: `--raw`, `--processed`, `--process` (defaults `$CT_RAW`, `$CT_PROCESSED`, `$CT_PROCESS` under `/media/cbtil3/WhiteSD/`). Tools: [processing/README.md](../README.md#tools).
