# HCC-TACE-Seg — processing

Dataset description: [task README](../../README.md#hcc-tace-seg).

| | |
|---|---|
| Source | [TCIA HCC-TACE-Seg](https://www.cancerimagingarchive.net/collection/hcc-tace-seg/) |
| Raw (cb) | `/media/cbtil3/WhiteSD/CTRaw/Abdomen/HCC-TACE-Seg ` |
| Anchor organ | liver (CADS 5) |

## Tables — metadata and clinical (`tables/`)

- `01_tcia_metadata.py` — TCIA manifest → `<ds>_metadata_series.csv`

## Pipeline — produces the dataset (`pipeline/`)

- `01_standardize.py` — DICOM → 1 mm RAS float32 HU; earliest study = baseline; rigid to baseline
- `02_seg_to_mask.py` — DICOM-SEG (liver, mass, portal vein, aorta) → NIfTI labels on the standardized baseline grid
- `03_cads.py` — CADS Task-551 organ masks → `_cads551/`
- `04_segvol.py` — SegVol text-prompted tumour masks → `_segvol/` (`--calibrated` uses [Combined/01](../Combined/))
- `05_segvol_fill_empty.py` — optional: re-run SegVol on empty masks listed in `--report`
- `06_build_longitudinal.py` — anchor-organ registration (ANTs SyNRA on signed-distance maps) + intensity matching → `CTProcess/`

## Report — text reports (`report/`)

- `01_report.py` — per-timepoint RECIST progression report (`lib/report/`, MedGemma + Qwen2.5-VL)

`lib/` holds this dataset's copies of the engines the steps run, and `lib/report/` its report generator. Figures and QC across all datasets: [Combined/](../Combined/).

## Run

```bash
bash run.sh                                   # pipeline steps 1 2 3 4 6 (raw data already downloaded)
bash run.sh --stage pipeline --steps "2 3"    # selected pipeline steps
bash run.sh --stage tables  -- --help         # tables/ scripts; arguments after -- pass through
bash run.sh --stage report  -- --patient <id> # report/ scripts
python3 <stage>/<NN_step>.py --help           # every step takes its own input/output paths
```

Path flags: `--raw`, `--processed`, `--process` (defaults `$CT_RAW`, `$CT_PROCESSED`, `$CT_PROCESS` under `/media/cbtil3/WhiteSD/`). Tools: [processing/README.md](../README.md#tools).
