# CLAUDE.md — HOPE liver project

HOPE (Barts Cancer Institute / QMUL): earlier detection of liver cancer recurrence from longitudinal contrast CT + AFP. Current deliverable: MICCAI 2027 paper on detecting malignant liver lesions on multi-phase CT; requirements live in `Liver Lesion Detection on Multi-phase CT — MICCAI 2027 Plan v2.md`.

## Repository layout

| Path | Purpose |
|---|---|
| `data_pipeline/` | MICCAI data pipeline. `PLAN.md` (steps, decisions, open items), `STANDARD.md` (output format), `configs/datasets.yaml` (all settings) are the source of truth; update them when the team changes a decision |
| `processing/` | Team registration pipeline (per dataset: `run.sh`, `pipeline/NN_*.py`, `lib/`). `processing/Combined/` is not in this repo |
| `experiments/registration/` | Registration results, reports, concerns; `simpleitk_baseline/` is a comparison baseline only |
| `docs/`, `literature_review/` | Project overview, dataset statistics, papers |
| `Datasets/` | Local raw data, gitignored (`/Datasets/`, anchored — an unanchored rule also hid `docs/datasets/` on Windows) |

## Conventions

- **Git:** small commits, one-line message, only the files you changed (never `git add .` — the user keeps unrelated uncommitted work in the tree). End messages with the Co-Authored-By trailer. The user runs `git push origin main`.
- **Never commit imaging data** (`*.dcm`, `*.nii*` are ignored); commit code, configs, split files and small result tables only. Check dataset licences (e.g. LiTS is non-commercial) and cite every dataset.
- `.gitattributes` forces LF for `*.sh` and `*.py`; keep it, scripts run on Linux.
- Explanations for the user: simple language, numbered steps, tables.

## Data standard (MICCAI, Plan v2)

- Development: MCT-LTDiag + PLC-CECT (patient-level 70/10/20, seed 0, stratified by dataset × tumour type × lesion count × diameter bin). External, never trained or tuned on: WAW-TACE, TCIA Colorectal-Liver-Metastases, LiTS. HCC-TACE-Seg is dropped.
- Positive class = malignant (HCC, ICC, cHCC-CCA, metastasis); haemangioma own class where annotated, ignore elsewhere; cysts ignore. Label codes: 0 background, 1 malignant, 2 haemangioma, 3 ignore.
- Images: LPS, 1 × 1 × 1 mm, linear interpolation (masks nearest), int16 HU clipped to [-200, 200], NC/AP/PVP/DP registered to PVP. Models use this spacing as-is.
- Crop = automatic (TotalSegmentator) PVP liver box + 20 mm for every case.
- Lesion sizes are measured on the native scan, never after resampling; strata by in-plane max diameter `<10`, `10–19`, `20–29`, `>=30` mm.

## Never do

- Never replace the team's pipeline with a self-built one; propose options and let the user decide. Changes to `processing/` code need the user's approval.
- Never build test/external crops from tumour masks (LiverPreprocessing's liver ∪ tumour crop leaks the answer).
- Never choose thresholds, post-processing or operating points on test or external data — validation only.
- Never split by scan or phase; all of a patient's data stays in one split.
- Never run two copies of a batch job that share a work folder.
- Never trust the overlap (Dice) of organs that also drove the registration as the only quality measure; report an independent one (gradient NCC, landmark TRE).

## Gotchas

- **HCC-TACE-Seg stacked passes:** some "LIVER 3 PHASE" DICOM folders contain two contrast passes with identical slice positions, separated only by `AcquisitionNumber`; reading the folder as one series duplicates the anatomy. Phases are unlabelled; "PRE" series have no dye. `processing/HCC-TACE-Seg/lib/standardize.py` splits passes and matches them by aorta / portal-vein HU (CADS labels 7 and 9).
- **DICOM-SEG slice lookup:** build and look up slice positions with the same `np.round` (+ `np.searchsorted`); mixing Python `round()` raises `KeyError` (e.g. `-223.91`).
- **Manifests downloaded on Windows** use `\` in `File Location`; convert to `/` before joining paths on Linux.
- **PLC-CECT headers are wrong** (geometry must be recovered, LiverPreprocessing stage 0); HU are released pre-clipped to [-200, 200].
- **MCT-LTDiag** excludes tumours < 1 cm and has 5 mm slices: tiny connected components in its masks are likely fragments, not small lesions.
- **Team pipeline `longitudinal_pipeline.py`:** `wdice_deformed` is logged even when `REVERT_RAW` / `KEPT_RIGID` discard the bending (delivered value = rigid); results JSON is written only at the end of a run (set `STD_SHARD_TAG` / `LONG_SHARD_TAG` per run so reruns don't overwrite); patients are processed in text order (`…69, 7, 72…`).
- **WAW-TACE** "timepoints" in the team pipeline are contrast phases of one visit, not follow-ups.

## Environment (Linux GPU machine)

- Lightning AI terminal is zsh: run `bash` first (zsh does not split `$VAR` lists; heredocs failed).
- Set `MPLBACKEND=Agg` for every subprocess (notebook backends crash the tool environments).
- Separate environments, created with `uv`:
  - CADS: Python 3.11, `torch==2.5.1` `torchvision==0.20.1` (cu124), `pip install -e CADS` (TPTBox pins scikit-image 0.22 / numpy < 2; no wheels for Python ≥ 3.13).
  - SegVol: Python 3.11, `torch==2.6.0` (transformers refuses `.bin` weights with older torch), `transformers>=4.40,<5`, `monai==1.3.2` (SegVol passes `pos_embed`, removed in MONAI 1.4), `numpy<2`.
- Team pipeline env vars: `CT_RAW`, `CT_PROCESSED`, `CT_PROCESS`, `PYTHON`, `CADS_PYTHON`, `CADS_REPO`, `SEGVOL_PYTHON`, `CADS_WEIGHTS_PATH`, `HF_HOME`, `CT_GALLERY`. HCC-TACE-Seg expects the raw folder name `"HCC-TACE-Seg "` (trailing space) and its stages: `bash run.sh --stage tables -- --manifest <dir> --out-dir "$CT_RAW/HCC-TACE-Seg "`, then `--steps 1` (only step taking `-- --patients …`), `2`, `3`, `4`, `6` run separately. WAW-TACE default steps `1 2 3 5`; step 1 processes every patient folder.
- Disk/RAM: full runs need several hundred GB and ≥ 16 GB RAM; the QMUL teaching hub (≈5 GB home) and the laptop (8 GB RAM) are for development only.

## Useful commands

- TCIA public download (no login): `https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID=<uid>`
- Contrast phase prediction: `totalseg_get_phase -i ct.nii.gz -o contrast_phase.json`
- nnU-Net export check: `nnUNetv2_plan_and_preprocess -d <id> --verify_dataset_integrity`
