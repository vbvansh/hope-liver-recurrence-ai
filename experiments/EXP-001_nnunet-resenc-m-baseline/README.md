# EXP-001: nnU-Net ResEnc M baseline (4 phases, official split)

| Field | Value |
|---|---|
| ID | EXP-001 |
| Owner | data-lead |
| Status | planned <!-- planned / running / done / failed / abandoned --> |
| Started / finished | 2026-11-27 / – |
| Parent experiment | none (first experiment; the reference for every later gain) |
| Data | Release v1.1 (843 patients, split 588 / 85 / 170), 1 × 1 × 1 mm grid as released (no resampling) |
| Phases used | NC, AP, PVP, DP as 4 input channels; a phase missing in the release (3 train/val, 2 test cases) = all-zero image (team default, PLAN.md open item 6) |
| Split evaluated | validation (85 cases); the internal test set (170) stays closed |
| Code commit | <!-- fill in when the training job starts: git rev-parse --short HEAD --> |
| Job IDs | <!-- smoke test, preprocessing, training, scoring --> |

## 1. Question and hypothesis

How well does a properly tuned, standard nnU-Net find liver lesions on our release, especially small ones (10–19 mm)? Reviewers expect this baseline in every paper, and every later method is compared with it. No hypothesis beyond "a strong, reproducible reference"; The team's detector report (21 Nov) gives nnU-Net test FROC mean < 20 mm 0.650 on its own grid and settings, so a similar order of magnitude is expected.

## 2. What changes compared with the parent

None: first experiment. Settings follow nnU-Net's current recommendation (residual-encoder "M" plans), chosen by the data lead on 27 Nov 2026.

## 3. Setup

| Item | Value |
|---|---|
| Model / architecture | nnU-Net v2.8.1, `nnUNetPlannerResEncM` → plans `nnUNetResEncUNetMPlans`, configuration `3d_fullres`, trainer `nnUNetTrainer` (defaults) |
| Input (phases, patch size, grid) | 4 channels (NC, AP, PVP, DP, each "CT" normalisation); patch size planned by nnU-Net (see plans file after preprocessing); release 1 mm grid |
| Labels | 0 background, 1 malignant, 2 haemangioma, 3 = nnU-Net ignore label (not trained on) |
| Loss | nnU-Net default (Dice + cross entropy, deep supervision) |
| Sampling / augmentation | nnU-Net default |
| Training | nnU-Net default: 1,000 epochs × 250 iterations; fold 0 = the official split (`splits_final.json`: train 588, val 85) |
| Seeds | nnU-Net default (single run) |
| Hardware and time | 1 GPU (A6000, 48 GB), host RAM 48 GB, 8 data-augmentation workers, `torch.compile` off (old compiler toolchain on the cluster); ~1.5–3 days |
| Data on disk | raw = links to the release (`processed/nnunet/nnUNet_raw/Dataset501_HOPELiverV11`, made by `data_pipeline/export/nnunet_export.py`); preprocessed ≈ 34–40 GB (`processed/nnunet/nnUNet_preprocessed`) |

## 4. How to run

```bash
cd <project path>/hope-liver-recurrence-ai
qsub data_pipeline/jobs/nnunet_smoke_test.sh                                     # 0: 4-case test, ~15-30 min (GPU)
qsub -hold_jid hope_nnunet_smoke data_pipeline/jobs/nnunet_preprocess.sh          # 1: preprocessing, ~1-2 h (CPU)
qsub -hold_jid hope_nnunet_pre experiments/EXP-001_nnunet-resenc-m-baseline/train.sh   # 2: training, ~1.5-3 days (GPU)
qsub -hold_jid hope_exp001_train experiments/EXP-001_nnunet-resenc-m-baseline/score.sh # 3: scoring, ~10-30 min (CPU)
```

Each job checks that the previous one succeeded (`SMOKE_DONE`, `PREPROCESS_DONE`, `TRAIN_DONE`) and stops at once otherwise. Training is resumable: resubmit `train.sh` and it continues from the last checkpoint (every 50 epochs).

## 5. Evaluation

| Item | Value |
|---|---|
| **Positive class (fixed team rule, 26 Nov 2026)** | **Lesion level: every lesion numbered in `lesion_instances.nii.gz`, i.e. malignant (class 1) and haemangioma (class 2), as released. Ignore regions (class 3) are neutral: a detection on them is neither a hit nor a false alarm. Patient level: malignant diagnosis vs others (haemangioma patients and controls are negatives).** |
| Evaluation code and version | HOPE shared scorer `evaluation/scorer.py` 1.0 (2026-11-27); candidates from nnU-Net probabilities with `evaluation/candidates_from_nnunet.py` (lesion probability = malignant + haemangioma; components at 0.5 / 0.2 / 0.05 / 0.01 / 0.002, ≥ 15 voxels, non-overlapping; score = highest probability) |
| Lesion matching rule (fixed team rule, 26 Nov 2026) | A candidate hits a lesion when their 3D masks have IoU ≥ 0.1, matched one-to-one (extra candidates on a found lesion are false positives) |
| FROC points | 1/8, 1/4, 1/2, 1, 2, 4, 8 false positives per case |
| Size strata | < 20 mm (< 10 mm merged, Plan v2 rule), 20–29 mm, >= 30 mm |
| Ignore regions | class 3 never counted as true or false positive |
| Plan v2 "unverifiable" rule | Reported both ways (main: off) |

## 6. Results

<!-- Filled in by the scoring job: results/summary.md, metrics.csv, froc.csv. Copy the main numbers into ../tracker.csv. -->

| Metric | Value |
|---|---|
| FROC mean, lesions < 20 mm | |
| Sensitivity < 20 mm at 0.5 / 1 / 2 FP per case | |
| FROC mean, all lesions | |
| AP, all lesions | |
| Patient AUROC | |

## 7. What we learned

<!-- One paragraph: keep / drop / change, and why. -->

## 8. Next step

<!-- e.g. compare with the team's earlier nnU-Net numbers; PVP-only run; the small-lesion method. -->

## 9. Files

Raw links and preprocessed data: `processed/nnunet/` (shared by all nnU-Net experiments on release v1.1). Model, logs, validation predictions: `checkpoints/EXP-001/nnUNet_results/Dataset501_HOPELiverV11/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/`. Candidates and full scorer output (with case ids): `outputs/experiments/EXP-001/`. Small result tables: `results/` in this folder.

---
Before committing this card: no patient IDs, no private paths, user names or host names, no images or per-case tables.
