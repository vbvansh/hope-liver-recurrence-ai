# Data pipeline plan

Goal: all five datasets in **one standard format**, with **fixed patient-level splits**, packaged so every baseline (nnU-Net v2, MedNeXt, U-Mamba, nnDetection, PLAN, multiphase models) uses exactly the same data.

Delivered in two releases so baseline work can start early:

- **Release v1:** MCT-LTDiag + PLC-CECT, 70/10/20 split, nnU-Net export → Akshat starts training.
- **Release v2:** external test sets WAW-TACE, LiTS, HCC-TACE-Seg added.

Status legend: ⬜ not started · 🔄 in progress · ✅ done

---

## Step 0 — Setup ⬜

**What:** folder, Python environment (Python 3.11), dataset paths in `configs/datasets.yaml`, a pinned checkout of LiverPreprocessing in `third_party/` (not committed; record its commit hash here).

**Compute:** laptop for writing and testing code on a few patients; a server or Lightning machine (T4 GPU, many CPU cores, several hundred GB disk) for full runs.

## Step 1 — Get the raw data and take inventory ⬜

**What:** download or receive all five datasets, then run an inventory script that lists, per dataset: patients, scans per patient, phases present, slice thickness, mask files.

**Why:** we must know exactly what we have before converting anything, and the numbers must match each dataset's publication.

| Dataset | Source | Status |
|---|---|---|
| MCT-LTDiag | from Yifan | waiting |
| PLC-CECT | Science Data Bank (online) | to download |
| WAW-TACE | Zenodo / Hugging Face; **all four CT zips** (the 164 four-phase patients span all of them) | have zip 1 of 4 |
| LiTS | online; 131 scans with labels | to download |
| HCC-TACE-Seg | TCIA | have it |

**Output:** `inventory_<dataset>.csv` per dataset.

## Step 2 — Fix the standard ⬜

**What:** one written definition of the output (see [STANDARD.md](STANDARD.md)): orientation, spacing, intensity type, phase order, mask labels, file names, `case.json` fields, missing-phase rule, crop rule.

**Why:** every converter writes the same thing, and every model reads the same thing. It follows LiverPreprocessing's format so his scripts keep working.

## Step 3 — Convert each dataset to the standard ⬜

| Sub-step | Dataset | How | Compute |
|---|---|---|---|
| 3a | MCT-LTDiag | LiverPreprocessing (phases already share one grid) | CPU |
| 3b | PLC-CECT | LiverPreprocessing, including geometry repair (stage 0) and phase registration | **GPU** for stage 0, CPU for registration |
| 3c | WAW-TACE | LiverPreprocessing (four-phase patients, registration to PVP) | CPU |
| 3d | LiTS | **new converter**: PVP channel only; split the label file into liver and tumour masks | CPU |
| 3e | HCC-TACE-Seg | **new converter**: pre-TACE visit only; split stacked passes; identify each scan's phase; register to PVP; move the expert mask from the scan it was drawn on into PVP space | CPU + **GPU** for phase identification |

**Why:** each dataset arrives with its own layout and problems (PLC-CECT has wrong headers, HCC-TACE-Seg has unlabelled and stacked phases). After this step they all look the same.

## Step 4 — Lesion labels and sizes ⬜

**What:** number every lesion (connected component), measure its size (equivalent spherical diameter, plus longest axial diameter), size bin (0–5, 5–10, 10–15, >15 mm; also ≤20 mm), lesion type per patient. LiverPreprocessing stage 2 does this for its datasets; extend it to LiTS and HCC-TACE-Seg.

**Output:** per-case lesion table used by the evaluation.

## Step 5 — Data check ⬜

**What:** a QC report per dataset: counts, phases present, slice thickness, lesion counts per size bin, failed or unreliable cases, duplicate patients, and a sample of very small lesions checked by eye (e.g. MCT-LTDiag 0–5 mm lesions on 5 mm slices may be fragments).

**Why:** problems found now are cheap; problems found after training are expensive. The report also becomes the paper's dataset table.

## Step 6 — Splits ⬜

**What:** extend LiverPreprocessing's `make_splits.py` (stratified protocol) to use **only MCT-LTDiag + PLC-CECT** for train/validation/internal test (70/10/20, patient-level, stratified by dataset, diagnosis and lesion size, fixed seed). WAW-TACE, LiTS and HCC-TACE-Seg are written as **external** lists only.

**Output:** split JSON/CSV files, committed to git.

## Step 7 — Leak-free test crops ⬜

**What:** training cases keep the crop "liver + tumour + 20 mm". Validation, internal test and external cases are cropped from an **automatic liver mask only** (TotalSegmentator), never from the tumour mask.

**Why:** using the tumour mask to crop test scans uses the answer to prepare the question; it can also hide false positives outside the box.

**Compute:** GPU (TotalSegmentator liver).

## Step 8 — Export for the baselines ⬜

**What:**
- nnU-Net v2 **single-phase** dataset (PVP channel) and **multiphase** dataset (channels `_0000`–`_0003` = NC, AP, PVP, DP; missing phase per the agreed rule), with `dataset.json` and a fixed `splits_final.json` so nnU-Net, MedNeXt and U-Mamba all use our split.
- nnDetection export from the same files.
- Internal and each external test set in separate folders.
- Lesion tables for the lesion-level evaluation (rule agreed with Akshat).

## Step 9 — Verification and handover ⬜

Automatic checks that must all pass:

1. LiverPreprocessing's own tests and `verify_unified_v2_output.py` pass.
2. No patient appears in more than one split.
3. No external patient appears in train or validation.
4. Re-running the split with the same seed gives identical files.
5. Every split contains lesions in every size bin.
6. Test and external crops were built without the tumour mask.
7. Channel order is NC, AP, PVP, DP; missing phases are flagged.
8. Visual QC: overlay images look right for a sample of every dataset.
9. Patient and lesion counts match the dataset publications.
10. nnU-Net's dataset integrity check passes on the export.

Then: README for users, QC report, release tag (`data-v1`, `data-v2`).

---

## Decisions needed from the team

| Decision | Proposal |
|---|---|
| Missing phases in multiphase models | Zero-filled channel + "phase present" flag; phase-dropout training possible |
| Crop for test and external cases | Automatic liver mask only, 20 mm margin |
| HCC-TACE-Seg scans used | Pre-TACE visit only (expert masks exist only there) |
| Lesion-level evaluation rule | Agree with Akshat: hit criterion, false-positive definition, size measure |
| Use of LiverPreprocessing code | Confirm with Yifan; credit in the paper; pin the commit |

## Timeline (target)

| Week | Work |
|---|---|
| 1 | Steps 0–2; downloads; inventory of local datasets |
| 2 | Steps 3a–3c (LiverPreprocessing runs); step 4 for these |
| 3 | Steps 5–6; nnU-Net export of MCT-LTDiag + PLC-CECT → **release v1** |
| 4 | Steps 3d–3e (LiTS, HCC-TACE-Seg) |
| 5 | Steps 7–9 for all; **release v2** |
