# Data pipeline plan (aligned with MICCAI 2027 Plan v2)

Goal: MCT-LTDiag and PLC-CECT harmonised into one standard for training, validation and internal testing, plus three external test sets processed the same way, so every detector (Akshat's baselines and our method) uses identical data.

Source of the requirements: *Liver Lesion Detection on Multi-phase CT — MICCAI 2027 Plan v2* (Yifan, 14 Oct 2026), section 2, plus the team decisions below.

Status legend: ⬜ not started · 🔄 in progress · ✅ done

---

## Datasets

| Dataset | Role | Phases | Patients |
|---|---|---|---|
| MCT-LTDiag | train / validation / internal test | NC, AP, PVP, DP (already registered to PVP by the authors) | 517 |
| PLC-CECT | train / validation / internal test | NC, AP, PVP, DP (not registered) | 361 (278 cancer + 83 controls) |
| WAW-TACE | external test | any subset of NC, AP, PVP, DP | 233 |
| TCIA Colorectal-Liver-Metastases | external test | PVP only | 197 |
| LiTS | external test | PVP only | 131 |

Dropped: HCC-TACE-Seg. On hold: TriALS.

## Decisions taken (Oct 2026)

| Topic | Decision |
|---|---|
| Positive class | Malignant lesions (HCC, ICC, cHCC-CCA, metastasis). Haemangioma is its own class where annotated (MCT-LTDiag), ignore elsewhere; cysts are ignore |
| Spacing | 1 × 1 × 1 mm, isotropic |
| Interpolation | Images: linear (B-spline allowed); masks: nearest neighbour |
| Models | Use our spacing as-is (no further resampling) |
| Intensity | Clip MCT-LTDiag to [-200, 200] HU, matching PLC-CECT (released pre-clipped) |
| Crop | One liver-box crop (+ 20 mm) from the automatic PVP liver mask, for every case; never from the tumour mask |
| Lesion size | Measured on the **original** scan: in-plane maximum diameter on the annotated phase, plus equivalent diameter and volume |
| Size strata | < 10 mm (exploratory), 10–19, 20–29, ≥ 30 mm |
| Registration QC | Liver Dice per phase (TotalSegmentator liver on every phase); failure when Dice < 0.9 or TRE > 5 mm |
| Landmarks | 5 fixed vessel landmarks per patient; 20 random patients per dataset (fixed seed); on every phase where they can be placed, liver Dice only where they cannot (e.g. NC) |
| Lesion-free masks | Remove malignant lesions only; one mask per phase |
| Split | Patient-level 70 / 10 / 20, fixed seed, stratified by dataset × tumour type × lesion count (1 / ≥ 2) × max-diameter bin; PLC-CECT controls spread proportionally |

## Open items (ask the team; defaults in brackets)

1. Registration for PLC-CECT: re-implement the MCT-LTDiag authors' hierarchical B-spline, or run LiverPreprocessing's rigid → affine → B-spline? [start from LiverPreprocessing's registration code]
2. The 5 named landmarks, who places them, which tool, and whether points are placed on native phases or registered images. [portal and hepatic vein branch points; 3D Slicer; placed on native phases and mapped with the saved transform]
3. What happens to a failed case (Dice < 0.9 or TRE > 5 mm): exclude, flag, or PVP only. [flag in `manifest.csv`, list in `exclusion.csv`, decide per experiment]
4. PLC-CECT +40 HU offset recommended by LiverPreprocessing: still apply before clipping? [ask Yifan]
5. Dilation margin around lesions in lesion-free masks. [none]

---

## Steps

### Step 0 — Setup ⬜
Folder, Python 3.11 environment, dataset paths in `configs/datasets.yaml`, LiverPreprocessing checked out in `third_party/` at a pinned commit (recorded here). Develop on the laptop with a few patients; full runs on a GPU machine with several hundred GB of disk.

### Step 1 — Raw data and inventory ⬜
Download or receive all five datasets; an inventory script lists, per dataset, patients, phases present, native spacing and slice thickness, mask files, and checks counts against each publication. Output: `inventory_<dataset>.csv`.

| Dataset | Source | Status |
|---|---|---|
| MCT-LTDiag | from Yifan (Harvard Dataverse, CC0) | waiting |
| PLC-CECT | Science Data Bank, CC BY 4.0 | to download |
| WAW-TACE | Zenodo 12741586, all four CT zips + tumour masks, CC BY 4.0 | have 1 of 4 zips |
| TCIA Colorectal-Liver-Metastases | TCIA, CC BY 4.0 | to download |
| LiTS | LiTS challenge, 131 labelled scans | to download |

### Step 2 — Label maps and annotation protocols ⬜
For each dataset: how its labels map into {malignant, haemangioma, ignore}, and whether sub-centimetre lesions were annotated (MCT-LTDiag excludes tumours < 1 cm; LiTS omitted small uncertain structures). Stored in `configs/label_maps.yaml`. This decides which unmatched small detections are "unverifiable" in the evaluation.

### Step 3 — Convert MCT-LTDiag ⬜
Read the four phases (already registered to PVP), keep native geometry for lesion measurement, map labels, clip to [-200, 200] HU.

### Step 4 — Convert and register PLC-CECT ⬜
Recover the true geometry (LiverPreprocessing stage 0, GPU), then register NC, AP and DP to PVP with hierarchical B-spline (open item 1).

### Step 5 — Per-phase liver masks and registration QC ⬜
TotalSegmentator liver on every phase (GPU); liver Dice of each registered phase against PVP; failure flag below 0.9. Output: `qc/registration_qc.csv`.

### Step 6 — Landmark check ⬜
20 random patients per dataset (fixed seed); 5 fixed vessel landmarks on every phase where visible; TRE per phase; failure when > 5 mm. Output: `qc/landmarks/` and `qc/landmark_tre.csv`. This is the slowest manual step; start it as soon as Step 4 has registered cases.

### Step 7 — Lesions on the original scan ⬜
Number each lesion (3D connected components; manual review of multi-lesion cases), record the phase it was drawn on, measure volume, equivalent diameter and in-plane maximum diameter on the native grid, size stratum, class, native slice thickness. Output: `lesions.csv`.

### Step 8 — Resample, crop, clip ⬜
Resample every phase and mask once to 1 × 1 × 1 mm (linear for images, nearest for masks); crop to the automatic PVP liver box + 20 mm; clip to [-200, 200] HU; write the standard case folder (see [STANDARD.md](STANDARD.md)).

### Step 9 — Lesion-free masks ⬜
Per phase: that phase's liver mask minus malignant lesions. Output: `lesion_free_<phase>.nii.gz` per case.

### Step 10 — Data check and manifest ⬜
`manifest.csv` (phases present, native thickness, QC flags per case), `exclusion.csv` (excluded cases with reasons), QC report with counts per dataset, class and size stratum, overlay images for a sample of every dataset. The report feeds the paper's dataset table.

### Step 11 — Splits ⬜
MCT-LTDiag + PLC-CECT only, patient-level 70 / 10 / 20, fixed seed, stratified as decided; external lists kept separate. Split files committed to git.

### Step 12 — Exports for the baselines ⬜
nnU-Net v2 single-phase (PVP) and multi-phase (NC, AP, PVP, DP as `_0000`–`_0003`; missing phase zero-filled) datasets with the ignore class as nnU-Net's ignore label, a fixed `splits_final.json`, and an nnDetection export; internal and each external test set in separate folders.

### Step 13 — External test sets ⬜
- **WAW-TACE:** all 233 patients with whatever phases exist; tumour masks kept on the phase they were drawn on (scored there with the centroid rule); haemangiomas and benign lesions are unannotated, so FP/study is an upper bound.
- **TCIA Colorectal-Liver-Metastases:** PVP only, same preprocessing as LiTS; keep hepatic and portal vein masks for the vessel-vs-parenchyma false-positive analysis.
- **LiTS:** PVP only; < 10 mm lesions only counted on volumes with native slice thickness ≤ 2 mm.

### Step 14 — Verification and handover ⬜
Automatic checks that must all pass:
1. No patient in more than one split; no external patient in train or validation.
2. Re-running the split with the same seed gives identical files.
3. Every split contains each tumour type and size stratum.
4. Crops never used a tumour mask.
5. Channel order NC, AP, PVP, DP; missing phases flagged.
6. All images 1 × 1 × 1 mm, LPS, clipped to [-200, 200] HU.
7. Lesion counts and patient counts match the inventory and the publications.
8. `lesions.csv` sizes come from the native grid.
9. Visual QC overlays look right for a sample of every dataset.
10. nnU-Net's dataset integrity check passes on the export.

Then a README for users, the QC report, and release tags.

---

## Releases and timeline (target)

| Week | Work |
|---|---|
| 1 (19–25 Oct) | Steps 0–2: setup, downloads, inventory, label maps |
| 2 (26 Oct – 1 Nov) | Steps 3–4: MCT-LTDiag conversion; PLC-CECT geometry and registration |
| 3 (2–8 Nov) | Steps 5–6: per-phase liver masks, registration QC, landmarks |
| 4 (9–15 Nov) | Steps 7–12 for MCT-LTDiag + PLC-CECT → **release v1** (training data for Akshat) |
| 5–6 (16–29 Nov) | Step 13 external sets, Step 14 checks → **release v2** |
| December | Buffer for fixes requested by the team |
