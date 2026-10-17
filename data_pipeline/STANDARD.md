# Output standard

Every processed patient, from every dataset, follows this format. Requirements come from MICCAI 2027 Plan v2 and the decisions recorded in [PLAN.md](PLAN.md).

## Folder layout

```
<output-root>/
├── cases/<DATASET>_<patient>/
│   ├── image_nc.nii.gz   image_ap.nii.gz   image_pvp.nii.gz   image_dp.nii.gz
│   ├── liver_nc.nii.gz   liver_ap.nii.gz   liver_pvp.nii.gz   liver_dp.nii.gz
│   ├── lesion_label.nii.gz
│   ├── lesion_instances.nii.gz
│   ├── lesion_free_nc.nii.gz   lesion_free_ap.nii.gz   lesion_free_pvp.nii.gz   lesion_free_dp.nii.gz
│   └── case.json
├── manifest.csv              one row per case
├── lesions.csv               one row per lesion
├── exclusion.csv             excluded cases and reasons
├── qc/
│   ├── registration_qc.csv   liver Dice per phase
│   ├── landmark_tre.csv      TRE per landmark and phase (20 random cases per dataset)
│   ├── landmarks/            landmark files
│   └── overlays/             visual QC images
├── splits/                   split files (also committed to git)
└── exports/                  nnU-Net v2 / nnDetection datasets
```

`<DATASET>` is one of `MCT-LTDiag`, `PLC-CECT`, `WAW-TACE`, `TCIA-CRLM`, `LiTS`. A phase that does not exist for a patient has no files.

## Images

| Property | Value |
|---|---|
| Orientation | LPS |
| Spacing | 1 × 1 × 1 mm, resampled once (native spacing recorded in `case.json`) |
| Interpolation | Linear for images (B-spline allowed); nearest neighbour for every mask |
| Values | int16 HU, clipped to [-200, 200] |
| Phases | NC, AP, PVP, DP on the PVP grid (PVP is the registration reference) |
| Crop | Box around the automatic PVP liver mask (TotalSegmentator) + 20 mm, identical for all phases and masks of a case; never built from a lesion mask |

## Masks

| File | Type | Values |
|---|---|---|
| `liver_<phase>.nii.gz` | uint8 | 0 / 1, TotalSegmentator liver on that phase (used for registration QC and lesion-free masks) |
| `lesion_label.nii.gz` | uint8 | 0 background · 1 malignant · 2 haemangioma · 3 ignore |
| `lesion_instances.nii.gz` | uint16 | 0 background · 1..N one id per lesion (3D connected component, multi-lesion cases reviewed by hand) |
| `lesion_free_<phase>.nii.gz` | uint8 | 0 / 1, `liver_<phase>` minus malignant lesions |

Class 1 (malignant) covers HCC, ICC, cHCC-CCA and metastasis. Class 2 is used only where haemangiomas are annotated (MCT-LTDiag). Class 3 covers cysts and any lesion type a dataset leaves unannotated; per-dataset rules are in `configs/label_maps.yaml`. Lesions keep the mask they were drawn on (e.g. WAW-TACE tumours stay on their own phase; recorded as `drawn_phase` in `lesions.csv`).

## `case.json`

| Field | Meaning |
|---|---|
| `dataset`, `patient_id`, `case_id` | identifiers |
| `role` | `development` (MCT-LTDiag, PLC-CECT) or `external` |
| `phases_present` | {nc, ap, pvp, dp} → true / false |
| `native_spacing_mm` | per phase, before resampling |
| `registration` | per phase: method, liver Dice, failure flag |
| `landmark_case` | true for the 20 random landmark cases per dataset |
| `crop_box_mm` | crop box in physical coordinates |
| `diagnosis` | patient-level tumour type(s) |
| `pipeline_commit` | git commit of this repository (and of LiverPreprocessing if used) |

## `manifest.csv`

One row per case: `case_id, dataset, role, diagnosis, phases_present, native_slice_mm, liver_dice_nc, liver_dice_ap, liver_dice_dp, registration_failed, landmark_case, tre_failed, n_malignant_lesions, max_diameter_bin, excluded`.

## `lesions.csv`

One row per lesion, all measurements on the **original (native) scan**:

`dataset, patient, case_id, lesion_id, class, drawn_phase, volume_ml, equivalent_diameter_mm, inplane_max_diameter_mm, size_stratum, native_slice_mm, multi_lesion_reviewed`

`size_stratum` uses the in-plane maximum diameter: `<10` (exploratory), `10-19`, `20-29`, `>=30` mm.

## `exclusion.csv`

`case_id, dataset, stage, reason` — every case left out of any split, with the step that excluded it.
