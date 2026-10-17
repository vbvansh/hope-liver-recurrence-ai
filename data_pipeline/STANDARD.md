# Output standard

Every processed patient, from every dataset, follows this format. It matches LiverPreprocessing's `unified_v2` layout, so its scripts (`make_splits.py`, `evaluate_by_lesion_size.py`, `verify_unified_v2_output.py`) keep working, with additions marked **(new)**.

## Folder layout

```
<output-root>/
├── cases/<DATASET>_<patient>/
│   ├── image_nc.nii.gz      image_ap.nii.gz      image_pvp.nii.gz      image_dp.nii.gz
│   ├── liver_mask.nii.gz    tumor_mask.nii.gz
│   ├── lesion_instances.nii.gz              (new) each lesion numbered 1..N
│   └── case.json
├── dataset_manifest.csv                     one row per case
├── lesions.csv                              (new) one row per lesion, all datasets
├── splits/                                  split files (also committed to git)
└── exports/                                 nnU-Net v2 / nnDetection datasets
```

`<DATASET>` is one of `MCT-LTDiag`, `PLC-CECT`, `WAW-TACE`, `LiTS`, `HCC-TACE-Seg`.

## Images

| Property | Value |
|---|---|
| Orientation | LPS |
| Spacing | 1 × 1 × 1 mm (the native spacing is recorded in `case.json`; resampling adds no detail) |
| Values | int16 Hounsfield units, unnormalised. Dataset-specific corrections (e.g. PLC-CECT +40 HU) are recorded, not applied |
| Phases | NC, AP, PVP, DP, all on the PVP grid (PVP is the registration reference) |
| Missing phase | file not written; `phases_present` is false for it. Multiphase models fill it with zeros (rule to be confirmed by the team) |
| Failed phase (registration liver Dice < 0.80) | file written; `usable_<phase>` is false |

## Masks

| File | Type | Values |
|---|---|---|
| `liver_mask.nii.gz` | uint8 | 0 background, 1 liver |
| `tumor_mask.nii.gz` | uint8 | 0 background, 1 lesion (all lesion types) |
| `lesion_instances.nii.gz` (new) | uint16 | 0 background, 1..N one id per 26-connected lesion |

The lesion type (HCC, ICC, metastasis, haemangioma, …) is stored per patient in `case.json`, not in the mask.

## Crop

| Case role | Crop box |
|---|---|
| Train | liver ∪ tumour, + 20 mm |
| Validation, internal test, external test | **automatic** liver mask only (TotalSegmentator), + 20 mm. Never uses the tumour mask (new) |

`case.json` records which rule was used (`crop_source`).

## `case.json` additions (new)

| Field | Meaning |
|---|---|
| `dataset`, `patient_id`, `case_id` | identifiers |
| `phases_present` | {nc, ap, pvp, dp} → true/false |
| `phase_source` | how each phase was identified: `dataset_label` or `detected` (HCC-TACE-Seg) |
| `crop_source` | `liver_and_tumour` or `auto_liver` |
| `liver_mask_source` | `expert`, `dataset`, or `totalsegmentator` |
| `role` | `development` (MCT-LTDiag, PLC-CECT) or `external` |
| `pipeline_commit` | git commit of this repository and of LiverPreprocessing used |

## `lesions.csv` columns (new)

`case_id, lesion_id, voxels, volume_ml, esd_mm, longest_axial_mm, size_bin, centroid_x_mm, centroid_y_mm, centroid_z_mm, native_slice_mm, lesion_type`
