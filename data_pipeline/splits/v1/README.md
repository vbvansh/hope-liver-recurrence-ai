# Release v1 split (MCT-LTDiag + PLC-CECT)

Patient-level train / validation / test split of the development data of release v1 (878 patients).

| File | Content |
|---|---|
| `train.txt`, `val.txt`, `test.txt` | One case id per line (`<DATASET>_<patient>`, the dataset's own public patient id): 614 / 88 / 176 |
| `splits.csv` | `case_id, dataset, diagnosis, stratum_level, stratum, split` |
| `split_summary.md` | Counts per split: tumour types, lesion counts, size bins, lesions per class and size |

Made by `data_pipeline/steps/split.py` (seed 0) from the release's `manifest.csv`: 70 / 10 / 20, stratified by dataset × tumour type × number of lesions (0 / 1 / ≥ 2) × size bin of the largest lesion, small strata merged into coarser ones. Every tumour type of both datasets is in every split; rerunning gives identical files. External test sets (WAW-TACE, TCIA Colorectal-Liver-Metastases, LiTS) are never part of the split.

Data: MCT-LTDiag (Harvard Dataverse doi:10.7910/DVN/S3RW15, CC0) and PLC-CECT (Science Data Bank doi:10.57760/sciencedb.12207, CC BY 4.0); please cite both datasets.
