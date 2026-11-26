# {EXP_ID}: {TITLE}

| Field | Value |
|---|---|
| ID | {EXP_ID} |
| Owner | {OWNER} |
| Status | planned <!-- planned / running / done / failed / abandoned --> |
| Started / finished | {DATE} / – |
| Parent experiment | {PARENT} |
| Data | Release v1.1 (843 patients, split 588 / 85 / 170), 1 × 1 × 1 mm grid <!-- say if resampled --> |
| Phases used | NC, AP, PVP, DP <!-- or PVP only, etc. --> |
| Split evaluated | validation <!-- validation / dev-val / test (only with a frozen protocol) / external --> |
| Code commit | <!-- git rev-parse --short HEAD --> |
| Job IDs | <!-- SGE job IDs or other cluster IDs --> |

## 1. Question and hypothesis

<!-- One question this experiment answers, and what you expect. Example: "Does phase dropout raise small-lesion sensitivity? Expect +0.02 FROC mean < 20 mm." -->

## 2. What changes compared with the parent

<!-- Ideally ONE change, so the result can be attributed to it. List it here. -->

## 3. Setup

| Item | Value |
|---|---|
| Model / architecture | |
| Input (phases, patch size, grid) | |
| Loss | |
| Sampling / augmentation | |
| Training (epochs × iterations, batch, optimiser, learning rate) | |
| Seeds | |
| Hardware and time | |
| Config file | `config.yaml` <!-- in this folder --> |

## 4. How to run

```bash
cd <project path>/hope-liver-recurrence-ai
# exact commands, e.g. qsub experiments/{FOLDER}/train.sh
```

## 5. Evaluation

| Item | Value |
|---|---|
| **Positive class (fixed team rule, 26 Nov 2026)** | **Lesion level: every lesion numbered in `lesion_instances.nii.gz`, i.e. malignant (class 1) and haemangioma (class 2), as released. Ignore regions (class 3) are neutral: a detection on them is neither a hit nor a false alarm. Patient level: malignant diagnosis vs others (haemangioma patients and controls are negatives).** <!-- If this experiment adds a "malignant only" line, label those numbers clearly. --> |
| Evaluation code and version | |
| Lesion matching rule (fixed team rule, 26 Nov 2026) | A candidate hits a lesion when their 3D masks have IoU ≥ 0.1, matched one-to-one (extra candidates on a found lesion are false positives) |
| FROC points | 1/8, 1/4, 1/2, 1, 2, 4, 8 false positives per case |
| Size strata | < 20 mm (< 10 mm merged, Plan v2 rule), 20–29 mm, >= 30 mm |
| Ignore regions | class 3 never counted as true or false positive |

## 6. Results

<!-- Copy the main numbers into ../tracker.csv. Full tables go into results/ (small CSV files only). -->

| Metric | Value |
|---|---|
| FROC mean, lesions < 20 mm | |
| Sensitivity < 20 mm at 0.5 / 1 / 2 FP per case | |
| FROC mean, all lesions | |
| AP, all lesions | |
| Patient AUROC | |

## 7. What we learned

<!-- One paragraph: keep / drop / change, and why. Negative results are results: write them down. -->

## 8. Next step

<!-- The experiment this one leads to. -->

## 9. Files

<!-- Relative names only, never private paths. Big files stay on the cluster:
predictions -> outputs/experiments/{EXP_ID}/, checkpoints -> checkpoints/{EXP_ID}/ (both under the project storage, not in git). -->

---
Before committing this card: no patient IDs, no private paths, user names or host names, no images or per-case tables.
