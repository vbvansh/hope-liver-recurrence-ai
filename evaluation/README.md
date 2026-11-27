# HOPE shared scorer (evaluation)

> **Important:** every detection number in the HOPE MICCAI paper (FROC, sensitivity at fixed false positives, AP, patient AUROC, confidence intervals) must come from **this** scorer, with its fixed rules, for every model (ours, nnU-Net and the teammates' models). Numbers from any other script are not comparable. Any change to a rule = new scorer version, and every reported model is re-scored.

## What it does

A model only draws candidate blobs with confidence scores. The scorer marks them against the release truth, like the marking scheme of an exam:

1. reads the truth of each listed case from the release (`lesion_instances`, `lesion_label`, `lesions.csv` for sizes);
2. reads the model's candidates (one ID map per case on the release 1 mm grid + `scores.csv`);
3. matches candidates to lesions with the team rules below;
4. writes FROC by lesion size, sensitivity at 0.5 / 1 / 2 false positives per case, AP, patient AUROC and bootstrap 95 % intervals.

## Fixed rules (decided by the data lead, 26 Nov 2026)

| Rule | Value |
|---|---|
| Targets (lesion level) | Every lesion in `lesion_instances.nii.gz`: malignant (class 1) **and** haemangioma (class 2), as released |
| Neutral | Ignore regions (class 3): a candidate on them is neither a hit nor a false positive |
| Hit | Candidate and lesion masks overlap with **IoU ≥ 0.1**, matched **one-to-one** (greedy, highest score first); a candidate that only overlaps an already-found lesion is a false positive |
| Patient level | Score = highest candidate score; positive = malignant diagnosis (haemangioma patients and controls negative) |
| FROC points | 1/8, 1/4, 1/2, 1, 2, 4, 8 false positives per case; FROC mean = mean of the 7 |
| Size strata | In-plane max diameter from `lesions.csv`: < 10, 10–19, 20–29, ≥ 30 mm; plus < 20 mm and all |
| Plan v2 "unverifiable" rule | Unmatched candidates < 10 mm (equivalent diameter) on MCT-LTDiag / LiTS: reported both with (excluded) and without (counted as false positives, the main result) |
| Grid | Release 1 mm grid; predictions made on another grid must be mapped back first |

## Documented choices (to confirm with the team's detector code or predictions when available)

1. A candidate is neutral when ≥ 50 % of its voxels lie in ignore regions, or its IoU with a touched ignore region is ≥ 0.1.
2. Candidates from probability maps (`candidates_from_nnunet.py`): connected components at 0.5 / 0.2 / 0.05 / 0.01 / 0.002 and ≥ 15 voxels (as in the team's detector report of 21 Nov 2026); a lower-level component touching an accepted candidate is dropped, so candidates never overlap; score = highest probability inside.

## Files

| File | What |
|---|---|
| `scorer.py` | The scorer (version 1.0, 2026-11-27) |
| `candidates_from_nnunet.py` | nnU-Net probability maps (`--npz`) → candidate maps + `scores.csv` |
| `test_scorer.py` | Tests on made-up cases with hand-computed answers (statuses, FROC points and means in both modes, AP, AUROC, malignant-only targets, command line, candidate converter) |

## Usage

```bash
# candidates from nnU-Net validation probabilities
"$HOPE_PYTHON" -I evaluation/candidates_from_nnunet.py --npz-dir <fold_0/validation> --release <processed/v1> --out <cand_dir>
# score
"$HOPE_PYTHON" -I evaluation/scorer.py --release <processed/v1> --cases <processed/v1/splits/val.txt> \
    --pred <cand_dir> --out <out_dir> --bootstrap 1000
# tests (empty temporary folder)
"$HOPE_PYTHON" -I evaluation/test_scorer.py <empty folder>
```

Outputs: `metrics.csv` (one row = the tracker columns), `froc.csv`, `summary.md` (counts only), `settings.json`, and `lesions.csv` / `candidates.csv` (with case ids: keep out of git).

## Status

- 2026-11-27: written; all tests pass on made-up cases.
- Next check: reproduce one number of the team's detector report from its saved predictions (e.g. nnU-Net, test, < 20 mm at 2 false positives per case = 0.713, with the same rules), once they are shared.
