# Registration: open concerns and possible solutions

What the two registration experiments showed still needs fixing, and work by other researchers that can be built on.

- Experiments: [HCC-TACE-Seg report](HCC_TACE_Seg_registration_output/REPORT.md) (69 patients, before vs after TACE) and [WAW-TACE report](WAW-TACE_registration_output/REPORT.md) (44 patients, 114 contrast-phase pairs)
- Pipeline: team pipeline in [`processing/`](../../processing/)

---

## Where the concerns sit

HOPE's goal is to spot a **small returning liver tumour as early as possible** by comparing a patient's scans over time. Registration is the first part of that chain:

```
① pick the right scans → ② line them up → ③ make them comparable → ④ spot what changed → ⑤ small tumours must survive ②–④
```

The experiments show that **② works well**: the liver overlaps 91% across visits weeks apart (HCC-TACE-Seg) and 96–98% within one visit (WAW-TACE). The open concerns are in the other links.

| # | Concern | Link | Priority |
|---|---|---|---|
| 1 | Before and after scans are often not in the same contrast phase | ①, ③ | Highest: measured in half of the patients |
| 2 | It has not been tested whether a small new tumour survives the process | ④, ⑤ | High: this is HOPE's actual goal |
| 3 | The alignment is graded with its own answer key | measuring ② | Needed to trust any improvement |

---

## Concern 1: before and after scans are often not "like for like"

### The problem

Before a liver CT, the patient gets a dye injection. The dye moves through the body within minutes, so the same liver looks different depending on **when** the scan is taken: no dye, arterial (about 30 s), portal venous (about 70 s), delayed (about 3 min). These moments are called **contrast phases**. A liver tumour is mainly visible **because of** the dye.

At each visit the hospital made about 3 scans (with and without dye, at different timings). They are **not labelled** by phase. The pipeline picks **the scan with the most slices** and never checks the dye.

### Evidence from our experiments

| Finding | Number |
|---|---|
| HCC-TACE-Seg patients where a **no-dye** scan was used at one or both visits | **29 of 69** (10 at both visits, 19 at one) |
| Patients where brightness could not be evened out (`NO_COLOR`) | **35 of 69** |
| Leftover brightness difference in healthy tissue, mismatch vs matched patients | **37 HU vs 26 HU** (about 40% more) |
| Effect on alignment (liver overlap), mismatch vs matched | 0.92 vs 0.90: **none** |
| WAW-TACE: same-day, well-aligned phases still differ strongly | whole liver brighter in the difference image; 32–35 HU left between two dye phases |

Alignment follows organ **shapes**, so it is not harmed. The **"after minus before" picture** is: when the whole liver changes brightness because of the dye, a small tumour can be hidden, or a false one can appear.

### The fix

1. **Recognise the contrast phase** of every scan of every visit.
2. **Pick the same phase at both visits**, preferring scans with dye.
3. If no matching pair exists, **flag the patient** and analyse them separately.

The stacked-phase fix already does this for 9 patients (passes stored in one folder), using aorta and portal-vein brightness on CADS outlines. It needs extending to **all scans of a visit**.

### Work to build on

| Work | What it does | How we would use it |
|---|---|---|
| **TotalSegmentator `totalseg_get_phase`** ([tool](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/bin/totalseg_get_phase.py), [method](https://github.com/wasserth/TotalSegmentator/blob/master/resources/contrast_phase_prediction.md)) | Ready-made tool. Measures organ brightness and predicts the time since injection, mapped to no dye / early arterial / late arterial / portal venous, with a confidence value | Run on every candidate scan (`totalseg_get_phase -i ct.nii.gz -o contrast_phase.json`), then pick matching phases |
| **Segment-and-Classify** (2025, [arXiv 2501.14066](https://arxiv.org/abs/2501.14066)) | Organ-brightness features and a simple prediction model. **Trained on WAW-TACE**, tested on other hospitals' data (no dye 0.994, arterial 0.937 F1) | Closest to our idea. We have CADS outlines and true phase labels for all 56 WAW-TACE patients, so we can build and test the same kind of detector |
| Organ-outline phase classification (2024, [medRxiv](https://www.medrxiv.org/content/10.1101/2023.12.16.23299369.full.pdf), [PubMed](https://pubmed.ncbi.nlm.nih.gov/38629779)) | Same principle on 2,509 scans, 7 organs | Reference for which organs to measure |
| Two-step ResNet phase identification (2025, [Insights into Imaging](https://link.springer.com/article/10.1186/s13244-025-01995-7)) | Deep learning on the images directly; tested on 5 hospitals | Alternative if organ-based detection is not accurate enough |
| Phase recognition with random sampling (2022, [arXiv 2203.11206](https://arxiv.org/abs/2203.11206)) | Decides from sampled slices by majority vote | Background |

**Backups when no matching phase exists:**

| Work | What it does | Note |
|---|---|---|
| **MIND** (2012, [PubMed](https://pubmed.ncbi.nlm.nih.gov/22722056/)) | Compares local structure patterns instead of brightness, so dye differences do not fool the alignment | Helps alignment only |
| **ConvexAdam** ([GitHub](https://github.com/multimodallearning/convexAdam)) | Top-ranked Learn2Reg method; uses MIND; runs in seconds | Candidate for comparison with the team's method |
| **uniGradICON** (MICCAI 2024, [GitHub](https://github.com/uncbiag/uniGradICON)) | Pre-trained registration model that works on new data without retraining (GPU) | Candidate for comparison |
| Multi-phase CT synthesis from no-dye CT (2026, [J. Imaging](https://doi.org/10.3390/jimaging12080365)) | Generates the missing phase artificially | **Caution:** generated images can invent or erase small tumours |

These backups improve **alignment** of mismatched scans, which is already good. They do not remove the brightness difference in the comparison, so phase matching stays the main fix.

---

## Concern 2: does a small new tumour survive the process?

### The problem

The bending step (deformable registration) stretches and squeezes the follow-up scan to fit the organ shapes. A **new** tumour does not exist in the before scan, so the bending may treat it as a mismatch and **shrink or smear it**. The leftover brightness noise can also hide a faint small tumour.

### Evidence from our experiments

- Our scores describe **organs** (liver overlap 0.91). **None measures small tumours.**
- Leftover brightness difference in healthy tissue is **22–30 HU** even after alignment: the noise level a small tumour must stand out from.
- The safety checks threw the bending away in about half of the WAW-TACE pairs, which shows that bending does change image content.

HOPE targets lesions of a few millimetres, so this is the most important untested question.

### The fix

1. **Test directly:** place realistic artificial tumours (3 mm, 5 mm, 10 mm) into follow-up scans, run the pipeline, and check whether they remain, at the right size, in the difference picture.
2. **Check the squeeze:** measure how much the bending compressed or expanded each area (the pipeline stores its transforms in `xfm/`), especially inside tumours.
3. Later, **protect tumours** during bending and move to models that look at both scans together.

### Work to build on

| Work | What it does | How we would use it |
|---|---|---|
| **DiffTumor** (CVPR 2024, [GitHub](https://github.com/MrGiovanni/DiffTumor), [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Towards_Generalizable_Tumor_Synthesis_CVPR_2024_paper.pdf); already in `literature_review/`) | Creates realistic artificial early-stage liver tumours | Insert small tumours into follow-up scans for the survival test |
| **Temporal subtraction CT** ([Sci. Rep. 2021](https://www.nature.com/articles/s41598-021-97607-7), [Radiology 2017](https://pubmed.ncbi.nlm.nih.gov/28678671/)) | Radiologists given "after minus before" images found more bone metastases (57% → 66%) and read faster; false alarms rose slightly (0.17 → 0.21 per case) | Shows the subtraction idea works in practice, and that false alarms must be tracked |
| **New liver tumours in follow-up CT** (2017, [IJCARS](https://link.springer.com/article/10.1007/s11548-017-1660-z)) | First method built to find new liver tumours from baseline + follow-up scans; notes new tumours are typically small and easily missed | Closest earlier work to HOPE's goal |
| **SimU-Net** ([Medical Image Analysis](https://www.sciencedirect.com/science/article/abs/pii/S1361841522003036)) | Deep-learning model that analyses both scans together for liver lesion changes; 86% of lesions > 5 mm found, about 50% better precision than single-scan analysis | Template for HOPE's later detection model |
| **Deep Lesion Tracker** (CVPR 2021, [paper](https://arxiv.org/abs/2012.04872), [GitHub](https://github.com/JimmyCai91/DLT)) | Finds the same lesion in the next scan from appearance and position among organs (about 7 mm error), faster than registration | Alternative to pixel subtraction for following tumours |
| **Tumour-preserving registration** ([PubMed](https://pubmed.ncbi.nlm.nih.gov/42159478/), ["To deform or not"](https://arxiv.org/html/2401.09336), [tumour change with registration](https://pmc.ncbi.nlm.nih.gov/articles/PMC5496099/)) | Measures and prevents wrong shrinking or stretching of tumours during bending (breast MRI) | Method for the squeeze check and for protecting tumours |

---

## Concern 3: the alignment is graded with its own answer key

### The problem

The organ-overlap scores are computed with the **same CADS organ outlines that guided the bending**, like a student marking their own homework. Such scores tend to look better than the true alignment.

### Evidence from our experiments

- Organ and liver overlap are the headline scores in both reports and both use the CADS outlines.
- Only the edge-match score is independent. It improved too (HCC-TACE-Seg 0.17 → 0.29, WAW-TACE 0.48 → 0.55), so the improvement is real, but its size is uncertain.
- When a safety check discards the bending (`REVERT_RAW`), the log still records the overlap the bending reached, which needs care when reporting.

### The fix

Grade the alignment with references that were **not used to align**:

1. **Radiologist tumour outlines.** WAW-TACE has 377 of them, often on several phases of the same patient. After alignment they should lie on top of each other.
2. **Landmarks:** points such as vessel branchings marked in both scans; measure their distance after alignment.
3. Report the **delivered** result separately from the bending attempt.

### Work to build on

| Work | What it does | How we would use it |
|---|---|---|
| **Learn2Reg challenge** ([arXiv 2112.04489](https://arxiv.org/pdf/2112.04489)) | Community standard for comparing registration methods with independent outlines and landmarks; includes an abdomen CT task | Follow its evaluation principle with WAW-TACE's radiologist outlines |
| **ConvexAdam**, **uniGradICON** (see Concern 1) | Strong published registration methods | Run on the same patients and compare with the team's method under the independent measure |

---


