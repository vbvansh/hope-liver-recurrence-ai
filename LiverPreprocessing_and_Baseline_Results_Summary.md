# Liver CT Preprocessing Pipeline & Small-Lesion Baseline Results

## 1. Big Picture

This document combines the two materials discussed:

1.  **LiverPreprocessing repository** --- prepares and standardizes
    three liver CT datasets so that they can be used together.
2.  **Baselines on Small Held-out Lesions** --- evaluates several
    segmentation approaches on that standardized dataset, with a special
    focus on whether models trained on **large tumors (\>15 mm)** can
    detect **small tumors**.

### The overall workflow

``` text
Raw CT datasets
      ↓
Fix geometry / metadata problems
      ↓
Standardize orientation and intensity representation
      ↓
Register the four CT phases
      ↓
Resample to a common 1 × 1 × 1 mm grid
      ↓
Crop around liver + tumor
      ↓
Create standardized masks and metadata
      ↓
Unified dataset (unified_v2)
      ↓
Small-held-out train/validation/test split
      ↓
Train different segmentation approaches
      ↓
Evaluate detection by lesion size
```

------------------------------------------------------------------------

# Part I --- Liver CT Preprocessing Pipeline

## 2. Why is preprocessing needed?

The project combines three datasets:

-   **MCT-LTDiag**
-   **PLC-CECT**
-   **WAW-TACE**

They were collected independently, so they do not naturally have the
same:

-   voxel spacing
-   image orientation
-   CT intensity representation
-   phase alignment
-   image geometry
-   annotation format

A deep-learning model expects its inputs to be reasonably consistent.

So the preprocessing pipeline creates a **common representation**.

The key idea is:

> **Do not simply put the three datasets together. First make their
> physical representation and labels comparable.**

------------------------------------------------------------------------

## 3. Four CT phases

The pipeline works with four phases:

  Phase     Meaning
  --------- ---------------------
  **NC**    Non-Contrast
  **AP**    Arterial Phase
  **PVP**   Portal Venous Phase
  **DP**    Delayed Phase

The final input is therefore conceptually:

``` text
Channel 0 → NC
Channel 1 → AP
Channel 2 → PVP
Channel 3 → DP
```

These four phases contain different information about how tissue and
tumors enhance after contrast.

------------------------------------------------------------------------

## 4. The PLC-CECT geometry problem

One of the most important preprocessing issues is **PLC-CECT**.

Its released image metadata cannot simply be trusted. The stored headers
can claim a standard-looking geometry even though the actual physical
slice spacing and orientation are different.

If the geometry is wrong, then things such as:

-   tumor size
-   spatial distances
-   registration
-   liver volume
-   lesion measurements

can also become wrong.

Therefore, the pipeline first tries to **recover the actual physical
geometry**.

------------------------------------------------------------------------

## 5. How PLC geometry is recovered

The pipeline uses anatomical structures as a physical reference.

In particular, vertebrae are detected and used like a **ruler**.

Conceptually:

``` text
Detect vertebrae
      ↓
Find how many CT slices separate neighboring vertebrae
      ↓
Use their expected physical distances
      ↓
Estimate actual slice spacing
```

It also checks whether the stored slice direction is reversed.

The important idea is:

> **Instead of blindly trusting broken metadata, use anatomy and image
> information to recover the geometry.**

TotalSegmentator is used as part of this geometry-recovery process; it
is not the final tumor segmentation model.

------------------------------------------------------------------------

## 6. Intensity / HU handling

The pipeline keeps CT values in a physically meaningful form rather than
immediately normalizing everything for a neural network.

The unified input is stored as **int16 HU**.

PLC-CECT is special because its released values are not directly stored
like the other datasets. The pipeline reconstructs an approximate HU
representation and records a recommended **+40 HU offset** for PLC-CECT
in the manifest.

Important principle:

> Dataset-specific corrections are explicitly recorded rather than
> silently changing the raw converted images.

The later training pipeline can apply the recommended intensity offset
and clipping.

------------------------------------------------------------------------

## 7. Standard orientation

The images are standardized to a common **LPS orientation**:

-   L = Left
-   P = Posterior
-   S = Superior

This gives the datasets a consistent anatomical coordinate convention.

------------------------------------------------------------------------

## 8. Registering the four phases

The same patient's NC, AP, PVP and DP images are not necessarily
perfectly aligned.

For example:

``` text
NC  → liver in one position
AP  → liver shifted slightly
PVP → liver shifted again
DP  → another small shift
```

If these are given directly to a model, the same tumor may appear at
slightly different locations in the four channels.

### PVP is used as the reference

The pipeline approximately does:

``` text
NC ──┐
AP ──┤
DP ──┤──→ PVP coordinate system
     │
PVP ─┘
```

Registration can involve:

1.  liver-centroid based initialization
2.  rigid registration
3.  affine registration
4.  deformable B-spline registration

The goal is to make the anatomy line up across phases.

------------------------------------------------------------------------

## 9. Registration quality control

The pipeline does not blindly trust registration.

It checks how well the liver masks overlap.

A common measure is **Dice**:

``` text
Dice = 1.0 → perfect overlap
Dice = 0.0 → no overlap
```

A registration quality gate around **0.80 liver Dice** is used.

Importantly, failed/poor phases are not simply hidden. The case metadata
records phase usability, allowing downstream code to decide whether a
phase should be used.

------------------------------------------------------------------------

## 10. Tumor masks and registration

Tumor annotations are not used to drive the registration transform.

Instead, registration is based on image/anatomical information such as
the liver.

For WAW-TACE, when the tumor annotation exists in one phase, the
corresponding transform can be used to move the tumor mask into the
common PVP space.

This helps produce a consistent final tumor label.

------------------------------------------------------------------------

## 11. Resampling to a common grid

After alignment, the images are resampled to:

``` text
1 × 1 × 1 mm
```

This gives all datasets a common coordinate grid.

### Important warning

A 1-mm output grid does **not** mean that the original CT had 1-mm
physical resolution.

For example:

``` text
Original scan:
0.7 × 0.7 × 5 mm

After resampling:
1 × 1 × 1 mm
```

The output voxels are 1 mm apart, but the original scan did not contain
1-mm detail along its 5-mm direction.

This is especially important for **very small tumors**.

------------------------------------------------------------------------

## 12. Cropping

The pipeline creates a crop around:

``` text
liver + tumor
```

and adds approximately a **20 mm margin**.

This reduces unnecessary background and makes training more efficient.

Using both liver and tumor information helps avoid accidentally cropping
away tumor tissue.

------------------------------------------------------------------------

## 13. Final standardized case

A processed case can be thought of as:

``` text
Patient
 ├── NC
 ├── AP
 ├── PVP
 ├── DP
 ├── liver mask
 ├── tumor mask
 └── metadata / quality information
```

The metadata is important because it records things such as:

-   dataset
-   geometry information
-   transforms
-   phase usability
-   registration quality
-   diagnosis/labels
-   lesion information
-   preprocessing information

Think of `case.json` as the **passport of the processed case**.

------------------------------------------------------------------------

# Part II --- Small-Held-Out Baseline Experiment

## 14. What question does the experiment ask?

The baseline study asks:

> **If a model is trained only on patients whose tumors are larger than
> 15 mm, how well can it detect smaller tumors that were not represented
> in training?**

This is deliberately harder than a normal random train/test experiment.

The split is called:

**small-held-out, seed 0**

------------------------------------------------------------------------

## 15. Dataset split

The unified dataset contains:

-   **1,042 total patients**
-   **517 MCT-LTDiag**
-   **361 PLC-CECT**
-   **164 WAW-TACE**

The split is:

  Split          Patients
  ------------ ----------
  Train               517
  Validation           67
  Test                439

There are **no lesions ≤15 mm in training or validation**.

The test set contains the small lesions.

------------------------------------------------------------------------

## 16. An important experimental caveat: dataset shift

The split also changes the dataset composition.

### Training

PLC-CECT is the largest contributor.

### Test

MCT-LTDiag becomes the dominant dataset.

Approximately:

``` text
Training:
PLC-CECT → 262 cases
MCT-LTDiag → 144
WAW-TACE → 111

Test:
MCT-LTDiag → 354
PLC-CECT → 48
WAW-TACE → 37
```

Therefore the experiment tests two things at once:

``` text
Large-lesion training
        ↓
Small-lesion testing

PLUS

Dataset distribution shift
```

So poor performance cannot automatically be attributed only to small
tumor size.

------------------------------------------------------------------------

# Part III --- The Seven Experiments

## 17. The seven runs

The study compares:

1.  **nnU-Net**
2.  **Plain 3D U-Net**
3.  **U-Net + Synthetic Small Tumors**
4.  **U-Net + CC-DiceCE**
5.  **U-Net + Phase Disentanglement**
6.  **PLAN Stage 1**
7.  **PLAN Stage 2**

------------------------------------------------------------------------

## 18. Baseline 1 --- nnU-Net

nnU-Net is used as a standard strong segmentation baseline.

Results:

-   lesion recall: **0.413**
-   lesion precision: **0.786**
-   F1: **0.542**
-   false positives/case: **0.54**

------------------------------------------------------------------------

## 19. Baseline 2 --- Plain 3D U-Net

The plain 3D U-Net gets:

-   lesion recall: **0.413**
-   lesion precision: **0.797**
-   F1: **0.544**
-   false positives/case: **0.50**

The two baselines are essentially tied.

This gives a useful reference point for judging the later modifications.

------------------------------------------------------------------------

# Part IV --- Synthetic Small Tumors

## 20. Why create synthetic tumors?

The training data does not contain small lesions.

So the researchers create artificial small tumors during training.

The idea is:

``` text
Real training data:
mostly large tumors

+

Artificial:
small tumors

↓

model gets exposure to small lesions
```

------------------------------------------------------------------------

## 21. How are they created?

The synthetic lesions are approximately:

-   3--15 mm equivalent spherical diameter
-   1--3 lesions per selected patch
-   placed inside the liver
-   away from real tumors
-   given phase-specific intensity changes

Synthetic augmentation is applied with probability **0.5 per patch**.

------------------------------------------------------------------------

## 22. Result of synthetic tumors

Overall lesion recall:

``` text
Plain U-Net       0.413
Synthetic U-Net   0.508
```

This is the **highest overall recall** of the seven experiments.

More importantly:

### 5--10 mm lesions

``` text
Plain U-Net → 0.061
Synthetic   → 0.242
```

### 10--15 mm lesions

``` text
Plain U-Net → 0.441
Synthetic   → 0.616
```

So synthetic training clearly helps the model find smaller lesions.

------------------------------------------------------------------------

## 23. The problem with synthetic tumors

The model becomes much more likely to produce false positives.

``` text
Precision:
0.797 → 0.583

False positives/case:
0.50 → 1.73
```

So the model becomes:

> **more sensitive but much less precise.**

This is a classic recall-vs-precision trade-off.

For 0--5 mm lesions, synthetic training produces **338 false-positive
components**, compared with **65** for the plain U-Net.

------------------------------------------------------------------------

# Part V --- CC-DiceCE

## 24. Why change the loss?

A large tumor contains many more voxels than a tiny tumor.

For example:

``` text
Large tumor → 10,000 voxels
Small tumor → 100 voxels
```

A normal voxel-based loss can therefore give much more influence to the
large tumor.

CC-DiceCE tries to make **each lesion count more equally**, rather than
allowing large lesions to dominate.

------------------------------------------------------------------------

## 25. CC-DiceCE result

Compared with Plain U-Net:

``` text
Recall:
0.413 → 0.455

F1:
0.544 → 0.573
```

Small-lesion sensitivity also improves:

``` text
5–10 mm:
0.061 → 0.105

10–15 mm:
0.441 → 0.529
```

This is a good compromise because it improves recall without the
dramatic precision collapse seen with synthetic augmentation.

------------------------------------------------------------------------

# Part VI --- Phase Disentanglement

## 26. What is the idea?

The four CT phases contain:

-   information common across phases
-   information specific to each phase

The experiment tries to explicitly separate those two types of
information.

Conceptually:

``` text
NC ─┐
AP ─┤
PVP ┤ → common information
DP ─┘

plus phase-specific information
```

------------------------------------------------------------------------

## 27. Result

It did not help.

Results:

-   recall: **0.406**
-   F1: **0.536**

It also took about **11.8 hours** to train.

The disentanglement loss collapsed toward zero within about ten epochs.

Conclusion:

> This version of phase disentanglement did not provide a useful
> improvement and was not pursued further.

------------------------------------------------------------------------

# Part VII --- PLAN

## 28. PLAN Stage 1

PLAN-1 uses a segmentation/pixel branch and a different training setup,
including:

-   RAdam optimizer
-   lesion-forced sampling
-   PLAN's post-processing

Results:

-   recall: **0.455**
-   precision: **0.797**
-   F1: **0.579**
-   FP/case: **0.55**

This is one of the best balanced results.

------------------------------------------------------------------------

## 29. What is lesion-forced sampling?

Instead of randomly sampling patches and sometimes getting no tumor, the
training process deliberately samples patches containing lesions more
often.

The model therefore sees more lesion examples during training.

------------------------------------------------------------------------

# Part VIII --- PLAN Stage 2

## 30. What does PLAN-2 add?

PLAN-2 starts with PLAN-1 and adds:

### Lesion branch

A branch specifically designed to reason about individual lesions.

### Patient branch

A branch that predicts whether a patient has a tumor.

Conceptually:

``` text
                 PLAN
                   │
        ┌──────────┼──────────┐
        │          │          │
      Pixel      Lesion     Patient
      branch     branch     branch
        │          │          │
        └──────────┼──────────┘
                   ↓
              final output
```

------------------------------------------------------------------------

## 31. PLAN-2 results

PLAN-2 achieves:

-   case Dice: **0.721** --- highest
-   lesion precision: **0.862** --- highest
-   F1: **0.583** --- highest
-   false positives/case: **0.34** --- lowest

So PLAN-2 is the **cleanest and most precise model overall**.

------------------------------------------------------------------------

## 32. But PLAN-2 does NOT improve small-lesion recall

This is an important finding.

PLAN-1:

``` text
Lesion recall = 0.455
```

PLAN-2:

``` text
Lesion recall = 0.440
```

And size-specific sensitivity for PLAN-2 is approximately:

``` text
0–5 mm     → 6.3%
5–10 mm   → 10.1%
10–15 mm  → 48.8%
>15 mm    → 89.4%
```

Therefore:

> Adding explicit lesion and patient branches improved precision and
> reduced false positives, but did not solve the small-lesion detection
> problem.

------------------------------------------------------------------------

# Part IX --- Most Important Results by Tumor Size

## 33. This is the most important table

  -------------------------------------------------------------------------------
  Lesion        nnU-Net      Plain   Synthetic   CC-DiceCE      PLAN-1     PLAN-2
  size                       U-Net                                     
  ---------- ---------- ---------- ----------- ----------- ----------- ----------
  **0--5           5.1%       6.5%   **12.4%**        8.3%        7.3%       6.3%
  mm**                                                                 

  **5--10          7.8%       6.1%   **24.2%**       10.5%       11.6%      10.1%
  mm**                                                                 

  **10--15        42.5%      44.1%   **61.6%**       52.9%       51.2%      48.8%
  mm**                                                                 

  **\>15          87.1%      86.3%       88.8%       89.8%   **90.6%**      89.4%
  mm**                                                                 
  -------------------------------------------------------------------------------

### The pattern is very clear

``` text
Large tumors:
        EASY

10–15 mm:
        HARDER

5–10 mm:
        VERY HARD

0–5 mm:
        EXTREMELY HARD
```

Even the best method detects only **12.4% of 0--5 mm lesions**.

------------------------------------------------------------------------

# Part X --- Main Overall Results

  Model               Recall   Precision          F1    FP/case
  -------------- ----------- ----------- ----------- ----------
  nnU-Net              0.413       0.786       0.542       0.54
  Plain U-Net          0.413       0.797       0.544       0.50
  Synthetic        **0.508**       0.583       0.543       1.73
  CC-DiceCE            0.455       0.774       0.573       0.63
  Disentangled         0.406       0.787       0.536       0.52
  PLAN-1               0.455       0.797       0.579       0.55
  PLAN-2               0.440   **0.862**   **0.583**   **0.34**

------------------------------------------------------------------------

# Part XI --- Detection vs Segmentation Quality

## 34. A key conclusion

The different methods change **lesion detection** much more than they
change the quality of the actual tumor boundary.

Across the seven runs:

-   case Dice stays around **0.703--0.721**
-   pooled voxel Dice stays around **0.825--0.851**

So the main challenge is not simply:

> "Can the model draw the tumor boundary?"

It is:

> **"Can the model find the tiny tumor in the first place?"**

This is especially true for lesions below 10 mm.

------------------------------------------------------------------------

# Part XII --- Post-processing

## 35. PLAN's post-processing is important

PLAN removes certain obvious false positives, including:

-   components outside/too far from the liver
-   very small predicted components
-   keeps the relevant liver component

The post-processing substantially improves precision while preserving
lesion sensitivity.

Without post-processing:

``` text
PLAN-1 precision ≈ 0.669
PLAN-2 precision ≈ 0.680
```

After post-processing:

``` text
PLAN-1 precision = 0.797
PLAN-2 precision = 0.862
```

False positives also fall substantially.

So post-processing is not just a minor detail; it contributes
meaningfully to PLAN's final performance.

------------------------------------------------------------------------

# Part XIII --- What Should Be Taken Forward?

## 36. Most promising directions from this experiment

### 1. Combine synthetic tumors + CC-DiceCE

This is the clearest next experiment.

Synthetic tumors provide:

``` text
small-lesion recall ↑↑
```

CC-DiceCE provides:

``` text
better balance
```

The obvious question is:

> Can both be combined to get the synthetic recall improvement without
> the huge false-positive increase?

------------------------------------------------------------------------

### 2. Reduce synthetic augmentation probability

The synthetic experiment used probability **0.5 per patch**.

A lower probability could potentially reduce false positives while
retaining some small-lesion benefit.

------------------------------------------------------------------------

### 3. Tune inference filtering carefully

A minimum predicted-component size can remove tiny false positives.

But it must be tuned carefully because real 3--5 mm tumors also exist.

------------------------------------------------------------------------

### 4. Focus on detection, not only Dice

For small-lesion research, always report:

-   lesion recall
-   lesion precision
-   lesion F1
-   false positives per case
-   sensitivity by lesion size
-   voxel Dice
-   case Dice

A single Dice number can hide poor small-lesion detection.

------------------------------------------------------------------------

# Part XIV --- Important Caveats

## 37. 1-mm resampling is not the same as 1-mm original resolution

This is critical when studying tiny tumors.

A scan originally acquired with thick slices does not magically gain
fine z-axis information just because it is resampled to 1 mm.

------------------------------------------------------------------------

## 38. The test set contains strong dataset shift

The training set is dominated by PLC-CECT while the test set is
dominated by MCT-LTDiag.

Therefore, the reported test performance reflects:

``` text
large → small generalization
+
cross-dataset generalization
```

These should ideally be separated in future experiments if the
scientific question is specifically about small-lesion generalization.

------------------------------------------------------------------------

## 39. Very small lesions remain unsolved

The biggest unresolved problem is:

``` text
0–5 mm lesions
```

Even the best sensitivity is only:

``` text
12.4%
```

with synthetic training.

That means there is substantial room for research.

------------------------------------------------------------------------

# 40. Final Mental Model

The entire project can be remembered as:

``` text
THREE DIFFERENT CT DATASETS
          ↓
   PREPROCESSING
          ↓
 ┌─────────────────────┐
 │ Fix geometry        │
 │ Align phases        │
 │ Standardize HU      │
 │ Standardize LPS     │
 │ Resample to 1 mm    │
 │ Crop liver + tumor  │
 │ Create masks/labels │
 │ Save QC metadata    │
 └─────────────────────┘
          ↓
       unified_v2
          ↓
  SMALL-HELD-OUT SPLIT
          ↓
Train only on >15 mm lesions
          ↓
Test on small lesions
          ↓
 ┌─────────────────────────────┐
 │ nnU-Net                     │
 │ Plain U-Net                 │
 │ + Synthetic tumors          │
 │ + CC-DiceCE                 │
 │ + Disentanglement           │
 │ PLAN-1                      │
 │ PLAN-2                      │
 └─────────────────────────────┘
          ↓
      MAIN FINDING
          ↓
Small-lesion detection is the
real bottleneck.
```

## The single most important conclusion

> **Better segmentation architecture alone does not solve tiny-lesion
> detection. Synthetic small-tumor training substantially improves
> sensitivity, especially for 5--10 mm and 10--15 mm lesions, but
> creates many false positives. PLAN-2 gives the best overall
> precision/F1 and fewest false positives, but still does not
> substantially improve detection of very small lesions.**

The most promising next step identified by the experiment is therefore
to explore **synthetic small-tumor training + CC-DiceCE**, while
carefully controlling false positives.

------------------------------------------------------------------------

## Sources

-   [LiverPreprocessing
    repository](https://github.com/lyf7374/LiverPreprocessing)
-   **Baselines on Small Held-out Lesions**, experiment report,
    September 27--29, 2026.
