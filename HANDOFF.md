# HANDOFF — HOPE liver project (state as of 19 Oct 2026)

Written for a reader with no memory of the previous session. Repo root: `D:\Research Internship BCI\HOPE_Liver_Recurrence_AI` (GitHub `https://github.com/vbvansh/hope-liver-recurrence-ai`, branch `main`). User: Vansh (intern, git user `vb2910`), learning from basics; wants simple language, numbered steps, tables.

---

## 1. Goal

**Long-term project (HOPE, Barts Cancer Institute / QMUL, CRUK-funded):** earlier detection of liver cancer (HCC) recurrence from longitudinal contrast CT + AFP blood tests, targeting small (≤ 5–10 mm) recurrent lesions. No public recurrence dataset exists; the in-house dataset (≈85 patients, ≈3 visits each, 39% of series with "Unknown" phase) is delayed.

**Current goal: MICCAI 2027 paper** (deadline estimated 25–26 Feb 2027, not yet announced; conference 26 Sep–1 Oct 2027, Auckland). Task defined in `Liver Lesion Detection on Multi-phase CT — MICCAI 2027 Plan v2.md` (repo root, by teammate Yifan Li, 14 Oct 2026):
- Lesion-level **detection of malignant focal liver lesions** (HCC, ICC, cHCC-CCA, metastasis) on registered multi-phase CT (NC / AP / PVP / DP); headline claim in the **10–19 mm** stratum (in-plane max diameter); sensitivity at 0.5 and 1 FP/study, patient-level sens/spec, FROC (1/8–8 FP), precision/recall/F1 at a validation-frozen operating point; Dice secondary.
- Input: any subset of phases (phase dropout); output: lesion candidates (centre + radius/box + confidence).

**Vansh's role: data lead.** Build the preprocessing/data pipeline for 5 datasets in `data_pipeline/` so all baselines (teammate Akshat: nnU-Net v2 2D/3D, MedNeXt, U-Mamba, PLAN, nnDetection, multi-phase nnU-Net, Reciprocal Cross-Modal Guidance BSPC 2024; plus Akshat's coarse-to-fine method) use identical data and splits.

---

## 2. Current status

### Done
- **`data_pipeline/` documents (no code yet):** `README.md`, `PLAN.md` (steps 0–14, decisions, open items, timeline), `STANDARD.md` (output format), `configs/datasets.yaml` (all settings), `.gitignore` (ignores `outputs/`, `logs/`, `third_party/`). All committed and pushed (last commit `5f133d8 Update pipeline README datasets`).
- **Registration experiments with the team pipeline `processing/`** (earlier phase of the project, now secondary to MICCAI):
  - HCC-TACE-Seg, 70 patients on Lightning AI → `experiments/registration/HCC_TACE_Seg_registration_output/` with `REPORT.md`, `per_patient_metrics.csv`, `report_figures/`.
  - WAW-TACE, 44 of 56 patients → `experiments/registration/WAW-TACE_registration_output/` with `REPORT.md`, `per_pair_metrics.csv`, `report_figures/`.
  - `experiments/registration/CONCERNS_AND_SOLUTIONS.md` (concerns 1–3 + literature), `RESULTS_VISUAL_GUIDE.md` (images wrapped in page-break-safe HTML), `RESULTS_VISUAL_GUIDE_PRESENTATION.md` (sections A2, A5, B4, Part C removed; summary table added).
- **Fixes committed in the team pipeline** `processing/HCC-TACE-Seg/` (approved by user): stacked-phase split + phase matching (`lib/standardize.py`), rounding KeyError (`pipeline/02_seg_to_mask.py`), Windows backslash paths (`lib/tcia_metadata.py`).
- `experiments/registration/simpleitk_baseline/`: an earlier self-built SimpleITK pipeline (notebook + `run_batch_registration.py`, 91/95 patients done) — kept only as a comparison baseline.

### In progress / waiting
- Nothing is running. No pipeline code has been written yet for `data_pipeline/`.
- Waiting on: MCT-LTDiag raw data from Yifan; downloads of PLC-CECT, TCIA Colorectal-Liver-Metastases, LiTS (131 labelled), WAW-TACE zips 2–4; compute machine for full runs; team answers to open items.

### Uncommitted files belonging to the user (do not touch / do not `git add .`)
`experiments/registration/CONCERNS_AND_SOLUTIONS.md` (modified), `experiments/registration/RESULTS_VISUAL_GUIDE.md` (modified), `HOPE_Experiment_Log_&_Results.docx`, `Liver Lesion Detection on Multi-phase CT — MICCAI 2027 Plan v2.md`, `experiments/registration/RESULTS_VISUAL_GUIDE_PRESENTATION.md` and `.pdf`, `literature_review/.../Health system learning enables generalist neuroimaging models.pdf`.

---

## 3. Key findings (with evidence)

### Datasets
1. **HCC-TACE-Seg** (`Datasets/HCC_TACE_Seg/manifest-1643035385102/`, 27.9 GB): 105 patients, 211 studies (104 patients with 2 sessions, 1 with 3), pre→post TACE median 63 days (9–212), 572 CT + 105 SEG series. Expert DICOM-SEG (Liver, Mass, Portal vein, Abdominal aorta) **only on the first (pre-TACE) session**; 104/105 SEGs carry `ReferencedSeriesSequence` + per-frame source SOP UIDs (HCC_048 has none).
2. **Stacked contrast passes:** some "LIVER 3 PHASE" folders hold 2 passes with identical slice positions, distinguished only by `AcquisitionNumber` (file names `1-xxx.dcm` / `2-xxx.dcm`). The team's `standardize.py` read them as one volume → anatomy appears twice (coronal view of HCC_002). Using the team's own `select_series`: **21 of 105 patients (30 of 205 timepoints)** affected: HCC_002, 003, 004, 007, 008, 010, 011, 030, 068, 075, 092, 093, 094, 095, 096, 098, 100, 101, 102, 104, 105.
3. **Phase mismatch between visits (HCC-TACE-Seg):** visits have ~2.8 CT series (with dye "LIVER 3 PHASE …", no dye "PRE"/"PRE LIVER"), none labelled by phase. `standardize.select_series` picks the series with **most slices**, ignoring dye. Of 69 registered pairs: 40 dye↔dye, **19 dye↔no-dye, 10 no-dye↔no-dye** (e.g. HCC_012 used "Recon 2 PRE LIVER" at both visits although dye series existed).
4. **WAW-TACE:** 233 treatment-naive HCC patients, one pre-TACE visit, up to 4 labelled phases (`<pid>_<0..3>_scan.nii.gz`, 0 = native, 1 = arterial, 2 = portal, 3 = delayed); CT zips hold whole patients: zip 1 = 56 patients (IDs 2…77), zip 2 = 57 (79…198), zip 3 = 57 (303…399), zip 4 = 63 (404…543); `tumor_masks_wawtace_v1_08_05_2024.zip` = 377 masks for all 233 (`<pid>_<phase>_<n>_tumor_seg.nrrd`). Locally only zip 1 + masks (`Datasets/WAW-TACE/`). 37 of 56 local patients have 4 phases, 19 have 3; lesion1 median diameter 45 mm (only 21/233 ≤ 20 mm).
5. **Clinical tables contain outcomes:** HCC-TACE-Seg (`HCC-TACE-Seg_clinical_data-V2.xlsx`, 105 rows): TTP, `Censored_0_progressed_1` (45 progressed), OS, AFP (all 105), mRECIST/RECIST/EASL. WAW-TACE (`clinical_data_wawtace_v2_15_07_2024.xlsx`, 233 rows): `progression` (56 events), `progression_time` (108 non-empty), `death` (170), `lab_afp` (all), `initial_tace_answer`, `initial_LR_TR`, lesion diameters / LI-RADS. Codes and units must be checked in `supplementary_table_s1_definitions_v2.xlsx`.
6. **MCT-LTDiag "small lesions" are probably fragments:** an earlier statistics document counted 508 lesions 0–5 mm and 458 of 5–10 mm (26-connected components, no minimum size filter), but per Plan v2 MCT-LTDiag's annotation protocol **excludes tumours < 1 cm** and slices are 5 mm. Not yet verified by eye.
7. **Positive-label inconsistency across datasets:** MCT-LTDiag masks include haemangiomas (benign) and exclude tumours < 1 cm; PLC-CECT marks HCC/ICC/cHCC-CCA only (+83 controls); WAW-TACE marks HCC only (benign unmarked); LiTS omits small uncertain structures; TCIA-CRLM marks metastases. Plan v2 resolves it with {malignant, haemangioma, ignore} label maps and an "unverifiable" bucket for unmatched < 10 mm candidates.
8. **Yifan's earlier work** (`LiverPreprocessing_and_Baseline_Results_Summary.md`, repo root): unified_v2 = 1,042 patients (517 MCT, 361 PLC, 164 WAW four-phase); small-held-out split (train only > 15 mm) 517/67/439; sensitivity 0–5 / 5–10 / 10–15 / > 15 mm best = 12.4% / 24.2% / 61.6% (synthetic tumours) / 90.6% (PLAN-1); synthetic tumours raise FP/case 0.50 → 1.73; PLAN-2 precision 0.862, F1 0.583, FP/case 0.34. His public code `github.com/lyf7374/LiverPreprocessing` crops by **liver ∪ tumour + 20 mm** (a test-time leak) and has no nnU-Net export; his processed outputs **cannot be shared**.

### Registration results (team pipeline `processing/`)
9. **HCC-TACE-Seg, 70 patients (Lightning T4):** 70 standardised, 69 registered, 0 errors (HCC_054 had only one usable visit). Medians: organ wDice 0.57 → 0.76 (better in 69/69); liver Dice 0.78 → 0.91 (n = 64; 63/64 improved; 41/64 ≥ 0.9); gradNCC 0.17 → 0.29; PSNR 11.9 → 13.3 dB (61/69); shared organs 13; organ shift 11 mm (2–42); residual 30 HU. Flags: NO_COLOR 35, KEPT_RIGID/NCC_REJECT 5 (HCC_011, 013, 016, 026, 046), REVERT_RAW 2 (HCC_020, 041), no flag 31. NO_COLOR vs matched: residual 37 vs 26 HU, liver Dice 0.92 vs 0.90 (alignment unaffected, comparison degraded). Stacked fix: 9 patients / 13 timepoints split; HCC_002 aorta/PV 332/114 vs 335/106 HU (arterial both), wDice 0.61 → 0.77 (Colab pilot without fix: 0.02 → 0.37, 3 shared organs). Run time ≈ 12.6 h for 68 patients (step 1 7.7 h CPU, CADS + SegVol 1.6 h GPU, step 6 3.3 h CPU).
10. **WAW-TACE, 44 patients / 114 phase pairs:** wDice 0.87 → 0.92 attempt / **0.91 delivered**; bending improved 103/113 but delivered in only 57/114; gradNCC 0.48 → 0.55; PSNR 17.1 → 17.7; organ shift 2 mm; residual 22 HU; NO_COLOR 40 (expected), REVERT_RAW 30, KEPT_RIGID 19, NCC_REJECT 9, NO_SHARED_ORGAN 1 (patient 25). Two-dye-phase pairs keep 32–35 HU vs 17–19 HU against no-dye. Liver Dice (test patients 2, 3 only) 0.96–0.98. Worst: 33 (0.40), 59 (0.40), 34 (0.42) — scans covering different body parts. Patients 62, 63, 65, 66, 67, 69, 7, 72, 73, 74, 77, 8 not registered (credits ran out; step 5 iterates in **text order**, so 7 and 8 come last).
11. **REVERT_RAW subtlety** (`processing/*/lib/longitudinal_pipeline.py`): `wdice_deformed` in logs/metadata is what the bending reached even when the safety net reverts to the raw scan; the delivered value equals rigid. HCC report footnote added for HCC_020.
12. **SegVol tumour masks unreliable** without the team's missing `processing/Combined/` folder (no `_segvol_calib/recommended.json`, no `_totalseg` organ-confine): HCC_005 baseline 4,724 vs expert 307,777 voxels; patient 58 tumour outside the liver.
13. **Concerns** (`CONCERNS_AND_SOLUTIONS.md`): (1) phase mismatch between visits, (2) small-lesion survival through deformable registration untested, (3) overlap metrics graded with the CADS outlines that drive registration (gradNCC is the independent one).

### Literature found (cited in `CONCERNS_AND_SOLUTIONS.md`)
14. Phase detection: TotalSegmentator `totalseg_get_phase -i ct.nii.gz -o contrast_phase.json`; **Segment-and-Classify** (arXiv 2501.14066, trained on WAW-TACE, F1 no-dye 0.994, arterial 0.937). Small synthetic tumours: Label-Free Liver Tumor Segmentation (CVPR 2023, SyntheticTumors), DiffTumor (CVPR 2024). Longitudinal: SimU-Net (Med Image Anal), Deep Lesion Tracker (CVPR 2021), Vivanti/Joskowicz IJCARS 2017, temporal subtraction CT (Sci Rep 2021: sensitivity 57 → 66%). Registration: MIND, ConvexAdam, uniGradICON, Learn2Reg.

---

## 4. Decisions made

### MICCAI data pipeline (user + Yifan, Oct 2026; recorded in `data_pipeline/PLAN.md`)
| Decision | Why |
|---|---|
| Datasets: train/val/internal test MCT-LTDiag + PLC-CECT; external WAW-TACE (all 233), TCIA Colorectal-Liver-Metastases (197, PVP), LiTS (131, PVP) | Plan v2 |
| HCC-TACE-Seg **dropped**; TriALS **on hold** | Plan v2: one lesion per patient, no delayed phase, single vendor 2002–2012 |
| Positive = malignant; haemangioma own class where annotated (MCT), ignore elsewhere; cysts ignore | Consistent labels across datasets (finding 7) |
| Spacing **1 × 1 × 1 mm isotropic**; images **linear** (B-spline allowed), masks **nearest**; models use it as-is | Team decision; simple and keeps PLC's thin-slice detail |
| Clip both datasets to **[-200, 200] HU** | PLC-CECT is released pre-clipped |
| **One crop from the automatic PVP liver mask + 20 mm**, never from a tumour mask | Avoids the test-time leak in LiverPreprocessing (liver ∪ tumour) |
| Lesion sizes measured on the **original/native scan**; strata by in-plane max diameter `<10` (exploratory), `10–19`, `20–29`, `>=30` mm | Resampling changes sizes; LI-RADS-aligned (Plan v2) |
| Registration QC: TotalSegmentator liver on **every phase**, liver Dice ≥ **0.9**; **5 fixed vessel landmarks**, **20 random patients per dataset** (seed 0), all phases where visible, liver Dice only where not (e.g. NC); **TRE > 5 mm = fail** | Independent check (concern 3) |
| Lesion-free masks: remove **malignant lesions only**, **one mask per phase**, margin 0 | Team decision; per phase allows per-phase alignment error |
| Split 70/10/20, patient-level, seed 0, stratified by dataset × tumour type × lesion count (1 / ≥ 2) × max-diameter bin; PLC controls proportional | Plan v2 |
| Reuse Yifan's public code where useful (PLC geometry recovery) instead of rewriting | Saves weeks; outputs unavailable but code public |
| Two releases: v1 = MCT + PLC (for Akshat), v2 = externals | Data lead is on the critical path |

### Earlier (registration work)
| Decision | Why / alternative rejected |
|---|---|
| Use the **team pipeline `processing/`**, not a self-built one | User explicitly rejected substitution ("huge mistake") |
| Stacked-phase fix = "Way 2": split by `AcquisitionNumber`, choose passes minimising mean \|ΔHU\| of aorta (CADS label 7) + portal/splenic vein (label 9) vs baseline, via one CADS call per patient | Uses CADS already in the pipeline; Way 3 (TotalSegmentator `get_phase`) would add a model/env |
| Lightning AI (T4 16 GB, 400 GB) for full runs | Hub disk too small; Colab sessions time out; laptop RAM too small |

---

## 5. Failed approaches (do not repeat)

1. **Self-built SimpleITK pipeline as a substitute** for the team pipeline → user called it a huge mistake. Never substitute; propose options and ask.
2. **QMUL JupyterHub** (`hub.comp-teach.qmul.ac.uk`, `~/Vansh_HA26001`): home disk 1 GB → 2 GB → 4.9 GB, root and `/var/tmp` read-only, `/tmp` 1 GB tmpfs; CADS + SegVol weights alone ≈ 1.5 GB plus environments → cannot run the team pipeline. RAM limit 15 GiB (`free -h` misleadingly shows 1.5 TB); GPU A100 sliced to 16 GB; Python 3.12; torch 2.9.1+cu128 preinstalled; conda not writable.
3. **Google Colab** (free tier, Python 3.13, T4 15 GB, ~5 h sessions): worked after fixes but user rejected it for long runs. Errors met and fixed (relevant again for any new environment):
   - `pip install -e CADS` → scikit-image metadata build failure (TPTBox==0.3.0 pins scikit-image 0.22 and numpy < 2, no wheels for 3.13) → **CADS env = Python 3.11 via `uv`**.
   - `ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value for backend` → set **`MPLBACKEND=Agg`** for every subprocess.
   - SegVol `TypeError: ViT.__init__() got an unexpected keyword argument 'pos_embed'` → **MONAI 1.3.2** (pos_embed removed in 1.4).
   - `ValueError: Due to a serious vulnerability issue in torch.load … upgrade torch to at least v2.6` (CVE-2025-32434, `pytorch_model.bin`) → **torch 2.6** in the SegVol env.
4. **Three concurrent copies** of `run_batch_registration.py` sharing `work/` deleted each other's downloads (`FileNotFoundError …/before/00000001.dcm`, `FileExistsError`, "could not download") → added a PID lock file; temporary failures retried, `ValueError` = permanent skip.
5. **Rounding KeyError** in SEG loading (`KeyError: -223.91`, HCC_021): `np.round` vs Python `round` disagree → fixed with one `np.round` + `np.searchsorted` (team `02_seg_to_mask.py`, baseline script, notebook).
6. **Windows-downloaded `metadata.csv`** has `.\HCC-TACE-Seg\…` paths → on Linux `posixpath.normpath` keeps backslashes → 0/677 series found → fixed in `lib/tcia_metadata.py`.
7. **Lightning terminal is zsh:** `PATIENTS="HCC_002 HCC_005"` passed as one argument; heredoc-created `~/hope_env.sh` was missing → always run `bash` first.
8. **WAW-TACE resume** after an accidental stop: credits had run out, no progress; stage-5 results file `metadata_full56.json` was never written (it is only written at the end), so 42 patients' scores were recovered from log lines; per-organ (liver) Dice exists only for test patients 2 and 3.
9. **WSL Ubuntu on the laptop**: system Python 3.10 without numpy; `pip3` belongs to python3.9 running as root → abandoned to avoid modifying the system.
10. **Asking Yifan for his processed unified_v2 outputs** → not possible; only his public code can be reused.
11. Local scratch venv (`…\scratchpad\venv`, Python 3.11) now has a broken pip (`No module named pip.__main__`, `_distutils_hack` warnings); Excel files were read with stdlib `zipfile` + XML instead.

---

## 6. Important references

### Files
| Path | Content |
|---|---|
| `Liver Lesion Detection on Multi-phase CT — MICCAI 2027 Plan v2.md` | Authoritative MICCAI plan (task, positive class, strata, metrics, datasets, harmonisation deliverables 2.2, external scoring rules 2.3) |
| `LiverPreprocessing_and_Baseline_Results_Summary.md` | Yifan's earlier pipeline + 7 baselines |
| `data_pipeline/PLAN.md`, `STANDARD.md`, `configs/datasets.yaml`, `README.md` | Data pipeline plan, format and settings |
| `docs/PROJECT_OVERVIEW.md`, `docs/datasets/` | HOPE overview; dataset statistics (`Three_Dataset_Statistics_EN.docx`, `Dataset_Summary_Sep2026.pdf`) |
| `processing/HCC-TACE-Seg/`, `processing/WAW-TACE/` | Team registration pipeline (`run.sh`, `pipeline/`, `lib/`); `processing/Combined/` is missing |
| `processing/HCC-TACE-Seg/lib/standardize.py` | Added `split_passes`, `build_standard_candidates`, `blood_pool_hu` (CADS labels 7 aorta, 9 portal/splenic vein, one `predict_images.py -task 551` call), `phase_name`, `choose_passes`, `resolve_passes`; metadata fields `n_passes, acquisition, aorta_hu, portal_vein_hu, phase, pass_selection` |
| `processing/*/lib/longitudinal_pipeline.py` | Registration engine (ANTs SyNRA on organ SDMs; flags NO_COLOR, KEPT_RIGID, NCC_REJECT, REVERT_RAW, NO_SHARED_ORGAN; writes `metadata_<LONG_SHARD_TAG>.json` only at the end) |
| `experiments/registration/*_registration_output/REPORT.md` | Both registration reports |
| `Datasets/` (gitignored, `/Datasets/` anchored) | Local raw data: HCC-TACE-Seg (full) + clinical xlsx; WAW-TACE zip 1 extracted + masks + clinical/metadata files |
| Memory dir `C:\Users\bhavs\.claude\projects\d--Research-Internship-BCI-HOPE-Liver-Recurrence-AI\memory\` | `MEMORY.md` index + notes (user profile, compute, team-pipeline rule, MICCAI pipeline, small commits, HCC facts) |

### Key settings (`data_pipeline/configs/datasets.yaml`)
`spacing_mm: [1,1,1]`, `interpolation: {image: linear, mask: nearest}`, `hu_clip: [-200, 200]`, `reference_phase: pvp`, `crop: {source: auto_pvp_liver, margin_mm: 20}`, `liver_dice_min: 0.9`, `landmarks: {points_per_case: 5, cases_per_dataset: 20, seed: 0, tre_max_mm: 5}`, `size_strata_mm: [10, 20, 30]`, `lesion_free_masks: {remove: malignant, per_phase: true, margin_mm: 0}`, `split: {ratios: 0.70/0.10/0.20, seed: 0}`. Label codes in `lesion_label.nii.gz`: 0 background, 1 malignant, 2 haemangioma, 3 ignore.

### Dataset sources
MCT-LTDiag: Harvard Dataverse doi:10.7910/DVN/S3RW15 (CC0), copy from Yifan. PLC-CECT: Science Data Bank doi:10.57760/sciencedb.12207 (CC BY 4.0). WAW-TACE: `https://zenodo.org/records/12741586` (download URL pattern `…/files/<name>?download=1`). TCIA Colorectal-Liver-Metastases: `https://wiki.cancerimagingarchive.net/x/wQWwC`. LiTS: 131 labelled training volumes. TCIA NBIA public API (no login): `https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID=<uid>`.

### Working environment recipe (Linux GPU machine; used on Lightning)
```bash
bash                                   # Lightning terminal is zsh
cat > ~/hope_env.sh <<'EOF'
export HOPE=~/hope_data
export CT_RAW=$HOPE/CTRaw/Abdomen CT_PROCESSED=$HOPE/CTProcessed CT_PROCESS=$HOPE/CTProcess
export PYTHON=$(which python)
export CADS_REPO=$HOPE/CADS CADS_PYTHON=$HOPE/envs/cads/bin/python SEGVOL_PYTHON=$HOPE/envs/segvol/bin/python
export CADS_WEIGHTS_PATH=$HOPE/weights/cads HF_HOME=$HOPE/weights/huggingface CT_GALLERY=$HOPE/gallery MPLBACKEND=Agg
EOF
source ~/hope_env.sh
pip install -q antspyx pydicom SimpleITK nibabel pandas matplotlib scipy uv
git clone -q --depth 1 https://github.com/murong-xu/CADS.git $CADS_REPO
uv venv -q --python 3.11 $HOPE/envs/cads
uv pip install -q --python $CADS_PYTHON torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install -q --python $CADS_PYTHON -e $CADS_REPO
uv venv -q --python 3.11 $HOPE/envs/segvol
uv pip install -q --python $SEGVOL_PYTHON torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -q --python $SEGVOL_PYTHON 'transformers>=4.40,<5' monai==1.3.2 'numpy<2' nibabel matplotlib einops scipy
```
CADS Task-551 weights ≈ 762 MB (reference licence, auto-download to `CADS_WEIGHTS_PATH`); SegVol `BAAI/SegVol` ≈ 724 MB. Team pipeline env vars also used: `STD_SHARD_TAG`, `LONG_SHARD_TAG` (per-run metadata file names).

### Compute facts
Laptop: HP Pavilion 15-ec2xxx, Ryzen 5 5600H (6c/12t), **8 GB RAM**, GTX 1650 4 GB, free disk F: ~169 GB, C: ~92 GB, D: ~49 GB → development only. Lightning AI: T4 16 GB, 400 GB disk, 0.55 credits/h; the account used for HCC/WAW ran out of credits. Hub: see failed approach 2.

### Git conventions
Claude makes small one-line commits of only its own files with trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; never `git add .`; the user pushes with `git push origin main`. `.gitattributes` forces LF for `*.sh` and `*.py`.

---

## 7. Open issues

### Open decisions (defaults in `data_pipeline/PLAN.md` → "Open items")
1. PLC-CECT registration: re-implement MCT-LTDiag authors' hierarchical B-spline, or run LiverPreprocessing's rigid → affine → B-spline? (default: start from LiverPreprocessing's code)
2. The 5 named landmarks, who places them, tool, native vs registered placement (default: portal/hepatic vein branch points, 3D Slicer, native + mapped).
3. Failed-case handling (Dice < 0.9 or TRE > 5 mm): exclude / flag / PVP-only (default: flag + `exclusion.csv`).
4. Whether LiverPreprocessing's PLC-CECT **+40 HU offset** still applies before clipping (ask Yifan).
5. Margin around lesions for lesion-free masks (default 0).
6. Missing-phase convention for multi-phase exports (proposal: zero-filled channel + presence flag; Plan v2 says phase dropout).
7. Lesion-level evaluation rule agreed with Akshat (Plan v2: centroid rule on WAW-TACE; earlier LiverPreprocessing used ≥ 10% voxel hit).
8. Plan v2 says "MCT-LTDiag multifocal mask policy to confirm with the authors".

### Known data risks
- MCT-LTDiag small components likely fragments (finding 6) — verify before any small-lesion claim.
- WAW-TACE: benign lesions unannotated → FP/study is an upper bound; phase availability varies.
- LiTS < 10 mm lesions only scorable when native slice ≤ 2 mm.

### Registration-work leftovers (lower priority now)
- WAW-TACE: 12 patients not registered (credits); liver Dice missing for 42 patients.
- `processing/Combined/` missing → SegVol runs uncalibrated without organ-confine.
- Phase matching for all HCC-TACE-Seg series (not only stacked folders) not implemented.
- `experiments/README.md` still says "Team pipeline full run … running on Lightning AI" — stale wording.

---

## 8. Next steps (ordered)

1. **Read** `data_pipeline/PLAN.md`, `STANDARD.md`, `configs/datasets.yaml` and the Plan v2 file before writing code.
2. **Step 1 inventory script** (`data_pipeline/scripts/inventory.py` or similar): per dataset list patients, phases present, native spacing/slice thickness, mask files, and compare with `expected_patients` in the config; output `inventory_<dataset>.csv`. Build and test first on the local WAW-TACE data (`Datasets/WAW-TACE/ct_scans_1_4_wawtace_09_05_24/<pid>/<pid>_<phase>_scan.nii.gz`, masks in `Datasets/WAW-TACE/tumor_masks_wawtace_v1_08_05_2024/`). Commit in small steps.
3. **Step 2 label maps**: create `data_pipeline/configs/label_maps.yaml` mapping each dataset's labels to {malignant, haemangioma, ignore} and recording whether sub-centimetre lesions were annotated.
4. When MCT-LTDiag arrives from Yifan: inventory it, then **Step 3** (read the already-registered phases, map labels, clip HU; keep native grid for lesion measurement).
5. Download PLC-CECT; pin a commit of `github.com/lyf7374/LiverPreprocessing` into `data_pipeline/third_party/` (gitignored) and record the hash in PLAN.md; **Step 4** geometry recovery (`estimate_plc_geometry.py`, TotalSegmentator task 297, GPU) and NC/AP/DP → PVP registration.
6. **Steps 5–6**: TotalSegmentator liver per phase + liver Dice QC; select 20 random patients per dataset (seed 0) and start the landmark/TRE work early (slowest manual step).
7. **Steps 7–12** for MCT + PLC → **release v1** (target ~15 Nov 2026): `lesions.csv` on native grid with manual review of multi-lesion cases; resample 1 mm, auto-liver crop + 20 mm, clip; lesion-free masks per phase; `manifest.csv`, `exclusion.csv`, QC report; splits (seed 0); nnU-Net v2 single/multi-phase exports with fixed `splits_final.json` and ignore label; nnDetection export.
8. **Step 13** externals (WAW-TACE all 233 incl. zips 2–4, TCIA-CRLM with vein masks, LiTS) → **Step 14** verification checks → **release v2** (target ~29 Nov 2026).
9. Ask the team the open decisions in section 7 and update `PLAN.md` "Decisions taken" whenever they answer.
10. Secure a compute machine for full runs (GPU for TotalSegmentator/PLC geometry; many CPU cores + RAM for registration; several hundred GB disk).
