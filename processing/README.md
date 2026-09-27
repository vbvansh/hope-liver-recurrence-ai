# Abdomen Cancer-Progression — processing

`CTRaw/Abdomen/<DS>` → `CTProcessed/<DS>` (1 mm RAS HU CT + organ/tumour masks) → `CTProcess/<DS>` (registered + intensity-matched). All under `/media/cbtil3/WhiteSD/` on cb.

Each dataset folder is self-contained: `pipeline/` (required steps, in order), `tables/` (metadata / clinical, where present), `report/` (text reports), `run.sh` (`--stage` / `--steps`), and its own `lib/` (engines + `lib/report/`).

| Dataset | Folder | Anchor organ |
|---|---|---|
| CPTAC-CCRCC | [CPTAC-CCRCC/](CPTAC-CCRCC/) | kidney |
| CPTAC-PDA | [CPTAC-PDA/](CPTAC-PDA/) | pancreas |
| EAY131 | [EAY131/](EAY131/) | auto |
| HCC-TACE-Seg | [HCC-TACE-Seg/](HCC-TACE-Seg/) | liver |
| NLST | [NLST/](NLST/) | lung lobes |
| RIDER | [RIDER/](RIDER/) | lung lobes |
| WAW-TACE | [WAW-TACE/](WAW-TACE/) | liver |

Order per dataset: download / cohort / clinical → standardize → CADS organs → SegVol tumour → longitudinal → report. Cross-dataset steps: [Combined/](Combined/) — `pipeline/` (SegVol calibration, watchers, rollout), `analysis/` (SegVol tuning, PSNR, mask quality), `visualization/`, `report/`.

## Tools

| Tool | Steps | Install |
|---|---|---|
| pydicom, SimpleITK, nibabel | standardize | conda env `mri` (`/home/cbtil3/hao/software/miniforge3/envs/mri`) |
| ANTsPy | build_longitudinal | `pip install antspyx` |
| CADS (nnU-Net Task-551) | cads | clone murong-xu/CADS; set `CADS_PYTHON`, `CADS_REPO` |
| SegVol (BAAI/SegVol) | segvol | env `segvol`; set `SEGVOL_PYTHON` |
| TotalSegmentator | `Combined/legacy/segment.py` | `pip install TotalSegmentator` |
| Qwen2.5-VL, MedGemma | report | `<DS>/lib/report/requirements.txt` |

Figures from the batch/demo engines go to `$CT_GALLERY` (default `/home/cbtil3/hao/repo_outputs/Awesome-Medical-Datasets/Collections/Abdomen/Cancer-Progression/visualization`).
