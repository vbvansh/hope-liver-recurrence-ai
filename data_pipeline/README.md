# Liver CT data pipeline (MICCAI 2027)

Prepares five public liver CT datasets in one standard format, with fixed patient-level splits, so that every baseline and new method is trained and tested under the same conditions.

| Dataset | Role | Phases |
|---|---|---|
| MCT-LTDiag | train / validation / internal test | NC, AP, PVP, DP |
| PLC-CECT | train / validation / internal test | NC, AP, PVP, DP |
| WAW-TACE | external test | NC, AP, PVP, DP (patients with all four) |
| LiTS | external test | PVP only |
| HCC-TACE-Seg | external test | pre-TACE visit; phases identified from the images |

The three multiphase datasets are converted with Yifan Li's public [LiverPreprocessing](https://github.com/lyf7374/LiverPreprocessing) code (run by us, not rewritten). This folder adds what the MICCAI plan needs on top: converters for LiTS and HCC-TACE-Seg, a data check, the new splits, leak-free test crops, and exports for the baselines.

- Plan and status: [PLAN.md](PLAN.md)
- Output format: [STANDARD.md](STANDARD.md)
- Dataset settings: [configs/datasets.yaml](configs/datasets.yaml)

Raw data and processed outputs are **not** stored in this repository; their locations are set in the config.
