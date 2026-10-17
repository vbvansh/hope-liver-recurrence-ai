# Liver CT data pipeline (MICCAI 2027)

Prepares five public liver CT datasets in one standard format, with fixed patient-level splits, so that every baseline and new method is trained and tested under the same conditions.

| Dataset | Role | Phases |
|---|---|---|
| MCT-LTDiag | train / validation / internal test | NC, AP, PVP, DP |
| PLC-CECT | train / validation / internal test | NC, AP, PVP, DP |
| WAW-TACE | external test | any subset of NC, AP, PVP, DP |
| TCIA Colorectal-Liver-Metastases | external test | PVP only |
| LiTS | external test | PVP only |

Requirements follow *MICCAI 2027 Plan v2* (Yifan, Oct 2026). Where useful, parts of Yifan Li's public [LiverPreprocessing](https://github.com/lyf7374/LiverPreprocessing) code are run (e.g. PLC-CECT geometry recovery) rather than rewritten.

- Plan and status: [PLAN.md](PLAN.md)
- Output format: [STANDARD.md](STANDARD.md)
- Dataset settings: [configs/datasets.yaml](configs/datasets.yaml)

Raw data and processed outputs are **not** stored in this repository; their locations are set in the config.
