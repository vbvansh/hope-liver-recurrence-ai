# HOPE — AI for Earlier Detection of Liver Cancer Recurrence

Barts Cancer Institute, Queen Mary University of London · 12-month proof-of-concept

HOPE investigates whether AI can detect HCC recurrence earlier by learning how a patient's liver changes across sequential contrast-enhanced CT scans, combined with longitudinal AFP blood-test trajectories.

Project page: https://vivek231.github.io/HOPE/

## Repository layout

```text
HOPE_Liver_Recurrence_AI/
├── docs/
│   ├── PROJECT_OVERVIEW.md                 project goals, concepts, roadmap
│   └── datasets/
│       ├── Dataset_Summary_Sep2026.pdf     in-house + public dataset summary slides
│       └── Three_Dataset_Statistics_EN.docx  MCT-LTDiag / PLC-CECT / WAW-TACE statistics
├── literature_review/
│   ├── research_papers/
│   │   ├── research_novelty_papers/        method and application papers
│   │   └── survey_papers/                  systematic reviews and surveys
│   └── sota_benchmarks/                    state-of-the-art result tables
├── experiments/
│   └── registration/
│       └── simpleitk_baseline/             lightweight CPU pipeline for the 1 GB hub
├── processing/                             CT preprocessing + longitudinal registration pipelines
└── README.md
```

## Where to start

1. [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) — what we are building and why.
2. [docs/datasets/](docs/datasets/) — which datasets are available and their statistics.
3. [processing/README.md](processing/README.md) — how raw CT is standardised, segmented and registered across timepoints (runs on the lab GPU server).
