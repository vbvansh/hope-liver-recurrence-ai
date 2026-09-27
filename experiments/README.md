# Experiments

```text
experiments/
└── registration/
    └── simpleitk_baseline/     lightweight CPU-only pipeline built for the 1 GB JupyterHub disk
        ├── 01_registration_basics.ipynb     one patient, step by step (learning)
        ├── run_batch_registration.py        all HCC-TACE-Seg patients, one at a time
        └── results/
            ├── 01_registration_basics/      notebook outputs (HCC_002)
            └── 02_batch_registration/       batch outputs: summary.md, summary.png, per-patient CSV
```

The main registration experiments use the team pipeline in [`../processing/`](../processing/) directly (run on Lightning AI).

| Experiment | Pipeline | Dataset | Status |
|---|---|---|---|
| Registration basics (1 patient) | simpleitk_baseline | HCC-TACE-Seg (HCC_002) | done |
| E1 batch registration baseline | simpleitk_baseline | HCC-TACE-Seg (95 eligible) | 91 done, 4 to re-run |
| Team pipeline pilot (3 patients, Colab) | processing/ | HCC-TACE-Seg | done: runs; found stacked-phase + phase-mismatch problems |
| Team pipeline full run, with stacked-phase fix | processing/ | HCC-TACE-Seg (HCC_001–HCC_070) | running on Lightning AI |
| E2 compare baseline vs team registration | both | HCC-TACE-Seg | after the team pipeline run |
