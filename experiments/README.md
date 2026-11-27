# Experiments

Every model experiment for the MICCAI 2027 paper lives here, one folder per experiment, with one row in `tracker.csv`. A folder is self-contained, so it can be shared with a teammate as it is.

> **Every result in the tracker and in the paper is scored with the HOPE shared scorer in [`../evaluation/`](../evaluation/) (team rules fixed 26 Nov 2026). Never report numbers from another scoring script.**

```text
experiments/
├── README.md               this file: rules and how to start an experiment
├── tracker.csv             one row per experiment (opens in Excel / Google Sheets)
├── TEMPLATE.md             the experiment card; copied into each new folder as README.md
├── new_experiment.py       creates EXP-<NNN>_<slug>/ from the template and adds the tracker row
├── EXP-001_<slug>/
│   ├── README.md           experiment card: question, change, setup, commands, results, conclusion
│   ├── config.yaml         settings (optional)
│   ├── *.py, *.sh          code and job scripts specific to this experiment
│   └── results/            small result tables only (metrics.csv, froc.csv)
└── registration/           earlier registration experiments (see the end of this file)
```

## Start an experiment

```bash
cd <project path>/hope-liver-recurrence-ai
python3 experiments/new_experiment.py "Reproduce nnU-Net baseline" --owner YOUR_NAME
python3 experiments/new_experiment.py "Phase dropout 0.5" --owner YOUR_NAME --parent EXP-002
```

This creates the next free ID (`EXP-001`, `EXP-002`, …), the folder with its card and `results/`, and a `planned` row in `tracker.csv`. Then fill in the card, run, and copy the main numbers into the tracker row.

## Rules

1. **One question, ideally one change, per experiment.** Name the parent experiment it changes, so every gain can be attributed.
2. **Develop on validation only.** The internal test set (170 patients) and the external sets (release v2) are evaluated only with a frozen protocol, and every opening is recorded in the `test_opened` column (date and reason). Never choose thresholds or settings on test or external data (CLAUDE.md "Never do").
3. **Same evaluation for everyone:** the same evaluation code version, the lesion-matching rule (decided 26 Nov 2026: a candidate hits a lesion when their 3D masks have IoU ≥ 0.1, matched one-to-one), FROC points (1/8–8 false positives per case) and size strata (< 20 mm with < 10 mm merged, 20–29, ≥ 30 mm), written in each card.
4. **Positive class (team rule, decided 26 Nov 2026; state it in every card):** at lesion level the positives are **every lesion numbered in `lesion_instances.nii.gz`: malignant (class 1) and haemangioma (class 2), exactly as released**. Ignore regions (class 3) are neutral: a detection on them is neither a hit nor a false alarm. At patient level: malignant diagnosis vs others (haemangioma patients and PLC-CECT controls are negatives). A "malignant only" result (class 1) may appear as a clearly labelled secondary line. In the external sets haemangiomas are not annotated, so detections on them count as false alarms. In release v1.1 this gives 87 lesions < 20 mm in the test set (78 malignant + 9 haemangioma) and 30 in validation (26 + 4).
5. **Negative results count:** write down what did not work and why; set the status to `failed` or `abandoned`, never delete the row.
6. **Reproducible:** each card records the git commit, config, seeds, data release (v1.1 / v2), exact commands and job IDs.
7. **Big files stay out of git:** predictions go to `outputs/experiments/<EXP_ID>/` and checkpoints to `checkpoints/<EXP_ID>/` on the project storage. `.gitignore` already blocks `outputs/`, `runs/`, `*.nii*`, `*.pt`, `*.pth`, `*.ckpt`.
8. **Public repo:** no patient IDs, no private paths, user names or host names, no images and no per-case tables in any committed file. Check before every commit.

## Tracker columns

| Column | Meaning |
|---|---|
| `exp_id`, `title`, `owner`, `status` | ID, short title, who runs it, planned / running / done / failed / abandoned |
| `date_started`, `date_finished`, `parent` | Dates (YYYY-MM-DD) and the experiment it builds on |
| `data_release`, `split_evaluated`, `positive_class`, `phases` | e.g. v1.1, validation, `malignant + haemangioma (as released)` (the team rule, rule 4), NC+AP+PVP+DP |
| `model`, `key_change`, `seeds` | Architecture, the one change, seeds run |
| `val_froc_mean_lt20` | FROC mean (7 points) for lesions < 20 mm (malignant + haemangioma), validation (all metric columns follow rule 4) |
| `val_sens_lt20_at_0.5fp`, `_at_1fp`, `_at_2fp` | Sensitivity for lesions < 20 mm at 0.5 / 1 / 2 false positives per case (Plan v2 headline: 0.5 and 1) |
| `val_froc_mean_all`, `val_ap_all`, `val_patient_auroc` | All lesions FROC mean, average precision, patient-level AUROC |
| `test_opened` | Empty, or date + reason when the test / external sets were evaluated |
| `conclusion`, `card` | One-line take-away; path to the card |

## Registration experiments (earlier work)

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

The MICCAI data pipeline that produced releases v1.1 and v2 is in [`../data_pipeline/`](../data_pipeline/).
