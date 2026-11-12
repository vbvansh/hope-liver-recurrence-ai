#!/usr/bin/env python3
"""Step 11: patient-level train / validation / test split of the development data (MCT-LTDiag + PLC-CECT).

Why: every model (baselines and our method) must train, tune and test on the same patients. PLAN.md decision:
patient level 70 / 10 / 20, fixed seed, stratified by dataset x tumour type x lesion count (0 / 1 / >= 2) x
largest-lesion size bin; PLC-CECT controls spread proportionally. External sets never enter the split.

What: reads <release>/manifest.csv (one row per case = patient; excluded cases left out, registration-flagged cases kept)
and <release>/lesions.csv (lesions in the release, for the counts).
  Strata: cases are grouped at the finest level that gives a group of at least --min-stratum cases:
    L0 dataset | diagnosis | lesion_count_group | max_diameter_bin
    L1 dataset | diagnosis | lesion_count_group | coarse bin (< 20 mm, >= 20 mm, none)
    L2 dataset | diagnosis | coarse bin
    L3 dataset | diagnosis (every remaining case)
  Assignment: within each stratum the cases (sorted by case id) are shuffled with numpy RandomState(seed); validation
  and test counts follow running totals of 10 % and 20 % over the strata in a fixed order, so the overall sizes are
  70 / 10 / 20 after rounding. Same inputs + same seed -> identical files.
Outputs (<release>/splits/): splits.csv (case_id, dataset, diagnosis, stratum_level, stratum, split), train.txt,
val.txt, test.txt (case ids), split_summary.md (counts only: patients per split by dataset, tumour type, lesion count,
size bin; lesions per class and size stratum; malignant lesions < 10 mm per split = Plan v2's reporting-rule count).
Usage: python -I split.py --release <project>/processed/v1 [--seed 0] [--min-stratum 10]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FRACTIONS = {"val": 0.1, "test": 0.2}
COARSE = {"<10": "<20", "10-19": "<20", "20-29": ">=20", ">=30": ">=20", "none": "none"}
SMALL_MIN_INTERNAL = 50  # Plan v2: the < 10 mm stratum is reported only with >= 50 such lesions in the internal test


def stratum_keys(m):
    g = m.lesion_count_group.astype(str)
    b = m.max_diameter_bin.fillna("none").astype(str)
    base = m.dataset + "|" + m.diagnosis.astype(str)
    return [("L0", base + "|" + g + "|" + b), ("L1", base + "|" + g + "|" + b.map(COARSE).fillna(b)),
            ("L2", base + "|" + b.map(COARSE).fillna(b)), ("L3", base)]


def assign_strata(m, min_n):
    level, stratum = pd.Series("", index=m.index), pd.Series("", index=m.index)
    left = pd.Series(True, index=m.index)
    for name, key in stratum_keys(m):
        sizes = key[left].value_counts()
        ok = left & key.map(sizes).ge(min_n if name != "L3" else 0).fillna(False)
        level[ok], stratum[ok] = name, key[ok]
        left &= ~ok
    return level, stratum


def split(m, seed, min_n):
    m = m.sort_values("case_id").reset_index(drop=True)
    m["stratum_level"], m["stratum"] = assign_strata(m, min_n)
    m["split"] = "train"
    rng = np.random.RandomState(seed)
    acc = {k: 0.0 for k in FRACTIONS}
    given = {k: 0 for k in FRACTIONS}
    for s in sorted(m.stratum.unique()):
        idx = m.index[m.stratum == s].to_numpy()
        idx = idx[rng.permutation(len(idx))]
        start = 0
        for k, f in FRACTIONS.items():
            acc[k] += f * len(idx)
            n = int(np.floor(acc[k] + 0.5)) - given[k]
            n = max(0, min(n, len(idx) - start))
            m.loc[idx[start:start + n], "split"] = k
            given[k] += n
            start += n
    return m


def summary(m, les):
    order = ["train", "val", "test"]
    lines = ["# Step 11 split (MCT-LTDiag + PLC-CECT)", "",
             f"Patients: {len(m)}; " + ", ".join(f"{k} {int((m.split == k).sum())}" for k in order)
             + f" ({', '.join(f'{(m.split == k).mean() * 100:.1f} %' for k in order)}).",
             "Strata used: " + ", ".join(f"{lv} {n}" for lv, n in m.drop_duplicates('stratum').stratum_level.value_counts()
                                          .sort_index().items()), ""]

    def table(title, col):
        t = pd.crosstab([m.dataset, m[col].fillna("none")], m.split).reindex(columns=order, fill_value=0)
        return [f"## {title}", "", "| Dataset | " + col + " | train | val | test |", "|---|---|---|---|---|"] + [
            f"| {d} | {v} | " + " | ".join(str(int(x)) for x in row) + " |" for (d, v), row in t.iterrows()] + [""]

    lines += table("Patients by tumour type", "diagnosis")
    lines += table("Patients by lesion count", "lesion_count_group")
    lines += table("Patients by largest-lesion size bin", "max_diameter_bin")
    lr = les[les.in_release.astype(bool)].merge(m[["case_id", "split"]], on="case_id")
    cls = pd.crosstab([lr.dataset, lr["class"]], lr.split).reindex(columns=order, fill_value=0)
    lines += ["## Lesions in the release by class (1 malignant, 2 haemangioma)", "",
              "| Dataset | class | train | val | test |", "|---|---|---|---|---|"]
    lines += [f"| {d} | {c} | " + " | ".join(str(int(x)) for x in row) + " |" for (d, c), row in cls.iterrows()]
    mal = lr[lr["class"] == 1]
    st = pd.crosstab(mal.size_stratum, mal.split).reindex(columns=order, fill_value=0).reindex(
        ["<10", "10-19", "20-29", ">=30"], fill_value=0)
    lines += ["", "## Malignant lesions by in-plane size (native scan)", "", "| Size | train | val | test |",
              "|---|---|---|---|"] + [f"| {s} | " + " | ".join(str(int(x)) for x in row) + " |" for s, row in st.iterrows()]
    small_test = int(st.loc["<10", "test"])
    lines += ["", f"Plan v2 < 10 mm rule: internal test holds {small_test} malignant lesions < 10 mm "
                  f"({'meets' if small_test >= SMALL_MIN_INTERNAL else 'below'} the {SMALL_MIN_INTERNAL} needed"
                  + ("" if small_test >= SMALL_MIN_INTERNAL else ": the stratum is merged into < 20 mm for the internal test")
                  + ")."]
    missing = []
    for d, g in m.groupby("dataset"):
        for dx in sorted(g.diagnosis.unique()):
            for k in order:
                if not ((g.diagnosis == dx) & (g.split == k)).any():
                    missing.append(f"{d} {dx} not in {k}")
    lines += ["", "Every tumour type of each dataset in every split: "
              + ("yes" if not missing else "NO: " + "; ".join(missing))]
    return "\n".join(lines) + "\n", missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-stratum", type=int, default=10)
    args = ap.parse_args()
    m = pd.read_csv(args.release / "manifest.csv")
    m = m[(m.role == "development") & ~m.excluded.astype(bool)]
    les = pd.read_csv(args.release / "lesions.csv")
    m = split(m[["case_id", "dataset", "diagnosis", "lesion_count_group", "max_diameter_bin"]].copy(), args.seed,
              args.min_stratum)
    out = args.release / "splits"
    out.mkdir(parents=True, exist_ok=True)
    m[["case_id", "dataset", "diagnosis", "stratum_level", "stratum", "split"]].to_csv(out / "splits.csv", index=False)
    for k in ("train", "val", "test"):
        (out / f"{k}.txt").write_text("\n".join(m.case_id[m.split == k]) + "\n")
    text, missing = summary(m, les)
    (out / "split_summary.md").write_text(text)
    print(text)
    print(f"done: {len(m)} patients split (seed {args.seed})")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
