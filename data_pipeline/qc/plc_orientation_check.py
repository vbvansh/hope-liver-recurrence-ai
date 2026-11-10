#!/usr/bin/env python3
"""Check every corrected PLC-CECT volume for an upside-down scan or a failed liver mask (Step 4 / 5 QC).

Why: LiverPreprocessing stage 0 decides per volume whether to flip the slice order; on 2026-11-09 the only 2 of 1,444
volumes it leaves unflipped both looked upside down (lungs at the bottom, TotalSegmentator liver 31 ml or far too
large), while the same patients' other phases looked right. This check looks at all 1,444 volumes so that no other
error is missed before registration and standardisation.

What, per volume (data/PLC-CECT/fixed/<patient>/ct_<phase>.nii.gz):
  - share of air / lung voxels (<= -150 HU) in the central 200 x 200 pixels of the bottom and the top third of the
    slices; the lungs belong at the top (the files are LPS, slice index growing towards the head);
  - TotalSegmentator liver volume, and its ratio to the median over the patient's four phases;
  - Dice of the TotalSegmentator liver with the released PLC liver mask, where LiverPreprocessing does not flag that mask as bad;
  - the stage-0 flip, confidence and orientation source from geometry.json.
Flags: orientation_differs (this phase has more air at the bottom than at the top while the patient's other three
phases all show the opposite, or the other way round, by at least 0.05; "more air at the bottom" alone is common, e.g. bowel gas or a
scan ending below the lungs, and was seen in all 4 phases of a correctly flipped patient), liver_ratio outside
0.6-1.6, dice_released < 0.8. A flag means "look at it", not "wrong".

Outputs (--out): plc_check.csv (one row per volume; patient IDs stay on disk) and summary.md (counts only).
Usage: python -I plc_orientation_check.py --data <project>/data/PLC-CECT/fixed \
           --livers <project>/outputs/liver_masks/PLC-CECT --out <project>/outputs/plc_check [--workers 4] [--limit N]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

PHASES = ["nc", "ap", "pvp", "dp"]


def check_patient(args):
    data, livers, pid = args
    geo = json.loads((data / pid / "geometry.json").read_text())["phases"]
    rows = []
    for ph in PHASES:
        row = {"patient": pid, "phase": ph, "k_axis_flip": geo[ph]["k_axis_flip"],
               "spacing_z_confidence": geo[ph]["spacing_z_confidence"],
               "orientation_source": geo[ph]["orientation_source"]}
        ct = sitk.ReadImage(str(data / pid / f"ct_{ph}.nii.gz"))
        a = sitk.GetArrayViewFromImage(ct)
        c = a[:, 156:356, 156:356] <= -150
        n = c.shape[0] // 3
        row["air_bottom"] = round(float(c[:n].mean()), 3)
        row["air_top"] = round(float(c[-n:].mean()), 3)
        voxel_ml = float(np.prod(ct.GetSpacing())) / 1000
        ts = sitk.GetArrayFromImage(sitk.ReadImage(str(livers / pid / f"liver_{ph}.nii.gz"))) > 0
        row["liver_ml"] = round(float(ts.sum() * voxel_ml), 1)
        rel = data / pid / f"liver_{ph}.nii.gz"
        if rel.exists():
            r = sitk.GetArrayFromImage(sitk.ReadImage(str(rel))) > 0
            s = ts.sum() + r.sum()
            row["released_liver_ml"] = round(float(r.sum() * voxel_ml), 1)
            row["dice_released"] = round(float(2 * (ts & r).sum() / s), 4) if s else np.nan
        rows.append(row)
    med = np.median([r["liver_ml"] for r in rows])
    bottom = [r["air_bottom"] > r["air_top"] for r in rows]
    for i, r in enumerate(rows):
        r["liver_ratio"] = round(r["liver_ml"] / med, 3) if med else np.nan
        others = bottom[:i] + bottom[i + 1:]
        flags = []
        if len(set(others)) == 1 and bottom[i] != others[0] and abs(r["air_bottom"] - r["air_top"]) >= 0.05:
            flags.append("orientation_differs")
        if not 0.6 <= r["liver_ratio"] <= 1.6:
            flags.append("liver_ratio")
        if r.get("dice_released", 1) < 0.8:
            flags.append("dice_released")
        r["flags"] = "|".join(flags)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="data/PLC-CECT/fixed")
    ap.add_argument("--livers", type=Path, required=True, help="outputs/liver_masks/PLC-CECT")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="first N patients only (test)")
    args = ap.parse_args()
    pats = sorted(p.name for p in args.data.iterdir() if p.is_dir() and not p.name.endswith(".part"))[: args.limit]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(check_patient, [(args.data, args.livers, p) for p in pats]), 1):
            rows += res
            if i % 50 == 0 or i == len(pats):
                print(f"  {i}/{len(pats)} patients", flush=True)
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "plc_check.csv", index=False)
    flagged = df[df["flags"] != ""]
    lines = ["# PLC-CECT orientation and liver check", "",
             f"Volumes: {len(df)} ({df.patient.nunique()} patients); flagged: {len(flagged)} volumes in "
             f"{flagged.patient.nunique()} patients.", "",
             "| Flag | Volumes |", "|---|---|"]
    for f in ["orientation_differs", "liver_ratio", "dice_released"]:
        lines.append(f"| {f} | {int(df['flags'].str.contains(f).sum())} |")
    lines += ["", "Flag combinations by stage-0 flip:", "", "```",
              flagged.groupby(["k_axis_flip", "flags"]).size().to_string() if len(flagged) else "none", "```", "",
              f"Dice with the released liver (where present, {int(df.dice_released.notna().sum())} volumes): "
              f"median {df.dice_released.median():.3f}, < 0.8 in {int((df.dice_released < 0.8).sum())}."]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"done: {len(df)} volumes, {len(flagged)} flagged")


if __name__ == "__main__":
    main()
