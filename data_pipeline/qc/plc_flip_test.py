#!/usr/bin/env python3
"""Decide the slice order of the PLC-CECT volumes that LiverPreprocessing stage 0 leaves unflipped (Step 4 QC).

Why: stage 0 flips 1,442 of 1,444 volumes; for the other 2 ("weakly confirmed") two quick signals disagree (air at
the bottom says upside down; the liver's shape says one of them is the right way up). The decision (flip or not) must
not make a correct volume wrong, so this test asks the anatomy directly.

What: for every volume with k_axis_flip false in data/PLC-CECT/fixed/<patient>/geometry.json, TotalSegmentator (total
task, same weights as the liver masks) segments liver, heart, lower lung lobes, kidneys and vertebrae T10 / L1 / L3 /
L5 on the volume as stored and on a copy with the slice order reversed (scratch disk only; data/ is not changed).
Files are LPS, so the head is at high z. Checks per version: heart and lungs above the liver centre, kidneys below it,
and each pair of the four vertebrae in head-to-foot order. The version that passes more checks is the right way up.

Output: <out>/flip_test.csv (one row per volume and version; patient IDs stay on disk). Prints counts only.
Usage (envs/totalseg, GPU): python -u plc_flip_test.py --data <project>/data/PLC-CECT/fixed --out <project>/outputs/plc_check
"""
import argparse
import csv
import json
import tempfile
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROI = ["liver", "heart", "lung_lower_lobe_left", "lung_lower_lobe_right", "kidney_left", "kidney_right",
       "vertebrae_T10", "vertebrae_L1", "vertebrae_L3", "vertebrae_L5"]
VERTEBRAE = ["vertebrae_T10", "vertebrae_L1", "vertebrae_L3", "vertebrae_L5"]  # head to foot
MIN_VOXELS = 200


def organ_z(seg_path, names):
    """Centre z (mm) of each organ found with >= MIN_VOXELS voxels."""
    seg = sitk.ReadImage(str(seg_path))
    a = sitk.GetArrayFromImage(seg)
    out = {}
    for label, name in names.items():
        if name not in ROI:
            continue
        idx = np.argwhere(a == label)
        if len(idx) >= MIN_VOXELS:
            z_index = idx[:, 0].mean()
            out[name] = seg.TransformContinuousIndexToPhysicalPoint((0.0, 0.0, float(z_index)))[2]
    return out


def checks(z):
    """List of (check, passed) for the organs present."""
    res = []
    if "liver" in z:
        above = [z[n] for n in ("heart", "lung_lower_lobe_left", "lung_lower_lobe_right") if n in z]
        below = [z[n] for n in ("kidney_left", "kidney_right") if n in z]
        if above:
            res.append(("heart/lungs above liver", np.mean(above) > z["liver"]))
        if below:
            res.append(("kidneys below liver", np.mean(below) < z["liver"]))
    present = [v for v in VERTEBRAE if v in z]
    for upper, lower in combinations(present, 2):
        res.append((f"{upper[-3:]} above {lower[-2:]}", z[upper] > z[lower]))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="data/PLC-CECT/fixed")
    ap.add_argument("--out", type=Path, required=True, help="output folder (flip_test.csv)")
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    import torch
    from totalsegmentator.map_to_binary import class_map
    from totalsegmentator.python_api import totalsegmentator
    torch.set_num_threads(args.threads)
    names = class_map["total"]
    volumes = []
    for d in sorted(p for p in args.data.iterdir() if p.is_dir() and not p.name.endswith(".part")):
        for ph, g in json.loads((d / "geometry.json").read_text())["phases"].items():
            if not g.get("k_axis_flip_stage0", g["k_axis_flip"]):
                volumes.append((d.name, ph))
    print(f"{len(volumes)} volumes that stage 0 leaves unflipped", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k, (pid, ph) in enumerate(volumes, 1):
            ct = sitk.ReadImage(str(args.data / pid / f"ct_{ph}.nii.gz"))
            flipped = sitk.GetImageFromArray(sitk.GetArrayFromImage(ct)[::-1].copy())
            flipped.CopyInformation(ct)
            score = {}
            for version, img in (("as_stored", ct), ("slices_reversed", flipped)):
                src, seg = tmp / f"{version}.nii.gz", tmp / f"{version}_seg.nii.gz"
                sitk.WriteImage(img, str(src))
                t0 = time.time()
                totalsegmentator(str(src), str(seg), ml=True, roi_subset=ROI, device="gpu", quiet=True,
                                 nr_thr_resamp=args.threads, nr_thr_saving=args.threads)
                z = organ_z(seg, names)
                res = checks(z)
                passed = sum(p for _, p in res)
                liver_ml = float((sitk.GetArrayFromImage(sitk.ReadImage(str(seg))) ==
                                  next(l for l, n in names.items() if n == "liver")).sum()
                                 * np.prod(ct.GetSpacing()) / 1000)
                score[version] = passed / len(res) if res else float("nan")
                rows.append({"patient": pid, "phase": ph, "version": version, "organs_found": len(z),
                             "checks": len(res), "passed": passed, "liver_ml": round(liver_ml, 1),
                             "failed_checks": "; ".join(c for c, p in res if not p),
                             "seconds": round(time.time() - t0, 1)})
                print(f"[{k}/{len(volumes)}] {ph} {version}: {len(z)} organs, {passed}/{len(res)} checks passed, "
                      f"liver {liver_ml:.0f} ml, {time.time() - t0:.0f} s", flush=True)
            verdict = ("right way up as stored" if score["as_stored"] > score["slices_reversed"] else
                       "upside down as stored (flip it)" if score["slices_reversed"] > score["as_stored"]
                       else "undecided")
            print(f"[{k}/{len(volumes)}] {ph}: {verdict}", flush=True)
            for r in rows[-2:]:
                r["verdict"] = verdict
    with (args.out / "flip_test.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(volumes)} volumes tested")


if __name__ == "__main__":
    main()
