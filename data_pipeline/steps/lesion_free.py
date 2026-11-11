#!/usr/bin/env python3
"""Step 9: lesion-free (healthy) liver masks per phase for the standard case folders (STANDARD.md, PLAN.md).

Why: the method needs a map of liver that holds no annotated lesion. Decided by the team (Oct 2026): remove every
annotated region (lesion_label 1 malignant, 2 haemangioma, 3 ignore) plus a 5 mm exclusion margin around each; keep a
no-margin copy for the ablation.

What, per case (cases/<DATASET>_<patient>/, written by steps/standardise.py) and phase:
  lesion_free_nomargin_<phase>.nii.gz = liver_<phase> AND NOT (lesion_label > 0)
  lesion_free_<phase>.nii.gz          = liver_<phase> AND NOT (within 5 mm of lesion_label > 0)
uint8 0/1 on the case's 1 mm grid (Euclidean distance; 1 mm voxels, so 5 voxels). All phases share the PVP grid after
registration, so the lesion regions (drawn on PVP) apply to every phase. case.json gets a "lesion_free" entry
(margin, volumes per phase). Datasets that leave haemangiomas or cysts undrawn cannot have them removed (STANDARD.md).

Outputs: the 8 files above per case, and <release>/qc/lesion_free_<DATASET>.csv (one row per case and phase).
A case that already has all 8 files is skipped. Logs show counts only.
Usage: python -I -u lesion_free.py --release <project>/processed/v1 --dataset MCT-LTDiag|PLC-CECT [--workers 4] [--limit N]
"""
import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

PHASES = ["nc", "ap", "pvp", "dp"]
MARGIN_MM = 5.0
FIELDS = ["case_id", "phase", "liver_ml", "lesion_free_nomargin_ml", "lesion_free_ml", "removed_share"]


def write(arr, ref, path):
    im = sitk.GetImageFromArray(arr.astype(np.uint8))
    im.CopyInformation(ref)
    part = path.with_name(path.name.replace(".nii.gz", f".{os.getpid()}.part.nii.gz"))
    sitk.WriteImage(im, str(part), useCompression=True)
    os.replace(part, path)


def one_case(folder):
    """Write the masks of one case; returns QC rows, or None if it was done already."""
    outputs = [folder / f"lesion_free{s}_{ph}.nii.gz" for ph in PHASES for s in ("", "_nomargin")]
    record = json.loads((folder / "case.json").read_text())
    if all(p.exists() for p in outputs) and "lesion_free" in record:
        return None
    lab_img = sitk.ReadImage(str(folder / "lesion_label.nii.gz"))
    spacing = lab_img.GetSpacing()
    lesion = sitk.GetArrayFromImage(lab_img) > 0
    if lesion.any():
        near = ndimage.distance_transform_edt(~lesion, sampling=spacing[::-1]) <= MARGIN_MM
    else:
        near = lesion
    rows, info = [], {"margin_mm": MARGIN_MM, "removed_classes": [1, 2, 3], "phases": {}}
    voxel_ml = float(np.prod(spacing)) / 1000
    for ph in PHASES:
        liver_path = folder / f"liver_{ph}.nii.gz"
        if not liver_path.exists():  # a phase this case does not have (external sets)
            continue
        liver_img = sitk.ReadImage(str(liver_path))
        liver = sitk.GetArrayFromImage(liver_img) > 0
        nomargin = liver & ~lesion
        free = liver & ~near
        write(nomargin, liver_img, folder / f"lesion_free_nomargin_{ph}.nii.gz")
        write(free, liver_img, folder / f"lesion_free_{ph}.nii.gz")
        r = {"case_id": record["case_id"], "phase": ph, "liver_ml": round(liver.sum() * voxel_ml, 1),
             "lesion_free_nomargin_ml": round(nomargin.sum() * voxel_ml, 1),
             "lesion_free_ml": round(free.sum() * voxel_ml, 1),
             "removed_share": round(1 - free.sum() / liver.sum(), 4) if liver.any() else None}
        rows.append(r)
        info["phases"][ph] = {k: r[k] for k in ("liver_ml", "lesion_free_nomargin_ml", "lesion_free_ml")}
    record["lesion_free"] = info
    tmp = folder / f"case.json.{os.getpid()}.part"
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, folder / "case.json")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", type=Path, required=True, help="release folder with cases/ (e.g. processed/v1)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="first N cases only (test)")
    args = ap.parse_args()
    cases = sorted(p for p in (args.release / "cases").glob(f"{args.dataset}_*")
                   if p.is_dir() and not p.name.endswith(".part"))[: args.limit]
    print(f"{args.dataset}: {len(cases)} cases, {args.workers} workers", flush=True)
    (args.release / "qc").mkdir(parents=True, exist_ok=True)
    qc_path = args.release / "qc" / f"lesion_free_{args.dataset}.csv"
    new = not qc_path.exists()
    counts, t0 = {}, time.time()
    with qc_path.open("a", newline="") as fh, ProcessPoolExecutor(args.workers) as ex:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        futures = [ex.submit(one_case, c) for c in cases]
        for k, fut in enumerate(futures, 1):
            try:
                rows = fut.result()
                status = "skipped" if rows is None else "done"
                for r in rows or []:
                    w.writerow(r)
            except Exception as e:  # keep going; counted and shown
                status = "failed"
                print(f"[{k}/{len(cases)}] failed: {type(e).__name__}: {e}", flush=True)
            counts[status] = counts.get(status, 0) + 1
            if k % 50 == 0 or k == len(cases):
                fh.flush()
                print(f"[{k}/{len(cases)}] {counts}, {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"done: {counts}")
    if counts.get("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
