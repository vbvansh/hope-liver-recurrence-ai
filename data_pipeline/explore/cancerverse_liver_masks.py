#!/usr/bin/env python3
"""Measure the liver tumour masks of CancerVerse straight from its label archive, without unpacking it.

What: streams CancerVerse_Label.tar.gz once (CancerVerse/CV_XXXXXXXX/segmentations/<13 masks>.nii.gz), opens
only the chosen mask files (default liver_lesion.nii.gz) in memory, and records per scan whether the mask is
empty, its grid, and every 3D connected lesion (26-connectivity) measured on the scan's own grid with our release
convention (lib/components.measure, diameters between voxel centres: volume, slices, in-plane and 3D max
diameter, equivalent diameter). Also counts every mask file name in the archive.

Why: the CancerVerse metadata does not say which scans carry liver tumour outlines. Unpacking the archive would
create ~317,000 files (more than our free file quota) and ~40 GB, so it is read as a stream instead.

Outputs (in --out, written as .part and renamed when complete; counts only in the log):
  scans.csv    one row per mask file: cv_id, mask, shape, spacing_mm, slice_axis, dtype, values, voxels, lesions
  lesions.csv  one row per lesion: cv_id, mask, lesion, voxels, volume_ml, slices, inplane_max_diameter_mm,
               max3d_diameter_mm, equivalent_diameter_mm
  members.csv  every mask file name in the archive and how often it occurs

Usage (job: data_pipeline/jobs/cancerverse_liver_masks.sh):
  python -I data_pipeline/explore/cancerverse_liver_masks.py --archive <tar.gz> --out <dir> [--workers 8]
  python -I data_pipeline/explore/cancerverse_liver_masks.py --archive <tar.gz> --out <dir> --limit 20   # test
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import gzip
import os
import sys
import tarfile
import time

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import components as C  # noqa: E402

SCAN_COLS = ["cv_id", "mask", "shape", "spacing_mm", "slice_axis", "dtype", "values", "voxels", "lesions"]
LESION_COLS = ["cv_id", "mask", "lesion", "voxels", "volume_ml", "slices", "inplane_max_diameter_mm",
               "max3d_diameter_mm", "equivalent_diameter_mm"]


def cv_id_of(path):
    return next((p for p in path.split("/") if p.startswith("CV_")), "")


def measure_mask(name, data):
    """Open one gzipped NIfTI mask from bytes; return its scan row and lesion rows."""
    raw = gzip.decompress(data)
    img = nib.Nifti1Image.from_bytes(raw)
    arr = np.asanyarray(img.dataobj)
    del raw
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    codes = nib.aff2axcodes(img.affine)
    slice_axis = next((i for i, c in enumerate(codes) if c in ("S", "I")), 2)
    nz = arr != 0
    voxels = int(nz.sum())
    values = sorted(np.unique(arr[nz]).tolist())[:5] if voxels else []
    mask = os.path.basename(name).replace(".nii.gz", "")
    cv = cv_id_of(name)
    lesion_rows = []
    if voxels:
        where = np.nonzero(nz)
        lo = [int(w.min()) for w in where]
        hi = [int(w.max()) + 1 for w in where]
        sub = nz[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        labels, n = ndimage.label(sub, structure=C.FULL)
        for k, box in enumerate(ndimage.find_objects(labels), 1):
            idx = np.argwhere(labels[box] == k) + [s.start for s in box] + lo
            m = C.measure(idx, spacing, slice_axis, centres=True)
            lesion_rows.append({"cv_id": cv, "mask": mask, "lesion": k, "voxels": m["voxels"],
                                "volume_ml": round(m["volume_mm3"] / 1000.0, 3), "slices": m["slices"],
                                "inplane_max_diameter_mm": m["inplane_max_diameter_mm"],
                                "max3d_diameter_mm": m["max3d_diameter_mm"],
                                "equivalent_diameter_mm": m["equivalent_diameter_mm"]})
    scan = {"cv_id": cv, "mask": mask, "shape": "x".join(map(str, arr.shape)),
            "spacing_mm": "x".join("%.3f" % s for s in spacing), "slice_axis": slice_axis, "dtype": str(arr.dtype),
            "values": " ".join(str(v) for v in values), "voxels": voxels, "lesions": len(lesion_rows)}
    return scan, lesion_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", required=True, help="CancerVerse_Label.tar.gz")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--masks", default="liver_lesion", help="comma-separated mask names to measure")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NSLOTS", "2")))
    ap.add_argument("--limit", type=int, default=0, help="stop after this many measured masks (test)")
    ap.add_argument("--force", action="store_true", help="redo even if the outputs exist")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    final = {k: os.path.join(args.out, k + ".csv") for k in ("scans", "lesions", "members")}
    if all(os.path.exists(p) for p in final.values()) and not args.force:
        print("outputs exist, nothing to do (use --force to redo)")
        return
    wanted = {m.strip() + ".nii.gz" for m in args.masks.split(",") if m.strip()}
    members = collections.Counter()
    scans, lesions, failed = [], [], 0
    t0 = time.time()
    max_inflight = 4 * args.workers
    print("archive %s, masks %s, workers %d, limit %d" % (os.path.basename(args.archive), sorted(wanted),
                                                           args.workers, args.limit), flush=True)

    def collect(futures):
        nonlocal failed
        for f in futures:
            try:
                s, ls = f.result()
                scans.append(s)
                lesions.extend(ls)
            except Exception as e:  # keep going; count failures (no IDs in the log)
                failed += 1
                print("failed mask: %s" % type(e).__name__, flush=True)

    submitted = 0
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool, tarfile.open(args.archive, "r|gz") as tf:
        pending = set()
        for m in tf:
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            members[base] += 1
            if base not in wanted or (args.limit and submitted >= args.limit):
                continue
            data = tf.extractfile(m).read()
            pending.add(pool.submit(measure_mask, m.name, data))
            submitted += 1
            if len(pending) >= max_inflight:
                done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                collect(done)
            if submitted % 1000 == 0:
                print("%d masks submitted, %d measured, %d non-empty, %.0f s" % (
                    submitted, len(scans), sum(1 for s in scans if s["voxels"]), time.time() - t0), flush=True)
            if args.limit and submitted >= args.limit:
                break
        done, _ = cf.wait(pending)
        collect(done)

    for key, rows, cols in (("scans", sorted(scans, key=lambda r: (r["cv_id"], r["mask"])), SCAN_COLS),
                            ("lesions", sorted(lesions, key=lambda r: (r["cv_id"], r["mask"], r["lesion"])),
                             LESION_COLS),
                            ("members", [{"name": k, "count": v} for k, v in sorted(members.items())],
                             ["name", "count"])):
        part = final[key] + ".part.%d" % os.getpid()
        with open(part, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        os.replace(part, final[key])
    nonempty = sum(1 for s in scans if s["voxels"])
    print("done: %d mask files measured, %d non-empty, %d lesions, %d failed; %d file names in the archive "
          "(%d distinct); %.0f s" % (len(scans), nonempty, len(lesions), failed, sum(members.values()),
                                     len(members), time.time() - t0), flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
