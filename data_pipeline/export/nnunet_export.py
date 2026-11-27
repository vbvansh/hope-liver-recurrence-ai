#!/usr/bin/env python3
"""Export release v1.1 as an nnU-Net v2 raw dataset (links, no copies) with the official split as fold 0.

What: writes <raw>/Dataset<ID>_<name>/ with
  imagesTr/<case_id>_0000..0003.nii.gz  symbolic links to the release images NC, AP, PVP, DP (4 channels, all "CT")
  labelsTr/<case_id>.nii.gz             link to lesion_label.nii.gz (0 background, 1 malignant, 2 haemangioma,
                                        3 ignore = nnU-Net's ignore label)
  imagesTs/<case_id>_0000..0003.nii.gz  the same for the internal test cases (prediction only; never trained on)
  dataset.json
and <preprocessed>/Dataset<ID>_<name>/splits_final.json with ONE fold: train = release train, val = release val.
A phase missing in the release (3 train/val and 2 test cases in v1.1) is written as a real all-zero int16 image on
the case's grid (team default for missing phases, PLAN.md open item 6, decided by the data lead 27 Nov 2026).
Only cases in the v1.1 split (splits.csv) are exported; excluded cases are left out. Rerunning skips existing
links; a link that points elsewhere is an error.

Why: nnU-Net needs its own folder layout; links keep the 30 GB release from being copied.

Usage: python -I data_pipeline/export/nnunet_export.py --release <processed/v1> --raw <nnUNet_raw>
           --preprocessed <nnUNet_preprocessed> [--dataset-id 501] [--name HOPELiverV11] [--limit N] [--test-only-cases]
Outputs: the folders above; a summary line (counts only).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import SimpleITK as sitk

PHASES = ["nc", "ap", "pvp", "dp"]


def link(src, dst):
    if os.path.islink(dst):
        if os.readlink(dst) != src:
            raise RuntimeError("existing link points elsewhere: %s" % os.path.basename(dst))
        return False
    if os.path.exists(dst):
        raise RuntimeError("a real file is in the way: %s" % os.path.basename(dst))
    os.symlink(src, dst)
    return True


def zero_image(ref_path, dst):
    """All-zero int16 image on the grid of ref_path (written as .part, then renamed)."""
    if os.path.exists(dst):
        return False
    ref = sitk.ReadImage(ref_path)
    img = sitk.GetImageFromArray(np.zeros(sitk.GetArrayViewFromImage(ref).shape, np.int16))
    img.CopyInformation(ref)
    part = dst[:-7] + ".part%d.nii.gz" % os.getpid()
    sitk.WriteImage(img, part, True)
    os.replace(part, dst)
    return True


def export_case(case_dir, case_id, img_dir, lab_dir):
    """Links (or zero images) for one case; returns (new links, zero-filled phases)."""
    new = filled = 0
    present = [p for p in PHASES if os.path.exists(os.path.join(case_dir, "image_%s.nii.gz" % p))]
    if "pvp" not in present and not present:
        raise RuntimeError("no image at all in a case")
    ref = os.path.join(case_dir, "image_%s.nii.gz" % ("pvp" if "pvp" in present else present[0]))
    for k, ph in enumerate(PHASES):
        src = os.path.join(case_dir, "image_%s.nii.gz" % ph)
        dst = os.path.join(img_dir, "%s_%04d.nii.gz" % (case_id, k))
        if ph in present:
            new += link(os.path.abspath(src), dst)
        else:
            filled += zero_image(ref, dst)
            if os.path.islink(dst):
                raise RuntimeError("missing phase must be a real zero image")
    if lab_dir:
        new += link(os.path.abspath(os.path.join(case_dir, "lesion_label.nii.gz")),
                    os.path.join(lab_dir, case_id + ".nii.gz"))
    return new, filled


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", required=True, help="release folder (processed/v1: cases/, splits/splits.csv)")
    ap.add_argument("--raw", required=True, help="nnUNet_raw folder")
    ap.add_argument("--preprocessed", required=True, help="nnUNet_preprocessed folder (for splits_final.json)")
    ap.add_argument("--dataset-id", type=int, default=501)
    ap.add_argument("--name", default="HOPELiverV11")
    ap.add_argument("--limit", type=int, default=0, help="only the first N train, N val and N test cases (test)")
    args = ap.parse_args()

    with open(os.path.join(args.release, "splits", "splits.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    by = {s: sorted(r["case_id"] for r in rows if r["split"] == s) for s in ("train", "val", "test")}
    if args.limit:
        # keep a case with a missing phase in the test selection, if there is one
        def pick(ids):
            miss = [c for c in ids if len([p for p in PHASES if os.path.exists(
                os.path.join(args.release, "cases", c, "image_%s.nii.gz" % p))]) < 4]
            return sorted(set(ids[: args.limit - 1] + (miss[:1] or ids[args.limit - 1: args.limit])))
        by = {s: pick(v) for s, v in by.items()}
    dname = "Dataset%03d_%s" % (args.dataset_id, args.name)
    root = os.path.join(args.raw, dname)
    dirs = {k: os.path.join(root, k) for k in ("imagesTr", "labelsTr", "imagesTs")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    new = filled = 0
    for split in ("train", "val", "test"):
        for c in by[split]:
            cd = os.path.join(args.release, "cases", c)
            if split == "test":
                n, z = export_case(cd, c, dirs["imagesTs"], None)
            else:
                n, z = export_case(cd, c, dirs["imagesTr"], dirs["labelsTr"])
            new, filled = new + n, filled + z
    n_tr = len(by["train"]) + len(by["val"])
    meta = {"name": dname,
            "description": "HOPE release v1.1 (MCT-LTDiag + PLC-CECT), 1 mm, liver crop, NC/AP/PVP/DP registered "
                           "to PVP; labels: malignant, haemangioma, ignore (nnU-Net ignore label)",
            "channel_names": {str(k): "CT" for k in range(4)},
            "labels": {"background": 0, "malignant": 1, "haemangioma": 2, "ignore": 3},
            "numTraining": n_tr, "file_ending": ".nii.gz", "overwrite_image_reader_writer": "SimpleITKIO"}
    with open(os.path.join(root, "dataset.json"), "w") as f:
        json.dump(meta, f, indent=2)
    pre = os.path.join(args.preprocessed, dname)
    os.makedirs(pre, exist_ok=True)
    with open(os.path.join(pre, "splits_final.json"), "w") as f:
        json.dump([{"train": by["train"], "val": by["val"]}], f, indent=1)
    counts = {k: len([x for x in os.listdir(d) if x.endswith(".nii.gz")]) for k, d in dirs.items()}
    print("%s: train %d, val %d (imagesTr %d files, labelsTr %d), test %d (imagesTs %d files); new links %d, "
          "zero-filled phases %d" % (dname, len(by["train"]), len(by["val"]), counts["imagesTr"], counts["labelsTr"],
                                     len(by["test"]), counts["imagesTs"], new, filled))
    ok = counts["imagesTr"] == 4 * n_tr and counts["labelsTr"] == n_tr and counts["imagesTs"] == 4 * len(by["test"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
