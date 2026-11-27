#!/usr/bin/env python3
"""Turn nnU-Net probability maps into scorer candidates: one candidate-ID map per case + scores.csv.

What: reads <case>.npz (key 'probabilities', shape (classes, z, y, x), written by nnU-Net with --npz / --save_
probabilities, already in the original image space = the release 1 mm grid). Lesion probability = sum of the
target classes (default 1 malignant + 2 haemangioma, the team rule). Candidates = 3D connected components
(26-connectivity) of lesion probability >= each level in --levels, from the highest level down; a component at a
lower level is added only when it does not touch an already accepted candidate, so candidates never overlap (the
ID-map format). Components smaller than --min-voxels are dropped. Score = highest lesion probability inside.
Default levels 0.5, 0.2, 0.05, 0.01, 0.002 and 15 voxels follow the team's detector report (21 Nov 2026); its exact
handling of nested components is not documented, so this non-overlap rule is our documented choice.

Why: the shared scorer (evaluation/scorer.py) needs candidates with scores, not a single mask.

Outputs (--out): <case>_candidates.nii.gz (uint16, geometry copied from the release image_pvp, or the first
present phase) and scores.csv (case_id, candidate_id, score, level, voxels). Rerunning skips finished cases.

Usage: python -I evaluation/candidates_from_nnunet.py --npz-dir <fold_0/validation> --release <processed/v1> --out <dir>
"""
import argparse
import csv
import glob
import os

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

FULL = np.ones((3, 3, 3), bool)


def candidates(prob, levels, min_voxels):
    out = np.zeros(prob.shape, np.uint16)
    rows = []
    k = 0
    for lev in sorted(levels, reverse=True):
        lab, n = ndimage.label(prob >= lev, structure=FULL)
        if not n:
            continue
        for j, box in enumerate(ndimage.find_objects(lab), 1):
            m = lab[box] == j
            nv = int(m.sum())
            if nv < min_voxels or (out[box][m] > 0).any():
                continue
            k += 1
            out[box][m] = k
            rows.append((k, float(prob[box][m].max()), lev, nv))
    return out, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="1,2", help="probability channels summed into the lesion probability")
    ap.add_argument("--levels", default="0.5,0.2,0.05,0.01,0.002")
    ap.add_argument("--min-voxels", type=int, default=15)
    args = ap.parse_args()
    classes = [int(c) for c in args.classes.split(",")]
    levels = [float(x) for x in args.levels.split(",")]
    os.makedirs(args.out, exist_ok=True)
    score_file = os.path.join(args.out, "scores.csv")
    done = set()
    if os.path.exists(score_file):
        with open(score_file, newline="") as f:
            done = {r["case_id"] for r in csv.DictReader(f)}
    files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    new = 0
    with open(score_file, "a", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["case_id", "candidate_id", "score", "level", "voxels"])
        for p in files:
            cid = os.path.basename(p)[:-4]
            dst = os.path.join(args.out, cid + "_candidates.nii.gz")
            if cid in done and os.path.exists(dst):
                continue
            prob = np.load(p)["probabilities"]
            lesion = prob[classes].astype(np.float32).sum(axis=0)
            cd = os.path.join(args.release, "cases", cid)
            ref = next(os.path.join(cd, "image_%s.nii.gz" % ph) for ph in ("pvp", "ap", "dp", "nc")
                       if os.path.exists(os.path.join(cd, "image_%s.nii.gz" % ph)))
            refimg = sitk.ReadImage(ref)
            if tuple(reversed(refimg.GetSize())) != lesion.shape:
                raise SystemExit("probabilities are not on the release grid for a case")
            cmap, rows = candidates(lesion, levels, args.min_voxels)
            img = sitk.GetImageFromArray(cmap)
            img.CopyInformation(refimg)
            part = dst[:-7] + ".part%d.nii.gz" % os.getpid()
            sitk.WriteImage(img, part, True)
            os.replace(part, dst)
            if rows:
                for r in rows:
                    w.writerow([cid, r[0], "%.6f" % r[1], r[2], r[3]])
            else:  # a row per case keeps the resume logic simple; candidate 0 never exists in a map
                w.writerow([cid, 0, "0", "", 0])
            f.flush()
            new += 1
    print("candidates written for %d new cases (%d probability files)" % (new, len(files)))


if __name__ == "__main__":
    main()
