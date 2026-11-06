#!/usr/bin/env python3
"""Overlay pictures of the standard lesion labels (PLAN.md Step 2) for a visual check, N random patients per dataset.

For each picked patient and each phase with lesions: the axial slice where the lesions cover the most area,
shown twice side by side (CT alone, CT with labels): malignant red, haemangioma blue, ignore yellow. CT is shown
at -200..200 HU (PLC-CECT as stored, 0-255). Images are in radiological view (patient's right on the left).
Patients are picked with a fixed seed among those with at least one lesion; file names carry a running number,
the number-to-patient list is written next to the pictures (it stays on the cluster).

Usage: python -I label_overlays.py --dataset MCT-LTDiag --data <project>/data --inventory <project>/outputs/inventory
         --out <project>/outputs/label_overlays [--n 4] [--seed 0]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import labels as L  # noqa: E402
import mct_reader  # noqa: E402

COLOURS = {1: (230, 57, 70), 2: (66, 135, 245), 3: (240, 200, 40)}


def labels_for(dataset, data, pid, row):
    """(Labels list, function giving the CT array of a Labels' grid) for one patient."""
    if dataset == "MCT-LTDiag":
        tar = data / "MCT-LTDiag" / "tars" / f"{pid}.tar"
        return L.mct(tar, row.diagnosis), lambda lab: np.asarray(mct_reader.read(tar, ("pvp",))["pvp"].dataobj)
    if dataset == "PLC-CECT":
        return L.plc(data / "PLC-CECT", pid, row.diagnosis), lambda lab: np.asarray(lab.image.dataobj)
    if dataset == "WAW-TACE":
        cts, masks = L.waw_files(data / "WAW-TACE")
        return L.waw(cts, masks, pid), lambda lab: np.asarray(lab.image.dataobj)
    if dataset == "TCIA-CRLM":
        return L.tcia(data / "TCIA-CRLM" / "nifti" / pid), lambda lab: np.asarray(lab.image.dataobj)
    raw = data / "LiTS" / "Task03_Liver"
    return (L.lits(raw / "imagesTr" / f"{pid}.nii.gz", raw / "labelsTr" / f"{pid}.nii.gz"),
            lambda lab: np.asarray(lab.image.dataobj))


def to_display(arr, affine):
    """Reorient a 3D array to RAS (closest axes)."""
    ornt = nib.orientations.io_orientation(affine)
    return nib.orientations.apply_orientation(arr, ornt)


def panel(ct, lab, dataset):
    """RGB image: CT slice and the same slice with labels, side by side (radiological view)."""
    view = lambda a: np.flip(a.T, axis=(0, 1))  # noqa: E731  RAS (x, y) -> rows anterior at top, patient's right left
    if dataset == "PLC-CECT":
        grey = np.clip(ct, 0, 255).astype(np.uint8)
    else:
        grey = ((np.clip(ct, -200, 200) + 200) / 400 * 255).astype(np.uint8)
    grey, lab = view(grey), view(lab)
    base = np.stack([grey] * 3, axis=-1).astype(float)
    over = base.copy()
    for cls, colour in COLOURS.items():
        m = lab == cls
        if m.any():
            over[m] = 0.55 * over[m] + 0.45 * np.array(colour)
            edge = m & ~ndimage.binary_erosion(m)
            over[edge] = colour
    img = np.concatenate([base, np.full((base.shape[0], 8, 3), 255.0), over], axis=1).astype(np.uint8)
    return Image.fromarray(img)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=["MCT-LTDiag", "PLC-CECT", "WAW-TACE", "TCIA-CRLM", "LiTS"])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diagnosis", help="only this diagnosis, e.g. HH (pictures go to <dataset>_<diagnosis>)")
    args = ap.parse_args()
    inv = pd.read_csv(args.inventory / f"inventory_{args.dataset}.csv", dtype={"patient_id": str},
                      keep_default_na=False)
    has = inv[(inv[["lesions_class1", "lesions_class2", "lesions_class3"]].astype(int).sum(axis=1) > 0)
              & ~inv.notes.str.contains("BLOCK")]
    if args.diagnosis:
        has = has[has.diagnosis == args.diagnosis]
    picked = has.sample(n=min(args.n, len(has)), random_state=args.seed)
    out = args.out / (f"{args.dataset}_{args.diagnosis}" if args.diagnosis else args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for i, row in enumerate(picked.itertuples(), 1):
        found, ct_of = labels_for(args.dataset, args.data, row.patient_id, row)
        for lab in found:
            ct = to_display(ct_of(lab), lab.image.affine)
            lb = to_display(lab.label, lab.image.affine)
            area = ((lb == 1) | (lb == 2)).sum(axis=(0, 1))
            z = int(area.argmax()) if area.max() > 0 else int((lb == 3).sum(axis=(0, 1)).argmax())
            img = panel(ct[:, :, z], lb[:, :, z], args.dataset)
            ImageDraw.Draw(img).text((6, 6), f"{args.dataset} #{i}  {lab.phase}  slice {z}  "
                                             f"lesions {len(lab.lesions)}", fill=(255, 255, 0))
            name = f"{args.dataset}_{i:02d}_{lab.phase}.png"
            img.save(out / name)
            counts = {c: int(((lb == c).sum(axis=(0, 1)) > 0).sum()) for c in (1, 2, 3)}
            index.append({"picture": name, "patient_id": row.patient_id, "phase": lab.phase,
                          "lesions": len(lab.lesions), "slices_with_class1": counts[1],
                          "slices_with_class2": counts[2], "slices_with_class3": counts[3]})
            print(f"#{i} {lab.phase}: {len(lab.lesions)} lesions, classes on slices {counts}", flush=True)
    with open(out / "pictures_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    print(f"done: {len(index)} pictures in {out}")


if __name__ == "__main__":
    main()
