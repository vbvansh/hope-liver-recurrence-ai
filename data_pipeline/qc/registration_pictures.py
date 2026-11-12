#!/usr/bin/env python3
"""Check pictures of phase alignment for cases of a release (Step 5 QC, open item 11 of PLAN.md).

Why: the liver-Dice flag alone cannot tell a real misalignment from a poor automatic liver mask (TotalSegmentator is
weaker on non-contrast scans). A person looks at the pictures to judge.

What: for the --worst cases with the lowest liver Dice of any phase, --moderate random flagged cases with a worst
phase between 0.85 and 0.9, and --reference random unflagged cases (seed 0), one picture per case: each phase (NC, AP,
PVP, DP) on the case's 1 mm grid, axial and coronal slice through the PVP liver centre, with the PVP liver outline in
green and that phase's own TotalSegmentator liver outline in red. Aligned phases: the outlines lie on top of each
other.

Output: <release>/qc/registration_pictures/<dataset>/<dataset>_<group>_<k>.png + pictures_index.csv (case, group, worst
phase and its Dice). Patient IDs only in the index file. Uses Pillow (no matplotlib in envs/hope).
Usage: python -I registration_pictures.py --release <project>/processed/v1 --dataset MCT-LTDiag [--worst 7]
           [--moderate 5] [--reference 2]
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image, ImageDraw
from scipy import ndimage

PHASES = ["nc", "ap", "pvp", "dp"]


def outline(mask):
    return mask & ~ndimage.binary_erosion(mask)


def picture(folder, out_png, title):
    g = lambda n: sitk.GetArrayFromImage(sitk.ReadImage(str(folder / n)))  # noqa: E731
    pvp_liver = g("liver_pvp.nii.gz") > 0
    z, y, _ = [int(v) for v in np.argwhere(pvp_liver).mean(axis=0)]
    cuts = (lambda a: a[z], lambda a: a[::-1, y, :])
    tiles = []
    for ph in PHASES:
        img, liver = g(f"image_{ph}.nii.gz"), g(f"liver_{ph}.nii.gz") > 0
        column = []
        for cut in cuts:
            grey = ((np.clip(cut(img), -200, 200) + 200) / 400 * 255).astype(np.uint8)
            rgb = np.repeat(grey[..., None], 3, axis=2)
            rgb[outline(cut(liver))] = (255, 40, 40)
            rgb[outline(cut(pvp_liver))] = (40, 255, 40)
            column.append(Image.fromarray(rgb))
        tiles.append(column)
    w = max(t.width for col in tiles for t in col)
    h = [max(col[r].height for col in tiles) for r in range(2)]
    sheet = Image.new("RGB", (4 * w, h[0] + h[1] + 34), "black")
    for i, (ph, col) in enumerate(zip(PHASES, tiles)):
        sheet.paste(col[0], (i * w, 18))
        sheet.paste(col[1], (i * w, 18 + h[0]))
        ImageDraw.Draw(sheet).text((i * w + 4, 3), f"{ph.upper()}", fill="white")
    ImageDraw.Draw(sheet).text((4, sheet.height - 14), f"{title}. Green = PVP liver, red = this phase's liver "
                               "(TotalSegmentator). Top axial, bottom coronal.", fill="white")
    sheet.save(out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--worst", type=int, default=7)
    ap.add_argument("--moderate", type=int, default=5)
    ap.add_argument("--reference", type=int, default=2)
    args = ap.parse_args()
    m = pd.read_csv(args.release / "manifest.csv")
    m = m[m.dataset == args.dataset].copy()
    dice = m[["liver_dice_nc", "liver_dice_ap", "liver_dice_dp"]]
    m["worst_dice"] = dice.min(axis=1)
    m["worst_phase"] = dice.idxmin(axis=1).str.replace("liver_dice_", "")
    rng = random.Random(0)
    worst = m.nsmallest(args.worst, "worst_dice")
    band = m[(m.worst_dice >= 0.85) & (m.worst_dice < 0.9)]
    moderate = band.loc[rng.sample(list(band.index), min(args.moderate, len(band)))]
    ok = m[~m.registration_failed.astype(bool)]
    reference = ok.loc[rng.sample(list(ok.index), min(args.reference, len(ok)))]
    out = args.release / "qc" / "registration_pictures" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for group, rows in (("worst", worst), ("moderate", moderate), ("reference", reference)):
        for k, r in enumerate(rows.itertuples(), 1):
            png = out / f"{args.dataset}_{group}_{k:02d}.png"
            picture(args.release / "cases" / r.case_id, png,
                    f"{group} {k}: worst phase {r.worst_phase.upper()} liver Dice {r.worst_dice:.3f}")
            index.append({"picture": png.name, "case_id": r.case_id, "group": group,
                          "worst_phase": r.worst_phase, "worst_dice": round(r.worst_dice, 4)})
    pd.DataFrame(index).to_csv(out / "pictures_index.csv", index=False)
    print(f"done: {len(index)} pictures in {out.name}/ (worst {len(worst)}, moderate {len(moderate)}, "
          f"reference {len(reference)}); worst-phase Dice of the worst group "
          f"{worst.worst_dice.min():.3f}-{worst.worst_dice.max():.3f}")


if __name__ == "__main__":
    main()
