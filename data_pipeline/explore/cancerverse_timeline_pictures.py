#!/usr/bin/env python3
"""Draw one timeline picture per CancerVerse patient: every scan date side by side, with the liver tumour outlines.

What: for each patient in the subset lists (cancerverse_subsets.py), reads every downloaded scan
(<ct-dir>/<cv_id>/ct.nii.gz + segmentations/liver_lesion.nii.gz), puts it in RAS orientation and draws one column
per date: top = coronal slice through the largest tumour (else the middle of the scan) with the tumour outlines of
the whole volume projected on it in red, to show where the scan starts and ends and whether the liver is in it;
bottom = axial slice with the largest tumour area that date, tumour outline in red (middle slice if no tumour).
Above each column: day since the first date, phase, number of lesions (cancerverse_subsets.py count), T / -.
Images in radiological view (patient's right on the left), liver window -160..240 HU, slices scaled to their
physical aspect.

Why: so the data lead can check by eye whether tumour outlines are right and whether "no tumour" dates are real
or scans without the liver, before any model uses this data.

Outputs: <out>/<subset>/<subset>_<NN>.png (patients numbered, no IDs in the picture) and <out>/<subset>/index.csv
(NN -> dataset patient ID, dates, course; disk only). Needs Pillow (envs/hope).

Usage: python -I cancerverse_timeline_pictures.py --lists <a.csv> [...] --ct-dir <dir> --out <dir> [--limit N]
"""
import argparse
import csv
import os

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

PANEL_H = 260
WINDOW = (-160.0, 240.0)


def load_ras(path):
    """Stored (unscaled) array in RAS orientation, its scaling (slope, inter) and spacing. Keeping the stored
    integer type and scaling only the drawn slices keeps memory near the file's own size."""
    img = nib.load(path)
    ornt = nib.orientations.io_orientation(img.affine)
    raw = np.asanyarray(img.dataobj.get_unscaled())
    slope, inter = img.dataobj.slope, img.dataobj.inter
    raw = nib.orientations.apply_orientation(raw, ornt)
    zooms = np.array(img.header.get_zooms()[:3], float)[ornt[:, 0].astype(int)]
    return raw, float(slope), float(inter), zooms


def to_gray(sl):
    lo, hi = WINDOW
    return (np.clip((sl - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def panel(gray, region, spacing_rc, height=PANEL_H):
    """gray / region (filled tumour area or None): 2D (rows, cols); spacing_rc: mm per row and per column.
    The image is scaled to its physical aspect; the region is scaled with nearest neighbour and its outline drawn
    after scaling, so thin outlines survive shrinking. Returns an RGB PIL image."""
    rows, cols = gray.shape
    h_mm, w_mm = rows * spacing_rc[0], cols * spacing_rc[1]
    width = max(1, int(round(height * w_mm / h_mm)))
    rgb = np.array(Image.fromarray(gray).resize((width, height), Image.BILINEAR).convert("RGB"))
    if region is not None and region.any():
        small = np.array(Image.fromarray(region.astype(np.uint8) * 255).resize((width, height), Image.NEAREST)) > 0
        if not small.any():  # a region smaller than one pixel after scaling: mark its position
            r, c = np.argwhere(region).mean(axis=0)
            small[int(r * height / rows), int(c * width / cols)] = True
        rgb[edges(small) | (small & ~ndimage.binary_erosion(small, iterations=1))] = (255, 40, 40)
    return Image.fromarray(rgb)


def edges(mask2d):
    return mask2d & ~ndimage.binary_erosion(mask2d)


def column(ct_path, mask_path):
    ct, slope, inter, z = load_ras(ct_path)
    mask = None
    if os.path.exists(mask_path):
        mk, ms, mi, _ = load_ras(mask_path)
        if mk.shape == ct.shape:
            mask = (mk > 0) if (ms > 0 and mi == 0) else (mk.astype(np.float32) * ms + mi > 0)
    if mask is not None and mask.any():
        zs = mask.sum(axis=(0, 1))
        k = int(zs.argmax())
        ys = mask.sum(axis=(0, 2))
        j = int(ys.argmax())
    else:
        k, j = ct.shape[2] // 2, ct.shape[1] // 2
    # RAS: axis0 -> right, axis1 -> anterior, axis2 -> superior. Radiological view: patient's right on the left.
    cor = ct[::-1, j, ::-1].T.astype(np.float32) * slope + inter   # rows: superior at top; cols: patient's right left
    ax = ct[::-1, ::-1, k].T.astype(np.float32) * slope + inter    # rows: anterior at top; cols: patient's right left
    cor_o = ax_o = None
    if mask is not None and mask.any():
        cor_o = mask.any(axis=1)[::-1, ::-1].T        # filled tumour areas, projected front to back
        ax_o = mask[::-1, ::-1, k].T
    top = panel(to_gray(cor), cor_o, (z[2], z[0]))
    bottom = panel(to_gray(ax), ax_o, (z[1], z[0]))
    return top, bottom


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lists", nargs="+", required=True)
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask", default="liver_lesion")
    ap.add_argument("--limit", type=int, default=0, help="patients per list (test)")
    args = ap.parse_args()
    font = ImageFont.load_default()
    done = missing = 0
    for lst in args.lists:
        name = os.path.splitext(os.path.basename(lst))[0]
        with open(lst, newline="") as f:
            rows = list(csv.DictReader(f))
        pats = []
        for r in rows:
            if r["patient"] not in pats:
                pats.append(r["patient"])
        pats = sorted(pats)[: args.limit] if args.limit else sorted(pats)
        od = os.path.join(args.out, name)
        os.makedirs(od, exist_ok=True)
        index = []
        for n, p in enumerate(pats, 1):
            scans = sorted((r for r in rows if r["patient"] == p), key=lambda r: r["date"])
            d0 = np.datetime64(scans[0]["date"])
            cols, heads = [], []
            for r in scans:
                ct_path = os.path.join(args.ct_dir, r["cv_id"], "ct.nii.gz")
                if not os.path.exists(ct_path):
                    missing += 1
                    continue
                top, bottom = column(ct_path, os.path.join(args.ct_dir, r["cv_id"], "segmentations",
                                                           args.mask + ".nii.gz"))
                cols.append((top, bottom))
                ph = {"portal_venous": "PV", "venous": "V", "arterial_early": "early A", "arterial_late": "late A",
                      "arterial": "A", "non-contrast": "NC"}.get(r["phase"], "?")
                heads.append("day %d | %s | %s lesions | %s" % (
                    int((np.datetime64(r["date"]) - d0).astype(int)), ph, r["n_lesions"],
                    "T" if r["outlined"] in ("True", "true", "1") else "-"))
            if not cols:
                continue
            pad, head_h = 8, 18
            w = sum(max(t.width, b.width) for t, b in cols) + pad * (len(cols) + 1)
            h = head_h + 2 * PANEL_H + 3 * pad + 16
            canvas = Image.new("RGB", (w, h), (0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((pad, h - 14), "%s_%02d: %d dates; top: coronal + tumour outlines of the whole scan (red); "
                      "bottom: axial slice with the largest tumour" % (name, n, len(cols)), fill=(200, 200, 200), font=font)
            x = pad
            for (top, bottom), head in zip(cols, heads):
                cw = max(top.width, bottom.width)
                draw.text((x, 3), head, fill=(255, 255, 0), font=font)
                canvas.paste(top, (x, head_h + pad))
                canvas.paste(bottom, (x, head_h + 2 * pad + PANEL_H))
                x += cw + pad
            out_png = os.path.join(od, "%s_%02d.png" % (name, n))
            canvas.save(out_png + ".part.png")
            os.replace(out_png + ".part.png", out_png)
            index.append({"picture": os.path.basename(out_png), "patient": p, "dates": len(cols),
                          "course": " ".join("T" if r["outlined"] in ("True", "true", "1") else "-" for r in scans)})
            done += 1
            print("%s: %d / %d pictures" % (name, n, len(pats)), flush=True)
        with open(os.path.join(od, "index.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["picture", "patient", "dates", "course"])
            wr.writeheader()
            wr.writerows(index)
    print("done: %d pictures, %d scans missing on disk" % (done, missing), flush=True)


if __name__ == "__main__":
    main()
