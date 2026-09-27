#!/usr/bin/env python3
"""WAW-TACE longitudinal/multi-phase HCC-CT pipeline (dataset-specific).

Unlike the TCIA collections, WAW-TACE ships already as NIfTI: per patient up to
four CT phases `<pid>/<pid>_<phase>_scan.nii.gz` (phase 0 = native/pre, 1 = late
arterial, 2 = portal-venous, 3 = delayed) plus per-phase tumour masks
`tumor_masks_.../<pid>/<pid>_<phase>_<n>_tumor_seg.nrrd`. So there is no DICOM
step; we reuse the SHARED standardize.py helpers (1 mm RAS iso resample, rigid
candidate registration, body-Dice QC) and map the phase axis onto the standard
timepoint layout so segment.py / register.py work downstream unchanged:

  Processed/WAW-TACE/<pid>/
    000_baseline/   ct.nii.gz   tumor.nii.gz   (phase 0)
    001_followup_1/ ct.nii.gz   tumor.nii.gz   (phase 1)  ...
  visualization/<pid>.png   rows=phases, cols=3 levels, tumour outlined red

Each non-baseline phase is rigidly registered onto phase 0 (best of MI /
silhouette-z / centred by body-Dice, same as the DICOM path); the chosen
transform warps BOTH the CT (linear) and the tumour mask (nearest) so lesions
stay aligned in the registered frame — this is the CT+mask cross-phase product.

Usage:
  python3 process.py --n-debug 3
  python3 process.py --patients 10 102 142
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import SimpleITK as sitk
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import standardize as S   # shared helpers: resample_iso, register_to_baseline, ...

RAW = os.path.join(os.environ.get("CT_RAW", "/media/cbtil3/WhiteSD/CTRaw/Abdomen"), "WAW-TACE", "extracted")
MASK_ROOT = os.path.join(RAW, "tumor_masks_wawtace_v1_08_05_2024")
OUT = os.path.join(os.environ.get("CT_PROCESSED", "/media/cbtil3/WhiteSD/CTProcessed"), "WAW-TACE")
PHASE_NAME = {0: "native", 1: "arterial", 2: "portal", 3: "delayed"}


def std_image(nii_path):
    """NIfTI CT -> RAS+, 1mm iso, float32 HU (same standard as the DICOM path)."""
    img = sitk.ReadImage(nii_path, sitk.sitkFloat32)
    img = sitk.DICOMOrient(img, "RAS")
    img = S.resample_iso(img, S.ISO_MM)
    return sitk.Cast(img, sitk.sitkFloat32)


def load_phase_mask(pid, phase, ref_std):
    """Union of all tumour-seg NRRDs for (pid, phase), resampled onto ref_std's
    standardized grid (nearest). Returns a uint8 sitk mask or None."""
    pat = os.path.join(MASK_ROOT, str(pid))
    hits = sorted(glob.glob(os.path.join(pat, f"{pid}_{phase}_*_tumor_seg.nrrd")))
    if not hits:
        return None
    union = None
    for h in hits:
        m = sitk.ReadImage(h)
        m = sitk.Resample(m, ref_std, sitk.Transform(), sitk.sitkNearestNeighbor,
                          0, sitk.sitkUInt8)
        union = m if union is None else sitk.Or(union, m > 0)
    return union > 0


def phases_for(pid):
    out = []
    for ph in (0, 1, 2, 3):
        p = os.path.join(RAW, str(pid), f"{pid}_{ph}_scan.nii.gz")
        if os.path.exists(p):
            out.append((ph, p))
    return out


def write(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(img, path)


def render(pat_out, png, pid, phase_labels):
    stages = sorted(d for d in os.listdir(pat_out)
                    if os.path.isdir(os.path.join(pat_out, d)) and d[:1].isdigit())
    vols = []
    for st in stages:
        ct = os.path.join(pat_out, st, "ct.nii.gz")
        if not os.path.exists(ct):
            continue
        tm = os.path.join(pat_out, st, "tumor.nii.gz")
        mask = nib.load(tm).get_fdata() > 0 if os.path.exists(tm) else None
        vols.append((st, nib.load(ct).get_fdata().astype(np.float32), mask))
    if not vols:
        return False
    base = vols[0][1]
    ranges = [S.body_z_range(a) for _, a, _ in vols if a.shape == base.shape]
    z0 = max(r[0] for r in ranges); z1 = min(r[1] for r in ranges)
    if z1 - z0 < 30:
        z0, z1 = S.body_z_range(base)
    # bias levels toward any tumour-bearing slices so lesions are visible
    tz = [np.where(m.any(axis=(0, 1)))[0] for _, _, m in vols if m is not None and m.any()]
    if tz:
        c = int(np.median(np.concatenate(tz)))
        levels = [int(np.clip(c + d, z0, z1)) for d in (-25, 0, 25)]
    else:
        levels = [int(z0 + f * (z1 - z0)) for f in (0.35, 0.50, 0.65)]
    nrows = len(vols)
    fig = plt.figure(figsize=(3 * 2.8, nrows * 2.8 + 0.7), facecolor="#111111")
    fig.suptitle(f"{pid}  ·  WAW-TACE (multi-phase HCC CT, L40/W400, red=tumour)",
                 color="white", fontsize=11, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(nrows, 3, figure=fig, hspace=0.06, wspace=0.04,
                           top=0.93, bottom=0.02, left=0.12, right=0.98)
    for r, (st, arr, mask) in enumerate(vols):
        for c, z in enumerate(levels):
            ax = fig.add_subplot(gs[r, c]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(["lower", "mid", "upper"][c], color="white", fontsize=10)
            if c == 0:
                ax.set_ylabel(phase_labels.get(st, st), color="#cccccc", fontsize=8)
            ax.imshow(S.axial(arr, z), cmap="gray", vmin=S.WIN_LO, vmax=S.WIN_HI,
                      aspect="equal", interpolation="bilinear")
            if mask is not None:
                mz = S.axial(mask.astype(float), z)
                if mz.any():
                    ax.contour(mz, levels=[0.5], colors="red", linewidths=0.8)
    os.makedirs(os.path.dirname(png), exist_ok=True)
    fig.savefig(png, dpi=120, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    return True


def process(pid):
    phs = phases_for(pid)
    if len(phs) < 2:
        print(f"  {pid}: <2 phases, skip"); return None
    pat_out = os.path.join(OUT, str(pid))
    # build standardized phase images
    imgs = []
    for ph, p in phs:
        try:
            imgs.append((ph, std_image(p)))
        except Exception as e:
            print(f"  {pid} phase {ph}: ERROR {e}")
    if len(imgs) < 2:
        return None
    baseline = imgs[0][1]
    tps = []
    for i, (ph, img) in enumerate(imgs):
        label = "000_baseline" if i == 0 else f"{i:03d}_followup_{i}"
        reg = {"phase": ph, "stage": label, "registered": False, "body_dice": None}
        mask = load_phase_mask(pid, ph, img)   # mask on this phase's std grid
        if i == 0:
            out_img, out_mask = img, mask
        else:
            tx, _ = S.register_to_baseline(baseline, img)
            sz, _ = S.silhouette_z_init(baseline, img)
            geo = sitk.CenteredTransformInitializer(
                baseline, img, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY)
            cands = {"mi": tx, "silhouette_z": sz, "centred": geo}
            scored = {k: (t, S.body_dice(baseline, sitk.Resample(
                img, baseline, t, sitk.sitkLinear, -1024.0, sitk.sitkFloat32)))
                for k, t in cands.items()}
            how = max(scored, key=lambda k: scored[k][1])
            best_tx, dice = scored[how]
            out_img = sitk.Resample(img, baseline, best_tx, sitk.sitkLinear,
                                    -1024.0, sitk.sitkFloat32)
            out_mask = (sitk.Resample(sitk.Cast(mask, sitk.sitkUInt8), baseline,
                        best_tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
                        if mask is not None else None)
            reg.update(registered=True, body_dice=round(dice, 4), aligned_by=how)
        write(out_img, os.path.join(pat_out, label, "ct.nii.gz"))
        if out_mask is not None:
            write(out_mask, os.path.join(pat_out, label, "tumor.nii.gz"))
            reg["has_tumor"] = True
        tps.append(reg)
        print(f"  {pid}/{label} (phase {ph}, {PHASE_NAME.get(ph,'?')}): "
              f"dice={reg['body_dice']} tumor={'has_tumor' in reg}")
    labels = {t["stage"]: f"{t['stage'][:3]} {PHASE_NAME.get(t['phase'],'?')}"
              for t in tps}
    render(pat_out, os.path.join(OUT, "visualization", f"{pid}.png"), pid, labels)
    return {"patient": str(pid), "timepoints": tps}


def main():
    global RAW, MASK_ROOT, OUT
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--patients", nargs="*")
    g.add_argument("--n-debug", type=int)
    g.add_argument("--all", action="store_true")
    ap.add_argument("--input", default=RAW, help="extracted WAW-TACE raw dir")
    ap.add_argument("--output", default=OUT, help="processed output dir")
    args = ap.parse_args()
    RAW, OUT = args.input, args.output
    MASK_ROOT = os.path.join(RAW, "tumor_masks_wawtace_v1_08_05_2024")
    os.makedirs(OUT, exist_ok=True)

    all_pids = sorted((d for d in os.listdir(RAW)
                       if os.path.isdir(os.path.join(RAW, d)) and d.isdigit()),
                      key=int)
    if args.patients:
        pids = args.patients
    elif args.n_debug:
        # prefer patients that actually have a tumour mask, then most phases
        def score(p):
            return (os.path.isdir(os.path.join(MASK_ROOT, p)), len(phases_for(p)))
        pids = sorted(all_pids, key=score, reverse=True)[:args.n_debug]
    else:
        pids = all_pids

    summary = []
    for pid in pids:
        print(f"{pid}:")
        r = process(pid)
        if r:
            summary.append(r)
    with open(os.path.join(OUT, "metadata.json"), "w") as f:
        json.dump({"dataset": "WAW-TACE",
                   "standard": {"orientation": "RAS+", "spacing_mm": [1, 1, 1],
                                "values": "raw HU", "axis": "CT contrast phase "
                                "(0 native /1 arterial /2 portal /3 delayed) "
                                "mapped to timepoint folders; non-baseline phases "
                                "rigidly registered to phase 0; tumour.nii.gz "
                                "warped by the same transform"},
                   "patients": summary}, f, indent=2)
    print(f"WAW-TACE done: {len(summary)} patients")


if __name__ == "__main__":
    main()
