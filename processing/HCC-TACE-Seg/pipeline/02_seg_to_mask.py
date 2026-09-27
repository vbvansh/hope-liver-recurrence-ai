#!/usr/bin/env python3
"""HCC-TACE-Seg ships expert DICOM-SEG masks (segments: Liver, Mass=tumour,
Portal vein, Abdominal aorta) on the PRE-TACE CT. This turns the SEG into a
NIfTI label volume on the SAME standardized grid as the processed baseline CT
(Processed/HCC-TACE-Seg/<pid>/000_baseline/ct.nii.gz) so the lesion overlays
the registered images — the CT+mask product the user asked for.

No pydicom_seg dependency (it pins pydicom<3): we read the SEG with plain
pydicom and reconstruct a physical-space sitk image from the per-frame
ImagePositionPatient / SharedFunctionalGroups geometry, then resample onto the
baseline grid (nearest). SimpleITK keeps everything in LPS world coordinates, so
the raw-DICOM-built SEG and the baseline image (sitk-read + DICOMOrient RAS,
which preserves world points) share a frame of reference -> a plain
reference-image resample lines them up.

Writes:
  Processed/HCC-TACE-Seg/<pid>/000_baseline/tumor.nii.gz   (binary Mass)
  Processed/HCC-TACE-Seg/<pid>/000_baseline/organs.nii.gz  (multi-label all segs)
  Processed/HCC-TACE-Seg/visualization/<pid>_mask.png      (overlay montage)

Usage:
  python3 seg_to_mask.py --all
  python3 seg_to_mask.py --patients HCC_001 HCC_078
"""
import argparse
import glob
import os
import sys

import numpy as np
import pydicom
import SimpleITK as sitk
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import standardize as S

META = os.path.join(os.environ.get("CT_RAW", "/media/cbtil3/WhiteSD/CTRaw/Abdomen"),
                    "HCC-TACE-Seg ", "hcc-tace-seg_metadata_series.csv")  # raw dir name has a trailing space
OUT = os.path.join(os.environ.get("CT_PROCESSED", "/media/cbtil3/WhiteSD/CTProcessed"), "HCC-TACE-Seg")
TUMOR_LABELS = ("mass", "tumor", "tumour", "lesion")


def seg_file_for(pid, meta):
    rows = meta[(meta.PatientID == pid) & (meta.Modality == "SEG")]
    for _, r in rows.iterrows():
        f = glob.glob(os.path.join(r.SeriesDir, "*.dcm"))
        f = [x for x in f if not os.path.basename(x).startswith("._")]
        if f:
            return f[0]
    return None


def build_seg_image(seg_path):
    """DICOM-SEG -> (multilabel sitk image in LPS world, {segnum: label_name}).
    Each segment gets its SegmentNumber as the voxel value (last-wins on overlap,
    tumour kept by re-stamping it on top)."""
    ds = pydicom.dcmread(seg_path)
    seg_names = {int(s.SegmentNumber): str(getattr(s, "SegmentLabel", s.SegmentNumber))
                 for s in ds.SegmentSequence}
    frames = ds.pixel_array
    if frames.ndim == 2:
        frames = frames[None]
    sfg = ds.SharedFunctionalGroupsSequence[0]
    ps = [float(x) for x in sfg.PixelMeasuresSequence[0].PixelSpacing]  # [row, col]
    iop = [float(x) for x in sfg.PlaneOrientationSequence[0].ImageOrientationPatient]
    row_dir = np.array(iop[:3]); col_dir = np.array(iop[3:])
    slice_dir = np.cross(row_dir, col_dir)

    # per-frame: (segment number, position, frame index)
    rec = []
    for i, pf in enumerate(ds.PerFrameFunctionalGroupsSequence):
        seg_n = int(pf.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        pos = np.array([float(x) for x in pf.PlanePositionSequence[0].ImagePositionPatient])
        rec.append((seg_n, pos, i))
    # build the slice grid from unique positions along the slice normal
    proj = np.array([np.dot(p, slice_dir) for _, p, _ in rec])
    uniq = np.unique(np.round(proj, 2))
    uniq.sort()
    nz = len(uniq)
    dz = float(np.median(np.diff(uniq))) if nz > 1 else \
        float(sfg.PixelMeasuresSequence[0].SliceThickness)
    rows, cols = frames.shape[1], frames.shape[2]

    vol = np.zeros((nz, rows, cols), dtype=np.uint8)
    # stamp non-tumour first, tumour last so it wins on overlap
    order = sorted(rec, key=lambda r: seg_names.get(r[0], "").lower()
                   in TUMOR_LABELS)
    origin = None
    for seg_n, pos, i in order:
        # look the slice up with the same np.round used to build `uniq`: Python's
        # round() can disagree on the last decimal (e.g. -223.905) -> KeyError
        k = int(np.searchsorted(uniq, np.round(np.dot(pos, slice_dir), 2)))
        m = frames[i] > 0
        vol[k][m] = seg_n
        if k == 0:
            origin = pos
    if origin is None:
        origin = rec[int(np.argmin(proj))][1]

    img = sitk.GetImageFromArray(vol)            # (z,y,x)
    img.SetSpacing((ps[1], ps[0], abs(dz)))      # (col, row, slice)
    d = np.column_stack([row_dir, col_dir, slice_dir]).flatten()
    img.SetDirection([float(x) for x in d])
    img.SetOrigin([float(x) for x in origin])
    return img, seg_names


def overlay(pid, base_ct, tumor, organs):
    base = nib.load(base_ct).get_fdata().astype(np.float32)
    tm = nib.load(tumor).get_fdata() > 0 if os.path.exists(tumor) else np.zeros_like(base, bool)
    org = nib.load(organs).get_fdata() if os.path.exists(organs) else np.zeros_like(base)
    if tm.any():
        c = int(np.median(np.where(tm.any(axis=(0, 1)))[0]))
    else:
        z0, z1 = S.body_z_range(base); c = (z0 + z1) // 2
    zs = [int(np.clip(c + d, 0, base.shape[2] - 1)) for d in (-20, 0, 20)]
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.4), facecolor="k")
    for j, z in enumerate(zs):
        ax[j].imshow(S.axial(base, z), cmap="gray", vmin=S.WIN_LO, vmax=S.WIN_HI)
        oz = S.axial(org, z)
        for lab in np.unique(oz):
            if lab == 0:
                continue
            ax[j].contour((oz == lab).astype(float), levels=[0.5],
                          colors=["#33aaff"], linewidths=0.6)
        tz = S.axial(tm.astype(float), z)
        if tz.any():
            ax[j].contour(tz, levels=[0.5], colors="red", linewidths=1.0)
        ax[j].axis("off"); ax[j].set_title(f"z={z}", color="w", fontsize=8)
    fig.suptitle(f"{pid}  baseline + DICOM-SEG (red=tumour, blue=organs)",
                 color="w", fontsize=10)
    plt.tight_layout()
    png = os.path.join(OUT, "visualization", f"{pid}_mask.png")
    os.makedirs(os.path.dirname(png), exist_ok=True)
    fig.savefig(png, dpi=120, facecolor="k"); plt.close(fig)
    return png


def process(pid, meta):
    base_ct = os.path.join(OUT, pid, "000_baseline", "ct.nii.gz")
    if not os.path.exists(base_ct):
        print(f"  {pid}: no processed baseline, skip"); return False
    seg_path = seg_file_for(pid, meta)
    if not seg_path:
        print(f"  {pid}: no SEG, skip"); return False
    seg_img, names = build_seg_image(seg_path)
    ref = sitk.ReadImage(base_ct)
    # multilabel organs on the baseline grid (written via sitk -> identical
    # space to ct.nii.gz, so the nib-based overlay/register steps line up).
    organs_img = sitk.Resample(seg_img, ref, sitk.Transform(),
                               sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    arr = sitk.GetArrayFromImage(organs_img)
    tum_nums = [n for n, nm in names.items() if nm.lower() in TUMOR_LABELS]
    tumor_img = sitk.Cast(sitk.GetImageFromArray(
        np.isin(arr, tum_nums).astype(np.uint8)), sitk.sitkUInt8)
    tumor_img.CopyInformation(organs_img)
    sitk.WriteImage(organs_img, os.path.join(OUT, pid, "000_baseline", "organs.nii.gz"))
    sitk.WriteImage(tumor_img, os.path.join(OUT, pid, "000_baseline", "tumor.nii.gz"))
    tumor = sitk.GetArrayFromImage(tumor_img)
    png = overlay(pid, base_ct,
                  os.path.join(OUT, pid, "000_baseline", "tumor.nii.gz"),
                  os.path.join(OUT, pid, "000_baseline", "organs.nii.gz"))
    print(f"  {pid}: segs={names} tumor_vox={int(tumor.sum())} -> {png}")
    return True


def main():
    global OUT
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--patients", nargs="*")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--meta", default=META, help="TCIA series metadata CSV (input)")
    ap.add_argument("--output", default=OUT, help="processed HCC-TACE-Seg dir (read + written)")
    args = ap.parse_args()
    OUT = args.output
    meta = pd.read_csv(args.meta, dtype=str).fillna("")
    if args.patients:
        pids = args.patients
    else:
        pids = sorted(d for d in os.listdir(OUT)
                      if d.startswith("HCC_") and os.path.isdir(os.path.join(OUT, d)))
    for pid in pids:
        print(f"{pid}:")
        process(pid, meta)


if __name__ == "__main__":
    main()
