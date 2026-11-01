#!/usr/bin/env python3
"""Convert TCIA Colorectal-Liver-Metastases DICOM (CT + SEG) to NIfTI at native resolution, plus the inventory.

Reads the folders written by data_pipeline/download/tcia_crlm.py. Per patient: the CT series with SimpleITK
(HU, int16), the SEG with highdicom, one mask per segment on the CT grid (segments overlap: the liver holds the
tumours and the veins, so a single label map would lose voxels). The SEG frames must lie exactly on the CT slices
they reference (checked here, because the SEG files don't declare it). No resampling, cropping or clipping here.

Output (inside --raw):
  nifti/<patient>/ct.nii.gz                       int16 HU, native grid
  nifti/<patient>/liver.nii.gz, liver_remnant.nii.gz, hepatic_vein.nii.gz, portal_vein.nii.gz   uint8 0/1
  nifti/<patient>/tumours.nii.gz                  uint8, 0 background, k = SEG segment "Tumor_k"
  nifti/<patient>/info.json                       inventory row of this patient
  inventory_TCIA-CRLM.csv                         all patients (PLAN.md Step 1)
  failed_conversion.csv                           only when patients fail

A patient folder appears only when complete, so a rerun skips finished patients. The log shows counts only.
Usage: python -I -u tcia_crlm.py --raw <project>/data/TCIA-CRLM [--limit N]
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import highdicom as hd
import numpy as np
import pydicom
import SimpleITK as sitk

ORGANS = {"Liver": "liver", "Liver Remnant": "liver_remnant", "Hepatic": "hepatic_vein", "Portal": "portal_vein"}
TUMOUR = re.compile(r"Tumor_(\d+)")
MAX_OFFSET_MM = 0.01  # SEG frame position vs its CT slice
MIN_FREE_BYTES = 30e9  # never fill the project storage completely


def floats(values):
    return np.array([float(v) for v in values])


def read_ct(ct_dir):
    """CT image (int16 HU) and the slice headers in image order."""
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(ct_dir))
    headers = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    ct = reader.Execute()
    if ct.GetPixelID() != sitk.sitkInt16:  # e.g. float when the rescale slope is not 1
        arr = np.round(sitk.GetArrayFromImage(ct))
        if arr.min() < -32768 or arr.max() > 32767:
            raise ValueError(f"HU outside int16: {arr.min()}..{arr.max()}")
        img = sitk.GetImageFromArray(arr.astype(np.int16))
        img.CopyInformation(ct)
        ct = img
    return ct, headers


def check_geometry(seg, headers, ct_series_uid):
    """Raise unless every SEG frame lies on the CT slice it references, with the CT's orientation and pixel grid."""
    if not any(r.SeriesInstanceUID == ct_series_uid for r in seg.ReferencedSeriesSequence):
        raise ValueError("SEG does not reference this CT series")
    shared = seg.SharedFunctionalGroupsSequence[0]
    if not np.allclose(floats(shared.PlaneOrientationSequence[0].ImageOrientationPatient),
                       floats(headers[0].ImageOrientationPatient), atol=1e-4):
        raise ValueError("SEG orientation differs from CT")
    if not np.allclose(floats(shared.PixelMeasuresSequence[0].PixelSpacing), floats(headers[0].PixelSpacing)):
        raise ValueError("SEG pixel spacing differs from CT")
    if (seg.Rows, seg.Columns) != (headers[0].Rows, headers[0].Columns):
        raise ValueError("SEG rows/columns differ from CT")
    position = {h.SOPInstanceUID: floats(h.ImagePositionPatient) for h in headers}
    worst = 0.0
    for frame in seg.PerFrameFunctionalGroupsSequence:
        ref = frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
        if ref not in position:
            raise ValueError("SEG frame references a slice outside the CT series")
        worst = max(worst, np.abs(floats(frame.PlanePositionSequence[0].ImagePositionPatient) - position[ref]).max())
    if worst > MAX_OFFSET_MM:
        raise ValueError(f"SEG frame {worst:.3f} mm away from its CT slice")
    return worst


def convert(patient_dir, out_dir):
    """Write one patient's NIfTI files into out_dir; return its inventory row."""
    ct_dir, seg_dir = next(patient_dir.glob("CT_*")), next(patient_dir.glob("SEG_*"))
    ct, headers = read_ct(ct_dir)
    seg = hd.seg.segread(next(seg_dir.glob("*.dcm")))
    offset = check_geometry(seg, headers, ct_dir.name.split("_", 1)[1])
    uids = [h.SOPInstanceUID for h in headers]

    def mask(segment_number):  # (slices, rows, cols) uint8 in CT order; slices without frames are empty
        return seg.get_pixels_by_source_instance(
            source_sop_instance_uids=uids, segment_numbers=[segment_number], combine_segments=False,
            assert_missing_frames_are_empty=True, ignore_spatial_locations=True)[..., 0].astype(np.uint8)

    def write(arr, name):
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(ct)
        sitk.WriteImage(img, str(out_dir / name), useCompression=True)

    sitk.WriteImage(ct, str(out_dir / "ct.nii.gz"), useCompression=True)
    labels = {n: str(seg.get_segment_description(n).segment_label) for n in seg.segment_numbers}
    unknown = [lab for lab in labels.values() if lab not in ORGANS and not TUMOUR.fullmatch(lab)]
    if unknown:
        raise ValueError(f"unknown SEG labels {unknown}")
    row = {}
    liver = None
    for n, lab in labels.items():
        if lab in ORGANS:
            m = mask(n)
            write(m, f"{ORGANS[lab]}.nii.gz")
            row[f"{ORGANS[lab]}_voxels"] = int(m.sum())
            liver = m if lab == "Liver" else liver
    tumours = np.zeros(sitk.GetArrayViewFromImage(ct).shape, np.uint8)
    overlap = 0
    for n, lab in labels.items():
        if TUMOUR.fullmatch(lab):
            m = mask(n).astype(bool)
            overlap += int((m & (tumours > 0)).sum())  # counted, then the higher number wins
            tumours[m] = int(TUMOUR.fullmatch(lab).group(1))
    write(tumours, "tumours.nii.gz")

    z = np.array([floats(h.ImagePositionPatient) @ np.cross(*floats(headers[0].ImageOrientationPatient).reshape(2, 3))
                  for h in headers])
    steps = np.diff(z)
    ct_arr = sitk.GetArrayViewFromImage(ct)
    sx, sy, sz = ct.GetSpacing()
    row = {"patient_id": patient_dir.name, "ct_series_uid": ct_dir.name.split("_", 1)[1],
           "slices": ct.GetSize()[2], "rows": ct.GetSize()[1], "cols": ct.GetSize()[0],
           "spacing_x_mm": round(sx, 4), "spacing_y_mm": round(sy, 4), "spacing_z_mm": round(sz, 4),
           "slice_thickness_mm": float(headers[0].SliceThickness),
           "z_step_spread_mm": round(float(steps.max() - steps.min()), 4) if len(steps) else 0.0,
           "hu_min": int(ct_arr.min()), "hu_max": int(ct_arr.max()),
           "manufacturer": str(getattr(headers[0], "Manufacturer", "")),
           "seg_frames": int(seg.number_of_frames), "seg_max_offset_mm": round(float(offset), 4),
           **row,
           "n_tumours": sum(bool(TUMOUR.fullmatch(lab)) for lab in labels.values()),
           "tumour_voxels": int((tumours > 0).sum()), "tumour_overlap_voxels": overlap,
           "tumour_outside_liver_voxels": int(((tumours > 0) & (liver == 0)).sum()) if liver is not None else -1}
    (out_dir / "info.json").write_text(json.dumps(row, indent=1))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True, help="TCIA-CRLM raw folder, e.g. <project>/data/TCIA-CRLM")
    ap.add_argument("--limit", type=int, help="only the first N patients: test run")
    args = ap.parse_args()
    nifti = args.raw / "nifti"
    nifti.mkdir(exist_ok=True)
    patients = sorted(p for p in (args.raw / "dicom").iterdir() if p.is_dir())
    if args.limit:
        patients = patients[:args.limit]
    todo = [p for p in patients if not (nifti / p.name).exists()]
    st = os.statvfs(args.raw)
    print(f"{len(patients)} patients, {len(patients) - len(todo)} already converted, {len(todo)} to do "
          f"| free: {st.f_bavail * st.f_frsize / 1e9:.0f} GB, {st.f_ffree:,} files")
    if st.f_bavail * st.f_frsize < MIN_FREE_BYTES or st.f_ffree < 10 * len(todo) + 1000:
        sys.exit("Not enough free space or files on the project storage: stopping.")

    failed, t_run = [], time.time()
    for k, p in enumerate(todo, 1):
        t0, tmp = time.time(), nifti / (p.name + ".part")
        shutil.rmtree(tmp, ignore_errors=True)  # leftover of an interrupted run
        tmp.mkdir()
        try:
            row = convert(p, tmp)
        except Exception as e:  # unreadable DICOM, geometry mismatch, unknown labels
            shutil.rmtree(tmp, ignore_errors=True)
            failed.append((p.name, f"{type(e).__name__}: {e}"))
            print(f"[{k:3d}/{len(todo)}] FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        os.replace(tmp, nifti / p.name)
        print(f"[{k:3d}/{len(todo)}] ok: {row['slices']} slices, {row['spacing_z_mm']} mm, {row['n_tumours']} tumours, "
              f"tumour voxels outside liver {row['tumour_outside_liver_voxels']}, {time.time() - t0:.1f} s", flush=True)

    rows = [json.loads((nifti / p.name / "info.json").read_text()) for p in patients if (nifti / p.name).exists()]
    if rows:
        with open(args.raw / "inventory_TCIA-CRLM.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"done: {len(rows)}/{len(patients)} patients converted, {sum(r['n_tumours'] for r in rows)} tumours, "
          f"{sum(r['tumour_overlap_voxels'] > 0 for r in rows)} patients with overlapping tumours, "
          f"{sum(r['tumour_outside_liver_voxels'] > 0 for r in rows)} with tumour voxels outside the liver mask; "
          f"this run {len(todo) - len(failed)} in {(time.time() - t_run) / 60:.0f} min")
    failed_csv = args.raw / "failed_conversion.csv"
    if failed:
        with open(failed_csv, "w", newline="") as f:
            csv.writer(f).writerows([("patient_id", "error"), *failed])
        sys.exit(f"{len(failed)} patients failed (listed in failed_conversion.csv).")
    failed_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
