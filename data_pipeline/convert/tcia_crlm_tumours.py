#!/usr/bin/env python3
"""Write each TCIA-CRLM tumour segment as its own mask (decided 2026-11-05: overlapping tumours stay separate).

convert/tcia_crlm.py stored the tumours as one numbered map, in which a tumour lying wholly under another one
disappears (34 of 524). This reads the SEG files again, with the same geometry check, and writes per patient:
  nifti/<patient>/tumour_stack.nii.gz   uint8, 4D: one 0/1 volume per Tumor_k segment, on the grid of ct.nii.gz
  nifti/<patient>/tumour_stack.json     segment names in stack order and voxels per segment
A rerun skips patients that already have both files. The log shows counts only.
Usage: python -I -u tcia_crlm_tumours.py --raw <project>/data/TCIA-CRLM [--limit N]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import highdicom as hd
import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tcia_crlm as T  # noqa: E402  (convert/tcia_crlm.py: geometry check and segment names)


def stack_for(patient_dir, nifti_dir):
    """Write the tumour stack of one patient; return the number of tumour segments."""
    ct_dir, seg_dir = next(patient_dir.glob("CT_*")), next(patient_dir.glob("SEG_*"))
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(ct_dir))  # same slice order as ct.nii.gz
    headers = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    seg = hd.seg.segread(next(seg_dir.glob("*.dcm")))
    T.check_geometry(seg, headers, ct_dir.name.split("_", 1)[1])
    uids = [h.SOPInstanceUID for h in headers]
    ct = nib.load(nifti_dir / "ct.nii.gz")
    names, volumes = [], []
    for n in seg.segment_numbers:
        name = str(seg.get_segment_description(n).segment_label)
        if not T.TUMOUR.fullmatch(name):
            continue
        m = seg.get_pixels_by_source_instance(
            source_sop_instance_uids=uids, segment_numbers=[n], combine_segments=False,
            assert_missing_frames_are_empty=True, ignore_spatial_locations=True)[..., 0]
        m = np.transpose(m > 0, (2, 1, 0))  # (slices, rows, cols) -> nibabel's (x, y, z) of ct.nii.gz
        if m.shape != ct.shape:
            raise ValueError(f"segment {name} has shape {m.shape}, CT {ct.shape}")
        names.append(name)
        volumes.append(m.astype(np.uint8))
    order = sorted(range(len(names)), key=lambda i: int(T.TUMOUR.fullmatch(names[i]).group(1)))
    names, volumes = [names[i] for i in order], [volumes[i] for i in order]
    tmp = nifti_dir / "tumour_stack.part.nii.gz"
    nib.save(nib.Nifti1Image(np.stack(volumes, axis=3), ct.affine), tmp)
    os.replace(tmp, nifti_dir / "tumour_stack.nii.gz")
    (nifti_dir / "tumour_stack.json").write_text(json.dumps(
        {"segments": names, "voxels": [int(v.sum()) for v in volumes]}, indent=1))
    return len(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True, help="TCIA-CRLM raw folder, e.g. <project>/data/TCIA-CRLM")
    ap.add_argument("--limit", type=int, help="only the first N patients: test run")
    args = ap.parse_args()
    patients = sorted(p for p in (args.raw / "dicom").iterdir() if p.is_dir())
    if args.limit:
        patients = patients[:args.limit]
    done = lambda p: all((args.raw / "nifti" / p.name / f).exists()  # noqa: E731
                         for f in ("tumour_stack.nii.gz", "tumour_stack.json"))
    todo = [p for p in patients if not done(p)]
    print(f"{len(patients)} patients, {len(patients) - len(todo)} already done, {len(todo)} to do", flush=True)
    failed, total, t0 = [], 0, time.time()
    for k, p in enumerate(todo, 1):
        try:
            total += stack_for(p, args.raw / "nifti" / p.name)
        except Exception as e:  # unreadable SEG, geometry mismatch
            failed.append(p.name)
            print(f"[{k:3d}/{len(todo)}] FAILED: {type(e).__name__}: {e}", flush=True)
        if k % 25 == 0 or k == len(todo):
            print(f"[{k:3d}/{len(todo)}] {total} tumour segments written, {time.time() - t0:.0f} s", flush=True)
    print(f"done: {len(todo) - len(failed)} patients, {total} tumour segments, {len(failed)} failed")
    if failed:
        sys.exit(f"{len(failed)} patients failed.")


if __name__ == "__main__":
    main()
