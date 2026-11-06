#!/usr/bin/env python3
"""GPU test of TotalSegmentator's liver mask on one scan per dataset, before the full runs (PLAN.md Steps 5 and 8).

For the first patient of TCIA-CRLM, LiTS, MCT-LTDiag (PVP, from its tar) and WAW-TACE (portal phase): runs
TotalSegmentator (task total, liver only) on the GPU, times it, and compares its liver with the dataset's own liver
outline (Dice): MCT the authors' liver_mask_pvp, WAW the authors' TotalSegmentator organ mask, TCIA the Liver
segment, LiTS labels 1 + 2. PLC-CECT waits for the geometry recovery (its headers carry a placeholder 1 mm).
Runs in envs/totalseg. Writes the masks to --out; prints TotalSegmentator's step timings and one result line
per scan, no patient IDs.

Usage: python -I -u totalseg_check.py --data <project>/data --out <project>/outputs/totalseg_test
"""
import argparse
import os
import re
import sys
import tarfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from totalsegmentator.python_api import totalsegmentator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import mct_reader  # noqa: E402


def cases(data, tmp):
    """[(dataset, CT path, reference liver mask as a boolean array or None)] for one patient per dataset."""
    out = []
    tcia = sorted(p for p in (data / "TCIA-CRLM" / "nifti").iterdir() if p.is_dir())[0]
    out.append(("TCIA-CRLM", tcia / "ct.nii.gz", np.asarray(nib.load(tcia / "liver.nii.gz").dataobj) > 0))
    lits = data / "LiTS" / "Task03_Liver"
    out.append(("LiTS", lits / "imagesTr" / "liver_0.nii.gz",
                np.asarray(nib.load(lits / "labelsTr" / "liver_0.nii.gz").dataobj) > 0))
    table = mct_reader.patients(data / "MCT-LTDiag")
    tar = table.tar.iloc[0]
    ct_path = tmp / "mct_pvp.nii.gz"
    with tarfile.open(tar) as tf:
        member = next(m for m in tf.getmembers() if re.sub(r"^\./", "", m.name) == mct_reader.MEMBERS["pvp"])
        ct_path.write_bytes(tf.extractfile(member).read())
    ref = mct_reader.read(tar, ("liver",))["liver"]
    out.append(("MCT-LTDiag", ct_path, np.asarray(ref.dataobj) > 0))

    waw = data / "WAW-TACE"
    for ct in sorted(waw.glob("ct_scans_*/*/*_2_scan.nii.gz"), key=lambda p: int(p.parent.name)):
        pid = ct.parent.name
        organ = sorted(waw.glob(f"organ_masks_*/**/{pid}_2_total_segmentator*.nii.gz"))
        if organ:
            out.append(("WAW-TACE", ct, np.asarray(nib.load(organ[0]).dataobj) == 5))  # TotalSegmentator v2: 5 = liver
            break

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="raw data root, e.g. <project>/data")
    ap.add_argument("--out", type=Path, required=True, help="output folder for the liver masks")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    threads = len(os.sched_getaffinity(0))  # CPU cores this job may use (the GPU queue gives one slot per GPU)
    torch.set_num_threads(threads)
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU available: {torch.cuda.is_available()}"
          + (f", {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
          + f"; CPU cores usable: {threads} of {os.cpu_count()}", flush=True)
    for name, ct, ref in cases(args.data, args.out):
        torch.cuda.reset_peak_memory_stats()
        out = args.out / f"{name}_liver.nii.gz"
        t0 = time.time()
        print(f"--- {name}", flush=True)
        totalsegmentator(str(ct), str(out), ml=True, roi_subset=["liver"], device="gpu", quiet=False, verbose=True,
                         nr_thr_resamp=threads, nr_thr_saving=threads)
        seconds = time.time() - t0
        liver = np.asarray(nib.load(out).dataobj) > 0
        shape = nib.load(ct).shape
        dice = (2 * (liver & ref).sum() / (liver.sum() + ref.sum())) if liver.shape == ref.shape else float("nan")
        print(f"{name}: scan {shape}, {seconds:.0f} s, GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB, "
              f"liver {liver.sum() * np.prod(nib.load(ct).header.get_zooms()[:3]) / 1e3:.0f} ml, "
              f"Dice vs the dataset's own liver {dice:.3f}", flush=True)
    (args.out / "mct_pvp.nii.gz").unlink(missing_ok=True)
    print("done")


if __name__ == "__main__":
    main()
