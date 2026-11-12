#!/usr/bin/env python3
"""Step 4b: re-register the flagged MCT-LTDiag cases with our method (PLAN.md open item 11, Option B, 2026-11-12).

Why: on the 1 mm grid, 105 of 517 MCT cases have a phase with liver Dice < 0.9 against PVP after the authors'
registration. A test on the 7 worst (job 7735635) showed real liver misalignment (breathing) that our PLC method fixes in
4 of 7 when judged inside the liver. Decision: re-register the flagged cases, then keep per phase whichever version
(authors' or ours) lines up the liver better, and mark a phase unusable where neither does (a later step).

What, per array task K: the flagged cases are read from <release>/manifest.csv (MCT-LTDiag, registration_failed true),
sorted by case id, and cases (K-1)*per_task+1 .. K*per_task are processed:
  1. prepare: the four phases from the tar (PVP header for every phase, decided 2026-11-10) and their TotalSegmentator
     livers (same header) are written to <work>/data/<patient>/ct_<phase>.nii.gz and <work>/livers/<patient>/;
  2. register: steps/register_plc.py's register_patient (liver-centre shift + the authors' default B-spline, 3
     resolutions, 20 mm grid) for NC, AP and DP onto PVP; transforms in <out>/<patient>/<phase>_{shift,to_pvp}.txt,
     QC rows (liver Dice, box and liver-only edge agreement, before = authors', after = ours) in <out>/qc/task_<K>.csv.
The prepared images have exactly the geometry Step 8 uses for MCT, so <work> can be deleted after the run.
Finished registrations are skipped. Logs show numbers only.
Usage: python -I -u register_mct.py --release <project>/processed/v1 --project <project> --work <project>/work/register_mct
           --out <project>/outputs/registration/MCT-LTDiag (--task K [--per-task 4] | --list) [--threads 4]
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_plc as RP  # noqa: E402
import standardise as S  # noqa: E402

PHASES = ["nc", "ap", "pvp", "dp"]


def flagged(release):
    m = pd.read_csv(release / "manifest.csv")
    m = m[(m.dataset == "MCT-LTDiag") & m.registration_failed.astype(bool)]
    return sorted(m.case_id.str.split("_", n=1).str[1])


def prepare(project, work, pid):
    data, livers = work / "data" / pid, work / "livers" / pid
    if (data / "ct_dp.nii.gz").exists() and (livers / "liver_dp.nii.gz").exists():
        return
    data.mkdir(parents=True, exist_ok=True)
    livers.mkdir(parents=True, exist_ok=True)
    imgs, _ = S.mct_images(project / "data" / "MCT-LTDiag" / "tars" / f"{pid}.tar")
    for ph in PHASES:
        sitk.WriteImage(imgs[ph], str(data / f"ct_{ph}.nii.gz"), useCompression=True)
        liver = sitk.ReadImage(str(project / "outputs" / "liver_masks" / "MCT-LTDiag" / pid / f"liver_{ph}.nii.gz"),
                               sitk.sitkUInt8)
        sitk.WriteImage(S.on_grid_of(liver, imgs[ph]), str(livers / f"liver_{ph}.nii.gz"), useCompression=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", type=int)
    g.add_argument("--list", action="store_true", help="print how many cases and tasks, then stop")
    ap.add_argument("--per-task", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    pats = flagged(args.release)
    n_tasks = -(-len(pats) // args.per_task)
    if args.list:
        print(f"{len(pats)} flagged MCT-LTDiag cases -> {n_tasks} tasks of {args.per_task}")
        return
    chosen = pats[(args.task - 1) * args.per_task: args.task * args.per_task]
    print(f"task {args.task} of {n_tasks}: cases {(args.task - 1) * args.per_task + 1}-"
          f"{(args.task - 1) * args.per_task + len(chosen)} of {len(pats)} flagged", flush=True)
    import itk  # after argument parsing (slow import)
    itk.MultiThreaderBase.SetGlobalMaximumNumberOfThreads(args.threads)
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(args.threads)
    (args.out / "qc").mkdir(parents=True, exist_ok=True)
    qc_path = args.out / "qc" / f"task_{args.task:04d}.csv"
    new = not qc_path.exists()
    counts, t0 = {}, time.time()
    with qc_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RP.QC_FIELDS)
        if new:
            w.writeheader()
        for i, pid in enumerate(chosen, 1):
            prepare(args.project, args.work, pid)
            for st in RP.register_patient(itk, pid, args.work / "data", args.work / "livers", args.out, w,
                                          args.threads, False, f"[{i}/{len(chosen)}]"):
                counts[st.split(":")[0]] = counts.get(st.split(":")[0], 0) + 1
            fh.flush()
    print(f"done: {len(chosen)} cases, {counts} registrations, {(time.time() - t0) / 60:.1f} min")
    if counts.get("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
