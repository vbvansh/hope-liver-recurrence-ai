#!/usr/bin/env python3
"""Test the PLC registration method on the MCT-LTDiag cases the authors' registration aligns worst (PLAN.md open
item 11, Option B decided 2026-11-11). Nothing in the release changes; this only measures.

Why: on the 1 mm grid, 105 of 517 MCT cases have a phase with liver Dice < 0.9 against PVP after the authors'
registration. Before deciding whether to re-register MCT, test whether our method (steps/register_plc.py:
liver-centre shift + the authors' default B-spline) fixes the worst cases.

Subcommands:
  prepare  for the --n cases with the lowest worst-phase liver Dice in <release>/manifest.csv, write the four phases
           (from the tar, PVP header for every phase as decided 2026-11-10) to <work>/data/<patient>/ct_<phase>.nii.gz
           and their TotalSegmentator livers (same header) to <work>/livers/<patient>/liver_<phase>.nii.gz, in the
           layout steps/register_plc.py reads. Then run register_plc.py --data <work>/data --livers <work>/livers
           --out <work>/registration --task 1 --per-task <n> (jobs/mct_registration_test.sh does both).
  summary  compare per case and phase the liver Dice and gradient NCC before (= the authors' registration, phases
           overlaid as released) and after our registration, from <work>/registration/qc/task_0001.csv.
Logs show numbers only, never patient IDs.
Usage: python -I -u mct_registration_test.py prepare --release <project>/processed/v1 --project <project>
           --work <project>/work/mct_registration_test [--n 7]
       python -I -u mct_registration_test.py summary --work <project>/work/mct_registration_test
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "steps"))
import standardise as S  # noqa: E402  (mct_images, on_grid_of)

PHASES = ["nc", "ap", "pvp", "dp"]


def prepare(release, project, work, n):
    m = pd.read_csv(release / "manifest.csv")
    m = m[m.dataset == "MCT-LTDiag"].copy()
    m["worst_dice"] = m[["liver_dice_nc", "liver_dice_ap", "liver_dice_dp"]].min(axis=1)
    chosen = m.nsmallest(n, "worst_dice")
    for k, cid in enumerate(chosen.case_id, 1):
        pid = cid.split("_", 1)[1]
        data, livers = work / "data" / pid, work / "livers" / pid
        if (data / "ct_dp.nii.gz").exists() and (livers / "liver_dp.nii.gz").exists():
            print(f"[{k}/{n}] already prepared", flush=True)
            continue
        data.mkdir(parents=True, exist_ok=True)
        livers.mkdir(parents=True, exist_ok=True)
        imgs, _ = S.mct_images(project / "data" / "MCT-LTDiag" / "tars" / f"{pid}.tar")
        for ph in PHASES:
            sitk.WriteImage(imgs[ph], str(data / f"ct_{ph}.nii.gz"), useCompression=True)
            liver = sitk.ReadImage(str(project / "outputs" / "liver_masks" / "MCT-LTDiag" / pid / f"liver_{ph}.nii.gz"),
                                   sitk.sitkUInt8)
            sitk.WriteImage(S.on_grid_of(liver, imgs[ph]), str(livers / f"liver_{ph}.nii.gz"), useCompression=True)
        print(f"[{k}/{n}] prepared (worst-phase liver Dice on the 1 mm grid {chosen.worst_dice.iloc[k - 1]:.3f})",
              flush=True)
    print(f"done: {n} cases in {work.name}/data and livers")


def summary(work):
    q = pd.read_csv(work / "registration" / "qc" / "task_0001.csv", dtype={"patient": str})
    q = q[q.status == "ok"].groupby(["patient", "phase"]).tail(1)
    q["case"] = q.patient.astype("category").cat.codes + 1
    print("case phase | liver Dice: authors' -> shift -> ours | gradient NCC: authors' -> ours")
    for r in q.sort_values(["case", "phase"]).itertuples():
        print(f"{r.case:4d} {r.phase:5s} | {r.liver_dice_before:.3f} -> {r.liver_dice_shift:.3f} -> "
              f"{r.liver_dice_after:.3f} | {r.grad_ncc_before:.3f} -> {r.grad_ncc_after:.3f}")
    worst_before = q.groupby("case").liver_dice_before.min()
    worst_after = q.groupby("case").liver_dice_after.min()
    print(f"cases whose worst phase reaches Dice >= 0.9 after: {int((worst_after >= 0.9).sum())} of {len(worst_after)}; "
          f"median worst-phase Dice {worst_before.median():.3f} -> {worst_after.median():.3f}; gradient NCC "
          f"median {q.grad_ncc_before.median():.3f} -> {q.grad_ncc_after.median():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["prepare", "summary"])
    ap.add_argument("--release", type=Path)
    ap.add_argument("--project", type=Path)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--n", type=int, default=7)
    args = ap.parse_args()
    if args.command == "prepare":
        prepare(args.release, args.project, args.work, args.n)
    else:
        summary(args.work)


if __name__ == "__main__":
    main()
