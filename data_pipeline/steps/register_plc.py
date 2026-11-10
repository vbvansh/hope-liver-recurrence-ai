#!/usr/bin/env python3
"""Step 4: register PLC-CECT NC, AP and DP onto PVP with the MCT-LTDiag authors' settings, plus a first QC.

Why: PLC-CECT phases are not registered (MCT-LTDiag's are). Method decided in PLAN.md ("PLC-CECT registration"):
ITKElastix 0.20.0, default B-spline parameter map with 3 resolutions and a 20 mm final grid spacing
(itk.ParameterObject.GetDefaultParameterMap("bspline", 3, 20.0)), fixed = PVP, moving = NC / AP / DP, no rigid or
affine step first. This reproduces the settings of the authors' Data_processing/regis.py
(github.com/Hoyant-Su/Multi-phase_LT_Benchmark, no licence file, so it is not copied).
One addition (Option B, decided 2026-11-09): PLC lost the scanner positions (every phase has origin 0, 0, 0), so
phases can start up to ~57 mm apart, which a 20 mm B-spline grid cannot close. Each moving phase therefore starts from
a translation that puts its TotalSegmentator liver centre on the PVP liver centre (automatic liver masks only, never
lesion masks), given to elastix as its initial transform; the B-spline map itself is unchanged.

Inputs: the corrected PLC files of Step 4 (<data>/<patient>/ct_<phase>.nii.gz: true geometry, HU + 40) and the
TotalSegmentator livers (<livers>/<patient>/liver_<phase>.nii.gz).

What: per patient and moving phase, one elastix registration. Saved is the transform parameter file, not the
deformation field (about 1 GB each); Step 8 applies it once, straight onto the 1 mm grid. Each registration also gets
one QC row (Step 5 adds more):
  - liver Dice with the PVP liver, before (phases overlaid by position) and after (liver warped, nearest neighbour);
  - gradient NCC inside the PVP liver box + 20 mm, before and after: how well the edges line up, a measure the
    registration did not optimise (CLAUDE.md: never judge a registration by organ overlap alone);
  - distance between the liver centres before registration, the shift, liver Dice after the shift alone, run time.

Outputs (--out): <patient>/<phase>_shift.txt (the starting translation), <patient>/<phase>_to_pvp.txt (the B-spline;
its InitialTransformParameterFileName names the shift file by absolute path, so transformix applies both; if the
folder moves, point that line at the new path) and qc/<task>.csv (one row per registration; rows append, so after a
rerun the last row of a registration counts). A registration whose transform file exists is skipped (--force redoes
it), so a task can simply be submitted again. Logs show counts and numbers only, never patient IDs.

Usage (envs/hope with itk-elastix 0.20.0):
  python -u register_plc.py --data <project>/data/PLC-CECT/fixed --livers <project>/outputs/liver_masks/PLC-CECT \
      --out <project>/outputs/registration/PLC-CECT --task K [--per-task 4] [--threads 4]
  --test N   N patients (the largest scan plus random ones, seed 0) instead of a task, for timing and checks;
             with --skip-largest random ones only (login node: 7 GB RAM; the largest scan was killed at 5.5 GB)
"""
import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

MOVING = ["nc", "ap", "dp"]
QC_FIELDS = ["patient", "phase", "status", "seconds", "liver_dice_before", "liver_dice_shift", "liver_dice_after",
             "grad_ncc_before", "grad_ncc_after", "liver_centre_distance_before_mm", "shift_mm", "fixed_slices",
             "moving_slices"]
BOX_MARGIN_MM = 20


def patients(data):
    return sorted(p.name for p in data.iterdir() if p.is_dir() and not p.name.endswith(".part"))


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def liver_box(mask, spacing_zyx):
    """Index slices of the mask's bounding box + BOX_MARGIN_MM, clipped to the volume."""
    idx = np.argwhere(mask)
    if not len(idx):
        return tuple(slice(None) for _ in range(3))
    m = np.ceil(BOX_MARGIN_MM / np.array(spacing_zyx)).astype(int)
    lo = np.maximum(idx.min(0) - m, 0)
    hi = np.minimum(idx.max(0) + m + 1, mask.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def grad_ncc(a, b, spacing_zyx):
    """Normalised cross-correlation of the gradient magnitudes of two arrays on the same grid."""
    ga = np.sqrt(sum(g ** 2 for g in np.gradient(a.astype(np.float32), *spacing_zyx)))
    gb = np.sqrt(sum(g ** 2 for g in np.gradient(b.astype(np.float32), *spacing_zyx)))
    ga, gb = ga - ga.mean(), gb - gb.mean()
    d = np.sqrt((ga ** 2).sum() * (gb ** 2).sum())
    return float((ga * gb).sum() / d) if d else float("nan")


def write_translation(shift, fixed, path):
    """Elastix transform parameter file of a translation (fixed point x -> x + shift) on the fixed image's grid."""
    q = lambda v: f'"{v}"'  # noqa: E731
    direction = np.array(fixed.GetDirection()).reshape(3, 3).T.flatten()  # elastix lists the matrix column by column
    entries = {"Transform": q("TranslationTransform"), "NumberOfParameters": "3",
               "TransformParameters": " ".join(f"{v:.6f}" for v in shift),
               "InitialTransformParameterFileName": q("NoInitialTransform"), "HowToCombineTransforms": q("Compose"),
               "FixedImageDimension": "3", "MovingImageDimension": "3",
               "FixedInternalImagePixelType": q("float"), "MovingInternalImagePixelType": q("float"),
               "Size": " ".join(str(v) for v in fixed.GetSize()), "Index": "0 0 0",
               "Spacing": " ".join(f"{v:.10f}" for v in fixed.GetSpacing()),
               "Origin": " ".join(f"{v:.10f}" for v in fixed.GetOrigin()),
               "Direction": " ".join(f"{v:.10f}" for v in direction), "UseDirectionCosines": q("true"),
               "ResampleInterpolator": q("FinalBSplineInterpolator"), "FinalBSplineInterpolationOrder": "3",
               "Resampler": q("DefaultResampler"), "DefaultPixelValue": "0", "ResultImageFormat": q("nii"),
               "ResultImagePixelType": q("float"), "CompressResultImage": q("false")}
    part = path.with_name(path.name.replace(".txt", f".{os.getpid()}.part.txt"))  # unique: safe if a task runs twice
    part.write_text("".join(f"({k} {v})\n" for k, v in entries.items()))
    os.replace(part, path)


def centre_mm(mask_img):
    a = sitk.GetArrayViewFromImage(mask_img) > 0
    if not a.any():
        return None
    zyx = np.argwhere(a).mean(axis=0)
    return np.array(mask_img.TransformContinuousIndexToPhysicalPoint(zyx[::-1].tolist()))


def register_patient(itk, pid, data, livers, out, writer, threads, force, label):
    """All three registrations of one patient; returns a list of status strings."""
    folder = out / pid
    todo = [ph for ph in MOVING if force or not (folder / f"{ph}_to_pvp.txt").exists()]
    if not todo:
        return ["skipped"] * len(MOVING)
    missing = [ph for ph in ["pvp"] + MOVING if not (livers / pid / f"liver_{ph}.nii.gz").exists()]
    if missing:  # e.g. a flipped volume waiting for its new TotalSegmentator liver; rerun the task later
        print(f"{label} no liver mask yet for {'+'.join(missing)}: patient left for a rerun", flush=True)
        return ["no liver mask"] * len(todo) + ["skipped"] * (len(MOVING) - len(todo))
    folder.mkdir(parents=True, exist_ok=True)
    f_ct = sitk.ReadImage(str(data / pid / "ct_pvp.nii.gz"), sitk.sitkFloat32)
    f_liv = sitk.ReadImage(str(livers / pid / "liver_pvp.nii.gz"), sitk.sitkUInt8)
    f_arr = sitk.GetArrayFromImage(f_ct)
    f_mask = sitk.GetArrayFromImage(f_liv) > 0
    sp = f_ct.GetSpacing()[::-1]
    box = liver_box(f_mask, sp)
    f_centre = centre_mm(f_liv)
    fixed_itk = itk.GetImageFromArray(f_arr)
    fixed_itk.SetSpacing(f_ct.GetSpacing()); fixed_itk.SetOrigin(f_ct.GetOrigin())
    fixed_itk.SetDirection(itk.matrix_from_array(np.array(f_ct.GetDirection()).reshape(3, 3)))
    statuses = []
    for ph in MOVING:
        if ph not in todo:
            statuses.append("skipped")
            continue
        t0 = time.time()
        row = {"patient": pid, "phase": ph, "fixed_slices": f_ct.GetSize()[2]}
        try:
            m_ct = sitk.ReadImage(str(data / pid / f"ct_{ph}.nii.gz"), sitk.sitkFloat32)
            m_liv = sitk.ReadImage(str(livers / pid / f"liver_{ph}.nii.gz"), sitk.sitkUInt8)
            row["moving_slices"] = m_ct.GetSize()[2]
            # before: overlay by position (identity transform)
            before_ct = sitk.GetArrayFromImage(sitk.Resample(m_ct, f_ct, sitk.Transform(), sitk.sitkLinear, 0.0))
            before_liv = sitk.GetArrayFromImage(
                sitk.Resample(m_liv, f_liv, sitk.Transform(), sitk.sitkNearestNeighbor, 0)) > 0
            m_centre = centre_mm(m_liv)
            shift = np.zeros(3)
            if f_centre is not None and m_centre is not None:
                shift = m_centre - f_centre  # fixed (PVP) point x -> moving point x + shift
                row["liver_centre_distance_before_mm"] = round(float(np.linalg.norm(shift)), 1)
            row["shift_mm"] = " ".join(f"{v:.1f}" for v in shift)
            shifted_liv = sitk.GetArrayFromImage(sitk.Resample(
                m_liv, f_liv, sitk.TranslationTransform(3, shift.tolist()), sitk.sitkNearestNeighbor, 0)) > 0
            row["liver_dice_shift"] = round(dice(f_mask, shifted_liv), 4)
            del shifted_liv
            shift_file = folder / f"{ph}_shift.txt"
            write_translation(shift, f_ct, shift_file)
            row["liver_dice_before"] = round(dice(f_mask, before_liv), 4)
            row["grad_ncc_before"] = round(grad_ncc(f_arr[box], before_ct[box], sp), 4)
            del before_ct, before_liv
            moving_itk = itk.GetImageFromArray(sitk.GetArrayFromImage(m_ct))
            moving_itk.SetSpacing(m_ct.GetSpacing()); moving_itk.SetOrigin(m_ct.GetOrigin())
            moving_itk.SetDirection(itk.matrix_from_array(np.array(m_ct.GetDirection()).reshape(3, 3)))
            po = itk.ParameterObject.New()
            po.AddParameterMap(po.GetDefaultParameterMap("bspline", 3, 20.0))
            # threads come from the global ITK setting in main(); passing number_of_threads here crashes
            # itk-elastix 0.20.0 / ITK 5.4.7 with a segmentation fault (tested 2026-11-09)
            result, tp = itk.elastix_registration_method(
                fixed_itk, moving_itk, parameter_object=po, log_to_console=False,
                initial_transform_parameter_file_name=str(shift_file.resolve()))
            part = folder / f"{ph}_to_pvp.{os.getpid()}.part.txt"  # unique per process: safe if a task is submitted twice
            tp.WriteParameterFile(tp.GetParameterMap(0), str(part))
            os.replace(part, folder / f"{ph}_to_pvp.txt")
            after_ct = itk.GetArrayFromImage(result)
            row["grad_ncc_after"] = round(grad_ncc(f_arr[box], after_ct[box], sp), 4)
            del after_ct, result
            # after: warp the moving liver with the same transform, nearest neighbour
            tmap = tp.GetParameterMap(0)
            tmap["FinalBSplineInterpolationOrder"] = ["0"]
            tpo = itk.ParameterObject.New(); tpo.AddParameterMap(tmap)
            m_liv_itk = itk.GetImageFromArray(sitk.GetArrayFromImage(m_liv).astype(np.float32))
            m_liv_itk.SetSpacing(m_liv.GetSpacing()); m_liv_itk.SetOrigin(m_liv.GetOrigin())
            m_liv_itk.SetDirection(itk.matrix_from_array(np.array(m_liv.GetDirection()).reshape(3, 3)))
            warped = itk.GetArrayFromImage(itk.transformix_filter(m_liv_itk, tpo, log_to_console=False)) > 0.5
            row["liver_dice_after"] = round(dice(f_mask, warped), 4)
            row["status"] = "ok"
        except Exception as e:  # keep going; the row says what failed
            row["status"] = f"failed: {type(e).__name__}"
        row["seconds"] = round(time.time() - t0, 1)
        writer.writerow(row)
        statuses.append(row["status"])
        print(f"{label} {ph}: {row['status']}, {row['seconds']:.0f} s, liver Dice "
              f"{row.get('liver_dice_before', float('nan')):.3f} -> shift {row.get('liver_dice_shift', float('nan')):.3f}"
              f" -> {row.get('liver_dice_after', float('nan')):.3f}, "
              f"gradient NCC {row.get('grad_ncc_before', float('nan')):.3f} -> "
              f"{row.get('grad_ncc_after', float('nan')):.3f}, liver centres "
              f"{row.get('liver_centre_distance_before_mm', float('nan'))} mm apart before", flush=True)
    return statuses


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="data/PLC-CECT/fixed")
    ap.add_argument("--livers", type=Path, required=True, help="outputs/liver_masks/PLC-CECT")
    ap.add_argument("--out", type=Path, required=True, help="outputs/registration/PLC-CECT")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", type=int, help="array task number K (1-based)")
    g.add_argument("--test", type=int, help="number of test patients (largest scan + random, seed 0)")
    ap.add_argument("--per-task", type=int, default=4, help="patients per task (default 4: 91 tasks)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="redo registrations that already have a transform file")
    ap.add_argument("--skip-largest", action="store_true",
                    help="--test: random patients only (the login node's 7 GB RAM can't hold the largest scan)")
    args = ap.parse_args()

    import itk  # slow to import (~20 s); after argument parsing
    itk.MultiThreaderBase.SetGlobalMaximumNumberOfThreads(args.threads)
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(args.threads)
    pats = patients(args.data)
    if args.task:
        chosen = pats[(args.task - 1) * args.per_task: args.task * args.per_task]
        tag = f"task_{args.task:04d}"
        print(f"task {args.task}: patients {(args.task - 1) * args.per_task + 1}-"
              f"{(args.task - 1) * args.per_task + len(chosen)} of {len(pats)}")
    else:
        slices = {}
        for p in pats:
            r = sitk.ImageFileReader(); r.SetFileName(str(args.data / p / "ct_pvp.nii.gz")); r.ReadImageInformation()
            slices[p] = r.GetSize()[2]
        largest = max(pats, key=slices.get)
        rest = [p for p in pats if p != largest]
        random.Random(0).shuffle(rest)
        chosen = rest[: args.test] if args.skip_largest else [largest] + rest[: args.test - 1]
        tag = "test"
        print(f"test: {len(chosen)} patients (PVP slices {[slices[p] for p in chosen]}; largest of all "
              f"{slices[largest]})")
    qc = args.out / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    qc_path = qc / f"{tag}.csv"
    new = not qc_path.exists()
    t0 = time.time()
    counts = {}
    with qc_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=QC_FIELDS)
        if new:
            w.writeheader()
        for i, pid in enumerate(chosen, 1):
            for s in register_patient(itk, pid, args.data, args.livers, args.out, w, args.threads, args.force,
                                      f"[{i}/{len(chosen)}]"):
                key = s.split(":")[0]
                counts[key] = counts.get(key, 0) + 1
            fh.flush()
    print(f"done: {len(chosen)} patients, {counts} registrations, {(time.time() - t0) / 60:.1f} min")
    if counts.get("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
