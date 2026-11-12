#!/usr/bin/env python3
"""Step 8: standard case folders for MCT-LTDiag and PLC-CECT (STANDARD.md): 1 mm, liver crop, HU clip, labels.

Why: every detector must see identical data. Each case is resampled once onto a 1 x 1 x 1 mm grid (LPS axes,
identity direction) that covers the PVP TotalSegmentator liver box + 20 mm, clipped to the PVP scan's extent (no
made-up voxels; decided 2026-11-10). The crop never uses a lesion mask.

What, per patient (all phases end up on the same 1 mm grid):
  - image_<phase>.nii.gz: int16 HU clipped to [-200, 200], linear interpolation (decided 2026-11-10).
      MCT-LTDiag: phases as registered by the authors; NC / AP / DP take PVP's header (voxel order trusted for the 14
      ICC patients whose headers are off, decided 2026-11-10).
      PLC-CECT: corrected files (data/PLC-CECT/fixed); NC / AP / DP are warped with their Step 4 transform (liver-
      centre shift + B-spline) straight onto the 1 mm grid, so they too are interpolated only once. Parts of the grid
      a phase does not cover are -200 HU; the covered share is recorded in case.json.
  - liver_<phase>.nii.gz: uint8 TotalSegmentator liver of that phase on the same grid (nearest neighbour; PLC phases
    warped with the same transform).
  - lesion_label.nii.gz (uint8 0-3) and lesion_instances.nii.gz (uint16) from lib/labels.py (the same instance ids as
    outputs/lesions/lesions_<dataset>.csv), nearest neighbour.
  - case.json (STANDARD.md fields) with per-phase registration QC on the 1 mm grid: liver Dice with the PVP liver and
    gradient NCC with the PVP image inside the PVP liver box; failure when Dice < 0.9 (PLAN.md).
Lesion-free masks (Step 9) and the landmark flag (Step 6) come later.

Outputs (--out, e.g. processed/v1): cases/<DATASET>_<patient>/ (written to .part, renamed when complete; finished
cases are skipped) and qc/standardise_<DATASET>_<task>.csv (one row per case and phase). Logs show counts only.
Usage: python -I -u standardise.py --dataset MCT-LTDiag|PLC-CECT --project <project root> --out <project>/processed/v1
           (--task K [--per-task 20] | --test N)
"""
import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import labels as LB  # noqa: E402
import mct_reader  # noqa: E402

PHASES = ["nc", "ap", "pvp", "dp"]
MARGIN_MM, SPACING_MM, HU_MIN, HU_MAX = 20.0, 1.0, -200, 200
OUTSIDE = -10000.0  # transformix fill value for grid voxels a warped phase does not cover; becomes HU_MIN after clipping
DICE_FAIL = 0.9
QC_FIELDS = ["case_id", "phase", "status", "liver_ml", "liver_dice_pvp", "grad_ncc_pvp", "coverage", "seconds"]
LP_COMMIT = "ded5eb74c85f"


# ---------- reading ----------

def mct_images(tar):
    """{phase: CT}, PVP header for every phase (same voxel grid; voxel order trusted)."""
    out = {}
    with tarfile.open(tar) as tf, tempfile.TemporaryDirectory() as tmp:
        members = {re.sub(r"^\./", "", m.name): m for m in tf.getmembers() if m.isfile()}
        for ph in PHASES:
            path = Path(tmp) / f"{ph}.nii.gz"
            path.write_bytes(tf.extractfile(members[mct_reader.MEMBERS[ph]]).read())
            out[ph] = sitk.ReadImage(str(path), sitk.sitkFloat32)
    native = {ph: list(im.GetSpacing()) for ph, im in out.items()}
    for ph in ("nc", "ap", "dp"):
        if out[ph].GetSize() != out["pvp"].GetSize():
            raise ValueError(f"{ph} is not on the PVP voxel grid")
        out[ph].CopyInformation(out["pvp"])
    return out, native


def on_grid_of(img, ref):
    """img with ref's header (same voxel grid, header trusted from ref)."""
    if img.GetSize() != ref.GetSize():
        raise ValueError("mask is not on the image's voxel grid")
    img.CopyInformation(ref)
    return img


def labels_image(lab, ref):
    """lib/labels.py arrays (nibabel x, y, z order) as SimpleITK images on ref's grid."""
    def to_img(arr):
        im = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(arr, (2, 1, 0))))
        return on_grid_of(im, ref)
    return to_img(lab.label), to_img(lab.instances)


# ---------- grid and resampling ----------

def crop_grid(pvp, pvp_liver):
    """1 mm reference image over the PVP liver box + 20 mm, clipped to the PVP scan, LPS axes."""
    idx = np.argwhere(sitk.GetArrayViewFromImage(pvp_liver) > 0)
    if not len(idx):
        raise ValueError("empty PVP liver mask")
    def corners(lo, hi):  # (z, y, x) index box -> physical corners
        pts = []
        for z in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for x in (lo[2], hi[2]):
                    pts.append(pvp.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z))))
        return np.array(pts)
    liver = corners(idx.min(0) - 0.5, idx.max(0) + 0.5)  # outer voxel edges, so the margin is >= 20 mm everywhere
    scan = corners(np.zeros(3), np.array(pvp.GetSize()[::-1]) - 1)
    lo = np.maximum(liver.min(0) - MARGIN_MM, scan.min(0))
    hi = np.minimum(liver.max(0) + MARGIN_MM, scan.max(0))
    size = [int(np.floor((h - l) / SPACING_MM)) + 1 for l, h in zip(lo, hi)]
    ref = sitk.Image(size, sitk.sitkInt16)
    ref.SetSpacing((SPACING_MM,) * 3)
    ref.SetOrigin(tuple(float(v) for v in lo))
    ref.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return ref, [lo.tolist(), (lo + (np.array(size) - 1) * SPACING_MM).tolist()]


def to_hu(img):
    a = np.rint(np.clip(sitk.GetArrayFromImage(img), HU_MIN, HU_MAX)).astype(np.int16)
    out = sitk.GetImageFromArray(a)
    out.CopyInformation(img)
    return out


def resample(img, ref, nearest=False, default=0.0):
    return sitk.Resample(img, ref, sitk.Transform(), sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear,
                         default, img.GetPixelID())


def warp(itk, img, transform_file, ref, order, default):
    """PLC phase -> 1 mm grid with its Step 4 transform (shift + B-spline), via transformix."""
    po = itk.ParameterObject.New()
    po.ReadParameterFile(str(transform_file))
    m = po.GetParameterMap(0)
    m["Size"] = [str(v) for v in ref.GetSize()]
    m["Index"] = ["0", "0", "0"]
    m["Spacing"] = [f"{v:.10f}" for v in ref.GetSpacing()]
    m["Origin"] = [f"{v:.10f}" for v in ref.GetOrigin()]
    m["Direction"] = [f"{v:.10f}" for v in np.array(ref.GetDirection()).reshape(3, 3).T.flatten()]
    m["FinalBSplineInterpolationOrder"] = [str(order)]
    m["DefaultPixelValue"] = [str(default)]
    m["ResultImagePixelType"] = ["float"]
    po2 = itk.ParameterObject.New()
    po2.AddParameterMap(m)
    arr = np.ascontiguousarray(sitk.GetArrayFromImage(sitk.Cast(img, sitk.sitkFloat32)))  # kept alive while ITK uses it
    moving = itk.GetImageFromArray(arr)
    moving.SetSpacing(img.GetSpacing())
    moving.SetOrigin(img.GetOrigin())
    moving.SetDirection(itk.matrix_from_array(np.array(img.GetDirection()).reshape(3, 3)))
    result = itk.transformix_filter(moving, po2, log_to_console=False)
    out = sitk.GetImageFromArray(np.array(itk.GetArrayViewFromImage(result), copy=True))
    out.CopyInformation(ref)
    del result, moving, arr
    return out


# ---------- QC ----------

def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def grad_ncc(a, b, box):
    ga = np.sqrt(sum(g ** 2 for g in np.gradient(a[box].astype(np.float32))))
    gb = np.sqrt(sum(g ** 2 for g in np.gradient(b[box].astype(np.float32))))
    ga, gb = ga - ga.mean(), gb - gb.mean()
    d = np.sqrt((ga ** 2).sum() * (gb ** 2).sum())
    return float((ga * gb).sum() / d) if d else float("nan")


def liver_box(mask):
    idx = np.argwhere(mask)
    if not len(idx):
        return tuple(slice(None) for _ in range(3))
    return tuple(slice(int(a), int(b) + 1) for a, b in zip(idx.min(0), idx.max(0)))


# ---------- one case ----------

def case(dataset, pid, diagnosis, project, out, itk, commit, overrides=None):
    """Write cases/<dataset>_<pid>/; returns QC rows. Skips a finished case whose phase decisions are unchanged.

    overrides: {(patient, phase): (action, reason)} from qc/phase_overrides.py: "use_our_registration" (MCT phase warped
    with outputs/registration/MCT-LTDiag/) or "missing" (phase left out, phases_present false). A finished case whose
    case.json does not record the same decisions is moved to <out>_replaced_<date>/cases/ and rebuilt.
    """
    case_id = f"{dataset}_{pid}"
    final = out / "cases" / case_id
    ov = {ph: v for (p, ph), v in (overrides or {}).items() if p == pid}
    signature = sorted(f"{ph}:{a}" for ph, (a, _) in ov.items())
    if final.exists():
        done = json.loads((final / "case.json").read_text()) if (final / "case.json").exists() else {}
        if done.get("overrides_applied", []) == signature:
            return None
        backup = out.parent / f"{out.name}_replaced_{time.strftime('%Y-%m-%d')}" / "cases"
        backup.mkdir(parents=True, exist_ok=True)
        target = backup / case_id
        if target.exists():
            target = backup / f"{case_id}.{int(time.time())}"
        os.replace(final, target)
    present = [ph for ph in PHASES if ov.get(ph, ("",))[0] != "missing"]
    if "pvp" not in present:
        raise ValueError("PVP cannot be missing")
    livers = project / "outputs" / "liver_masks" / dataset / pid
    if dataset == "MCT-LTDiag":
        tar = project / "data" / "MCT-LTDiag" / "tars" / f"{pid}.tar"
        imgs, native = mct_images(tar)
        lab = LB.mct(tar, diagnosis)[0]
        method = "MCT-LTDiag authors' registration (released on one grid)"
    else:
        fixed = project / "data" / "PLC-CECT" / "fixed" / pid
        imgs = {ph: sitk.ReadImage(str(fixed / f"ct_{ph}.nii.gz"), sitk.sitkFloat32) for ph in PHASES}
        native = {ph: list(im.GetSpacing()) for ph, im in imgs.items()}
        lab = LB.plc(project / "data" / "PLC-CECT", pid, diagnosis)[0]
        method = "elastix B-spline (MCT-LTDiag authors' settings) after a liver-centre shift"
    liv = {ph: on_grid_of(sitk.ReadImage(str(livers / f"liver_{ph}.nii.gz"), sitk.sitkUInt8), imgs[ph]) for ph in present}
    ref, box_mm = crop_grid(imgs["pvp"], liv["pvp"])
    lesion_label, lesion_instances = labels_image(lab, imgs["pvp"])

    tmp = out / "cases" / (case_id + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    arrays, rows, registration = {}, [], {}
    phase_method = {}
    for ph in present:
        t0 = time.time()
        coverage = 1.0
        ours = ph != "pvp" and (dataset == "PLC-CECT" or ov.get(ph, ("",))[0] == "use_our_registration")
        phase_method[ph] = (method if not (dataset == "MCT-LTDiag" and ours) else
                            "elastix B-spline (MCT-LTDiag authors' settings) after a liver-centre shift; re-registered "
                            "because it lines the liver up better than the authors' (2026-11-12)")
        if ours:
            tfile = project / "outputs" / "registration" / dataset / pid / f"{ph}_to_pvp.txt"
            image = warp(itk, imgs[ph], tfile, ref, 1, OUTSIDE)  # OUTSIDE marks grid voxels this phase doesn't cover
            outside = sitk.GetArrayViewFromImage(image) <= OUTSIDE / 2
            coverage = float(1 - outside.mean())
            liver = sitk.Cast(warp(itk, liv[ph], tfile, ref, 0, 0) > 0.5, sitk.sitkUInt8)
        else:
            image = resample(imgs[ph], ref, default=HU_MIN)
            liver = resample(liv[ph], ref, nearest=True)
        image = to_hu(image)
        sitk.WriteImage(image, str(tmp / f"image_{ph}.nii.gz"), useCompression=True)
        sitk.WriteImage(sitk.Cast(liver, sitk.sitkUInt8), str(tmp / f"liver_{ph}.nii.gz"), useCompression=True)
        arrays[ph] = (sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(liver) > 0)
        rows.append({"case_id": case_id, "phase": ph, "status": "ok", "coverage": round(coverage, 4),
                     "liver_ml": round(float(arrays[ph][1].sum()) / 1000, 1), "seconds": round(time.time() - t0, 1)})
    pvp_img, pvp_liv = arrays["pvp"]
    box = liver_box(pvp_liv)
    for r in rows:
        if r["phase"] != "pvp":
            img, lv = arrays[r["phase"]]
            r["liver_dice_pvp"] = round(dice(pvp_liv, lv), 4)
            r["grad_ncc_pvp"] = round(grad_ncc(pvp_img, img, box), 4)
            registration[r["phase"]] = {"method": phase_method[r["phase"]], "liver_dice": r["liver_dice_pvp"],
                                        "grad_ncc": r["grad_ncc_pvp"], "coverage": r["coverage"],
                                        "failed": bool(not r["liver_dice_pvp"] >= DICE_FAIL)}
    sitk.WriteImage(sitk.Cast(resample(lesion_label, ref, nearest=True), sitk.sitkUInt8),
                    str(tmp / "lesion_label.nii.gz"), useCompression=True)
    inst = resample(lesion_instances, ref, nearest=True)
    sitk.WriteImage(sitk.Cast(inst, sitk.sitkUInt16), str(tmp / "lesion_instances.nii.gz"), useCompression=True)
    kept = set(np.unique(sitk.GetArrayViewFromImage(inst))) - {0}
    record = {"dataset": dataset, "patient_id": pid, "case_id": case_id, "role": "development",
              "phases_present": {ph: ph in present for ph in PHASES},
              "phase_notes": {ph: f"{a}: {why}" for ph, (a, why) in ov.items()}, "overrides_applied": signature,
              "native_spacing_mm": native, "registration": registration, "landmark_case": None,
              "crop_box_mm": box_mm, "grid": {"spacing_mm": [SPACING_MM] * 3, "size": list(ref.GetSize()),
                                               "origin_mm": list(ref.GetOrigin()), "direction": "identity (LPS)"},
              "hu_clip": [HU_MIN, HU_MAX], "diagnosis": diagnosis,
              "lesions": {"instances_native": len(lab.lesions), "instances_on_1mm_grid": len(kept)},
              "pipeline_commit": commit}
    if dataset == "PLC-CECT":
        record["liverpreprocessing_commit"] = LP_COMMIT
    (tmp / "case.json").write_text(json.dumps(record, indent=1))
    os.replace(tmp, final)
    return rows


def repo_commit():
    """Commit of this repository, read from .git without the git program (not on every node's PATH)."""
    git = Path(__file__).resolve().parents[2] / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head[:7]
        ref = head[5:]
        if (git / ref).exists():
            return (git / ref).read_text().strip()[:7]
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0][:7]
    except OSError:
        pass
    return "unknown"


def patient_list(project, dataset):
    """[(patient, diagnosis)] in a fixed order."""
    if dataset == "MCT-LTDiag":
        t = mct_reader.patients(project / "data" / "MCT-LTDiag")
        return list(zip(t.patient_id, t.type))
    fixed = project / "data" / "PLC-CECT" / "fixed"
    out = []
    for d in sorted(p for p in fixed.iterdir() if p.is_dir() and not p.name.endswith(".part")):
        out.append((d.name, json.loads((d / "geometry.json").read_text())["diagnosis"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=["MCT-LTDiag", "PLC-CECT"], required=True)
    ap.add_argument("--project", type=Path, required=True, help="project root (data/, outputs/)")
    ap.add_argument("--out", type=Path, required=True, help="release folder, e.g. <project>/processed/v1")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", type=int, help="array task K (1-based)")
    g.add_argument("--test", type=int, help="N random patients (seed 0)")
    ap.add_argument("--per-task", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--overrides", type=Path, help="phase decisions CSV (default <project>/outputs/phase_overrides.csv)")
    args = ap.parse_args()
    import itk  # transformix: PLC phases and re-registered MCT phases
    itk.MultiThreaderBase.SetGlobalMaximumNumberOfThreads(args.threads)
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(args.threads)
    overrides = {}
    ov_path = args.overrides or (args.project / "outputs" / "phase_overrides.csv")
    if ov_path.exists():
        o = pd.read_csv(ov_path, dtype=str)
        overrides = {(r.patient, r.phase): (r.action, r.reason) for r in o[o.dataset == args.dataset].itertuples()}
    print(f"phase decisions for {args.dataset}: {len(overrides)}", flush=True)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
    commit = repo_commit()
    pats = patient_list(args.project, args.dataset)
    if args.task:
        chosen = pats[(args.task - 1) * args.per_task: args.task * args.per_task]
        tag = f"{args.task:04d}"
    else:
        chosen = random.Random(0).sample(pats, args.test)
        tag = "test"
    print(f"{args.dataset}: {len(chosen)} patients (task {args.task or 'test'}, {len(pats)} in total)", flush=True)
    (args.out / "qc").mkdir(parents=True, exist_ok=True)
    qc_path = args.out / "qc" / f"standardise_{args.dataset}_{tag}.csv"
    new = not qc_path.exists()
    counts, t_all = {}, time.time()
    with qc_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=QC_FIELDS)
        if new:
            w.writeheader()
        for i, (pid, dx) in enumerate(chosen, 1):
            t0 = time.time()
            try:
                rows = case(args.dataset, pid, dx, args.project, args.out, itk, commit, overrides)
                status = "skipped" if rows is None else "done"
            except Exception as e:  # keep going; the row says what failed
                rows, status = [{"case_id": f"{args.dataset}_{pid}", "phase": "-",
                                 "status": f"failed: {type(e).__name__}: {e}"}], "failed"
                shutil.rmtree(args.out / "cases" / f"{args.dataset}_{pid}.part", ignore_errors=True)
            for r in rows or []:
                w.writerow(r)
            fh.flush()
            counts[status] = counts.get(status, 0) + 1
            if rows and status == "done":
                reg = ", ".join(f"{r['phase']} Dice {r['liver_dice_pvp']:.3f} NCC {r['grad_ncc_pvp']:.3f}"
                                for r in rows if r["phase"] != "pvp")
                print(f"[{i}/{len(chosen)}] {status}, {time.time() - t0:.0f} s; {reg}", flush=True)
            else:
                print(f"[{i}/{len(chosen)}] {status}" + (f": {rows[0]['status']}" if status == "failed" else ""),
                      flush=True)
    print(f"done: {counts}, {(time.time() - t_all) / 60:.1f} min")
    if counts.get("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
