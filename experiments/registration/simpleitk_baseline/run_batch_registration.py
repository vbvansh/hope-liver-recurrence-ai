#!/usr/bin/env python3
"""Batch registration of HCC-TACE-Seg (experiment E1): pre-TACE vs post-TACE CT.

For every eligible patient, one at a time:
  1. download the expert mask (DICOM-SEG) and the two CT series from TCIA
  2. register the post-TACE scan onto the pre-TACE scan: none -> rigid -> deformable
  3. save metrics (result.json), transforms and one QC figure
  4. delete the downloaded scans (so disk use stays ~ one patient)
Resumable: patients that already have a result.json are skipped.

How the scans are chosen:
  * BEFORE = the exact CT series + acquisition the radiologist outlined (read from the SEG).
  * AFTER  = the most similar contrast series in the next session (same description if
             possible, else any "PHASE" series); among its acquisitions (a "3 PHASE" folder can
             hold 2 stacked phases) we keep the one whose aorta brightness after rigid alignment
             is closest to BEFORE, i.e. the same contrast phase.

Usage (run from the folder that holds this script and metadata.csv):
  python run_batch_registration.py --plan                        # list chosen scans, small downloads only
  python run_batch_registration.py --patients HCC_002 HCC_003    # quick test on 2 patients
  nohup python run_batch_registration.py > batch_log.txt 2>&1 &  # all patients, keeps running in background
  python run_batch_registration.py --summary                     # rebuild summary table + charts
"""
import argparse
import csv
import gc
import json
import os
import re
import shutil
import sys
import time
import traceback
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TCIA_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID="
WIN_LO, WIN_HI = -160, 240          # abdomen soft-tissue window (HU) used for display and scores
MIN_FREE_MB = 500                   # stop before downloading if the disk is fuller than this
LEVELS = ("none", "rigid", "deformable")

sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(8)


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ── metadata: which series belong to which patient/session ──────────────────
def norm_desc(s):
    s = re.sub(r"(\d)PHASE", r"\1 PHASE", str(s).upper())
    return re.sub(r"\s+", " ", s).strip()


def recon_token(desc):
    m = re.search(r"RECON (\d)", desc)
    return m.group(1) if m else None


def read_metadata(path):
    by_patient = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["date"] = datetime.strptime(r["Study Date"], "%m-%d-%Y")
            r["desc"] = norm_desc(r["Series Description"])
            by_patient[r["Subject ID"]].append(r)
    return by_patient


def pick_followup(base_row, followup_rows):
    """Most comparable contrast CT series in the follow-up session, or None."""
    b = base_row["desc"]

    def score(r):
        d = r["desc"]
        s = 4 if d == b else 0
        if "PHASE" in d:
            s += 2
        if recon_token(d) and recon_token(d) == recon_token(b):
            s += 1
        if "PRE" in d.split() and d != b:
            s = 0                                  # "PRE" = before contrast injection
        return s, int(r["Number of Images"])

    cands = [r for r in followup_rows if r["Modality"] == "CT"]
    if not cands:
        return None
    best = max(cands, key=score)
    return best if score(best)[0] >= 2 else None


# ── download ─────────────────────────────────────────────────────────────────
def download_series(uid, dest, expected=None):
    """Download one series from TCIA into `dest`; returns the list of .dcm files."""
    zpath = dest.parent / (dest.name + ".zip")
    for attempt in range(1, 4):
        try:
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True)
            with urllib.request.urlopen(TCIA_API + uid, timeout=600) as resp, open(zpath, "wb") as fh:
                shutil.copyfileobj(resp, fh, 1 << 20)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(dest)
            zpath.unlink()
            files = sorted(p for p in dest.rglob("*.dcm") if not p.name.startswith("._"))
            if not files or (expected and len(files) != expected):
                raise IOError(f"got {len(files)} files, expected {expected}")
            return files
        except Exception as e:  # network hiccup, truncated zip, ...
            log(f"      download attempt {attempt}/3 failed: {e}")
            zpath.unlink(missing_ok=True)
            time.sleep(15 * attempt)
    raise RuntimeError(f"could not download series {uid}")


def free_mb(path):
    return shutil.disk_usage(path).free / 2**20


# ── DICOM loading ────────────────────────────────────────────────────────────
def slice_position(h):
    iop = np.array(h.ImageOrientationPatient, float)
    return float(np.dot(np.cross(iop[:3], iop[3:]), np.array(h.ImagePositionPatient, float)))


def load_acquisitions(files):
    """{AcquisitionNumber: {"image", "sops", "n"}}: one clean volume per stacked phase."""
    groups = defaultdict(list)
    for f in files:
        h = pydicom.dcmread(f, stop_before_pixels=True)
        groups[int(getattr(h, "AcquisitionNumber", 0) or 0)].append(
            (slice_position(h), str(f), str(h.SOPInstanceUID)))
    out = {}
    for acq, items in sorted(groups.items()):
        items.sort()
        pos = [p for p, _, _ in items]
        if len(items) < 10 or len(set(np.round(pos, 2))) != len(pos):
            continue                               # too short, or still mixed phases
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([f for _, f, _ in items])
        out[acq] = {"image": sitk.Cast(reader.Execute(), sitk.sitkFloat32),
                    "sops": {s for _, _, s in items}, "n": len(items)}
    return out


def seg_info(seg_file):
    """Referenced series UID and SOP UIDs of the CT slices the radiologist outlined."""
    ds = pydicom.dcmread(seg_file, stop_before_pixels=True)
    ref_series = ds.ReferencedSeriesSequence[0].SeriesInstanceUID if "ReferencedSeriesSequence" in ds else None
    sops = set()
    for pf in ds.PerFrameFunctionalGroupsSequence:
        for der in getattr(pf, "DerivationImageSequence", []):
            for src in getattr(der, "SourceImageSequence", []):
                sops.add(str(src.ReferencedSOPInstanceUID))
    return ref_series, sops


def load_seg(seg_path):
    """DICOM-SEG -> (label image in world coordinates, {number: name}).
    Adapted from processing/HCC-TACE-Seg/pipeline/02_seg_to_mask.py."""
    ds = pydicom.dcmread(seg_path)
    names = {int(s.SegmentNumber): str(s.SegmentLabel) for s in ds.SegmentSequence}
    frames = ds.pixel_array
    if frames.ndim == 2:
        frames = frames[None]
    sfg = ds.SharedFunctionalGroupsSequence[0]
    ps = [float(x) for x in sfg.PixelMeasuresSequence[0].PixelSpacing]
    iop = [float(x) for x in sfg.PlaneOrientationSequence[0].ImageOrientationPatient]
    row_dir, col_dir = np.array(iop[:3]), np.array(iop[3:])
    normal = np.cross(row_dir, col_dir)
    rec = []
    for i, pf in enumerate(ds.PerFrameFunctionalGroupsSequence):
        seg_n = int(pf.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        pos = np.array([float(x) for x in pf.PlanePositionSequence[0].ImagePositionPatient])
        rec.append((seg_n, pos, i))
    # round once and look slices up in that same array: mixing np.round and Python round()
    # can disagree on the last decimal (e.g. -223.905) and crash with a KeyError
    proj = np.round([np.dot(p, normal) for _, p, _ in rec], 2)
    uniq = np.unique(proj)
    slice_of = np.searchsorted(uniq, proj)
    dz = float(np.median(np.diff(uniq))) if len(uniq) > 1 else 1.0
    vol = np.zeros((len(uniq), frames.shape[1], frames.shape[2]), np.uint8)
    for seg_n, pos, i in sorted(rec, key=lambda r: names.get(r[0], "").lower() == "mass"):
        vol[slice_of[i]][frames[i] > 0] = seg_n
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((ps[1], ps[0], abs(dz)))
    img.SetDirection([float(x) for x in np.column_stack([row_dir, col_dir, normal]).flatten()])
    img.SetOrigin([float(x) for x in rec[int(np.argmin(proj))][1]])
    return img, names


# ── registration ─────────────────────────────────────────────────────────────
def body_mask(img):
    return sitk.Cast(sitk.BinaryThreshold(img, -500, 3000), sitk.sitkUInt8)


def clamp(img):
    return sitk.Clamp(img, sitk.sitkFloat32, -1000, 1500)


def base_method(fixed):
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.02, seed=42)
    reg.SetMetricFixedMask(body_mask(fixed))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return reg


def register_rigid(fixed, moving):
    init = sitk.CenteredTransformInitializer(
        sitk.Cast(body_mask(fixed), sitk.sitkFloat32), sitk.Cast(body_mask(moving), sitk.sitkFloat32),
        sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg = base_method(fixed)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-3, numberOfIterations=300)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SetInitialTransform(init, inPlace=False)
    out = reg.Execute(clamp(fixed), clamp(moving))
    return sitk.Euler3DTransform(out.GetNthTransform(0) if out.GetName() == "CompositeTransform" else out)


def register_deformable(fixed, moving, rigid, grid_nodes):
    bspline = sitk.BSplineTransformInitializer(fixed, grid_nodes)
    reg = base_method(fixed)
    reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)
    reg.SetShrinkFactorsPerLevel([4, 2])
    reg.SetSmoothingSigmasPerLevel([2, 1])
    reg.SetMovingInitialTransform(rigid)
    reg.SetInitialTransform(bspline, inPlace=True)
    reg.Execute(clamp(fixed), clamp(moving))
    return bspline


def resample(moving, fixed, transform):
    return sitk.GetArrayFromImage(sitk.Resample(moving, fixed, transform, sitk.sitkLinear, -1024.0, sitk.sitkFloat32))


# ── scores ───────────────────────────────────────────────────────────────────
def scores(fixed_arr, warped, region):
    m = region & (warped > -1000)
    if m.sum() < 1000:
        return {"MAE_HU": np.nan, "NCC": np.nan, "PSNR_dB": np.nan, "coverage_pct": 0.0}
    a = np.clip(fixed_arr[m], WIN_LO, WIN_HI)
    b = np.clip(warped[m], WIN_LO, WIN_HI)
    mse = float(np.mean((a - b) ** 2))
    return {"MAE_HU": float(np.mean(np.abs(a - b))), "NCC": float(np.corrcoef(a, b)[0, 1]),
            "PSNR_dB": float(20 * np.log10(WIN_HI - WIN_LO) - 10 * np.log10(max(mse, 1e-6))),
            "coverage_pct": float(100 * m.sum() / region.sum())}


def region_median(arr, mask):
    v = arr[mask & (arr > -1000)]
    return float(np.median(v)) if v.size > 50 else np.nan


def jacobian_stats(bspline, fixed, liver, tumour):
    """How much the bending squeezes (<1) or stretches (>1) tissue; <=0 means folding (impossible)."""
    ref = sitk.Shrink(fixed, [4, 4, 1])          # a smooth B-spline needs no full-resolution grid
    disp = sitk.TransformToDisplacementField(bspline, sitk.sitkVectorFloat64, ref.GetSize(),
                                             ref.GetOrigin(), ref.GetSpacing(), ref.GetDirection())
    jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(disp))
    small = lambda m: m[:, ::4, ::4][:, :jac.shape[1], :jac.shape[2]]
    lv, tu = small(liver), small(tumour)
    return {"jac_fold_pct_liver": float(100 * np.mean(jac[lv] <= 0)) if lv.any() else np.nan,
            "jac_min_liver": float(jac[lv].min()) if lv.any() else np.nan,
            "jac_mean_tumour": float(jac[tu].mean()) if tu.any() else np.nan}


def qc_figure(path, pid, fixed_arr, warped_none, warped_def, liver, tumour, z, title_extra):
    f = np.clip(fixed_arr[z], WIN_LO, WIN_HI)
    w = np.clip(warped_def[z], WIN_LO, WIN_HI)
    tiles = (np.indices(f.shape) // 64).sum(axis=0) % 2 == 0
    panels = [(f, "gray", None, "BEFORE (pre-TACE)"),
              (w, "gray", None, "AFTER aligned (deformable)"),
              (np.where(tiles, f, w), "gray", None, "checkerboard"),
              (np.clip(warped_none[z], WIN_LO, WIN_HI) - f, "bwr", 200, "difference, no registration"),
              (w - f, "bwr", 200, "difference, deformable")]
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.8))
    for a, (img, cmap, lim, t) in zip(ax, panels):
        a.imshow(img, cmap=cmap, vmin=-lim if lim else None, vmax=lim)
        a.contour(liver[z], colors="lime", linewidths=0.6)
        a.contour(tumour[z], colors="yellow", linewidths=1)
        a.set_title(t); a.axis("off")
    fig.suptitle(f"{pid}  {title_extra}   (lime = liver, yellow = tumour; red = brighter after, blue = darker after)")
    plt.tight_layout(); plt.savefig(path, dpi=60); plt.close(fig)


# ── one patient ──────────────────────────────────────────────────────────────
def process_patient(pid, rows, work, out_dir, grid_nodes):
    t_start = time.time()
    dates = sorted({r["date"] for r in rows})
    if len(dates) < 2:
        raise ValueError("only one session")
    seg_row = next((r for r in rows if r["Modality"] == "SEG"), None)
    if seg_row is None:
        raise ValueError("no expert segmentation")

    pwork = work / pid
    shutil.rmtree(pwork, ignore_errors=True)
    pwork.mkdir(parents=True)
    try:
        log("   downloading expert mask")
        seg_file = download_series(seg_row["Series UID"], pwork / "seg")[0]
        ref_uid, ref_sops = seg_info(seg_file)
        base_row = next((r for r in rows if r["Series UID"] == ref_uid), None)
        if base_row is None:
            raise ValueError("segmentation does not say which CT series it was drawn on")
        fu_row = pick_followup(base_row, [r for r in rows if r["date"] == dates[1]])
        if fu_row is None:
            raise ValueError(f"no comparable contrast series after TACE for '{base_row['Series Description']}'")

        log(f"   BEFORE {base_row['Study Date']}  '{base_row['Series Description']}'  ({base_row['Number of Images']} files)")
        log(f"   AFTER  {fu_row['Study Date']}  '{fu_row['Series Description']}'  ({fu_row['Number of Images']} files)")
        base_files = download_series(base_row["Series UID"], pwork / "before", int(base_row["Number of Images"]))
        fu_files = download_series(fu_row["Series UID"], pwork / "after", int(fu_row["Number of Images"]))
        t_download = time.time() - t_start

        base_acqs = load_acquisitions(base_files)
        fu_acqs = load_acquisitions(fu_files)
        if not base_acqs or not fu_acqs:
            raise ValueError("could not build a clean 3D volume")
        base_acq = max(base_acqs, key=lambda a: len(base_acqs[a]["sops"] & ref_sops))
        fixed = base_acqs[base_acq]["image"]

        seg_img, names = load_seg(seg_file)
        labels = sitk.GetArrayFromImage(sitk.Resample(seg_img, fixed, sitk.Transform(), sitk.sitkNearestNeighbor, 0))
        ids = {v.lower(): k for k, v in names.items()}
        liver = labels == ids.get("liver", -1)
        tumour = labels == ids.get("mass", -1)
        aorta = labels == ids.get("abdominal aorta", -1)
        if liver.sum() < 1000:
            raise ValueError("liver mask missing or not on this scan")
        healthy = liver & ~ndimage.binary_dilation(tumour, iterations=3)
        fixed_arr = sitk.GetArrayFromImage(fixed)
        aorta_before = region_median(fixed_arr, aorta)

        # rigid for every AFTER acquisition; keep the one whose aorta matches BEFORE best (same phase)
        t0 = time.time()
        cands = {}
        for acq, d in fu_acqs.items():
            rig = register_rigid(fixed, d["image"])
            w = resample(d["image"], fixed, rig)
            cands[acq] = (rig, abs(region_median(w, aorta) - aorta_before) if aorta.any() else 0.0, w)
        fu_acq = min(cands, key=lambda a: (np.nan_to_num(cands[a][1], nan=1e9), -fu_acqs[a]["n"]))
        rigid, _, warped_rigid = cands[fu_acq]
        del cands
        moving = fu_acqs[fu_acq]["image"]
        t_rigid = time.time() - t0

        t0 = time.time()
        bspline = register_deformable(fixed, moving, rigid, grid_nodes)
        deformable = sitk.CompositeTransform([rigid, bspline])
        t_def = time.time() - t0

        warped_none = resample(moving, fixed, sitk.Transform(3, sitk.sitkIdentity))
        warped_def = resample(moving, fixed, deformable)
        per_level = {"none": scores(fixed_arr, warped_none, healthy),
                     "rigid": scores(fixed_arr, warped_rigid, healthy),
                     "deformable": scores(fixed_arr, warped_def, healthy)}

        # brightness shift of healthy liver (contrast timing) and the error left once it is removed
        m = healthy & (warped_def > -1000)
        diff = np.clip(warped_def[m], WIN_LO, WIN_HI) - np.clip(fixed_arr[m], WIN_LO, WIN_HI)
        shift = float(np.median(diff))

        tv = tumour & (warped_def > -1000)
        vox_ml = float(np.prod(fixed.GetSpacing()) / 1000)
        spacing_b, spacing_a = fixed.GetSpacing(), moving.GetSpacing()
        result = {
            "patient": pid, "days_between": (dates[1] - dates[0]).days,
            "before_series": base_row["Series Description"], "after_series": fu_row["Series Description"],
            "before_acq": base_acq, "after_acq": fu_acq,
            "before_acq_count": len(base_acqs), "after_acq_count": len(fu_acqs),
            "before_slice_mm": round(spacing_b[2], 2), "after_slice_mm": round(spacing_a[2], 2),
            "liver_ml": round(float(liver.sum()) * vox_ml, 1), "tumour_ml": round(float(tumour.sum()) * vox_ml, 1),
            "rigid_shift_mm": round(float(np.linalg.norm(rigid.GetTranslation())), 1),
            "rigid_rotation_deg": round(float(np.degrees(np.max(np.abs(
                [rigid.GetAngleX(), rigid.GetAngleY(), rigid.GetAngleZ()])))), 2),
            "aorta_HU_before": aorta_before, "aorta_HU_after": region_median(warped_def, aorta),
            "liver_shift_HU": shift, "MAE_shift_corrected_HU": float(np.mean(np.abs(diff - shift))),
            "tumour_HU_before": region_median(fixed_arr, tumour), "tumour_HU_after": region_median(warped_def, tv),
            "tumour_p95_before": float(np.percentile(fixed_arr[tv], 95)) if tv.sum() > 50 else np.nan,
            "tumour_p95_after": float(np.percentile(warped_def[tv], 95)) if tv.sum() > 50 else np.nan,
            "grid_nodes": list(grid_nodes),
            "time_download_s": round(t_download), "time_rigid_s": round(t_rigid), "time_deformable_s": round(t_def),
        }
        for level, s in per_level.items():
            for k, v in s.items():
                result[f"{level}_{k}"] = v
        result.update(jacobian_stats(bspline, fixed, liver, tumour))

        z = int(np.argmax(tumour.sum(axis=(1, 2)))) if tumour.any() else fixed_arr.shape[0] // 2
        out_dir.mkdir(parents=True, exist_ok=True)
        qc_figure(out_dir / "qc.png", pid, fixed_arr, warped_none, warped_def, liver, tumour, z,
                  f"MAE {per_level['none']['MAE_HU']:.0f} -> {per_level['rigid']['MAE_HU']:.0f} -> "
                  f"{per_level['deformable']['MAE_HU']:.0f} HU")
        sitk.WriteTransform(rigid, str(out_dir / "rigid.tfm"))
        sitk.WriteTransform(bspline, str(out_dir / "bspline.tfm"))
        result["time_total_s"] = round(time.time() - t_start)
        with open(out_dir / "result.json", "w") as fh:     # written last = "this patient is done"
            json.dump(result, fh, indent=1, default=float)
        return result
    finally:
        shutil.rmtree(pwork, ignore_errors=True)            # always free the disk
        gc.collect()


# ── summary across patients ──────────────────────────────────────────────────
def fmt(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return "-"
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return f"{med:.2f} [{q1:.2f}-{q3:.2f}]"


def summarize(res_dir):
    results = []
    for p in sorted((res_dir / "patients").glob("*/result.json")):
        with open(p) as fh:
            results.append(json.load(fh))
    done = {r["patient"] for r in results}
    latest = {}                                   # one line per patient: its most recent reason
    for r in read_failures(res_dir / "failures.csv"):
        if r["patient"] not in done:
            latest[r["patient"]] = r
    failures = [latest[p] for p in sorted(latest)]
    if not results:
        log("no finished patients yet")
        return

    keys = list(results[0].keys())
    with open(res_dir / "per_patient_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})

    col = lambda k: [r.get(k) for r in results]
    n = len(results)
    skipped = [f for f in failures if is_permanent(f["reason"])]
    retry = [f for f in failures if not is_permanent(f["reason"])]
    lines = [f"# HCC-TACE-Seg registration summary ({n} patients done, {len(skipped)} skipped for data "
             f"reasons, {len(retry)} failed temporarily)", "",
             "Scores inside the healthy liver (tumour excluded). Values: median [25th-75th percentile].", "",
             "| Level | MAE (HU), lower = better | NCC, higher = better | PSNR (dB), higher = better | Liver coverage (%) |",
             "|---|---|---|---|---|"]
    for lv in LEVELS:
        lines.append(f"| {lv} | {fmt(col(f'{lv}_MAE_HU'))} | {fmt(col(f'{lv}_NCC'))} | "
                     f"{fmt(col(f'{lv}_PSNR_dB'))} | {fmt(col(f'{lv}_coverage_pct'))} |")
    def better(a, b):
        pairs = [(r[f"{a}_MAE_HU"], r[f"{b}_MAE_HU"]) for r in results]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
        return f"{sum(x < y for x, y in pairs)}/{len(pairs)}"

    no_overlap = sum(1 for r in results if not r.get("none_coverage_pct"))
    lines += ["",
              f"- Rigid better than none: {better('rigid', 'none')} patients "
              f"({no_overlap} more had NO overlap at all before registration, so 'none' cannot be scored)",
              f"- Deformable better than rigid: {better('deformable', 'rigid')} patients",
              f"- Rigid movement needed: shift {fmt(col('rigid_shift_mm'))} mm, rotation {fmt(col('rigid_rotation_deg'))} deg",
              f"- Healthy-liver brightness shift after/before: {fmt(col('liver_shift_HU'))} HU; "
              f"MAE once that shift is removed: {fmt(col('MAE_shift_corrected_HU'))} HU",
              f"- Aorta brightness before {fmt(col('aorta_HU_before'))} HU vs after {fmt(col('aorta_HU_after'))} HU "
              "(similar values = same contrast phase)",
              f"- Warp folding inside liver: {fmt(col('jac_fold_pct_liver'))} % of voxels (should be 0); "
              f"tumour region volume factor {fmt(col('jac_mean_tumour'))} (1 = untouched)",
              f"- Tumour median HU before {fmt(col('tumour_HU_before'))} vs after {fmt(col('tumour_HU_after'))}",
              f"- Time per patient: {fmt(col('time_total_s'))} s", ""]
    ranked = sorted(results, key=lambda r: r["deformable_MAE_HU"])
    lines.append("Best 5 (lowest deformable MAE): " + ", ".join(
        f"{r['patient']} ({r['deformable_MAE_HU']:.0f})" for r in ranked[:5]))
    lines.append("Worst 5 (highest deformable MAE): " + ", ".join(
        f"{r['patient']} ({r['deformable_MAE_HU']:.0f})" for r in ranked[-5:]))
    if skipped:
        lines += ["", "Skipped for data reasons (expected, not retried):"] + [
            f"- {f['patient']}: {f['reason']}" for f in skipped]
    if retry:
        lines += ["", "Failed temporarily (run the script again to retry):"] + [
            f"- {f['patient']}: {f['reason']}" for f in retry]
    text = "\n".join(lines)
    (res_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)

    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    for a, metric, label in [(ax[0], "MAE_HU", "MAE in healthy liver (HU) - lower is better"),
                             (ax[1], "PSNR_dB", "PSNR (dB) - higher is better")]:
        data = [[v for v in col(f"{lv}_{metric}") if v is not None and np.isfinite(v)] for lv in LEVELS]
        a.boxplot(data)
        a.set_xticks([1, 2, 3]); a.set_xticklabels(LEVELS)
        a.set_title(label)
    for r in results:
        ax[2].plot(range(3), [r[f"{lv}_MAE_HU"] for lv in LEVELS], color="tab:blue", alpha=0.3)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(LEVELS)
    ax[2].set_title("MAE per patient (one line = one patient)")
    ax[3].scatter(col("aorta_HU_before"), col("aorta_HU_after"), s=12)
    lim = [0, np.nanmax([x for x in col("aorta_HU_before") + col("aorta_HU_after") if x is not None] + [300])]
    ax[3].plot(lim, lim, "k--", lw=0.8)
    ax[3].set_xlabel("aorta HU before"); ax[3].set_ylabel("aorta HU after")
    ax[3].set_title("contrast phase match (dots near the line = same phase)")
    plt.tight_layout(); plt.savefig(res_dir / "summary.png", dpi=90); plt.close(fig)
    log(f"summary written to {res_dir / 'summary.md'} and summary.png")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=HERE / "metadata.csv", help="TCIA metadata.csv")
    ap.add_argument("--results", type=Path, default=HERE / "results" / "02_batch_registration")
    ap.add_argument("--work", type=Path, default=HERE / "work", help="temporary download folder")
    ap.add_argument("--patients", nargs="*", help="only these patients, e.g. HCC_002 HCC_003")
    ap.add_argument("--limit", type=int, help="stop after this many new patients")
    ap.add_argument("--grid", type=int, nargs=3, default=[6, 6, 6], help="B-spline control points per axis")
    ap.add_argument("--retry-failed", action="store_true", help="try previously failed patients again")
    ap.add_argument("--plan", action="store_true", help="only list which series would be used (downloads masks only)")
    ap.add_argument("--summary", action="store_true", help="only rebuild the summary from finished patients")
    args = ap.parse_args()

    res_dir = args.results
    (res_dir / "patients").mkdir(parents=True, exist_ok=True)
    if args.summary:
        summarize(res_dir)
        return

    # only ONE copy may run: two copies share the work folder and delete each other's downloads
    lock = res_dir / "run.lock"
    if not acquire_lock(lock):
        log(f"ANOTHER COPY IS ALREADY RUNNING (process {lock.read_text().strip()}). "
            "Wait for it to finish, or stop it with:  pkill -f run_batch_registration")
        return 1
    try:
        return run(args, res_dir)
    finally:
        lock.unlink(missing_ok=True)


def acquire_lock(lock):
    """Create the lock file; a lock left behind by a process that no longer exists is replaced."""
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            if os.name != "posix":      # os.kill(pid, 0) would terminate the process on Windows
                return False
            try:
                os.kill(int(lock.read_text().strip()), 0)   # signal 0 = "are you alive?"
                return False
            except (ProcessLookupError, ValueError):
                lock.unlink(missing_ok=True)                 # stale lock from a crashed/stopped run
            except PermissionError:
                return False
    return False


def read_failures(path):
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def is_permanent(reason):
    """ValueError = something about the data itself (skip for good); anything else
    (download error, missing file, ...) is temporary and is retried on the next run."""
    return reason.startswith("ValueError")


def run(args, res_dir):
    meta = read_metadata(args.metadata)
    pids = args.patients or sorted(meta)
    fail_path = res_dir / "failures.csv"
    failed = set() if args.retry_failed else {
        r["patient"] for r in read_failures(fail_path) if is_permanent(r["reason"])}

    if args.plan:
        with open(res_dir / "plan.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["patient", "before_series", "after_series", "status"])
            for pid in pids:
                rows = meta[pid]
                seg_row = next((r for r in rows if r["Modality"] == "SEG"), None)
                dates = sorted({r["date"] for r in rows})
                try:
                    tmp = args.work / "plan"
                    shutil.rmtree(tmp, ignore_errors=True)
                    ref_uid, _ = seg_info(download_series(seg_row["Series UID"], tmp)[0])
                    shutil.rmtree(tmp, ignore_errors=True)
                    base = next((r for r in rows if r["Series UID"] == ref_uid), None)
                    fu = pick_followup(base, [r for r in rows if r["date"] == dates[1]]) if base else None
                    status = "OK" if fu else ("no SEG reference" if not base else "no comparable AFTER series")
                    w.writerow([pid, base["Series Description"] if base else "", fu["Series Description"] if fu else "", status])
                    log(f"{pid}: {status}   {base['Series Description'] if base else ''} -> {fu['Series Description'] if fu else ''}")
                except Exception as e:
                    w.writerow([pid, "", "", f"error: {e}"])
                    log(f"{pid}: error {e}")
        log(f"plan written to {res_dir / 'plan.csv'}")
        return

    todo = [p for p in pids if not (res_dir / "patients" / p / "result.json").exists() and p not in failed]
    log(f"{len(pids)} patients requested, {len(todo)} to do "
        f"({len(pids) - len(todo)} already done or permanently skipped)")
    new = 0
    for i, pid in enumerate(todo, 1):
        if args.limit and new >= args.limit:
            break
        if free_mb(args.work.parent) < MIN_FREE_MB:
            log(f"STOP: only {free_mb(args.work.parent):.0f} MB free on disk (need {MIN_FREE_MB} MB)")
            break
        log(f"[{i}/{len(todo)}] {pid}")
        try:
            r = process_patient(pid, meta[pid], args.work, res_dir / "patients" / pid, args.grid)
            new += 1
            log(f"   done in {r['time_total_s']} s | MAE none {r['none_MAE_HU']:.1f} -> rigid {r['rigid_MAE_HU']:.1f}"
                f" -> deformable {r['deformable_MAE_HU']:.1f} HU")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            if is_permanent(reason):
                log(f"   SKIPPED (data reason, will not retry): {reason}")
            else:
                log(f"   FAILED (temporary, retried on the next run): {reason}")
                traceback.print_exc()
            new_file = not fail_path.exists()
            with open(fail_path, "a", newline="") as fh:
                w = csv.writer(fh)
                if new_file:
                    w.writerow(["patient", "reason", "time"])
                w.writerow([pid, reason, f"{datetime.now():%Y-%m-%d %H:%M}"])
    summarize(res_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
