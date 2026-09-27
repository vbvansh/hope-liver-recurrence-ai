#!/usr/bin/env python3
"""Abdomen-CT preprocessing to the MU-Glioma-Post OUTPUT STANDARD.

The MU pipeline is brain-MRI specific (skull-strip, MNI atlas, BraTS seg). CT of
the abdomen has none of those, so this is the faithful CT analogue of the same
*output standard* (folder layout, NIfTI conventions, per-patient montage):

  OUTPUT/EAY131/
  ├── <patient_id>/
  │   ├── 000_baseline/        earliest study date (longitudinal reference)
  │   │   └── ct.nii.gz        float32, RAS+, 1.0mm isotropic, raw Hounsfield
  │   ├── 001_followup_1/
  │   └── 00N_followup_N/
  ├── visualization/<patient_id>.png   rows=timepoints, cols=3 axial levels
  ├── metadata.json
  └── processing_log.txt

Standard (set here for image generation):
  • orientation  RAS+
  • spacing      1.0 x 1.0 x 1.0 mm (linear resample)
  • dtype        float32
  • values       raw Hounsfield Units (LOSSLESS — not pre-windowed). Recommended
                 generation window = abdomen soft tissue L40 / W400  → clip
                 [-160, 240] then scale to [0,1]. (Stored raw so any window works.)
  • timepoints   one folder per study DATE, ordered 000_baseline, 001_followup_N
  • series       per timepoint, the primary AXIAL CT series is auto-selected
                 (most slices, excluding scout/localizer/recon/MIP/cor/sag etc.)

NOTE: cross-timepoint registration is intentionally NOT done (rigid abdomen CT
alignment across months is a separate problem); the montage uses matched
fractional body-z levels for QC. Add a realign step later if generation needs it.

Usage:
  python3 preprocess_ct.py --input  /media/cbtil3/WhiteSD/CTRaw/Abdomen/EAY131 \
                           --output /media/cbtil3/WhiteSD/CTProcessed \
                           --n-debug 3
  python3 preprocess_ct.py --input ... --output ... --patients EAY131-359750 ...
"""
import argparse
import datetime
import glob
import json
import os
import re

import numpy as np
import pydicom
import SimpleITK as sitk
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATASET = "EAY131"          # default; overridden by --dataset
ISO_MM = 1.0
# abdomen soft-tissue display window (level 40, width 400)
WIN_LO, WIN_HI = -160.0, 240.0
# obvious derived/non-diagnostic series (plane handled separately via orientation).
# NB: match "dose" only as a standalone dose-report token — a bare "dose"
# substring wrongly kills "iDose" (Philips iterative recon) and "Low_Dose"
# (a real low-dose CT protocol), both of which are diagnostic volumes to keep.
EXCLUDE = re.compile(r"(?:scout|localiz|topogram|scano|surview|dose[ _]?report|"
                     r"dose[ _]?record|mip|ssd|monitor|bolus|track|reformat|"
                     r"smartprep|scrr)",
                     re.I)
# body-region keywords (matched against StudyDesc + SeriesDescription)
ABDOMEN_KW = ("abd", "pelvi", "liver", "hepat", "cbapc", "renal", "kidney",
              "adrenal", "pancrea", "abdo")
CHEST_KW = ("chest", "thorax", "lung", "pulmonary")
OTHER_KW = ("head", "brain", "neck", "femur", "hand", "knee", "spine", "skull",
            "sinus", "facial", "wrist", "foot", "ankle", "shoulder", "extrem",
            "breast", "mamm")


def classify_region(desc):
    """Body region from description text: 'abdomen' (incl CAP/abd-pelvis),
    'chest', 'other' (head/limb/etc), or 'unknown'."""
    d = str(desc).lower()
    has_abd = any(k in d for k in ABDOMEN_KW)
    if any(k in d for k in OTHER_KW) and not has_abd:
        return "other"
    if has_abd:
        return "abdomen"
    if any(k in d for k in CHEST_KW):
        return "chest"
    return "unknown"


def series_plane(series_dir):
    """Acquisition plane from ImageOrientationPatient: 'axial'|'coronal'|
    'sagittal'|None. Axial = slice normal along patient z (S/I)."""
    dcms = [f for f in glob.glob(os.path.join(series_dir, "*.dcm"))
            if not os.path.basename(f).startswith("._")]
    if not dcms:
        return None
    try:
        d = pydicom.dcmread(dcms[0], stop_before_pixels=True)
        iop = [float(x) for x in d.ImageOrientationPatient]
    except Exception:
        return None
    row, col = np.array(iop[:3]), np.array(iop[3:])
    normal = np.abs(np.cross(row, col))
    return ["sagittal", "coronal", "axial"][int(np.argmax(normal))]


def log(msg, fh=None):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


# ── series selection ─────────────────────────────────────────────────────────
def select_series(meta, pid, patient_root, region="auto"):
    """Pick one primary AXIAL CT series per study date, restricted to a single
    consistent body region so timepoints overlap and can be registered.

    region="auto" (default): per-patient, choose the region (abdomen vs chest)
    that yields the MOST distinct study dates — i.e. the patient's longest
    co-registerable run — preferring abdomen on a tie. This salvages
    longitudinal series in a heterogeneous collection (NCI-MATCH mixes
    abdomen / chest / CAP / pulmonary-CTA across a patient's timepoints) instead
    of forcing one region and silently dropping the others. Explicit
    "abdomen"/"chest" force that region; "any" disables the region screen.

    Drops derived series by name, confirms axial plane via DICOM orientation.
    Returns (chosen_region, [(stage_label, StudyDate, uid, desc), ...])."""
    sub = meta[(meta.PatientID.astype(str) == pid) & (meta.Modality == "CT")].copy()
    if sub.empty:
        return None, []
    descs = (sub.SeriesDescription.fillna("") + " " + sub.StudyDesc.fillna(""))
    sub = sub[~descs.str.contains(EXCLUDE)]
    sub = sub[sub.ImageCount.fillna(0) >= 20]            # drop tiny/derived
    if sub.empty:
        return None, []
    sub["region"] = (sub.StudyDesc.fillna("") + " "
                     + sub.SeriesDescription.fillna("")).map(classify_region)

    if region == "auto":
        dates_per = {r: sub[sub.region == r].StudyDate.nunique()
                     for r in ("abdomen", "chest")}
        # most timepoints wins; abdomen breaks ties (collection's focus)
        chosen_region = max(("abdomen", "chest"),
                            key=lambda r: (dates_per[r], r == "abdomen"))
        if dates_per[chosen_region] == 0:               # neither matched: keep all
            chosen_region = "any"
    else:
        chosen_region = region

    if chosen_region != "any":
        sub = sub[sub.region == chosen_region]
    if sub.empty:
        return chosen_region, []

    chosen = []
    for date in sorted(sub.StudyDate.dropna().unique()):
        rows = sub[sub.StudyDate == date].sort_values("ImageCount",
                                                       ascending=False)
        pick = None
        for _, r in rows.iterrows():
            sdir = find_series_dir(patient_root, r.SeriesInstanceUID)
            if sdir and series_plane(sdir) == "axial":
                pick = r
                break
        if pick is None:                                  # no axial: take largest
            pick = rows.iloc[0]
        chosen.append((date, pick.SeriesInstanceUID, pick.SeriesDescription))

    out = []
    for i, (date, uid, sd) in enumerate(chosen):
        label = "000_baseline" if i == 0 else f"{i:03d}_followup_{i}"
        out.append((label, date, uid, sd))
    return chosen_region, out


# UID -> on-disk series directory, populated from the metadata CSV's optional
# `SeriesDir` column (TCIA manifest downloads name series folders by description,
# not UID, so the glob below can't find them). Empty for API-downloaded
# collections (EAY131, NLST) where folders ARE named by UID -> glob path used.
UID2DIR = {}


def find_series_dir(patient_root, uid):
    d = UID2DIR.get(str(uid))
    if d and os.path.isdir(d):
        return d
    hits = glob.glob(os.path.join(patient_root, "*", uid))
    return hits[0] if hits else None


# ── DICOM -> standardized NIfTI ──────────────────────────────────────────────
def read_ct_series(series_dir):
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(series_dir)
    if not ids:
        return None
    files = reader.GetGDCMSeriesFileNames(series_dir, ids[0])
    reader.SetFileNames(files)
    img = reader.Execute()            # GDCM applies RescaleSlope/Intercept -> HU
    return img


def resample_iso(img, mm, interp=sitk.sitkLinear):
    osp = img.GetSpacing()
    osz = img.GetSize()
    nsz = [max(1, int(round(osz[i] * osp[i] / mm))) for i in range(3)]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing((mm, mm, mm))
    rs.SetSize(nsz)
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputDirection(img.GetDirection())
    rs.SetInterpolator(interp)
    rs.SetDefaultPixelValue(-1024.0)  # air
    return rs.Execute(img)


def build_standard_img(series_dir):
    """DICOM series -> standardized sitk image (RAS+, 1mm iso, float32, HU)."""
    img = read_ct_series(series_dir)
    if img is None:
        return None
    img = sitk.DICOMOrient(img, "RAS")
    img = resample_iso(img, ISO_MM)
    return sitk.Cast(img, sitk.sitkFloat32)


def register_to_baseline(fixed, moving):
    """Rigid (Euler3D) register moving CT -> fixed baseline CT with Mattes MI,
    multi-resolution. Returns (transform, final_metric). Lower metric = better."""
    init = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.10, seed=42)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(init, inPlace=False)
    tx = R.Execute(fixed, moving)
    return tx, float(R.GetMetricValue())


def silhouette_z_init(fixed, moving):
    """Contrast-independent coarse translation: align the two scans along z by
    cross-correlating their per-slice body-area profiles, and centre them in
    x/y (geometry). Mutual-information can grossly slide a partial-FOV follow-up
    along z (it locks onto a textured sub-region); this profile match is immune
    to that and to contrast phase. Returns (Euler3DTransform, corr) where corr
    rates how confidently the silhouettes line up (≈1 = strong, single-peak)."""
    af = (sitk.GetArrayFromImage(fixed) > -500).sum(axis=(1, 2)).astype(float)
    am = (sitk.GetArrayFromImage(moving) > -500).sum(axis=(1, 2)).astype(float)
    af0 = (af - af.mean()) / (af.std() + 1e-6)
    am0 = (am - am.mean()) / (am.std() + 1e-6)
    best_s, best_c = 0, -1e9
    for s in range(-len(am) + 20, len(af) - 20):       # moving slice i ↔ fixed i+s
        lo, hi = max(0, s), min(len(af), len(am) + s)
        if hi - lo < 30:
            continue
        c = float(np.corrcoef(af0[lo:hi], am0[lo - s:hi - s])[0, 1])
        if c > best_c:
            best_c, best_s = c, s
    geo = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY).GetTranslation()
    # z translation (1mm iso, RAS identity dir): fixed point + t samples moving;
    # fixed slice k ↔ moving slice k-s ⇒ t_z = origin_m.z - origin_f.z - s.
    tz = moving.GetOrigin()[2] - fixed.GetOrigin()[2] - best_s
    e = sitk.Euler3DTransform()
    e.SetTranslation((geo[0], geo[1], tz))
    return e, best_c


def body_dice(fixed, resampled):
    """Contrast-independent registration QC: Dice of the body silhouette
    (HU > -500) between the fixed baseline and a resampled follow-up, computed
    over the z-range where BOTH have body (so partial coverage isn't penalised).

    This is the metric to trust — Mattes-MI values swing with contrast phase
    (arterial/portal/delayed) and read low even when the spine is perfectly
    aligned. A low Dice means the two scans don't share anatomy (e.g. a chest
    timepoint vs an abdomen baseline), not that the optimiser failed."""
    a = sitk.GetArrayFromImage(fixed) > -500          # z, y, x
    b = sitk.GetArrayFromImage(resampled) > -500
    za = np.where(a.sum(axis=(1, 2)) > 0)[0]
    zb = np.where(b.sum(axis=(1, 2)) > 0)[0]
    if len(za) == 0 or len(zb) == 0:
        return 0.0
    z0, z1 = max(za.min(), zb.min()), min(za.max(), zb.max())
    if z1 <= z0:
        return 0.0                                    # no shared z-extent
    a, b = a[z0:z1 + 1], b[z0:z1 + 1]
    denom = a.sum() + b.sum()
    return float(2 * (a & b).sum() / denom) if denom else 0.0


def write_img(img, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(img, out_path)


# ── montage (CT analogue of revisualize) ─────────────────────────────────────
def body_z_range(arr):
    """(z0, z1) bounding the body-bearing axial slices (RAS z = axis 2)."""
    body = (arr > -500).sum(axis=(0, 1))            # voxels clearly not air
    zs = np.where(body > body.max() * 0.1)[0] if body.max() > 0 else []
    if len(zs) == 0:
        return 0, arr.shape[2] - 1
    return int(zs.min()), int(zs.max())


def body_z_levels(arr, fracs=(0.35, 0.50, 0.65)):
    """Fractional axial levels within a single volume's body z-range."""
    z0, z1 = body_z_range(arr)
    return [int(z0 + f * (z1 - z0)) for f in fracs]


def axial(arr, z):
    z = max(0, min(z, arr.shape[2] - 1))
    return np.rot90(arr[:, :, z])


def render_patient(pat_out_dir, out_png, pid):
    stages = sorted(d for d in os.listdir(pat_out_dir)
                    if os.path.isdir(os.path.join(pat_out_dir, d))
                    and d[:1].isdigit())
    vols = []
    for st in stages:
        p = os.path.join(pat_out_dir, st, "ct.nii.gz")
        if os.path.exists(p):
            vols.append((st, nib.load(p).get_fdata().astype(np.float32)))
    if not vols:
        return False
    col_labels = ["lower", "mid", "upper"]
    base_arr = vols[0][1]
    same_grid = all(arr.shape == base_arr.shape for _, arr in vols)
    # Follow-ups are registered onto the baseline grid, so the SAME z-index is
    # the same anatomy across rows. Anchor the QC levels to the z-range every
    # timepoint actually covers (intersection of body z-ranges) -> each row
    # shows comparable, present anatomy and partial-coverage scans no longer
    # render empty panels. If a follow-up barely overlaps the baseline (region
    # mismatch), the intersection collapses; fall back to the baseline's own
    # range so the montage still renders (the low body-Dice flags the mismatch).
    if same_grid:
        ranges = [body_z_range(arr) for _, arr in vols]
        z0 = max(r[0] for r in ranges); z1 = min(r[1] for r in ranges)
        if z1 - z0 < 30:                                # negligible overlap
            z0, z1 = body_z_range(base_arr)
        common_levels = [int(z0 + f * (z1 - z0)) for f in (0.35, 0.50, 0.65)]
    else:
        common_levels = None
    nrows, ncols = len(vols), 3
    fig = plt.figure(figsize=(ncols * 2.8, nrows * 2.8 + 0.7),
                     facecolor="#111111")
    fig.suptitle(f"{pid}  ·  {DATASET} (abdomen CT, L40/W400)",
                 color="white", fontsize=12, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.06, wspace=0.04,
                           top=0.93, bottom=0.02, left=0.10, right=0.98)
    for r, (st, arr) in enumerate(vols):
        levels = common_levels if (common_levels and arr.shape == base_arr.shape) \
            else body_z_levels(arr)
        for c, z in enumerate(levels):
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor("#000000"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(col_labels[c], color="white", fontsize=10, pad=4)
            if c == 0:
                ax.set_ylabel(st, color="#cccccc", fontsize=8, labelpad=5)
            ax.imshow(axial(arr, z), cmap="gray", vmin=WIN_LO, vmax=WIN_HI,
                      aspect="equal", interpolation="bilinear")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight",
                facecolor="#111111", edgecolor="none")
    plt.close(fig)
    return True


# ── driver ───────────────────────────────────────────────────────────────────
def main():
    global DATASET
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset", default="EAY131",
                    help="dataset id = output subfolder, e.g. EAY131 | NLST. "
                         "Metadata defaults to <input>/<dataset-lower>_"
                         "metadata_series.csv (override with --meta-csv).")
    ap.add_argument("--meta-csv", default=None,
                    help="series metadata CSV (cols PatientID, Modality, "
                         "StudyDate, SeriesInstanceUID, SeriesDescription, "
                         "StudyDesc, ImageCount). Same schema for EAY131/NLST.")
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--n-debug", type=int, default=0,
                    help="process first N patients (>=2 timepoints) as a sample")
    ap.add_argument("--no-register", action="store_true",
                    help="skip rigid longitudinal registration to baseline")
    ap.add_argument("--region", default="auto",
                    help="timepoint region screen: auto (default; per-patient "
                         "pick the region with the most timepoints, abdomen on "
                         "tie) | abdomen | chest | any")
    ap.add_argument("--min-timepoints", type=int, default=1,
                    help="only write patients with at least this many usable "
                         "same-region CT timepoints (2 = genuine longitudinal "
                         "cohort; single-volume patients are skipped)")
    args = ap.parse_args()
    DATASET = args.dataset

    import pandas as pd
    meta_csv = args.meta_csv or os.path.join(
        args.input, f"{DATASET.lower()}_metadata_series.csv")
    meta = pd.read_csv(meta_csv)
    if "SeriesDir" in meta.columns:
        UID2DIR.update({str(u): d for u, d in
                        zip(meta.SeriesInstanceUID.astype(str), meta.SeriesDir)
                        if isinstance(d, str) and d})
    # Enumerate patients straight from the series metadata (per-patient CT
    # timepoint count = #distinct study dates with a CT series) instead of a
    # separate patient CSV — keeps this dataset-agnostic.
    ct = meta[meta.Modality == "CT"]
    tp_count = ct.groupby("PatientID").StudyDate.nunique()

    out_root = os.path.join(args.output, DATASET)
    os.makedirs(out_root, exist_ok=True)
    fh = open(os.path.join(out_root, "processing_log.txt"), "a")
    if UID2DIR:
        log(f"loaded {len(UID2DIR)} SeriesDir paths from metadata", fh)

    if args.patients:
        pids = args.patients
    elif args.n_debug:
        # sample: prefer multi-timepoint patients so montages show longitudinal
        pids = tp_count[tp_count >= 2].sort_values(
            ascending=False).index.tolist()[:args.n_debug]
    else:
        pids = sorted(tp_count.index.tolist())                     # full cohort
    pids = [str(p) for p in pids]            # NLST PatientIDs are numeric
    log(f"processing {len(pids)} patients (region={args.region}) -> {out_root}",
        fh)

    summary = {}
    regions_used = {}
    for pid in pids:
        chosen_region, series = select_series(
            meta, pid, os.path.join(args.input, pid), region=args.region)
        if not series:
            log(f"  {pid}: no usable CT series, skip", fh); continue
        if len(series) < args.min_timepoints:
            log(f"  {pid}: {len(series)} same-region CT tp < "
                f"min {args.min_timepoints}, skip (not longitudinal)", fh)
            continue
        regions_used[pid] = chosen_region
        log(f"  {pid}: region={chosen_region}, {len(series)} timepoint(s)", fh)
        pat_out = os.path.join(out_root, pid)
        # 1. build standardized images for every timepoint
        imgs = []
        for label, date, uid, sd in series:
            sdir = find_series_dir(os.path.join(args.input, pid), uid)
            if not sdir:
                log(f"  {pid}/{label}: series dir for {uid} missing", fh); continue
            try:
                img = build_standard_img(sdir)
                if img is not None:
                    imgs.append([label, date, uid, sd, img])
            except Exception as e:
                log(f"  {pid}/{label}: ERROR {e}", fh)
        if not imgs:
            log(f"  {pid}: no images built, skip", fh); continue

        # 2. rigid-register every follow-up onto the baseline grid
        baseline = imgs[0][4]
        base_dice = body_dice(baseline, baseline)   # 1.0 sanity reference
        tps = []
        for i, (label, date, uid, sd, img) in enumerate(imgs):
            reg = {"registered": False, "metric": None, "body_dice": None,
                   "overlap_ok": None}
            if i == 0:
                out_img = img
            elif not args.no_register:
                try:
                    # Three candidate alignments; keep the one with the best
                    # body-silhouette Dice. MI is usually best, but it can
                    # grossly slide a partial-FOV follow-up along z — in which
                    # case the silhouette-z (contrast-independent) or the plain
                    # geometry-centred candidate wins and rescues it.
                    tx, metric = register_to_baseline(baseline, img)
                    sz, sz_corr = silhouette_z_init(baseline, img)
                    geo = sitk.CenteredTransformInitializer(
                        baseline, img, sitk.Euler3DTransform(),
                        sitk.CenteredTransformInitializerFilter.GEOMETRY)
                    cands = {"mi": tx, "silhouette_z": sz, "centred": geo}
                    scored = {k: (t, body_dice(baseline,
                              sitk.Resample(img, baseline, t, sitk.sitkLinear,
                                            -1024.0, sitk.sitkFloat32)))
                              for k, t in cands.items()}
                    how = max(scored, key=lambda k: scored[k][1])
                    best_tx, dice = scored[how]
                    out_img = sitk.Resample(img, baseline, best_tx,
                                            sitk.sitkLinear, -1024.0,
                                            sitk.sitkFloat32)
                    if how != "mi":
                        log(f"  {pid}/{label}: MI Dice {scored['mi'][1]:.3f} "
                            f"< {how} {dice:.3f}; used {how}", fh)
                    # body-Dice < 0.70 ⇒ the two scans don't really image the
                    # same region (e.g. an abdomen baseline paired with a chest
                    # follow-up): their torsos still overlap enough for a
                    # mediocre Dice, but the registration is meaningless. Valid
                    # same-region pairs in this collection sit at 0.72–0.90;
                    # genuine cross-region mismatches fall to ~0.67. Flag it.
                    reg = {"registered": True, "metric": round(metric, 4),
                           "body_dice": round(dice, 4), "aligned_by": how,
                           "silhouette_corr": round(sz_corr, 3),
                           "overlap_ok": bool(dice >= 0.70)}
                except Exception as e:
                    log(f"  {pid}/{label}: registration failed ({e}); native", fh)
                    out_img = img
            else:
                out_img = img
            out_ct = os.path.join(pat_out, label, "ct.nii.gz")
            write_img(out_img, out_ct)
            log(f"  {pid}/{label}: {sd} -> {out_img.GetSize()} "
                f"reg={reg['registered']} dice={reg['body_dice']} "
                f"overlap_ok={reg['overlap_ok']}", fh)
            tps.append({"stage": label, "study_date": date, "series_uid": uid,
                        "series_desc": sd, "shape": list(out_img.GetSize()),
                        **reg})
        if tps:
            png = os.path.join(out_root, "visualization", f"{pid}.png")
            # The montage is a QC nicety, not the deliverable: never let a bad
            # read (e.g. a transient truncated ct.nii.gz under heavy concurrent
            # I/O) crash the whole shard and skip the remaining patients.
            try:
                render_patient(pat_out, png, pid)
                log(f"  {pid}: montage -> {png}", fh)
            except Exception as e:
                log(f"  {pid}: montage failed ({e}); ct.nii.gz still written", fh)
            summary[pid] = {"region": chosen_region, "timepoints": tps}

    meta_out = {
        "dataset": DATASET,
        "standard": {
            "orientation": "RAS+", "spacing_mm": [ISO_MM] * 3,
            "dtype": "float32", "values": "raw Hounsfield Units",
            "recommended_window": {"level": 40, "width": 400,
                                   "clip": [WIN_LO, WIN_HI], "scale_to": [0, 1]},
            "timepoint_order": "by study date: 000_baseline, 001_followup_N",
            "series_selection": "primary axial CT (max slices, excl "
                                "scout/localizer/recon/MIP/cor/sag)",
            "region_filter": args.region,
            "region_per_patient": ("auto = the region (abdomen/chest) giving "
                                   "the most timepoints, abdomen on tie; see "
                                   "each patient's 'region' field")
            if args.region == "auto" else args.region,
            "registration": ("rigid Euler3D (Mattes MI) of each follow-up to "
                             "000_baseline, resampled onto baseline grid. QC = "
                             "body-silhouette Dice over shared z-range (trust "
                             "this, not MI which swings with contrast phase); "
                             "overlap_ok=False ⇒ timepoints don't share anatomy")
            if not args.no_register else "none",
        },
        "patients": summary,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    # Shard-safe output name: parallel patient-subset runs set STD_SHARD_TAG so
    # they each write metadata_<tag>.json instead of clobbering one another's
    # metadata.json (per-patient ct.nii.gz / montages never collide). A later
    # merge step combines the shards back into a single metadata.json.
    _tag = os.environ.get("STD_SHARD_TAG", "")
    _meta_name = f"metadata_{_tag}.json" if _tag else "metadata.json"
    with open(os.path.join(out_root, _meta_name), "w") as f:
        json.dump(meta_out, f, indent=2)
    log(f"wrote {_meta_name} ({len(summary)} patients)", fh)
    fh.close()


if __name__ == "__main__":
    main()
