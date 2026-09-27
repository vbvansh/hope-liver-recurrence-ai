#!/usr/bin/env python3
"""Longitudinal abdomen-CT STANDARD: cross-timepoint registration + intensity
("colour") matching, so a baseline-vs-follow-up subtraction reflects the lesion
change and not the acquisition protocol.

This is a NEW pipeline (it does not use the old standardize.py / register.py).
It REUSES the already-computed intermediates under <processed-in> (default
/media/cbtil3/WhiteSD/CTProcessed), which are all on one shared per-patient grid:

  <DS>/<pid>/<stage>/ct.nii.gz            standardized 1mm RAS float32 HU CT
  <DS>/_cads551/<pid>/<stage>_seg.nii.gz  CADS Task-551 organ multilabel (THE organ seg)
  <DS>/_segvol/<pid>/<stage>_tumor.nii.gz SegVol lesion mask

and produces, under <out> (default /media/cbtil3/WhiteSD/CTProcess/Abdomen):

  <DS>/<pid>/000_baseline/{ct,organs,tumor}.nii.gz      reference (copied/relabelled)
  <DS>/<pid>/00N_followup_N/{ct,organs,tumor}.nii.gz    registered + intensity-matched
  <DS>/<pid>/00N_followup_N/diff.nii.gz                 follow-up minus baseline (HU)
  <DS>/<pid>/00N_followup_N/xfm/                         persisted ANTs transform
  <DS>/visualization/<pid>.png                          QC: rows=timepoints + Δ row
  <DS>/metadata.json, processing_log.txt

THE TWO STANDARDS  (see registration.md for the design rationale)
  1. Registration (spatial / pixel alignment). Intensity MI is fooled by contrast
     phase, so we drive registration on ANATOMY: ANTs SyNRA on the signed-distance
     maps of the SHARED organ set -- the organs present in BOTH timepoints (an
     organ truncated out of one scan's FOV is dropped, so only the overlap is
     matched). A union channel sets global pose; per-organ channels for the
     largest shared organs land each on its counterpart (liver<->liver, ...). The
     CTs already share a grid (rigid baseline), so this is a deformable refinement.
     SyN runs on 2mm SDMs (smooth -> downsampling is lossless and ~8x faster); the
     transform is persisted (xfm/) and applied to the full-res CT + organ + tumor
     masks identically. Kept only when mean per-organ Dice beats the rigid result.
  2. Intensity / colour matching. After alignment, the follow-up is mapped onto the
     baseline by an organ-anchored monotone piecewise-linear transform (one knot per
     shared organ's median HU) fit ONLY on non-lesion tissue (tumor dilated out).
     Healthy aligned tissue then subtracts to ~0; the lesion keeps its true HU
     change. HU is already calibrated, so this corrects residual contrast/kernel/
     dose drift, not real anatomy. Falls back to body-tissue Nyul if too few organs.

The match keys off the auto-detected shared organ set; --anchor (CADS-551 ids or
"auto") now only labels the montage level and biases the per-organ channel order.
"""
import argparse
import datetime
import json
import os
import shutil

import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

WIN_LO, WIN_HI = -160.0, 240.0          # abdomen soft-tissue window L40/W400
SDM_CLIP_MM = 40.0                      # signed-distance clip (must exceed organ shift)
REG_MM = 2.0                            # registration grid for the SDM channels
TUMOR_DILATE = 3                        # voxels to grow the lesion before excluding it
DICE_FLOOR = 0.20                       # below this the anchor barely overlaps
# CADS-551 soft organs usable as an "auto" anchor, large-to-small preference order.
AUTO_ORGANS = {"liver": 5, "spleen": 1, "stomach": 6, "kidney_l": 3,
               "kidney_r": 2, "pancreas": 10}

# ── multi-organ matching (registration.md) ────────────────────────────────────
# The SHARED organ set (organs present in BOTH timepoints) drives geometry,
# colour and the accept gate. An organ truncated out of one scan's FOV fails the
# "present in both" test and is dropped -> that IS the "use only the overlap"
# rule. CADS-551 here is a 17-label subset (no skeleton), so aorta/IVC are the
# vertical anchors that stand in for it.
CADS_IDS = tuple(range(1, 18))
CADS551_NAME = {1: "spleen", 2: "kidney_r", 3: "kidney_l", 4: "gallbladder",
                5: "liver", 6: "stomach", 7: "aorta", 8: "ivc",
                9: "portal_splenic_vein", 10: "pancreas", 11: "adrenal_r",
                12: "adrenal_l", 13: "lung_ul_l", 14: "lung_ll_l",
                15: "lung_ul_r", 16: "lung_ml_r", 17: "lung_ll_r"}
VMIN_VOX = 1500            # organ must exceed this in BOTH timepoints to be shared
TOPK = 5                  # per-organ deformable channels (largest shared organs)
VERTICAL_IDS = (7, 8)     # aorta, IVC -- z/roll anchors (substitute for skeleton)
VERTICAL_FLOOR_W = 0.3    # floor weight for the vertical anchors
DICE_MARGIN = 0.02        # deformable must beat rigid mean-Dice by this to be kept
# seg-quality-driven robustness (seg_quality.py findings, registration.md 7b):
# per-pair organ volume ratio must stay in this band, else the organ is truncated
# out of one FOV or seg-failed -> drop it from the driving set (RIDER/PDA fix).
VOL_RATIO_LO, VOL_RATIO_HI = 0.6, 1.667
# reliability prior per CADS-551 id (vol_cv across timepoints): tier-1 stable
# (liver/spleen/kidneys/aorta/IVC/lungs), tier-2 moderate (pancreas/stomach),
# tier-3 erratic (gallbladder/adrenals/portal vein). Multiplies the channel weight.
RELIABILITY = {5: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 7: 1.0, 8: 1.0,
               13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0,
               10: 0.5, 6: 0.4, 4: 0.2, 11: 0.2, 12: 0.2, 9: 0.2}
NCC_TOL = 0.02            # deformable may not worsen gradient-NCC by more than this
# iteration-2: CC-driven SyN refine on the matched CT after the organ-SDM step,
# to pull INTERNAL structure (vessels, parenchyma, mobile kidneys) into register
# that boundary-only organ-SDM registration leaves off. CC is invariant to
# residual linear intensity, so contrast differences don't drive it. Toggle off
# with LONG_REFINE=0 (it roughly doubles per-follow-up registration time).
REFINE = os.environ.get("LONG_REFINE", "0") != "0"   # full-res SyNCC: too slow, off
PSNR_TOL = 0.3            # processed must not lose >this much PSNR vs raw, else
                         # revert (do-no-harm: never ship a worse follow-up).


# ── small helpers ─────────────────────────────────────────────────────────────
def log(msg, fh=None):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def stages(pat_dir):
    """Ordered timepoint folders that actually hold a ct.nii.gz."""
    return sorted(d for d in os.listdir(pat_dir)
                  if d[:1].isdigit() and ("baseline" in d or "followup" in d)
                  and os.path.exists(os.path.join(pat_dir, d, "ct.nii.gz")))


def dice(a, b):
    a = a.astype(bool); b = b.astype(bool)
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 0.0


def body(ct):
    return ct > -500


def grad_ncc(a, b, mask):
    """Contrast-robust, CADS-INDEPENDENT alignment score: normalized cross-
    correlation of CT GRADIENT magnitudes within `mask` (organ/structure edges
    align regardless of contrast phase, and it uses no organ labels, so it breaks
    the circular 'CADS judging a CADS-driven warp' gate). ~[-1,1], higher=better."""
    if mask.sum() < 1000:
        return float("nan")
    ga = np.sqrt(sum(g * g for g in np.gradient(np.clip(a, WIN_LO, WIN_HI))))[mask]
    gb = np.sqrt(sum(g * g for g in np.gradient(np.clip(b, WIN_LO, WIN_HI))))[mask]
    ga = ga - ga.mean(); gb = gb - gb.mean()
    da = np.sqrt((ga * ga).sum()); db = np.sqrt((gb * gb).sum())
    return float((ga * gb).sum() / (da * db)) if da > 0 and db > 0 else float("nan")


def psnr(a, b, mask, peak=WIN_HI - WIN_LO):
    """PSNR (dB) of soft-tissue-windowed CT `b` vs reference `a` over `mask`.
    Higher = the two CTs agree better on that region. We report it on the common
    FOV with the tumor excluded, before vs after processing, so it measures how
    much registration + colour-match made the HEALTHY anatomy line up."""
    if mask.sum() < 1000:
        return float("nan")
    d = (np.clip(a, WIN_LO, WIN_HI)[mask] - np.clip(b, WIN_LO, WIN_HI)[mask])
    mse = float(np.mean(d * d))
    return float(20 * np.log10(peak) - 10 * np.log10(mse)) if mse > 0 else float("inf")


def com_vox(mask):
    """Voxel centre-of-mass of a binary mask, or None if empty."""
    if mask.sum() == 0:
        return None
    idx = np.array(np.nonzero(mask), dtype=np.float64)
    return idx.mean(axis=1)


def to_base_grid(arr, src_nib, base_nib, order):
    """Resample a follow-up array onto the baseline voxel grid when their shapes
    differ (reused data is usually already co-gridded; this is the safety net)."""
    import ants
    if arr.shape == base_nib.shape:
        return arr
    interp = "genericLabel" if order == 0 else "linear"
    ref = _ants_from(np.zeros(base_nib.shape, np.float32), base_nib)
    out = ants.apply_transforms(ref, _ants_from(arr.astype(np.float32), src_nib),
                                transformlist=[], interpolator=interp,
                                defaultvalue=(-1024.0 if order != 0 else 0.0))
    return out.numpy()


# ── registration ──────────────────────────────────────────────────────────────
def _ants_from(arr, ref_nib):
    import ants
    return ants.from_numpy(
        arr, origin=[float(x) for x in ref_nib.affine[:3, 3]],
        spacing=[float(abs(ref_nib.affine[i, i])) for i in range(3)])


def _sdm_lowres(mask_full, ref_nib):
    """Signed-distance map of a full-res binary mask, computed at REG_MM (fast,
    smooth). Returns an ANTs image on the downsampled grid."""
    import ants
    a = _ants_from(mask_full.astype(np.float32), ref_nib)
    a = ants.resample_image(a, (REG_MM, REG_MM, REG_MM), False, 1)  # nearest
    m = a.numpy() > 0.5
    if m.sum() == 0:
        sdm = np.full(m.shape, SDM_CLIP_MM, np.float32)
    else:
        d = (ndimage.distance_transform_edt(~m)
             - ndimage.distance_transform_edt(m)) * REG_MM
        sdm = np.clip(d, -SDM_CLIP_MM, SDM_CLIP_MM).astype(np.float32)
    out = a.new_image_like(sdm)
    return out


# ── intensity / colour matching ───────────────────────────────────────────────
def match_intensity(moving, fixed, mask_mov, mask_fix, lo=-150.0, hi=300.0):
    """Monotone piecewise-linear landmark map of `moving` onto `fixed`, fit on the
    masked, soft-tissue-windowed voxels (Nyul). Applied to the whole volume."""
    ps = np.concatenate([[0.5, 1, 2], np.arange(5, 100, 5), [98, 99, 99.5]])
    sm = moving[mask_mov]; sf = fixed[mask_fix]
    sm = sm[(sm >= lo) & (sm <= hi)]; sf = sf[(sf >= lo) & (sf <= hi)]
    if sm.size < 100 or sf.size < 100:
        return moving.astype(np.float32)            # too little tissue, skip
    lm = np.maximum.accumulate(np.percentile(sm, ps))
    lf = np.maximum.accumulate(np.percentile(sf, ps))
    xp = np.concatenate([[moving.min() - 1], lm, [moving.max() + 1]])
    fp = np.concatenate([[moving.min() - 1 + (lf[0] - lm[0])], lf,
                         [moving.max() + 1 + (lf[-1] - lm[-1])]])
    for i in range(1, len(xp)):                     # strictly increasing knots
        if xp[i] <= xp[i - 1]:
            xp[i] = xp[i - 1] + 1e-3
    return np.interp(moving, xp, fp).astype(np.float32)


def match_intensity_organ(moving, fixed, mseg_w, bseg, shared, excl,
                          lo=-150.0, hi=300.0):
    """Organ-anchored monotone HU remap (the registration.md choice): one knot
    per shared organ = (median HU in the moving organ -> median HU in the
    baseline organ), measured on non-lesion tissue, plus an air anchor. Monotone,
    applied to the whole volume. Returns None when too few organs anchor it, so
    the caller can fall back to the body-tissue Nyul match."""
    bm = body(moving) & ~excl
    bf = body(fixed) & ~excl
    xs, ys = [-1000.0], [-1000.0]                   # air anchor
    for L in shared:
        mm = (mseg_w == L) & bm
        mf = (bseg == L) & bf
        if mm.sum() < 50 or mf.sum() < 50:
            continue
        xm = float(np.median(moving[mm])); xf = float(np.median(fixed[mf]))
        if lo <= xm <= hi and lo <= xf <= hi:
            xs.append(xm); ys.append(xf)
    if len(xs) < 3:                                 # air + <2 organs -> fallback
        return None
    order = np.argsort(xs)
    xs = np.array(xs)[order]; ys = np.maximum.accumulate(np.array(ys)[order])
    for i in range(1, len(xs)):                     # strictly increasing knots
        if xs[i] <= xs[i - 1]:
            xs[i] = xs[i - 1] + 1e-3
    xp = np.concatenate([[moving.min() - 1], xs, [moving.max() + 1]])
    fp = np.concatenate([[ys[0] + (moving.min() - 1 - xs[0])], ys,
                         [ys[-1] + (moving.max() + 1 - xs[-1])]])  # slope-1 ends
    return np.interp(moving, xp, fp).astype(np.float32)


# ── visualization ─────────────────────────────────────────────────────────────
def axial(a, z):
    z = max(0, min(z, a.shape[2] - 1))
    return np.rot90(a[:, :, z])


def render_patient(pat_out, out_png, pid, dataset, organ_label, anchor_idx):
    sts = stages(pat_out)
    vols, organs, tumors = [], [], []
    for st in sts:
        vols.append((st, nib.load(os.path.join(pat_out, st, "ct.nii.gz"))
                     .get_fdata().astype(np.float32)))
        op = os.path.join(pat_out, st, "organs.nii.gz")
        tp = os.path.join(pat_out, st, "tumor.nii.gz")
        organs.append(nib.load(op).get_fdata().astype(np.int16)
                      if os.path.exists(op) else None)
        tumors.append(nib.load(tp).get_fdata() > 0 if os.path.exists(tp) else None)
    if not vols:
        return False
    base_arr = vols[0][1]
    # montage z-levels: prefer the BASELINE TUMOR (this is a progression QC, so the
    # lesion must be in frame), else the anchor organ, else evenly across the body.
    seg0 = organs[0]
    tum0 = tumors[0]
    anc0 = np.isin(seg0, anchor_idx) if seg0 is not None else None
    focus = tum0 if (tum0 is not None and tum0.sum() > 0) else anc0
    if focus is not None and focus.sum() > 0:
        prof = focus.sum(axis=(0, 1)).astype(float)
        zc = int(round(np.average(np.arange(len(prof)), weights=prof)))
        zext = np.where(focus.any(axis=(0, 1)))[0]
        d = max(15, int((zext.max() - zext.min()) // 4))
        levels = [max(0, zc - d), zc, min(base_arr.shape[2] - 1, zc + d)]
    else:
        b = (base_arr > -500).sum(axis=(0, 1))
        z = np.where(b > b.max() * 0.1)[0] if b.max() > 0 else [0]
        z0, z1 = int(z.min()), int(z.max())
        levels = [int(z0 + f * (z1 - z0)) for f in (0.35, 0.5, 0.65)]
    # shared crop-to-body box
    foot = None
    for _, arr in vols:
        if arr.shape == base_arr.shape:
            fp = np.rot90((arr > -500).any(axis=2))
            foot = fp if foot is None else (foot | fp)
    crop = None
    if foot is not None and foot.any():
        ys, xs = np.nonzero(foot); pad = 6
        crop = (foot.shape, (slice(max(0, ys.min() - pad), min(foot.shape[0], ys.max() + pad + 1)),
                             slice(max(0, xs.min() - pad), min(foot.shape[1], xs.max() + pad + 1))))

    def fit(sl):
        return sl[crop[1]] if (crop and sl.shape == crop[0]) else sl

    nrows = len(vols) + (1 if len(vols) > 1 else 0)   # extra Δ row
    ncols = 3
    fig = plt.figure(figsize=(ncols * 2.8, nrows * 2.8 + 0.7), facecolor="#111111")
    fig.suptitle(f"{pid}  ·  {dataset}  (CT L40/W400, anchor={organ_label}; "
                 f"tumor = red, anchor organ = cyan; bottom row = follow-up−baseline)",
                 color="white", fontsize=11, fontweight="bold", y=0.997)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.06, wspace=0.04,
                           top=0.94, bottom=0.02, left=0.10, right=0.98)
    for r, (st, arr) in enumerate(vols):
        tm = tumors[r]
        og = organs[r]
        for c, z in enumerate(levels):
            ax = fig.add_subplot(gs[r, c])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#000")
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(["inferior", f"{organ_label} centre", "superior"][c],
                             color="white", fontsize=9, pad=4)
            if c == 0:
                ax.set_ylabel(st, color="#ccc", fontsize=8)
            ax.imshow(fit(axial(arr, z)), cmap="gray", vmin=WIN_LO, vmax=WIN_HI,
                      aspect="equal", interpolation="bilinear")
            # Brain-standard overlay: tumor mask in semi-transparent red, so the
            # lesion is visible at the SAME registered level across timepoints.
            if tm is not None and tm.shape == arr.shape:
                ts = fit(axial(tm.astype(np.float32), z)) > 0.5
                if ts.any():
                    rgba = np.zeros((*ts.shape, 4), np.float32)
                    rgba[..., 0] = 1.0                 # red
                    rgba[..., 3] = ts * 0.45           # alpha only on the lesion
                    ax.imshow(rgba, aspect="equal", interpolation="nearest")
            # anchor-organ CADS contour (cyan): exposes seg quality + cross-
            # timepoint alignment of the organ the registration was driven on.
            if og is not None and anchor_idx and og.shape == arr.shape:
                om = fit(axial(np.isin(og, anchor_idx).astype(np.float32), z))
                if om.any():
                    ax.contour(om, levels=[0.5], colors="cyan", linewidths=0.6)
    if len(vols) > 1:                       # Δ row: last follow-up minus baseline
        d = (np.clip(vols[-1][1], WIN_LO, WIN_HI)
             - np.clip(base_arr, WIN_LO, WIN_HI))
        for c, z in enumerate(levels):
            ax = fig.add_subplot(gs[nrows - 1, c])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel("Δ (FU−base)", color="#ff8", fontsize=8)
            ax.imshow(fit(axial(d, z)), cmap="bwr", vmin=-150, vmax=150,
                      aspect="equal", interpolation="nearest")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    return True


# ── per-patient ───────────────────────────────────────────────────────────────
def cads_path(ds_in, dataset, pid, st):
    return os.path.join(ds_in, dataset, "_cads551", pid, f"{st}_seg.nii.gz")


def segvol_path(ds_in, dataset, pid, st):
    return os.path.join(ds_in, dataset, "_segvol", pid, f"{st}_tumor.nii.gz")


def load_organ(p):
    return nib.load(p).get_fdata().astype(np.int16) if os.path.exists(p) else None


def load_tumor(p, shape):
    if os.path.exists(p):
        return nib.load(p).get_fdata() > 0
    return np.zeros(shape, bool)


def resolve_anchor(anchor, segs):
    """Return (label, [cads ids]). anchor is a list of ids, or 'auto'."""
    if anchor != "auto":
        return ("+".join(map(str, anchor)), list(anchor))
    if any(s is None for s in segs):
        return (None, [])
    best, best_vol = None, 0
    for nm, idx in AUTO_ORGANS.items():
        vols = [int((s == idx).sum()) for s in segs]
        if min(vols) < 2000:
            continue
        if min(vols) > best_vol:
            best_vol, best = min(vols), (nm, idx)
    return (best[0], [best[1]]) if best else (None, [])


def process_patient(pid, dataset, ds_in, out_root, anchor, do_intensity=True,
                    fh=None):
    pat_in = os.path.join(ds_in, dataset, pid)
    sts = stages(pat_in)
    if len(sts) < 2:
        log(f"  {pid}: <2 timepoints, skip", fh); return None
    pat_out = os.path.join(out_root, pid)

    # load baseline + all organ segs (for auto anchor)
    base_nib = nib.load(os.path.join(pat_in, sts[0], "ct.nii.gz"))
    base_ct = base_nib.get_fdata().astype(np.float32)
    segs = [load_organ(cads_path(ds_in, dataset, pid, st)) for st in sts]
    if segs[0] is None:
        log(f"  {pid}: no baseline organ seg, skip", fh); return None
    # anchor now only labels the montage level; the match keys off the shared set
    organ_label, anchor_idx = resolve_anchor(anchor, segs)
    if not anchor_idx:
        organ_label, anchor_idx = "auto", []

    import ants  # noqa: F401  (ensure available before heavy work)
    base_seg = segs[0]
    base_tumor = load_tumor(segvol_path(ds_in, dataset, pid, sts[0]), base_ct.shape)
    base_tum_dil = (ndimage.binary_dilation(base_tumor, iterations=TUMOR_DILATE)
                    if base_tumor.any() else base_tumor)
    # organs present (>=VMIN) on the baseline; SDMs cached lazily and reused
    # across every follow-up (geometry + colour both key off the shared set).
    base_present = [L for L in CADS_IDS if int((base_seg == L).sum()) >= VMIN_VOX]
    if not base_present:
        log(f"  {pid}: no baseline organ >= {VMIN_VOX} vox, skip", fh); return None
    _bsdm = {}

    def base_sdm(mask, key):
        if key not in _bsdm:
            _bsdm[key] = _sdm_lowres(mask, base_nib)
        return _bsdm[key]

    # baseline outputs (reference; copied as-is)
    _write(base_ct, base_nib, os.path.join(pat_out, sts[0], "ct.nii.gz"))
    _write(base_seg.astype(np.int16), base_nib, os.path.join(pat_out, sts[0], "organs.nii.gz"))
    _write(base_tumor.astype(np.uint8), base_nib, os.path.join(pat_out, sts[0], "tumor.nii.gz"))

    tp_meta = [{"stage": sts[0], "role": "baseline", "anchor": organ_label}]
    for st in sts[1:]:
        mnib = nib.load(os.path.join(pat_in, st, "ct.nii.gz"))
        mct = mnib.get_fdata().astype(np.float32)
        mseg = load_organ(cads_path(ds_in, dataset, pid, st))
        mtum = load_tumor(segvol_path(ds_in, dataset, pid, st), mct.shape)
        # safety net: co-grid onto baseline (CTProcessed is normally already on
        # one shared per-patient grid, so this is usually a no-op).
        if mseg is not None and mct.shape != base_ct.shape:
            mct = to_base_grid(mct, mnib, base_nib, 1)
            mseg = to_base_grid(mseg, mnib, base_nib, 0).astype(np.int16)
            mtum = to_base_grid(mtum.astype(np.float32), mnib, base_nib, 0) > 0.5
        mct0 = mct.copy()       # raw co-gridded follow-up (pre shift/warp/colour)
        mseg0 = (mseg.copy() if mseg is not None else None)   # raw, for do-no-harm
        mtum0 = mtum.copy()                                   # revert targets

        # SHARED organ set = organs present (>=VMIN) in BOTH timepoints AND with a
        # sane follow-up/baseline volume ratio. A wild ratio means the organ is
        # truncated out of one FOV or seg-failed -> drop it (the RIDER/PDA fix).
        # Organs absent in one scan drop too -> "use only the overlap".
        shared = []
        if mseg is not None:
            for L in base_present:
                vb = int((base_seg == L).sum()); vm = int((mseg == L).sum())
                if vm >= VMIN_VOX and VOL_RATIO_LO <= vm / vb <= VOL_RATIO_HI:
                    shared.append(L)
        shared.sort(key=lambda L: min(int((base_seg == L).sum()),
                                      int((mseg == L).sum())), reverse=True)
        flags, tx = [], None
        per_organ = {}
        ncc_rig = ncc_def = float("nan")
        if not shared:
            # nothing common (disjoint FOV / failed seg) -> keep reused-rigid copy
            log(f"  {pid}/{st}: no shared organ, kept rigid (copy)", fh)
            w_ct, w_seg = mct, (mseg if mseg is not None else np.zeros_like(base_seg))
            w_tum = mtum
            shift_mm = 0.0; mean_rig = mean_def = float("nan")
            flags.append("NO_SHARED_ORGAN")
        else:
            import ants
            mtum_dil = (ndimage.binary_dilation(mtum, iterations=TUMOR_DILATE)
                        if mtum.any() else mtum)
            b_union = np.isin(base_seg, shared)
            m_union = np.isin(mseg, shared)
            # COM pre-shift on the shared UNION -- ONLY as a rescue for offsets
            # beyond the SyN's reach (SDM clip). The inputs are already rigidly
            # co-registered (the BODY matches), so a small organ offset means the
            # organ moved relative to an aligned body; shifting the whole volume to
            # chase it would knock the body wall out by that much. For those (the
            # common case) we let SyN deform the organ LOCALLY and keep the body.
            delta = com_vox(b_union) - com_vox(m_union)
            shift_mm = float(np.linalg.norm(delta))
            if shift_mm > SDM_CLIP_MM:
                mct = ndimage.shift(mct, delta, order=1, cval=-1024.0, prefilter=False)
                mseg = ndimage.shift(mseg.astype(np.float32), delta, order=0,
                                     cval=0.0, prefilter=False).astype(np.int16)
                mtum = ndimage.shift(mtum.astype(np.float32), delta, order=0,
                                     cval=0.0, prefilter=False) > 0.5
                mtum_dil = (ndimage.binary_dilation(mtum, iterations=TUMOR_DILATE)
                            if mtum.any() else mtum)
                m_union = np.isin(mseg, shared)
            # per-organ channels, TUMOR-MASKED (align HEALTHY organ tissue, not the
            # changing/under-segmented tumor region) and weighted by reliability x
            # size; aorta/IVC keep a floor weight as vertical anchors.
            topk = shared[:TOPK]
            raw = {L: min(int((base_seg == L).sum()), int((mseg == L).sum()))
                   for L in topk}
            wmax = max(raw.values())

            def wt(L):
                w = (raw[L] / wmax) * RELIABILITY.get(L, 0.5)
                return max(w, VERTICAL_FLOOR_W) if L in VERTICAL_IDS else w
            extras = [["meansquares",
                       base_sdm((base_seg == L) & ~base_tum_dil, L),
                       _sdm_lowres((mseg == L) & ~mtum_dil, mnib), wt(L), 0]
                      for L in topk]
            reg = ants.registration(
                fixed=base_sdm(b_union & ~base_tum_dil,
                               "U:" + ",".join(map(str, shared))),
                moving=_sdm_lowres(m_union & ~mtum_dil, mnib),
                type_of_transform="SyNRA",
                aff_metric="meansquares", syn_metric="meansquares",
                multivariate_extras=extras, verbose=False)
            tx = reg["fwdtransforms"]
            ref = _ants_from(base_ct, base_nib)
            w_ct = ants.apply_transforms(ref, _ants_from(mct, mnib), tx,
                                         interpolator="linear",
                                         defaultvalue=-1024.0).numpy()
            w_seg = ants.apply_transforms(
                ref, _ants_from(mseg.astype(np.float32), mnib), tx,
                interpolator="genericLabel").numpy().astype(np.int16)
            w_tum = ants.apply_transforms(
                ref, _ants_from(mtum.astype(np.float32), mnib), tx,
                interpolator="genericLabel").numpy() > 0.5
            # gate 1: reliability-WEIGHTED per-organ Dice, rigid(post-shift) vs warp
            d_rig = {L: dice(base_seg == L, mseg == L) for L in shared}
            d_def = {L: dice(base_seg == L, w_seg == L) for L in shared}
            ws = {L: RELIABILITY.get(L, 0.5) for L in shared}; sw = sum(ws.values())
            mean_rig = float(sum(d_rig[L] * ws[L] for L in shared) / sw)
            mean_def = float(sum(d_def[L] * ws[L] for L in shared) / sw)
            # gate 2 (CADS-INDEPENDENT): gradient-NCC of the CT on healthy overlap
            # tissue -- catches a high-Dice warp that actually mis-aligns the CT.
            ncc_mask = (ndimage.binary_dilation(b_union, iterations=4)
                        & body(base_ct) & ~base_tum_dil & ~mtum_dil)
            ncc_rig = grad_ncc(base_ct, mct, ncc_mask & body(mct))
            ncc_def = grad_ncc(base_ct, w_ct, ncc_mask & body(w_ct))
            keep = (mean_def >= mean_rig + DICE_MARGIN and mean_def >= DICE_FLOOR
                    and (np.isnan(ncc_rig) or np.isnan(ncc_def)
                         or ncc_def >= ncc_rig - NCC_TOL))
            if keep:
                per_organ = {CADS551_NAME.get(L, L): [_r(d_rig[L]), _r(d_def[L])]
                             for L in shared}
            else:                                   # SyN didn't help -> keep rigid
                w_ct, w_seg, w_tum = mct, mseg, mtum
                tx = None
                flags.append("KEPT_RIGID" if mean_def < mean_rig + DICE_MARGIN
                             else "NCC_REJECT")
                per_organ = {CADS551_NAME.get(L, L): _r(d_rig[L]) for L in shared}

        # fine CT refine (BEFORE colour): the organ-SDM SyN aligns organ BOUNDARIES
        # but leaves internal structure (vessels, parenchyma, mobile kidneys) off; a
        # CC-driven SyN on the CT, masked to healthy organ tissue, pulls it into
        # register and lifts PSNR. CC is intensity-robust so contrast doesn't drive
        # it. Only refine an accepted organ-SDM warp.
        if REFINE and shared and tx is not None:
            import ants
            rmask = _ants_from((ncc_mask & body(w_ct)).astype(np.float32), base_nib)
            reg2 = ants.registration(
                fixed=_ants_from(np.clip(base_ct, WIN_LO, WIN_HI), base_nib),
                moving=_ants_from(np.clip(w_ct, WIN_LO, WIN_HI), base_nib),
                type_of_transform="SyNOnly", syn_metric="CC", mask=rmask,
                verbose=False)
            tx2 = reg2["fwdtransforms"]
            ref = _ants_from(base_ct, base_nib)
            w_ct = ants.apply_transforms(ref, _ants_from(w_ct, base_nib), tx2,
                                         interpolator="linear",
                                         defaultvalue=-1024.0).numpy()
            w_seg = ants.apply_transforms(
                ref, _ants_from(w_seg.astype(np.float32), base_nib), tx2,
                interpolator="genericLabel").numpy().astype(np.int16)
            w_tum = ants.apply_transforms(
                ref, _ants_from(w_tum.astype(np.float32), base_nib), tx2,
                interpolator="genericLabel").numpy() > 0.5
            tx = list(tx2) + list(tx)            # composed (refine ∘ organ-SDM)
            flags.append("CC_REFINED")
            d_def = {L: dice(base_seg == L, w_seg == L) for L in shared}
            mean_def = float(sum(d_def[L] * ws[L] for L in shared) / sw)
            per_organ = {CADS551_NAME.get(L, L): [_r(d_rig[L]), _r(d_def[L])]
                         for L in shared}
            ncc_def = grad_ncc(base_ct, w_ct, ncc_mask & body(w_ct))

        excl = ndimage.binary_dilation(base_tumor | w_tum, iterations=TUMOR_DILATE)
        fov = body(base_ct) & body(mct0) & body(w_ct) & ~excl

        def _resid(x):
            if not fov.any():
                return float("inf")
            return float(np.median(np.abs(np.clip(x, WIN_LO, WIN_HI)[fov]
                                          - np.clip(base_ct, WIN_LO, WIN_HI)[fov])))

        # intensity / colour match (organ-anchored monotone), DO-NO-HARM: a monotone
        # map CANNOT fix opposite-sign per-organ offsets (differential contrast
        # phase) -- it then adds a global offset and HURTS. Keep the colour map only
        # when it actually lowers the healthy-tissue residual.
        if do_intensity:
            cand = (match_intensity_organ(w_ct, base_ct, w_seg, base_seg, shared, excl)
                    if shared else None)
            if cand is None:
                cand = match_intensity(w_ct, base_ct, body(w_ct) & ~excl,
                                       body(base_ct) & ~excl)
            if _resid(cand) <= _resid(w_ct):
                w_ct_m = cand
            else:
                w_ct_m = w_ct; flags.append("NO_COLOR")    # differential contrast
        else:
            w_ct_m = w_ct

        # PSNR (common FOV, healthy tissue) raw vs processed.
        psnr_before = psnr(base_ct, mct0, fov)
        psnr_after = psnr(base_ct, w_ct_m, fov)
        # global DO-NO-HARM: if processing made the follow-up WORSE than the raw
        # (over-aggressive warp on few organs / unfixable contrast), ship the raw.
        if psnr_after < psnr_before - PSNR_TOL and mseg0 is not None:
            w_ct_m, w_seg, w_tum, tx = mct0, mseg0, mtum0, None
            flags.append("REVERT_RAW")
            excl = ndimage.binary_dilation(base_tumor | w_tum, iterations=TUMOR_DILATE)
            fov = body(base_ct) & body(mct0) & ~excl
            psnr_after = psnr(base_ct, w_ct_m, fov)
            if shared:
                ncc_def = grad_ncc(base_ct, w_ct_m, ncc_mask & body(w_ct_m))

        _write(w_ct_m, base_nib, os.path.join(pat_out, st, "ct.nii.gz"))
        _write(w_seg, base_nib, os.path.join(pat_out, st, "organs.nii.gz"))
        _write(w_tum.astype(np.uint8), base_nib, os.path.join(pat_out, st, "tumor.nii.gz"))
        diff = (np.clip(w_ct_m, WIN_LO, WIN_HI) - np.clip(base_ct, WIN_LO, WIN_HI))
        _write(diff.astype(np.float32), base_nib, os.path.join(pat_out, st, "diff.nii.gz"))
        # persist the transform so any mask can be re-warped later without re-reg
        if tx:
            xdir = os.path.join(pat_out, st, "xfm"); os.makedirs(xdir, exist_ok=True)
            saved = []
            for k, tpath in enumerate(tx):
                if os.path.exists(tpath):
                    dst = os.path.join(xdir, f"{k}_{os.path.basename(tpath)}")
                    shutil.copy(tpath, dst); saved.append(os.path.basename(dst))
            with open(os.path.join(xdir, "transforms.json"), "w") as jf:
                json.dump({"order": saved, "applies": "followup->baseline"}, jf)

        # QC: residual on aligned healthy shared-organ tissue (should be small)
        anc = ndimage.binary_dilation(np.isin(base_seg, shared) |
                                      np.isin(w_seg, shared), iterations=4)
        paren = anc & ~ndimage.binary_dilation(base_tumor | w_tum, iterations=TUMOR_DILATE)
        paren &= body(base_ct) & body(w_ct_m)
        resid = float(np.median(np.abs(diff[paren]))) if paren.sum() else float("nan")
        log(f"  {pid}/{st}: shared={len(shared)} organs  wDice "
            f"{mean_rig:.3f}->{mean_def:.3f}  gradNCC {ncc_rig:.3f}->{ncc_def:.3f}  "
            f"PSNR {psnr_before:.1f}->{psnr_after:.1f}dB  (COM {shift_mm:.0f}mm)  "
            f"med|Δ|={resid:.1f}HU{'  [' + ','.join(flags) + ']' if flags else ''}", fh)
        tp_meta.append({"stage": st, "role": "followup",
                        "shared_organs": len(shared),
                        "wdice_rigid": _r(mean_rig),
                        "wdice_deformed": _r(mean_def),
                        "gradncc_rigid": _r(ncc_rig),
                        "gradncc_deformed": _r(ncc_def),
                        "psnr_before_db": _r(psnr_before),
                        "psnr_after_db": _r(psnr_after),
                        "per_organ_dice": per_organ,
                        "centroid_shift_mm": _r(shift_mm),
                        "healthy_resid_med_hu": _r(resid),
                        "flags": flags,
                        "intensity_matched": bool(do_intensity)})

    out_png = os.path.join(out_root, "visualization", f"{pid}.png")
    try:
        render_patient(pat_out, out_png, pid, dataset, organ_label, anchor_idx)
    except Exception as e:
        log(f"  {pid}: montage failed ({e})", fh)
    return {"anchor": organ_label, "timepoints": tp_meta}


def _r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def _write(arr, ref_nib, path):
    # affine only (no source header) so nibabel stores each array in its own
    # dtype: float32 CT/diff, int16 organs, uint8 tumor — not coerced to float32.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(arr, ref_nib.affine), path)


# ── driver ────────────────────────────────────────────────────────────────────
def run(dataset, anchor, ds_in, out, patients=None, limit=0,
        do_intensity=True, force=False):
    out_root = os.path.join(out, dataset)
    os.makedirs(os.path.join(out_root, "visualization"), exist_ok=True)
    fh = open(os.path.join(out_root, "processing_log.txt"), "a")
    src = os.path.join(ds_in, dataset)
    all_pids = [d for d in sorted(os.listdir(src))
                if not d.startswith((".", "_")) and d != "visualization"
                and os.path.isdir(os.path.join(src, d))
                and len(stages(os.path.join(src, d))) >= 2]
    if patients:
        all_pids = [p for p in all_pids if p in set(patients)]
    if limit:
        all_pids = all_pids[:limit]
    log(f"=== {dataset}: {len(all_pids)} patients (>=2 tp), anchor={anchor} -> "
        f"{out_root}", fh)
    summary = {}
    for i, pid in enumerate(all_pids, 1):
        # resumable: skip if every follow-up already has ct + diff
        po = os.path.join(out_root, pid)
        sts = stages(os.path.join(src, pid))
        done = all(os.path.exists(os.path.join(po, st, "diff.nii.gz"))
                   for st in sts[1:]) and os.path.exists(
                       os.path.join(po, sts[0], "ct.nii.gz"))
        if done and not force:
            log(f"[{i}/{len(all_pids)}] {pid}: already done, skip", fh); continue
        log(f"[{i}/{len(all_pids)}] {pid}", fh)
        try:
            res = process_patient(pid, dataset, ds_in, out_root, anchor,
                                  do_intensity=do_intensity, fh=fh)
            if res:
                summary[pid] = res
        except Exception as e:
            import traceback
            log(f"  {pid}: ERROR {e}\n{traceback.format_exc()}", fh)
    meta = {"dataset": dataset, "anchor_organ": anchor,
            "processed_in": ds_in, "registration":
            "ANTs SyNRA on 2mm signed-distance maps of the SHARED organ set "
            "(union channel + per-organ channels for the largest shared organs); "
            "mean-per-organ-Dice gated; transform persisted under xfm/ and "
            "applied to full-res CT and masks",
            "intensity_matching":
            "organ-anchored monotone HU remap on non-lesion tissue "
            "(follow-up -> baseline)" if do_intensity else "none",
            "standard": {"orientation": "RAS+", "spacing_mm": [1, 1, 1],
                         "dtype": "float32", "values": "HU"},
            "patients": summary,
            "generated": datetime.datetime.now().isoformat(timespec="seconds")}
    tag = os.environ.get("LONG_SHARD_TAG", "")
    name = f"metadata_{tag}.json" if tag else "metadata.json"
    with open(os.path.join(out_root, name), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote {name} ({len(summary)} new patients)", fh)
    fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--anchor", required=True,
                    help="comma-separated CADS-551 ids (e.g. 2,3) or 'auto'")
    ap.add_argument("--processed-in", default="/media/cbtil3/WhiteSD/CTProcessed")
    ap.add_argument("--out", default="/media/cbtil3/WhiteSD/CTProcess/Abdomen")
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-intensity-match", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="reprocess patients even if their output already exists")
    args = ap.parse_args()
    anchor = "auto" if args.anchor.strip().lower() == "auto" else \
        [int(x) for x in args.anchor.split(",")]
    run(args.dataset, anchor, args.processed_in, args.out,
        patients=args.patients, limit=args.limit,
        do_intensity=not args.no_intensity_match, force=args.force)


if __name__ == "__main__":
    main()
