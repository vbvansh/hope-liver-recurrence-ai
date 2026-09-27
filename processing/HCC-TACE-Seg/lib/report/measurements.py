"""Per-timepoint quantitative measurements from the ALREADY-PROCESSED CT volumes.

Training-free, graceful. The tumor mask comes from the cohort's lesion seg:
  - a shipped GROUND-TRUTH `tumor.nii.gz` in the stage dir (HCC-TACE-Seg, WAW-TACE), else
  - the SegVol pre-annotation at `_segvol/<pid>/<stage>_tumor.nii.gz` (all 7 datasets).
Organ context (which organ the lesion sits in, organ volume) comes from the cached
TotalSegmentator multilabel at `_totalseg/<pid>/<stage>_seg.nii.gz`.

These numbers are what the VLM reasons over (it must not eyeball sizes). The
RECIST-1.1 measure is the lesion's LONGEST AXIAL DIAMETER (mm); tumor VOLUME (cc)
is the burden backbone used for the nadir and the trajectory.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib

# TotalSegmentator v2 'total' task: label id -> organ group (lobes folded to the
# parent organ). Matches segvol_demo.TOTALSEG_ORGAN used to build the masks.
ORGAN_LABELS = {
    1: "spleen", 2: "kidney", 3: "kidney", 4: "gallbladder", 5: "liver",
    6: "stomach", 7: "pancreas", 8: "adrenal", 9: "adrenal",
    10: "lung", 11: "lung", 12: "lung", 13: "lung", 14: "lung",
}


def stage_dirs(patient_dir: Path) -> List[Path]:
    return sorted(d for d in patient_dir.iterdir()
                  if d.is_dir() and d.name[:1].isdigit() and not d.name.startswith("."))


def _voxel_cc(affine) -> float:
    return float(abs(np.linalg.det(affine[:3, :3])) / 1000.0)  # mm^3 -> cc


def _voxel_mm(affine) -> Tuple[float, float, float]:
    return tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))


def ct_path(stage_dir: Path) -> Optional[Path]:
    for name in ("ct.nii.gz", "ct_labelguided.nii.gz", "ct_rigid.nii.gz"):
        p = stage_dir / name
        if p.exists():
            return p
    return None


def tumor_mask_path(proc_ds: Path, patient: str, stage_dir: Path) -> Tuple[Optional[Path], str]:
    """Lesion mask + its source. Prefer a shipped GT mask, else the SegVol pre-annotation."""
    gt = stage_dir / "tumor.nii.gz"
    if gt.exists():
        return gt, "gt"
    sv = proc_ds / "_segvol" / patient / f"{stage_dir.name}_tumor.nii.gz"
    if sv.exists():
        return sv, "segvol"
    return None, "none"


def organ_seg_path(proc_ds: Path, patient: str, stage: str) -> Optional[Path]:
    p = proc_ds / "_totalseg" / patient / f"{stage}_seg.nii.gz"
    return p if p.exists() else None


def longest_diameter_mm(mask: np.ndarray, affine) -> Optional[float]:
    """RECIST-1.1 longest diameter: the longest straight line within a single axial
    slice of the lesion. Computed on the few largest-area axial slices (cheap,
    numpy-only). SegVol/GT masks are already largest-connected-component, so this
    measures the dominant lesion, not scattered noise."""
    if mask.sum() == 0:
        return None
    sx, sy, _ = _voxel_mm(affine)
    areas = mask.sum(axis=(0, 1))
    best = 0.0
    for z in np.argsort(areas)[-5:]:          # top-5 axial slices by area
        if areas[z] == 0:
            continue
        xs, ys = np.nonzero(mask[:, :, int(z)])
        if xs.size < 2:
            continue
        pts = np.stack([xs * sx, ys * sy], axis=1)
        if pts.shape[0] > 400:                # bound the O(n^2) pairwise step
            sel = np.linspace(0, pts.shape[0] - 1, 400).astype(int)
            pts = pts[sel]
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        best = max(best, float(d.max()))
    return round(best, 1)


def organ_context(tmask: np.ndarray, organ_seg: Optional[Path], vcc: float) -> Dict:
    """Which organ the lesion sits in (max tumor overlap) + that organ's volume."""
    if organ_seg is None or tmask.sum() == 0:
        return {}
    od = np.asarray(nib.load(str(organ_seg)).dataobj)
    if od.shape != tmask.shape:
        return {}
    overlap: Dict[str, int] = {}
    labels_in_tumor = od[tmask > 0]
    for lid in np.unique(labels_in_tumor):
        if int(lid) == 0:
            continue
        name = ORGAN_LABELS.get(int(lid))
        if name:
            overlap[name] = overlap.get(name, 0) + int((labels_in_tumor == lid).sum())
    if not overlap:
        return {"organ": "extra-organ / not localized"}
    organ = max(overlap, key=overlap.get)
    organ_ids = [k for k, v in ORGAN_LABELS.items() if v == organ]
    organ_vox = int(np.isin(od, organ_ids).sum())
    in_frac = round(overlap[organ] / max(1, int((tmask > 0).sum())), 2)
    return {"organ": organ,
            "organ_vol_cc": round(organ_vox * vcc, 1),
            "tumor_in_organ_frac": in_frac}


def measure_stage(proc_ds: Path, patient: str, stage_dir: Path) -> Dict:
    """Return a measurement dict for one timepoint (CT + lesion mask + organ context)."""
    out: Dict = {"stage": stage_dir.name}
    tpath, tsrc = tumor_mask_path(proc_ds, patient, stage_dir)
    out["tumor_source"] = tsrc
    if tpath is not None:
        im = nib.load(str(tpath))
        d = np.asarray(im.dataobj)
        tmask = (d > 0)
        vcc = _voxel_cc(im.affine)
        out["seg_present"] = True
        out["tumor_vol_cc"] = round(float(tmask.sum()) * vcc, 2)
        out["lesion_vol_cc"] = out["tumor_vol_cc"]          # PRIMARY burden backbone
        out["longest_diameter_mm"] = longest_diameter_mm(tmask, im.affine)
        out.update(organ_context(tmask, organ_seg_path(proc_ds, patient, stage_dir.name), vcc))
    else:
        out["seg_present"] = False
        out["tumor_vol_cc"] = None
        out["lesion_vol_cc"] = None
        out["longest_diameter_mm"] = None
    out["modalities"] = ["ct"] if ct_path(stage_dir) else []
    return out


def _burden(x: Optional[Dict]) -> Optional[float]:
    return None if x is None else x.get("lesion_vol_cc")


def _diam(x: Optional[Dict]) -> Optional[float]:
    return None if x is None else x.get("longest_diameter_mm")


def _pct(cur, ref) -> Optional[float]:
    if cur is None or ref in (None, 0):
        return None
    return round(100.0 * (cur - ref) / abs(ref), 1)


def with_deltas(m: Dict, baseline: Optional[Dict], nadir: Optional[Dict]) -> Dict:
    """% change vs baseline / nadir for BOTH volume (backbone) and RECIST diameter."""
    cur, curd = _burden(m), _diam(m)
    m["burden_metric"] = "tumor_vol_cc" if cur is not None else "none"
    m["delta_vs_baseline_pct"] = _pct(cur, _burden(baseline))
    m["delta_vs_nadir_pct"] = _pct(cur, _burden(nadir))
    m["diam_delta_vs_baseline_pct"] = _pct(curd, _diam(baseline))
    m["diam_delta_vs_nadir_pct"] = _pct(curd, _diam(nadir))
    m["baseline_burden"] = _burden(baseline)
    m["nadir_burden"] = _burden(nadir)
    m["baseline_diam_mm"] = _diam(baseline)
    m["nadir_diam_mm"] = _diam(nadir)
    return m


def with_prior_delta(m: Dict, prior: Optional[Dict]) -> Dict:
    """% change vs the immediately-prior timepoint (change-from-last-point)."""
    cur, curd = _burden(m), _diam(m)
    m["delta_vs_prior_pct"] = _pct(cur, _burden(prior))
    m["diam_delta_vs_prior_pct"] = _pct(curd, _diam(prior))
    m["prior_burden"] = _burden(prior)
    m["prior_diam_mm"] = _diam(prior)
    return m


def _trend(delta_pct: Optional[float], cur, prior) -> str:
    if cur is not None and cur >= 0.5 and prior in (None, 0):
        return "new"
    if cur is not None and cur < 0.5 and prior and prior >= 0.5:
        return "resolved"
    if delta_pct is None:
        return "stable"
    if delta_pct >= 20:          # RECIST progression boundary (diameter)
        return "increasing"
    if delta_pct <= -30:         # RECIST partial-response boundary (diameter)
        return "decreasing"
    return "stable"


def build_target_lesions(m: Dict, prior: Optional[Dict]) -> List[Dict]:
    """Deterministic target lesion from SEGMENTATION (never eyeballed by the VLM).

    One measurable target lesion (RECIST tracks the dominant lesion here): its
    volume + longest diameter and their change. Sizes come straight from the mask,
    so they cannot be fabricated."""
    cur = m.get("lesion_vol_cc")
    if cur is None:
        return []
    organ = m.get("organ") or "abdomen"
    site = f"{organ} tumor (RECIST target)"
    return [{"id": "L1", "site": site,
             "now": cur, "prior": (prior or {}).get("lesion_vol_cc"),
             "now_diam_mm": m.get("longest_diameter_mm"),
             "prior_diam_mm": (prior or {}).get("longest_diameter_mm"),
             "trend": _trend(m.get("diam_delta_vs_prior_pct", m.get("delta_vs_prior_pct")),
                             cur, (prior or {}).get("lesion_vol_cc")),
             "delta_vs_nadir_pct": m.get("diam_delta_vs_nadir_pct",
                                         m.get("delta_vs_nadir_pct"))}]


def factual_impression(m: Dict, prior_stage: Optional[str], label: str,
                       has_prior: bool) -> str:
    """A numbers-grounded one-liner — the trustworthy core of the impression."""
    cur = m.get("lesion_vol_cc")
    curd = m.get("longest_diameter_mm")
    organ = m.get("organ") or "abdominal"

    def size_str():
        s = f"{cur} cc" if cur is not None else "—"
        if curd is not None:
            s += f", longest diameter {curd} mm"
        return s

    if not has_prior:
        if cur is None:
            return "Baseline study; no prior for comparison."
        return (f"Baseline study; no prior for comparison. {organ.capitalize()} tumor "
                f"burden {size_str()} by segmentation.")
    pb = m.get("prior_burden")
    dp = m.get("diam_delta_vs_prior_pct")
    dpv = m.get("delta_vs_prior_pct")
    dn = m.get("diam_delta_vs_nadir_pct", m.get("delta_vs_nadir_pct"))
    ref = prior_stage or "the prior scan"
    if cur is not None:
        if (pb in (None, 0)) and cur >= 0.5:
            return (f"Compared to {ref}, new/recurrent measurable {organ} tumor "
                    f"({size_str()}; none on prior). RECIST: {label}.")
        if cur < 0.5 and pb and pb >= 0.5:
            return (f"Compared to {ref}, resolution of measurable {organ} tumor "
                    f"(was {pb} cc). RECIST: {label}.")
    primary = dp if dp is not None else dpv
    verb = ("increased" if (primary or 0) >= 20 else
            "decreased" if (primary or 0) <= -30 else "was largely stable")
    parts = [f"Compared to {ref}, measurable {organ} tumor burden {verb}"]
    if dp is not None:
        parts.append(f" {dp:+.0f}% by diameter ({size_str()})")
    elif dpv is not None:
        parts.append(f" {dpv:+.0f}% by volume ({size_str()})")
    if dn is not None:
        parts.append(f", {dn:+.0f}% vs nadir")
    parts.append(f". RECIST: {label}.")
    return "".join(parts)
