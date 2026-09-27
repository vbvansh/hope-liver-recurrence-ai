"""Deterministic triangulation between S1 (3D numbers) and S2 (2D image read).

This is the AUDITABLE core of "trustworthiness": it does not depend on the
generative model. It decides how reliable the burden is, whether the two
independent sources agree on the direction of change, and a single trust score
the fusion report and the training condition both carry.

Trust = data-reliability x cross-source agreement.
  - lesion mask present -> burden is a real volume + RECIST diameter (high reliability)
  - mask absent         -> no quantitative burden (low); lean on the image read
  - S1 sign and S2 visual direction agree -> raise trust; conflict -> drop it

Guideline is RECIST 1.1 (solid abdominal/thoracic tumors), not RANO.
"""
from __future__ import annotations
import math
from typing import Dict, Optional

# RECIST 1.1 thresholds, applied to the longest-diameter (mm) measure.
PD_DIAM_PCT = 20.0     # >=20% AND >=5mm increase vs nadir => progression
PD_DIAM_ABS = 5.0      # mm
PR_DIAM_PCT = -30.0    # >=30% decrease vs baseline => partial response
DIR_THRESH = 10.0      # |%Δ| below this is "flat" for direction agreement


def verify_findings_3d(meas: Dict, prior: Optional[Dict] = None) -> Dict:
    """Deterministic QC of the 3D lesion-mask numbers (S1) — the analog of the
    removal-only S2 image-read verify. The trusted core is only trustworthy if the
    segmentation itself is sane; this catches mask artifacts that would corrupt the
    RECIST call: NaN/inf, tumor > containing organ, implausibly large lesion, and
    'huge %% change from a sub-cc base' (noise, not biology). SegVol masks are
    unvalidated pre-annotations, so they are demoted relative to ground-truth masks.

    Returns {checked, seg_reliable, qc_passed, warnings}. When seg_reliable is
    False the caller demotes the seg-based label to ADVISORY and lowers trust.
    """
    if not meas.get("seg_present"):
        return {"checked": False, "seg_reliable": False, "qc_passed": True,
                "warnings": ["no lesion mask (image-read only)"]}
    warns: list = []
    reliable = True
    vol = meas.get("tumor_vol_cc")
    organ_vol = meas.get("organ_vol_cc")
    for k in ("tumor_vol_cc", "longest_diameter_mm", "organ_vol_cc"):
        v = meas.get(k)
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            warns.append(f"{k} is NaN/inf"); reliable = False
    if vol is not None and organ_vol not in (None, 0) and vol > organ_vol:
        warns.append(f"tumor {vol}cc > containing organ {organ_vol}cc — mask artifact")
        reliable = False
    if vol is not None and vol > 4000:
        warns.append(f"tumor {vol}cc implausibly large — likely over-segmentation")
        reliable = False
    if meas.get("tumor_source") == "segvol":
        warns.append("SegVol pre-annotation (unvalidated mask)")
    # huge %% from a sub-cc base = noise, not progression
    dp = meas.get("delta_vs_prior_pct")
    pb = meas.get("prior_burden")
    cur = meas.get("lesion_vol_cc")
    if dp is not None and abs(dp) >= 500 and (pb is None or pb < 1.0) and (cur or 0) < 2.0:
        warns.append(f"{dp}% change from sub-cc base — likely noise, low confidence")
    return {"checked": True, "seg_reliable": reliable, "qc_passed": reliable,
            "warnings": warns}


def _s1_direction(meas: Dict) -> str:
    """Sign of change from the most reliable available delta (prior > nadir > baseline),
    preferring the RECIST diameter delta over the volume delta."""
    for key in ("diam_delta_vs_prior_pct", "diam_delta_vs_nadir_pct",
                "diam_delta_vs_baseline_pct", "delta_vs_prior_pct",
                "delta_vs_nadir_pct", "delta_vs_baseline_pct"):
        d = meas.get(key)
        if d is None:
            continue
        if d >= DIR_THRESH:
            return "up"
        if d <= -DIR_THRESH:
            return "down"
        return "flat"
    return "unknown"


def _reliability(meas: Dict) -> str:
    return "segmentation" if (meas.get("seg_present")
                              or meas.get("lesion_vol_cc") is not None) else "none"


def _s2_new_lesion(findings2d: Optional[Dict]) -> bool:
    labs = (findings2d or {}).get("labels") or {}
    for lid in ("new_lesion", "distant_metastasis"):
        e = labs.get(lid, {})
        if int(e.get("present", 0) or 0) and str(e.get("change")) in ("new", "increased") \
                and float(e.get("confidence", 0) or 0) >= 0.5:
            return True
    return False


def recist_hint(meas: Dict, findings2d: Optional[Dict], has_prior: bool) -> Dict:
    """Deterministic RECIST-1.1 label SUGGESTION from the trusted numbers (S1) +
    the new-lesion flag (S2). The fusion model gets this as strong guidance so a
    clear quantitative call is not overturned by a noisy image read. `certain` is
    True only when burden is a real lesion volume that passed 3D-QC."""
    if not has_prior:
        return {"label": "baseline", "basis": "first timepoint, no prior", "certain": True}
    seg = meas.get("seg_present") or meas.get("lesion_vol_cc") is not None
    sure = seg and not meas.get("seg_unreliable")
    cur = meas.get("lesion_vol_cc")
    curd = meas.get("longest_diameter_mm")
    nadir_b = meas.get("nadir_burden")
    nadir_d = meas.get("nadir_diam_mm")
    dn_d = meas.get("diam_delta_vs_nadir_pct")
    db_d = meas.get("diam_delta_vs_baseline_pct")
    new_les = _s2_new_lesion(findings2d)
    MEASURABLE = 0.5   # cc; below this = no measurable disease
    if not seg:
        if new_les:
            return {"label": "PD", "basis": "new lesion (S2); no mask", "certain": False}
        return {"label": "SD?", "basis": "no lesion mask; image-read only", "certain": False}
    # disappearance of all measurable disease
    if cur is not None and cur < MEASURABLE:
        return {"label": "CR", "basis": "no measurable disease", "certain": sure}
    # reappearance of measurable disease from a zero nadir (recurrence) => PD
    if cur is not None and cur >= MEASURABLE and nadir_b == 0:
        return {"label": "PD", "basis": f"new measurable disease ({cur} cc) from zero nadir",
                "certain": sure}
    if new_les:
        return {"label": "PD", "basis": "new measurable lesion (S2)", "certain": sure}
    # RECIST progression: >=20% AND >=5mm diameter increase vs nadir
    if dn_d is not None and nadir_d is not None and curd is not None:
        if dn_d >= PD_DIAM_PCT and (curd - nadir_d) >= PD_DIAM_ABS:
            return {"label": "PD", "basis": f"+{dn_d}% diameter vs nadir "
                    f"({nadir_d}->{curd} mm)", "certain": sure}
    # RECIST partial response: >=30% diameter decrease vs baseline
    if db_d is not None and db_d <= PR_DIAM_PCT:
        return {"label": "PR", "basis": f"{db_d}% diameter vs baseline", "certain": sure}
    # volume fallback when diameters are unavailable
    if dn_d is None:
        dn_v = meas.get("delta_vs_nadir_pct")
        db_v = meas.get("delta_vs_baseline_pct")
        if dn_v is not None and dn_v >= 73:        # ~+20% diameter in volume terms
            return {"label": "PD", "basis": f"+{dn_v}% volume vs nadir", "certain": sure}
        if db_v is not None and db_v <= -66:       # ~-30% diameter in volume terms
            return {"label": "PR", "basis": f"{db_v}% volume vs baseline", "certain": sure}
    return {"label": "SD", "basis": "size change within RECIST thresholds", "certain": sure}


def triangulate(meas: Dict, findings2d: Optional[Dict], has_prior: bool = True) -> Dict:
    """Return the triangulation block + a scalar trust in [0,1]."""
    reliability = _reliability(meas)
    s1 = _s1_direction(meas)
    s2 = (findings2d or {}).get("image_burden_direction") or "stable"
    s2_conf = float((findings2d or {}).get("image_confidence") or 0.0)

    s2_dir = {"increased": "up", "decreased": "down", "stable": "flat"}.get(s2, "flat")
    has_s1 = s1 != "unknown"
    has_s2 = findings2d is not None and bool(findings2d)

    if not has_prior:
        agreement = "baseline"
    elif has_s1 and has_s2:
        agreement = "agree" if s1 == s2_dir else (
            "partial" if "flat" in (s1, s2_dir) else "conflict")
    elif has_s1:
        agreement = "s1-only"
    elif has_s2:
        agreement = "s2-only"
    else:
        agreement = "none"

    base = {"segmentation": 0.8, "none": 0.2}[reliability]
    if agreement == "baseline":
        trust = base
    elif agreement == "agree":
        trust = min(1.0, base + 0.2 * s2_conf + 0.05)
    elif agreement == "conflict":
        trust = max(0.05, base - 0.3)
    elif agreement == "s2-only":
        trust = min(0.7, 0.3 + 0.4 * s2_conf)
    else:
        trust = base
    seg_unreliable = bool(meas.get("seg_unreliable"))
    if seg_unreliable:
        trust = max(0.05, trust - 0.3)
    return {
        "s1_direction": s1,
        "s2_direction": s2,
        "agreement": agreement,
        "burden_reliability": ("segmentation*untrusted" if seg_unreliable
                               and reliability == "segmentation" else reliability),
        "s2_confidence": round(s2_conf, 2),
        "seg_qc_reliable": not seg_unreliable,
        "trust": round(float(trust), 3),
    }
