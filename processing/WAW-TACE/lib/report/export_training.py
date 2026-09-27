#!/usr/bin/env python3
"""Export per-patient ledgers into a TRAINING SET for a longitudinal CT
image-generation model that uses the progression report as a CONSTRAINT on how
the lesion progresses.

A training example is a TRANSITION (prior stage -> current stage):

    input    : prior + baseline volumes        (what the generator conditions on)
    target   : current volumes                 (what the generator must produce)
    condition: { caption, vector, fields }     <-- the report, made model-readable
    qc       : { burden_metric, qc_passed, confidence }   <-- for filtering

`condition` is the whole point: a fixed-length VECTOR (for cross-attention / FiLM /
concat conditioning) plus a free-text CAPTION (for a text-encoder branch), both
describing the change to render — Δ-volume / Δ-diameter vs prior and vs baseline,
RECIST label, new lesions, treatment-effect risk, where/how the lesion grew.

Volume paths are emitted as RELATIVE paths the training loader joins with its own
CTProcessed root:
    ct     -> <DS>/<pid>/<stage>/ct.nii.gz
    tumor  -> <DS>/_segvol/<pid>/<stage>_tumor.nii.gz
    organs -> <DS>/_totalseg/<pid>/<stage>_seg.nii.gz

Usage:
    python3 export_training.py                      # scan reports root, write ./_train
    python3 export_training.py --include-baseline   # also emit the t0 (no-change) row
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
DEFAULT_REPORTS = os.environ.get("REPORTS", "/media/cbtil3/WhiteSD/CTProcessed-Reports")

# ---- fixed-length condition vector schema ---------------------------------
LABELS = ["CR", "PR", "SD", "PD"]              # RECIST categories (one-hot)
PHASES = ["baseline", "follow-up"]            # stage type (one-hot)
RISK = {"low": 0.0, "moderate": 0.5, "medium": 0.5, "high": 1.0}

VECTOR_FIELDS = (
    [f"phase={p}" for p in PHASES] +
    [f"label={l}" for l in LABELS] +
    ["confirmed",                  # PD/PR/CR confirmed (1) / unconfirmed (0)
     "new_lesions",                # any new lesion (1/0)
     "treatment_effect_risk",      # 0 low .. 1 high
     "burden_is_seg",              # 1 = volume from a lesion mask, 0 = none
     "burden_now_cc",              # raw current tumor volume (cc)
     "diam_now_mm",                # raw current longest diameter (mm)
     "delta_vol_vs_prior_pct",     # % volume change vs immediately-prior scan
     "delta_vol_vs_baseline_pct",  # % volume change vs baseline
     "delta_vol_vs_nadir_pct",     # % volume change vs nadir
     "delta_diam_vs_nadir_pct",    # % diameter change vs nadir (RECIST reference)
     "weeks_since_treatment"]      # -1.0 if unknown
)


def _onehot(value, options) -> List[float]:
    return [1.0 if value == o else 0.0 for o in options]


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def encode_vector(fields: Dict) -> List[float]:
    v: List[float] = []
    v += _onehot(fields.get("phase"), PHASES)
    v += _onehot(fields.get("label"), LABELS)
    v.append(1.0 if fields.get("confirmed") else 0.0)
    v.append(1.0 if fields.get("new_lesions") else 0.0)
    v.append(RISK.get(str(fields.get("treatment_effect_risk", "")).lower(), 0.0))
    v.append(1.0 if fields.get("burden_is_seg") else 0.0)
    v.append(_f(fields.get("burden_now_cc")))
    v.append(_f(fields.get("diam_now_mm")))
    v.append(_f(fields.get("delta_vol_vs_prior_pct")))
    v.append(_f(fields.get("delta_vol_vs_baseline_pct")))
    v.append(_f(fields.get("delta_vol_vs_nadir_pct")))
    v.append(_f(fields.get("delta_diam_vs_nadir_pct")))
    v.append(_f(fields.get("weeks_since_treatment"), -1.0))
    assert len(v) == len(VECTOR_FIELDS), (len(v), len(VECTOR_FIELDS))
    return v


# ---- triangulated-trust + 2D image-read (S2) extension --------------------
AGREE_SCALAR = {"agree": 1.0, "partial": 0.5, "s1-only": 0.0, "s2-only": 0.0,
                "none": 0.0, "conflict": -1.0}
REL_ORD = {"segmentation": 1.0, "none": 0.0}
CHANGE_SIGN = {"new": 1.0, "increased": 1.0, "stable": 0.0,
               "decreased": -1.0, "resolved": -1.0, "absent": 0.0}
TRUST_FIELDS = ["trust", "s1s2_agreement", "burden_reliability_ord"]


def load_label_ids(path: Optional[str]) -> List[str]:
    p = Path(path) if path else (HERE / "labels_abdomen.json")
    try:
        return [l["id"] for l in json.loads(p.read_text()).get("labels", [])]
    except Exception:
        return []


def encode_trust(report: Dict) -> List[float]:
    tri = report.get("triangulation", {}) or {}
    return [_f(tri.get("trust")),
            AGREE_SCALAR.get(tri.get("agreement"), 0.0),
            REL_ORD.get(tri.get("burden_reliability"), 0.0)]


def encode_image(report: Dict, label_ids: List[str]) -> List[float]:
    labs = ((report.get("image_findings") or {}).get("labels")) or {}
    out = []
    for lid in label_ids:
        e = labs.get(lid, {})
        present = 1.0 if e.get("present") else 0.0
        sign = CHANGE_SIGN.get(str(e.get("change", "absent")), 0.0)
        conf = _f(e.get("confidence"))
        out.append(round(present * sign * conf, 3))
    return out


def full_vector(fields: Dict, report: Dict, label_ids: List[str]) -> List[float]:
    return encode_vector(fields) + encode_trust(report) + encode_image(report, label_ids)


def full_field_names(label_ids: List[str]) -> List[str]:
    return VECTOR_FIELDS + TRUST_FIELDS + [f"img:{i}" for i in label_ids]


# ---- caption (text-encoder branch) ----------------------------------------
FINDING_ORDER = ["target_tumor", "new_lesions", "nodes_metastasis",
                 "invasion_vascular", "necrosis_hemorrhage"]


def build_caption(report: Dict, fields: Dict) -> str:
    f = report.get("findings", {}) or {}
    parts = [str(f[k]).strip() for k in FINDING_ORDER
             if f.get(k) and str(f[k]).strip().lower() not in ("none", "none.", "stub")]
    body = " ".join(parts) if parts else (report.get("impression") or "").strip()
    dp = fields.get("delta_vol_vs_prior_pct")
    db = fields.get("delta_vol_vs_baseline_pct")
    quant = []
    if dp is not None:
        quant.append(f"{dp:+.0f}% vol vs prior")
    if db is not None:
        quant.append(f"{db:+.0f}% vol vs baseline")
    tag = f" Burden {', '.join(quant)}." if quant else ""
    lab = fields.get("label")
    return f"{body}{tag} RECIST: {lab}." if lab else f"{body}{tag}"


# ---- per-patient processing -----------------------------------------------
def _burden(meas: Dict) -> Optional[float]:
    return meas.get("lesion_vol_cc")


def _relpaths(dataset: str, patient: str, stage: str, meas: Dict) -> Dict[str, str]:
    out = {"ct": f"{dataset}/{patient}/{stage}/ct.nii.gz"}
    if meas.get("seg_present"):
        if meas.get("tumor_source") == "gt":
            out["tumor"] = f"{dataset}/{patient}/{stage}/tumor.nii.gz"
        else:
            out["tumor"] = f"{dataset}/_segvol/{patient}/{stage}_tumor.nii.gz"
    out["organs"] = f"{dataset}/_totalseg/{patient}/{stage}_seg.nii.gz"
    return out


def _split_of(patient: str, val=0.1, test=0.1) -> str:
    h = int(hashlib.md5(patient.encode()).hexdigest(), 16) % 1000 / 1000.0
    if h < test:
        return "test"
    if h < test + val:
        return "val"
    return "train"


def export_patient(ledger_path: Path, include_baseline: bool,
                   label_ids: List[str]) -> List[Dict]:
    L = json.loads(ledger_path.read_text())
    dataset, patient = L.get("dataset"), L.get("patient")
    visits = sorted(L.get("visits", []), key=lambda v: v.get("stage") or "")
    if not visits:
        return []
    baseline = visits[0]
    nadir_stage = (L.get("nadir") or {}).get("stage")
    rows: List[Dict] = []
    for i, v in enumerate(visits):
        meas = v.get("measurements", {}) or {}
        report = v.get("report", {}) or {}
        is_baseline = (i == 0)
        if is_baseline and not include_baseline:
            continue
        prior = visits[i - 1] if i > 0 else None
        conf = report.get("confounders", {}) or {}
        fields = {
            "phase": v.get("phase") or report.get("phase"),
            "label": (v.get("progression_label") or report.get("progression_label")),
            "confirmed": (v.get("confirmation_status") == "confirmed"),
            "new_lesions": bool(str((report.get("findings", {}) or {})
                                    .get("new_lesions", "none")).strip().lower()
                                not in ("none", "none.", "", "stub")),
            "treatment_effect_risk": conf.get("treatment_effect_risk", "low"),
            "burden_is_seg": (meas.get("burden_metric") == "tumor_vol_cc"
                              or meas.get("seg_present") is True),
            "burden_now_cc": meas.get("tumor_vol_cc"),
            "diam_now_mm": meas.get("longest_diameter_mm"),
            "delta_vol_vs_prior_pct": meas.get("delta_vs_prior_pct"),
            "delta_vol_vs_baseline_pct": meas.get("delta_vs_baseline_pct"),
            "delta_vol_vs_nadir_pct": meas.get("delta_vs_nadir_pct"),
            "delta_diam_vs_nadir_pct": meas.get("diam_delta_vs_nadir_pct"),
            "weeks_since_treatment": conf.get("weeks_since_treatment", -1),
        }
        rows.append({
            "id": f"{dataset}__{patient}__{v.get('stage')}",
            "dataset": dataset, "patient": patient,
            "transition": {"from": prior.get("stage") if prior else None,
                           "to": v.get("stage"), "baseline": baseline.get("stage"),
                           "nadir": nadir_stage},
            "input": {
                "prior": _relpaths(dataset, patient, prior.get("stage"),
                                   prior.get("measurements", {})) if prior else None,
                "baseline": _relpaths(dataset, patient, baseline.get("stage"),
                                      baseline.get("measurements", {})),
            },
            "target": _relpaths(dataset, patient, v.get("stage"), meas),
            "condition": {
                "caption": build_caption(report, fields),
                "vector": full_vector(fields, report, label_ids),
                "fields": fields,
                "image_findings": (report.get("image_findings") or {}).get("labels"),
                "triangulation": report.get("triangulation"),
            },
            "qc": {"burden_metric": meas.get("burden_metric"),
                   "qc_passed": report.get("_qc_passed"),
                   "confidence": report.get("confidence"),
                   "trust": (report.get("triangulation") or {}).get("trust")},
            "split": _split_of(patient),
        })
    return rows


# ---- per-timepoint CSV summary (one row per visit, all patients) ----------
CSV_COLS = ["dataset", "patient", "stage", "stage_type", "phase", "visit_idx",
            "n_visits", "organ", "days_from_baseline", "days_since_prior",
            "progression_label", "confirmation_status", "confidence",
            "burden_metric", "tumor_vol_cc", "longest_diameter_mm", "tumor_source",
            "delta_vol_vs_prior_pct", "delta_vol_vs_baseline_pct", "delta_vol_vs_nadir_pct",
            "diam_delta_vs_nadir_pct", "trust", "agreement", "burden_reliability",
            "seg_qc_reliable", "s1_dir", "s2_dir", "treatment_effect_risk",
            "weeks_since_treatment", "new_lesions", "qc_passed", "recist_basis", "impression"]


def summary_rows(ledger_path: Path) -> List[Dict]:
    L = json.loads(ledger_path.read_text())
    dataset, patient = L.get("dataset"), L.get("patient")
    visits = sorted(L.get("visits", []), key=lambda v: v.get("stage") or "")
    rows = []
    for i, v in enumerate(visits):
        meas = v.get("measurements", {}) or {}
        rep = v.get("report", {}) or {}
        tri = rep.get("triangulation", {}) or {}
        conf = rep.get("confounders", {}) or {}
        fnd = rep.get("findings", {}) or {}
        new_les = str(fnd.get("new_lesions", "")).strip().lower() \
            not in ("none", "none.", "", "stub")
        imp = (rep.get("impression") or "").replace("\n", " ").strip()
        rows.append({
            "dataset": dataset, "patient": patient, "stage": v.get("stage"),
            "stage_type": meas.get("stage_type"),
            "phase": v.get("phase"), "visit_idx": i + 1, "n_visits": len(visits),
            "organ": meas.get("organ"),
            "days_from_baseline": meas.get("days_from_baseline"),
            "days_since_prior": meas.get("days_since_prior"),
            "tumor_vol_cc": meas.get("tumor_vol_cc"),
            "longest_diameter_mm": meas.get("longest_diameter_mm"),
            "tumor_source": meas.get("tumor_source"),
            "progression_label": v.get("progression_label"),
            "confirmation_status": v.get("confirmation_status"),
            "confidence": rep.get("confidence"),
            "burden_metric": meas.get("burden_metric"),
            "delta_vol_vs_prior_pct": meas.get("delta_vs_prior_pct"),
            "delta_vol_vs_baseline_pct": meas.get("delta_vs_baseline_pct"),
            "delta_vol_vs_nadir_pct": meas.get("delta_vs_nadir_pct"),
            "diam_delta_vs_nadir_pct": meas.get("diam_delta_vs_nadir_pct"),
            "trust": tri.get("trust"), "agreement": tri.get("agreement"),
            "burden_reliability": tri.get("burden_reliability"),
            "seg_qc_reliable": (meas.get("measurement_qc") or {}).get("seg_reliable"),
            "s1_dir": tri.get("s1_direction"), "s2_dir": tri.get("s2_direction"),
            "treatment_effect_risk": conf.get("treatment_effect_risk"),
            "weeks_since_treatment": conf.get("weeks_since_treatment"),
            "new_lesions": int(new_les),
            "qc_passed": rep.get("_qc_passed"),
            "recist_basis": rep.get("recist_basis"),
            "impression": imp,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=DEFAULT_REPORTS,
                    help="root holding <DATASET>/<patient>/ledger.json")
    ap.add_argument("--out", default=str(HERE.parent.parent / "report" / "_train"))
    ap.add_argument("--include-baseline", action="store_true",
                    help="also emit the t0 row (no prior; zero-change condition)")
    ap.add_argument("--require-qc", action="store_true",
                    help="drop rows whose report did not pass the QC gate")
    ap.add_argument("--min-trust", type=float, default=0.0,
                    help="drop rows whose triangulated trust is below this")
    ap.add_argument("--labels", default="",
                    help="finding taxonomy used by S2 (default: labels_abdomen.json)")
    args = ap.parse_args()

    label_ids = load_label_ids(args.labels or None)
    reports = Path(args.reports)
    ledgers = sorted(reports.glob("*/*/ledger.json"))
    if not ledgers:
        raise SystemExit(f"no ledgers under {reports} (run run_report.py first)")

    rows: List[Dict] = []
    for lp in ledgers:
        try:
            rows += export_patient(lp, args.include_baseline, label_ids)
        except Exception as e:
            print(f"  [warn] {lp}: {e}")
    if args.require_qc:
        rows = [r for r in rows if r["qc"]["qc_passed"] is not False]
    if args.min_trust > 0:
        rows = [r for r in rows if (r["qc"].get("trust") or 0.0) >= args.min_trust]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    field_names = full_field_names(label_ids)
    (out / "condition_schema.json").write_text(json.dumps({
        "vector_fields": field_names,
        "vector_dim": len(field_names),
        "blocks": {"quant": len(VECTOR_FIELDS), "trust": len(TRUST_FIELDS),
                   "image_2d": len(label_ids)},
        "labels": LABELS, "phases": PHASES, "image_labels": label_ids,
        "notes": "vector = [quant | trust | image_2d]. quant=S1 numbers (RECIST), "
                 "trust=deterministic triangulation, image_2d=per-label signed "
                 "confidence from the S2 image read (present*sign*conf, +growth/-shrink). "
                 "raw scalars (burden_now_cc, diam_now_mm, delta_*_pct, weeks_since_treatment) "
                 "are UNnormalized — fit a scaler on the train split. -1 means unknown.",
    }, indent=2))

    srows: List[Dict] = []
    for lp in ledgers:
        try:
            srows += summary_rows(lp)
        except Exception as e:
            print(f"  [warn] csv {lp}: {e}")
    with (out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(srows)
    print(f"wrote {len(srows)} timepoint rows -> {out/'summary.csv'}")

    by_split = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    by_label: Dict[str, int] = {}
    for r in rows:
        k = r["condition"]["fields"].get("label") or "?"
        by_label[k] = by_label.get(k, 0) + 1
    print(f"wrote {len(rows)} transitions -> {out/'dataset.jsonl'}")
    print(f"  split: {by_split}")
    print(f"  label: {by_label}")
    print(f"  vector_dim: {len(field_names)} "
          f"[quant {len(VECTOR_FIELDS)} | trust {len(TRUST_FIELDS)} | "
          f"image_2d {len(label_ids)}] -> {out/'condition_schema.json'}")


if __name__ == "__main__":
    main()
