"""Optional model-ready per-patient CLINICAL JSON joined into the report context.

The abdominal CT cohorts do not (yet) ship a `clinical_data/model_ready/<DS>/
<patient>.json` like the brain pipeline does, so everything here degrades
gracefully: missing file => fields are None and the report runs on imaging alone.
If you later build such JSONs (demographics, diagnosis, treatment dates, a
`timepoints` list with per-scan `date`/`days_from_baseline`), this module surfaces
them automatically — same interface as the brain pipeline.

Stage type (baseline / follow-up N) is always derivable from the stage dir name,
so timing.stage_type is populated even without a clinical record.
"""
from __future__ import annotations
import json
import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
DEFAULT_CLIN = os.environ.get("CLIN_ROOT", str(HERE.parent.parent / "clinical_data" / "model_ready"))


def load_clinical(dataset: str, patient: str, root: str = DEFAULT_CLIN) -> Optional[Dict]:
    f = Path(root) / dataset / f"{patient}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return None
    return None


def _parse(d) -> Optional[date]:
    try:
        return date.fromisoformat(str(d)[:10])
    except Exception:
        return None


def _treatment(clin: Optional[Dict]) -> Dict:
    if not clin:
        return {}
    t = clin.get("treatment")
    return t if isinstance(t, dict) else {}


def patient_summary(clin: Optional[Dict]) -> str:
    if not clin:
        return "(no clinical record found)"
    demo = clin.get("demographics") or {}
    dx = clin.get("diagnosis") or {}
    out = clin.get("outcomes") or {}
    tx = _treatment(clin)
    bits = []
    if demo.get("age") is not None or demo.get("sex"):
        bits.append(f"{demo.get('age', '?')}y {demo.get('sex', '?')}")
    if dx.get("histology"):
        bits.append(str(dx["histology"]) + (f" {dx['stage']}" if dx.get("stage") else ""))
    if dx.get("primary_site"):
        bits.append(f"primary {dx['primary_site']}")
    if tx.get("text"):
        bits.append(str(tx["text"]))
    if out.get("pfs_days") is not None:
        bits.append(f"PFS {out['pfs_days']}d")
    if out.get("os_days") is not None:
        bits.append(f"OS {out['os_days']}d")
    return "; ".join(str(b) for b in bits) if bits else "(clinical record present, sparse)"


def stage_record(clin: Optional[Dict], stage: str) -> Dict:
    if not clin:
        return {}
    for tp in clin.get("timepoints", []):
        if tp.get("stage") == stage:
            return tp
    return {}


def _stage_type(stage: str) -> str:
    s = stage.lower()
    if "baseline" in s or s.endswith("_0") or s.startswith("000"):
        return "baseline"
    if "followup" in s or "follow_up" in s:
        return "follow-up"
    return "follow-up"


def timing(clin: Optional[Dict], stage: str, prior_stage: Optional[str]) -> Dict:
    """Per-timepoint timing relative to baseline / prior + weeks since treatment.

    stage_type is always set (from the dir name). Dates/days/weeks are only filled
    when a clinical record with a matching `timepoints` entry exists.
    """
    rec = stage_record(clin, stage)
    dfb = rec.get("days_from_baseline")
    out = {"stage_type": rec.get("stage_type") or _stage_type(stage),
           "scan_date": rec.get("date"),
           "days_from_baseline": dfb,
           "days_since_prior": None,
           "weeks_since_treatment": None}
    if prior_stage:
        pdfb = stage_record(clin, prior_stage).get("days_from_baseline")
        if dfb is not None and pdfb is not None:
            out["days_since_prior"] = dfb - pdfb
    tx_end = _parse(_treatment(clin).get("end"))
    sd = _parse(rec.get("date"))
    if tx_end and sd:
        out["weeks_since_treatment"] = round((sd - tx_end).days / 7.0, 1)
    return out


def treatment_timeline(clin: Optional[Dict]) -> List[Dict]:
    """Treatment events (surgery / chemo / TACE / RT) when a clinical record exists."""
    if not clin:
        return []
    ev = []
    tx = _treatment(clin)
    for key, label in (("surgery_date", "surgery"), ("start", "treatment start"),
                       ("end", "treatment end")):
        if tx.get(key):
            ev.append({"date": tx[key], "event": label, "detail": tx.get("text") or ""})
    for e in (clin.get("events") or []):
        if isinstance(e, dict) and e.get("date"):
            ev.append({"date": e["date"], "event": e.get("event", "event"),
                       "detail": e.get("detail", "")})
    return ev
