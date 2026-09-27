"""S2 — the 2D image-read pass (MR-RATE-style structured findings FROM THE IMAGE).

A VLM looks at the co-registered prior/current/difference CT slices WITHOUT any
numbers and reports, per finding label: present, change-vs-prior, confidence, and
the slice it saw it on. This is the independent "radiology factors from the model"
source the fusion pass triangulates against the 3D numbers.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

from prompts import (SYSTEM_FINDINGS2D, build_findings2d_prompt,
                     SYSTEM_FINDINGS2D_VERIFY, build_findings2d_verify_prompt)
from vlm import parse_report

HERE = Path(__file__).resolve().parent
_DEFAULT_LABELS = HERE / "labels_abdomen.json"


def load_label_set(path: Path = _DEFAULT_LABELS) -> Dict:
    return json.loads(Path(path).read_text())


def run_findings_2d(client, images: List, label_set: Dict,
                    has_prior: bool = True) -> Dict:
    """Call the VLM on images only; return normalized findings dict (+ raw think)."""
    labels = label_set.get("labels", [])
    image_labels = [l for l, _ in images]
    user = build_findings2d_prompt(labels, image_labels, has_prior)
    raw = client.generate(SYSTEM_FINDINGS2D, user, images)
    out, think = parse_report(raw)
    if out.get("_parse_error"):
        out = {"_parse_error": out.get("_parse_error"), "labels": {}, "findings": {}}
    out.setdefault("labels", {})
    out.setdefault("image_burden_direction", "stable")
    out.setdefault("image_confidence", 0.0)
    if not has_prior:
        out["image_burden_direction"] = "stable"   # no prior => no change to call
    norm: Dict[str, Dict] = {}
    for l in labels:
        e = out["labels"].get(l["id"], {})
        change = e.get("change", "absent")
        if not has_prior:
            change = "na" if int(e.get("present", 0) or 0) else "absent"
        norm[l["id"]] = {
            "present": int(e.get("present", 0) or 0),
            "change": change,
            "confidence": float(e.get("confidence", 0.0) or 0.0),
            "evidence": e.get("evidence", ""),
        }
    out["labels"] = norm
    out["_think"] = think
    return out


def verify_findings_2d(client, images: List, findings: Dict) -> Dict:
    """MR-RATE step-3: a SEPARATE removal-only pass that drops false-positive
    PRESENT findings (the weak 2D VLM over-calls). Can only set present 1->0 and
    lower confidence; never adds a finding. Returns the pruned findings dict.
    """
    labs = findings.get("labels", {})
    present = [{"id": lid, "change": e.get("change"), "confidence": e.get("confidence"),
                "evidence": e.get("evidence")}
               for lid, e in labs.items() if int(e.get("present", 0) or 0)]
    if not present:
        return findings
    image_labels = [l for l, _ in images]
    raw = client.generate(SYSTEM_FINDINGS2D_VERIFY,
                          build_findings2d_verify_prompt(present, image_labels), images)
    out, _ = parse_report(raw)
    ver = out.get("verified", {}) if isinstance(out, dict) else {}
    dropped = []
    for lid, v in ver.items():
        if lid in labs and not int(v.get("keep", 1) or 0):
            labs[lid]["present"] = 0
            labs[lid]["change"] = "absent"
            labs[lid]["confidence"] = 0.0
            dropped.append(lid)
        elif lid in labs and v.get("confidence") is not None:
            try:
                labs[lid]["confidence"] = min(float(labs[lid].get("confidence", 0) or 0),
                                              float(v["confidence"]))
            except (TypeError, ValueError):
                pass
    findings["_verify_dropped"] = dropped
    if dropped and findings.get("image_burden_direction") == "increased" \
            and not any(int(labs[l].get("present", 0) or 0)
                        and labs[l].get("change") in ("new", "increased")
                        for l in labs):
        findings["image_burden_direction"] = "stable"
    return findings
