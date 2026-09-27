"""Patient-wide longitudinal MEMORY (the 'memory module').

A per-patient JSON that accumulates one entry per timepoint. It is the external,
training-free memory the VLM reads and writes each visit. Retrieval is a
deterministic, RECIST-aligned policy (baseline + nadir + immediately-prior +
rolling summary) — not learned, no vector store needed for a single patient's
handful of visits.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional


class Ledger:
    def __init__(self, path: Path, dataset: str, patient: str):
        self.path = Path(path)
        self.data = {"dataset": dataset, "patient": patient,
                     "baseline_stage": None, "nadir": None,
                     "treatment_timeline": [], "visits": []}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            self.data.setdefault("treatment_timeline", [])

    def set_treatment_timeline(self, events: list):
        if events:
            self.data["treatment_timeline"] = events

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))

    # ---- write-back --------------------------------------------------------
    def add_visit(self, report: Dict, measurements: Dict):
        """Append a produced report + its measurements; update nadir."""
        entry = {"stage": measurements.get("stage"),
                 "phase": report.get("phase"),
                 "progression_label": report.get("progression_label"),
                 "confirmation_status": report.get("confirmation_status"),
                 "burden_metric": measurements.get("burden_metric"),
                 "burden": measurements.get("lesion_vol_cc"),
                 "diameter_mm": measurements.get("longest_diameter_mm"),
                 "impression": report.get("impression"),
                 "watch_items": report.get("carry_forward", {}).get("watch_items", []),
                 "report": report, "measurements": measurements}
        # de-dup: replace same-stage entry if re-run
        self.data["visits"] = [v for v in self.data["visits"]
                               if v.get("stage") != entry["stage"]]
        self.data["visits"].append(entry)
        self.data["visits"].sort(key=lambda v: v.get("stage") or "")
        if self.data["baseline_stage"] is None and self.data["visits"]:
            self.data["baseline_stage"] = self.data["visits"][0]["stage"]
        self._recompute_nadir()

    def _recompute_nadir(self):
        cand = [v for v in self.data["visits"]
                if isinstance(v.get("burden"), (int, float))]
        if cand:
            n = min(cand, key=lambda v: v["burden"])
            self.data["nadir"] = {"stage": n["stage"], "value": n["burden"]}

    # ---- retrieval (deterministic, RECIST-aligned) -------------------------
    def visit(self, stage: str) -> Optional[Dict]:
        return next((v for v in self.data["visits"] if v.get("stage") == stage), None)

    def priors_before(self, stage: str) -> List[Dict]:
        return [v for v in self.data["visits"] if (v.get("stage") or "") < stage]

    def retrieve(self, current_stage: str) -> Dict:
        priors = self.priors_before(current_stage)
        baseline = self.data["visits"][0] if self.data["visits"] else None
        nadir_stage = (self.data.get("nadir") or {}).get("stage")
        nadir = self.visit(nadir_stage) if nadir_stage else None
        prior = priors[-1] if priors else None
        return {"baseline": baseline, "nadir": nadir, "prior": prior,
                "summary": self.summary_table(priors),
                "trajectory": self.trajectory_summary(priors),
                "prior_findings": self.findings_text(prior),
                "treatment_timeline": self.data.get("treatment_timeline", [])}

    def trajectory_summary(self, visits: List[Dict]) -> str:
        if not visits:
            return "(first timepoint — no trajectory yet)"
        line = " → ".join(f"{v.get('stage')}({v.get('burden')}cc,{v.get('progression_label')})"
                          for v in visits)
        nad = self.data.get("nadir")
        tail = f"\nnadir = {nad}" if nad else ""
        last_pd = next((v.get("stage") for v in reversed(visits)
                        if v.get("progression_label") == "PD"), None)
        if last_pd:
            tail += f"\nprior PD at {last_pd} (confirmation logic applies)"
        return line + tail

    @staticmethod
    def findings_text(visit: Optional[Dict]) -> str:
        if not visit:
            return "(none)"
        r = visit.get("report", {})
        f = r.get("findings")
        if isinstance(f, dict):
            return json.dumps(f, ensure_ascii=False)
        return Ledger.report_text(visit)

    @staticmethod
    def summary_table(visits: List[Dict]) -> str:
        if not visits:
            return "(no prior timepoints)"
        rows = ["stage | phase | label | burden", "------|-------|-------|-------"]
        for v in visits:
            rows.append(f"{v.get('stage')} | {v.get('phase')} | "
                        f"{v.get('progression_label')} | {v.get('burden')}")
        return "\n".join(rows)

    @staticmethod
    def report_text(visit: Optional[Dict]) -> str:
        if not visit:
            return "(none)"
        r = visit.get("report", {})
        return r.get("impression") or json.dumps(r, ensure_ascii=False)[:1500]
