#!/usr/bin/env python3
"""Regenerate the clean <stub>.report.md for every existing report.json, using the
current renderer + re-derived target lesions (no model re-run needed)."""
from __future__ import annotations
import argparse
import glob
import json
import os
from pathlib import Path

import measurements as M
import clinical as C
from report_md import render_md

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=os.environ.get(
        "REPORTS", "/media/cbtil3/WhiteSD/CTProcessed-Reports"))
    ap.add_argument("--rewrite-json", action="store_true",
                    help="also rewrite report.json with re-derived target_lesions")
    args = ap.parse_args()
    root = Path(args.reports)
    patients = sorted({Path(p).parent for p in glob.glob(str(root / "*/*/ledger.json"))})
    n = 0
    for pdir in patients:
        dataset, patient = pdir.parent.name, pdir.name
        clin = C.load_clinical(dataset, patient)
        csum = C.patient_summary(clin)
        reps = sorted(pdir.glob("*.report.json"))
        prior_meas = None
        for rp in reps:
            r = json.loads(rp.read_text())
            meas = r.get("measurements", {}) or {}
            r["target_lesions"] = M.build_target_lesions(meas, prior_meas)
            rp.with_suffix("").with_suffix(".report.md").write_text(
                render_md(r, dataset, patient, csum))
            if args.rewrite_json:
                rp.write_text(json.dumps(r, indent=2))
            prior_meas = meas
            n += 1
    print(f"wrote {n} clean .report.md under {root}")


if __name__ == "__main__":
    main()
