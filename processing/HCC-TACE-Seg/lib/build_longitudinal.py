#!/usr/bin/env python3
"""Longitudinal CT standard for one dataset: cross-timepoint registration +
intensity ("colour") matching, driven by the shared core longitudinal_pipeline.py.

Replaces the seven per-dataset build_longitudinal.py drivers; the only per-dataset
setting is the anchor organ (CADS-551 label ids, or "auto").

  python3 build_longitudinal.py --dataset NLST                 # full cohort (>=2 timepoints)
  python3 build_longitudinal.py --dataset NLST --limit 3       # quick sample
  python3 build_longitudinal.py --dataset NLST --patients <id> ...
  python3 build_longitudinal.py --dataset NLST --force         # reprocess existing outputs
  python3 build_longitudinal.py --dataset NLST --no-intensity-match   # registration only
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longitudinal_pipeline import run   # noqa: E402

# dataset -> anchor organ (CADS-551 ids, or "auto" = largest soft organ shared across timepoints)
ANCHORS = {
    "CPTAC-CCRCC": [2, 3],               # kidney (L/R)
    "CPTAC-PDA": [10],                   # pancreas
    "EAY131": "auto",                    # mixed-site
    "HCC-TACE-Seg": [5],                 # liver
    "NLST": [13, 14, 15, 16, 17],        # lung lobes
    "RIDER": [13, 14, 15, 16, 17],       # lung lobes
    "WAW-TACE": [5],                     # liver
}


def main():
    ap = argparse.ArgumentParser(description="Abdomen longitudinal pipeline")
    ap.add_argument("--dataset", required=True, choices=sorted(ANCHORS))
    ap.add_argument("--processed-in", default="/media/cbtil3/WhiteSD/CTProcessed")
    ap.add_argument("--out", default="/media/cbtil3/WhiteSD/CTProcess")
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-intensity-match", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a.dataset, ANCHORS[a.dataset], a.processed_in, a.out, patients=a.patients,
        limit=a.limit, do_intensity=not a.no_intensity_match, force=a.force)


if __name__ == "__main__":
    main()
