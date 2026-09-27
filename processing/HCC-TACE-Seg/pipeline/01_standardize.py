#!/usr/bin/env python3
"""Step 2 (HCC-TACE-Seg) - DICOM -> 1 mm RAS float32 HU, rigid to baseline.

Runs lib/standardize.py with --dataset HCC-TACE-Seg preset; every other argument passes through (see --help).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import standardize  # noqa: E402

if __name__ == "__main__":
    if not any(a == "--dataset" or a.startswith("--dataset=") for a in sys.argv[1:]):
        sys.argv[1:1] = ["--dataset", "HCC-TACE-Seg"]
    standardize.main()
