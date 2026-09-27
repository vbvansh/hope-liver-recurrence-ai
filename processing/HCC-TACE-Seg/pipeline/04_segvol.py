#!/usr/bin/env python3
"""Step 5 (HCC-TACE-Seg) - SegVol text-prompted tumour masks -> _segvol/.

Runs lib/segvol_batch.py with --datasets HCC-TACE-Seg preset; every other argument passes through (see --help).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import segvol_batch  # noqa: E402

if __name__ == "__main__":
    if not any(a == "--datasets" or a.startswith("--datasets=") for a in sys.argv[1:]):
        sys.argv[1:1] = ["--datasets", "HCC-TACE-Seg"]
    segvol_batch.main()
