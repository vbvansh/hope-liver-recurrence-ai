#!/usr/bin/env python3
"""Step 4 (HCC-TACE-Seg) - CADS Task-551 organ masks -> _cads551/.

Runs lib/cads_batch.py with --datasets HCC-TACE-Seg preset; every other argument passes through (see --help).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import cads_batch  # noqa: E402

if __name__ == "__main__":
    if not any(a == "--datasets" or a.startswith("--datasets=") for a in sys.argv[1:]):
        sys.argv[1:1] = ["--datasets", "HCC-TACE-Seg"]
    cads_batch.main()
