#!/usr/bin/env python3
"""Step 7 (HCC-TACE-Seg) - anchor-organ registration + intensity matching -> CTProcess/.

Runs lib/build_longitudinal.py with --dataset HCC-TACE-Seg preset; every other argument passes through (see --help).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import build_longitudinal  # noqa: E402

if __name__ == "__main__":
    if not any(a == "--dataset" or a.startswith("--dataset=") for a in sys.argv[1:]):
        sys.argv[1:1] = ["--dataset", "HCC-TACE-Seg"]
    build_longitudinal.main()
