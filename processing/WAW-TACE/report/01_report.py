#!/usr/bin/env python3
"""Step 6 (WAW-TACE) - per-timepoint RECIST progression report.

Runs lib/run_report.py with --dataset WAW-TACE preset; every other argument passes through (see --help).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib", "report"))
import run_report  # noqa: E402

if __name__ == "__main__":
    if not any(a == "--dataset" or a.startswith("--dataset=") for a in sys.argv[1:]):
        sys.argv[1:1] = ["--dataset", "WAW-TACE"]
    run_report.main()
