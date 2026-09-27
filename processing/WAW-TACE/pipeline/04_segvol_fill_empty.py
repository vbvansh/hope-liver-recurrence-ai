#!/usr/bin/env python3
"""Step 4 (WAW-TACE) - re-run SegVol on empty masks listed in --report.

Runs lib/segvol_fill_empty.py; every other argument passes through (see --help).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import segvol_fill_empty  # noqa: E402

if __name__ == "__main__":
    segvol_fill_empty.main()
