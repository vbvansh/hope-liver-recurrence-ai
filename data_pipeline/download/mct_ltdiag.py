#!/usr/bin/env python3
"""Download MCT-LTDiag from Harvard Dataverse (doi:10.7910/DVN/S3RW15, CC0): per-patient tars and the two tables.

The tars stay as they are (180 GB for all 517; PLAN.md: the reader opens patients straight from them). A tar gets
its final name only after its MD5 matches the one Dataverse lists, so a rerun skips finished files and redoes the
rest. --sample takes the first tar (by name) for each letter a-e in the file names, one per tumour type (the
letter-to-type mapping is to be confirmed with meta_info_patient), to build the reader and the blob check before
the full download. Never run two copies at once on the same folder.

Output folder (--out):
  metadata/dataset.json, files.csv                      Dataverse listing (file name, id, size, MD5)
  metadata/meta_info_patient.csv, meta_info_tumor.csv   the two tables in their original format
  tars/<name>.tar
  failed_files.csv                                      only when files still fail after all retries

The log shows counts and sizes only, no file names (they are patient IDs).
Usage: python -I -u mct_ltdiag.py --out <project>/data/MCT-LTDiag [--sample]
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

API = "https://dataverse.harvard.edu/api"  # public files need no API token
DOI = "doi:10.7910/DVN/S3RW15"
TAR = re.compile(r"\d+([a-e])\d+\.tar")
RETRIES = 3
MIN_FREE_BYTES = 20e9  # never fill the project storage completely


def fetch(url, dest, timeout=300):
    """Stream url into dest (following Dataverse's redirect to its storage); return the MD5 of what was written."""
    h = hashlib.md5()
    req = urllib.request.Request(url, headers={"User-Agent": "hope-data-pipeline"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="raw data folder, e.g. <project>/data/MCT-LTDiag")
    ap.add_argument("--sample", action="store_true", help="only one tar per letter a-e (test set for the reader)")
    args = ap.parse_args()
    meta, tars = args.out / "metadata", args.out / "tars"
    meta.mkdir(parents=True, exist_ok=True)
    tars.mkdir(exist_ok=True)

    fetch(f"{API}/datasets/:persistentId/?persistentId={DOI}", meta / "dataset.json", timeout=120)
    version = json.loads((meta / "dataset.json").read_text())["data"]["latestVersion"]
    files = sorted((f["dataFile"] for f in version["files"]), key=lambda f: f["filename"])
    with open(meta / "files.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "id", "filesize", "md5"])
        w.writerows([d["filename"], d["id"], d["filesize"], d["checksum"]["value"]] for d in files)
    print(f"Dataverse version {version['versionNumber']}.{version['versionMinorNumber']}: {len(files)} files, "
          f"{sum(TAR.fullmatch(d['filename']) is not None for d in files)} patient tars, "
          f"{sum(d['filesize'] for d in files) / 1e9:.1f} GB")

    for d in files:  # the two tables, in the format the authors uploaded
        if d["filename"].endswith(".tab"):
            dest = meta / (d.get("originalFileName") or d["filename"])
            digest = fetch(f"{API}/access/datafile/{d['id']}?format=original", dest, timeout=120)
            print(f"table {dest.name}: {dest.stat().st_size:,} bytes, "
                  f"MD5 {'matches' if digest == d['checksum']['value'] else 'differs from'} the Dataverse listing")

    todo_all = [d for d in files if TAR.fullmatch(d["filename"])]
    if args.sample:
        first = {}
        for d in todo_all:
            first.setdefault(TAR.fullmatch(d["filename"]).group(1), d)
        todo_all = [first[k] for k in sorted(first)]
    todo = [d for d in todo_all if not (tars / d["filename"]).exists()]
    need = sum(d["filesize"] for d in todo)
    st = os.statvfs(args.out)
    free = st.f_bavail * st.f_frsize
    print(f"{len(todo_all) - len(todo)} of {len(todo_all)} tars already here; to download: {len(todo)} tars, "
          f"{need / 1e9:.2f} GB | free: {free / 1e9:.0f} GB, {st.f_ffree:,} files")
    if free < 1.05 * need + MIN_FREE_BYTES or st.f_ffree < len(todo) + 1000:
        sys.exit("Not enough free space or files on the project storage: stopping before any download.")

    failed, got, t_run = [], 0, time.time()
    for k, d in enumerate(todo, 1):
        final, part = tars / d["filename"], tars / (d["filename"] + ".part")
        tag = f"[{k:3d}/{len(todo)}] type {TAR.fullmatch(d['filename']).group(1)}"
        for attempt in range(1, RETRIES + 1):
            t0 = time.time()
            try:
                digest = fetch(f"{API}/access/datafile/{d['id']}", part)
                size = part.stat().st_size
                if size != d["filesize"] or digest != d["checksum"]["value"]:
                    raise ValueError(f"{size:,} bytes (listed {d['filesize']:,}), MD5 "
                                     f"{'ok' if digest == d['checksum']['value'] else 'mismatch'}")
            except Exception as e:  # network errors, timeouts, broken files
                print(f"{tag} attempt {attempt}/{RETRIES} failed: {type(e).__name__}: {e}", flush=True)
                part.unlink(missing_ok=True)
                if attempt < RETRIES:
                    time.sleep(60 * attempt)
                continue
            os.replace(part, final)
            got += size
            dt = max(time.time() - t0, 1e-3)
            print(f"{tag} {size / 1e6:7.1f} MB {dt:5.0f} s {size / 1e6 / dt:5.1f} MB/s md5 ok "
                  f"| {got / 1e9:6.2f}/{need / 1e9:.2f} GB", flush=True)
            break
        else:
            failed.append(d)

    here = [d for d in todo_all if (tars / d["filename"]).exists()]
    print(f"done: {len(here)}/{len(todo_all)} tars here "
          f"({dict(sorted(collections.Counter(TAR.fullmatch(d['filename']).group(1) for d in here).items()))}), "
          f"{sum(d['filesize'] for d in here) / 1e9:.2f} GB; this run {got / 1e9:.2f} GB in "
          f"{(time.time() - t_run) / 60:.0f} min")
    failed_csv = args.out / "failed_files.csv"
    if failed:
        with open(failed_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "id"])
            w.writerows([d["filename"], d["id"]] for d in failed)
        sys.exit(f"{len(failed)} tars failed {RETRIES} times (listed in failed_files.csv); rerun to retry them.")
    failed_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
