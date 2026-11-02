#!/usr/bin/env python3
"""Download PLC-CECT (Luo et al., Sci Data 2025; CC BY 4.0) from its Google Drive copy, file by file.

The official release is one 42 GB tar on Science Data Bank (doi:10.57760/sciencedb.12207), but it arrives at under
1 MB/s here (tested 2026-11-02, also with parallel streams). The Drive folder holds the same files unpacked
(ct_files/, liver_mask_files/, mask_files/ and patient_data.csv, the same layout as the tar) and downloads much
faster. Drive lists no checksums for these files, so each one must be a complete gzip (its built-in CRC) holding a
NIfTI header before it gets its final name; patient_data.csv is compared with the MD5 Science Data Bank lists.
A rerun skips finished files. Never run two copies at once on the same folder.

Output folder (--out):
  metadata/files.csv                          Drive listing (folder, file name, Drive id)
  patient_data.csv
  ct_files/, liver_mask_files/, mask_files/   NIfTI files as on the Drive
  failed_files.csv                            only when files still fail after all retries

The log shows counts and sizes only, no file names (they hold patient IDs).
Usage: python -I -u plc_cect.py --out <project>/data/PLC-CECT [--workers 4] [--limit N]
"""
import argparse
import collections
import csv
import gzip
import hashlib
import html
import os
import re
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DRIVE_ROOT = "1KQiLUFyQZfy1jXYNb98_cVAU15wyo5Uo"  # PLC-CECT_Dataset folder shared with the team
LIST_URL = "https://drive.google.com/embeddedfolderview?id={}"  # lists every item (the normal page shows 50)
FILE_URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
EXPECTED = {"ct_files": 1444, "liver_mask_files": 1444, "mask_files": 1112}  # 361 x 4 phases; masks: 278 cancers
PATIENT_DATA_MD5 = "f12a18577c714627afcafe1eb84b1f1a"  # patient_data.csv on Science Data Bank
OFFICIAL_BYTES = 42_051_328_000  # CECT.tar on Science Data Bank: the size estimate for the space check
ENTRY = re.compile(r'<div class="flip-entry" id="entry-([A-Za-z0-9_-]+)".*?<a href="([^"]+)".*?'
                   r'class="flip-entry-title">([^<]+)<', re.S)
RETRIES = 5
MIN_FREE_BYTES = 20e9  # never fill the project storage completely
HEADERS = {"User-Agent": "hope-data-pipeline"}


def listing(folder_id):
    """[(name, id, is_folder)] for every item in a public Drive folder."""
    req = urllib.request.Request(LIST_URL.format(folder_id), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        page = r.read().decode("utf-8", "replace")
    return [(html.unescape(name), fid, "/drive/folders/" in href) for fid, href, name in ENTRY.findall(page)]


def check_nifti_gz(path):
    """Raise unless path is a complete gzip file holding a NIfTI-1 or NIfTI-2 header."""
    with gzip.open(path, "rb") as f:
        head = f.read(540)
        while f.read(1 << 24):  # reading to the end checks the gzip CRC and length
            pass
    if head[344:348] != b"n+1\0" and head[4:8] != b"n+2\0":
        raise ValueError("not a NIfTI file")


def fetch(file_id, dest):
    """Download one Drive file into dest.part; return its size and MD5 (no final name yet)."""
    part, h = dest.with_name(dest.name + ".part"), hashlib.md5()
    with urllib.request.urlopen(urllib.request.Request(FILE_URL.format(file_id), headers=HEADERS), timeout=300) as r:
        if r.headers.get_content_type() == "text/html":
            raise RuntimeError("Drive sent a web page instead of the file (download limit or access)")
        with open(part, "wb") as f:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                f.write(chunk)
                h.update(chunk)
    return part, part.stat().st_size, h.hexdigest()


def get_nifti(item, out):
    """Download and check one NIfTI file with retries; return (bytes, error or None)."""
    folder, name, file_id = item
    dest = out / folder / name
    error = None
    for attempt in range(1, RETRIES + 1):
        try:
            part, size, _ = fetch(file_id, dest)
            check_nifti_gz(part)
            os.replace(part, dest)
            return size, None
        except Exception as e:  # network errors, Drive limits, broken files
            error = f"{type(e).__name__}: {e}"
            dest.with_name(dest.name + ".part").unlink(missing_ok=True)
            if attempt < RETRIES:
                time.sleep(30 * attempt)
    return 0, error


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="raw data folder, e.g. <project>/data/PLC-CECT")
    ap.add_argument("--workers", type=int, default=4, help="files downloaded at the same time")
    ap.add_argument("--limit", type=int, help="only the first N files of each folder: test run")
    args = ap.parse_args()
    meta = args.out / "metadata"
    meta.mkdir(parents=True, exist_ok=True)

    top = {name: (fid, is_folder) for name, fid, is_folder in listing(DRIVE_ROOT)}
    raw = {name: fid for name, fid, is_folder in listing(top["raw"][0]) if is_folder}
    items = []
    for folder in EXPECTED:
        files = sorted((name, fid) for name, fid, is_folder in listing(raw[folder])
                       if not is_folder and name.endswith(".nii.gz"))
        print(f"{folder}: {len(files)} files on the Drive (expected {EXPECTED[folder]}), "
              f"{len({n.split('_')[0].split('-')[0] for n, _ in files})} patients", flush=True)
        items += [(folder, name, fid) for name, fid in (files[:args.limit] if args.limit else files)]
    with open(meta / "files.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["folder", "filename", "drive_id"])
        w.writerows(items)

    if not (args.out / "patient_data.csv").exists():
        part, size, digest = fetch(top["patient_data.csv"][0], args.out / "patient_data.csv")
        os.replace(part, args.out / "patient_data.csv")
        print(f"patient_data.csv: {size:,} bytes, MD5 {'matches' if digest == PATIENT_DATA_MD5 else 'DIFFERS from'} "
              f"the Science Data Bank file", flush=True)

    for folder in EXPECTED:
        (args.out / folder).mkdir(exist_ok=True)
    todo = [it for it in items if not (args.out / it[0] / it[1]).exists()]
    total = sum(EXPECTED.values())
    st = os.statvfs(args.out)
    free = st.f_bavail * st.f_frsize
    need = OFFICIAL_BYTES * len(todo) / total
    print(f"{len(items) - len(todo)} of {len(items)} files already here; to download: {len(todo)} files, "
          f"about {need / 1e9:.1f} GB | free: {free / 1e9:.0f} GB, {st.f_ffree:,} files", flush=True)
    if free < 1.1 * need + MIN_FREE_BYTES or st.f_ffree < len(todo) + 1000:
        sys.exit("Not enough free space or files on the project storage: stopping before any download.")

    done, got, failed, errors = 0, 0, [], collections.Counter()
    lock, t0 = threading.Lock(), time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(get_nifti, it, args.out): it for it in todo}
        for fut in as_completed(futures):
            size, error = fut.result()
            with lock:
                done += 1
                got += size
                if error:
                    failed.append(futures[fut])
                    errors[error.split(":")[0]] += 1
                    print(f"[{done:4d}/{len(todo)}] failed after {RETRIES} tries: {error}", flush=True)
                if done % 100 == 0 or done == len(todo):
                    print(f"[{done:4d}/{len(todo)}] {got / 1e9:5.1f} GB, {got / 1e6 / (time.time() - t0):5.1f} MB/s, "
                          f"{len(failed)} failed", flush=True)

    here = collections.Counter(it[0] for it in items if (args.out / it[0] / it[1]).exists())
    print(f"done: {dict(here)} files here of {dict(collections.Counter(it[0] for it in items))}; "
          f"this run {got / 1e9:.1f} GB in {(time.time() - t0) / 60:.0f} min")
    failed_csv = args.out / "failed_files.csv"
    if failed:
        with open(failed_csv, "w", newline="") as f:
            csv.writer(f).writerows([("folder", "filename", "drive_id"), *failed])
        sys.exit(f"{len(failed)} files failed ({dict(errors)}; listed in failed_files.csv); rerun later to retry.")
    failed_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
