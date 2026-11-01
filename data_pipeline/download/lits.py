#!/usr/bin/env python3
"""Download LiTS as Medical Segmentation Decathlon Task03_Liver and unpack only the 131 labelled scans.

LiTS: Bilic et al., Med Image Anal 2023 (cite it); non-commercial licence. The tar (28.9 GB) comes from the MONAI
mirror, resumes after an interruption, and must match MONAI's published MD5 before anything is unpacked. From it
only imagesTr/, labelsTr/ and dataset.json are unpacked (not imagesTs/, not the macOS "._" files), through a
temporary folder, so the final folder only ever appears complete. The tar is kept until the user agrees to delete it.

Output folder (--out):
  Task03_Liver.tar
  Task03_Liver/imagesTr/liver_<n>.nii.gz, labelsTr/liver_<n>.nii.gz, dataset.json

Usage: python -I -u lits.py --out <project>/data/LiTS
"""
import argparse
import collections
import hashlib
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar"
MD5 = "a90ec6c4aa7f6a3d087205e23d4e6397"  # MONAI DecathlonDataset, monai/apps/datasets.py
KEEP = re.compile(r"(?:\./)?Task03_Liver/(?:(imagesTr|labelsTr)/liver_\d+\.nii\.gz|dataset\.json)")
EXPECTED = 131  # labelled scans
RETRIES = 10  # each retry resumes where the last one stopped
MIN_FREE_BYTES = 20e9  # never fill the project storage completely


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "hope-data-pipeline"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def download(url, part, size):
    """Fill part up to size bytes, resuming with HTTP range requests."""
    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            return
        headers = {"User-Agent": "hope-data-pipeline"}
        if have:
            headers["Range"] = f"bytes={have}-"
        t0, got, mark = time.time(), 0, 0
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as r, \
                    open(part, "ab") as f:
                if have and r.status != 206:
                    raise RuntimeError(f"server ignored the resume request (HTTP {r.status})")
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    f.write(chunk)
                    got += len(chunk)
                    if got - mark >= 1e9:
                        mark = got
                        print(f"  {(have + got) / 1e9:5.1f}/{size / 1e9:.1f} GB, "
                              f"{got / 1e6 / (time.time() - t0):5.1f} MB/s", flush=True)
            if have + got < size:
                print(f"  attempt {attempt}/{RETRIES}: connection ended early at {(have + got) / 1e9:.1f} GB",
                      flush=True)
        except Exception as e:  # network errors, timeouts
            print(f"  attempt {attempt}/{RETRIES} stopped at {(have + got) / 1e9:.1f} GB: {type(e).__name__}: {e}",
                  flush=True)
            if attempt < RETRIES:
                time.sleep(60 * min(attempt, 5))
    if not part.exists() or part.stat().st_size != size:
        sys.exit(f"Download incomplete after {RETRIES} attempts; rerun to resume.")


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack(tar_path, out):
    """Unpack the labelled scans into out (= <folder>/Task03_Liver) through a temporary folder."""
    tmp = out.with_name(out.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)  # leftover of an interrupted run
    tmp.mkdir()
    sizes = {}
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if m.isfile() and KEEP.fullmatch(m.name):
                tf.extract(m, tmp, filter="data")  # the data filter refuses absolute paths and ".."
                sizes[Path(m.name)] = m.size  # Path drops a leading "./"
    src = tmp / "Task03_Liver"
    images = sorted(p.name for p in (src / "imagesTr").glob("liver_*.nii.gz"))
    labels = sorted(p.name for p in (src / "labelsTr").glob("liver_*.nii.gz"))
    wrong_size = [p for p, s in sizes.items() if (tmp / p).stat().st_size != s]
    if len(images) != EXPECTED or images != labels or not (src / "dataset.json").exists() or wrong_size:
        shutil.rmtree(tmp)
        sys.exit(f"Unpack check failed: {len(images)} images, {len(labels)} labels, names match {images == labels}, "
                 f"{len(wrong_size)} files with the wrong size.")
    os.replace(src, out)
    tmp.rmdir()
    return len(images), sum(sizes.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="raw data folder, e.g. <project>/data/LiTS")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tar_path, out = args.out / "Task03_Liver.tar", args.out / "Task03_Liver"
    part = tar_path.with_name(tar_path.name + ".part")
    if out.exists():
        print("Task03_Liver/ is already unpacked: nothing to do.")
        return

    if not tar_path.exists():
        size = remote_size(URL)
        have = part.stat().st_size if part.exists() else 0
        st = os.statvfs(args.out)
        free = st.f_bavail * st.f_frsize
        print(f"tar: {size / 1e9:.2f} GB, {have / 1e9:.2f} GB already here | free: {free / 1e9:.0f} GB")
        if free < (size - have) + 0.75 * size + MIN_FREE_BYTES:  # tar + labelled scans (~70% of it)
            sys.exit("Not enough free space on the project storage: stopping before the download.")
        t0 = time.time()
        download(URL, part, size)
        print(f"downloaded in {(time.time() - t0) / 60:.0f} min; checking MD5 ...", flush=True)
        digest = md5sum(part)
        if digest != MD5:
            part.unlink()
            sys.exit(f"MD5 mismatch ({digest}, expected {MD5}): broken download removed; rerun to download again.")
        os.replace(part, tar_path)
        print("MD5 ok", flush=True)

    t0 = time.time()
    n, nbytes = unpack(tar_path, out)
    print(f"done: {n} labelled scans (images + labels) unpacked, {nbytes / 1e9:.2f} GB, "
          f"in {(time.time() - t0) / 60:.0f} min; Task03_Liver.tar kept")


if __name__ == "__main__":
    main()
