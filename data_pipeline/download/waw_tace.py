#!/usr/bin/env python3
"""Download WAW-TACE from Zenodo (record 12741586, CC BY 4.0) and unpack its zips.

All ten files of the record: four CT zips (NIfTI per patient and phase), tumour masks, organ masks (TotalSegmentator,
from the authors), clinical data, radiomics features, scan metadata and definitions. Each file resumes after an
interruption and must match Zenodo's MD5 before it gets its final name. Each zip is then unpacked into a folder of
its own name through a temporary folder; file count and sizes are checked against the zip's directory (the CT zips
are ZIP64, which Python reads natively). The zips are kept until the user agrees to delete them.

Output folder (--out):
  metadata/record.json
  <file>                       the ten files as on Zenodo
  <zip name without .zip>/     unpacked content of each zip

Usage: python -I -u waw_tace.py --out <project>/data/WAW-TACE
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/12741586"
RETRIES = 10  # each retry resumes where the last one stopped
MIN_FREE_BYTES = 20e9  # never fill the project storage completely
HEADERS = {"User-Agent": "hope-data-pipeline"}


def download(url, part, size):
    """Fill part up to size bytes, resuming with HTTP range requests."""
    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            return
        headers = dict(HEADERS, **({"Range": f"bytes={have}-"} if have else {}))
        t0, got, mark = time.time(), 0, 0
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as r, \
                    open(part, "ab") as f:
                if have and r.status != 206:
                    raise RuntimeError(f"server ignored the resume request (HTTP {r.status})")
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    f.write(chunk)
                    got += len(chunk)
                    if got - mark >= 2e9:
                        mark = got
                        print(f"    {(have + got) / 1e9:5.1f}/{size / 1e9:.1f} GB, "
                              f"{got / 1e6 / (time.time() - t0):5.1f} MB/s", flush=True)
            if have + got < size:
                print(f"    attempt {attempt}/{RETRIES}: connection ended early at {(have + got) / 1e9:.1f} GB",
                      flush=True)
        except Exception as e:  # network errors, timeouts
            print(f"    attempt {attempt}/{RETRIES} stopped at {(have + got) / 1e9:.1f} GB: {type(e).__name__}: {e}",
                  flush=True)
            if attempt < RETRIES:
                time.sleep(60 * min(attempt, 5))
    if not part.exists() or part.stat().st_size != size:
        raise RuntimeError(f"download incomplete after {RETRIES} attempts")


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack(zip_path, out):
    """Unpack zip_path into out through out.part; return (files, bytes)."""
    tmp = out.with_name(out.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)  # leftover of an interrupted run
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        zf.extractall(tmp)  # zipfile drops absolute paths and ".." from member names; reading checks each CRC
    wrong = [i.filename for i in infos if not (tmp / i.filename).is_file()
             or (tmp / i.filename).stat().st_size != i.file_size]
    if wrong:
        shutil.rmtree(tmp)
        raise ValueError(f"{len(wrong)} of {len(infos)} files missing or with the wrong size after unpacking")
    os.replace(tmp, out)
    return len(infos), sum(i.file_size for i in infos)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="raw data folder, e.g. <project>/data/WAW-TACE")
    args = ap.parse_args()
    meta = args.out / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(RECORD, headers=HEADERS), timeout=120) as r:
        record = json.load(r)
    (meta / "record.json").write_text(json.dumps(record, indent=1))
    files = sorted(record["files"], key=lambda f: f["size"])  # tables and masks first
    print(f"Zenodo record {record['id']}: {len(files)} files, {sum(f['size'] for f in files) / 1e9:.1f} GB", flush=True)

    unpacked = lambda f: args.out / f["key"][:-4]  # noqa: E731
    need = sum(f["size"] for f in files if not (args.out / f["key"]).exists()) + \
        sum(f["size"] for f in files if f["key"].endswith(".zip") and not unpacked(f).exists())
    st = os.statvfs(args.out)
    free = st.f_bavail * st.f_frsize
    print(f"to download and unpack: about {need / 1e9:.1f} GB | free: {free / 1e9:.0f} GB, {st.f_ffree:,} files",
          flush=True)
    if free < 1.05 * need + MIN_FREE_BYTES:
        sys.exit("Not enough free space on the project storage: stopping before any download.")

    failed, t_run = [], time.time()
    for k, f in enumerate(files, 1):
        dest, part = args.out / f["key"], args.out / (f["key"] + ".part")
        tag = f"[{k:2d}/{len(files)}] {f['key']} ({f['size'] / 1e9:.2f} GB)"
        try:
            if not dest.exists():
                t0 = time.time()
                download(f["links"]["self"], part, f["size"])
                digest = md5sum(part)
                if f"md5:{digest}" != f["checksum"]:
                    part.unlink()
                    raise ValueError(f"MD5 mismatch ({digest}, Zenodo {f['checksum']}): broken download removed")
                os.replace(part, dest)
                print(f"{tag}: downloaded in {(time.time() - t0) / 60:.1f} min, MD5 ok", flush=True)
            else:
                print(f"{tag}: already here", flush=True)
            if f["key"].endswith(".zip") and not unpacked(f).exists():
                t0 = time.time()
                n, nbytes = unpack(dest, unpacked(f))
                print(f"{tag}: unpacked {n} files, {nbytes / 1e9:.2f} GB, in {(time.time() - t0) / 60:.1f} min",
                      flush=True)
        except Exception as e:  # incomplete download, wrong MD5, bad zip
            failed.append(f["key"])
            print(f"{tag}: FAILED: {type(e).__name__}: {e}", flush=True)

    print(f"done: {len(files) - len(failed)}/{len(files)} files ready in {(time.time() - t_run) / 60:.0f} min; "
          f"zips kept")
    if failed:
        sys.exit(f"{len(failed)} files failed ({', '.join(failed)}); rerun to resume them.")


if __name__ == "__main__":
    main()
