#!/usr/bin/env python3
"""Download TCIA Colorectal-Liver-Metastases (doi:10.7937/QXK2-QG03, CC BY 4.0): CT + SEG series and clinical table.

The NBIA API sends one zip per DICOM series. A zip gets its final name only after it passes the checks (file count
matches the TCIA listing, zip CRC, MD5 of every file when the zip carries TCIA's md5hashes.csv); then it is unpacked
into its own series folder (via a temporary folder, so a finished folder is always complete). A rerun skips series
that are already unpacked and redoes the rest. The zips are kept until the user agrees to delete them.
Never run two copies at once on the same folder.

Output folder (--out):
  metadata/series.json, series.csv      TCIA series listing (patient, study, series, modality, file count, size)
  metadata/*.xlsx, *.tcia               clinical table, official manifest and its digest (collection page)
  dicom_zips/<SeriesInstanceUID>.zip
  dicom/<PatientID>/<Modality>_<SeriesInstanceUID>/   unpacked DICOM files (+ md5hashes.csv)
  failed_series.csv                     only when series still fail after all retries

The log shows counts and sizes only, no patient IDs.
Usage: python -I -u tcia_crlm.py --out <project>/data/TCIA-CRLM [--limit N]
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"  # public collections need no login
COLLECTION = "Colorectal-Liver-Metastases"
EXTRAS = [  # linked from https://www.cancerimagingarchive.net/collection/colorectal-liver-metastases/
    "https://www.cancerimagingarchive.net/wp-content/uploads/Colorectal-Liver-Metastases-Clinical-data-April-2023.xlsx",
    "https://www.cancerimagingarchive.net/wp-content/uploads/Colorectal-Liver-Metastases-November-2022-manifest.tcia",
    "https://www.cancerimagingarchive.net/wp-content/uploads/Colorectal-Liver-Metastases-November-2022-manifes-nbia-digest.xlsx",
]
RETRIES = 3
MIN_FREE_BYTES = 20e9  # never fill the project storage completely


def fetch(url, dest, timeout=300):
    """Stream url into dest; return the number of bytes written."""
    req = urllib.request.Request(url, headers={"User-Agent": "hope-data-pipeline"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    return dest.stat().st_size


def md5_list(zf):
    """File name -> MD5 from the md5hashes.csv inside an NBIA zip, or None when the zip has none."""
    lists = [n for n in zf.namelist() if Path(n).name.lower() == "md5hashes.csv"]
    if not lists:
        return None
    hashes = {}
    with zf.open(lists[0]) as f:
        for row in csv.reader(line.decode("utf-8", "replace") for line in f):
            cells = [c.strip() for c in row if c.strip()]
            md5 = next((c.lower() for c in cells if re.fullmatch(r"[0-9a-fA-F]{32}", c)), None)
            name = next((c for c in cells if c.lower() != md5), None)
            if md5 and name:  # the header row has no MD5 and is skipped
                hashes[Path(name).name] = md5
    return hashes


def check_zip(path, expected_files):
    """Return (problem or None, whether MD5s were checked)."""
    with zipfile.ZipFile(path) as zf:
        files = [i for i in zf.infolist() if not i.is_dir() and Path(i.filename).name.lower() != "md5hashes.csv"]
        if len(files) != expected_files:
            return f"{len(files)} files in zip, TCIA lists {expected_files}", False
        hashes = md5_list(zf)
        for i in files:
            h = hashlib.md5()
            with zf.open(i) as f:  # reading a member to the end also checks its zip CRC
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if hashes is not None and hashes.get(Path(i.filename).name) != h.hexdigest():
                return f"MD5 mismatch for {Path(i.filename).name}", True
    return None, hashes is not None


def unzip(zip_path, dest, expected_files):
    """Unpack into dest through dest.part, so dest only ever appears complete."""
    tmp = dest.with_name(dest.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)  # leftover of an interrupted run
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)  # zipfile drops absolute paths and ".." from member names
    n = sum(1 for p in tmp.rglob("*") if p.is_file() and p.name.lower() != "md5hashes.csv")
    if n != expected_files:
        shutil.rmtree(tmp)
        raise ValueError(f"{n} files after unzip, TCIA lists {expected_files}")
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="raw data folder, e.g. <project>/data/TCIA-CRLM")
    ap.add_argument("--limit", type=int, help="only the first N series (sorted by patient, modality): test run")
    args = ap.parse_args()
    meta, zips, dicom = args.out / "metadata", args.out / "dicom_zips", args.out / "dicom"
    for d in (meta, zips, dicom):
        d.mkdir(parents=True, exist_ok=True)

    # series listing from the API (fresh every run); the official manifest only cross-checks it
    fetch(f"{API}/getSeries?Collection={COLLECTION}", meta / "series.json", timeout=120)
    series = sorted(json.loads((meta / "series.json").read_text()),
                    key=lambda s: (s["PatientID"], s["Modality"], s["SeriesInstanceUID"]))
    with open(meta / "series.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "study_uid", "series_uid", "modality", "image_count", "file_size_bytes"])
        for s in series:
            w.writerow([s["PatientID"], s["StudyInstanceUID"], s["SeriesInstanceUID"], s["Modality"],
                        s["ImageCount"], s.get("FileSize", "")])
    for url in EXTRAS:
        dest = meta / url.rsplit("/", 1)[1]
        if not dest.exists():
            part = meta / (dest.name + ".part")
            fetch(url, part, timeout=120)
            os.replace(part, dest)

    modalities = collections.Counter(s["Modality"] for s in series)
    print(f"TCIA lists {len(series)} series ({dict(modalities)}) of {len({s['PatientID'] for s in series})} patients, "
          f"{sum(s['ImageCount'] for s in series):,} files, {sum(s.get('FileSize') or 0 for s in series) / 1e9:.2f} GB")
    manifest = (meta / EXTRAS[1].rsplit("/", 1)[1]).read_text()
    manifest_uids = {ln.strip() for ln in manifest.splitlines() if re.fullmatch(r"[0-9]+(\.[0-9]+)+", ln.strip())}
    api_uids = {s["SeriesInstanceUID"] for s in series}
    print(f"official manifest: {len(manifest_uids)} series, {len(manifest_uids - api_uids)} missing from the API "
          f"listing, {len(api_uids - manifest_uids)} extra in it")

    def zip_of(s):
        return zips / f"{s['SeriesInstanceUID']}.zip"

    def folder_of(s):
        return dicom / s["PatientID"] / f"{s['Modality']}_{s['SeriesInstanceUID']}"

    if args.limit:
        series = series[:args.limit]
    todo = [s for s in series if not folder_of(s).exists()]
    need_dl = sum(s.get("FileSize") or 0 for s in todo if not zip_of(s).exists())
    need_unzip = sum(s.get("FileSize") or 0 for s in todo)
    need_files = sum(s["ImageCount"] + 3 for s in todo)  # DICOM files + md5 list, folder, zip
    st = os.statvfs(args.out)
    free, free_files = st.f_bavail * st.f_frsize, st.f_ffree
    print(f"{len(series) - len(todo)} series already unpacked; to do: {len(todo)} series "
          f"(download {need_dl / 1e9:.2f} GB, unpacked {need_unzip / 1e9:.2f} GB, {need_files:,} files) "
          f"| free: {free / 1e9:.0f} GB, {free_files:,} files")
    if free < 1.1 * (need_dl + need_unzip) + MIN_FREE_BYTES or free_files < need_files + 1000:
        sys.exit("Not enough free space or files on the project storage: stopping before any download.")

    failed, no_md5, got, t_run = [], 0, 0, time.time()
    for k, s in enumerate(todo, 1):
        uid, n = s["SeriesInstanceUID"], s["ImageCount"]
        zip_path, part = zip_of(s), zips / f"{uid}.zip.part"
        tag = f"[{k:3d}/{len(todo)}] {s['Modality']:3s} {n:4d} files"
        for attempt in range(1, RETRIES + 1):
            t0 = time.time()
            try:
                note = "zip already on disk"
                if not zip_path.exists():
                    size = fetch(f"{API}/getImageWithMD5Hash?SeriesInstanceUID={uid}", part)
                    problem, md5_checked = check_zip(part, n)
                    if problem:
                        raise ValueError(problem)
                    os.replace(part, zip_path)
                    got += size
                    no_md5 += not md5_checked
                    note = (f"{size / 1e6:7.1f} MB {size / 1e6 / max(time.time() - t0, 1e-3):5.1f} MB/s "
                            f"{'md5 ok' if md5_checked else 'crc ok, no md5 list'}")
                folder_of(s).parent.mkdir(exist_ok=True)
                unzip(zip_path, folder_of(s), n)
            except Exception as e:  # network errors, timeouts, broken zips, wrong file counts
                print(f"{tag} attempt {attempt}/{RETRIES} failed: {type(e).__name__}: {e}", flush=True)
                part.unlink(missing_ok=True)
                if attempt < RETRIES:
                    time.sleep(60 * attempt)
                continue
            print(f"{tag} {note}, unpacked {time.time() - t0:5.0f} s | {got / 1e9:5.2f}/{need_dl / 1e9:.2f} GB",
                  flush=True)
            break
        else:
            failed.append(s)

    done = [s for s in series if folder_of(s).exists()]
    zip_bytes = sum(zip_of(s).stat().st_size for s in series if zip_of(s).exists())
    print(f"done: {len(done)}/{len(series)} series unpacked ({dict(collections.Counter(s['Modality'] for s in done))}), "
          f"{sum(s['ImageCount'] for s in done):,} DICOM files; zips on disk {zip_bytes / 1e9:.2f} GB; "
          f"this run downloaded {got / 1e9:.2f} GB in {(time.time() - t_run) / 60:.0f} min, "
          f"{no_md5} zips without an MD5 list")
    failed_csv = args.out / "failed_series.csv"
    if failed:
        with open(failed_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "series_uid", "modality"])
            w.writerows([s["PatientID"], s["SeriesInstanceUID"], s["Modality"]] for s in failed)
        sys.exit(f"{len(failed)} series failed {RETRIES} times (listed in failed_series.csv); rerun to retry them.")
    failed_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
