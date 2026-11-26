#!/usr/bin/env python3
"""Download chosen CancerVerse CT volumes (checksum-verified) and extract only their liver masks from the label archive.

What: reads one or more subset lists (outputs/cancerverse/subsets/<name>.csv from cancerverse_subsets.py: cv_id,
size_bytes, sha256), downloads each CancerVerse/<cv_id>/ct.nii.gz from Hugging Face into <out>/<cv_id>/ct.nii.gz
(as .part, kept only if size and SHA-256 match, retried), then reads the label archive once as a stream and writes
only the wanted masks (default liver_lesion) to <out>/<cv_id>/segmentations/<mask>.nii.gz. Finished files are
skipped, so a rerun continues where it stopped. The archive is never unpacked (it would exceed our file quota).

Why: to look at a few HCC patients' whole follow-up (scans + tumour outlines) without the 3 TB image set.

Outputs: the files above and <out>/downloaded.csv (cv_id, ct_ok, mask_ok). Log: counts only.

Usage (job: data_pipeline/jobs/cancerverse_download.sh):
  python -I cancerverse_download.py --lists <a.csv> [<b.csv> ...] --out <dir> --archive <CancerVerse_Label.tar.gz>
  ... --test 2   # only the 2 lowest scan IDs (the archive is sorted by ID, so mask extraction stops early)
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import os
import subprocess
import sys
import tarfile
import time

URL = "https://huggingface.co/datasets/BodyMaps/CancerVerse/resolve/main/CancerVerse/%s/ct.nii.gz"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fetch(cv, size, digest, out, tries=3):
    """Download one CT volume; True when the file on disk has the expected size and SHA-256."""
    dest = os.path.join(out, cv, "ct.nii.gz")
    ok_mark = dest + ".sha256ok"
    if os.path.exists(dest) and os.path.exists(ok_mark):
        return True
    if os.path.exists(dest) and os.path.getsize(dest) == size and sha256(dest) == digest:
        open(ok_mark, "w").close()
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part.%d" % os.getpid()
    for attempt in range(tries):
        r = subprocess.run(["curl", "-fsSL", "--retry", "5", "--retry-delay", "20", "-o", part, URL % cv],
                           capture_output=True, timeout=3600)  # cluster curl 7.29: no --retry-all-errors
        if r.returncode == 0 and os.path.exists(part) and os.path.getsize(part) == size and sha256(part) == digest:
            os.replace(part, dest)
            open(ok_mark, "w").close()
            return True
        time.sleep(30 * (attempt + 1))
    if os.path.exists(part):
        os.remove(part)
    return False


def extract_masks(archive, out, wanted, masks):
    """Stream the archive once; write the wanted masks of the wanted scans. Returns the set of scans done."""
    names = {m + ".nii.gz" for m in masks}
    todo = {cv for cv in wanted
            if not all(os.path.exists(os.path.join(out, cv, "segmentations", n)) for n in names)}
    done = set(wanted) - todo
    if not todo:
        return done
    last = max(todo)
    found = {cv: 0 for cv in todo}
    with tarfile.open(archive, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            cv = next((p for p in m.name.split("/") if p.startswith("CV_")), "")
            base = os.path.basename(m.name)
            if cv in found and base in names:
                d = os.path.join(out, cv, "segmentations")
                os.makedirs(d, exist_ok=True)
                part = os.path.join(d, base + ".part.%d" % os.getpid())
                with open(part, "wb") as f:
                    f.write(tf.extractfile(m).read())
                os.replace(part, os.path.join(d, base))
                found[cv] += 1
            if cv > last:  # the archive is sorted by scan ID: nothing wanted after this point
                break
    done |= {cv for cv, k in found.items() if k == len(names)}
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lists", nargs="+", required=True, help="subset CSV files (cv_id, size_bytes, sha256)")
    ap.add_argument("--out", required=True, help="folder for <cv_id>/ct.nii.gz and masks")
    ap.add_argument("--archive", required=True, help="CancerVerse_Label.tar.gz")
    ap.add_argument("--masks", default="liver_lesion", help="comma-separated mask names to extract")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads")
    ap.add_argument("--test", type=int, default=0, help="only the N lowest scan IDs")
    args = ap.parse_args()

    rows = {}
    for path in args.lists:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[r["cv_id"]] = (int(r["size_bytes"]), r["sha256"])
    ids = sorted(rows)[: args.test] if args.test else sorted(rows)
    total_gb = sum(rows[c][0] for c in ids) / 1e9
    print("lists %d, scans %d, %.1f GB, workers %d" % (len(args.lists), len(ids), total_gb, args.workers), flush=True)
    t0 = time.time()
    ok = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch, c, rows[c][0], rows[c][1], args.out): c for c in ids}
        for k, f in enumerate(cf.as_completed(futs), 1):
            ok[futs[f]] = f.result()
            if k % 20 == 0 or k == len(ids):
                print("CT: %d / %d done, %d verified, %.0f s" % (k, len(ids), sum(ok.values()), time.time() - t0),
                      flush=True)
    masks = [m.strip() for m in args.masks.split(",") if m.strip()]
    mask_ok = extract_masks(args.archive, args.out, ids, masks)
    print("masks: %d / %d scans have %s, %.0f s" % (len(mask_ok), len(ids), ",".join(masks), time.time() - t0),
          flush=True)
    report = os.path.join(args.out, "downloaded.csv")
    with open(report + ".part", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cv_id", "ct_ok", "mask_ok"])
        for c in ids:
            w.writerow([c, ok[c], c in mask_ok])
    os.replace(report + ".part", report)
    bad = sum(1 for c in ids if not ok[c]) + sum(1 for c in ids if c not in mask_ok)
    print("done: CT verified %d / %d, masks %d / %d, %.1f GB, %.0f s" % (
        sum(ok.values()), len(ids), len(mask_ok), len(ids), total_gb, time.time() - t0), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
