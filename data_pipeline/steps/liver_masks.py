#!/usr/bin/env python3
"""Steps 5 and 8: TotalSegmentator liver mask for every scan (every phase of every patient), on the scan's own grid.

Runs in envs/totalseg (the job unpacks it on the GPU node's local disk, see jobs/liver_masks.sh). Same call and
settings as qc/totalseg_check.py: task total, liver only (roi_subset), full-resolution model. PLC-CECT is read from
its corrected files (Step 4) and listed last.

Modes:
  --make-list      write <out>/scans.csv: one row per scan (MCT-LTDiag, TCIA-CRLM, LiTS, WAW-TACE, PLC-CECT)
  --task K         process scans (K-1)*per_task+1 .. K*per_task of scans.csv (one array-job task)
  --test           the four scans of qc/totalseg_check.py plus the other three phases of its MCT patient; compares
                   the four with outputs/totalseg_test (same software, so Dice should be 1.000)
Output: <out>/<dataset>/<patient>/liver_<phase>.nii.gz (uint8 0/1) and <out>/log/<task>.csv (time, liver volume).
A scan whose mask exists is skipped, so a task can simply be submitted again. The log shows counts only.
"""
import argparse
import csv
import os
import re
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import mct_reader  # noqa: E402

MCT_PHASES = ["nc", "ap", "pvp", "dp"]
WAW_PHASES = {"0": "nc", "1": "ap", "2": "pvp", "3": "dp"}
FIELDS = ["scan_id", "dataset", "patient", "phase", "source", "member"]


def make_list(data):
    """[{scan_id, dataset, patient, phase, source, member}] for every scan."""
    rows = []
    for t in mct_reader.patients(data / "MCT-LTDiag").itertuples():
        rows += [{"dataset": "MCT-LTDiag", "patient": t.patient_id, "phase": ph, "source": str(t.tar),
                  "member": mct_reader.MEMBERS[ph]} for ph in MCT_PHASES]
    for folder in sorted(p for p in (data / "TCIA-CRLM" / "nifti").iterdir() if p.is_dir()):
        rows.append({"dataset": "TCIA-CRLM", "patient": folder.name, "phase": "pvp",
                     "source": str(folder / "ct.nii.gz"), "member": ""})
    lits = sorted((data / "LiTS" / "Task03_Liver" / "imagesTr").glob("liver_*.nii.gz"),
                  key=lambda p: int(re.search(r"\d+", p.name).group()))
    rows += [{"dataset": "LiTS", "patient": p.name[:-7], "phase": "pvp", "source": str(p), "member": ""} for p in lits]
    waw = sorted((data / "WAW-TACE").glob("ct_scans_*/*/*_scan.nii.gz"),
                 key=lambda p: (int(p.parent.name), p.name))
    for p in waw:
        pid, code = re.fullmatch(r"(\d+)_(\d)_scan\.nii\.gz", p.name).groups()
        rows.append({"dataset": "WAW-TACE", "patient": pid, "phase": WAW_PHASES[code], "source": str(p), "member": ""})
    fixed = data / "PLC-CECT" / "fixed"  # corrected PLC-CECT files (Step 4), appended last: tasks 131 on
    for folder in sorted(p for p in fixed.iterdir() if p.is_dir() and not p.name.endswith(".part")):
        rows += [{"dataset": "PLC-CECT", "patient": folder.name, "phase": ph, "source": str(folder / f"ct_{ph}.nii.gz"),
                  "member": ""} for ph in MCT_PHASES]
    for r in rows:
        r["scan_id"] = f"{r['dataset']}_{r['patient']}_{r['phase']}"
    return rows


def ct_file(scan, tmp):
    """Path of the scan as a NIfTI file (MCT-LTDiag phases are copied out of the patient's tar first)."""
    if not scan["member"]:
        return Path(scan["source"])
    path = Path(tmp) / f"{scan['scan_id']}.nii.gz"
    with tarfile.open(scan["source"]) as tf:
        member = next(m for m in tf.getmembers() if re.sub(r"^\./", "", m.name) == scan["member"])
        path.write_bytes(tf.extractfile(member).read())
    return path


def run(scans, out, threads, log_name):
    """Liver mask for each scan not done yet; one log row per scan."""
    import torch
    from totalsegmentator.python_api import totalsegmentator
    torch.set_num_threads(threads)
    (out / "log").mkdir(parents=True, exist_ok=True)
    done, t_run = 0, time.time()
    with tempfile.TemporaryDirectory() as tmp, open(out / "log" / f"{log_name}.csv", "a", newline="") as f:
        w = csv.writer(f)
        for k, scan in enumerate(scans, 1):
            dest = out / scan["dataset"] / scan["patient"] / f"liver_{scan['phase']}.nii.gz"
            if dest.exists():
                continue
            t0 = time.time()
            try:
                ct = ct_file(scan, tmp)
                raw = Path(tmp) / "ts.nii.gz"
                totalsegmentator(str(ct), str(raw), ml=True, roi_subset=["liver"], device="gpu", quiet=True,
                                 nr_thr_resamp=threads, nr_thr_saving=threads)
                seg = nib.load(raw)
                liver = (np.asarray(seg.dataobj) > 0).astype(np.uint8)
                dest.parent.mkdir(parents=True, exist_ok=True)
                part = dest.with_name(dest.name.replace(".nii.gz", ".part.nii.gz"))
                nib.save(nib.Nifti1Image(liver, seg.affine), part)
                os.replace(part, dest)
                ml = liver.sum() * np.prod(seg.header.get_zooms()[:3]) / 1000
                status = "ok" if ml > 0 else "empty"
                if ct.parent == Path(tmp):
                    ct.unlink()
            except Exception as e:  # unreadable scan, TotalSegmentator failure
                ml, status = 0.0, f"failed: {type(e).__name__}: {e}"
            seconds = time.time() - t0
            w.writerow([scan["scan_id"], round(seconds, 1), round(float(ml), 1), status])
            f.flush()
            done += 1
            print(f"[{k}/{len(scans)}] {scan['dataset']} {scan['phase']}: {seconds:.0f} s, liver {ml:.0f} ml, {status}",
                  flush=True)
    print(f"done: {done} scans in {(time.time() - t_run) / 60:.1f} min", flush=True)


def test_scans(data):
    """The scans of qc/totalseg_check.py (in its order) plus the other phases of its MCT patient."""
    rows = make_list(data)
    by = {(r["dataset"], r["patient"], r["phase"]): r for r in rows}
    tcia = sorted(p for p in (data / "TCIA-CRLM" / "nifti").iterdir() if p.is_dir())[0].name
    mct = mct_reader.patients(data / "MCT-LTDiag").patient_id.iloc[0]
    waw = next(r["patient"] for r in rows if r["dataset"] == "WAW-TACE" and r["phase"] == "pvp"
               and list((data / "WAW-TACE").glob(f"organ_masks_*/**/{r['patient']}_2_total_segmentator*.nii.gz")))
    picked = [by[("TCIA-CRLM", tcia, "pvp")], by[("LiTS", "liver_0", "pvp")], by[("MCT-LTDiag", mct, "pvp")],
              by[("WAW-TACE", waw, "pvp")]]
    return picked + [by[("MCT-LTDiag", mct, ph)] for ph in ("nc", "ap", "dp")]


def compare(scans, out, reference):
    """Dice of the new masks against the earlier test masks for the four test scans."""
    for scan in scans[:4]:
        old = reference / f"{scan['dataset']}_liver.nii.gz"
        new = out / scan["dataset"] / scan["patient"] / f"liver_{scan['phase']}.nii.gz"
        if old.exists() and new.exists():
            a, b = np.asarray(nib.load(old).dataobj) > 0, np.asarray(nib.load(new).dataobj) > 0
            print(f"same as the earlier test, {scan['dataset']}: Dice {2 * (a & b).sum() / (a.sum() + b.sum()):.4f}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="raw data root, e.g. <project>/data")
    ap.add_argument("--out", type=Path, required=True, help="output folder, e.g. <project>/outputs/liver_masks")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-list", action="store_true")
    mode.add_argument("--task", type=int)
    mode.add_argument("--test", action="store_true")
    ap.add_argument("--per-task", type=int, default=25, help="scans per array-job task")
    ap.add_argument("--threads", type=int, default=3, help="CPU threads (fair share of a GPU node)")
    ap.add_argument("--reference", type=Path, help="earlier test masks (--test), e.g. <project>/outputs/totalseg_test")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.make_list:
        rows = make_list(args.data)
        with open(args.out / "scans.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        counts = {d: sum(r["dataset"] == d for r in rows) for d in dict.fromkeys(r["dataset"] for r in rows)}
        print(f"{len(rows)} scans {counts}; {-(-len(rows) // args.per_task)} tasks of {args.per_task}")
        return
    if args.test:
        scans = test_scans(args.data)
        run(scans, args.out, args.threads, "test")
        if args.reference:
            compare(scans, args.out, args.reference)
        return
    rows = list(csv.DictReader(open(args.out / "scans.csv")))
    chunk = rows[(args.task - 1) * args.per_task: args.task * args.per_task]
    first = (args.task - 1) * args.per_task + 1
    print(f"task {args.task}: scans {first}-{first + len(chunk) - 1} of {len(rows)}", flush=True)
    run(chunk, args.out, args.threads, f"task_{args.task:04d}")


if __name__ == "__main__":
    main()
