#!/usr/bin/env python3
"""CADS Task-551 batch: render every CT of every multi-timepoint patient.

Goal: across one or more processed datasets, keep only patients that have a CT
at >=2 timepoints (filtering OUT single-CT patients), run CADS Task-551 on
EVERY timepoint's CT, and render one img+seg multi-slice figure per CT.

Single-GPU by design: pins CUDA_VISIBLE_DEVICES to one device and runs CADS
inference ONCE over all selected CTs, so the nnUNet model loads a single time
instead of once per volume. With dozens of CTs that is the difference between
minutes and an hour of pure model-load overhead.

It reuses the shared engine `cads_demo.py` (same figure layout, colors, slice
picking that skips blank padding slices) and the same on-disk cache, so it is
resumable: a CT whose mask already exists at
`<dataset>/_cads551/<pid>/<stage>_seg.nii.gz` is skipped on re-runs.

Layout assumed (per dataset under <processed>):
  <dataset>/<patient>/<stage>/ct.nii.gz      e.g. 000_baseline, 001_followup_1
A "timepoint" is any stage dir holding a `ct.nii.gz`; a patient qualifies when
it has at least --min-timepoints of them.

Inputs are symlinked (not copied) into the staging dir, so no extra disk is
needed for the CT volumes during inference.

Outputs:
  <dataset>/_cads551/<pid>/<stage>_seg.nii.gz        cached multilabel mask
  <dataset>/visualization/<pid>_<stage>_cads551.png  per-CT figure (data drive)
  <gallery>/<dataset>_<pid>_<stage>_cads551.png      flat gallery (in this repo)

Usage:
  export CADS_PYTHON=/path/to/envs/CADS_env/bin/python
  export CADS_REPO=/path/to/CADS
  python3 cads_batch.py --gpu 0                       # all datasets, GPU 0
  python3 cads_batch.py --datasets NLST WAW-TACE --gpu 1
  python3 cads_batch.py --list                        # just print the worklist
  python3 cads_batch.py --render-only                 # re-draw from cached masks
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

# Reuse the single-example engine for everything figure/cache related.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cads_demo as eng

DEFAULT_DATASETS = ["CPTAC-CCRCC", "CPTAC-PDA", "EAY131", "HCC-TACE-Seg",
                    "NLST", "RIDER", "WAW-TACE"]


class Item:
    """One CT to segment + render."""
    __slots__ = ("dataset", "pid", "stage", "ct", "seg", "png", "token")

    def __init__(self, root, dataset, pid, stage):
        self.dataset = dataset
        self.pid = pid
        self.stage = stage
        self.ct = os.path.join(root, dataset, pid, stage, "ct.nii.gz")
        self.seg = os.path.join(root, dataset, "_cads551", pid,
                                f"{stage}_seg.nii.gz")
        self.png = os.path.join(root, dataset, "visualization",
                                f"{pid}_{stage}_cads551.png")
        # filename-safe, unique key used both as the staged input name and the
        # CADS output subdir name (predict names outputs after the basename).
        self.token = f"{dataset}__{pid}__{stage}"


def _stable(ct_path, min_age, now):
    """True if ct.nii.gz exists and was last modified >= min_age seconds ago.

    Guards against feeding a volume the converter is still writing (writes are
    not atomic): an in-flight file keeps getting fresh mtimes, so it stays
    'unstable' until standardize.py stops touching it."""
    if not os.path.exists(ct_path):
        return False
    if min_age and (now - os.path.getmtime(ct_path)) < min_age:
        return False
    return True


def discover(root, datasets, min_tp, min_age=0.0):
    """Worklist of Items for every timepoint of every >=min_tp patient.

    With min_age > 0, a timepoint whose ct.nii.gz is younger than min_age
    seconds is treated as not-yet-ready (still converting) and excluded; such a
    patient may temporarily drop below min_tp and get picked up on a later pass.
    """
    now = time.time()
    items, skipped_single = [], 0
    for ds in datasets:
        dpath = os.path.join(root, ds)
        if not os.path.isdir(dpath):
            eng.log(f"  [skip] dataset not found: {dpath}")
            continue
        for pid in sorted(os.listdir(dpath)):
            if pid.startswith((".", "_")) or pid == "visualization":
                continue
            pp = os.path.join(dpath, pid)
            if not os.path.isdir(pp):
                continue
            stages = [s for s in sorted(os.listdir(pp))
                      if not s.startswith((".", "_"))
                      and _stable(os.path.join(pp, s, "ct.nii.gz"),
                                  min_age, now)]
            if len(stages) < min_tp:
                if len(stages) >= 1:
                    skipped_single += 1
                continue
            for s in stages:
                items.append(Item(root, ds, pid, s))
    return items, skipped_single


def run_cads_batch(to_run, out_dir, cads_python, cads_repo, use_cpu, gpu):
    """Segment every Item in `to_run` with ONE predict call (model loads once)."""
    predict = os.path.join(cads_repo, "cads", "scripts", "predict_images.py")
    if not os.path.exists(predict):
        sys.exit(f"CADS inference script not found: {predict}\n"
                 f"set --cads-repo / $CADS_REPO to the cloned murong-xu/CADS")
    with tempfile.TemporaryDirectory(prefix="cads551_batch_") as tmp:
        in_dir = os.path.join(tmp, "in")
        os.makedirs(in_dir)
        for it in to_run:
            os.symlink(os.path.abspath(it.ct),
                       os.path.join(in_dir, f"{it.token}.nii.gz"))
        env = dict(os.environ)
        if gpu is not None and not use_cpu:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [cads_python, predict, "-in", in_dir, "-out", out_dir,
               "-task", "551"]
        if use_cpu:
            cmd.append("--cpu")
        eng.log(f"  segmenting {len(to_run)} CT(s) on "
                f"{'CPU' if use_cpu else 'GPU ' + str(gpu)} (model loads once)")
        eng.log("  $ " + " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)
    # distribute produced masks into the per-dataset cache
    done, failed = [], []
    for it in to_run:
        produced = os.path.join(out_dir, it.token, f"{it.token}_part_551.nii.gz")
        if os.path.exists(produced):
            os.makedirs(os.path.dirname(it.seg), exist_ok=True)
            shutil.copy(produced, it.seg)
            done.append(it)
        else:
            failed.append(it)
    return done, failed


def render_item(it, n_slices, colors, gallery):
    """Draw the img+seg figure for one Item; mirror into the flat gallery."""
    ct, seg = eng.load_aligned(it.ct, it.seg)
    slices = eng.pick_slices(ct, seg, n_slices)
    title = f"{it.dataset}  {it.pid}  {it.stage}  (CADS Task-551)"
    eng.render(ct, seg, slices, colors, title, it.png)
    if gallery:
        os.makedirs(gallery, exist_ok=True)
        shutil.copy(it.png, os.path.join(gallery, f"{it.token}_cads551.png"))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="CADS Task-551 batch over multi-timepoint patients.")
    ap.add_argument("--processed", default=os.environ.get(
        "CT_PROCESSED", "/media/cbtil3/WhiteSD/CTProcessed"),
        help="root of processed datasets ($CT_PROCESSED)")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                    help="dataset folders to process (default: all known)")
    ap.add_argument("--min-timepoints", type=int, default=2,
                    help="keep patients with at least this many CT timepoints")
    ap.add_argument("--min-age", type=float, default=0.0,
                    help="skip CTs whose ct.nii.gz was modified within this many "
                         "seconds (avoid volumes still being written by the "
                         "converter); 0 disables the check")
    ap.add_argument("--n-slices", type=int, default=6,
                    help="number of axial slices per figure")
    ap.add_argument("--gpu", type=int, default=0,
                    help="single GPU index to pin (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    ap.add_argument("--gallery", default=os.environ.get("CT_GALLERY", "/home/cbtil3/hao/repo_outputs/Awesome-Medical-Datasets/Collections/Abdomen/Cancer-Progression/visualization"),
                    help="flat folder to also collect figures into "
                         "(dataset-prefixed names); '' to disable")
    ap.add_argument("--cads-python", default=os.environ.get(
        "CADS_PYTHON", "python3"), help="interpreter of the CADS env")
    ap.add_argument("--cads-repo", default=os.environ.get("CADS_REPO", ""),
                    help="cloned murong-xu/CADS checkout ($CADS_REPO)")
    ap.add_argument("--force", action="store_true",
                    help="re-segment even if a cached mask exists")
    ap.add_argument("--render-only", action="store_true",
                    help="skip inference; only (re)draw figures from cached masks")
    ap.add_argument("--rerender", action="store_true",
                    help="redraw figures even if the PNG is already newer than "
                         "its mask (default: skip up-to-date figures, so repeat "
                         "passes during a watch loop stay cheap)")
    ap.add_argument("--list", action="store_true",
                    help="print the selected worklist and exit")
    args = ap.parse_args()

    root = args.processed
    if not os.path.isdir(root):
        sys.exit(f"processed root not found: {root}")
    items, skipped_single = discover(root, args.datasets, args.min_timepoints,
                                     args.min_age)
    if not items:
        sys.exit("no patient meets the timepoint filter")

    n_pat = len({(it.dataset, it.pid) for it in items})
    eng.log(f"selected {n_pat} patients with >= {args.min_timepoints} "
            f"timepoints -> {len(items)} CT(s)  "
            f"(filtered out {skipped_single} single-CT patients)")
    if args.list:
        for it in items:
            mark = "cached" if os.path.exists(it.seg) else "to-run"
            eng.log(f"  [{mark}] {it.dataset}/{it.pid}/{it.stage}")
        return

    cads_repo = args.cads_repo or os.path.dirname(here)  # parent has none; require env
    # Inference (one shared predict call over all not-yet-cached CTs).
    if not args.render_only:
        to_run = items if args.force else [
            it for it in items if not os.path.exists(it.seg)]
        if to_run:
            out_dir = os.path.join(root, "_cads551_batch_out")
            done, failed = run_cads_batch(
                to_run, out_dir, args.cads_python, cads_repo, args.cpu, args.gpu)
            shutil.rmtree(out_dir, ignore_errors=True)
            eng.log(f"  segmented {len(done)} / {len(to_run)} "
                    f"({len(failed)} failed)")
            for it in failed:
                eng.log(f"    [FAIL] {it.dataset}/{it.pid}/{it.stage}")
        else:
            eng.log("  all masks already cached; nothing to segment")

    # Render every CT that has a mask. Figures whose PNG is already newer than
    # the mask are skipped unless --rerender, so repeat passes (watch loop) do
    # not redraw the whole cohort every time.
    colors = eng.label_colors()
    rendered = missing = fig_cached = 0
    for it in items:
        if not os.path.exists(it.seg):
            missing += 1
            eng.log(f"  [no mask] {it.dataset}/{it.pid}/{it.stage}")
            continue
        if (not args.rerender and os.path.exists(it.png)
                and os.path.getmtime(it.png) >= os.path.getmtime(it.seg)):
            fig_cached += 1
            continue
        try:
            render_item(it, args.n_slices, colors, args.gallery or None)
            rendered += 1
        except Exception as e:  # one bad volume shouldn't sink the batch
            missing += 1
            eng.log(f"  [render-fail] {it.dataset}/{it.pid}/{it.stage}: {e}")
    eng.log(f"done: rendered {rendered} figure(s), {fig_cached} up-to-date, "
            f"{missing} missing/failed")
    if args.gallery:
        eng.log(f"gallery -> {args.gallery}")


if __name__ == "__main__":
    main()
