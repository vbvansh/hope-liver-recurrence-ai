#!/usr/bin/env python3
"""SegVol tumor/cancer batch: segment every CT of every patient, across datasets.

Counterpart to `cads_batch.py`. Where CADS shells out to an external nnU-Net CLI
once, SegVol runs in-process, so the win is the same but simpler: load the model
ONE time and stream every CT through it. Each dataset is driven with its own
tumor text prompt (segvol_demo.DATASET_PROMPTS), so one run produces a uniform
"general tumor/cancer" product over all 7 abdomen datasets.

It reuses the single-example engine `segvol_demo.py` (same prompts, mask cache,
figure layout) and is resumable: a CT whose mask already exists at
`<dataset>/_segvol/<pid>/<stage>_tumor.nii.gz` is skipped on re-runs.

Layout assumed (per dataset under <processed>):
  <dataset>/<patient>/<stage>/ct.nii.gz      e.g. 000_baseline, 001_followup_1

Outputs:
  <dataset>/_segvol/<pid>/<stage>_tumor.nii.gz        cached binary mask
  <dataset>/visualization/<pid>_<stage>_segvol.png    per-CT figure (data drive)
  <gallery>/<dataset>_<pid>_<stage>_segvol.png        flat gallery (in this repo)

Usage:
  export SEGVOL_PYTHON=/path/to/envs/segvol/bin/python
  $SEGVOL_PYTHON segvol_batch.py --gpu 0                  # all datasets, GPU 0
  $SEGVOL_PYTHON segvol_batch.py --datasets HCC-TACE-Seg WAW-TACE --gpu 1
  $SEGVOL_PYTHON segvol_batch.py --list                  # just print the worklist
  $SEGVOL_PYTHON segvol_batch.py --render-only           # re-draw from cached masks
  $SEGVOL_PYTHON segvol_batch.py --generic               # force prompt "tumor"
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cads_demo as eng        # shared log / load_aligned / pick_slices
import segvol_demo as sv       # SegVol model + prompts + render

DEFAULT_DATASETS = ["CPTAC-CCRCC", "CPTAC-PDA", "EAY131", "HCC-TACE-Seg",
                    "NLST", "RIDER", "WAW-TACE"]


class Item:
    """One CT to segment + render."""
    __slots__ = ("dataset", "pid", "stage", "prompt", "organ", "threshold",
                 "ct", "seg", "png", "totalseg")

    def __init__(self, root, dataset, pid, stage, cfg):
        self.dataset = dataset
        self.pid = pid
        self.stage = stage
        self.prompt = cfg["prompt"]
        self.organ = cfg["organ"]
        self.threshold = cfg["threshold"]
        self.ct = os.path.join(root, dataset, pid, stage, "ct.nii.gz")
        self.seg = os.path.join(root, dataset, "_segvol", pid,
                                f"{stage}_tumor.nii.gz")
        self.png = os.path.join(root, dataset, "visualization",
                                f"{pid}_{stage}_segvol.png")
        self.totalseg = os.path.join(root, dataset, "_totalseg", pid,
                                     f"{stage}_seg.nii.gz")


def _ct_stable(ct_path, min_age):
    """True if ct.nii.gz exists and wasn't modified within the last min_age
    seconds. Guards against handing inference a volume the converter is still
    writing (writes are NOT atomic). min_age<=0 disables the check."""
    if not os.path.exists(ct_path):
        return False
    if min_age <= 0:
        return True
    import time
    return (time.time() - os.path.getmtime(ct_path)) >= min_age


def discover(root, datasets, override, generic, min_tp, raw, calib=None,
             min_age=0.0):
    """Worklist of Items for every timepoint of every qualifying patient."""
    items, skipped = [], 0
    for ds in datasets:
        dpath = os.path.join(root, ds)
        if not os.path.isdir(dpath):
            eng.log(f"  [skip] dataset not found: {dpath}")
            continue
        if raw:   # plain text-only at 0.5, no organ-confine (legacy behavior)
            cfg = dict(prompt=sv.prompt_for(ds, override, generic),
                       organ=None, threshold=0.5)
        else:     # the STANDARD: type-prompt + organ-confine + low threshold
            cfg = sv.standard_config(ds, override, generic)
            # calibration overrides the prompt/threshold the samples picked
            if calib and ds in calib:
                cfg["prompt"] = calib[ds].get("prompt", cfg["prompt"])
                cfg["threshold"] = calib[ds].get("threshold", cfg["threshold"])
        for pid in sorted(os.listdir(dpath)):
            if pid.startswith((".", "_")) or pid == "visualization":
                continue
            pp = os.path.join(dpath, pid)
            if not os.path.isdir(pp):
                continue
            stages = [s for s in sorted(os.listdir(pp))
                      if not s.startswith((".", "_"))
                      and _ct_stable(os.path.join(pp, s, "ct.nii.gz"), min_age)]
            if len(stages) < min_tp:
                if stages:
                    skipped += 1
                continue
            for s in stages:
                items.append(Item(root, ds, pid, s, cfg))
    return items, skipped


def render_item(it, n_slices, gallery):
    """Draw the img + tumor figure for one Item; mirror into the flat gallery."""
    ct, seg = eng.load_aligned(it.ct, it.seg)
    slices = eng.pick_slices(ct, seg, n_slices)
    title = f"{it.dataset}  {it.pid}  {it.stage}  (SegVol: {it.prompt})"
    sv.render(ct, seg, slices, title, it.prompt, it.png)
    if gallery:
        os.makedirs(gallery, exist_ok=True)
        shutil.copy(it.png, os.path.join(
            gallery, f"{it.dataset}_{it.pid}_{it.stage}_segvol.png"))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="SegVol tumor batch over all datasets (model loads once).")
    ap.add_argument("--processed", default=os.environ.get(
        "CT_PROCESSED", "/media/cbtil3/WhiteSD/CTProcessed"),
        help="root of processed datasets ($CT_PROCESSED)")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                    help="dataset folders to process (default: all 7)")
    ap.add_argument("--prompt", help="override every dataset's tumor prompt")
    ap.add_argument("--generic", action="store_true",
                    help='use the bare prompt "tumor" for all datasets')
    ap.add_argument("--raw", action="store_true",
                    help="legacy text-only mode (threshold 0.5, no organ-confine, "
                         "no largest-CC); default is the STANDARD organ-confined recipe")
    ap.add_argument("--calibrated", action="store_true",
                    help="override prompt/threshold from _segvol_calib/recommended.json "
                         "(label-free calibration from segvol_calibrate.py)")
    ap.add_argument("--min-timepoints", type=int, default=1,
                    help="only patients with >= this many CT timepoints (default 1=all)")
    ap.add_argument("--min-age", type=float, default=0.0,
                    help="skip any ct.nii.gz modified within this many seconds "
                         "(treats it as still being written; 0=off). Use when "
                         "running concurrently with the DICOM->NIfTI converter.")
    ap.add_argument("--n-slices", type=int, default=6,
                    help="number of axial slices per figure")
    ap.add_argument("--model", default=sv.DEFAULT_MODEL,
                    help="SegVol HF id or local snapshot path ($SEGVOL_MODEL)")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index (ignored with --cpu)")
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    ap.add_argument("--gallery", default=os.environ.get("CT_GALLERY", "/home/cbtil3/hao/repo_outputs/Awesome-Medical-Datasets/Collections/Abdomen/Cancer-Progression/visualization"),
                    help="flat folder to also collect figures into; '' to disable")
    ap.add_argument("--force", action="store_true",
                    help="re-segment even if a cached mask exists")
    ap.add_argument("--render-only", action="store_true",
                    help="skip inference; only (re)draw figures from cached masks")
    ap.add_argument("--render-new-only", action="store_true",
                    help="only render CTs whose figure is missing or older than its "
                         "mask; skip figures already up to date (avoids redundant "
                         "re-rendering of the whole corpus every watcher pass)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of CTs to segment this run (0 = no cap)")
    ap.add_argument("--list", action="store_true",
                    help="print the selected worklist and exit")
    ap.add_argument("--shard", default="",
                    help="process a disjoint slice 'i/n' of the worklist (0-based i), "
                         "e.g. --shard 0/2 and --shard 1/2 on two GPUs (no races)")
    args = ap.parse_args()

    root = args.processed
    if not os.path.isdir(root):
        sys.exit(f"processed root not found: {root}")
    calib = None
    if args.calibrated:
        cpath = os.path.join(root, "_segvol_calib", "recommended.json")
        if os.path.exists(cpath):
            import json
            calib = json.load(open(cpath))
            eng.log(f"using calibrated configs from {cpath}")
        else:
            eng.log(f"[warn] --calibrated but {cpath} missing; using STANDARD")
    items, skipped = discover(root, args.datasets, args.prompt, args.generic,
                              args.min_timepoints, args.raw, calib, args.min_age)
    if not items:
        sys.exit("no CT found for the selected datasets")

    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        items = [it for k, it in enumerate(items) if k % n == i]
        eng.log(f"shard {i}/{n}: {len(items)} of the worklist on this process")

    n_pat = len({(it.dataset, it.pid) for it in items})
    mode = "RAW text-only" if args.raw else "STANDARD (type-prompt + organ-confine)"
    eng.log(f"mode: {mode}")
    eng.log(f"selected {n_pat} patients -> {len(items)} CT(s) across "
            f"{len(args.datasets)} dataset(s)"
            + (f"  (skipped {skipped} below --min-timepoints)" if skipped else ""))
    if args.list:
        for it in items:
            mark = "cached" if os.path.exists(it.seg) else "to-run"
            conf = it.organ or "none"
            eng.log(f"  [{mark}] {it.dataset}/{it.pid}/{it.stage}  "
                    f"'{it.prompt}'  organ={conf} t={it.threshold}")
        return

    # Inference: load SegVol once, stream every not-yet-cached CT through it.
    if not args.render_only:
        to_run = items if args.force else [
            it for it in items if not os.path.exists(it.seg)]
        if args.limit:
            to_run = to_run[:args.limit]
        if to_run:
            device = "cuda" if not args.cpu else "cpu"
            if not args.cpu:
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
            model, device = sv.load_model(args.model, device, args.cpu)
            done = failed = 0
            import nibabel as nib
            for i, it in enumerate(to_run, 1):
                conf = it.organ or "none"
                eng.log(f"[{i}/{len(to_run)}] {it.dataset}/{it.pid}/{it.stage} "
                        f"'{it.prompt}'  organ={conf} t={it.threshold}")
                try:
                    organ_mask = None
                    if it.organ:
                        organ_mask = sv.load_organ_mask(
                            it.totalseg, nib.load(it.ct), it.organ)
                        if organ_mask is None:
                            eng.log(f"    [warn] no {it.organ} TotalSeg mask; "
                                    f"running without organ-confine")
                    sv.segment_tumor(model, device, it.ct, it.prompt, it.seg,
                                     organ_mask=organ_mask, threshold=it.threshold,
                                     keep_largest=not args.raw)
                    done += 1
                except Exception as e:  # one bad volume shouldn't sink the batch
                    failed += 1
                    eng.log(f"    [FAIL] {e}")
            eng.log(f"  segmented {done} / {len(to_run)} ({failed} failed)")
        else:
            eng.log("  all masks already cached; nothing to segment")

    # Render every CT that has a mask.
    rendered = missing = uptodate = 0
    for it in items:
        if not os.path.exists(it.seg):
            missing += 1
            continue
        # --render-new-only: skip figures already current (png exists and is at
        # least as new as its mask). Re-segmented masks (newer than the png) and
        # brand-new ones still get (re)drawn; the unchanged bulk is skipped.
        if args.render_new_only and os.path.exists(it.png) and \
                os.path.getmtime(it.png) >= os.path.getmtime(it.seg):
            uptodate += 1
            continue
        try:
            render_item(it, args.n_slices, args.gallery or None)
            rendered += 1
        except Exception as e:
            missing += 1
            eng.log(f"  [render-fail] {it.dataset}/{it.pid}/{it.stage}: {e}")
    eng.log(f"done: rendered {rendered} figure(s), {missing} missing/failed"
            + (f", {uptodate} already current" if uptodate else ""))
    if args.gallery:
        eng.log(f"gallery -> {args.gallery}")


if __name__ == "__main__":
    main()
