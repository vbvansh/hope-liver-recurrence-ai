#!/usr/bin/env python3
"""Re-segment SegVol masks that came out empty/tiny, relaxing until non-empty.

For each target CT we compute the SegVol probability volume ONCE, then walk a
relaxation ladder and keep the first setting that yields a non-empty mask:
  1. calibrated threshold, organ-confined           (the original recipe)
  2. progressively lower thresholds, organ-confined  (0.10, 0.05, 0.02, 0.01)
  3. same thresholds WITHOUT organ-confine
  4. last resort: the single strongest blob -- voxels >= 0.5*max(prob),
     largest connected component (guaranteed non-empty as long as prob has
     any signal).
The winning mask overwrites the empty one and its QC figure is redrawn.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import nibabel as nib
import cads_demo as eng
import segvol_demo as sv

LADDER_THRESHOLDS = [0.10, 0.05, 0.02, 0.01]


def cfg_for(ds, calib):
    cfg = sv.standard_config(ds, None, False)
    if calib and ds in calib:
        cfg["prompt"] = calib[ds].get("prompt", cfg["prompt"])
        cfg["threshold"] = calib[ds].get("threshold", cfg["threshold"])
    return cfg


def fill_one(model, device, root, relpath, calib, n_slices, gallery):
    # relpath = "<ds>/_segvol/<pid>/<stage>_tumor.nii.gz"
    parts = relpath.split("/")
    ds, pid = parts[0], parts[2]
    stage = parts[3].replace("_tumor.nii.gz", "")
    ct = os.path.join(root, ds, pid, stage, "ct.nii.gz")
    seg = os.path.join(root, ds, "_segvol", pid, f"{stage}_tumor.nii.gz")
    totalseg = os.path.join(root, ds, "_totalseg", pid, f"{stage}_seg.nii.gz")
    if not os.path.exists(ct):
        eng.log(f"  [skip] CT missing: {ct}")
        return None

    cfg = cfg_for(ds, calib)
    prompt, organ, base_t = cfg["prompt"], cfg["organ"], cfg["threshold"]
    prob, ct_img = sv.predict_prob_grid(model, device, ct, prompt)

    organ_mask = None
    if organ:
        organ_mask = sv.load_organ_mask(totalseg, ct_img, organ)

    # build the ladder: (threshold, use_confine)
    rungs = [(base_t, True)] + [(t, True) for t in LADDER_THRESHOLDS if t < base_t]
    rungs += [(base_t, False)] + [(t, False) for t in LADDER_THRESHOLDS if t < base_t]

    chosen = None
    for thr, confine in rungs:
        m = (prob > thr).astype(np.uint8)
        if confine and organ_mask is not None:
            m = (m & organ_mask.astype(np.uint8)).astype(np.uint8)
        m = sv.largest_cc(m)
        v = int(m.sum())
        if v > 0:
            chosen = (m, thr, confine, v)
            break

    if chosen is None:
        # last resort: strongest blob relative to this volume's own max prob
        pmax = float(prob.max())
        m = sv.largest_cc((prob >= 0.5 * pmax).astype(np.uint8)) if pmax > 0 \
            else np.zeros(prob.shape, np.uint8)
        chosen = (m, f"0.5*max({pmax:.3f})", False, int(m.sum()))

    mask, thr, confine, v = chosen
    nib.save(nib.Nifti1Image(mask, ct_img.affine, ct_img.header), seg)
    eng.log(f"  {ds}/{pid}/{stage}: thr={thr} confine={confine} -> {v} voxels")

    # redraw figure
    png = os.path.join(root, ds, "visualization", f"{pid}_{stage}_segvol.png")
    try:
        ctv, segv = eng.load_aligned(ct, seg)
        slices = eng.pick_slices(ctv, segv, n_slices)
        title = f"{ds}  {pid}  {stage}  (SegVol refilled: {prompt})"
        sv.render(ctv, segv, slices, title, prompt, png)
        if gallery:
            import shutil
            os.makedirs(gallery, exist_ok=True)
            shutil.copy(png, os.path.join(gallery, f"{ds}_{pid}_{stage}_segvol.png"))
    except Exception as e:
        eng.log(f"    [render-fail] {e}")
    return v


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=os.environ.get(
        "CT_PROCESSED", "/media/cbtil3/WhiteSD/CTProcessed"))
    ap.add_argument("--report", required=True,
                    help="segvol_empty_report.json from the scan")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--n-slices", type=int, default=6)
    ap.add_argument("--gallery", default=os.environ.get("CT_GALLERY", "/home/cbtil3/hao/repo_outputs/Awesome-Medical-Datasets/Collections/Abdomen/Cancer-Progression/visualization"))
    ap.add_argument("--include-tiny", action="store_true", default=True,
                    help="also re-do the <10-voxel 'tiny' masks (default on)")
    args = ap.parse_args()

    root = args.processed
    rep = json.load(open(args.report))
    targets = list(rep.get("empty", []))
    if args.include_tiny:
        targets += [p for p, _ in rep.get("tiny", [])]
    targets = sorted(set(targets))
    eng.log(f"refilling {len(targets)} mask(s)")

    calib = None
    cpath = os.path.join(root, "_segvol_calib", "recommended.json")
    if os.path.exists(cpath):
        calib = json.load(open(cpath))

    device = "cpu" if args.cpu else "cuda"
    if not args.cpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    model, device = sv.load_model(sv.DEFAULT_MODEL, device, args.cpu)

    still_empty = []
    for i, rel in enumerate(targets, 1):
        eng.log(f"[{i}/{len(targets)}] {rel}")
        try:
            v = fill_one(model, device, root, rel, calib, args.n_slices,
                         args.gallery or None)
            if v is not None and v == 0:
                still_empty.append(rel)
        except Exception as e:
            eng.log(f"    [FAIL] {e}")
            still_empty.append(rel)

    eng.log(f"done. still empty: {len(still_empty)}")
    for p in still_empty:
        eng.log(f"  STILL EMPTY: {p}")


if __name__ == "__main__":
    main()
