#!/usr/bin/env python3
"""CADS-551 segmentation demo (shared, dataset-agnostic).

Deliverable: for ONE example patient of a dataset, run the CADS Task-551 model
(17 abdominal organs / lung lobes / great vessels) on the standardized baseline
CT and render a single multi-slice figure showing the CT and the CT+segmentation
overlay side by side (img + seg).

It mirrors `segment.py`: it shells out to the CADS inference CLI (TotalSegmentator
is to `segment.py` what CADS is here), caches the multilabel mask so re-runs that
only tweak the figure are cheap, then draws the overlay with matplotlib.

CADS Task-551 labels (from the model labelmap):
  1 Spleen            7 Aorta                  13 Upper lobe lung L
  2 Kidney R          8 Inferior vena cava     14 Lower lobe lung L
  3 Kidney L          9 Portal & splenic vein  15 Upper lobe lung R
  4 Gallbladder      10 Pancreas               16 Middle lobe lung R
  5 Liver            11 Adrenal gland R        17 Lower lobe lung R
  6 Stomach          12 Adrenal gland L

CADS is not the pipeline's main conda env; install it separately (its README
pins torch 2.5.1 + nnunetv2) and point this script at that interpreter and repo:

  export CADS_PYTHON=/path/to/envs/CADS_env/bin/python
  export CADS_REPO=/path/to/CADS          # the cloned murong-xu/CADS checkout

Then, per dataset (or via the thin wrapper in each dataset folder):

  python3 cads_demo.py --dataset EAY131                 # auto-pick first patient
  python3 cads_demo.py --dataset HCC-TACE-Seg --patient HCC_001 --n-slices 6
  python3 cads_demo.py --dataset NLST --cpu             # force CPU inference

Outputs (under <processed>/<dataset>/):
  _cads551/<pid>/<stage>_seg.nii.gz        cached multilabel mask (resumable)
  visualization/<pid>_<stage>_cads551.png  the img+seg figure
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# Task-551 label id -> structure name (id 0 is background, omitted).
CADS551_LABELS = {
    1: "Spleen", 2: "Kidney R", 3: "Kidney L", 4: "Gallbladder", 5: "Liver",
    6: "Stomach", 7: "Aorta", 8: "Inferior vena cava",
    9: "Portal & splenic vein", 10: "Pancreas", 11: "Adrenal gland R",
    12: "Adrenal gland L", 13: "Upper lobe lung L", 14: "Lower lobe lung L",
    15: "Upper lobe lung R", 16: "Middle lobe lung R", 17: "Lower lobe lung R",
}
N_LABELS = max(CADS551_LABELS)  # 17

# Abdomen soft-tissue display window (HU): level 40, width 400.
CT_WMIN, CT_WMAX = -160.0, 240.0


def log(msg):
    print(msg, flush=True)


def label_colors():
    """A distinct RGBA per label id (index 0 = background = transparent)."""
    base = plt.get_cmap("tab20").colors  # 20 distinct colors
    cols = [(0, 0, 0, 0.0)]              # background -> transparent
    for i in range(N_LABELS):
        r, g, b = base[i % len(base)]
        cols.append((r, g, b, 1.0))
    return cols


def find_patient_dir(root, patient):
    """Resolve the example patient dir; auto-pick the first valid one if None."""
    if patient:
        pd = os.path.join(root, patient)
        if not os.path.isdir(pd):
            sys.exit(f"patient dir not found: {pd}")
        return patient, pd
    for d in sorted(os.listdir(root)):
        if d.startswith((".", "_")) or d == "visualization":
            continue
        pd = os.path.join(root, d)
        if os.path.isdir(pd) and os.path.exists(
                os.path.join(pd, "000_baseline", "ct.nii.gz")):
            return d, pd
    sys.exit(f"no patient with a baseline ct.nii.gz under {root}")


def run_cads(ct_path, out_seg_path, example_id, cads_python, cads_repo,
             use_cpu):
    """CADS Task-551 -> multilabel NIfTI at out_seg_path (skip if present).

    The CLI walks an input *directory* and names outputs after each file's
    basename, so we stage the CT into a temp dir under a unique name to avoid
    the universal `ct.nii.gz` colliding across patients.
    """
    if os.path.exists(out_seg_path):
        log(f"  reuse {out_seg_path}")
        return
    predict = os.path.join(cads_repo, "cads", "scripts", "predict_images.py")
    if not os.path.exists(predict):
        sys.exit(f"CADS inference script not found: {predict}\n"
                 f"set --cads-repo / $CADS_REPO to the cloned murong-xu/CADS")
    with tempfile.TemporaryDirectory(prefix="cads551_") as tmp:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(in_dir)
        staged = os.path.join(in_dir, f"{example_id}.nii.gz")
        shutil.copy(ct_path, staged)
        cmd = [cads_python, predict, "-in", in_dir, "-out", out_dir,
               "-task", "551"]
        if use_cpu:
            cmd.append("--cpu")
        log("  $ " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        produced = os.path.join(out_dir, example_id,
                                f"{example_id}_part_551.nii.gz")
        if not os.path.exists(produced):
            sys.exit(f"CADS produced no mask at {produced}")
        os.makedirs(os.path.dirname(out_seg_path), exist_ok=True)
        shutil.copy(produced, out_seg_path)
        log(f"  wrote {out_seg_path}")


def load_aligned(ct_path, seg_path):
    """Return (ct_array, seg_array) on the CT grid; nearest-resample seg if needed."""
    ct_img = nib.load(ct_path)
    seg_img = nib.load(seg_path)
    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    if seg_img.shape != ct_img.shape or not np.allclose(
            seg_img.affine, ct_img.affine):
        seg_img = resample_from_to(seg_img, ct_img, order=0)  # nearest
    seg = np.asarray(seg_img.dataobj).astype(np.int16)
    return ct, seg


def pick_slices(ct, seg, n):
    """n axial (z) slice indices evenly spanning the labeled extent.

    Only slices that hold BOTH a mask label and real CT content count: CADS
    resamples its mask back onto the original grid and can smear labels by one
    slice onto blank boundary padding (uniform air), which would otherwise
    render as a colored overlay floating on an empty CT. Requiring CT body
    voxels (HU > -500) excludes those padding slices. Falls back to the middle
    of the volume when nothing qualifies.
    """
    nz = seg.shape[2]
    has_seg = seg.sum(axis=(0, 1)) > 0
    has_body = (ct > -500).sum(axis=(0, 1)) > 100  # >100 vox of non-air tissue
    fg = np.where(has_seg & has_body)[0]
    if not fg.size:                       # no overlap: fall back to seg alone
        fg = np.where(has_seg)[0]
    if fg.size:
        lo, hi = int(fg.min()), int(fg.max())
    else:
        lo, hi = nz // 4, 3 * nz // 4
    if hi <= lo:
        lo, hi = max(0, nz // 2 - 1), min(nz - 1, nz // 2 + 1)
    return [int(round(z)) for z in np.linspace(lo, hi, n)]


def render(ct, seg, slices, colors, title, out_png):
    """Two-row figure: top = CT, bottom = CT + Task-551 overlay."""
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, N_LABELS + 1, 1.0), cmap.N)
    ct_disp = np.clip(ct, CT_WMIN, CT_WMAX)

    n = len(slices)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
    if n == 1:
        axes = axes.reshape(2, 1)
    for col, z in enumerate(slices):
        base = np.rot90(ct_disp[:, :, z])
        msk = np.rot90(seg[:, :, z])
        for row in (0, 1):
            ax = axes[row, col]
            ax.imshow(base, cmap="gray", vmin=CT_WMIN, vmax=CT_WMAX,
                      interpolation="nearest", aspect="equal")
            if row == 1:
                ax.imshow(np.ma.masked_where(msk == 0, msk), cmap=cmap,
                          norm=norm, alpha=0.55, interpolation="nearest",
                          aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, col].set_title(f"z={z}", fontsize=8)
    axes[0, 0].set_ylabel("CT", fontsize=10)
    axes[1, 0].set_ylabel("CT + CADS-551", fontsize=10)

    present = sorted(int(v) for v in np.unique(seg) if v != 0)
    handles = [Patch(facecolor=colors[v], edgecolor="none",
                     label=f"{v} {CADS551_LABELS.get(v, v)}") for v in present]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
                   fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"  figure -> {out_png}  ({len(present)} structures present)")


def main():
    ap = argparse.ArgumentParser(
        description="CADS Task-551 demo: one example, multi-slice CT+seg figure.")
    ap.add_argument("--dataset", required=True,
                    help="dataset folder under <processed> (e.g. EAY131, NLST)")
    ap.add_argument("--patient", help="patient id; default = first with a baseline")
    ap.add_argument("--stage", default="000_baseline",
                    help="timepoint/phase dir holding ct.nii.gz (default baseline)")
    ap.add_argument("--processed", default=os.environ.get(
        "CT_PROCESSED", "/media/cbtil3/WhiteSD/CTProcessed"),
        help="root of processed datasets ($CT_PROCESSED)")
    ap.add_argument("--n-slices", type=int, default=6,
                    help="number of axial slices in the figure")
    ap.add_argument("--cads-python", default=os.environ.get(
        "CADS_PYTHON", "python3"), help="interpreter of the CADS env ($CADS_PYTHON)")
    ap.add_argument("--cads-repo", default=os.environ.get("CADS_REPO", ""),
                    help="cloned murong-xu/CADS checkout ($CADS_REPO)")
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    args = ap.parse_args()

    root = os.path.join(args.processed, args.dataset)
    if not os.path.isdir(root):
        sys.exit(f"dataset not found: {root}")
    pid, pat_dir = find_patient_dir(root, args.patient)
    ct_path = os.path.join(pat_dir, args.stage, "ct.nii.gz")
    if not os.path.exists(ct_path):
        sys.exit(f"CT not found: {ct_path}")
    log(f"{args.dataset}: example patient {pid}  stage {args.stage}")

    example_id = f"{args.dataset}_{pid}_{args.stage}"
    seg_path = os.path.join(root, "_cads551", pid, f"{args.stage}_seg.nii.gz")
    run_cads(ct_path, seg_path, example_id, args.cads_python,
             args.cads_repo or os.path.dirname(os.path.abspath(__file__)),
             args.cpu)

    ct, seg = load_aligned(ct_path, seg_path)
    slices = pick_slices(ct, seg, args.n_slices)
    out_png = os.path.join(root, "visualization",
                           f"{pid}_{args.stage}_cads551.png")
    render(ct, seg, slices, label_colors(),
           f"{args.dataset}  {pid}  {args.stage}  (CADS Task-551)", out_png)
    log("done")


if __name__ == "__main__":
    main()
