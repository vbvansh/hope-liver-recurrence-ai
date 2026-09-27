#!/usr/bin/env python3
"""SegVol tumor/cancer segmentation demo (shared, dataset-agnostic).

Counterpart to `cads_demo.py`. CADS Task-551 segments *organs*; SegVol is a
promptable 3-D foundation model (BAAI/SegVol) that segments a *lesion* from a
free-text prompt, so it is the one we use for the "general tumor/cancer"
deliverable. For ONE example patient it runs SegVol on the standardized baseline
CT with a tumor text prompt and renders a two-row figure: CT on top, CT + tumor
overlay on the bottom (img + seg) -- same layout, slice picking and gallery as
the CADS demo, so the two products look uniform.

Why text-only prompts: point/bbox prompts need a ground-truth mask to seed them,
which only HCC-TACE-Seg and WAW-TACE ship. Text prompting needs nothing but the
CT, so the SAME path runs on all 7 datasets. Each dataset gets an anatomically
appropriate default tumor prompt (see DATASET_PROMPTS); override with --prompt,
or use --generic to force the bare word "tumor" everywhere.

SegVol is NOT the pipeline's main conda env (it pins its own torch + transformers
+ monai). Install it separately and run THIS script with that interpreter:

  # one-time, in a fresh env
  pip install "torch" "transformers>=4.40" monai nibabel matplotlib einops
  export SEGVOL_PYTHON=/path/to/envs/segvol/bin/python
  # optional: pre-download the weights to a local dir and point at it
  export SEGVOL_MODEL=BAAI/SegVol          # or a local snapshot path

Then, per dataset (or via the thin wrapper in each dataset folder):

  $SEGVOL_PYTHON segvol_demo.py --dataset HCC-TACE-Seg              # liver tumor
  $SEGVOL_PYTHON segvol_demo.py --dataset CPTAC-CCRCC --patient C3L-00610
  $SEGVOL_PYTHON segvol_demo.py --dataset NLST --prompt "lung nodule" --cpu

Outputs (under <processed>/<dataset>/):
  _segvol/<pid>/<stage>_tumor.nii.gz        cached binary tumor mask (resumable)
  visualization/<pid>_<stage>_segvol.png    the img + tumor figure

The model-call block (load_model / segment_tumor) follows the BAAI/SegVol HF
model card. If you pin a different SegVol revision, that block is the only thing
to re-check; everything else is plain nibabel/matplotlib shared with cads_demo.
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import nibabel as nib

# Reuse the CADS engine's IO/slice/log helpers (identical for any mask).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cads_demo as eng  # eng.log, eng.load_aligned, eng.pick_slices

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Default HF repo (or a local snapshot dir). Override with --model / $SEGVOL_MODEL.
DEFAULT_MODEL = os.environ.get("SEGVOL_MODEL", "BAAI/SegVol")

# Per-dataset tumor prompt. SegVol responds best to an organ-qualified lesion
# name; --generic collapses all of these to the bare word "tumor".
DATASET_PROMPTS = {
    "CPTAC-CCRCC":  "kidney tumor",       # clear-cell renal cell carcinoma
    "CPTAC-PDA":    "pancreas tumor",      # pancreatic ductal adenocarcinoma
    "EAY131":       "tumor",               # NCI-MATCH: mixed solid tumors
    "HCC-TACE-Seg": "liver tumor",         # hepatocellular carcinoma
    "NLST":         "lung tumor",          # lung cancer screening
    "RIDER":        "lung tumor",          # lung test-retest
    "WAW-TACE":     "liver tumor",         # HCC, multi-phase
}
GENERIC_PROMPT = "tumor"

# Tumor overlay color (single foreground class) -> opaque red.
TUMOR_RGBA = (0.90, 0.10, 0.10, 1.0)

# --------------------------------------------------------------------------- #
# THE STANDARD label-free config: type-prompt + organ-confine.
#
# The deployable recipe (validated on the labelled liver datasets, see README):
#   tumor-type text prompt  +  confine to the lesion's organ via TotalSegmentator
#   +  largest-connected-component  +  a low sigmoid threshold.
# It needs NO tumor label at inference -- only the tumor TYPE (-> prompt) and the
# ORGAN (-> TotalSeg mask), which you know per dataset. TotalSeg itself is
# label-free and already cached at Processed/<DS>/_totalseg/<pid>/<stage>_seg.nii.gz.
# --------------------------------------------------------------------------- #

# TotalSegmentator v2 'total' task label ids, by organ (verified on this cohort).
TOTALSEG_ORGAN = {
    "liver":    [5],
    "kidney":   [2, 3],
    "pancreas": [7],
    "lung":     [10, 11, 12, 13, 14],
}

# Per-dataset standard. The two liver rows are Dice-validated (prompt
# 'hepatocellular carcinoma', threshold 0.20 won the ablation); the others use
# the clinically-correct tumor-type name + its organ and the same low-threshold
# prior -- re-tune the threshold per domain once a few labels exist (segvol_tune).
STANDARD = {
    "HCC-TACE-Seg": dict(prompt="hepatocellular carcinoma",            organ="liver",    threshold=0.20),
    "WAW-TACE":     dict(prompt="hepatocellular carcinoma",            organ="liver",    threshold=0.20),
    "CPTAC-CCRCC":  dict(prompt="renal cell carcinoma",               organ="kidney",   threshold=0.25),
    "CPTAC-PDA":    dict(prompt="pancreatic ductal adenocarcinoma",   organ="pancreas", threshold=0.25),
    "NLST":         dict(prompt="lung tumor",                          organ="lung",     threshold=0.25),
    "RIDER":        dict(prompt="lung tumor",                          organ="lung",     threshold=0.25),
    "EAY131":       dict(prompt="tumor",                               organ=None,       threshold=0.25),
}
DEFAULT_STANDARD = dict(prompt="tumor", organ=None, threshold=0.25)
DILATE = 5  # organ-mask dilation (voxels) so a peri-organ lesion isn't clipped


def prompt_for(dataset, override=None, generic=False):
    """Resolve the text prompt for a dataset (override > generic > default)."""
    if override:
        return override
    if generic:
        return GENERIC_PROMPT
    return DATASET_PROMPTS.get(dataset, GENERIC_PROMPT)


def standard_config(dataset, prompt=None, generic=False):
    """The STANDARD (prompt, organ, threshold) for a dataset, with overrides."""
    cfg = dict(STANDARD.get(dataset, DEFAULT_STANDARD))
    if prompt:
        cfg["prompt"] = prompt
    if generic:
        cfg["prompt"] = GENERIC_PROMPT
    return cfg


def largest_cc(mask):
    """Keep only the largest connected component of a binary mask."""
    from scipy import ndimage
    if mask.sum() == 0:
        return mask
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask.astype(np.uint8)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return (lab == (1 + int(np.argmax(sizes)))).astype(np.uint8)


def load_organ_mask(totalseg_path, ref_img, organ, dilate=DILATE):
    """Dilated organ envelope on ref_img's grid from a TotalSeg mask.

    Returns a uint8 mask, or None if the organ is unset / the seg is missing /
    the organ is absent -- callers then skip confinement for that volume.
    """
    if organ is None or not totalseg_path or not os.path.exists(totalseg_path):
        return None
    from scipy import ndimage
    seg_img = nib.load(totalseg_path)
    if seg_img.shape != ref_img.shape or not np.allclose(
            seg_img.affine, ref_img.affine):
        from nibabel.processing import resample_from_to
        seg_img = resample_from_to(seg_img, ref_img, order=0)
    seg = np.asarray(seg_img.dataobj)
    m = np.isin(seg, TOTALSEG_ORGAN.get(organ, []))
    if m.sum() == 0:
        return None
    if dilate > 0:
        m = ndimage.binary_dilation(m, iterations=dilate)
    return m.astype(np.uint8)


# --------------------------------------------------------------------------- #
# SegVol model (BAAI/SegVol HF model card). Heavy import: only torch/transformers
# here so the rest of the module imports fine in the plain pipeline env too.
# --------------------------------------------------------------------------- #
def load_model(model_id, device, use_cpu):
    """Load SegVol once and return (model, device). Reuse across many CTs.

    Applies two compatibility patches for MONAI >= 1.3 (SegVol was authored
    against MONAI < 1.0, both verified live against the BAAI/SegVol snapshot):
      * its `LoadImage()` now returns only a tensor (image_only defaults True),
        but `preprocess_ct_gt` unpacks `(img, meta)` -> restore image_only=False.
      * `CropForegroundd` no longer writes `foreground_{start,end}_coord` into the
        dict by default, yet `save_preds` needs those coords -> rebuild the test
        transform with the coord keys (reusing the model's own DimTranspose /
        MinMaxNormalization so normalization is byte-identical to training).
    """
    import importlib
    from transformers import AutoModel, AutoTokenizer
    from monai import transforms

    dev = "cpu" if use_cpu else device
    eng.log(f"  loading SegVol '{model_id}' on {dev} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                      test_mode=True)
    # The CLIP text encoder needs its tokenizer wired up (per the model card).
    model.model.text_encoder.tokenizer = tokenizer

    proc = model.processor
    proc.img_loader = transforms.LoadImage(image_only=False)
    rm = importlib.import_module(type(proc).__module__)  # SegVol's remote module
    proc.transform4test = transforms.Compose([
        rm.DimTranspose(keys=["image", "label"]),
        rm.MinMaxNormalization(),
        transforms.CropForegroundd(
            keys=["image", "label"], source_key="image",
            start_coord_key="foreground_start_coord",
            end_coord_key="foreground_end_coord"),
        transforms.ToTensord(keys=["image", "label"]),
    ])

    model.eval()
    model.to(dev)
    eng.log("  SegVol ready")
    return model, dev


def predict_prob_grid(model, device, ct_path, prompt, use_zoom=True):
    """Text-prompted SegVol -> tumor probability volume on the ORIGINAL CT grid.

    Mirrors SegVol's own save_preds restore (DimTranspose axis swap + foreground
    paste-back) but keeps the sigmoid probability instead of binarizing at 0.5,
    so the caller can threshold/confine/clean afterwards. Returns (prob, ct_img).
    """
    import torch

    ct_img = nib.load(ct_path)
    # preprocess_ct_gt wants a GT path; for pure inference we hand it an all-zero
    # label of matching geometry so the foreground crop / intensity normalization
    # run exactly as in training. The label is never used -- text drives the model.
    with tempfile.TemporaryDirectory(prefix="segvol_") as tmp:
        gt_path = os.path.join(tmp, "zeros_gt.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros(ct_img.shape, np.uint8),
                                 ct_img.affine, ct_img.header), gt_path)
        ct_npy, gt_npy = model.processor.preprocess_ct_gt(
            ct_path, gt_path, category=[prompt])
        data_item = model.processor.zoom_transform(ct_npy, gt_npy)

    image = data_item["image"].unsqueeze(0).to(device)
    zoom_out_image = data_item["zoom_out_image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model.forward_test(
            image=image, zoomed_image=zoom_out_image,
            point_prompt_group=None,     # no spatial prompt -> text-only
            bbox_prompt_group=None,
            text_prompt=[prompt],
            use_zoom=use_zoom,           # zoom-in-zoom-out refinement
        )
    prob_crop = torch.sigmoid(logits[0][0]).transpose(-1, -3)
    sc = list(data_item["foreground_start_coord"])
    ec = list(data_item["foreground_end_coord"])
    sc[-1], sc[-3] = sc[-3], sc[-1]
    ec[-1], ec[-3] = ec[-3], ec[-1]
    prob = np.zeros(ct_img.shape, dtype=np.float32)
    prob[sc[0]:ec[0], sc[1]:ec[1], sc[2]:ec[2]] = prob_crop.numpy()
    return prob, ct_img


def segment_tumor(model, device, ct_path, prompt, out_path,
                  organ_mask=None, threshold=0.5, keep_largest=False):
    """SegVol tumor mask -> out_path on the original CT grid; returns the mask.

    The STANDARD label-free pipeline passes organ_mask (TotalSeg envelope),
    threshold (per-dataset, ~0.2-0.25) and keep_largest=True. With the defaults
    (threshold 0.5, no confine) it reproduces the plain text-only behavior.
    """
    prob, ct_img = predict_prob_grid(model, device, ct_path, prompt)
    mask = (prob > threshold).astype(np.uint8)
    if organ_mask is not None:
        mask = (mask & organ_mask.astype(np.uint8)).astype(np.uint8)
    if keep_largest:
        mask = largest_cc(mask)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(nib.Nifti1Image(mask, ct_img.affine, ct_img.header), out_path)
    eng.log(f"  wrote {out_path}  ({int(mask.sum())} tumor voxels)")
    return mask


# --------------------------------------------------------------------------- #
# Figure: top row CT, bottom row CT + tumor overlay. Mirrors cads_demo.render
# but for a single binary class.
# --------------------------------------------------------------------------- #
def render(ct, seg, slices, title, prompt, out_png):
    ct_disp = np.clip(ct, eng.CT_WMIN, eng.CT_WMAX)
    overlay = np.zeros((*seg.shape, 4), dtype=np.float32)
    overlay[seg > 0] = TUMOR_RGBA

    n = len(slices)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
    if n == 1:
        axes = axes.reshape(2, 1)
    for col, z in enumerate(slices):
        base = np.rot90(ct_disp[:, :, z])
        ov = np.rot90(overlay[:, :, z])
        for row in (0, 1):
            ax = axes[row, col]
            ax.imshow(base, cmap="gray", vmin=eng.CT_WMIN, vmax=eng.CT_WMAX,
                      interpolation="nearest", aspect="equal")
            if row == 1:
                ax.imshow(ov, interpolation="nearest", aspect="equal", alpha=0.55)
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, col].set_title(f"z={z}", fontsize=8)
    axes[0, 0].set_ylabel("CT", fontsize=10)
    axes[1, 0].set_ylabel("CT + SegVol", fontsize=10)

    n_vox = int((seg > 0).sum())
    handle = [Patch(facecolor=TUMOR_RGBA, edgecolor="none",
                    label=f'"{prompt}"  ({n_vox} vox)')]
    fig.legend(handles=handle, loc="lower center", fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    eng.log(f"  figure -> {out_png}  ({n_vox} tumor voxels)")


def main():
    ap = argparse.ArgumentParser(
        description="SegVol tumor demo: one example, multi-slice CT+tumor figure.")
    ap.add_argument("--dataset", required=True,
                    help="dataset folder under <processed> (e.g. HCC-TACE-Seg)")
    ap.add_argument("--patient", help="patient id; default = first with a baseline")
    ap.add_argument("--stage", default="000_baseline",
                    help="timepoint/phase dir holding ct.nii.gz (default baseline)")
    ap.add_argument("--processed", default=os.environ.get(
        "CT_PROCESSED", "/media/cbtil3/WhiteSD/CTProcessed"),
        help="root of processed datasets ($CT_PROCESSED)")
    ap.add_argument("--prompt", help="override the per-dataset tumor text prompt")
    ap.add_argument("--generic", action="store_true",
                    help='use the bare prompt "tumor" instead of the dataset default')
    ap.add_argument("--n-slices", type=int, default=6,
                    help="number of axial slices in the figure")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="SegVol HF id or local snapshot path ($SEGVOL_MODEL)")
    ap.add_argument("--gpu", type=int, default=0,
                    help="GPU index to use (ignored with --cpu)")
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    ap.add_argument("--force", action="store_true",
                    help="re-segment even if a cached mask exists")
    args = ap.parse_args()

    root = os.path.join(args.processed, args.dataset)
    if not os.path.isdir(root):
        sys.exit(f"dataset not found: {root}")
    pid, pat_dir = eng.find_patient_dir(root, args.patient)
    ct_path = os.path.join(pat_dir, args.stage, "ct.nii.gz")
    if not os.path.exists(ct_path):
        sys.exit(f"CT not found: {ct_path}")
    prompt = prompt_for(args.dataset, args.prompt, args.generic)
    eng.log(f"{args.dataset}: example patient {pid}  stage {args.stage}  "
            f"prompt='{prompt}'")

    seg_path = os.path.join(root, "_segvol", pid, f"{args.stage}_tumor.nii.gz")
    if args.force or not os.path.exists(seg_path):
        device = f"cuda:{args.gpu}" if not args.cpu else "cpu"
        if not args.cpu:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
            device = "cuda"
        model, device = load_model(args.model, device, args.cpu)
        segment_tumor(model, device, ct_path, prompt, seg_path)
    else:
        eng.log(f"  reuse {seg_path}")

    ct, seg = eng.load_aligned(ct_path, seg_path)
    slices = eng.pick_slices(ct, seg, args.n_slices)
    out_png = os.path.join(root, "visualization",
                           f"{pid}_{args.stage}_segvol.png")
    render(ct, seg, slices,
           f"{args.dataset}  {pid}  {args.stage}  (SegVol: {prompt})",
           prompt, out_png)
    eng.log("done")


if __name__ == "__main__":
    main()
