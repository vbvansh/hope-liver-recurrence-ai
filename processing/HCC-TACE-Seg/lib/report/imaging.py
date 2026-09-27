"""Extract a few key 2D CT slices per timepoint to feed the VLM (a 2D model).

We pass co-registered PRIOR | CURRENT | DIFFERENCE axial slices at the most
informative tumor plane, windowed to a fixed HU range (soft-tissue or lung) so
brightness is identical across timepoints WITHOUT per-slice rescaling. 3D burden
comes from the lesion-mask numbers (measurements.py); these slices give the VLM
the morphology it needs (new lesions, organ invasion, necrosis, lymphadenopathy).
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import nibabel as nib

# Fixed CT display windows (HU): (level, width) -> [level-width/2, level+width/2].
WINDOWS = {
    "soft": (40.0, 400.0),     # abdomen soft-tissue (liver/kidney/pancreas)
    "lung": (-600.0, 1500.0),  # lung parenchyma (NLST / RIDER)
}


def window_hu(name: str) -> Tuple[float, float]:
    lvl, wid = WINDOWS.get(name, WINDOWS["soft"])
    return lvl - wid / 2.0, lvl + wid / 2.0


def _ct_file(stage_dir: Path) -> Optional[Path]:
    for name in ("ct.nii.gz", "ct_labelguided.nii.gz", "ct_rigid.nii.gz"):
        p = stage_dir / name
        if p.exists():
            return p
    return None


def _to_uint8(slc: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.clip((slc - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def _body_z(vol: np.ndarray) -> int:
    """Fallback slice: the axial plane with the most body tissue (HU > -500)."""
    counts = (vol > -500).sum(axis=(0, 1))
    return int(np.argmax(counts)) if counts.max() else vol.shape[2] // 2


def tumor_z(tmask_path: Optional[Path], organ_path: Optional[Path],
            fallback_vol: Optional[np.ndarray] = None):
    """Axial index of the MOST PROMINENT TUMOR plane (what the 2D VLM must read).

    Priority: lesion-mask area > organ-mask area > body-area max. Volumes are
    co-registered, so the same z is the same plane across timepoints. Returns
    (z, source) where source in {tumor, organ, body}.
    """
    if tmask_path is not None and tmask_path.exists():
        prof = (np.asarray(nib.load(str(tmask_path)).dataobj) > 0).sum(axis=(0, 1))
        if prof.max() > 0:
            return int(np.argmax(prof)), "tumor"
    if organ_path is not None and organ_path.exists():
        prof = (np.asarray(nib.load(str(organ_path)).dataobj) > 0).sum(axis=(0, 1))
        if prof.max() > 0:
            return int(np.argmax(prof)), "organ"
    if fallback_vol is not None:
        return _body_z(fallback_vol), "body"
    return None, None


def comparison_slices(cur_dir: Path, prior_dir: Optional[Path],
                      tmask_path: Optional[Path] = None,
                      organ_path: Optional[Path] = None,
                      window: str = "soft",
                      save_dir: Optional[Path] = None):
    """Co-registered PRIOR | CURRENT | DIFFERENCE CT slices at the MOST PROMINENT
    TUMOR plane, so the 2D VLM actually reads the lesion (not a random slice).

    Slice z is chosen from the lesion mask (max tumor area) when available, else
    the organ mask, else the largest body cross-section. The chosen plane is
    reported in each label (e.g. 'ct:current@z75[tumor]') so the read is auditable.
    Returns list of (label, PIL.Image).
    """
    try:
        from PIL import Image
    except Exception:
        return []
    cur_p = _ct_file(cur_dir)
    if cur_p is None:
        return []
    cur = np.asarray(nib.load(str(cur_p)).get_fdata(), dtype=np.float32)
    z, src = tumor_z(tmask_path, organ_path, fallback_vol=cur)
    if z is None:
        z, src = cur.shape[2] // 2, "mid"
    lo, hi = window_hu(window)
    tag = f"@z{z}[{src}]"
    out = []
    zc = z if z < cur.shape[2] else cur.shape[2] // 2
    cur_slc = cur[:, :, zc]
    out.append((f"ct:current{tag}",
                Image.fromarray(np.rot90(_to_uint8(cur_slc, lo, hi))).convert("RGB")))
    pri_p = _ct_file(prior_dir) if prior_dir else None
    if pri_p is not None:
        pri = np.asarray(nib.load(str(pri_p)).get_fdata(), dtype=np.float32)
        zz = z if z < pri.shape[2] else pri.shape[2] // 2
        pri_slc = pri[:, :, zz]
        out.append((f"ct:prior{tag}",
                    Image.fromarray(np.rot90(_to_uint8(pri_slc, lo, hi))).convert("RGB")))
        if pri_slc.shape == cur_slc.shape:
            diff = cur_slc - pri_slc
            a = np.clip(np.rot90(diff) / (np.abs(diff).max() + 1e-6) * 127 + 128,
                        0, 255).astype(np.uint8)
            out.append((f"ct:diff{tag}", Image.fromarray(a).convert("RGB")))
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        for lbl, im in out:
            im.save(str(save_dir / f"{cur_dir.name}_{lbl.replace(':', '_').replace('@', '_')}.png"))
    return out
