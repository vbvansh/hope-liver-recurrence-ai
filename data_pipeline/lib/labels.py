"""Standard lesion labels per dataset (PLAN.md Step 2, rules in configs/label_maps.yaml).

Each function returns a list of Labels, one per phase that has lesions drawn on it (one for every dataset except
WAW-TACE, where tumours may be drawn on different phases). Arrays follow nibabel's voxel order (x, y, z) of the
image they sit on, on its native grid: no resampling here. lesion_instances numbers lesions (classes 1-2); where
lesions overlap, the smaller one keeps the overlapping voxels so that every lesion stays visible. Where lesions can
overlap (WAW-TACE, TCIA-CRLM), each lesion dict also carries "idx", the voxel indices of its own full outline.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

import components as C
import mct_reader

MALIGNANT, HAEMANGIOMA, IGNORE = 1, 2, 3
MCT_CLASS = {"HCC": MALIGNANT, "ICC": MALIGNANT, "CRLM": MALIGNANT, "BCLM": MALIGNANT, "HH": HAEMANGIOMA}
MCT_MIN_VOXELS = 50  # authors' VOI rule: keep pieces with more than 50 voxels on more than 1 slice
PLC_PHASES = {"P": "nc", "C1": "ap", "C2": "pvp", "C3": "dp"}
WAW_PHASES = {"0": "nc", "1": "ap", "2": "pvp", "3": "dp"}


@dataclass
class Labels:
    phase: str                 # phase the lesions were drawn on
    image: object              # nibabel image whose grid the arrays follow
    label: np.ndarray          # uint8, 0 background / 1 malignant / 2 haemangioma / 3 ignore
    instances: np.ndarray      # uint16, 0 or the lesion's id
    lesions: list = field(default_factory=list)   # one dict per lesion: instance, class, source, voxels, ...


def paint(shape, parts):
    """Label and instance arrays from [(instance id or 0, class, boolean mask)]; ignore first, then large to small."""
    label, instances = np.zeros(shape, np.uint8), np.zeros(shape, np.uint16)
    order = sorted(parts, key=lambda p: (p[1] != IGNORE, -int(p[2].sum())))
    for iid, cls, mask in order:
        label[mask] = cls
        instances[mask] = iid
    return label, instances


def pieces(mask):
    """26-connected pieces of a mask: (piece-id array, [(piece id, voxels, slices along axis 2)], largest first)."""
    lab, _ = ndimage.label(mask, structure=C.FULL, output=np.uint16)
    voxels = np.bincount(lab.ravel())
    found = [(k, int(voxels[k]), box[2].stop - box[2].start) for k, box in enumerate(ndimage.find_objects(lab), 1)]
    return lab, sorted(found, key=lambda p: -p[1])


def from_pieces(lab, assign):
    """Label and instance arrays from piece ids and {piece id: (instance id or 0, class)}; pieces never overlap."""
    lut_class = np.zeros(int(lab.max()) + 1, np.uint8)
    lut_instance = np.zeros(int(lab.max()) + 1, np.uint16)
    for k, (iid, cls) in assign.items():
        lut_class[k], lut_instance[k] = cls, iid
    return lut_class[lab], lut_instance[lab]


def mct(tar, patient_type):
    """MCT-LTDiag: largest piece + pieces the authors' rule keeps are lesions; every other piece is ignore."""
    img = mct_reader.read(tar, ("tumour",))["tumour"]
    cls = MCT_CLASS[patient_type]
    lab, found = pieces(np.asarray(img.dataobj) > 0)
    assign, lesions = {}, []
    for rank, (k, voxels, slices) in enumerate(found, 1):
        keep = rank == 1 or (voxels > MCT_MIN_VOXELS and slices > 1)
        iid = len(lesions) + 1 if keep else 0
        assign[k] = (iid, cls if keep else IGNORE)
        if keep:
            lesions.append({"instance": iid, "class": cls, "source": f"piece {rank} by size", "voxels": voxels})
    label, instances = from_pieces(lab, assign)
    return [Labels("pvp", img, label, instances, lesions)]


def plc_ct(raw, pid, code):
    """PLC-CECT CT of one phase as a 3D nibabel image (a few are saved as identical colour channels: take one)."""
    img = nib.load(Path(raw) / "ct_files" / f"{pid}_ct_{code}.nii.gz")
    if len(img.shape) > 3:
        img = nib.Nifti1Image(np.asarray(img.dataobj)[..., 0, 0], img.affine)
    return img


def plc(raw, pid, cancer_type):
    """PLC-CECT: every piece of the PVP (C2) mask is a malignant lesion; controls (CN) have none.

    Reads the corrected files (convert/plc_cect_geometry.py: true voxel size, right way up, HU + 40).
    """
    fixed = Path(raw) / "fixed" / pid
    if not (fixed / "geometry.json").exists():
        raise FileNotFoundError("no corrected PLC-CECT files for this patient (run plc_cect_geometry.py apply)")
    img = nib.load(fixed / "ct_pvp.nii.gz")
    shape = img.shape[:3]
    mask_path = fixed / "lesion_pvp.nii.gz"
    if cancer_type == "CN" or not mask_path.exists():
        return [Labels("pvp", img, np.zeros(shape, np.uint8), np.zeros(shape, np.uint16), [])]
    lab, found = pieces(np.asarray(nib.load(mask_path).dataobj) > 0)
    lesions = [{"instance": i, "class": MALIGNANT, "source": f"piece {i} by size", "voxels": p[1]}
               for i, p in enumerate(found, 1)]
    label, instances = from_pieces(lab, {p[0]: (i, MALIGNANT) for i, p in enumerate(found, 1)})
    return [Labels("pvp", img, label, instances, lesions)]


def geometry(path):
    """(size, spacing, origin, direction) of an image file, without reading its pixels."""
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return r.GetSize(), r.GetSpacing(), r.GetOrigin(), r.GetDirection()


def place_mask(mask, named, grids):
    """(phase, array (z, y, x), "same grid" | "resampled") for a WAW tumour mask, or None.

    A mask lying exactly on one of the patient's scans was drawn on that scan, whatever its file name says
    (15 masks, decided 2026-11-05). Otherwise it is placed on its named phase by physical position, if that keeps
    its volume within 10 % (else on another phase that does).
    """
    geom = (mask.GetSize(), mask.GetSpacing(), mask.GetOrigin(), mask.GetDirection())
    same = lambda a, b: a[0] == b[0] and all(np.allclose(x, y, atol=1e-2) for x, y in zip(a[1:], b[1:]))  # noqa: E731
    order = [named] + [p for p in grids if p != named]
    for ph in order:
        if ph in grids and same(geom, grids[ph]):
            return ph, sitk.GetArrayFromImage(mask) > 0, "same grid"
    volume = float(sitk.GetArrayViewFromImage(mask).sum()) * np.prod(mask.GetSpacing())
    for ph in order:
        if ph not in grids:
            continue
        size, spacing, origin, direction = grids[ph]
        ref = sitk.Image(size, sitk.sitkUInt8)
        ref.SetSpacing(spacing)
        ref.SetOrigin(origin)
        ref.SetDirection(direction)
        arr = sitk.GetArrayFromImage(sitk.Resample(mask, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0)) > 0
        if volume > 0 and abs(arr.sum() * np.prod(spacing) - volume) <= 0.1 * volume:
            return ph, arr, "resampled"
    return None


def waw_files(raw):
    """{(patient, phase code): CT path} and {patient: [(phase code, k, mask path)]} for WAW-TACE."""
    cts, masks = {}, {}
    for p in Path(raw).glob("ct_scans_*/*/*_scan.nii.gz"):
        pid, code = re.fullmatch(r"(\d+)_(\d)_scan\.nii\.gz", p.name).groups()
        cts[(pid, code)] = p
    for p in Path(raw).glob("tumor_masks_*/**/*_tumor_seg.nrrd"):
        pid, code, k = re.fullmatch(r"(\d+)_(\d)_(\d+)_tumor_seg\.nrrd", p.name).groups()
        masks.setdefault(pid, []).append((code, int(k), p))
    return cts, masks


def waw(cts, masks, pid):
    """WAW-TACE: one tumour file = one malignant lesion, on the scan it was drawn on."""
    grids = {WAW_PHASES[code]: geometry(p) for (q, code), p in cts.items() if q == pid}
    paths = {WAW_PHASES[code]: p for (q, code), p in cts.items() if q == pid}
    placed = {}
    for code, k, path in sorted(masks.get(pid, [])):
        m = sitk.Cast(sitk.ReadImage(str(path)) > 0, sitk.sitkUInt8)
        found = place_mask(m, WAW_PHASES[code], grids)
        if found is None:
            raise ValueError(f"tumour mask {k} ({WAW_PHASES[code]}) fits no CT of this patient")
        ph, arr, how = found
        placed.setdefault(ph, []).append((code, k, np.transpose(arr, (2, 1, 0)), how))
    out = []
    for ph, items in placed.items():
        img = nib.load(paths[ph])
        parts, lesions = [], []
        for i, (code, k, m, how) in enumerate(items, 1):
            parts.append((i, MALIGNANT, m))
            lesions.append({"instance": i, "class": MALIGNANT, "source": f"tumour file {code}_{k}",
                            "named_phase": WAW_PHASES[code], "placement": how, "voxels": int(m.sum()),
                            "idx": np.argwhere(m)})  # own full outline, also where tumours overlap
        label, instances = paint(img.shape, parts)
        out.append(Labels(ph, img, label, instances, lesions))
    return out


def tcia(nifti_dir):
    """TCIA-CRLM: one SEG segment Tumor_k = one malignant lesion; overlapping segments stay separate."""
    folder = Path(nifti_dir)
    img = nib.load(folder / "ct.nii.gz")
    stack = nib.load(folder / "tumour_stack.nii.gz")
    names = json.loads((folder / "tumour_stack.json").read_text())["segments"]
    arr = np.asarray(stack.dataobj) > 0
    if arr.ndim == 3:
        arr = arr[..., None]
    parts, lesions = [], []
    for j, name in enumerate(names):
        k = int(re.fullmatch(r"Tumor_(\d+)", name).group(1))
        parts.append((k, MALIGNANT, arr[..., j]))
        lesions.append({"instance": k, "class": MALIGNANT, "source": f"segment {name}",
                        "voxels": int(arr[..., j].sum()), "idx": np.argwhere(arr[..., j])})  # own full outline
    label, instances = paint(img.shape, parts)
    return [Labels("pvp", img, label, instances, lesions)]


def lits(image_path, label_path):
    """LiTS: every 26-connected piece of label 2 ("cancer") is a malignant lesion; label 1 is liver."""
    img = nib.load(image_path)
    lab, found = pieces(np.asarray(nib.load(label_path).dataobj) == 2)
    lesions = [{"instance": i, "class": MALIGNANT, "source": f"piece {i} by size", "voxels": p[1]}
               for i, p in enumerate(found, 1)]
    label, instances = from_pieces(lab, {p[0]: (i, MALIGNANT) for i, p in enumerate(found, 1)})
    return [Labels("pvp", img, label, instances, lesions)]
