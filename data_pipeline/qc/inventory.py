#!/usr/bin/env python3
"""Inventory of one raw dataset (PLAN.md Step 1): one row per patient and one row per annotated lesion.

Per patient: which phases exist (NC, AP, PVP, DP) and which are missing, slices and spacing per phase, which masks
the dataset provides and on which phase, lesions per standard class (1 malignant, 2 haemangioma, 3 ignore) and
notes on anything that may stop a case from being used ("BLOCK: ..." stops it, "NOTE: ..." does not). Per lesion:
class, the phase it was drawn on, voxels, slices and a provisional in-plane maximum diameter (outer voxel surface;
Step 7 fixes the convention). Reads only; nothing in data/ changes.

Phase codes (to confirm in Step 2): PLC-CECT P / C1 / C2 / C3 = NC / AP / PVP / DP (the dataset README calls C1
arterial); WAW-TACE 0 / 1 / 2 / 3 = NC / AP / PVP / DP (team pipeline processing/WAW-TACE); TCIA-CRLM and LiTS are
taken as PVP. MCT-LTDiag lesions use the blob-check output and the proposed blob rule (open item 8, pending Yifan).

Output (--out folder): inventory_<dataset>.csv, lesions_<dataset>.csv
Usage: python -I -u inventory.py --dataset PLC-CECT --data <project>/data --out <project>/outputs/inventory
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import components as C  # noqa: E402
import mct_reader  # noqa: E402
from labels import geometry, place_mask  # noqa: E402

PHASES = ["nc", "ap", "pvp", "dp"]
ROLE = {"MCT-LTDiag": "development", "PLC-CECT": "development",
        "WAW-TACE": "external", "TCIA-CRLM": "external", "LiTS": "external"}
PLC_PHASES = {"P": "nc", "C1": "ap", "C2": "pvp", "C3": "dp"}
WAW_PHASES = {"0": "nc", "1": "ap", "2": "pvp", "3": "dp"}
LIMIT = None  # --limit: first N patients only (test run)


def take(seq):
    return list(seq)[:LIMIT] if LIMIT else list(seq)


def patient(dataset, pid, diagnosis):
    return {"dataset": dataset, "patient_id": pid, "role": ROLE[dataset], "diagnosis": diagnosis, "notes": []}


def add_phase(row, phase, shape, spacing, slice_axis):
    """Record slices and spacing of one phase (spacing per array axis)."""
    plane = [a for a in range(3) if a != slice_axis]
    row[f"slices_{phase}"] = int(shape[slice_axis])
    row[f"inplane_mm_{phase}"] = round(float(spacing[plane[0]]), 3)
    row[f"slice_mm_{phase}"] = round(float(spacing[slice_axis]), 3)


def slice_axis_of(affine):
    return [i for i, c in enumerate(nib.aff2axcodes(affine)) if c in "SI"][0]


def lesion(dataset, pid, lesion_id, cls, phase, m):
    return {"dataset": dataset, "patient_id": pid, "lesion_id": lesion_id, "class": cls, "drawn_phase": phase,
            "voxels": m["voxels"], "slices": m["slices"], "inplane_max_diameter_mm": m["inplane_max_diameter_mm"],
            "equivalent_diameter_mm": m["equivalent_diameter_mm"]}


def finish(row, lesions):
    """Phase summary columns, lesion counts per class, notes as text."""
    present = [p for p in PHASES if f"slices_{p}" in row]
    row["phases_present"] = "|".join(present)
    row["missing_phases"] = "|".join(p for p in PHASES if p not in present)
    row["n_phases"] = len(present)
    for cls in (1, 2, 3):
        row[f"lesions_class{cls}"] = sum(les["class"] == cls for les in lesions)
    row["notes"] = "; ".join(row["notes"])
    return row


def mct(data, blobs):
    table = mct_reader.patients(data / "MCT-LTDiag")
    pieces = pd.read_csv(blobs, dtype={"patient_id": str})
    rows, lesions = [], []
    table = table.iloc[:LIMIT] if LIMIT else table
    for k, t in enumerate(table.itertuples(), 1):
        row = patient("MCT-LTDiag", t.patient_id, t.type)
        hdr = mct_reader.headers(t.tar)
        for ph in PHASES:
            add_phase(row, ph, hdr[ph].get_data_shape(), hdr[ph].get_zooms(), slice_axis_of(hdr[ph].get_best_affine()))
        same = lambda a, b: (hdr[a].get_data_shape() == hdr[b].get_data_shape()  # noqa: E731
                             and np.allclose(hdr[a].get_best_affine(), hdr[b].get_best_affine(), atol=1e-3))
        offset = max(float(np.abs(hdr[ph].get_best_affine() - hdr["pvp"].get_best_affine()).max()) for ph in PHASES)
        row["phase_header_offset_mm"] = round(offset, 2)
        if any(hdr[ph].get_data_shape() != hdr["pvp"].get_data_shape() for ph in PHASES):
            row["notes"].append("BLOCK: phases not on one voxel grid")
        elif offset > 1e-3:  # 14 ICC patients: same voxel grid, but headers place NC/AP/DP elsewhere; Step 5 checks
            row["notes"].append(f"NOTE: NC/AP/DP headers up to {offset:.1f} mm from PVP (same voxel grid)")
        if not same("tumour", "pvp"):
            row["notes"].append("BLOCK: tumour mask not on the PVP grid")
        if not same("liver", "pvp"):
            row["notes"].append("NOTE: authors' liver mask on another grid")
        row["lesion_mask_phases"], row["liver_mask_phases"] = "pvp", "pvp"
        cls = 2 if t.type == "HH" else 1
        mine = pieces[pieces.patient_id == t.patient_id]
        if mine.empty:
            row["notes"].append("BLOCK: no blob-check result")
        for p in mine.itertuples():  # proposed rule: largest piece + pieces the authors' VOI rule keeps
            keep = p.rank == 1 or bool(p.authors_rule_keeps)
            lesions.append(lesion("MCT-LTDiag", t.patient_id, p.component, cls if keep else 3, "pvp", p._asdict()))
        rows.append(finish(row, [x for x in lesions if x["patient_id"] == t.patient_id]))
        progress(k, len(table))
    return rows, lesions


def plc(data):
    raw = data / "PLC-CECT"
    table = pd.read_csv(raw / "patient_data.csv")
    rows, lesions = [], []
    groups = take(table.groupby("patient_id"))
    for k, (pid, g) in enumerate(groups, 1):
        types = g.cancer_type.dropna().unique()
        row = patient("PLC-CECT", pid, "/".join(types))
        if len(types) != 1:
            row["notes"].append("NOTE: cancer type differs between phase rows")
        row["notes"].append("NOTE: placeholder 1 mm spacing and 0-255 values until geometry recovery (Step 4)")
        mine, mask_phases, liver_phases = [], [], []
        for code, ph in PLC_PHASES.items():
            ct = raw / "ct_files" / f"{pid}_ct_{code}.nii.gz"
            if not ct.exists():
                continue
            img = nib.load(ct)
            add_phase(row, ph, img.shape, img.header.get_zooms(), slice_axis_of(img.affine))
            if len(img.shape) > 3:  # 91 patients: grey CT saved as an RGB vector image (512, 512, z, 1, 3)
                a = np.asarray(img.dataobj)
                same = all(np.array_equal(a[..., 0, 0], a[..., 0, c]) for c in range(1, a.shape[-1]))
                row["notes"].append(f"NOTE: {ph} CT saved with {a.shape[-1]} colour channels "
                                    f"({'identical' if same else 'DIFFERENT'})")
            liver = raw / "liver_mask_files" / f"{pid}-livermask_{code}.nii.gz"
            if liver.exists():
                liver_phases.append(ph)
            mask = raw / "mask_files" / f"{pid}_mask_{code}.nii.gz"
            if mask.exists():
                mask_phases.append(ph)
                m = nib.load(mask)
                if m.shape[:3] != img.shape[:3]:
                    row["notes"].append(f"BLOCK: {ph} lesion mask not on the CT grid")
                    continue
                labels, n = ndimage.label(np.asarray(m.dataobj) > 0.5, structure=C.FULL)
                row[f"lesions_on_{ph}"] = n
                if ph == "pvp":  # lesion rows from the PVP mask; sizes in mm wait for the real spacing (Step 4)
                    axis, voxels = slice_axis_of(m.affine), np.bincount(labels.ravel())
                    mine = [lesion("PLC-CECT", pid, j, 1, "pvp",
                                   {"voxels": int(voxels[j]), "slices": box[axis].stop - box[axis].start,
                                    "inplane_max_diameter_mm": np.nan, "equivalent_diameter_mm": np.nan})
                            for j, box in enumerate(ndimage.find_objects(labels), 1)]
        if types[0] != "CN" and "pvp" not in mask_phases:
            row["notes"].append("BLOCK: cancer patient without a PVP lesion mask")
        row["lesion_mask_phases"], row["liver_mask_phases"] = "|".join(mask_phases), "|".join(liver_phases)
        lesions += mine
        rows.append(finish(row, mine))
        progress(k, len(groups))
    return rows, lesions


def waw(data):
    raw = data / "WAW-TACE"
    meta = pd.read_csv(raw / "ct_hcc_metadata_v2.csv", dtype={"patient_id": str, "ct_phase": str})
    cts = {}
    for p in raw.glob("ct_scans_*/*/*_scan.nii.gz"):
        pid, code = re.fullmatch(r"(\d+)_(\d)_scan\.nii\.gz", p.name).groups()
        cts[(pid, code)] = p
    masks = {}
    for p in raw.glob("tumor_masks_*/**/*_tumor_seg.nrrd"):
        pid, code, n = re.fullmatch(r"(\d+)_(\d)_(\d+)_tumor_seg\.nrrd", p.name).groups()
        masks.setdefault(pid, []).append((code, int(n), p))
    rows, lesions = [], []
    pids = take(sorted({pid for pid, _ in cts}, key=int))
    for k, pid in enumerate(pids, 1):
        row = patient("WAW-TACE", pid, "HCC")
        grids = {}
        for code, ph in WAW_PHASES.items():
            if (pid, code) in cts:
                img = nib.load(cts[(pid, code)])
                add_phase(row, ph, img.shape, img.header.get_zooms(), slice_axis_of(img.affine))
                grids[ph] = geometry(cts[(pid, code)])
                thick = meta[(meta.patient_id == pid) & (meta.ct_phase == code)].slice_thickness
                if len(thick):
                    row[f"thickness_mm_{ph}"] = float(thick.iloc[0])
        mine = []
        for code, n, path in sorted(masks.get(pid, [])):
            named = WAW_PHASES[code]
            m = sitk.Cast(sitk.ReadImage(str(path)) > 0, sitk.sitkUInt8)
            placed = place_mask(m, named, grids)
            if placed is None:
                row["notes"].append(f"BLOCK: tumour mask {n} ({named}) fits no CT of this patient")
                continue
            ph, arr, how = placed
            if ph != named:
                row["notes"].append(f"NOTE: tumour mask {n} named {named} lies on the {ph} CT grid")
            elif how == "resampled":
                row["notes"].append(f"NOTE: tumour mask {n} placed on the {ph} CT grid by position")
            idx = np.argwhere(arr)
            if len(idx) == 0:
                row["notes"].append(f"NOTE: tumour mask {n} on {ph} is empty")
                continue
            mine.append(lesion("WAW-TACE", pid, f"{code}_{n}", 1, ph, C.measure(idx, grids[ph][1][::-1], 0)))
        row["lesion_mask_phases"] = "|".join(sorted({x["drawn_phase"] for x in mine}, key=PHASES.index))
        row["liver_mask_phases"] = ""  # organ masks are TotalSegmentator output, not annotations
        listed = meta[meta.patient_id == pid].tumor_count.sum()
        if listed != len(masks.get(pid, [])):
            row["notes"].append(f"NOTE: metadata lists {listed} tumours, {len(masks.get(pid, []))} mask files")
        lesions += mine
        rows.append(finish(row, mine))
        progress(k, len(pids))
    return rows, lesions


def lits(data):
    raw = data / "LiTS" / "Task03_Liver"
    rows, lesions = [], []
    images = take(sorted((raw / "imagesTr").glob("liver_*.nii.gz"),
                         key=lambda p: int(re.search(r"\d+", p.name).group())))
    for k, img_path in enumerate(images, 1):
        pid = img_path.name[:-7]
        row = patient("LiTS", pid, "liver tumour (LiTS label 2)")
        img, lab = nib.load(img_path), nib.load(raw / "labelsTr" / img_path.name)
        axis = slice_axis_of(img.affine)
        add_phase(row, "pvp", img.shape, img.header.get_zooms(), axis)
        if lab.shape != img.shape:
            row["notes"].append("BLOCK: label not on the image grid")
            rows.append(finish(row, []))
            continue
        arr = np.asarray(lab.dataobj)
        _, comps = C.components(arr == 2, lab.header.get_zooms(), axis)
        mine = [lesion("LiTS", pid, c["component"], 1, "pvp", c) for c in comps]
        row["lesion_mask_phases"], row["liver_mask_phases"] = "pvp", "pvp" if (arr > 0).any() else ""
        lesions += mine
        rows.append(finish(row, mine))
        progress(k, len(images))
    return rows, lesions


def tcia(data):
    raw = data / "TCIA-CRLM"
    inv = pd.read_csv(raw / "inventory_TCIA-CRLM.csv", dtype={"patient_id": str})
    inv = inv.iloc[:LIMIT] if LIMIT else inv
    rows, lesions = [], []
    for k, t in enumerate(inv.itertuples(), 1):
        row = patient("TCIA-CRLM", t.patient_id, "CRLM")
        folder = raw / "nifti" / t.patient_id
        tum = nib.load(folder / "tumours.nii.gz")
        axis = slice_axis_of(tum.affine)
        add_phase(row, "pvp", tum.shape, tum.header.get_zooms(), axis)
        row["thickness_mm_pvp"] = t.slice_thickness_mm
        arr = np.asarray(tum.dataobj)
        mine = [lesion("TCIA-CRLM", t.patient_id, int(v), 1, "pvp",
                       C.measure(np.argwhere(arr == v), tum.header.get_zooms(), axis))
                for v in np.unique(arr) if v > 0]
        if t.n_tumours != len(mine):
            row["notes"].append(f"NOTE: {t.n_tumours - len(mine)} tumour segments hidden by overlapping ones")
        if t.tumour_outside_liver_voxels > 0.1 * max(t.tumour_voxels, 1):
            row["notes"].append("NOTE: > 10 % of tumour voxels outside the liver segment")
        row["lesion_mask_phases"], row["liver_mask_phases"] = "pvp", "pvp"
        lesions += mine
        rows.append(finish(row, mine))
        progress(k, len(inv))
    return rows, lesions


def progress(k, n):
    if k % 50 == 0 or k == n:
        print(f"  {k}/{n} patients", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=list(ROLE))
    ap.add_argument("--data", type=Path, required=True, help="raw data root, e.g. <project>/data")
    ap.add_argument("--out", type=Path, required=True, help="output folder, e.g. <project>/outputs/inventory")
    ap.add_argument("--blobs", type=Path, help="MCT blob-check components.csv (MCT-LTDiag only)")
    ap.add_argument("--limit", type=int, help="only the first N patients: test run")
    args = ap.parse_args()
    global LIMIT
    LIMIT = args.limit
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"{args.dataset}: reading ...", flush=True)
    if args.dataset == "MCT-LTDiag":
        rows, lesions = mct(args.data, args.blobs)
    else:
        rows, lesions = {"PLC-CECT": plc, "WAW-TACE": waw, "LiTS": lits, "TCIA-CRLM": tcia}[args.dataset](args.data)
    first = ["dataset", "patient_id", "role", "diagnosis", "phases_present", "missing_phases", "n_phases",
             "lesion_mask_phases", "liver_mask_phases", "lesions_class1", "lesions_class2", "lesions_class3", "notes"]
    inv = pd.DataFrame(rows)
    inv = inv[first + [c for c in inv.columns if c not in first]]
    inv.to_csv(args.out / f"inventory_{args.dataset}.csv", index=False)
    pd.DataFrame(lesions).to_csv(args.out / f"lesions_{args.dataset}.csv", index=False)
    blocked = inv.notes.str.contains("BLOCK").sum()
    print(f"done: {len(inv)} patients, {len(lesions)} lesions, {blocked} patients with a BLOCK note, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
