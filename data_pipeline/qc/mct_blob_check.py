#!/usr/bin/env python3
"""MCT blob check (PLAN.md Step 3, open item 8): measure every 3D connected component in all MCT-LTDiag tumour masks.

The masks hold more pieces than the dataset's one-tumour-per-patient table (meta_info_tumor.csv) and its rule
"tumours < 1 cm excluded, 5 mm slices" suggest. This script measures every 26-connected piece on the native grid
and compares it with the authors' VOI rule (VOI_extraction.py skips pieces with <= 50 voxels or on a single slice)
and with the table, so the team can choose a treatment. It decides nothing itself.

Output folder (--out), per-patient tables stay on the cluster:
  components.csv   one row per piece: size, slices, in-plane / 3D / equivalent diameter, share inside the
                   liver mask, gap to the patient's largest piece, kept by the authors' rule
  patients.csv     one row per patient: pieces, pieces kept by the authors' rule, largest piece vs the table
  summary.md       counts only (no patient IDs), for the team

Usage: python -I -u mct_blob_check.py --raw <project>/data/MCT-LTDiag --out <project>/outputs/mct_blob_check [--limit N]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import nibabel.processing
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import components as C  # noqa: E402
import mct_reader  # noqa: E402

AUTHORS_MIN_VOXELS = 50  # VOI_extraction.py: pieces with <= 50 voxels are skipped ("optional")
SIZE_BINS = [0, 5, 10, 20, np.inf]
SIZE_LABELS = ["<5 mm", "5-10 mm", "10-20 mm", ">=20 mm"]


def check_patient(tar):
    imgs = mct_reader.read(tar, ("tumour", "liver"))
    tumour, liver = imgs["tumour"], imgs["liver"]
    liver_regridded = liver.shape != tumour.shape or not np.allclose(liver.affine, tumour.affine, atol=1e-3)
    if liver_regridded:  # 3 patients: the authors' liver mask is on another grid; place it on the tumour grid by position
        liver = nibabel.processing.resample_from_to(liver, tumour, order=0)
    spacing = np.array(tumour.header.get_zooms()[:3], float)
    slice_axis = [i for i, c in enumerate(nib.aff2axcodes(tumour.affine)) if c in "SI"][0]
    mask = np.asarray(tumour.dataobj) > 0
    labels, comps = C.components(mask, spacing, slice_axis)
    liver_mask = np.asarray(liver.dataobj) > 0
    if comps:
        largest = max(comps, key=lambda c: c["voxels"])["component"]
        gap = ndimage.distance_transform_edt(labels != largest, sampling=spacing) if len(comps) > 1 else None
        for rank, c in enumerate(sorted(comps, key=lambda c: -c["voxels"]), 1):
            inside = labels == c["component"]
            c["rank"] = rank
            c["inside_liver_share"] = round(float(liver_mask[inside].mean()), 3)
            c["gap_to_largest_mm"] = 0.0 if c["component"] == largest else round(float(gap[inside].min()), 1)
            c["authors_rule_keeps"] = c["voxels"] > AUTHORS_MIN_VOXELS and c["slices"] > 1
    mask_max3d = C.max_distance(C.corners(np.argwhere(mask), spacing)) if comps else 0.0
    if comps:  # voxel-centre version too, to see which convention the dataset's Max3D_Diameter_mm used
        largest_idx = np.argwhere(labels == largest)
        comps[[c["component"] for c in comps].index(largest)]["max3d_centres_mm"] = round(
            C.max_distance(largest_idx * spacing), 1)
    return spacing, slice_axis, comps, round(mask_max3d, 1), liver_regridded


def summary(comp, pat, failed):
    """Markdown with counts only."""
    def table(df):
        return "```\n" + df.to_string() + "\n```"
    comp = comp.assign(size=pd.cut(comp.inplane_max_diameter_mm, SIZE_BINS, labels=SIZE_LABELS, right=False),
                       slices_group=np.where(comp.slices > 1, "2+ slices", "1 slice"),
                       main=np.where(comp["rank"] == 1, "largest", "other"))
    other = comp[comp.main == "other"]
    lines = ["# MCT-LTDiag blob check", "",
             f"Patients: {len(pat)} checked, {len(failed)} failed. Pieces (26-connected): {len(comp)}, "
             f"of which {len(other)} are not the patient's largest piece.", "",
             f"Patients with more than one piece: {(pat.pieces > 1).sum()}; "
             f"with more than one piece kept by the authors' rule: {(pat.pieces_kept_by_authors_rule > 1).sum()}.", "",
             "## All pieces by in-plane maximum diameter and number of slices", "",
             table(pd.crosstab(comp["size"], comp.slices_group, margins=True)), "",
             "## Pieces other than the largest: what the authors' rule (> 50 voxels and > 1 slice) does", "",
             table(pd.crosstab(other["size"], other.authors_rule_keeps.map({True: "kept", False: "removed"}),
                               margins=True)), "",
             "## Pieces other than the largest: distance to the largest piece", "",
             table(pd.crosstab(other["size"], pd.cut(other.gap_to_largest_mm, [0, 3, 10, 30, np.inf],
                                                   labels=["< 3 mm", "3-10 mm", "10-30 mm", ">= 30 mm"],
                                                   right=False), margins=True)), "",
             "## Pieces other than the largest, per tumour type", "",
             table(pd.crosstab(other.type, other.authors_rule_keeps.map({True: "kept", False: "removed"}),
                               margins=True)), "",
             "## Largest piece and whole mask vs meta_info_tumor.csv (one tumour per patient)", ""]
    for col, ref in [("largest_volume_mm3", "InExp_Volume_mm3"), ("mask_volume_mm3", "InExp_Volume_mm3"),
                     ("largest_volume_mm3", "Avg_Volume_mm3"), ("largest_max3d_mm", "Max3D_Diameter_mm"),
                     ("largest_max3d_centres_mm", "Max3D_Diameter_mm"), ("mask_max3d_mm", "Max3D_Diameter_mm")]:
        r = pat[col] / pat[ref]
        lines.append(f"- {col} / {ref}: median {r.median():.3f}, within ±10 %: {(r.sub(1).abs() <= 0.1).sum()} "
                     f"of {r.notna().sum()} patients")
    lines += ["", f"Patients whose liver mask lies on another grid than the tumour mask (placed on the tumour grid by "
                  f"position): {int(pat.liver_mask_regridded.sum())}.",
              "", f"Pieces with less than half their voxels inside the liver mask: "
                  f"{(comp.inside_liver_share < 0.5).sum()} (largest pieces among them: "
                  f"{((comp.inside_liver_share < 0.5) & (comp.main == 'largest')).sum()}).", "",
              "Sizes use the outer voxel surface (close to pyradiomics); slices are 5 mm, so a piece on one slice "
              "cannot be judged in 3D."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True, help="MCT-LTDiag raw folder (tars/ + metadata/)")
    ap.add_argument("--out", type=Path, required=True, help="output folder, e.g. <project>/outputs/mct_blob_check")
    ap.add_argument("--limit", type=int, help="only the first N patients: test run")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    table = mct_reader.patients(args.raw)
    tumour_meta = pd.read_csv(args.raw / "metadata" / "meta_info_tumor.csv", dtype={"ID": str})
    if args.limit:
        table = table.iloc[:args.limit]
    print(f"{len(table)} patients ({table.type.value_counts().to_dict()})", flush=True)

    comp_rows, pat_rows, failed, t0 = [], [], [], time.time()
    for k, row in enumerate(table.itertuples(), 1):
        try:
            spacing, slice_axis, comps, mask_max3d, liver_regridded = check_patient(row.tar)
        except Exception as e:  # unreadable tar or mask
            failed.append(row.patient_id)
            print(f"[{k:3d}/{len(table)}] FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        for c in comps:
            comp_rows.append({"patient_id": row.patient_id, "type": row.type, **c})
        largest = max(comps, key=lambda c: c["voxels"]) if comps else {}
        pat_rows.append({"patient_id": row.patient_id, "type": row.type,
                         "spacing_inplane_mm": round(float(spacing[[a for a in range(3) if a != slice_axis]][0]), 3),
                         "spacing_slice_mm": round(float(spacing[slice_axis]), 2),
                         "pieces": len(comps),
                         "pieces_kept_by_authors_rule": sum(c["authors_rule_keeps"] for c in comps),
                         "largest_volume_mm3": largest.get("volume_mm3", 0.0),
                         "largest_max3d_mm": largest.get("max3d_diameter_mm", 0.0),
                         "largest_max3d_centres_mm": largest.get("max3d_centres_mm", 0.0),
                         "mask_volume_mm3": round(sum(c["volume_mm3"] for c in comps), 1),
                         "mask_max3d_mm": mask_max3d,
                         "liver_mask_regridded": liver_regridded})
        if k % 50 == 0 or k == len(table):
            print(f"[{k:3d}/{len(table)}] {len(comp_rows)} pieces so far, {time.time() - t0:.0f} s", flush=True)

    comp = pd.DataFrame(comp_rows)
    pat = pd.DataFrame(pat_rows).merge(tumour_meta[["ID", "InExp_Volume_mm3", "Avg_Volume_mm3", "Max3D_Diameter_mm"]],
                                       left_on="patient_id", right_on="ID", how="left").drop(columns="ID")
    comp.to_csv(args.out / "components.csv", index=False)
    pat.to_csv(args.out / "patients.csv", index=False)
    (args.out / "summary.md").write_text(summary(comp, pat, failed))
    print(f"done: {len(pat)} patients, {len(comp)} pieces, {len(failed)} failed, {(time.time() - t0) / 60:.0f} min; "
          f"see summary.md")
    if failed:
        sys.exit(f"{len(failed)} patients failed.")


if __name__ == "__main__":
    main()
