#!/usr/bin/env python3
"""Step 7: measure every lesion on its original scan, before any resampling (PLAN.md Step 7, STANDARD.md lesions.csv).

Lesions come from the standard labels (lib/labels.py, configs/label_maps.yaml). Diameters are measured between voxel
centres (decided 2026-11-06), on the grid of the phase each lesion was drawn on: in-plane maximum diameter (largest
over the lesion's slices), 3D maximum diameter, volume and equivalent diameter. The size stratum uses the in-plane
maximum diameter: <10, 10-19, 20-29, >=30 mm. PLC-CECT is measured on its corrected files (Step 4: true voxel size).

Output (--out folder), per dataset:
  lesions_<dataset>.csv    one row per lesion (classes 1 and 2)
  patients_<dataset>.csv   one row per usable patient: lesion counts, largest lesion, its stratum, lesion-count group
                           (inputs of the stratified split, Step 11)
Patients with a BLOCK note in the inventory are skipped. The log shows counts only.
Usage: python -I -u lesions.py --dataset MCT-LTDiag --data <project>/data --inventory <project>/outputs/inventory
         --out <project>/outputs/lesions [--limit N]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import components as C  # noqa: E402
import labels as L  # noqa: E402

STRATA = [(10, "<10"), (20, "10-19"), (30, "20-29"), (np.inf, ">=30")]
GEOMETRY_UNKNOWN = set()  # PLC-CECT is read from its corrected files (Step 4)


def stratum(d):
    return next(name for limit, name in STRATA if d < limit) if pd.notna(d) else ""


def labels_of(dataset, data, row, waw_index):
    pid = row.patient_id
    if dataset == "MCT-LTDiag":
        return L.mct(data / "MCT-LTDiag" / "tars" / f"{pid}.tar", row.diagnosis)
    if dataset == "PLC-CECT":
        return L.plc(data / "PLC-CECT", pid, row.diagnosis)
    if dataset == "WAW-TACE":
        return L.waw(*waw_index, pid)
    if dataset == "TCIA-CRLM":
        return L.tcia(data / "TCIA-CRLM" / "nifti" / pid)
    raw = data / "LiTS" / "Task03_Liver"
    return L.lits(raw / "imagesTr" / f"{pid}.nii.gz", raw / "labelsTr" / f"{pid}.nii.gz")


def measure_patient(dataset, pid, found):
    """Lesion rows and the patient row for one patient."""
    rows, ignore = [], 0
    for lab in found:
        spacing = np.array(lab.image.header.get_zooms()[:3], float)
        axis = [i for i, c in enumerate(nib.aff2axcodes(lab.image.affine)) if c in "SI"][0]
        known = dataset not in GEOMETRY_UNKNOWN
        boxes = ndimage.find_objects(lab.instances)
        _, n_ignore = ndimage.label(lab.label == L.IGNORE, structure=C.FULL)
        ignore += n_ignore
        for les in lab.lesions:
            idx = les.get("idx")
            if idx is None:  # pieces never overlap: the instance map holds each lesion whole
                box = boxes[les["instance"] - 1]
                idx = np.argwhere(lab.instances[box] == les["instance"]) + [s.start for s in box]
            m = C.measure(idx, spacing if known else np.ones(3), axis, centres=True)
            size = (lambda v: v) if known else (lambda v: np.nan)
            rows.append({"dataset": dataset, "patient": pid, "case_id": f"{dataset}_{pid}",
                         "lesion_id": les["instance"], "class": les["class"], "drawn_phase": lab.phase,
                         "named_phase": les.get("named_phase", lab.phase), "source": les["source"],
                         "voxels": m["voxels"], "slices": m["slices"],
                         "volume_ml": size(round(m["volume_mm3"] / 1000, 3)),
                         "equivalent_diameter_mm": size(m["equivalent_diameter_mm"]),
                         "inplane_max_diameter_mm": size(m["inplane_max_diameter_mm"]),
                         "max3d_diameter_mm": size(m["max3d_diameter_mm"]),
                         "size_stratum": stratum(size(m["inplane_max_diameter_mm"])),
                         "native_slice_mm": round(float(spacing[axis]), 3) if known else np.nan,
                         "multi_lesion_reviewed": False})
    malignant = [r for r in rows if r["class"] == L.MALIGNANT]
    largest = max((r["inplane_max_diameter_mm"] for r in rows), default=np.nan)
    patient = {"dataset": dataset, "patient": pid, "case_id": f"{dataset}_{pid}",
               "malignant_lesions": len(malignant), "haemangiomas": len(rows) - len(malignant),
               "ignore_regions": ignore, "largest_inplane_mm": largest,
               "largest_stratum": stratum(largest) if rows else "none",
               "lesion_count_group": "0" if not rows else "1" if len(rows) == 1 else "2+"}
    return rows, patient


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=["MCT-LTDiag", "PLC-CECT", "WAW-TACE", "TCIA-CRLM", "LiTS"])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, help="only the first N patients: test run")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(args.inventory / f"inventory_{args.dataset}.csv", dtype={"patient_id": str},
                      keep_default_na=False)
    usable = inv[~inv.notes.str.contains("BLOCK")]
    print(f"{args.dataset}: {len(usable)} usable patients ({len(inv) - len(usable)} with a BLOCK note skipped)",
          flush=True)
    if args.limit:
        usable = usable.iloc[:args.limit]
    waw_index = L.waw_files(args.data / "WAW-TACE") if args.dataset == "WAW-TACE" else None
    lesions, patients, failed, t0 = [], [], [], time.time()
    for k, row in enumerate(usable.itertuples(), 1):
        try:
            rows, patient = measure_patient(args.dataset, row.patient_id,
                                            labels_of(args.dataset, args.data, row, waw_index))
        except Exception as e:  # unreadable file, mask off its grid
            failed.append(row.patient_id)
            print(f"[{k:3d}/{len(usable)}] FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        lesions += rows
        patients.append({**patient, "diagnosis": row.diagnosis})
        if k % 50 == 0 or k == len(usable):
            print(f"[{k:3d}/{len(usable)}] {len(lesions)} lesions so far, {time.time() - t0:.0f} s", flush=True)
    les, pat = pd.DataFrame(lesions), pd.DataFrame(patients)
    les.to_csv(args.out / f"lesions_{args.dataset}.csv", index=False)
    pat.to_csv(args.out / f"patients_{args.dataset}.csv", index=False)
    if len(les):
        by = les.groupby(["class", "size_stratum"]).size().to_dict()
        print(f"lesions by (class, stratum): {by}", flush=True)
    print(f"done: {len(pat)} patients, {len(les)} lesions, {len(failed)} failed, {(time.time() - t0) / 60:.1f} min")
    if failed:
        sys.exit(f"{len(failed)} patients failed.")


if __name__ == "__main__":
    main()
