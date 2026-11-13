#!/usr/bin/env python3
"""Step 14: final automatic checks of release v1 before upload (PLAN.md Step 14).

Why: a release must not ship a leaked split, a wrong format, a lost lesion or a crop that hints where the tumour is.

Checks (numbers as in PLAN.md Step 14; each prints PASS / FAIL with counts):
  1  no patient in more than one split; every split case is development data (external sets are not in v1)
  2  re-running steps/split.py with the same seed gives byte-identical files
  3  every split holds each tumour type of each dataset, haemangiomas, and malignant lesions of every size stratum
  4  crops never came from a lesion mask: on every case the PVP liver lies at most 21 mm from every crop face
     (liver box + 20 mm; less only where the scan ends) — a tumour-built crop would leave larger margins. A case over
     21 mm on the 1 mm grid is re-measured against its native-scan liver (outer voxel edges vs case.json crop_box_mm):
     resampling can drop a thin outer row of liver (seen once: 3 voxels, 1.6 mm), so the native check (<= 20.5 mm) decides
  5  phase files in order NC, AP, PVP, DP; every missing phase is flagged (phases_present false + a phase note)
  6  every image / mask: one grid per case, 1 x 1 x 1 mm, identity direction (LPS), int16 images, uint8 masks, uint16
     instances (headers of all files); image values within [-200, 200] on --value-sample cases per dataset
  7  patients match the publications (MCT-LTDiag 517, PLC-CECT 361); lesions in the release = Step 7 table minus the
     lesions outside the crop; per case, lesions on the 1 mm grid = lesions in the release
  8  lesions.csv sizes equal the Step 7 native-scan measurements (outputs/lesions/) and have a native slice thickness
  9  check pictures looked right (manual, by the data lead, 11-12 Nov 2026) — recorded, not computed
  10 nnU-Net integrity check — not part of v1 (Step 12 is done by the users, decided 2026-11-12)
  +  no unfinished files (.part) in the release; every case has its lesion-free masks and a complete case.json
Outputs: <release>/qc/release_tests.md (table) and release_tests_details.csv (failing cases, IDs on disk only).
Usage: python -I release_tests.py --release <project>/processed/v1 --project <project> [--value-sample 50]
"""
import argparse
import filecmp
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

PHASES = ["nc", "ap", "pvp", "dp"]
PUBLISHED = {"MCT-LTDiag": 517, "PLC-CECT": 361}
CASE_KEYS = ["dataset", "patient_id", "case_id", "role", "phases_present", "native_spacing_mm", "registration",
             "crop_box_mm", "diagnosis", "pipeline_commit", "lesions", "lesion_free"]


def header(path):
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return (r.GetSize(), r.GetSpacing(), r.GetOrigin(), r.GetDirection(), sitk.GetPixelIDValueAsString(r.GetPixelID()))


def native_margin_ok(project, cid, rec, tol=20.5):
    """Crop faces minus the native PVP liver's outer voxel edges, per axis and side; ok if none exceeds tol mm."""
    ds, pid = cid.split("_", 1)
    nat = sitk.ReadImage(str(project / "outputs" / "liver_masks" / ds / pid / "liver_pvp.nii.gz"))  # kept alive
    idx = np.argwhere(sitk.GetArrayViewFromImage(nat) > 0)
    lo_i, hi_i = idx.min(0) - 0.5, idx.max(0) + 0.5
    pts = np.array([nat.TransformContinuousIndexToPhysicalPoint([float(x), float(y), float(z)])
                    for z in (lo_i[0], hi_i[0]) for y in (lo_i[1], hi_i[1]) for x in (lo_i[2], hi_i[2])])
    lo, hi = np.array(rec["crop_box_mm"][0]), np.array(rec["crop_box_mm"][1])
    margins = np.concatenate([pts.min(0) - lo, hi - pts.max(0)])
    return bool((margins <= tol).all())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--value-sample", type=int, default=50)
    args = ap.parse_args()
    rel = args.release
    results, details = [], []

    def record(num, name, ok, note):
        results.append((num, name, "PASS" if ok else "FAIL", note))
        print(f"[{'PASS' if ok else 'FAIL'}] {num} {name}: {note}", flush=True)

    m = pd.read_csv(rel / "manifest.csv")
    sp = pd.read_csv(rel / "splits" / "splits.csv")
    les = pd.read_csv(rel / "lesions.csv")
    cases = {p.name: p for p in (rel / "cases").iterdir() if p.is_dir()}

    # 1
    dup = sp.case_id.duplicated().sum()
    ext = (~sp.dataset.isin(PUBLISHED)).sum()
    record("1", "no patient in two splits, no external data", dup == 0 and ext == 0 and set(sp.case_id) == set(m.case_id),
           f"{len(sp)} cases, {dup} duplicated, {ext} external, split = manifest: {set(sp.case_id) == set(m.case_id)}")
    # 2
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        for f in ("manifest.csv", "lesions.csv"):
            (t / f).write_bytes((rel / f).read_bytes())
        subprocess.run([sys.executable, "-I", str(Path(__file__).resolve().parents[1] / "steps" / "split.py"),
                        "--release", str(t)], check=True, capture_output=True)
        same = all(filecmp.cmp(t / "splits" / f, rel / "splits" / f, shallow=False)
                   for f in ("splits.csv", "train.txt", "val.txt", "test.txt"))
    record("2", "same seed gives the same split", same, "4 split files byte-identical" if same else "files differ")
    # 3
    msg, ok3 = [], True
    for (d, dx), g in sp.groupby(["dataset", "diagnosis"]):
        if set(g.split) != {"train", "val", "test"}:
            ok3 = False
            msg.append(f"{d} {dx} missing in {sorted({'train', 'val', 'test'} - set(g.split))}")
    lr = les[les.in_release.astype(bool)].merge(sp[["case_id", "split"]], on="case_id")
    for s in ("<10", "10-19", "20-29", ">=30"):
        if set(lr[(lr["class"] == 1) & (lr.size_stratum == s)].split) != {"train", "val", "test"}:
            ok3 = False
            msg.append(f"malignant {s} not in every split")
    if set(lr[lr["class"] == 2].split) != {"train", "val", "test"}:
        ok3 = False
        msg.append("haemangiomas not in every split")
    record("3", "every split has each tumour type and size stratum", ok3,
           "; ".join(msg) or f"{sp.groupby(['dataset', 'diagnosis']).ngroups} tumour types x 3 splits, 4 size strata + haemangiomas")
    # 4-6 and extras: one pass over the cases
    over21, rechecked, n4, bad5, bad6, part, incomplete = [], [], 0, [], [], [], []
    for cid, folder in sorted(cases.items()):
        if cid.endswith(".part"):
            part.append(cid)
            continue
        rec = json.loads((folder / "case.json").read_text())
        if any(k not in rec for k in CASE_KEYS):
            incomplete.append(cid)
        present = [ph for ph in PHASES if rec["phases_present"].get(ph)]
        for ph in PHASES:
            has = (folder / f"image_{ph}.nii.gz").exists()
            if has != (ph in present) or (ph not in present and ph not in rec.get("phase_notes", {})):
                bad5.append(cid)
        files = ([f"image_{p}.nii.gz" for p in present] + [f"liver_{p}.nii.gz" for p in present]
                 + ["lesion_label.nii.gz", "lesion_instances.nii.gz"]
                 + [f"lesion_free{s}_{p}.nii.gz" for p in present for s in ("", "_nomargin")])
        heads = {f: header(folder / f) for f in files if (folder / f).exists()}
        if len(heads) < len(files) or any(p.name.endswith((".part.nii.gz", ".part")) for p in folder.iterdir()):
            incomplete.append(cid)
        grids = {(h[0], tuple(np.round(h[2], 4))) for h in heads.values()}
        want = lambda f: ("16-bit signed integer" if f.startswith("image_") else  # noqa: E731
                          "16-bit unsigned integer" if f.startswith("lesion_instances") else "8-bit unsigned integer")
        if (len(grids) != 1 or any(tuple(np.round(h[1], 6)) != (1.0, 1.0, 1.0) for h in heads.values())
                or any(tuple(np.round(h[3], 6)) != (1, 0, 0, 0, 1, 0, 0, 0, 1) for h in heads.values())
                or any(h[4] != want(f) for f, h in heads.items())):
            bad6.append(cid)
        liver_img = sitk.ReadImage(str(folder / "liver_pvp.nii.gz"))  # kept alive while its array is used
        idx = np.argwhere(sitk.GetArrayViewFromImage(liver_img) > 0)
        size = np.array(liver_img.GetSize()[::-1])
        margins = np.concatenate([idx.min(0), size - 1 - idx.max(0)])  # 1 mm voxels
        n4 += 1
        if margins.max() > 21:
            rechecked.append(cid)
            if not native_margin_ok(args.project, cid, rec):
                over21.append(cid)
    for cid in over21 + bad5 + bad6 + part + incomplete:
        details.append({"case_id": cid, "check": "4" if cid in over21 else "5" if cid in bad5 else "6" if cid in bad6
                        else "extra"})
    record("4", "crops never from a lesion mask", not over21,
           f"{n4} cases: PVP liver within 21 mm of every crop face on the 1 mm grid in {n4 - len(rechecked)}; "
           f"{len(rechecked)} re-measured on the native-scan liver, {len(rechecked) - len(over21)} of them within "
           "20.5 mm; crop built in steps/standardise.py crop_grid() from the PVP liver mask only")
    n_missing = int((m.phases_present.str.split("|").str.len() < 4).sum())
    record("5", "phase order NC/AP/PVP/DP, missing phases flagged", not bad5,
           f"{len(cases)} cases; {n_missing} cases with a missing phase, all with a phase note" if not bad5
           else f"{len(set(bad5))} cases inconsistent")
    sample_bad, sampled = 0, 0
    for d in PUBLISHED:
        ids = sorted(c for c in cases if c.startswith(d + "_"))
        for cid in random.Random(0).sample(ids, min(args.value_sample, len(ids))):
            sampled += 1
            for ph in PHASES:
                f = cases[cid] / f"image_{ph}.nii.gz"
                if f.exists():
                    img = sitk.ReadImage(str(f))
                    a = sitk.GetArrayViewFromImage(img)
                    if a.min() < -200 or a.max() > 200:
                        sample_bad += 1
                        details.append({"case_id": cid, "check": "6 values"})
    record("6", "1 mm, LPS, int16 / uint8 / uint16, one grid; values in [-200, 200]", not bad6 and sample_bad == 0,
           f"headers of every file in {len(cases)} cases: {len(cases) - len(set(bad6))} fine; values: {sampled} sampled "
           f"cases, {sample_bad} images out of range")
    # 7
    counts = m.dataset.value_counts().to_dict()
    pub_ok = counts == PUBLISHED
    step7 = pd.concat([pd.read_csv(args.project / "outputs" / "lesions" / f"lesions_{d}.csv") for d in PUBLISHED])
    n_rel = int(les.in_release.astype(bool).sum())
    per_case_ok = 0
    in_rel = les[les.in_release.astype(bool)].groupby("case_id").size()
    for cid, folder in cases.items():
        rec = json.loads((folder / "case.json").read_text())
        if rec["lesions"]["instances_on_1mm_grid"] == int(in_rel.get(cid, 0)):
            per_case_ok += 1
    record("7", "counts match the publications and the lesion tables",
           pub_ok and len(les) == len(step7) and per_case_ok == len(cases),
           f"patients {counts} (published {PUBLISHED}); lesions.csv {len(les)} rows = Step 7 {len(step7)}, "
           f"{n_rel} in the release; per case lesions on the grid = release lesions in {per_case_ok}/{len(cases)}")
    # 8
    cols = ["case_id", "lesion_id", "inplane_max_diameter_mm", "max3d_diameter_mm", "volume_ml", "native_slice_mm"]
    j = les[cols].merge(step7[cols], on=["case_id", "lesion_id"], suffixes=("", "_s7"))
    same8 = all(np.allclose(j[c], j[c + "_s7"], equal_nan=True) for c in cols[2:])
    record("8", "lesion sizes from the native grid", same8 and len(j) == len(les) and les.native_slice_mm.notna().all(),
           f"{len(j)} lesions equal to the Step 7 native-scan measurements; native slice thickness known for "
           f"{int(les.native_slice_mm.notna().sum())}")
    record("9", "check pictures look right", True, "manual: 12 release pictures + 14 registration pictures checked by "
           "the data lead (11-12 Nov 2026); those checks found the duplicated PLC phases and the MCT misalignment, both fixed")
    record("10", "nnU-Net integrity check", True, "not part of v1: training files are built by the users (2026-11-12)")
    record("+", "no unfinished files, complete cases", not part and not incomplete,
           f"{len(part)} unfinished folders, {len(set(incomplete))} incomplete cases")
    lines = ["# Release v1 final checks (Step 14)", "", "| # | Check | Result | Details |", "|---|---|---|---|"]
    lines += [f"| {n} | {name} | **{r}** | {note} |" for n, name, r, note in results]
    (rel / "qc" / "release_tests.md").write_text("\n".join(lines) + "\n")
    pd.DataFrame(details, columns=["case_id", "check"]).to_csv(rel / "qc" / "release_tests_details.csv", index=False)
    fails = sum(r == "FAIL" for _, _, r, _ in results)
    print(f"done: {len(results)} checks, {fails} failed")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
