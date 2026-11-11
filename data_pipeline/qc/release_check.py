#!/usr/bin/env python3
"""Step 10: data check and release tables for the standard case folders (STANDARD.md, PLAN.md Step 10).

Why: before a release, every case must be complete and in the standard format, every lesion accounted for, and every
registration problem visible, with counts the paper's dataset table can use.

What (for the datasets given, from <release>/cases/ and the Step 7 tables in <project>/outputs/lesions/):
  manifest.csv           one row per case (STANDARD.md columns first, then QC extras)
  lesions.csv            one row per lesion, measured on the native scan (Step 7), for the cases present
  exclusion.csv          case_id, dataset, stage, reason, excluded_from_split: missing or broken cases (excluded) and
                         failed registrations (flagged and kept: decided Oct 2026, "decide per experiment")
  qc/registration_qc.csv liver Dice, gradient NCC and coverage per case and phase on the 1 mm grid
  qc/qc_report.md        counts per dataset, class and size stratum; registration QC; format checks (pass / fail)
  qc/overlays/<dataset>/ check pictures of --pictures random cases per dataset (seed 0) + pictures_index.csv
Format checks per case: all 19 files present (4 images, 4 livers, 2 lesion maps, 8 lesion-free masks, case.json);
every image and mask on one grid, 1 mm, identity direction (LPS); pixel types int16 / uint8 / uint16; lesions on the
1 mm grid = lesions in the Step 7 table. Values in [-200, 200] are checked on --value-sample cases per dataset.
Patient IDs stay in the files; the log and the report show counts only.
Usage: python -I -u release_check.py --release <project>/processed/v1 --project <project> \
           --datasets MCT-LTDiag PLC-CECT [--pictures 6] [--value-sample 20]
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

PHASES = ["nc", "ap", "pvp", "dp"]
FILES = ([f"image_{p}.nii.gz" for p in PHASES] + [f"liver_{p}.nii.gz" for p in PHASES]
         + ["lesion_label.nii.gz", "lesion_instances.nii.gz"]
         + [f"lesion_free{s}_{p}.nii.gz" for p in PHASES for s in ("", "_nomargin")])
TYPES = {"image": "16-bit signed integer", "lesion_instances": "16-bit unsigned integer"}
LESION_COLUMNS = ["dataset", "patient", "case_id", "lesion_id", "class", "drawn_phase", "named_phase", "volume_ml",
                  "equivalent_diameter_mm", "inplane_max_diameter_mm", "max3d_diameter_mm", "size_stratum",
                  "native_slice_mm", "multi_lesion_reviewed", "voxels", "slices", "source"]


def header(path):
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return (r.GetSize(), tuple(np.round(r.GetSpacing(), 6)), tuple(np.round(r.GetOrigin(), 4)),
            tuple(np.round(r.GetDirection(), 6)), sitk.GetPixelIDValueAsString(r.GetPixelID()))


def check_case(folder, n_table):
    """List of problems of one case folder (empty = fine) and its case.json."""
    problems = []
    missing = [f for f in FILES + ["case.json"] if not (folder / f).exists()]
    if missing:
        problems.append(f"missing files: {len(missing)}")
    record = json.loads((folder / "case.json").read_text()) if (folder / "case.json").exists() else {}
    grids = set()
    for f in FILES:
        if f in missing:
            continue
        size, spacing, origin, direction, ptype = header(folder / f)
        grids.add((size, origin))
        if spacing != (1.0, 1.0, 1.0):
            problems.append(f"{f}: spacing {spacing}")
        if direction != (1, 0, 0, 0, 1, 0, 0, 0, 1):
            problems.append(f"{f}: direction not identity")
        want = TYPES["image"] if f.startswith("image_") else TYPES.get(f[:-7], "8-bit unsigned integer")
        if ptype != want:
            problems.append(f"{f}: pixel type {ptype}")
    if len(grids) > 1:
        problems.append("files on different grids")
    les = record.get("lesions", {})
    if les and les.get("instances_native") != les.get("instances_on_1mm_grid"):
        problems.append(f"lesions lost in resampling: {les.get('instances_native')} -> {les.get('instances_on_1mm_grid')}")
    if les and n_table is not None and les.get("instances_native") != n_table:
        problems.append(f"lesions {les.get('instances_native')} vs Step 7 table {n_table}")
    if "lesion_free" not in record:
        problems.append("no lesion-free masks recorded (Step 9)")
    return problems, record


def values_ok(folder):
    for p in PHASES:
        img = sitk.ReadImage(str(folder / f"image_{p}.nii.gz"))  # keep the image alive while its array view is used
        a = sitk.GetArrayViewFromImage(img)
        if a.min() < -200 or a.max() > 200:
            return False
    return True


def picture(folder, out_png):
    """Axial and coronal slice through the largest lesion (else the liver centre), all four phases (Pillow)."""
    from PIL import Image, ImageDraw
    from scipy import ndimage
    g = lambda n: sitk.GetArrayFromImage(sitk.ReadImage(str(folder / n)))  # noqa: E731
    lab, inst = g("lesion_label.nii.gz"), g("lesion_instances.nii.gz")
    ids, counts = np.unique(inst[inst > 0], return_counts=True)
    target = (inst == ids[np.argmax(counts)]) if len(ids) else (g("liver_pvp.nii.gz") > 0)
    z, y, _ = [int(v) for v in np.argwhere(target).mean(axis=0)]
    colours = {1: (255, 0, 0), 2: (0, 100, 255), 3: (255, 215, 0)}
    tiles = []
    for ph in PHASES:
        img, free = g(f"image_{ph}.nii.gz"), g(f"lesion_free_{ph}.nii.gz") > 0
        column = []
        for cut in (lambda a: a[z], lambda a: a[::-1, y, :]):
            grey = ((np.clip(cut(img), -200, 200) + 200) / 400 * 255).astype(np.uint8)
            rgb = np.repeat(grey[..., None], 3, axis=2).astype(np.float32)
            for c, col in colours.items():
                m = cut(lab) == c
                rgb[m] = 0.55 * rgb[m] + 0.45 * np.array(col)
            f = cut(free)
            rgb[f & ~ndimage.binary_erosion(f)] = (0, 255, 255)
            column.append(Image.fromarray(rgb.astype(np.uint8)))
        tiles.append(column)
    w = max(t.width for col in tiles for t in col)
    h = [max(col[r].height for col in tiles) for r in range(2)]
    sheet = Image.new("RGB", (4 * w, h[0] + h[1] + 18), "black")
    for i, (ph, col) in enumerate(zip(PHASES, tiles)):
        sheet.paste(col[0], (i * w, 18))
        sheet.paste(col[1], (i * w, 18 + h[0]))
        ImageDraw.Draw(sheet).text((i * w + 4, 3), f"{ph.upper()}: axial (top), coronal (bottom)", fill="white")
    ImageDraw.Draw(sheet).text((4, sheet.height - 14), "red malignant, blue haemangioma, yellow ignore, "
                               "cyan = lesion-free liver (5 mm margin)", fill="white")
    sheet.save(out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--pictures", type=int, default=6)
    ap.add_argument("--value-sample", type=int, default=20)
    args = ap.parse_args()
    rel = args.release
    (rel / "qc").mkdir(parents=True, exist_ok=True)
    manifest, lesions, exclusion, regqc, report = [], [], [], [], ["# Release QC report", ""]
    for ds in args.datasets:
        table = pd.read_csv(args.project / "outputs" / "lesions" / f"lesions_{ds}.csv", dtype={"patient": str})
        pats = pd.read_csv(args.project / "outputs" / "lesions" / f"patients_{ds}.csv", dtype={"patient": str})
        folders = {p.name: p for p in (rel / "cases").glob(f"{ds}_*") if p.is_dir() and not p.name.endswith(".part")}
        n_les = table.groupby("case_id").size()
        problems_by_case, bad_values = {}, 0
        for cid in sorted(set(pats.case_id) - set(folders)):
            exclusion.append({"case_id": cid, "dataset": ds, "stage": "Step 8", "reason": "no case folder",
                              "excluded_from_split": True})
        for k, (cid, folder) in enumerate(sorted(folders.items()), 1):
            problems, rec = check_case(folder, int(n_les.get(cid, 0)))
            problems_by_case[cid] = problems
            if problems:
                exclusion.append({"case_id": cid, "dataset": ds, "stage": "Step 10 format check",
                                  "reason": "; ".join(problems), "excluded_from_split": True})
            p = pats[pats.case_id == cid].iloc[0]
            reg = rec.get("registration", {})
            failed = [ph for ph, v in reg.items() if v.get("failed")]
            for ph in failed:
                exclusion.append({"case_id": cid, "dataset": ds, "stage": "Step 5 registration QC",
                                  "reason": f"{ph} liver Dice {reg[ph]['liver_dice']:.3f} < 0.9 (flagged, kept)",
                                  "excluded_from_split": False})
            for ph, v in reg.items():
                regqc.append({"case_id": cid, "dataset": ds, "phase": ph, **{key: v.get(key) for key in
                              ("liver_dice", "grad_ncc", "coverage", "failed")}})
            manifest.append({
                "case_id": cid, "dataset": ds, "role": rec.get("role"), "diagnosis": rec.get("diagnosis"),
                "phases_present": "|".join(ph for ph, v in rec.get("phases_present", {}).items() if v),
                "native_slice_mm": round(rec["native_spacing_mm"]["pvp"][2], 3) if rec else None,
                **{f"liver_dice_{ph}": reg.get(ph, {}).get("liver_dice") for ph in ("nc", "ap", "dp")},
                "registration_failed": bool(failed), "landmark_case": rec.get("landmark_case"), "tre_failed": None,
                "n_malignant_lesions": int(p.malignant_lesions), "max_diameter_bin": p.largest_stratum,
                "excluded": bool(problems),
                **{f"grad_ncc_{ph}": reg.get(ph, {}).get("grad_ncc") for ph in ("nc", "ap", "dp")},
                "coverage_min": min((v.get("coverage", 1) for v in reg.values()), default=None),
                "n_haemangiomas": int(p["haemangiomas"]) if pd.notna(p.get("haemangiomas")) else 0,
                "n_ignore_regions": int(p["ignore_regions"]) if pd.notna(p.get("ignore_regions")) else 0,
                "lesion_count_group": p.lesion_count_group, "registration_failed_phases": "|".join(failed),
                "pipeline_commit": rec.get("pipeline_commit")})
            if k % 100 == 0:
                print(f"  {ds}: {k}/{len(folders)} cases checked", flush=True)
        sample = random.Random(0).sample(sorted(folders), min(args.value_sample, len(folders)))
        bad_values = sum(not values_ok(folders[c]) for c in sample)
        lesions.append(table[table.case_id.isin(folders)].reindex(columns=LESION_COLUMNS))
        # report section
        m = pd.DataFrame([r for r in manifest if r["dataset"] == ds])
        q = pd.DataFrame([r for r in regqc if r["dataset"] == ds])
        t = table[table.case_id.isin(folders)]
        size_gb = sum(f.stat().st_size for c in folders.values() for f in c.iterdir()) / 1e9
        report += [f"## {ds}", "",
                   f"- Cases: {len(folders)} of {len(pats)} patients; size {size_gb:.1f} GB",
                   f"- Format check: {sum(not v for v in problems_by_case.values())} of {len(folders)} cases pass "
                   f"(files, one grid, 1 mm, identity direction, pixel types, lesion counts, Step 9 masks)",
                   f"- Values within [-200, 200]: {len(sample) - bad_values} of {len(sample)} sampled cases",
                   f"- Lesions: malignant {int((t['class'] == 1).sum())}, haemangioma {int((t['class'] == 2).sum())}; "
                   f"ignore regions {int(m.n_ignore_regions.sum()) if len(m) else 0}",
                   "- Malignant lesions by in-plane size (native scan): "
                   + ", ".join(f"{s} mm {int(((t['class'] == 1) & (t.size_stratum == s)).sum())}"
                               for s in ("<10", "10-19", "20-29", ">=30")),
                   f"- Patients by lesion count: {m.lesion_count_group.value_counts().to_dict() if len(m) else {}}"]
        if len(q):
            report += ["- Registration QC on the 1 mm grid (liver Dice with PVP; gradient NCC; coverage):", "",
                       "| Phase | Dice median | Dice 10th pct | Dice < 0.9 | NCC median | Coverage min |",
                       "|---|---|---|---|---|---|"]
            for ph, g in q.groupby("phase"):
                report.append(f"| {ph} | {g.liver_dice.median():.3f} | {g.liver_dice.quantile(0.1):.3f} | "
                              f"{int(g.failed.sum())} | {g.grad_ncc.median():.3f} | {g.coverage.min():.3f} |")
            report += ["", f"- Cases with at least one failed phase (flagged, kept): {int(m.registration_failed.sum())}"]
        report.append("")
        # pictures
        pic_dir = rel / "qc" / "overlays" / ds
        pic_dir.mkdir(parents=True, exist_ok=True)
        index = []
        for j, cid in enumerate(random.Random(1).sample(sorted(folders), min(args.pictures, len(folders))), 1):
            png = pic_dir / f"{ds}_{j:02d}.png"
            picture(folders[cid], png)
            index.append({"picture": png.name, "case_id": cid})
        pd.DataFrame(index).to_csv(pic_dir / "pictures_index.csv", index=False)
        print(f"{ds}: {len(folders)} cases, {sum(bool(v) for v in problems_by_case.values())} with problems, "
              f"{len(index)} pictures", flush=True)
    pd.DataFrame(manifest).to_csv(rel / "manifest.csv", index=False)
    pd.concat(lesions).to_csv(rel / "lesions.csv", index=False)
    pd.DataFrame(exclusion, columns=["case_id", "dataset", "stage", "reason", "excluded_from_split"]).to_csv(
        rel / "exclusion.csv", index=False)
    pd.DataFrame(regqc).to_csv(rel / "qc" / "registration_qc.csv", index=False)
    ex = pd.DataFrame(exclusion, columns=["case_id", "dataset", "stage", "reason", "excluded_from_split"])
    report += ["## Exclusions", "", f"- Excluded from the split: {int(ex.excluded_from_split.sum())}",
               f"- Flagged and kept (failed registration of at least one phase): "
               f"{ex[~ex.excluded_from_split.astype(bool)].case_id.nunique()} cases", "",
               "Crops come from the automatic PVP liver mask, never from a lesion mask (steps/standardise.py)."]
    (rel / "qc" / "qc_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"done: {len(manifest)} cases, {sum(len(x) for x in lesions)} lesions, {len(exclusion)} exclusion rows")


if __name__ == "__main__":
    main()
