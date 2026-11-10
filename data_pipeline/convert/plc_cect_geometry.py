#!/usr/bin/env python3
"""PLC-CECT geometry fix (PLAN.md Step 4): true voxel size, slice order and HU for all 1,444 released volumes.

The released headers say 1 x 1 x 1 mm, 1,442 volumes are stored upside down, and voxels hold 0-255. The geometry
comes from LiverPreprocessing stage 0 (Yifan Li, data_pipeline/third_party/LiverPreprocessing at commit ded5eb74c85f,
reference/plc_geometry.csv): in-plane 0.78125 mm, slice spacing per volume from vertebra distances, slice order from
the liver profile and the vertebra order. Its own functions read the files (colour copies collapsed or averaged),
set the spacing, reverse the slices and map 0-255 to HU; we then add +40 HU (decided 2026-11-07).

Subcommands:
  spotcheck   rerun stage 0 (GPU, envs/totalseg) on --n patients picked across confidence levels and compare with
              the reference table: slice spacing and slice order per volume. Prints counts only.
  apply       write data/PLC-CECT/fixed/<patient>/ct_<phase>.nii.gz (int16 HU), lesion_<phase>.nii.gz and
              liver_<phase>.nii.gz (uint8, the released masks on the same corrected grid) and geometry.json, for every
              patient (phases nc / ap / pvp / dp). Patients already written are skipped.
              --flip-override <csv>: reverse the slice order of the volumes listed there (columns patient, phase)
              against the reference, and write only those patients again; their existing folders are moved to
              <out>_replaced_<date>/ first.

Flip override (decided 2026-11-09, applied 2026-11-10): the reference
flips 1,442 of 1,444 volumes; for the other 2 quick signals disagree, so only volumes that qc/plc_flip_test.py shows
upside down by their anatomy (heart / lungs / kidneys / vertebrae order) are listed in the override file.
geometry.json keeps stage 0's own value as k_axis_flip_stage0 and notes the override. Spacing and HU are unchanged.
Usage: python -I -u plc_cect_geometry.py {spotcheck,apply} --raw <project>/data/PLC-CECT --out <folder> [...]
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

LP = Path(__file__).resolve().parents[1] / "third_party" / "LiverPreprocessing"
LP_COMMIT = "ded5eb74c85f"
sys.path.insert(0, str(LP / "scripts"))
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402
import preprocess_multiphase_ct as P  # noqa: E402

REFERENCE = LP / "reference" / "plc_geometry.csv"
HU_OFFSET = 40.0  # the release was windowed at level 40, not 0 (open item 4, decided 2026-11-07)
OUR_PHASE = {"NC": "nc", "AP": "ap", "PVP": "pvp", "DP": "dp"}
CODE = {"NC": "P", "AP": "C1", "PVP": "C2", "DP": "C3"}


def use_unpacked_release(raw):
    """LiverPreprocessing looks files up in the release zips; ours is unpacked, so point its lookup at the folders."""
    index = {p.name: p for sub in ("ct_files", "liver_mask_files", "mask_files") for p in (raw / sub).glob("*.nii.gz")}
    P.zip_member_index = lambda archives: index
    P.required_zip_source = lambda idx, name: P.path_source(idx[name])


def reference_rows():
    return list(csv.DictReader(REFERENCE.open(encoding="utf-8-sig", newline="")))


def spotcheck(raw, out, n, seed):
    import estimate_plc_geometry as E
    rows = reference_rows()
    worst = {}
    rank = {"high": 0, "medium": 1, "low": 2}
    for r in rows:
        worst[r["patient_id"]] = max(worst.get(r["patient_id"], 0), rank[r["spacing_z_confidence"]])
    rng = random.Random(seed)
    picked = []
    for level in (2, 1, 0):  # some patients with a low, medium and only-high confidence volume each
        pool = sorted(p for p, w in worst.items() if w == level)
        picked += rng.sample(pool, min(len(pool), n // 3 + (1 if level == 2 else 0)))
    picked = picked[:n]
    use_unpacked_release(raw)
    for command in ("measure", "solve"):
        sys.argv = ["estimate_plc_geometry.py", command, "--plc-root", str(raw), "--output-root", str(out),
                    "--patients", *picked]
        E.main()
    written = out / "manifests" / "plc_geometry.csv"
    mine = {(r["patient_id"], r["phase"]): r for r in csv.DictReader(written.open(encoding="utf-8-sig"))}
    ref = {(r["patient_id"], r["phase"]): r for r in rows if r["patient_id"] in picked}
    ratios, flips_same, missing = [], 0, 0
    for key, r in ref.items():
        m = mine.get(key)
        if m is None:
            missing += 1
            continue
        ratios.append(float(m["spacing_z_mm"]) / float(r["spacing_z_mm"]))
        flips_same += m["k_axis_flip"] == r["k_axis_flip"]
    ratios = np.array(ratios)
    print(f"spot check: {len(picked)} patients, {len(ref)} volumes ({missing} not measured here)")
    print(f"slice order same as the reference: {flips_same} of {len(ratios)}")
    print(f"slice spacing here / reference: median {np.median(ratios):.4f}, min {ratios.min():.4f}, "
          f"max {ratios.max():.4f}; within 1 %: {int((abs(ratios - 1) <= 0.01).sum())} of {len(ratios)}")


def fixed_image(source, phase_geometry, temp, name, kind):
    """One volume read and corrected the LiverPreprocessing way; kind 'ct' -> int16 HU + 40, else uint8 0/1."""
    image, details = P.read_scalar_image(P.materialize(source, temp, name))
    image = P.apply_phase_geometry(image, phase_geometry)
    if kind == "ct":
        hu = P.plc_uint8_to_hu(image) + HU_OFFSET
        out = sitk.GetImageFromArray(np.rint(sitk.GetArrayFromImage(hu)).astype(np.int16))
        out.CopyInformation(hu)
        return out, details
    mask = sitk.Cast(image > 0.5, sitk.sitkUInt8)
    mask.CopyInformation(image)
    return mask, details


def apply_patient(case, out_root, flip_override=frozenset()):
    """Write the corrected volumes of one patient through a temporary folder; return a short status."""
    final = out_root / case["patient_id"]
    if final.exists():
        return "skipped"
    tmp = out_root / (case["patient_id"] + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    record = {"patient_id": case["patient_id"], "diagnosis": case["diagnosis"],
              "geometry_source": f"LiverPreprocessing {LP_COMMIT} reference/plc_geometry.csv",
              "hu_mapping": "HU = value / 255 * 400 - 200 + 40", "phases": {}}
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        for phase in P.PHASES:
            g = dict(case["phase_geometry"][phase])
            ours = OUR_PHASE[phase]
            stage0_flip = g["k_axis_flip"]
            if (case["patient_id"], ours) in flip_override:
                g["k_axis_flip"] = not stage0_flip
                record.setdefault("notes", []).append(
                    f"{ours}: slice order reversed against stage 0 (anatomy test, qc/plc_flip_test.py)")
            ct, details = fixed_image(case["phase_sources"][phase], g, temp, f"ct-{phase}", "ct")
            sitk.WriteImage(ct, str(tmp / f"ct_{ours}.nii.gz"), useCompression=True)
            liver_source = case["liver_sources"][phase] if case["liver_mask_origin"][phase] == "released" else None
            if liver_source:
                liver, _ = fixed_image(liver_source, g, temp, f"liver-{phase}", "mask")
                if liver.GetSize() == ct.GetSize():
                    liver.CopyInformation(ct)
                    sitk.WriteImage(liver, str(tmp / f"liver_{ours}.nii.gz"), useCompression=True)
            for source in case["tumour_sources_by_phase"][phase]:
                lesion, _ = fixed_image(source, g, temp, f"lesion-{phase}", "mask")
                if lesion.GetSize() != ct.GetSize():
                    if phase == "PVP":  # the main lesion mask must fit; the other phases' masks are for QC only
                        raise ValueError(f"{phase} lesion mask size {lesion.GetSize()} != CT {ct.GetSize()}")
                    record.setdefault("notes", []).append(
                        f"{ours} lesion mask has {lesion.GetSize()[2]} slices, the CT {ct.GetSize()[2]}: not written")
                    continue
                lesion.CopyInformation(ct)
                sitk.WriteImage(lesion, str(tmp / f"lesion_{ours}.nii.gz"), useCompression=True)
            record["phases"][ours] = {
                "source_code": CODE[phase], "size": list(ct.GetSize()), "spacing_mm": list(ct.GetSpacing()),
                "k_axis_flip": g["k_axis_flip"], "k_axis_flip_stage0": stage0_flip,
                "spacing_z_confidence": g["spacing_z_confidence"],
                "spacing_z_source": g["spacing_z_source"], "orientation_source": g["orientation_source"],
                "colour_channels": details.get("vector_fix"), "geometry_version": g["geometry_version"]}
    (tmp / "geometry.json").write_text(json.dumps(record, indent=1))
    os.replace(tmp, final)
    return "done"


def apply(raw, out, workers, limit, flip_override_csv=None):
    use_unpacked_release(raw)
    cases = P.build_plc_cases(raw, REFERENCE)
    flip_override = frozenset()
    if flip_override_csv:
        flip_override = frozenset((r["patient"], r["phase"]) for r in csv.DictReader(flip_override_csv.open()))
        cases = [c for c in cases if any(c["patient_id"] == p for p, _ in flip_override)]
        backup = out.parent / f"{out.name}_replaced_{time.strftime('%Y-%m-%d')}"
        backup.mkdir(exist_ok=True)
        for c in cases:
            if (out / c["patient_id"]).exists():
                os.replace(out / c["patient_id"], backup / c["patient_id"])
        print(f"flip overrides: {len(cases)} patients; their previous folders moved to {backup.name}/")
    if limit:
        cases = cases[:limit]
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(cases)} patients, {workers} workers", flush=True)
    counts, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=use_unpacked_release, initargs=(raw,)) as pool:
        futures = {pool.submit(apply_patient, case, out, flip_override): case for case in cases}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                status = fut.result()
            except Exception as e:  # unreadable file, mask on another grid
                status = f"failed: {type(e).__name__}: {e}"
                print(f"[{k}/{len(cases)}] FAILED: {type(e).__name__}: {e}", flush=True)
            counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
            if k % 25 == 0 or k == len(cases):
                print(f"[{k}/{len(cases)}] {counts}, {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"done: {counts}")
    if counts.get("failed"):
        sys.exit(f"{counts['failed']} patients failed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["spotcheck", "apply"])
    ap.add_argument("--raw", type=Path, required=True, help="PLC-CECT folder (ct_files/, mask_files/, ...)")
    ap.add_argument("--out", type=Path, required=True, help="spotcheck: output root; apply: <raw>/fixed")
    ap.add_argument("--n", type=int, default=10, help="spotcheck: number of patients")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="apply: patients in parallel")
    ap.add_argument("--limit", type=int, help="apply: only the first N patients (test run)")
    ap.add_argument("--flip-override", type=Path,
                    help="apply: CSV (patient, phase) of volumes to reverse against stage 0; rewrites only those patients")
    args = ap.parse_args()
    if args.command == "spotcheck":
        spotcheck(args.raw, args.out, args.n, args.seed)
    else:
        apply(args.raw, args.out, args.workers, args.limit, args.flip_override)


if __name__ == "__main__":
    main()
