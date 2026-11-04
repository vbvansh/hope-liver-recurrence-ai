#!/usr/bin/env python3
"""Dataset summary across all five datasets (PLAN.md Step 1, Yifan's request 2026-11-01).

Built from the inventories (inventory.py): per dataset the patients found vs the publication, phases per patient and
which are missing, lesion masks and on which phase, lesions per standard class, slice thickness, patients that
cannot be used (a "BLOCK" note) and the usable count, with the expected train / validation / test sizes for the
development sets. It also estimates Plan v2's < 10 mm reporting rule (internal test >= 50 such lesions, each external
set >= 30). Step 10 regenerates it with the QC exclusions.

Output (--out folder): dataset_summary.csv (one row per dataset) and dataset_summary.md (counts only, for the team).
Usage: python -I dataset_summary.py --inventory <project>/outputs/inventory --out <project>/outputs/dataset_summary
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ["MCT-LTDiag", "PLC-CECT", "WAW-TACE", "TCIA-CRLM", "LiTS"]
EXPECTED = {"MCT-LTDiag": 517, "PLC-CECT": 361, "WAW-TACE": 233, "TCIA-CRLM": 197, "LiTS": 131}
PHASES = ["nc", "ap", "pvp", "dp"]
TEXT = ["dataset", "patient_id", "role", "diagnosis", "phases_present", "missing_phases", "lesion_mask_phases",
        "liver_mask_phases", "notes"]
SPLIT = {"train": 0.7, "val": 0.1, "test": 0.2}
SMALL_MM, SMALL_MIN = 10, {"internal_test": 50, "external": 30}  # Plan v2 reporting rule for the < 10 mm stratum
LITS_SMALL_MAX_SLICE_MM = 2  # LiTS: < 10 mm lesions count only on slices <= 2 mm


def summarise(name, inv, les):
    blocked = inv.notes.str.contains("BLOCK")
    if inv.role.iloc[0] == "development":
        blocked |= ~inv.phases_present.str.contains("pvp")
    usable = int((~blocked).sum())
    reasons = (inv.notes[blocked].str.findall(r"BLOCK: ([^;]+)").explode().value_counts())
    notes = (inv.notes.str.findall(r"NOTE: ([^;]+)").explode().dropna()
             .str.replace(r"\d+(\.\d+)?", "#", regex=True).str.strip().value_counts())
    patients_noted = {k: int(inv.notes.str.replace(r"\d+(\.\d+)?", "#", regex=True).str.contains(k, regex=False).sum())
                      for k in notes.index}
    patterns = inv.missing_phases.replace("", "none").value_counts()
    slice_col = "slice_mm_pvp" if "slice_mm_pvp" in inv else next(c for c in inv if c.startswith("slice_mm_"))
    les_ok = les[les.patient_id.isin(inv.patient_id[~blocked])] if len(les) else les
    c1 = les_ok[les_ok["class"] == 1] if len(les_ok) else les_ok
    small = np.nan
    if len(c1) and c1.inplane_max_diameter_mm.notna().any():
        if name == "LiTS":
            thin = inv.patient_id[inv.slice_mm_pvp <= LITS_SMALL_MAX_SLICE_MM]
            c1 = c1[c1.patient_id.isin(thin)]
        small = int((c1.inplane_max_diameter_mm < SMALL_MM).sum())
    row = {"dataset": name, "role": inv.role.iloc[0],
           "patients_expected": EXPECTED[name], "patients_found": len(inv),
           "diagnoses": "; ".join(f"{k} {v}" for k, v in inv.diagnosis.value_counts().items()),
           **{f"with_{p}": int(inv.phases_present.str.contains(p).sum()) for p in PHASES},
           "all_four_phases": int((inv.n_phases == 4).sum()),
           "missing_phase_patterns": "; ".join(f"{k.replace('|', '+')} {v}" for k, v in patterns.items()),
           "patients_with_lesion_mask": int(inv.lesion_mask_phases.ne("").sum()),
           "lesion_mask_on": "; ".join(f"{k.replace('|', '+')} {v}" for k, v in
                                      inv.lesion_mask_phases.replace("", "none").value_counts().items()),
           "lesions_class1": int(inv.lesions_class1.sum()), "lesions_class2": int(inv.lesions_class2.sum()),
           "lesions_class3": int(inv.lesions_class3.sum()),
           "class1_lt10mm_provisional": small,
           "slice_mm_min": inv[slice_col].min(), "slice_mm_median": inv[slice_col].median(),
           "slice_mm_max": inv[slice_col].max(),
           "blocked": int(blocked.sum()), "blocked_reasons": "; ".join(f"{k} {v}" for k, v in reasons.items()),
           "notes": "; ".join(f"{k} ({v} patients)" for k, v in patients_noted.items()),
           "usable": usable}
    if row["role"] == "development":
        row.update({f"expected_{k}": round(usable * v) for k, v in SPLIT.items()})
    else:
        row["expected_test"] = usable
    return row


def markdown(df):
    def cell(v):
        return "–" if pd.isna(v) or v == "" else (f"{v:g}" if isinstance(v, float) else str(v))

    table = [("Role", lambda r: r.role),
             ("Patients found / in the paper", lambda r: f"{r.patients_found} / {r.patients_expected}"),
             ("With NC / AP / PVP / DP", lambda r: f"{r.with_nc} / {r.with_ap} / {r.with_pvp} / {r.with_dp}"),
             ("All four phases", lambda r: r.all_four_phases),
             ("With a lesion mask", lambda r: r.patients_with_lesion_mask),
             ("Lesions: malignant (1)", lambda r: r.lesions_class1),
             ("Lesions: haemangioma (2)", lambda r: r.lesions_class2),
             ("Regions: ignore (3)", lambda r: r.lesions_class3),
             ("Malignant lesions < 10 mm (provisional)", lambda r: cell(r.class1_lt10mm_provisional)),
             ("Slice spacing, mm (min / median / max)",
              lambda r: f"{cell(r.slice_mm_min)} / {cell(r.slice_mm_median)} / {cell(r.slice_mm_max)}"),
             ("Not usable yet", lambda r: r.blocked),
             ("**Usable**", lambda r: f"**{r.usable}**"),
             ("Expected train / val / test", lambda r: (f"{r.expected_train} / {r.expected_val} / {r.expected_test}"
                                                       if r.role == "development" else f"test {r.expected_test}"))]
    lines = ["# Dataset summary", "",
             "First version, from the Step 1 inventories. Counts only; per-patient tables stay on the cluster.", "",
             "| | " + " | ".join(df.dataset) + " |", "|---|" + "---|" * len(df)]
    lines += [f"| {label} | " + " | ".join(cell(get(r)) for r in df.itertuples()) + " |" for label, get in table]
    lines += ["", "## Details per dataset", ""]
    for r in df.itertuples():
        lines += [f"**{r.dataset}**", f"- Diagnoses: {r.diagnoses}",
                  f"- Missing phases: {r.missing_phase_patterns.replace('none', 'none missing')}",
                  f"- Lesion masks drawn on: {r.lesion_mask_on}"]
        if r.blocked_reasons:
            lines.append(f"- Not usable yet: {r.blocked_reasons}")
        if r.notes:
            lines.append(f"- Notes: {r.notes}")
        lines.append("")
    lines += ["## The < 10 mm rule (Plan v2)", ""]
    dev = df[df.role == "development"]
    if dev.class1_lt10mm_provisional.notna().any():
        est = SPLIT["test"] * dev.class1_lt10mm_provisional.sum()
        lines.append(f"- < 10 mm rule (Plan v2): internal test ≈ 20 % of the measurable development lesions < 10 mm "
                     f"= about {est:.0f} (needs >= {SMALL_MIN['internal_test']}; PLC-CECT sizes not yet measurable).")
    for r in df[df.role == "external"].itertuples():
        if pd.notna(r.class1_lt10mm_provisional):
            ok = "meets" if r.class1_lt10mm_provisional >= SMALL_MIN["external"] else "below"
            lines.append(f"- < 10 mm rule, {r.dataset}: {int(r.class1_lt10mm_provisional)} lesions ({ok} "
                         f"the {SMALL_MIN['external']} needed).")
    lines += ["", "## Assumptions", "",
              "PLC-CECT P/C1/C2/C3 and WAW-TACE 0/1/2/3 read as NC/AP/PVP/DP; TCIA-CRLM and LiTS taken "
              "as PVP; MCT-LTDiag lesions follow the proposed blob rule (largest piece + pieces the authors' rule "
              "keeps = lesions, the rest = class 3 ignore; pending Yifan); PLC-CECT lesion sizes wait for the "
              "geometry recovery (Step 4); diameters are provisional (outer voxel surface, Step 7 fixes the "
              "convention); LiTS < 10 mm lesions only on slices <= 2 mm; expected split sizes are 70 / 10 / 20 % "
              "of the usable development patients."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inventory", type=Path, required=True, help="folder with inventory_*.csv and lesions_*.csv")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in DATASETS:
        inv_path = args.inventory / f"inventory_{name}.csv"
        if not inv_path.exists():
            print(f"{name}: no inventory yet, skipped")
            continue
        inv = pd.read_csv(inv_path, dtype={c: str for c in TEXT}, keep_default_na=False)
        inv = inv.apply(lambda col: col if col.name in TEXT else pd.to_numeric(col.replace("", np.nan)))
        les_path = args.inventory / f"lesions_{name}.csv"
        les = pd.read_csv(les_path, dtype={"patient_id": str}) if les_path.stat().st_size > 1 else pd.DataFrame()
        rows.append(summarise(name, inv, les))
    df = pd.DataFrame(rows)
    for col in [c for c in df if c.startswith("expected_") or c == "class1_lt10mm_provisional"]:
        df[col] = df[col].astype("Int64")  # whole numbers, blank where they don't apply
    df.to_csv(args.out / "dataset_summary.csv", index=False)
    (args.out / "dataset_summary.md").write_text(markdown(df))
    print(markdown(df))


if __name__ == "__main__":
    main()
