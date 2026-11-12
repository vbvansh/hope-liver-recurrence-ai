#!/usr/bin/env python3
"""Per-phase decisions that change how Step 8 builds a case (PLAN.md decisions of 2026-11-12).

Why: three decisions change single phases of single patients, and Step 8 must apply them the same way every run:
  1. MCT-LTDiag re-registration (open item 11, Option B): for each phase the authors' registration flagged (liver
     Dice < 0.9 on the 1 mm grid), keep our registration (steps/register_mct.py) if it lines the liver up better:
     inside-the-liver edge agreement at least 0.05 higher and liver Dice not lower. Otherwise the authors' stays.
  2. A phase where even the better version has liver Dice < 0.5 and inside-the-liver agreement < 0.1 is treated as
     missing (one MCT NC phase, broken in the release; confirmed visually on 2026-11-12).
  3. PLC-CECT phases that are pixel-identical copies of another phase in the release are treated as missing (open item
     12, Option A, 2026-11-12): the DP of 4 patients (list from the pixel comparison of all same-size phase pairs).

Output: <out> (CSV: dataset, patient, phase, action, reason) with action "use_our_registration" or "missing".
Patient IDs stay in the file (on disk, not in git); the log shows counts only.
Usage: python -I phase_overrides.py --project <project> --duplicates <project>/outputs/plc_check/duplicate_phases.tsv
           --out <project>/outputs/phase_overrides.csv
"""
import argparse
from pathlib import Path

import pandas as pd

EDGE_GAIN, BROKEN_DICE, BROKEN_EDGE = 0.05, 0.5, 0.1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--duplicates", type=Path, required=True, help="TSV patient, phase_a, phase_b (pixel-identical)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    # 1-2: MCT-LTDiag flagged phases
    m = pd.read_csv(args.project / "processed" / "v1" / "manifest.csv")
    m = m[m.dataset == "MCT-LTDiag"]
    flagged = {(c.split("_", 1)[1], ph) for c, phs in zip(m.case_id, m.registration_failed_phases.fillna(""))
               for ph in str(phs).split("|") if ph}
    q = pd.concat([pd.read_csv(f, dtype={"patient": str})
                   for f in sorted((args.project / "outputs" / "registration" / "MCT-LTDiag" / "qc").glob("task_*.csv"))])
    q = q[q.status == "ok"].groupby(["patient", "phase"]).tail(1)
    for r in q.itertuples():
        if (r.patient, r.phase) not in flagged:
            continue
        best_dice = max(r.liver_dice_before, r.liver_dice_after)
        best_edge = max(r.liver_edge_before, r.liver_edge_after)
        note = (f"authors' liver Dice {r.liver_dice_before:.3f} / liver edges {r.liver_edge_before:.3f}; "
                f"ours {r.liver_dice_after:.3f} / {r.liver_edge_after:.3f}")
        if best_dice < BROKEN_DICE and best_edge < BROKEN_EDGE:
            rows.append(("MCT-LTDiag", r.patient, r.phase, "missing", "phase broken in the release; " + note))
        elif r.liver_edge_after >= r.liver_edge_before + EDGE_GAIN and r.liver_dice_after >= r.liver_dice_before:
            rows.append(("MCT-LTDiag", r.patient, r.phase, "use_our_registration", note))
    # 3: PLC-CECT duplicated phases: the copy (always DP here) is treated as missing
    dup = pd.read_csv(args.duplicates, sep="\t", dtype=str)
    for r in dup.itertuples():
        copy = "dp" if "dp" in (r.phase_a, r.phase_b) else r.phase_b
        source = r.phase_a if copy == r.phase_b else r.phase_b
        rows.append(("PLC-CECT", r.patient, copy, "missing", f"pixel-identical copy of {source.upper()} in the release"))
    out = pd.DataFrame(rows, columns=["dataset", "patient", "phase", "action", "reason"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"done: {len(out)} overrides: {out.groupby(['dataset', 'action']).size().to_dict()}; "
          f"cases affected: {out.groupby('dataset').patient.nunique().to_dict()}")


if __name__ == "__main__":
    main()
