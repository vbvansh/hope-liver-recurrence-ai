#!/usr/bin/env python3
"""Compare our 5 public datasets (releases v1.1 + v2) with the liver part of CancerVerse, counts only.

What: one table per question, same definitions for every dataset: patients, examinations (dates), CT volumes and
phases, native slice thickness, lesions by size (in-plane max diameter between voxel centres), and for CancerVerse
the longitudinal pairs (two consecutive scan dates of one patient) a follow-up model would learn from. Adds the
95 % confidence-interval half-width a test set of n small lesions gives for a sensitivity around 0.85, to judge
whether a test set is large enough.

Why: to judge whether CancerVerse's liver data is large enough to build and test a model, next to what we have.

Inputs: processed/v1/{manifest,lesions}.csv + splits/splits.csv (v1.1), processed/v2_external/{manifest,lesions}.csv
+ lists/test_all.txt, data/CancerVerse/metadata/CancerVerse_dataset_metadata.csv,
outputs/cancerverse/liver_masks/{scans,lesions}.csv. Output: outputs/cancerverse/comparison.md. Login node, seconds.

Lesion definitions: ours = lesions in the release of classes 1 + 2 (team rule, 26 Nov 2026); CancerVerse = liver
mask pieces without fragments (>= 2 slices and in-plane >= 5 mm), since its masks contain many tiny pieces.
"""
import argparse
import math
import os

import numpy as np
import pandas as pd

ORDER = ["<10", "10-19", "20-29", ">=30"]
P, A, ID = "Patient ID", "Encrypted Accession Number", "CancerVerse ID"
GB_PER_CT = 3.04e3 / 24422  # card: 3.04 TB for 24,422 CT volumes -> average GB per CT (estimate)


def stratum(d):
    return "<10" if d < 10 else "10-19" if d < 20 else "20-29" if d < 30 else ">=30"


def ci_half(n, p=0.85):
    return 1.96 * math.sqrt(p * (1 - p) / n) if n else float("nan")


def ours(root):
    """One dict per public dataset (patients in the v1.1 split or the v2 test lists)."""
    rows = []
    for rel, keep_file in (("v1", "splits/splits.csv"), ("v2_external", "lists/test_all.txt")):
        man = pd.read_csv(os.path.join(root, "processed", rel, "manifest.csv"))
        les = pd.read_csv(os.path.join(root, "processed", rel, "lesions.csv"))
        kp = os.path.join(root, "processed", rel, keep_file)
        keep = set(pd.read_csv(kp).case_id) if kp.endswith(".csv") else set(open(kp).read().split())
        man = man[man.case_id.isin(keep)]
        les = les[les.case_id.isin(keep) & les["class"].isin([1, 2]) & (les["in_release"].astype(str) == "True")]
        les = les.assign(stratum=les["inplane_max_diameter_mm"].apply(stratum))
        for ds, m in man.groupby("dataset"):
            l = les[les.dataset == ds]
            nph = m["phases_present"].astype(str).str.split("|").apply(len)
            th = pd.to_numeric(m["native_slice_mm"], errors="coerce")
            r = {"dataset": ds, "patients": len(m), "exams": len(m), "dates_per_patient": "1",
                 "ct_volumes": int(nph.sum()), "phases_per_exam": "%d–%d" % (nph.min(), nph.max()) if nph.min() != nph.max() else str(nph.min()),
                 "slice_mm": "%.2f" % th.median(), "lesions": len(l), "malignant": int((l["class"] == 1).sum()),
                 "patients_with_10_19": l.loc[l.stratum == "10-19", "case_id"].nunique(),
                 "longitudinal_patients": 0, "pairs_both_outlined": 0}
            for s in ORDER:
                r[s] = int((l.stratum == s).sum())
            rows.append(r)
    return rows


def cancerverse(root):
    meta = pd.read_csv(os.path.join(root, "data", "CancerVerse", "metadata", "CancerVerse_dataset_metadata.csv"),
                       low_memory=False)
    md = os.path.join(root, "outputs", "cancerverse", "liver_masks")
    scans = pd.read_csv(os.path.join(md, "scans.csv"))
    lesions = pd.read_csv(os.path.join(md, "lesions.csv"))
    scans = scans[scans["mask"] == "liver_lesion"]
    lesions = lesions[lesions["mask"] == "liver_lesion"]
    meta["date"] = pd.to_datetime(meta["exam_date"], errors="coerce")
    m = meta.merge(scans[["cv_id", "voxels", "spacing_mm", "slice_axis"]], left_on=ID, right_on="cv_id", how="inner")
    m["out"] = m["voxels"] > 0
    m["pv"] = m["phase"].isin(["portal_venous", "venous"])
    codes = meta.groupby(P)["icd10_code"].apply(lambda s: [str(c) for c in s.dropna()])
    hcc = set(p for p, cs in codes.items() if any(c.startswith("C22.0") for c in cs))
    liver_p = set(m.loc[m["out"], P])
    lp = m[m[P].isin(liver_p)]
    les = lesions.merge(m[[ID, P, "date"]], left_on="cv_id", right_on=ID)
    les = les[(les["slices"] >= 2) & (les["inplane_max_diameter_mm"] >= 5)]
    les = les.assign(stratum=les["inplane_max_diameter_mm"].apply(stratum))
    first_scan = set(lp[lp["out"]].sort_values("date").groupby(P)[ID].first())
    lf = les[les["cv_id"].isin(first_scan)]
    th = lp[lp["out"]].apply(lambda r: float(str(r["spacing_mm"]).split("x")[int(r["slice_axis"])]), axis=1)

    # longitudinal pairs: consecutive distinct dates of one patient (a date is outlined / portal-venous if any scan is)
    d = lp.dropna(subset=["date"]).groupby([P, "date"]).agg(out=("out", "any"), pv=("pv", "any"),
                                                             pv_out=("out", lambda s: bool((s & lp.loc[s.index, "pv"]).any()))
                                                             ).reset_index().sort_values([P, "date"])
    d["prev_out"] = d.groupby(P)["out"].shift()
    d["prev_pvout"] = d.groupby(P)["pv_out"].shift()
    d["gap"] = d.groupby(P)["date"].diff().dt.days
    pr = d.dropna(subset=["prev_out"])
    kinds = {"both outlined (tumour followed over time)": pr["prev_out"].astype(bool) & pr["out"],
             "earlier without, later with outline (appears; upper bound)": ~pr["prev_out"].astype(bool) & pr["out"],
             "earlier with, later without outline (gone)": pr["prev_out"].astype(bool) & ~pr["out"],
             "neither outlined": ~pr["prev_out"].astype(bool) & ~pr["out"],
             "both outlined, both portal-venous / venous": pr["prev_pvout"].astype(bool) & pr["pv_out"]}
    pairs = {k: (int(v.sum()), pr.loc[v, P].nunique(), pr.loc[v & pr[P].isin(hcc), P].nunique(),
                 float(pr.loc[v, "gap"].median()) if v.any() else float("nan")) for k, v in kinds.items()}
    per = lp[lp["out"]].groupby(P)["date"].nunique()
    long_p = set(per[per >= 2].index)
    r = {"dataset": "CancerVerse liver (outlined)", "patients": len(liver_p), "exams": lp[A].nunique(),
         "dates_per_patient": "1–%d (median %d)" % (lp.groupby(P)["date"].nunique().max(), lp.groupby(P)["date"].nunique().median()),
         "ct_volumes": len(lp), "phases_per_exam": "1", "slice_mm": "%.2f" % th.median(),
         "lesions": len(lf), "malignant": len(lf), "patients_with_10_19": lf.loc[lf.stratum == "10-19", P].nunique(),
         "longitudinal_patients": len(long_p), "pairs_both_outlined": pairs["both outlined (tumour followed over time)"][0]}
    for s in ORDER:
        r[s] = int((lf.stratum == s).sum())
    extra = {"outlined_scans": int(lp["out"].sum()), "all_lesions_no_fragments": len(les),
             "hcc_patients": len(liver_p & hcc), "hcc_longitudinal": len(long_p & hcc),
             "scans_of_longitudinal_patients": int(lp[lp[P].isin(long_p)].shape[0]),
             "outlined_scans_of_longitudinal_patients": int(lp[lp[P].isin(long_p) & lp["out"]].shape[0]),
             "pv_outlined_scans_of_longitudinal_patients": int(lp[lp[P].isin(long_p) & lp["out"] & lp["pv"]].shape[0]),
             "followup_lesions_10_19": int(((les.stratum == "10-19") & ~les["cv_id"].isin(first_scan)).sum())}
    return r, pairs, extra


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.path.abspath(os.path.join(here, "..", "..", "..")))
    args = ap.parse_args()
    pub = ours(args.root)
    cv, pairs, extra = cancerverse(args.root)
    allrows = pub + [cv]
    tot = {k: sum(r[k] for r in pub) for k in ["patients", "exams", "ct_volumes", "lesions", "malignant",
                                                "patients_with_10_19"] + ORDER}
    out = []
    w = out.append
    w("# Our 5 public datasets vs the liver part of CancerVerse (counts only)\n")
    w("Ours: patients in release v1.1's split (MCT-LTDiag, PLC-CECT) and release v2's test lists (WAW-TACE, TCIA-CRLM, LiTS); "
      "lesions = classes 1 + 2 in the release. CancerVerse: patients with a liver tumour outline on ≥ 1 scan; lesions = liver "
      "mask pieces with ≥ 2 slices and in-plane ≥ 5 mm (tiny pieces removed), counted on each patient's **first** outlined scan "
      "so a tumour followed over time is counted once. Sizes: in-plane max diameter between voxel centres.\n")
    w("## 1. Side by side\n")
    w("| Dataset | Patients | Exams (dates) | Dates per patient | CT volumes | Phases per exam | Slice (median mm) | Lesions | "
      "< 10 mm | 10–19 mm | 20–29 mm | ≥ 30 mm | Patients with a 10–19 mm lesion | Patients with outlines on ≥ 2 dates |")
    w("|---|" + "---|" * 13)
    for r in allrows:
        w("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["dataset"], f"{r['patients']:,}", f"{r['exams']:,}", r["dates_per_patient"], f"{r['ct_volumes']:,}",
            r["phases_per_exam"], r["slice_mm"], f"{r['lesions']:,}", *[f"{r[s]:,}" for s in ORDER],
            f"{r['patients_with_10_19']:,}", f"{r['longitudinal_patients']:,}"))
    w("| **All 5 public** | **%s** | **%s** | 1 | **%s** | 1–4 | — | **%s** | %s | %s | %s | %s | %s | **0** |" % (
        f"{tot['patients']:,}", f"{tot['exams']:,}", f"{tot['ct_volumes']:,}", f"{tot['lesions']:,}",
        *[f"{tot[s]:,}" for s in ORDER], f"{tot['patients_with_10_19']:,}"))
    w("\nCancerVerse liver: %s outlined scans; %s lesions on all outlined scans (without fragments); %s HCC patients "
      "(ICD C22.0) with outlines.\n" % (f"{extra['outlined_scans']:,}", f"{extra['all_lesions_no_fragments']:,}",
                                         f"{extra['hcc_patients']:,}"))
    w("## 2. Longitudinal pairs in CancerVerse (two consecutive scan dates of one patient)\n")
    w("| Pair type | Pairs | Patients | of which HCC | Median gap (days) |\n|---|---|---|---|---|")
    for k, (n, npat, nh, gap) in pairs.items():
        w("| %s | %s | %s | %s | %s |" % (k, f"{n:,}", f"{npat:,}", f"{nh:,}", "—" if np.isnan(gap) else "%d" % gap))
    w("\nOur 5 public datasets: **0** pairs (one examination per patient).\n")
    w("Patients with outlines on ≥ 2 dates: %s (HCC %s). Their scans: %s in total, %s outlined, %s outlined portal-venous / "
      "venous. 10–19 mm lesions on follow-up (not first) outlined scans: %s.\n" % (
          f"{cv['longitudinal_patients']:,}", f"{extra['hcc_longitudinal']:,}", f"{extra['scans_of_longitudinal_patients']:,}",
          f"{extra['outlined_scans_of_longitudinal_patients']:,}", f"{extra['pv_outlined_scans_of_longitudinal_patients']:,}",
          f"{extra['followup_lesions_10_19']:,}"))
    w("## 3. How precisely can a test set measure small-lesion sensitivity?\n")
    w("95 % confidence-interval half-width for a sensitivity of about 0.85 measured on n lesions (normal approximation; "
      "lesions of one patient are not independent, so real intervals are somewhat wider):\n")
    w("| Test set | n lesions | ± (95 % CI) |\n|---|---|---|")
    for name, n in (("Our internal test, < 20 mm (v1.1: 78 malignant + 9 haemangioma)", 87),
                    ("Our internal test, 10–19 mm malignant only", 78),
                    ("20 % of CancerVerse liver patients, 10–19 mm (first scans)", round(0.2 * cv["10-19"])),
                    ("A test set of 300 small lesions", 300)):
        w("| %s | %s | ± %.3f |" % (name, f"{n:,}", ci_half(n)))
    w("\n## 4. Storage needed to use it (estimate)\n")
    w("Average CT volume ≈ %.0f MB (card: 3.04 TB / 24,422). Scans of the %s longitudinal liver patients: %s volumes ≈ %.0f GB; "
      "outlined portal-venous / venous scans only: %s ≈ %.0f GB; all scans of all %s outlined patients: %s ≈ %.0f GB. Free on our "
      "project storage on 25 Nov: 108 GB.\n" % (
          GB_PER_CT * 1000, f"{cv['longitudinal_patients']:,}", f"{extra['scans_of_longitudinal_patients']:,}",
          extra["scans_of_longitudinal_patients"] * GB_PER_CT, f"{extra['pv_outlined_scans_of_longitudinal_patients']:,}",
          extra["pv_outlined_scans_of_longitudinal_patients"] * GB_PER_CT, f"{cv['patients']:,}", f"{cv['ct_volumes']:,}",
          cv["ct_volumes"] * GB_PER_CT))
    md = "\n".join(out) + "\n"
    path = os.path.join(args.root, "outputs", "cancerverse", "comparison.md")
    with open(path + ".part", "w") as f:
        f.write(md)
    os.replace(path + ".part", path)
    print(md)


if __name__ == "__main__":
    main()
