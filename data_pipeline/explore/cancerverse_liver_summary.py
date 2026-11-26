#!/usr/bin/env python3
"""Summarise the liver part of CancerVerse: mask results + metadata (dates, phases, ICD-10), counts only.

What: joins outputs/cancerverse/liver_masks/{scans,lesions,members}.csv (from cancerverse_liver_masks.py) with
data/CancerVerse/metadata/CancerVerse_dataset_metadata.csv and reports: how many scans / patients carry a liver
tumour outline, their ICD-10 diagnoses, follow-up (outlines on several dates, tumours appearing after an
outline-free scan), phases, slice thickness, lesions per scan and lesion sizes (in-plane max diameter between voxel
centres, our release convention), demographics.

Why: the metadata alone cannot say which scans have liver outlines; this is the complete liver picture.

Outputs (in --out): summary.md (counts only) and liver_patients.csv (one row per patient with a liver outline:
dataset patient ID, ICD group, scans, dated scans with / without outline, first / last outline date offset in
days). Login node, seconds.

Usage: python -I data_pipeline/explore/cancerverse_liver_summary.py [--root <project storage>]
"""
import argparse
import os

import numpy as np
import pandas as pd

P, A, ID = "Patient ID", "Encrypted Accession Number", "CancerVerse ID"


def icd_group(codes):
    codes = [str(c) for c in codes]
    if any(c.startswith("C22.0") for c in codes):
        return "HCC (C22.0)"
    if any(c.startswith("C22.1") for c in codes):
        return "intrahepatic cholangiocarcinoma (C22.1)"
    if any(c.startswith("C22") for c in codes):
        return "other primary liver cancer (C22.x)"
    if any(c.startswith("C78.7") for c in codes):
        return "liver metastases (C78.7)"
    if any(c.startswith("C") for c in codes):
        return "other cancer code only"
    if codes:
        return "non-cancer code only"
    return "no ICD-10 code"


def stratum(d):
    return "<10" if d < 10 else "10-19" if d < 20 else "20-29" if d < 30 else ">=30"


def table(series, name):
    vc = series.value_counts()
    return "| %s | Count |\n|---|---|\n" % name + "".join("| %s | %d |\n" % (k, v) for k, v in vc.items())


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.path.abspath(os.path.join(here, "..", "..", "..")))
    ap.add_argument("--masks-dir", help="mask results folder (default <root>/outputs/cancerverse/liver_masks)")
    args = ap.parse_args()
    mdir = args.masks_dir or os.path.join(args.root, "outputs", "cancerverse", "liver_masks")
    meta = pd.read_csv(os.path.join(args.root, "data", "CancerVerse", "metadata",
                                    "CancerVerse_dataset_metadata.csv"), low_memory=False)
    scans = pd.read_csv(os.path.join(mdir, "scans.csv"))
    lesions = pd.read_csv(os.path.join(mdir, "lesions.csv"))
    members = pd.read_csv(os.path.join(mdir, "members.csv"))
    scans = scans[scans["mask"] == "liver_lesion"]
    lesions = lesions[lesions["mask"] == "liver_lesion"]

    meta["date"] = pd.to_datetime(meta["exam_date"], errors="coerce")
    m = meta.merge(scans[["cv_id", "voxels", "lesions", "spacing_mm", "slice_axis"]], left_on=ID, right_on="cv_id",
                   how="left")
    m["has_file"] = m["cv_id"].notna()
    m["liver"] = m["voxels"].fillna(0) > 0
    codes = meta.groupby(P)["icd10_code"].apply(lambda s: list(s.dropna()))
    group = codes.apply(icd_group)
    liver_pats = set(m.loc[m["liver"], P])
    out = []
    w = out.append

    w("# CancerVerse: the liver part (counts only)\n")
    w("Sources: CancerVerse metadata (Hugging Face, SHA-256 checked) and the label archive "
      "`CancerVerse_Label.tar.gz` (authors' site), read as a stream by `cancerverse_liver_masks.py`. "
      "Liver outline = non-empty `liver_lesion.nii.gz`. Lesion = 3D connected component (26-connectivity); "
      "sizes on the scan's own grid, diameters between voxel centres (our release convention).\n")
    w("## 1. Archive and masks\n")
    w("| Item | Value |\n|---|---|")
    w("| Scans in the metadata | %d (patients %d) |" % (len(meta), meta[P].nunique()))
    w("| Mask file names in the archive | %d distinct, %d files |" % (len(members), members["count"].sum()))
    w("| Liver mask files read | %d (scans in the metadata without one: %d) |" % (len(scans), int((~m["has_file"]).sum())))
    w("| **Scans with a liver outline** | **%d** (card: 6,540) |" % int(m["liver"].sum()))
    w("| **Patients with a liver outline on ≥ 1 scan** | **%d** |" % len(liver_pats))
    w("| Liver lesions (connected pieces) over all outlined scans | %d |\n" % len(lesions))
    w("Mask names: " + ", ".join("`%s` %d" % (r.name.replace(".nii.gz", ""), r["count"])
                                 for _, r in members.set_index("name").iterrows()) + "\n")

    w("## 2. Diagnoses (ICD-10) of patients with a liver outline\n")
    g = group.reindex(sorted(liver_pats)).fillna("no ICD-10 code")
    w(table(g, "ICD-10 group (patients with a liver outline)"))
    icd_liver = set(group[group.str.contains("C22|C78.7")].index)
    w("ICD-coded liver-cancer patients (C22.x or C78.7): %d, of which %d have a liver outline and %d have none.\n"
      % (len(icd_liver), len(icd_liver & liver_pats), len(icd_liver - liver_pats)))
    nolc = [p for p in liver_pats if g.get(p) in ("other cancer code only", "non-cancer code only")]
    other = [c for p in nolc for c in set(codes.get(p, []))]
    if other:
        w("Most frequent codes (patients) among outlined patients without a liver code: " + ", ".join(
            "%s %d" % (k, v) for k, v in pd.Series(other).value_counts().head(12).items()) + "\n")

    w("## 3. Follow-up (scan dates) of patients with a liver outline\n")
    lp = m[m[P].isin(liver_pats)]
    per = lp.groupby(P).agg(scans=(ID, "size"), outlined=("liver", "sum"),
                            dates=("date", "nunique"),
                            outlined_dates=("date", lambda s: s[lp.loc[s.index, "liver"]].nunique()))
    first_out = lp[lp["liver"]].groupby(P)["date"].min()
    last_out = lp[lp["liver"]].groupby(P)["date"].max()
    withfile = lp[lp["has_file"]]  # only scans whose liver mask was read can show "no outline"
    first_any = withfile.groupby(P)["date"].min()
    last_any = withfile.groupby(P)["date"].max()
    appear = int(((first_any < first_out) & first_out.notna()).sum())
    vanish = int(((last_any > last_out) & last_out.notna()).sum())
    pv = lp[lp["liver"] & lp["phase"].isin(["portal_venous", "venous"])].groupby(P)["date"].nunique()
    w("| Item | Patients |\n|---|---|")
    for k in (1, 2, 3, 5):
        w("| Scans (any) on ≥ %d dates | %d |" % (k, int((per["dates"] >= k).sum())))
    for k in (2, 3, 5):
        w("| **Liver outlines on ≥ %d dates** | **%d** |" % (k, int((per["outlined_dates"] >= k).sum())))
    w("| Outlined portal-venous / venous scans on ≥ 2 dates | %d |" % int((pv >= 2).sum()))
    w("| **Tumour appears:** an earlier dated scan without liver outline, then outlined scans | %d |" % appear)
    w("| Tumour gone: outlined scans, then a later dated scan without liver outline | %d |" % vanish)
    w("| Patients whose scans have no date at all | %d |\n" % int((per["dates"] == 0).sum()))
    w("Caution: a scan without a liver outline may not show the liver at all (CancerVerse also has chest and "
      "pelvis CT), so \"tumour appears / gone\" are upper bounds until the field of view is checked.\n")
    per["icd_group"] = g.reindex(per.index)
    per["pv_outlined_dates"] = pv.reindex(per.index).fillna(0)
    per["appears"] = ((first_any < first_out) & first_out.notna()).reindex(per.index).fillna(False)
    w("| ICD-10 group | Patients | Outlines on ≥ 2 dates | ≥ 3 dates | ≥ 5 dates | Portal-venous outlines on ≥ 2 dates "
      "| Tumour appears (upper bound) |\n|---|---|---|---|---|---|---|")
    for gname, d in per.groupby("icd_group"):
        w("| %s | %d | %d | %d | %d | %d | %d |" % (gname, len(d), (d["outlined_dates"] >= 2).sum(),
          (d["outlined_dates"] >= 3).sum(), (d["outlined_dates"] >= 5).sum(), (d["pv_outlined_dates"] >= 2).sum(),
          d["appears"].sum()))
    w("")
    span = (last_out - first_out).dt.days[per["outlined_dates"] >= 2]
    if len(span):
        w("Span between first and last outlined scan (patients with ≥ 2 outlined dates): median %d days, "
          "IQR %d–%d, max %d.\n" % (span.median(), span.quantile(.25), span.quantile(.75), span.max()))
    w("Scans per patient: median %d, max %d; outlined scans per patient: median %d, max %d.\n" % (
        per["scans"].median(), per["scans"].max(), per["outlined"].median(), per["outlined"].max()))

    w("## 4. Outlined scans: phase and slice thickness\n")
    ls = m[m["liver"]].copy()
    w(table(ls["phase"].fillna("not recorded"), "Phase (outlined scans)"))
    th = ls.apply(lambda r: float(str(r["spacing_mm"]).split("x")[int(r["slice_axis"])]), axis=1)
    bins = pd.cut(th, [0, 1.5, 3.0, 5.0, 100], labels=["≤ 1.5 mm", "1.5–3 mm", "3–5 mm", "> 5 mm"])
    w(table(bins.astype(str), "Slice spacing (outlined scans)"))

    w("## 5. Lesions\n")
    les = lesions.merge(m[[ID, P, "date", "phase"]], left_on="cv_id", right_on=ID, how="left")
    les["stratum"] = les["inplane_max_diameter_mm"].apply(stratum)
    per_scan = ls["lesions"].astype(int)
    nb = pd.cut(per_scan, [0, 1, 2, 5, 10, 10 ** 6], labels=["1", "2", "3–5", "6–10", "> 10"])
    w(table(nb.astype(str), "Lesions per outlined scan"))
    order = ["<10", "10-19", "20-29", ">=30"]
    w("| Size stratum (in-plane max diameter) | Lesions on all outlined scans | Lesions on each patient's first "
      "outlined scan |\n|---|---|---|")
    first_scan = ls.sort_values("date").groupby(P)[ID].first()
    lf = les[les["cv_id"].isin(set(first_scan))]
    for s in order:
        w("| %s mm | %d | %d |" % (s, int((les["stratum"] == s).sum()), int((lf["stratum"] == s).sum())))
    w("| all | %d | %d |\n" % (len(les), len(lf)))
    w("Lesion volume median %.2f ml (IQR %.2f–%.2f); in-plane max diameter median %.1f mm.\n" % (
        les["volume_ml"].median(), les["volume_ml"].quantile(.25), les["volume_ml"].quantile(.75),
        les["inplane_max_diameter_mm"].median()))
    frag = (les["slices"] <= 1) | (les["inplane_max_diameter_mm"] < 5)
    w("Likely fragments (single slice or in-plane < 5 mm): %d of %d pieces; the < 10 mm row includes them.\n" % (
        int(frag.sum()), len(les)))
    real = les[~frag]
    lf_real = real[real["cv_id"].isin(set(first_scan))]
    w("Without fragments (pieces with ≥ 2 slices and in-plane ≥ 5 mm): %d pieces on all outlined scans, %d on each "
      "patient's first outlined scan.\n" % (len(real), len(lf_real)))
    w("| Size stratum, without fragments | All outlined scans | First outlined scan per patient | Patients (first scan) "
      "|\n|---|---|---|---|")
    for s in order:
        w("| %s mm | %d | %d | %d |" % (s, int((real["stratum"] == s).sum()), int((lf_real["stratum"] == s).sum()),
                                        lf_real.loc[lf_real["stratum"] == s, P].nunique()))
    rp = real.groupby("cv_id").size().reindex(ls[ID]).fillna(0).astype(int)
    nb2 = pd.cut(rp, [-1, 0, 1, 2, 5, 10, 10 ** 6], labels=["0 (fragments only)", "1", "2", "3–5", "6–10", "> 10"])
    w(table(nb2.astype(str), "Lesions per outlined scan, without fragments"))
    small = les[les["stratum"].isin(["<10", "10-19"])]
    w("Patients with ≥ 1 lesion < 20 mm on some outlined scan: %d; with ≥ 1 lesion of 10–19 mm: %d.\n" % (
        small[P].nunique(), les.loc[les["stratum"] == "10-19", P].nunique()))

    w("## 6. Patients with a liver outline: sex and age\n")
    first = meta[meta[P].isin(liver_pats)].drop_duplicates(P)
    ages = pd.to_numeric(first["age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    w("Sex: %s; age median %s (range %s–%s, known for %d).\n" % (
        first["sex"].fillna("missing").value_counts().to_dict(), ages.median(), ages.min(), ages.max(),
        int(ages.notna().sum())))

    md = "\n".join(out) + "\n"
    with open(os.path.join(mdir, "summary.md"), "w") as f:
        f.write(md)
    ref = first_out.min()
    pt = per.join((first_out - ref).dt.days.rename("first_outline_day")).join(
        (last_out - ref).dt.days.rename("last_outline_day"))
    pt.to_csv(os.path.join(mdir, "liver_patients.csv"))
    print(md)


if __name__ == "__main__":
    main()
