#!/usr/bin/env python3
"""Longitudinal structure of CancerVerse's liver part: time points, gaps, phases, tumour course (counts only).

What: from the metadata (exam date, phase, ICD-10) and the measured liver masks (outputs/cancerverse/liver_masks), one
time point = one exam date of one patient. Reports per patient group (all patients, patients with a liver outline,
and by ICD-10 group): time points per patient, gaps between consecutive time points, total follow-up span, scans
and phases per time point, phase consistency of consecutive pairs, the tumour-status course over time, and the change
in tumour burden (number of lesions, total tumour volume) between consecutive outlined time points.

Why: to know how much follow-up CancerVerse really offers before any image download (the images are ~3 TB).

Definitions: lesions = liver mask pieces with >= 2 slices and in-plane >= 5 mm (tiny fragments removed). When a
date has several outlined scans, the portal-venous one is used, else the one with the most tumour voxels. A date
without liver outline may be a scan that does not show the liver (field of view unchecked), so "no tumour" is an
upper bound. Lesions are not matched across dates: burden changes are counts and volumes, not per-lesion tracking.

Output: outputs/cancerverse/longitudinal.md. Login node, seconds.
"""
import argparse
import os

import numpy as np
import pandas as pd

P, ID = "Patient ID", "CancerVerse ID"
PV = {"portal_venous", "venous"}


def icd_group(codes):
    codes = [str(c) for c in codes]
    if any(c.startswith("C22.0") for c in codes):
        return "HCC"
    if any(c.startswith("C22.1") for c in codes):
        return "ICC"
    if any(c.startswith("C22") for c in codes):
        return "other primary liver"
    if any(c.startswith("C78.7") for c in codes):
        return "liver metastases (coded)"
    if any(c.startswith("C") for c in codes):
        return "other cancer code"
    return "no / non-cancer code"


def dist(series, bins, labels):
    return pd.cut(series, bins=bins, labels=labels, right=True).value_counts().reindex(labels).fillna(0).astype(int)


def mdtable(title, cols, rows):
    out = ["| %s | %s |" % (title, " | ".join(cols)), "|---|" + "---|" * len(cols)]
    for name, vals in rows:
        out.append("| %s | %s |" % (name, " | ".join(vals)))
    return "\n".join(out) + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.path.abspath(os.path.join(here, "..", "..", "..")))
    args = ap.parse_args()
    meta = pd.read_csv(os.path.join(args.root, "data", "CancerVerse", "metadata", "CancerVerse_dataset_metadata.csv"),
                       low_memory=False)
    md = os.path.join(args.root, "outputs", "cancerverse", "liver_masks")
    scans = pd.read_csv(os.path.join(md, "scans.csv"))
    les = pd.read_csv(os.path.join(md, "lesions.csv"))
    scans = scans[scans["mask"] == "liver_lesion"]
    les = les[(les["mask"] == "liver_lesion") & (les["slices"] >= 2) & (les["inplane_max_diameter_mm"] >= 5)]
    burden = les.groupby("cv_id").agg(n_les=("lesion", "size"), vol=("volume_ml", "sum"))

    meta["date"] = pd.to_datetime(meta["exam_date"], errors="coerce")
    m = meta.merge(scans[["cv_id", "voxels"]], left_on=ID, right_on="cv_id", how="inner")
    m = m.join(burden, on="cv_id")
    m["n_les"] = m["n_les"].fillna(0).astype(int)
    m["vol"] = m["vol"].fillna(0.0)
    m["out"] = m["voxels"] > 0
    m["pv"] = m["phase"].isin(PV)
    group = meta.groupby(P)["icd10_code"].apply(lambda s: icd_group(s.dropna()))
    liver_p = set(m.loc[m["out"], P])
    undated = m[m["date"].isna()]

    # one row per patient and date (time point)
    d = m.dropna(subset=["date"]).copy()
    d["rank"] = d["out"].astype(int) * 2 + d["pv"].astype(int)  # prefer outlined, then portal-venous
    d = d.sort_values([P, "date", "rank", "voxels"], ascending=[True, True, False, False])
    tp = d.groupby([P, "date"]).agg(n_scans=(ID, "size"), out=("out", "any"), pv=("pv", "any"),
                                    phases=("phase", lambda s: "+".join(sorted(set(s.dropna())))),
                                    n_les=("n_les", "first"), vol=("vol", "first"),
                                    rep_phase=("phase", "first")).reset_index()
    tp["tp_index"] = tp.groupby(P).cumcount() + 1
    tp["gap"] = tp.groupby(P)["date"].diff().dt.days
    tp["prev_out"] = tp.groupby(P)["out"].shift()
    tp["prev_phase"] = tp.groupby(P)["rep_phase"].shift()
    tp["prev_n"] = tp.groupby(P)["n_les"].shift()
    tp["prev_vol"] = tp.groupby(P)["vol"].shift()
    tp["group"] = tp[P].map(group).fillna("no / non-cancer code")

    groups = [("All CancerVerse patients", set(tp[P])),
              ("Patients with a liver outline", liver_p)]
    for gname in ["HCC", "ICC", "liver metastases (coded)", "other cancer code", "no / non-cancer code"]:
        groups.append(("  liver outline, " + gname, set(p for p in liver_p if group.get(p, "no / non-cancer code") == gname)))

    out = []
    w = out.append
    w("# CancerVerse liver part: longitudinal structure (counts only)\n")
    w("Time point = one exam date of one patient (scans without a date: %d of all scans, %d of the liver-outline patients' "
      "scans, left out). Lesions = liver mask pieces with ≥ 2 slices and in-plane ≥ 5 mm. \"No outline\" on a date may mean "
      "the scan does not show the liver (field of view unchecked), so tumour-absent counts are upper bounds. Lesions are "
      "not matched across dates.\n" % (meta["date"].isna().sum(), int(undated[P].isin(liver_p).sum())))

    # 1. time points per patient
    w("## 1. Time points per patient\n")
    bins, labels = [0, 1, 2, 3, 4, 5, 10, 20, 1000], ["1", "2", "3", "4", "5", "6–10", "11–20", "> 20"]
    rows = []
    for gname, pats in groups:
        n = tp[tp[P].isin(pats)].groupby(P).size()
        dd = dist(n, bins, labels)
        rows.append((gname, ["{:,}".format(len(n))] + ["{:,}".format(v) for v in dd] +
                     ["%.0f / %.1f / %d" % (n.median(), n.mean(), n.max())]))
    w(mdtable("Group", ["Patients"] + labels + ["median / mean / max"], rows))
    rows = []
    for gname, pats in groups[1:]:
        n = tp[tp[P].isin(pats) & tp["out"]].groupby(P).size()
        dd = dist(n, bins, labels)
        rows.append((gname, ["{:,}".format(len(n))] + ["{:,}".format(v) for v in dd]))
    w("Time points **with a liver outline** per patient:\n")
    w(mdtable("Group", ["Patients"] + labels, rows))

    # 2. gaps
    w("## 2. Time between consecutive time points\n")
    gb = [-1, 7, 30, 90, 180, 365, 730, 100000]
    gl = ["≤ 1 week", "1 week–1 month", "1–3 months", "3–6 months", "6–12 months", "1–2 years", "> 2 years"]
    rows = []
    for gname, pats in groups:
        g = tp[tp[P].isin(pats)]["gap"].dropna()
        dd = dist(g, gb, gl)
        rows.append((gname, ["{:,}".format(len(g))] + ["{:,}".format(v) for v in dd] +
                     ["%d (%d–%d)" % (g.median(), g.quantile(.25), g.quantile(.75)) if len(g) else "—"]))
    w(mdtable("Group (all consecutive dates)", ["Gaps"] + gl + ["median (IQR) days"], rows))
    rows = []
    for gname, pats in groups[1:]:
        o = tp[tp[P].isin(pats) & tp["out"]].copy()
        o["ogap"] = o.groupby(P)["date"].diff().dt.days
        g = o["ogap"].dropna()
        dd = dist(g, gb, gl)
        rows.append((gname, ["{:,}".format(len(g))] + ["{:,}".format(v) for v in dd] +
                     ["%d (%d–%d)" % (g.median(), g.quantile(.25), g.quantile(.75)) if len(g) else "—"]))
    w("Between consecutive time points **that both have a liver outline**:\n")
    w(mdtable("Group", ["Gaps"] + gl + ["median (IQR) days"], rows))

    # 3. follow-up span
    w("## 3. Total follow-up span (first to last time point)\n")
    sb = [-1, 0, 90, 180, 365, 730, 1825, 100000]
    sl = ["single date", "≤ 3 months", "3–6 months", "6–12 months", "1–2 years", "2–5 years", "> 5 years"]
    rows = []
    for gname, pats in groups:
        g = tp[tp[P].isin(pats)].groupby(P)["date"].agg(lambda s: (s.max() - s.min()).days)
        dd = dist(g, sb, sl)
        multi = g[g > 0]
        rows.append((gname, ["{:,}".format(len(g))] + ["{:,}".format(v) for v in dd] +
                     ["%d / %d" % (multi.median(), multi.max()) if len(multi) else "—"]))
    w(mdtable("Group", ["Patients"] + sl + ["median / max days (≥ 2 dates)"], rows))

    # 4. scans and phases per time point, phase consistency
    lt = tp[tp[P].isin(liver_p)]
    w("## 4. Scans and phases per time point (patients with a liver outline)\n")
    w("Time points: %s; with > 1 scan: %s; with > 1 phase: %s. Most frequent phase sets per time point: %s.\n" % (
        "{:,}".format(len(lt)), "{:,}".format(int((lt["n_scans"] > 1).sum())),
        "{:,}".format(int(lt["phases"].str.contains(r"\+").sum())),
        ", ".join("%s %s" % (k or "not recorded", "{:,}".format(v)) for k, v in lt["phases"].value_counts().head(6).items())))
    pr = lt.dropna(subset=["prev_out"])
    po = pr[pr["out"] & pr["prev_out"].astype(bool)]

    def ph(x):
        return "portal-venous" if x in PV else "arterial" if str(x).startswith("arterial") else \
            "non-contrast" if x == "non-contrast" else "not recorded"
    combo = (po["prev_phase"].map(ph) + " → " + po["rep_phase"].map(ph)).value_counts()
    w("Phase of the two scans in consecutive pairs **with a liver outline on both dates** (%s pairs): %s.\n" % (
        "{:,}".format(len(po)), ", ".join("%s %s" % (k, "{:,}".format(v)) for k, v in combo.head(8).items())))

    # 5. tumour-status course
    w("## 5. Tumour status over time (patients with a liver outline and ≥ 2 time points)\n")
    seq = lt.groupby(P)["out"].apply(lambda s: "".join("T" if x else "-" for x in s))
    seq = seq[seq.str.len() >= 2]

    def pattern(s):
        if "-" not in s:
            return "tumour on every date"
        first, last = s.index("T"), s.rindex("T")
        core = s[first:last + 1]
        lead, tail = first > 0, last < len(s) - 1
        if "-" in core:
            return "tumour, none, tumour again (intermittent)"
        if lead and not tail:
            return "none first, then tumour (appears; upper bound)"
        if tail and not lead:
            return "tumour first, then none (gone)"
        return "none, tumour, none"
    pat = seq.apply(pattern)
    rows = []
    for gname, pats in groups[1:]:
        pp = pat[pat.index.isin(pats)]
        rows.append((gname, ["{:,}".format(len(pp))] + ["{:,}".format(int((pp == k).sum())) for k in
                     ["tumour on every date", "none first, then tumour (appears; upper bound)",
                      "tumour first, then none (gone)", "tumour, none, tumour again (intermittent)", "none, tumour, none"]]))
    w(mdtable("Group", ["Patients", "Tumour on every date", "None, then tumour (appears)", "Tumour, then none (gone)",
                        "Tumour, none, tumour again", "None, tumour, none"], rows))
    w("Share of patients whose time point k has a liver outline (patients with a liver outline): " + ", ".join(
        "TP%d %d %% of %s" % (k, round(100 * lt[lt.tp_index == k]["out"].mean()), "{:,}".format(int((lt.tp_index == k).sum())))
        for k in range(1, 7) if (lt.tp_index == k).any()) + ".\n")

    # 6. burden change between consecutive outlined time points
    w("## 6. Change in tumour burden between consecutive time points with a liver outline\n")
    w("Number of lesions and total tumour volume of the representative scan per date. Volume change > +20 % = growth, "
      "< −20 % = shrinkage (a volume rule, not RECIST). Lesion counts depend on how the masks split tumours.\n")
    rows = []
    for gname, pats in groups[1:]:
        q = po[po[P].isin(pats)]
        dn = q["n_les"] - q["prev_n"]
        rv = (q["vol"] - q["prev_vol"]) / q["prev_vol"].replace(0, np.nan)
        rows.append((gname, ["{:,}".format(len(q)), "{:,}".format(int((dn > 0).sum())), "{:,}".format(int((dn == 0).sum())),
                             "{:,}".format(int((dn < 0).sum())), "{:,}".format(int((rv > 0.2).sum())),
                             "{:,}".format(int(((rv >= -0.2) & (rv <= 0.2)).sum())), "{:,}".format(int((rv < -0.2).sum())),
                             "%d" % q["gap"].median() if len(q) else "—"]))
    w(mdtable("Group", ["Pairs", "More lesions", "Same number", "Fewer lesions", "Volume grows > 20 %", "Stable ± 20 %",
                        "Shrinks > 20 %", "Median gap (days)"], rows))
    small = po[(po["prev_n"] < po["n_les"])]
    w("Pairs where the lesion count rises (a rough sign of new lesions): %s in %s patients (HCC %s patients).\n" % (
        "{:,}".format(len(small)), "{:,}".format(small[P].nunique()),
        "{:,}".format(small[small["group"] == "HCC"][P].nunique())))

    # 7. calendar
    first = lt.groupby(P)["date"].min().dt.year.value_counts().sort_index()
    w("## 7. Year of the first time point (patients with a liver outline)\n")
    w(", ".join("%d: %s" % (y, "{:,}".format(v)) for y, v in first.items()) + "\n")

    text = "\n".join(out) + "\n"
    path = os.path.join(args.root, "outputs", "cancerverse", "longitudinal.md")
    with open(path + ".part", "w") as f:
        f.write(text)
    os.replace(path + ".part", path)
    print(text)


if __name__ == "__main__":
    main()
