#!/usr/bin/env python3
"""Define small CancerVerse HCC subsets for download and get each CT file's exact size and SHA-256 (no images read).

What: from the metadata (dates, phases, ICD-10) and the measured liver masks (outputs/cancerverse/liver_masks), builds
file lists of CT volumes (CancerVerse/<cv_id>/ct.nii.gz) for these subsets of HCC patients (ICD-10 C22.0):
  pilot          10 random patients of hcc_pv_long (seed 0): all their outlined portal-venous scans
  hcc_pv_pair    patients with outlined portal-venous scans on >= 2 dates: only their FIRST two such dates
  hcc_recur_core the same patients, only 3 dates: last tumour date before the first gap, the first date without
                 tumour, the first date with tumour again
  hcc_recur_like patients whose tumour status goes tumour -> none -> tumour: one scan per date, all dates
  hcc_pv_long    patients with outlined portal-venous scans on >= 2 dates: all their outlined portal-venous scans
  pilot_all_dates the 10 pilot patients with ALL their dates (one scan per date), for looking at whole courses
  hcc_visual_review 12 other HCC patients with 4-8 dates and mostly portal-venous scans, 3 per course pattern
                 (tumour on every date and stable lesion count; tumour on every date and lesion count rising;
                 tumour -> none -> tumour; none first then tumour), all their dates, seed 0
One scan per patient and date: outlined + portal-venous first, then the one with most tumour voxels.
Sizes and checksums come from the Hugging Face paths-info API (metadata only).

Why: the full image set is ~3 TB and our storage has ~108 GB free; these lists let the team pick a subset that fits.

Outputs: outputs/cancerverse/subsets/<name>.csv (cv_id, patient, date, phase, outlined, n_lesions, size_bytes,
sha256) and summary.md (patients, scans, GB, download time at 10 / 20 / 50 MB/s, overlaps). Login node, ~1 min.
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

P, ID = "Patient ID", "CancerVerse ID"
PV = {"portal_venous", "venous"}
API = "https://huggingface.co/api/datasets/BodyMaps/CancerVerse/paths-info/main"


def file_info(cv_ids, batch=100):
    """{cv_id: (size_bytes, sha256)} from the Hugging Face API, in batches."""
    info = {}
    for i in range(0, len(cv_ids), batch):
        chunk = cv_ids[i:i + batch]
        data = urllib.parse.urlencode([("paths", "CancerVerse/%s/ct.nii.gz" % c) for c in chunk] +
                                      [("expand", "true")]).encode()
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(API, data=data, method="POST"), timeout=120) as r:
                    for e in json.load(r):
                        cv = e["path"].split("/")[1]
                        info[cv] = (int(e.get("size", 0)), (e.get("lfs") or {}).get("oid", ""))
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError("paths-info failed for a batch")
    return info


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
    nles = les.groupby("cv_id").size().rename("n_lesions")
    codes = meta.groupby(P)["icd10_code"].apply(lambda s: [str(c) for c in s.dropna()])
    hcc = set(p for p, cs in codes.items() if any(c.startswith("C22.0") for c in cs))

    m = meta.merge(scans[["cv_id", "voxels"]], left_on=ID, right_on="cv_id").join(nles, on="cv_id")
    m["n_lesions"] = m["n_lesions"].fillna(0).astype(int)
    m["date"] = pd.to_datetime(m["exam_date"], errors="coerce")
    m["outlined"] = m["voxels"] > 0
    m["pv"] = m["phase"].isin(PV)
    m = m[m[P].isin(hcc) & m["date"].notna()]
    m["rank"] = m["outlined"].astype(int) * 2 + m["pv"].astype(int)
    rep = (m.sort_values([P, "date", "rank", "voxels"], ascending=[True, True, False, False])
            .groupby([P, "date"]).head(1))                      # one scan per patient and date
    pvout = (m[m["outlined"] & m["pv"]].sort_values([P, "date", "voxels"], ascending=[True, True, False])
              .groupby([P, "date"]).head(1))                    # one outlined portal-venous scan per date
    long_p = pvout.groupby(P)["date"].nunique()
    long_p = sorted(long_p[long_p >= 2].index)

    seq = rep.groupby(P)["outlined"].apply(lambda s: "".join("T" if x else "-" for x in s))
    recur = sorted(p for p, s in seq.items() if "T" in s and "-" in s[s.index("T"):s.rindex("T") + 1])

    def core_dates(d):
        s = d["outlined"].tolist()
        first_t = s.index(True)
        gap = next(i for i in range(first_t + 1, len(s)) if not s[i])
        back = next(i for i in range(gap + 1, len(s)) if s[i])
        return d.iloc[[gap - 1, gap, back]]
    rr = rep[rep[P].isin(recur)]
    recur_core = pd.concat([core_dates(rr[rr[P] == p]) for p in recur]) if recur else rr.iloc[:0]

    rng = np.random.default_rng(0)
    pilot = sorted(rng.choice(long_p, size=min(10, len(long_p)), replace=False).tolist())
    # visual review: whole courses of a few HCC patients, 3 per pattern
    feat = rep.groupby(P).agg(n=("date", "size"), pv=("pv", "mean"),
                              seq=("outlined", lambda s: "".join("T" if x else "-" for x in s)),
                              nl=("n_lesions", list))
    feat = feat[(feat["n"] >= 4) & (feat["n"] <= 8) & (feat["pv"] >= 0.5) & ~feat.index.isin(pilot)]

    def kind(r):
        s, nl = r["seq"], r["nl"]
        if "T" not in s:
            return ""
        if "-" not in s:
            return "rising" if any(b > a for a, b in zip(nl, nl[1:])) and nl[-1] > nl[0] else "stable"
        core = s[s.index("T"):s.rindex("T") + 1]
        if "-" in core:
            return "recurrence-like"
        return "appears" if s.startswith("-") and s.endswith("T") else ""
    feat["kind"] = feat.apply(kind, axis=1)
    review = []
    for k in ["stable", "rising", "recurrence-like", "appears"]:
        cand = sorted(feat.index[feat["kind"] == k])
        review += rng.choice(cand, size=min(3, len(cand)), replace=False).tolist() if cand else []

    subsets = {
        "pilot": pvout[pvout[P].isin(pilot)],
        "hcc_pv_pair": pvout[pvout[P].isin(long_p)].groupby(P).head(2),
        "hcc_recur_core": recur_core,
        "hcc_recur_like": rep[rep[P].isin(recur)],
        "hcc_pv_long": pvout[pvout[P].isin(long_p)],
        "pilot_all_dates": rep[rep[P].isin(pilot)],
        "hcc_visual_review": rep[rep[P].isin(review)],
    }
    all_ids = sorted(set().union(*[set(d["cv_id"]) for d in subsets.values()]))
    info = file_info(all_ids)
    outdir = os.path.join(args.root, "outputs", "cancerverse", "subsets")
    os.makedirs(outdir, exist_ok=True)
    lines = ["# CancerVerse HCC download subsets (exact sizes from Hugging Face)\n",
             "| Subset | What | Patients | CT files | Size (GB) | Hours at 10 / 20 / 50 MB/s |", "|---|---|---|---|---|---|"]
    what = {"pilot": "10 random HCC patients with portal-venous tumour scans on ≥ 2 dates (all those scans)",
            "hcc_pv_pair": "every such HCC patient, first two portal-venous tumour dates only",
            "hcc_recur_core": "same 67 patients, only 3 dates each: last tumour, first without, tumour again",
            "hcc_recur_like": "HCC patients with tumour → none → tumour; one scan per date, all dates",
            "hcc_pv_long": "every such HCC patient, all portal-venous tumour scans",
            "pilot_all_dates": "the 10 pilot patients, every date (one scan per date)",
            "hcc_visual_review": "12 other HCC patients, 4–8 dates, 3 per course: stable / rising / recurrence-like / appears"}
    for name, d in subsets.items():
        d = d.assign(size_bytes=d["cv_id"].map(lambda c: info.get(c, (0, ""))[0]),
                     sha256=d["cv_id"].map(lambda c: info.get(c, (0, ""))[1]),
                     date=d["date"].dt.strftime("%Y-%m-%d"))
        missing = int((d["size_bytes"] == 0).sum())
        d[["cv_id", P, "date", "phase", "outlined", "n_lesions", "size_bytes", "sha256"]].rename(
            columns={P: "patient"}).to_csv(os.path.join(outdir, name + ".csv"), index=False)
        gb = d["size_bytes"].sum() / 1e9
        lines.append("| `%s` | %s | %d | %d%s | %.1f | %.1f / %.1f / %.1f |" % (
            name, what[name], d[P].nunique(), len(d), " (%d sizes missing)" % missing if missing else "", gb,
            gb * 1e3 / 10 / 3600, gb * 1e3 / 20 / 3600, gb * 1e3 / 50 / 3600))
    for a, b in (("hcc_pv_pair", "hcc_recur_core"), ("hcc_pv_pair", "hcc_recur_like")):
        both = set(subsets[a]["cv_id"]) | set(subsets[b]["cv_id"])
        lines.append("Union of `%s` + `%s`: %d files, %.1f GB; patients in both: %d." % (
            a, b, len(both), sum(info[c][0] for c in both) / 1e9, len(set(subsets[a][P]) & set(subsets[b][P]))))
    lines.append("")
    lines.append("Liver masks for these scans are in the label archive (13 small files per scan); only the needed ones "
                 "would be extracted.\n")
    text = "\n".join(lines)
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
