#!/usr/bin/env python3
"""The HOPE shared scorer: lesion-level detection metrics for every model, with one fixed set of rules.

What: compares a model's candidates (per case: a candidate-ID map on the release 1 mm grid + one score per
candidate) with the release truth and writes FROC by lesion size, sensitivity at fixed false positives per case,
average precision, patient-level AUROC and bootstrap 95 % confidence intervals.

Fixed team rules (CLAUDE.md, STANDARD.md "Evaluation convention", PLAN.md decisions of 26 Nov 2026):
  * Targets = every lesion numbered in lesion_instances.nii.gz whose class is in --positive (default 1,2 =
    malignant + haemangioma, as released). Ignore regions (lesion_label 3; plus any non-positive class) are neutral.
  * A candidate HITS a lesion when their 3D masks have IoU >= 0.1 (--iou). Matching is one-to-one and greedy in
    descending score: each candidate takes the unmatched lesion with the highest IoU >= 0.1. A candidate that only
    overlaps an already-found lesion is a false positive (duplicate).
  * An unmatched candidate is NEUTRAL (neither hit nor false positive) when >= 50 % of its voxels lie in neutral
    regions or its IoU with a touched neutral region is >= 0.1; otherwise it is a FALSE POSITIVE.
  * Optional (--unverifiable, Plan v2): unmatched candidates with an equivalent diameter < 10 mm on datasets whose
    annotators left out sub-centimetre lesions (MCT-LTDiag, LiTS) are "unverifiable" and excluded from false
    positives. Results are written with and without this rule.
  * Sizes: in-plane max diameter of each lesion from the release lesions.csv (native scan, voxel centres);
    strata < 10, 10-19, 20-29, >= 30 mm, plus < 20 mm and all.
  * FROC points: 1/8, 1/4, 1/2, 1, 2, 4, 8 false positives per case; sensitivity at a point = the highest sensitivity
    reached with at most that many false positives per case; FROC mean = mean of the 7 points.
  * Patient level: score = highest candidate score in the case (0 without candidates); positive = malignant
    diagnosis (diagnoses HH and CN, i.e. haemangioma patients and controls, are negative). Skipped when one class
    is missing (e.g. external sets, all patients have cancer).

Inputs:  --release <processed/v1 or processed/v2_external> (cases/, lesions.csv, manifest.csv)
         --cases <list of case ids, e.g. processed/v1/splits/val.txt>   (every listed case is scored; a case without
                  a candidate file is an error unless --allow-missing, then it counts as "no candidates")
         --pred <folder with <case_id>_candidates.nii.gz + scores.csv (case_id, candidate_id, score)>
Outputs (--out): metrics.csv (one row, the tracker columns), froc.csv, lesions.csv, candidates.csv, summary.md.
summary.md holds counts only.

Usage: python -I evaluation/scorer.py --release <dir> --cases <txt> --pred <dir> --out <dir> [--bootstrap 1000]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

FP_RATES = [0.125, 0.25, 0.5, 1, 2, 4, 8]
STRATA = ["<10", "10-19", "20-29", ">=30"]
GROUPS = {"<10": ["<10"], "10-19": ["10-19"], "20-29": ["20-29"], ">=30": [">=30"], "<20": ["<10", "10-19"],
          "all": STRATA + ["unknown"]}
NEGATIVE_DIAGNOSES = {"HH", "CN"}
SCORER_VERSION = "1.0 (2026-11-27)"


def stratum(d):
    return "<10" if d < 10 else "10-19" if d < 20 else "20-29" if d < 30 else ">=30"


def read_array(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def match_case(inst, lab, cand, scores, positive, iou_thr, dataset, unverifiable_on):
    """One case. Returns (lesion_rows, candidate_rows) with match results (no metrics yet)."""
    lesion_ids = [int(i) for i in np.unique(inst) if i]
    cls = {}
    for i in lesion_ids:
        v = lab[inst == i]
        v = v[v > 0]
        cls[i] = int(np.bincount(v).argmax()) if v.size else 0
    pos_ids = [i for i in lesion_ids if cls[i] in positive]
    neutral = (lab == 3) | np.isin(lab, [c for c in (1, 2) if c not in positive])
    neutral_lab, _ = ndimage.label(neutral, structure=np.ones((3, 3, 3), bool))

    cand_ids = [int(c) for c in np.unique(cand) if c]
    csize = dict(zip(*np.unique(cand[cand > 0], return_counts=True))) if cand_ids else {}
    pos_set = set(pos_ids)
    gsize = {i: int((inst == i).sum()) for i in pos_ids}
    # pairwise intersections candidate x positive lesion
    both = (cand > 0) & np.isin(inst, pos_ids) if pos_ids else np.zeros_like(cand, bool)
    inter = {}
    if both.any():
        pairs, counts = np.unique(np.stack([cand[both], inst[both]]), axis=1, return_counts=True)
        for (c, g), n in zip(pairs.T, counts):
            inter[(int(c), int(g))] = int(n)
    # neutral overlap per candidate
    neut_in = {}
    neut_iou = {}
    if cand_ids and neutral.any():
        nb = (cand > 0) & neutral
        if nb.any():
            for c, n in zip(*np.unique(cand[nb], return_counts=True)):
                neut_in[int(c)] = int(n)
            pairs, counts = np.unique(np.stack([cand[nb], neutral_lab[nb]]), axis=1, return_counts=True)
            nsize = dict(zip(*np.unique(neutral_lab[neutral_lab > 0], return_counts=True)))
            for (c, k), n in zip(pairs.T, counts):
                iou = n / (csize[c] + nsize[k] - n)
                neut_iou[int(c)] = max(neut_iou.get(int(c), 0.0), float(iou))

    matched = {}
    cand_rows = []
    for c in sorted(cand_ids, key=lambda k: -scores.get(k, 0.0)):
        s = float(scores.get(c, 0.0))
        ious = {g: n / (csize[c] + gsize[g] - n) for (cc, g), n in inter.items() if cc == c}
        free = {g: v for g, v in ious.items() if v >= iou_thr and g not in matched}
        if free:
            g = max(free, key=free.get)
            matched[g] = s
            cand_rows.append({"candidate": c, "score": s, "status": "TP", "lesion": g, "iou": round(free[g], 4)})
            continue
        if any(v >= iou_thr for v in ious.values()):
            status = "FP_duplicate"
        elif neut_in.get(c, 0) >= 0.5 * csize[c] or neut_iou.get(c, 0.0) >= iou_thr:
            status = "neutral"
        else:
            status = "FP"
        if status == "FP" and unverifiable_on and dataset in ("MCT-LTDiag", "LiTS"):  # not duplicates
            eq_d = (6 * csize[c] / np.pi) ** (1 / 3)  # 1 mm grid: voxels = mm^3
            if eq_d < 10:
                status = "unverifiable"
        cand_rows.append({"candidate": c, "score": s, "status": status, "lesion": "", "iou": ""})
    lesion_rows = [{"lesion": g, "class": cls[g], "score": matched.get(g, ""), "detected_any": g in matched}
                   for g in pos_ids]
    return lesion_rows, cand_rows, sorted(pos_set - set(matched))


def froc(lesions, cands, n_cases, groups=GROUPS):
    """lesions: list of (stratum, matched_score or None); cands: list of (score, is_fp). Returns dict."""
    out = {}
    fp_scores = np.sort(np.array([s for s, fp in cands if fp], float))[::-1]
    thr = np.unique(np.array([s for s, _ in cands], float))[::-1]
    fp_at = np.searchsorted(-fp_scores, -thr, side="right") / max(n_cases, 1)  # FP per case at threshold >= t
    for gname, strata in groups.items():
        ls = [sc for st, sc in lesions if st in strata]
        n = len(ls)
        det = np.array([sc for sc in ls if sc is not None], float)
        sens_at = np.array([(det >= t).sum() / n if n else np.nan for t in thr])
        row = {"lesions": n}
        for r in FP_RATES:
            ok = fp_at <= r + 1e-12
            row[r] = float(sens_at[ok].max()) if (n and ok.any()) else (0.0 if n else np.nan)
        row["froc_mean"] = float(np.mean([row[r] for r in FP_RATES])) if n else np.nan
        out[gname] = row
    return out


def average_precision(lesions, cands):
    n = len(lesions)
    if not n:
        return np.nan
    cs = sorted([(s, fp) for s, fp in cands], key=lambda x: -x[0])
    tp = 0
    ap = 0.0
    for k, (s, fp) in enumerate(cs, 1):
        if not fp:
            tp += 1
            ap += (tp / k) / n
    return ap


def auroc(scores, labels):
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    if y.all() or (~y).all():
        return np.nan
    ranks = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort").astype(float) + 1
    for v in np.unique(s):  # average ranks for ties
        m = s == v
        ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(case_data, groups=GROUPS):
    """case_data: list of dicts with lesions [(stratum, matched score or None)], cands [(score, status)],
    patient (score, is_positive). Main mode counts 'unverifiable' candidates as false positives; the
    'unverifiable' mode leaves them out (Plan v2 rule)."""
    lesions = [l for c in case_data for l in c["lesions"]]
    res = {}
    for mode, skip in (("main", ("neutral",)), ("unverifiable", ("neutral", "unverifiable"))):
        cands = [(s, st != "TP") for c in case_data for s, st in c["cands"] if st not in skip]
        res[mode] = {"froc": froc(lesions, cands, len(case_data), groups), "ap": average_precision(lesions, cands)}
    res["auroc"] = auroc([c["patient"][0] for c in case_data], [c["patient"][1] for c in case_data])
    return res


def bootstrap(case_data, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    keys = []
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(case_data), len(case_data))
        m = metrics([case_data[i] for i in idx])
        f = m["main"]["froc"]
        vals.append([f["<20"]["froc_mean"], f["<20"][0.5], f["<20"][1], f["<20"][2], f["all"]["froc_mean"],
                     m["main"]["ap"], m["auroc"]])
    keys = ["froc_mean_lt20", "sens_lt20_at_0.5fp", "sens_lt20_at_1fp", "sens_lt20_at_2fp", "froc_mean_all",
            "ap_all", "patient_auroc"]
    arr = np.array(vals, float)
    return {k: (float(np.nanpercentile(arr[:, j], 2.5)), float(np.nanpercentile(arr[:, j], 97.5)))
            for j, k in enumerate(keys)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", required=True)
    ap.add_argument("--cases", required=True, help="text file, one case id per line")
    ap.add_argument("--pred", required=True, help="folder with <case>_candidates.nii.gz and scores.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--positive", default="1,2", help="lesion classes that are targets (team rule: 1,2)")
    ap.add_argument("--iou", type=float, default=0.1, help="hit threshold (team rule: 0.1)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    positive = {int(x) for x in args.positive.split(",")}

    cases = [l.strip() for l in open(args.cases) if l.strip()]
    size = {}
    with open(os.path.join(args.release, "lesions.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["case_id"] in cases and str(r.get("in_release", "True")) == "True" and r["class"] in ("1", "2"):
                size[(r["case_id"], int(r["lesion_id"]))] = float(r["inplane_max_diameter_mm"])
    diag, dataset = {}, {}
    with open(os.path.join(args.release, "manifest.csv"), newline="") as f:
        for r in csv.DictReader(f):
            diag[r["case_id"]] = r.get("diagnosis", "")
            dataset[r["case_id"]] = r.get("dataset", "")
    scores = {}
    sp = os.path.join(args.pred, "scores.csv")
    if os.path.exists(sp):
        with open(sp, newline="") as f:
            for r in csv.DictReader(f):
                scores.setdefault(r["case_id"], {})[int(r["candidate_id"])] = float(r["score"])

    case_data, lesion_out, cand_out, missing = [], [], [], 0
    for cid in cases:
        cd = os.path.join(args.release, "cases", cid)
        inst = read_array(os.path.join(cd, "lesion_instances.nii.gz"))
        lab = read_array(os.path.join(cd, "lesion_label.nii.gz"))
        cp = os.path.join(args.pred, cid + "_candidates.nii.gz")
        if os.path.exists(cp):
            cand = read_array(cp).astype(np.int64)
            if cand.shape != inst.shape:
                sys.exit("candidate map on another grid than the release (a case)")
        elif args.allow_missing:
            cand = np.zeros(inst.shape, np.int64)
            missing += 1
        else:
            sys.exit("missing candidate file for a listed case (use --allow-missing to count it as empty)")
        sc = scores.get(cid, {})
        lrows, crows, _ = match_case(inst, lab, cand, sc, positive, args.iou, dataset.get(cid, ""), True)
        ls = []
        for r in lrows:
            d = size.get((cid, r["lesion"]))
            st = stratum(d) if d is not None else "unknown"
            ls.append((st, r["score"] if r["score"] != "" else None))
            lesion_out.append({"case_id": cid, "lesion_id": r["lesion"], "class": r["class"],
                               "inplane_max_diameter_mm": d, "stratum": st, "matched_score": r["score"]})
        for r in crows:
            cand_out.append({"case_id": cid, **r})
        case_data.append({"lesions": ls, "cands": [(r["score"], r["status"]) for r in crows],
                          "patient": (max([r["score"] for r in crows], default=0.0),
                                      diag.get(cid, "") not in NEGATIVE_DIAGNOSES)})

    m = metrics(case_data)
    ci = bootstrap(case_data, args.bootstrap) if args.bootstrap else {}
    os.makedirs(args.out, exist_ok=True)
    f = m["main"]["froc"]
    row = {"val_froc_mean_lt20": f["<20"]["froc_mean"], "val_sens_lt20_at_0.5fp": f["<20"][0.5],
           "val_sens_lt20_at_1fp": f["<20"][1], "val_sens_lt20_at_2fp": f["<20"][2],
           "val_froc_mean_all": f["all"]["froc_mean"], "val_ap_all": m["main"]["ap"], "val_patient_auroc": m["auroc"]}
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerow({k: "" if v != v else round(v, 4) for k, v in row.items()})
    with open(os.path.join(args.out, "froc.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "group", "lesions"] + ["sens_at_%gfp" % r for r in FP_RATES] + ["froc_mean"])
        for mode in ("main", "unverifiable"):
            for g, r in m[mode]["froc"].items():
                w.writerow([mode, g, r["lesions"]] + ["" if r[x] != r[x] else round(r[x], 4) for x in FP_RATES] +
                           ["" if r["froc_mean"] != r["froc_mean"] else round(r["froc_mean"], 4)])
    for name, rows in (("lesions.csv", lesion_out), ("candidates.csv", cand_out)):
        with open(os.path.join(args.out, name), "w", newline="") as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    status_counts = {}
    for r in cand_out:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    lines = ["# Scorer results", "",
             "Scorer %s · cases %d (%d without candidate file) · targets: classes %s · hit: IoU >= %g, one-to-one · "
             "neutral: ignore regions · bootstrap %d" % (SCORER_VERSION, len(cases), missing,
                                                         ",".join(map(str, sorted(positive))), args.iou, args.bootstrap),
             "", "Candidates: " + ", ".join("%s %d" % kv for kv in sorted(status_counts.items())), ""]
    for mode, title in (("main", "Main (Plan v2 'unverifiable' rule off)"),
                        ("unverifiable", "With the 'unverifiable' rule (unmatched < 10 mm on MCT-LTDiag / LiTS not counted)")):
        lines += ["## " + title, "", "| Lesions | n | " + " | ".join("%g FP" % r for r in FP_RATES) + " | FROC mean |",
                  "|---|---|" + "---|" * (len(FP_RATES) + 1)]
        for g, r in m[mode]["froc"].items():
            fmt = lambda v: "—" if v != v else "%.3f" % v
            lines.append("| %s | %d | %s | %s |" % (g, r["lesions"], " | ".join(fmt(r[x]) for x in FP_RATES),
                                                    fmt(r["froc_mean"])))
        lines += ["", "AP (all lesions): %s" % ("—" if m[mode]["ap"] != m[mode]["ap"] else "%.3f" % m[mode]["ap"]), ""]
    lines.append("Patient AUROC: %s" % ("— (one class only)" if m["auroc"] != m["auroc"] else "%.3f" % m["auroc"]))
    if ci:
        lines += ["", "## 95 % bootstrap intervals (cases resampled, main mode)", "", "| Metric | Value | 95 % CI |",
                  "|---|---|---|"]
        val = {"froc_mean_lt20": row["val_froc_mean_lt20"], "sens_lt20_at_0.5fp": row["val_sens_lt20_at_0.5fp"],
               "sens_lt20_at_1fp": row["val_sens_lt20_at_1fp"], "sens_lt20_at_2fp": row["val_sens_lt20_at_2fp"],
               "froc_mean_all": row["val_froc_mean_all"], "ap_all": row["val_ap_all"],
               "patient_auroc": row["val_patient_auroc"]}
        for k, (lo, hi) in ci.items():
            v = val[k]
            lines.append("| %s | %s | %s |" % (k, "—" if v != v else "%.3f" % v,
                                                "—" if lo != lo else "%.3f–%.3f" % (lo, hi)))
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "settings.json"), "w") as fh:
        json.dump({"scorer_version": SCORER_VERSION, "positive": sorted(positive), "iou": args.iou,
                   "bootstrap": args.bootstrap, "cases": len(cases), "missing_candidate_files": missing}, fh, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
