#!/usr/bin/env python3
"""Tests for the shared scorer and the nnU-Net candidate converter, on made-up cases with hand-computed answers.

What: builds a tiny fake release (2 cases on a 40^3 grid: an MCT-LTDiag HCC case with a malignant lesion, a
haemangioma and an ignore region; a PLC-CECT control without lesions) plus candidate maps, then checks every
status (hit, duplicate, near-miss below IoU 0.1, neutral on an ignore region, false positive, unverifiable), the
FROC points, FROC means, AP and patient AUROC against values worked out by hand, in both modes and with
targets = malignant only. Also runs the command-line scorer end to end and checks candidates_from_nnunet.

Usage: python -I evaluation/test_scorer.py <empty temporary folder>   (prints PASS / raises on failure)
"""
import csv
import os
import subprocess
import sys

import numpy as np
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import candidates_from_nnunet as C  # noqa: E402
import scorer as S  # noqa: E402

N = 40


def cube(a, z0, z1, y0, y1, x0, x1, v):
    a[z0:z1, y0:y1, x0:x1] = v


def write(arr, path):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, path, True)


def build(tmp):
    rel = os.path.join(tmp, "release")
    pred = os.path.join(tmp, "pred")
    os.makedirs(pred)
    # case A: MCT-LTDiag, HCC
    inst = np.zeros((N, N, N), np.uint16)
    lab = np.zeros((N, N, N), np.uint8)
    cube(inst, 5, 13, 5, 13, 5, 13, 1); cube(lab, 5, 13, 5, 13, 5, 13, 1)        # L1 malignant, 512 voxels
    cube(inst, 25, 31, 25, 31, 25, 31, 2); cube(lab, 25, 31, 25, 31, 25, 31, 2)  # L2 haemangioma, 216 voxels
    cube(lab, 5, 10, 25, 30, 25, 30, 3)                                          # ignore region
    cand = np.zeros((N, N, N), np.uint16)
    cube(cand, 5, 13, 5, 13, 5, 9, 1)      # c1: half of L1 (IoU 0.5), 0.9 -> TP
    cube(cand, 5, 10, 25, 30, 25, 30, 2)   # c2: on the ignore region, 0.8 -> neutral
    cube(cand, 30, 34, 5, 9, 5, 9, 3)      # c3: background, 64 voxels, 0.7 -> FP (unverifiable: < 10 mm, MCT)
    cube(cand, 5, 13, 5, 13, 9, 13, 4)     # c4: other half of L1, 0.6 -> FP_duplicate (stays FP)
    cube(cand, 29, 35, 29, 35, 29, 35, 5)  # c5: touches L2 with IoU 8/424 = 0.019, 0.65 -> FP (unverifiable)
    cube(cand, 25, 31, 25, 31, 25, 29, 7)  # c7: 2/3 of L2 (IoU 0.667), 0.3 -> TP
    # c5 and c7 must not overlap: c5 covers z,y,x 29..34 and c7 x 25..28 -> disjoint
    sa = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.65, 7: 0.3}
    # case B: PLC-CECT control
    instB = np.zeros((N, N, N), np.uint16)
    labB = np.zeros((N, N, N), np.uint8)
    candB = np.zeros((N, N, N), np.uint16)
    cube(candB, 10, 20, 10, 20, 10, 20, 1)  # 1000 voxels, 0.5 -> FP (PLC: never unverifiable)
    sb = {1: 0.5}
    for cid, i, l in (("MCT-LTDiag_A", inst, lab), ("PLC-CECT_B", instB, labB)):
        d = os.path.join(rel, "cases", cid)
        os.makedirs(d)
        write(i, os.path.join(d, "lesion_instances.nii.gz"))
        write(l, os.path.join(d, "lesion_label.nii.gz"))
        write(np.zeros((N, N, N), np.int16), os.path.join(d, "image_pvp.nii.gz"))
    write(cand, os.path.join(pred, "MCT-LTDiag_A_candidates.nii.gz"))
    write(candB, os.path.join(pred, "PLC-CECT_B_candidates.nii.gz"))
    with open(os.path.join(pred, "scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "candidate_id", "score"])
        for k, v in sa.items():
            w.writerow(["MCT-LTDiag_A", k, v])
        for k, v in sb.items():
            w.writerow(["PLC-CECT_B", k, v])
    with open(os.path.join(rel, "lesions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "lesion_id", "class", "inplane_max_diameter_mm", "in_release"])
        w.writerow(["MCT-LTDiag_A", 1, 1, 15.0, "True"])
        w.writerow(["MCT-LTDiag_A", 2, 2, 8.0, "True"])
    with open(os.path.join(rel, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "dataset", "diagnosis"])
        w.writerow(["MCT-LTDiag_A", "MCT-LTDiag", "HCC"])
        w.writerow(["PLC-CECT_B", "PLC-CECT", "CN"])
    with open(os.path.join(tmp, "cases.txt"), "w") as f:
        f.write("MCT-LTDiag_A\nPLC-CECT_B\n")
    return rel, pred, (inst, lab, cand, sa), (instB, labB, candB, sb)


def close(a, b, tol=1e-4):
    assert abs(a - b) < tol, (a, b)


def case_data(A, B, positive):
    out = []
    for (inst, lab, cand, sc), ds, diag, size in ((A, "MCT-LTDiag", "HCC", {1: 15.0, 2: 8.0}), (B, "PLC-CECT", "CN", {})):
        lrows, crows, _ = S.match_case(inst, lab, cand.astype(np.int64), sc, positive, 0.1, ds, True)
        out.append({"lesions": [(S.stratum(size[r["lesion"]]), r["score"] if r["score"] != "" else None) for r in lrows],
                    "cands": [(r["score"], r["status"]) for r in crows],
                    "patient": (max([r["score"] for r in crows], default=0.0), diag not in S.NEGATIVE_DIAGNOSES)})
    return out


def main():
    tmp = sys.argv[1]
    os.makedirs(tmp, exist_ok=True)
    assert not os.listdir(tmp), "give an empty folder"
    rel, pred, A, B = build(tmp)

    # statuses
    _, crows, _ = S.match_case(A[0], A[1], A[2].astype(np.int64), A[3], {1, 2}, 0.1, "MCT-LTDiag", True)
    st = {r["candidate"]: r["status"] for r in crows}
    assert st == {1: "TP", 2: "neutral", 3: "unverifiable", 4: "FP_duplicate", 5: "unverifiable", 7: "TP"}, st

    # metrics, targets = classes 1 + 2
    m = S.metrics(case_data(A, B, {1, 2}))
    f = m["main"]["froc"]
    assert [f["all"][r] for r in S.FP_RATES] == [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0], f["all"]
    close(f["all"]["froc_mean"], 5 / 7); close(f["<20"]["froc_mean"], 5 / 7)
    close(f["10-19"]["froc_mean"], 1.0); close(f["<10"]["froc_mean"], 3 / 7)
    close(m["main"]["ap"], 0.5 + (2 / 6) * 0.5)
    fu = m["unverifiable"]["froc"]
    assert [fu["all"][r] for r in S.FP_RATES] == [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0], fu["all"]
    close(fu["all"]["froc_mean"], 5.5 / 7)
    close(m["unverifiable"]["ap"], 0.5 + (2 / 4) * 0.5)
    close(m["auroc"], 1.0)

    # targets = malignant only: the haemangioma and its candidate become neutral
    m1 = S.metrics(case_data(A, B, {1}))
    assert m1["main"]["froc"]["all"]["lesions"] == 1
    close(m1["main"]["froc"]["all"]["froc_mean"], 1.0)

    # AUROC with ties and one class
    close(S.auroc([0.2, 0.2, 0.9], [False, True, True]), 0.75)
    assert S.auroc([0.1, 0.2], [True, True]) != S.auroc([0.1, 0.2], [True, True])  # NaN

    # command line, end to end
    out = os.path.join(tmp, "out")
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "scorer.py"), "--release", rel, "--cases",
                        os.path.join(tmp, "cases.txt"), "--pred", pred, "--out", out, "--bootstrap", "50"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with open(os.path.join(out, "metrics.csv")) as fh:
        row = next(csv.DictReader(fh))
    close(float(row["val_froc_mean_all"]), round(5 / 7, 4)); close(float(row["val_patient_auroc"]), 1.0)

    # candidate converter: core 0.9 with a 0.05 halo (nested -> one candidate), a 0.03 blob, a tiny 5-voxel blob
    prob = np.zeros((N, N, N), np.float32)
    cube(prob, 4, 12, 4, 12, 4, 12, 0.06); cube(prob, 6, 9, 6, 9, 6, 9, 0.9)   # 27-voxel core, 512-voxel halo
    cube(prob, 25, 29, 25, 29, 25, 27, 0.03)                                   # 32 voxels, only >= 0.01
    prob[35, 35, 30:35] = 0.9                                                  # 5 voxels: below 15
    cmap, rows = C.candidates(prob, [0.5, 0.2, 0.05, 0.01, 0.002], 15)
    assert [(k, round(s, 2), lev, nv) for k, s, lev, nv in rows] == [(1, 0.9, 0.5, 27), (2, 0.03, 0.01, 32)], rows
    assert cmap.max() == 2 and (cmap == 1).sum() == 27
    print("PASS: scorer statuses, FROC points and means (both modes), AP, AUROC, malignant-only targets, "
          "command line, candidate converter")


if __name__ == "__main__":
    main()
