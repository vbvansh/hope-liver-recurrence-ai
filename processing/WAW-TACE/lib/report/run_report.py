#!/usr/bin/env python3
"""Longitudinal tumor-progression report generation for abdominal/thoracic CT —
per timepoint, patient-wide memory, VLM reasoner, RECIST 1.1.

Workflow per timepoint t:
  1. measure t from the lesion mask (GT or SegVol) + organ context (TotalSeg)  [measurements]
  2. retrieve memory: baseline + nadir + prior + rolling summary               [ledger]
  3. assemble VLM context (phase + position + measurements + priors + slices)
  4a. S2: independent 2D image read over the CT slices (no numbers)            [findings_2d]
  4b. deterministic triangulation S1 vs S2 + RECIST hint                       [fusion]
  4c. S3: fuse into the final report JSON + label                             [vlm/prompts]
  5. QC verify (+1 retry); deterministic label/confirmation/impression overrides
  6. write-back report into the patient ledger (updates nadir, watch)          [ledger]

Examples:
  # single timepoint, dry-run to verify wiring
  python3 run_report.py --dataset HCC-TACE-Seg --patient HCC_001 \
        --timepoint 001_followup_1 --backend dryrun

  # whole patient timeline in order
  python3 run_report.py --dataset HCC-TACE-Seg --patient HCC_001 --all --backend dryrun
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import measurements as M
import clinical as C
from ledger import Ledger
from prompts import (SYSTEM_REPORT, build_report_prompt,
                     SYSTEM_FUSE, build_fuse_prompt,
                     SYSTEM_QC, build_qc_prompt)
from vlm import VLMClient, parse_report
from findings_2d import load_label_set, run_findings_2d, verify_findings_2d
from fusion import triangulate, recist_hint, verify_findings_3d

HERE = Path(__file__).resolve().parent
# Override on another machine with:  export PROC_ROOT=/path/to/CTProcessed
DEFAULT_PROC = os.environ.get("PROC_ROOT", "/media/cbtil3/WhiteSD/CTProcessed")
# Reports saved with the datasets on the same SSD (sibling of CTProcessed).
DEFAULT_OUT = os.environ.get("REPORTS", "/media/cbtil3/WhiteSD/CTProcessed-Reports")
LUNG_DATASETS = {"NLST", "RIDER"}


def infer_phase(stage_name: str) -> str:
    s = stage_name.lower()
    if "baseline" in s or s.startswith("000"):
        return "baseline"
    return "follow-up"


def process_timepoint(proc_ds, pdir, stage_dir, phase, position, ledger, client,
                      out_dir, save_slices, do_qc, label_set=None, use_fuse=True,
                      client_2d=None, clin=None, window="soft"):
    patient = ledger.data["patient"]
    # ---- 1. evidence S1: measurements + deltas vs prior/baseline/nadir ------
    meas = M.measure_stage(proc_ds, patient, stage_dir)
    mem = ledger.retrieve(stage_dir.name)
    base_m = mem["baseline"]["measurements"] if mem["baseline"] else None
    nad_m = mem["nadir"]["measurements"] if mem["nadir"] else None
    prior_m = mem["prior"]["measurements"] if mem["prior"] else None
    meas = M.with_deltas(meas, base_m, nad_m)
    meas = M.with_prior_delta(meas, prior_m)
    has_prior = mem["prior"] is not None

    # ---- 1a. verify the 3D numbers (S1): a bad mask must NOT silently drive the
    # trusted RECIST label — demote to advisory if the mask is implausible. -----
    mqc = verify_findings_3d(meas, prior_m)
    if mqc.get("checked") and not mqc.get("seg_reliable"):
        meas["seg_unreliable"] = True
    meas["measurement_qc"] = mqc

    # ---- 1b. timing (graceful: None unless a clinical record exists) ----------
    prior_stage_name = mem["prior"]["stage"] if has_prior else None
    tinfo = C.timing(clin, stage_dir.name, prior_stage_name)
    meas["days_from_baseline"] = tinfo.get("days_from_baseline")
    meas["days_since_prior"] = tinfo.get("days_since_prior")
    meas["stage_type"] = tinfo.get("stage_type")
    if tinfo.get("weeks_since_treatment") is not None:
        meas["weeks_since_treatment"] = tinfo["weeks_since_treatment"]

    # ---- 2. evidence: co-registered prior|current|difference CT slices --------
    images = []
    prior_dir = pdir / mem["prior"]["stage"] if mem["prior"] else None
    try:
        from imaging import comparison_slices
        tmask_path, _ = M.tumor_mask_path(proc_ds, patient, stage_dir)
        organ_path = M.organ_seg_path(proc_ds, patient, stage_dir.name)
        images = comparison_slices(stage_dir, prior_dir, tmask_path, organ_path, window,
                                   save_dir=(out_dir / "slices") if save_slices else None)
    except Exception as e:
        print(f"  [warn] slice extraction skipped: {e}")

    # ---- 2b. evidence S2: INDEPENDENT 2D image read (no numbers) -------------
    findings2d = None
    if use_fuse and label_set is not None:
        try:
            findings2d = run_findings_2d(client_2d or client, images, label_set,
                                         has_prior=has_prior)
            findings2d = verify_findings_2d(client_2d or client, images, findings2d)
        except Exception as e:
            print(f"  [warn] 2D image-read (S2) skipped: {e}")

    # ---- 2c. deterministic triangulation S1 vs S2 + RECIST hint --------------
    tri = triangulate(meas, findings2d, has_prior=has_prior)
    hint = recist_hint(meas, findings2d, has_prior)

    # ---- 3. assemble context & generate --------------------------------------
    ctx = {
        "dataset": ledger.data["dataset"], "patient": patient,
        "stage": stage_dir.name, "phase": phase, "position": position,
        "treatment_timeline": mem["treatment_timeline"],
        "trajectory": mem["trajectory"],
        "prior_findings": mem["prior_findings"],
        "prior_watch_items": (mem["prior"] or {}).get("watch_items", []),
        "baseline_report": Ledger.report_text(mem["baseline"]),
        "nadir_report": Ledger.report_text(mem["nadir"]),
        "measurements_json": meas,
        "n_images": len(images), "image_labels": [l for l, _ in images],
        "clinical_summary": C.patient_summary(clin),
        "timing": tinfo,
    }
    if use_fuse and findings2d is not None:
        sys_used = SYSTEM_FUSE
        user = build_fuse_prompt(ctx, findings2d, tri, hint, is_baseline=not has_prior)
    else:
        sys_used = SYSTEM_REPORT
        user = build_report_prompt(ctx)
    raw = client.generate(sys_used, user, images)
    report, think = parse_report(raw)

    # ---- 4. QC / verification pass (+ one retry) ------------------------------
    qc = None
    if do_qc and not report.get("_parse_error"):
        evidence = {"measurements": meas, "trajectory": mem["trajectory"],
                    "treatment_timeline": mem["treatment_timeline"],
                    "prior_findings": mem["prior_findings"],
                    "recist_label_hint": hint, "is_baseline": not has_prior}
        qc_raw = client.generate(SYSTEM_QC, build_qc_prompt(report, evidence), [])
        qc, _ = parse_report(qc_raw)
        if qc and qc.get("passed") is False:
            user2 = user + "\n\n[QC FEEDBACK — fix these and regenerate]\n" + \
                json.dumps(qc.get("issues", []), ensure_ascii=False)
            report, think = parse_report(client.generate(sys_used, user2, images))

    report["phase"] = phase
    report["timeline_position"] = position
    report["measurements"] = meas
    report["triangulation"] = tri
    report["recist_hint"] = hint
    # ---- enforce the trustworthy label: when the RECIST hint is CERTAIN (seg-based),
    # the number call is authoritative and overrides the VLM; for non-seg keep the
    # model's label but fall back to the hint if it is missing/invalid. ----
    VALID = {"CR", "PR", "SD", "PD", "baseline"}
    if not has_prior:
        report["progression_label"] = "baseline"
    elif hint.get("certain"):
        report["progression_label"] = hint["label"]
    elif report.get("progression_label") not in VALID:
        report["progression_label"] = (hint.get("label") or "SD").replace("?", "") or "SD"
    # ---- deterministic RECIST confirmation: a fresh PD/PR/CR is UNCONFIRMED until
    # a subsequent scan sustains it; confirmed only if the prior visit had the same
    # label (two consecutive). ----
    cur_label = report.get("progression_label")
    prev_label = (mem["prior"] or {}).get("progression_label")
    if not has_prior or cur_label in (None, "SD", "baseline"):
        report["confirmation_status"] = "n/a"
    elif prev_label == cur_label:
        report["confirmation_status"] = "confirmed"
    else:
        report["confirmation_status"] = "unconfirmed"
    # ---- trustworthy core: numbers from the mask, never eyeballed by the VLM ----
    prior_stage = mem["prior"]["stage"] if has_prior else None
    report["target_lesions"] = M.build_target_lesions(meas, prior_m)
    fact = M.factual_impression(meas, prior_stage, report.get("progression_label"),
                                has_prior)
    report["impression_model"] = (report.get("impression") or "").strip()
    report["impression"] = fact if meas.get("burden_metric") == "tumor_vol_cc" \
        else (fact + " " + report["impression_model"]).strip()
    # ---- timing + treatment-effect window (deterministic from clinical) ----
    report["timing"] = tinfo
    w = tinfo.get("weeks_since_treatment")
    in_window = (w is not None and 0 <= w <= 8)
    report["in_treatment_effect_window"] = in_window
    if isinstance(report.get("confounders"), dict):
        if w is not None:
            report["confounders"]["weeks_since_treatment"] = w
        if in_window and report.get("progression_label") == "PD":
            report["confounders"]["treatment_effect_risk"] = "high"
    if findings2d is not None:
        report["image_findings"] = {k: v for k, v in findings2d.items() if k != "_think"}
    report["_qc_passed"] = (qc.get("passed") if qc else None)

    # ---- 5. write-back into patient memory ------------------------------------
    ledger.add_visit(report, meas)
    ledger.save()

    stub = f"{ledger.data['dataset']}__{patient}__{stage_dir.name}"
    (out_dir / f"{stub}.report.json").write_text(json.dumps(report, indent=2))
    from report_md import render_md
    (out_dir / f"{stub}.report.md").write_text(
        render_md(report, ledger.data["dataset"], patient, ctx.get("clinical_summary", "")))
    (out_dir / f"{stub}.context.txt").write_text(user)
    if think:
        (out_dir / f"{stub}.think.txt").write_text(think)
    if qc is not None:
        (out_dir / f"{stub}.qc.json").write_text(json.dumps(qc, indent=2))
    if findings2d is not None:
        (out_dir / f"{stub}.findings2d.json").write_text(
            json.dumps({k: v for k, v in findings2d.items() if k != "_think"}, indent=2))
    seg_flag = "" if mqc.get("seg_reliable", True) or not mqc.get("checked") \
        else f"  3dQC!={mqc.get('warnings')}"
    print(f"  [{stage_dir.name}] organ={meas.get('organ','?')} -> label="
          f"{report.get('progression_label')} "
          f"({report.get('confirmation_status', '')})  "
          f"vol={meas.get('tumor_vol_cc')}cc diam={meas.get('longest_diameter_mm')}mm "
          f"trust={tri.get('trust')} agree={tri.get('agreement')}  "
          f"qc={qc.get('passed') if qc else 'off'}{seg_flag}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--patient", required=True)
    ap.add_argument("--proc-root", default=DEFAULT_PROC)
    ap.add_argument("--timepoint", default="", help="stage dir name or index (single)")
    ap.add_argument("--phase", default="", help="free-form phase label (default: derived)")
    ap.add_argument("--all", action="store_true", help="process the whole timeline")
    ap.add_argument("--phases", default="", help="comma list aligned to stages (--all)")
    # DEFAULT RUN = the real combine: MedGemma (S3 fusion/RECIST) + Qwen2.5-VL (S2
    # image read), in-process via transformers. Use --backend dryrun for a no-GPU
    # wiring check, or --backend endpoint to hit served vLLM models.
    # S3 fusion + QC reasoner
    ap.add_argument("--backend", default="hf", choices=["dryrun", "hf", "endpoint"])
    ap.add_argument("--model", default="google/medgemma-4b-it")
    ap.add_argument("--endpoint", default="")
    # S2 2D image-read. Different model from S3 => its own client is loaded.
    ap.add_argument("--backend-2d", default="",
                    choices=["", "dryrun", "hf", "endpoint"],
                    help="backend for the 2D image-read pass (default: same as --backend)")
    ap.add_argument("--model-2d", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--endpoint-2d", default="")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--save-slices", action="store_true")
    ap.add_argument("--events", default="",
                    help="JSON file: treatment timeline [{date/stage,event,detail}]")
    ap.add_argument("--qc", action="store_true",
                    help="run the verification pass + one retry (recommended)")
    ap.add_argument("--no-2d", action="store_true",
                    help="disable the S2 image-read + fusion; single-pass report only")
    ap.add_argument("--window", default="", choices=["", "soft", "lung"],
                    help="CT display window (default: lung for NLST/RIDER, else soft)")
    ap.add_argument("--labels", default="",
                    help="path to the S2 finding taxonomy (default: labels_abdomen.json)")
    args = ap.parse_args()

    use_fuse = not args.no_2d
    label_set = load_label_set(Path(args.labels)) if args.labels else \
        (load_label_set() if use_fuse else None)
    window = args.window or ("lung" if args.dataset in LUNG_DATASETS else "soft")

    proc_ds = Path(args.proc_root) / args.dataset
    pdir = proc_ds / args.patient
    if not pdir.is_dir():
        raise SystemExit(f"patient dir not found: {pdir}")
    stages = M.stage_dirs(pdir)
    if not stages:
        raise SystemExit(f"no timepoint dirs under {pdir}")

    out_dir = Path(args.out_dir) / args.dataset / args.patient
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out_dir / "ledger.json", args.dataset, args.patient)
    clin = C.load_clinical(args.dataset, args.patient)
    if args.events and Path(args.events).exists():
        ledger.set_treatment_timeline(json.loads(Path(args.events).read_text()))
    elif not ledger.data.get("treatment_timeline"):
        ledger.set_treatment_timeline(C.treatment_timeline(clin))
    client = VLMClient(args.backend, args.model, args.endpoint or None)
    # Reuse the SAME loaded model for S2 when it matches S3's backend+model.
    b2 = args.backend_2d or args.backend
    if not use_fuse:
        client_2d = None
    elif b2 == args.backend and args.model_2d == args.model and \
            (args.endpoint_2d or args.endpoint) == args.endpoint:
        client_2d = client
    else:
        client_2d = VLMClient(b2, args.model_2d, args.endpoint_2d or None)

    if args.all:
        phase_list = [p.strip() for p in args.phases.split(",")] if args.phases else []
        for i, st in enumerate(stages):
            phase = (phase_list[i] if i < len(phase_list) and phase_list[i]
                     else infer_phase(st.name))
            process_timepoint(proc_ds, pdir, st, phase, f"visit {i+1} of {len(stages)}",
                               ledger, client, out_dir, args.save_slices,
                               args.qc, label_set, use_fuse, client_2d, clin, window)
    else:
        if args.timepoint.isdigit():
            st = stages[int(args.timepoint)]
        else:
            st = next((s for s in stages if s.name == args.timepoint), None)
        if st is None:
            raise SystemExit(f"timepoint '{args.timepoint}' not found; "
                             f"available: {[s.name for s in stages]}")
        idx = stages.index(st)
        phase = args.phase or infer_phase(st.name)
        process_timepoint(proc_ds, pdir, st, phase, f"visit {idx+1} of {len(stages)}",
                           ledger, client, out_dir, args.save_slices,
                           args.qc, label_set, use_fuse, client_2d, clin, window)
    print(f"ledger → {ledger.path}")


if __name__ == "__main__":
    main()
