"""Render a report.json into ONE clean, human-readable Markdown report.

The report is the FINAL, reader-facing artifact: current lesion status + how it
compares to baseline, the previous timepoint, and the nadir (multi-directional),
plus the RECIST 1.1 call, the impression, the image findings, and a reliability
line. The .context.txt / .findings2d.json / ledger.json are intermediate files.
"""
from __future__ import annotations
from typing import Dict


def _fmt_cc(v):
    return f"{v:.1f} cc" if isinstance(v, (int, float)) else "—"


def _fmt_mm(v):
    return f"{v:.0f} mm" if isinstance(v, (int, float)) else "—"


def _delta(cur, ref_pct, ref_abs):
    """A readable change cell handling from-zero / to-zero / missing."""
    if cur is None or ref_abs is None:
        return "—"
    if ref_abs == 0:
        return "**new** (was 0)" if cur >= 0.5 else "≈0 (was 0)"
    if cur < 0.5 and ref_abs >= 0.5:
        return f"**resolved** (was {ref_abs:.1f} cc)"
    if ref_pct is None:
        return "—"
    arrow = "▲" if ref_pct >= 20 else ("▼" if ref_pct <= -30 else "▬")
    return f"{arrow} {ref_pct:+.0f}%"


def render_md(report: Dict, dataset: str = "", patient: str = "",
              clinical_summary: str = "") -> str:
    m = report.get("measurements", {}) or {}
    tri = report.get("triangulation", {}) or {}
    conf = report.get("confounders", {}) or {}
    tinfo = report.get("timing", {}) or {}
    label = report.get("progression_label", "—")
    status = report.get("confirmation_status", "")
    stage = m.get("stage") or report.get("visit_id", "")

    metric_is_seg = m.get("burden_metric") == "tumor_vol_cc"
    cur = m.get("lesion_vol_cc")
    curd = m.get("longest_diameter_mm")
    badge = {"PD": "🔴 PD", "PR": "🟢 PR", "CR": "🟢 CR", "SD": "🟡 SD",
             "baseline": "⚪ baseline"}.get(label, label)

    L = []
    L.append("# Tumor-Progression Report (CT · RECIST 1.1)")
    L.append(f"**{dataset} / {patient}** — timepoint **{stage}**  ·  "
             f"{report.get('timeline_position', '')}")
    meta = [f"phase: {report.get('phase', '—')}"]
    if tinfo.get("stage_type"):
        meta.append(f"type: {tinfo['stage_type']}")
    if m.get("organ"):
        meta.append(f"organ: {m['organ']}")
    if tinfo.get("days_from_baseline") is not None:
        meta.append(f"day {tinfo['days_from_baseline']} from baseline")
    if tinfo.get("days_since_prior") is not None:
        meta.append(f"+{tinfo['days_since_prior']}d since prior")
    L.append("  ·  ".join(meta))
    if clinical_summary and clinical_summary not in ("(no clinical record found)", ""):
        L.append(f"_Clinical:_ {clinical_summary}")
    L.append("")

    # ---- the verdict ----
    conf_txt = f" — {status}" if status and status != "n/a" else ""
    L.append(f"## RECIST 1.1:  {badge}{conf_txt}")
    if report.get("recist_basis"):
        L.append(f"**Basis:** {report['recist_basis']}")
    L.append("")

    # ---- multi-directional lesion status ----
    if metric_is_seg:
        L.append("## Target-lesion burden")
        L.append("| measure | now | vs previous | vs baseline | vs nadir |")
        L.append("|---|---|---|---|---|")
        L.append(f"| **volume** | **{_fmt_cc(cur)}** | "
                 f"{_delta(cur, m.get('delta_vs_prior_pct'), m.get('prior_burden'))} | "
                 f"{_delta(cur, m.get('delta_vs_baseline_pct'), m.get('baseline_burden'))} | "
                 f"{_delta(cur, m.get('delta_vs_nadir_pct'), m.get('nadir_burden'))} |")
        L.append(f"| **longest diameter (RECIST)** | **{_fmt_mm(curd)}** | "
                 f"{_delta(curd, m.get('diam_delta_vs_prior_pct'), m.get('prior_diam_mm'))} | "
                 f"{_delta(curd, m.get('diam_delta_vs_baseline_pct'), m.get('baseline_diam_mm'))} | "
                 f"{_delta(curd, m.get('diam_delta_vs_nadir_pct'), m.get('nadir_diam_mm'))} |")
        if m.get("organ_vol_cc") is not None:
            L.append(f"| {m.get('organ', 'organ')} volume | {_fmt_cc(m.get('organ_vol_cc'))} "
                     f"| — | — | — |")
    else:
        L.append("## Lesion status — no lesion mask")
        L.append("No segmentation available at this timepoint; size comes from the "
                 "image read below (qualitative only).")
    L.append("")

    # ---- impression ----
    if report.get("impression"):
        L.append("## Impression")
        L.append(report["impression"])
        L.append("")

    # ---- image findings (the 2D read, morphology) ----
    f = report.get("findings", {}) or {}
    rows = [(k, v) for k, v in f.items()
            if v and str(v).strip().lower() not in ("none", "stub", "n/a", "")]
    if rows:
        L.append("## Image findings (2D read)")
        if tri.get("agreement") == "conflict" and metric_is_seg:
            L.append("> ⚠️ The image read disagrees with the measured size change above "
                     "— **trust the measured (segmentation) change**; the 2D read is a "
                     "secondary, less reliable signal here.")
        nice = {"target_tumor": "Target tumor", "new_lesions": "New lesions",
                "nodes_metastasis": "Nodes / metastasis",
                "invasion_vascular": "Organ / vascular invasion",
                "necrosis_hemorrhage": "Necrosis / hemorrhage / treatment effect"}
        for k, v in rows:
            L.append(f"- **{nice.get(k, k)}:** {v}")
        L.append("")

    # ---- confounders ----
    if conf.get("treatment_effect_risk") or report.get("in_treatment_effect_window"):
        L.append("## Confounders")
        line = f"Treatment-effect risk: **{conf.get('treatment_effect_risk', '—')}**"
        if report.get("in_treatment_effect_window"):
            line += "  ⚠️ recent post-therapy window"
        L.append(line)
        if conf.get("rationale"):
            L.append(f"_{conf['rationale']}_")
        L.append("")

    if report.get("recommendation"):
        L.append("## Recommendation")
        L.append(report["recommendation"])
        L.append("")

    # ---- reliability footer ----
    src = tri.get("burden_reliability", "—")
    rel = ("segmentation (trusted)" if src == "segmentation" else
           "segmentation — UNTRUSTED (3D-QC flagged)" if "untrust" in str(src) else
           "no mask (image-read only)" if src == "none" else src)
    if m.get("tumor_source") == "segvol":
        rel += " · SegVol pre-annotation"
    mqc = (m.get("measurement_qc") or {})
    foot = [f"burden source: {rel}", f"S1↔S2: {tri.get('agreement', '—')}",
            f"trust: {tri.get('trust', '—')}"]
    if mqc.get("warnings"):
        foot.append("seg-QC: " + "; ".join(mqc["warnings"]))
    L.append("---")
    L.append("_" + "  ·  ".join(str(x) for x in foot) + "_")
    return "\n".join(L)
