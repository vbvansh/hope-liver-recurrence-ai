"""Prompts + schemas for AUTHENTIC longitudinal tumor-progression reporting on CT.

Three model passes (all inference-time, no training):
  FINDINGS2D : read co-registered prior/current/difference CT slices, NO numbers
               -> MR-RATE-style structured findings (S2, independent source)
  FUSE       : combine trusted measurements (S1) + image read (S2) + memory
               -> grounded findings + RECIST 1.1 label, with per-claim provenance
  QC         : verify the report against the evidence -> pass | issues (one retry)

Guideline is RECIST 1.1 for solid abdominal/thoracic tumors:
  - target lesion size = longest axial diameter (mm); we also carry volume (cc).
  - PD = >=20% AND >=5mm increase in the longest diameter vs the NADIR, OR any new
    lesion / unequivocal new metastasis.
  - PR = >=30% decrease vs the BASELINE.   CR = disappearance of all target disease.
  - otherwise SD. New PD not yet confirmed by a follow-up scan => "PD (unconfirmed)".
"""

SYSTEM_REPORT = """You are a body radiologist writing a LONGITUDINAL follow-up report for
oncologic CT (liver / kidney / pancreas / lung tumors), using RECIST 1.1.

You are given, for THIS timepoint:
  - the treatment TIMELINE (events + dates, if any) and the scan's PHASE,
  - co-registered KEY CT SLICES: prior, current, and their difference,
  - TRUSTED quantitative measurements (lesion volume + longest diameter from
    segmentation), including the target-lesion size tracked across visits,
  - the TRAJECTORY (burden over all prior visits, the nadir) and prior findings.

Hard rules:
  1. COMPARE to the prior scan AND to the nadir AND to baseline — the impression
     must open with an explicit comparison ("Compared to <prior>, ...").
  2. TRUST the provided numbers; never re-estimate sizes from pixels. Use images
     only for morphology: new lesions, lymphadenopathy, metastases, vascular/organ
     invasion, necrosis, hemorrhage, treatment effect.
  3. Track the TARGET LESION by its given id across visits; describe its change.
  4. Apply RECIST 1.1: >=20% AND >=5mm increase in longest diameter vs nadir OR any
     new lesion => PD; >=30% decrease vs baseline => PR; disappearance of all target
     disease => CR; otherwise SD. New PD not confirmed by a later scan => unconfirmed.
  5. Consider TREATMENT EFFECT (e.g. post-TACE necrosis/lipiodol, devascularization)
     before calling true progression; lower confidence when uncertain.
  6. State only what the evidence supports. No invented lesions or measurements.

Reason step by step inside <think>...</think> (restate comparison set; target-lesion
change; aggregate vs nadir/baseline; new lesions; treatment-effect check; trajectory
reconciliation; label + confirmation + confidence), then output ONLY the JSON object."""

REPORT_SCHEMA = """{
  "visit_id": "<stage id>",
  "phase": "<treatment phase or stage type>",
  "timeline_position": "<e.g. 'visit 3 of 7'>",
  "comparison": {"baseline": "<id>", "nadir": "<id>", "prior": "<id>"},
  "measurements": { ...echo provided numbers for audit... },
  "target_lesions": [
    {"id": "L1", "site": "<organ tumor>", "now": <num>, "prior": <num|null>,
     "now_diam_mm": <num|null>, "prior_diam_mm": <num|null>,
     "trend": "increasing|decreasing|stable|new|resolved",
     "delta_vs_nadir_pct": <num|null>}
  ],
  "findings": {
    "target_tumor": "<comparison-first str>",
    "new_lesions": "<str>",
    "nodes_metastasis": "<lymphadenopathy / distant metastasis>",
    "invasion_vascular": "<organ or vascular invasion / thrombus>",
    "necrosis_hemorrhage": "<necrosis / hemorrhage / treatment effect>"
  },
  "confounders": {"treatment_effect_risk": "low|moderate|high",
                  "weeks_since_treatment": <num|null>, "rationale": "<str>"},
  "progression_label": "CR | PR | SD | PD",
  "recist_basis": "<the measurement that drives the label, e.g. '+24% diameter vs nadir (38->47 mm)'>",
  "confirmation_status": "n/a | unconfirmed | confirmed",
  "confidence": 0.0,
  "impression": "<prose; MUST open with 'Compared to <prior>, ...'; state the decisive finding, the trajectory, and caveats>",
  "recommendation": "<clinical next step: confirmatory scan / interval / biopsy, etc.>",
  "carry_forward": {
    "updated_nadir": {"id": "<id>", "value": <num|null>},
    "target_lesions_to_track": ["L1", "..."],
    "watch_items": ["<str>", "..."],
    "pending_confirmation": false
  }
}"""

SYSTEM_QC = """You are a radiology-report QC auditor. Given the EVIDENCE (measurements,
target-lesion deltas, treatment timeline, prior trajectory) and a generated
REPORT, verify the report. Check ONLY against the evidence:
  - hallucination: does the report assert a lesion/measurement not in evidence?
  - omission: does evidence show a NEW lesion or >=20% nadir increase the report missed?
  - label consistency: is progression_label consistent with the RECIST thresholds
    applied to the numbers? (and CONFIRMED only if a prior visit had the same label)
  - treatment effect: if within a post-therapy window, was it considered?
  - comparison: does the impression actually compare to the prior?
Output ONLY JSON: {"passed": true|false, "issues": ["..."],
"suggested_label": "<CR|PR|SD|PD|unchanged>"}"""


# ---------------------------------------------------------------------------
# S2 — 2D IMAGE-READ pass (MR-RATE-style structured findings FROM THE IMAGE).
# Given ONLY the slices, NOT the numbers, so it is an INDEPENDENT source.
# ---------------------------------------------------------------------------
SYSTEM_FINDINGS2D = """You are a body radiologist reading oncologic CT slices, following the MR-RATE
discipline: ground every PRESENT call in what is actually visible, then self-verify
and DROP anything you are not sure you can see. The slice shown is the MOST
PROMINENT TUMOR plane (chosen from the lesion / organ mask); its label says the
plane and source, e.g. 'ct:current@z75[tumor]'. The SAME z is the SAME anatomical
plane across timepoints (volumes are co-registered). The CT uses a fixed HU window,
so brightness is comparable between prior and current.

You are NOT given any measurements — never invent numeric sizes. Judge only from the
pixels, and report your visual confidence honestly.

Process (MR-RATE style):
  1. THINK: for each label, name the exact slice and region you see it on (or state
     you do NOT see it).
  2. VERIFY: re-examine every label you marked present=1 and DOWNGRADE to present=0
     if the image does not clearly support it. Only removal is allowed. Default to
     ABSENT when unsure.

Strict inference rules (do NOT chain one finding into another):
  - peritumoral fat stranding or fluid does NOT imply tumor growth.
  - low-attenuation within the lesion may be NECROSIS or post-treatment change, not
    necessarily larger tumor — mark necrosis / treatment_effect, not new tumor.
  - a node is only lymphadenopathy if it looks enlarged/rounded; do not over-call.
  - only mark new_lesion / distant_metastasis if you see a lesion on CURRENT that is
    absent on PRIOR.
  - vascular_invasion needs tumor seen WITHIN/abutting a vessel, not mere proximity.

For EACH label in the LABEL SET output:
  - present: 1 only if clearly visible on the CURRENT slice, else 0
  - change : {new, increased, stable, decreased, resolved, absent} vs the PRIOR slice.
             If NO prior slice is shown, you cannot judge change — use "na".
  - confidence: 0.0-1.0 (visual certainty)
  - evidence: the slice label you saw it on + short note
Also give an overall IMAGE-DERIVED burden direction (increased|stable|decreased):
your independent visual impression of whether the tumor grew vs the prior slice,
ignoring numbers. If NO prior is shown, set it to "stable".

Reason in <think> (step 1 then step 2), then output ONLY the JSON object."""

SYSTEM_FINDINGS2D_VERIFY = """You are auditing a prior 2D CT image-read for FALSE POSITIVES, MR-RATE style.
You are shown the SAME slices and the list of findings the first pass marked
PRESENT. For EACH, look again and decide KEEP (the image clearly shows it) or DROP
(over-called, inferred from another finding, or not visible). You may ONLY remove or
weaken; never add a finding and never raise a confidence. Apply the strict inference
rules (fat stranding≠growth, necrosis≠new tumor, proximity≠vascular invasion).
Output ONLY JSON: {"verified": {"<label_id>": {"keep": 0|1, "confidence": 0.0-1.0,
"reason": "<short>"}}}. Think briefly in <think> first."""


def build_findings2d_verify_prompt(present_labels: list, image_labels: list) -> str:
    import json
    return "\n\n".join([
        _section("FINDINGS MARKED PRESENT (re-check each)",
                 json.dumps(present_labels, ensure_ascii=False)),
        _section("KEY SLICES ATTACHED", f"{image_labels}"),
        _section("TASK", "KEEP only what the image clearly supports; DROP false "
                 "positives. Output the verification JSON."),
    ])


FINDINGS2D_SCHEMA = """{
  "image_burden_direction": "increased | stable | decreased",
  "image_confidence": 0.0,
  "labels": {
    "<label_id>": {"present": 0, "change": "stable", "confidence": 0.0, "evidence": "<str>"}
  },
  "findings": {
    "target_tumor": "<what you see, current vs prior>",
    "new_lesions": "<str>",
    "nodes_metastasis": "<str>",
    "invasion_vascular": "<str>",
    "necrosis_hemorrhage": "<str>"
  },
  "uncertainty": "<what the images could not resolve (e.g. needs contrast phase)>"
}"""


# ---------------------------------------------------------------------------
# S3 — FUSION pass: combine trusted NUMBERS (S1) + independent IMAGE READ (S2)
# + memory into ONE report, with per-claim PROVENANCE and triangulated trust.
# ---------------------------------------------------------------------------
SYSTEM_FUSE = """You are a body radiologist producing the FINAL longitudinal report by FUSING
two independent evidence sources, under RECIST 1.1:

  S1 = QUANTITATIVE measurements (3D). Reliability is given per metric:
       'segmentation' = trust the size/%-change (volume + longest diameter);
       'none' = no lesion mask — DO NOT state a size, lean on the image read.
  S2 = an INDEPENDENT 2D IMAGE READ (findings + per-label change + confidence),
       produced WITHOUT seeing the numbers.

Fusion rules — this is how the report earns trust:
  1. QUANTITATIVE claims (volume, diameter, %-change vs prior/baseline/nadir) come
     from S1 when its reliability is 'segmentation'. If reliability is 'none',
     describe magnitude qualitatively and LOWER confidence.
  2. MORPHOLOGY / radiology factors (new lesions, lymphadenopathy, metastases,
     vascular/organ invasion, necrosis, hemorrhage, treatment-effect pattern,
     ascites/effusion) come from S2 (the image read).
  3. TRIANGULATE: if S1's direction (sign of Δ) AGREES with S2's image burden
     direction, raise confidence. If they CONFLICT, say so in 'conflicts', pick the
     better-supported side, and lower confidence.
  4. Tag every finding with its source in 'provenance'. Never assert a number S1 did
     not provide; never assert morphology S2 did not see.
  5. Apply RECIST 1.1 for the label (>=20% AND >=5mm diameter increase vs nadir OR any
     new measurable lesion => PD; >=30% decrease vs baseline => PR; all target disease
     gone => CR; else SD). A deterministic RECIST label hint computed from the trusted
     numbers is provided under 'RECIST LABEL GUIDANCE'; when burden reliability is
     'segmentation' that hint is AUTHORITATIVE for the size-based label — do not
     overturn it from image impression alone (only a genuinely new measurable lesion
     justifies escalating to PD). PD not yet confirmed by a later scan => 'unconfirmed'.
     Consider treatment effect from the TIMELINE. At BASELINE use label "baseline" and
     make no comparison.
  6. The final confidence must reflect: data reliability x cross-source agreement x
     treatment-effect risk.

Reason in <think>...</think> (reconcile S1 vs S2; reliability; agreement/conflict;
RECIST label; confidence), then output ONLY the JSON object matching the schema."""

FUSE_SCHEMA = """{
  "visit_id": "<stage id>", "phase": "<phase/stage type>", "timeline_position": "<str>",
  "comparison": {"baseline": "<id>", "nadir": "<id>", "prior": "<id>"},
  "measurements": { ...echo S1 numbers used... },
  "target_lesions": [
    {"id": "L1", "site": "<str>", "now": <num|null>, "prior": <num|null>,
     "now_diam_mm": <num|null>, "prior_diam_mm": <num|null>,
     "trend": "increasing|decreasing|stable|new|resolved", "delta_vs_nadir_pct": <num|null>}
  ],
  "findings": {
    "target_tumor": "<comparison-first>", "new_lesions": "<str>",
    "nodes_metastasis": "<str>", "invasion_vascular": "<str>", "necrosis_hemorrhage": "<str>"
  },
  "provenance": {
    "target_tumor": "S1+S2 | S1 | S2", "new_lesions": "S2", "burden": "S1(seg) | S2"
  },
  "triangulation": {
    "s1_direction": "up|down|flat|unknown", "s2_direction": "increased|stable|decreased",
    "agreement": "agree | conflict | partial | s1-only | s2-only",
    "burden_reliability": "segmentation | none"
  },
  "conflicts": ["<explicit S1-vs-S2 disagreement, if any>"],
  "confounders": {"treatment_effect_risk": "low|moderate|high", "weeks_since_treatment": <num|null>, "rationale": "<str>"},
  "progression_label": "CR|PR|SD|PD", "recist_basis": "<decisive metric>",
  "confirmation_status": "n/a|unconfirmed|confirmed", "confidence": 0.0,
  "impression": "<prose; opens with 'Compared to <prior>, ...'; states decisive finding, the source it rests on, and caveats>",
  "recommendation": "<next step>",
  "carry_forward": {"updated_nadir": {"id": "<id>", "value": <num|null>},
                    "target_lesions_to_track": ["L1"], "watch_items": ["<str>"],
                    "pending_confirmation": false}
}"""


def _section(title, body):
    return f"[{title}]\n{body}"


def build_report_prompt(ctx: dict) -> str:
    import json
    return "\n\n".join([
        _section("PATIENT", f"{ctx['dataset']} / {ctx['patient']}"),
        _section("CURRENT TIMEPOINT",
                 f"stage={ctx['stage']}  phase={ctx['phase']}  position={ctx['position']}"),
        _section("TREATMENT TIMELINE",
                 json.dumps(ctx.get("treatment_timeline", []), ensure_ascii=False)),
        _section("TRAJECTORY (burden over prior visits, nadir)",
                 ctx.get("trajectory", "(first timepoint)")),
        _section("PRIOR FINDINGS (most recent)", ctx.get("prior_findings", "(none)")),
        _section("PRIOR WATCH ITEMS",
                 json.dumps(ctx.get("prior_watch_items", []), ensure_ascii=False)),
        _section("BASELINE IMPRESSION", ctx.get("baseline_report", "(none)")),
        _section("NADIR IMPRESSION", ctx.get("nadir_report", "(none)")),
        _section("CURRENT MEASUREMENTS — trust these",
                 json.dumps(ctx.get("measurements_json", {}), indent=2, ensure_ascii=False)),
        _section("KEY SLICES",
                 f"{ctx.get('n_images', 0)} co-registered CT slices attached: "
                 f"{ctx.get('image_labels', [])} (prior / current / difference)."),
        _section("TASK",
                 "Produce the visit report JSON, schema exactly:\n" + REPORT_SCHEMA),
    ])


def build_findings2d_prompt(label_set: list, image_labels: list,
                            has_prior: bool = True) -> str:
    import json
    compact = [{"id": l["id"], "name": l["name"], "desc": l.get("desc", "")}
               for l in label_set]
    prior_note = ("Both PRIOR and CURRENT slices are shown — compare them for 'change'."
                  if has_prior else
                  "This is the BASELINE study: ONLY a current slice is shown, NO prior. "
                  "You cannot judge change — set every 'change' to 'na' and burden "
                  "direction to 'stable'. Report PRESENCE only.")
    return "\n\n".join([
        _section("LABEL SET (decide each)", json.dumps(compact, ensure_ascii=False, indent=0)),
        _section("KEY SLICES ATTACHED",
                 f"{image_labels}\n{prior_note} NO numbers given — read the images."),
        _section("TASK", "Output the image-read JSON, schema exactly:\n" + FINDINGS2D_SCHEMA),
    ])


def build_fuse_prompt(ctx: dict, findings2d: dict, tri: dict,
                      hint: dict = None, is_baseline: bool = False) -> str:
    import json
    hint = hint or {}
    if is_baseline:
        pos_note = ("This is the BASELINE study (first timepoint): there is NO prior, "
                    "nadir, or trajectory. Do NOT compare or claim any change. Produce a "
                    "baseline CHARACTERIZATION of the tumor. Set progression_label exactly "
                    "to \"baseline\", confirmation_status \"n/a\", and open the impression "
                    "with \"Baseline study; no prior for comparison.\" Leave delta fields null.")
    else:
        pos_note = ("Deterministic RECIST-1.1 label SUGGESTION from the trusted numbers: "
                    f"label={hint.get('label')} (basis: {hint.get('basis')}; "
                    f"certain={hint.get('certain')}). When burden reliability is "
                    "'segmentation' this size-based call is AUTHORITATIVE — adopt it unless "
                    "the IMAGE shows a new measurable lesion the numbers missed; you may not "
                    "flip it to PD on image impression alone. When reliability is 'none' the "
                    "hint is soft (no true size) — weigh S2 more.")
    tinfo = ctx.get("timing", {}) or {}
    timing_str = (f"stage_type={tinfo.get('stage_type')}; "
                  f"days_from_baseline={tinfo.get('days_from_baseline')}; "
                  f"days_since_prior={tinfo.get('days_since_prior')}; "
                  f"weeks_since_treatment={tinfo.get('weeks_since_treatment')} "
                  "(recent post-therapy change, e.g. post-TACE necrosis/lipiodol, can mimic "
                  "or mask tumor — weigh treatment effect, lower confidence if unsure)")
    return "\n\n".join([
        _section("PATIENT", f"{ctx['dataset']} / {ctx['patient']}"),
        _section("CLINICAL CONTEXT", ctx.get("clinical_summary", "(none)")),
        _section("CURRENT TIMEPOINT",
                 f"stage={ctx['stage']}  phase={ctx['phase']}  position={ctx['position']}"),
        _section("SCAN TIMING", timing_str),
        _section("RECIST LABEL GUIDANCE", pos_note),
        _section("TREATMENT TIMELINE",
                 json.dumps(ctx.get("treatment_timeline", []), ensure_ascii=False)),
        _section("TRAJECTORY (burden over prior visits, nadir)",
                 ctx.get("trajectory", "(first timepoint)")),
        _section("PRIOR FINDINGS (most recent)", ctx.get("prior_findings", "(none)")),
        _section("BASELINE IMPRESSION", ctx.get("baseline_report", "(none)")),
        _section("NADIR IMPRESSION", ctx.get("nadir_report", "(none)")),
        _section("S1 — QUANTITATIVE MEASUREMENTS (3D)",
                 json.dumps(ctx.get("measurements_json", {}), indent=2, ensure_ascii=False)),
        _section("S1 — BURDEN RELIABILITY",
                 tri.get("burden_reliability", "unknown") +
                 "  (segmentation = trust sizes; none = no mask, qualitative only)"),
        _section("S2 — INDEPENDENT 2D IMAGE READ",
                 json.dumps(findings2d, indent=2, ensure_ascii=False)),
        _section("TRIANGULATION (pre-computed)",
                 json.dumps(tri, ensure_ascii=False)),
        _section("TASK",
                 "FUSE S1 and S2 into the final report JSON, schema exactly:\n" + FUSE_SCHEMA),
    ])


def build_qc_prompt(report: dict, evidence: dict) -> str:
    import json
    return "\n\n".join([
        _section("EVIDENCE", json.dumps(evidence, indent=2, ensure_ascii=False)),
        _section("REPORT", json.dumps(report, indent=2, ensure_ascii=False)),
        _section("TASK", "Verify and output the QC JSON."),
    ])
