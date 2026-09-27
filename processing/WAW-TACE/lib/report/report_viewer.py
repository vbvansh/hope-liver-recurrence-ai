#!/usr/bin/env python3
"""GUI to review the generated longitudinal CT reports next to their image evidence.

Per timepoint it shows the ONE clean report and the actual co-registered tumor
slices (prior | current | difference) it was based on, plus the patient's burden
trajectory — so you can judge whether each report is reasonable.

Run:
  cd longitudinal_report
  export PROC_ROOT=/media/cbtil3/WhiteSD/CTProcessed
  export REPORTS=/media/cbtil3/WhiteSD/CTProcessed-Reports
  streamlit run report_viewer.py -- --reports <dir> --processed <dir>
"""
import argparse
import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd

import clinical as C
import measurements as M
from report_md import render_md
from imaging import comparison_slices

_ap = argparse.ArgumentParser(description="Streamlit viewer for longitudinal CT reports.")
_ap.add_argument("--reports", default=os.environ.get("REPORTS", "/media/cbtil3/WhiteSD/CTProcessed-Reports"))
_ap.add_argument("--processed", default=os.environ.get("PROC_ROOT", "/media/cbtil3/WhiteSD/CTProcessed"))
_args, _ = _ap.parse_known_args()
REPORTS = Path(_args.reports)
PROC = Path(_args.processed)
LUNG_DATASETS = {"NLST", "RIDER"}

st.set_page_config(page_title="CT Tumor-Progression Reports", layout="wide")


@st.cache_data(show_spinner=False)
def list_datasets():
    return sorted(d.name for d in REPORTS.iterdir()
                  if d.is_dir() and not d.name.startswith("_"))


@st.cache_data(show_spinner=False)
def list_patients(ds):
    return sorted(p.name for p in (REPORTS / ds).iterdir() if p.is_dir())


@st.cache_data(show_spinner=False)
def load_reports(ds, pt):
    reps = sorted((REPORTS / ds / pt).glob("*.report.json"))
    return [json.loads(r.read_text()) for r in reps]


@st.cache_data(show_spinner=True)
def gen_slices(ds, pt, stage, prior_stage):
    proc_ds = PROC / ds
    sd = proc_ds / pt / stage
    pdir = proc_ds / pt / prior_stage if prior_stage else None
    if not sd.is_dir():
        return []
    try:
        tmask, _ = M.tumor_mask_path(proc_ds, pt, sd)
        organ = M.organ_seg_path(proc_ds, pt, stage)
        window = "lung" if ds in LUNG_DATASETS else "soft"
        return [(lbl, im) for lbl, im in
                comparison_slices(sd, pdir, tmask, organ, window)]
    except Exception as e:
        return [("error", str(e))]


@st.cache_data(show_spinner=False)
def load_summary():
    f = REPORTS / "_train" / "summary.csv"
    return pd.read_csv(f) if f.exists() else None


# ----------------------------------------------------------------------------
st.sidebar.title("🩻 CT Reports")
view = st.sidebar.radio("View", ["Patient", "Overview (all)"])

if view == "Overview (all)":
    st.title("All timepoints — overview")
    df = load_summary()
    if df is None:
        st.warning("no summary.csv yet — run export_training.py")
    else:
        dss = st.multiselect("datasets", sorted(df.dataset.unique()))
        labs = st.multiselect("labels", sorted(df.progression_label.dropna().unique()))
        v = df
        if dss:
            v = v[v.dataset.isin(dss)]
        if labs:
            v = v[v.progression_label.isin(labs)]
        st.caption(f"{len(v)} timepoints")
        st.dataframe(v, use_container_width=True, height=600)
    st.stop()

ds = st.sidebar.selectbox("Dataset", list_datasets())
pt = st.sidebar.selectbox("Patient", list_patients(ds))
reports = load_reports(ds, pt)
if not reports:
    st.warning("no reports for this patient")
    st.stop()
stages = [r.get("measurements", {}).get("stage") or r.get("visit_id") for r in reports]
clin = C.load_clinical(ds, pt)
csum = C.patient_summary(clin)

st.title(f"{ds} / {pt}")
st.caption(csum)

# ---- trajectory ----
traj = []
for r in reports:
    m = r.get("measurements", {})
    traj.append({"stage": m.get("stage"),
                 "volume_cc": m.get("tumor_vol_cc"),
                 "diameter_mm": m.get("longest_diameter_mm"),
                 "label": r.get("progression_label")})
tdf = pd.DataFrame(traj)
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown("**Burden trajectory**")
    ycol = "diameter_mm" if tdf["diameter_mm"].notna().any() else "volume_cc"
    st.line_chart(tdf.set_index("stage")[[ycol]], height=200)
with c2:
    st.markdown("**Labels over time**")
    st.dataframe(tdf[["stage", "label", "volume_cc", "diameter_mm"]],
                 hide_index=True, use_container_width=True, height=200)

st.divider()
# ---- per-timepoint ----
idx = st.select_slider("Timepoint", options=list(range(len(stages))),
                       format_func=lambda i: stages[i])
r = reports[idx]
stage = stages[idx]
prior_stage = stages[idx - 1] if idx > 0 else None

left, right = st.columns([3, 2])
with left:
    st.markdown(render_md(r, ds, pt, csum))
with right:
    st.markdown("**Image evidence** — slice the report read"
                + (f" (vs prior `{prior_stage}`)" if prior_stage else " (baseline)"))
    imgs = gen_slices(ds, pt, stage, prior_stage)
    if not imgs:
        st.info("no slices (missing CT or data not on this machine)")
    for lbl, im in imgs:
        if lbl == "error":
            st.error(im)
        else:
            st.image(im, caption=lbl, width=260)
    with st.expander("raw report.json"):
        st.json(r)
