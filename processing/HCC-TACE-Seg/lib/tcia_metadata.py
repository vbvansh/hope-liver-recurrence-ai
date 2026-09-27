#!/usr/bin/env python3
"""Convert a TCIA NBIA `metadata.csv` (shipped inside every manifest download)
into the standard `<dataset>_metadata_series.csv` that standardize.py consumes.

The API-downloaded collections (EAY131, NLST) already have this CSV with series
folders named by SeriesInstanceUID, so standardize.py finds them by globbing.
Manifest downloads instead name series folders by *description*
("5.000000-NEPHRO PHASE AP-02551"), so the UID glob fails. We therefore also
emit a `SeriesDir` column (absolute path, resolved from the manifest's
`File Location`); standardize.py reads it into a UID->dir map and skips the glob.

TCIA metadata.csv columns ->
  Series UID, Collection, Subject ID, Study UID, Study Description, Study Date,
  Series Description, Manufacturer, Modality, Number of Images, File Size,
  File Location  (relative to the manifest dir, e.g. ./<Coll>/<pid>/<study>/<series>)

Study Date is MM-DD-YYYY in the manifest; standardize.py orders timepoints by
sorting the StudyDate string, so we rewrite it to ISO YYYY-MM-DD (otherwise
'12-...' sorts before '03-...' of a later year and baselines are picked wrong).

Usage:
  python3 tcia_metadata.py \
      --manifest "/media/cbtil3/WhiteSD/CTRaw/Abdomen/CPTAC-CCRCC/manifest-1692379830142" \
      --dataset  CPTAC-CCRCC \
      --out-dir  /media/cbtil3/WhiteSD/CTRaw/Abdomen/CPTAC-CCRCC
"""
import argparse
import os

import pandas as pd


def iso_date(s):
    """MM-DD-YYYY -> YYYY-MM-DD 00:00:00.0 (match the EAY131/NLST schema so the
    string sort standardize.py does is chronological). Pass through if already
    looks ISO or is empty."""
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return ""
    for fmt in ("%m-%d-%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt).strftime("%Y-%m-%d 00:00:00.0")
        except ValueError:
            continue
    # last resort: let pandas guess
    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d 00:00:00.0")
    except Exception:
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="manifest dir that holds metadata.csv + the collection tree")
    ap.add_argument("--dataset", required=True, help="dataset id, e.g. CPTAC-CCRCC")
    ap.add_argument("--out-dir", required=True,
                    help="where to write <dataset-lower>_metadata_series.csv")
    args = ap.parse_args()

    src = os.path.join(args.manifest, "metadata.csv")
    df = pd.read_csv(src, dtype=str).fillna("")

    def col(*names):
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series([""] * len(df))

    # manifests downloaded on Windows store ".\Coll\pid\..."; use "/" so the paths
    # resolve on Linux too
    file_loc = col("File Location").str.replace("\\", "/", regex=False)
    series_dir = file_loc.map(
        lambda p: os.path.normpath(os.path.join(args.manifest, p)) if p else "")

    out = pd.DataFrame({
        "PatientID": col("Subject ID"),
        "PatientSex": "",
        "PatientAge": "",
        "EventType": "",
        "EventOffsetDays": "",
        "StudyDate": col("Study Date").map(iso_date),
        "StudyDesc": col("Study Description"),
        "StudyInstanceUID": col("Study UID"),
        "SeriesInstanceUID": col("Series UID"),
        "SeriesNumber": col("Series Number"),
        "Modality": col("Modality"),
        "SeriesDescription": col("Series Description"),
        "ImageCount": pd.to_numeric(col("Number of Images"), errors="coerce")
                        .fillna(0).astype(int),
        "FileSizeBytes": "",
        "Manufacturer": col("Manufacturer"),
        "ManufacturerModelName": "",
        "SeriesDir": series_dir,
    })

    # sanity: how many series dirs actually exist on disk
    exists = out.SeriesDir.map(lambda p: bool(p) and os.path.isdir(p)).sum()
    os.makedirs(args.out_dir, exist_ok=True)
    dst = os.path.join(args.out_dir, f"{args.dataset.lower()}_metadata_series.csv")
    out.to_csv(dst, index=False)

    n_ct = (out.Modality == "CT").sum()
    n_pat = out.PatientID.nunique()
    ct_tp = (out[out.Modality == "CT"].groupby("PatientID").StudyDate.nunique())
    multi = (ct_tp >= 2).sum()
    print(f"{args.dataset}: {len(out)} series ({n_ct} CT), {n_pat} patients, "
          f"{multi} with >=2 CT study-dates")
    print(f"  modalities: {dict(out.Modality.value_counts())}")
    print(f"  series dirs on disk: {exists}/{len(out)}")
    print(f"  -> {dst}")


if __name__ == "__main__":
    main()
