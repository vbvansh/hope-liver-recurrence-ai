"""Read MCT-LTDiag patients straight from their per-patient tars, without unpacking them.

Each tar holds DICOM/ (the original series), NIFTI/{nc,art,pvp,delay}.nii.gz (all four phases, already registered
to PVP by the authors, on one grid), mask_pvp.nii.gz (tumour, 0/1) and liver_mask_pvp.nii.gz (0/1). The letter in
the patient ID is the tumour type (matches meta_info_patient.csv for all 517, checked 2026-11-02).
"""
import gzip
import re
import tarfile
from pathlib import Path

import nibabel as nib
import pandas as pd

MEMBERS = {"nc": "NIFTI/nc.nii.gz", "ap": "NIFTI/art.nii.gz", "pvp": "NIFTI/pvp.nii.gz", "dp": "NIFTI/delay.nii.gz",
           "tumour": "mask_pvp.nii.gz", "liver": "liver_mask_pvp.nii.gz"}
TYPES = {"a": "BCLM", "b": "HCC", "c": "HH", "d": "ICC", "e": "CRLM"}  # HH = haemangioma, BCLM = breast cancer mets
ID = re.compile(r"\d+([a-e])\d+")


def patients(raw_root):
    """Table of patient_id, type, tar path; stops if a tar and the patient table disagree."""
    raw_root = Path(raw_root)
    meta = pd.read_csv(raw_root / "metadata" / "meta_info_patient.csv", dtype=str)[["ID", "type"]]
    tars = pd.DataFrame({"patient_id": [p.stem for p in sorted((raw_root / "tars").glob("*.tar"))]})
    tars["tar"] = [raw_root / "tars" / f"{p}.tar" for p in tars.patient_id]
    tars = tars.merge(meta.rename(columns={"ID": "patient_id"}), on="patient_id", how="outer", indicator=True)
    if (tars["_merge"] != "both").any():
        raise ValueError(f"{(tars['_merge'] != 'both').sum()} tars and patient-table rows do not match")
    letters = tars.patient_id.map(lambda p: TYPES[ID.fullmatch(p).group(1)])
    if (letters != tars.type).any():
        raise ValueError(f"{(letters != tars.type).sum()} patients whose ID letter disagrees with the patient table")
    return tars.drop(columns="_merge")


def headers(tar_path, names=tuple(MEMBERS)):
    """{name: NIfTI header} for the requested members; only the start of each file is decompressed."""
    with tarfile.open(tar_path) as tf:
        members = {re.sub(r"^\./", "", m.name): m for m in tf.getmembers() if m.isfile()}
        return {n: nib.Nifti1Header.from_fileobj(gzip.GzipFile(fileobj=tf.extractfile(members[MEMBERS[n]])))
                for n in names}


def read(tar_path, names=("tumour", "liver")):
    """{name: nibabel image} for the requested members of one patient's tar; only those members are read."""
    with tarfile.open(tar_path) as tf:
        members = {re.sub(r"^\./", "", m.name): m for m in tf.getmembers() if m.isfile()}
        return {n: nib.Nifti1Image.from_bytes(gzip.decompress(tf.extractfile(members[MEMBERS[n]]).read()))
                for n in names}
