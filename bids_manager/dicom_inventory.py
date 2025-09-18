#!/usr/bin/env python3
"""
dicom_inventory.py — fully-commented, no-emoji version
------------------------------------------------------

Creates a long-format TSV describing every DICOM series in *root_dir*.

Why you want this
-----------------
* Lets you review **all** SeriesDescriptions, subjects, sessions and file counts
  before converting anything.
* Column `include` defaults to 1 except for scout/report/physio/physlog
  sequences, which start at 0 so they are skipped by default.
* Generated table is the single source of truth you feed into a helper script
  that writes the HeuDiConv heuristic.

Output columns (ordered as they appear)
---------------------------------------
subject        – GivenName shown only on the first row of each subject block
BIDS_name      – auto-assigned `sub-001`, `sub-002`, … (same GivenName → same ID)
session        – `ses-<label>` if exactly one unique session tag is present in
                 that folder, otherwise blank
source_folder  – relative path from the DICOM root to the folder containing the
                 series
include        – defaults to 1 but scout/report/physlog rows start at 0
sequence       – original SeriesDescription
series_uid     – DICOM SeriesInstanceUID identifying a specific acquisition
rep            – 1, 2, … if multiple SeriesInstanceUIDs share the same description
acq_time       – AcquisitionTime of the first file in that series
modality       – fine label inferred from patterns (T1w, bold, dwi, …)
modality_bids  – top-level container (anat, func, dwi, fmap) derived from
                 *modality*
n_files        – number of DICOM files (.dcm or .ima) with that SeriesDescription
GivenName … StudyDescription – demographics copied from the first header seen
"""

import os
import re
from collections import defaultdict
from typing import Optional
from pathlib import Path
from joblib import Parallel, delayed

import pandas as pd
import pydicom
from pydicom.multival import MultiValue

# Preview name helpers – loaded lazily so ``scan_dicoms_long`` can store
# proposed BIDS names directly in the TSV.  We guard the import to keep the
# inventory script functional even if the renamer dependencies are missing.
try:  # pragma: no cover - import errors simply disable preview generation
    from .schema_renamer import (
        load_bids_schema,
        SeriesInfo,
        build_preview_names,
    )
    from .schema_config import DEFAULT_SCHEMA_DIR
except Exception:  # pragma: no cover - best effort
    load_bids_schema = None  # type: ignore
    SeriesInfo = None  # type: ignore
    build_preview_names = None  # type: ignore
    DEFAULT_SCHEMA_DIR = Path(".")  # type: ignore

# Directory used to store persistent user preferences
PREF_DIR = Path(__file__).resolve().parent / "user_preferences"
SEQ_DICT_FILE = PREF_DIR / "sequence_dictionary.tsv"

# Acceptable DICOM file extensions (lower case)
# Siemens scanners sometimes omit extensions altogether, so we also
# perform a light-weight header check for files with no suffix.
DICOM_EXTS = (".dcm", ".ima")


def is_dicom_file(path: str) -> bool:
    """Return ``True`` if *path* looks like a DICOM file.

    Files with a known DICOM extension are accepted immediately.  For
    extensionless files we peek at the standard ``DICM`` marker located
    128 bytes into the file.  Any I/O error or missing marker results in
    ``False``.
    """

    name = Path(path).name.lower()
    if name.endswith(DICOM_EXTS):
        return True
    # If the filename contains a dot it has some other extension – skip it
    if "." in name:
        return False
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False

# ----------------------------------------------------------------------
# 1.  Patterns: SeriesDescription → fine-grained modality label
#    (order matters: first match wins)
# ----------------------------------------------------------------------
BIDS_PATTERNS = {
    # anatomy
    "T1w"    : (
        "t1w",
        "t1-weight",
        "t1_",
        "t1 ",
        "mprage",
        "tfl3d",
        "fspgr",
    ),
    "T2w"    : ("t2w", "space", "tse"),
    "FLAIR"  : ("flair",),
    "MTw"    : ("gre-mt", "gre_mt", "mt"),
    "PDw"    : ("gre-nm", "gre_nm"),
    "scout"  : ("localizer", "scout"),
    "report" : ("phoenixzipreport", "phoenix document", ".pdf", "report"),
    # functional
    # SBRef must precede bold to avoid misclassification when sequences
    # contain both "sbref" and "bold" tokens.
    "SBRef"  : (
        "sbref",
        "type-ref",
        "reference",
        "refscan",
        " ref",
        "_ref",
        "ref",
    ),
    # Physiological recordings also tend to include "fmri" or "bold" in
    # their sequence names.  List them before the generic bold patterns so
    # they are detected correctly.
    "physio" : ("physiolog", "physio", "pulse", "resp"),
    "bold"   : ("fmri", "bold", "task-"),
    # diffusion
    "dwi"    : ("dti", "dwi", "diff"),
    # field maps
    "fmap"   : (
        "gre_field",
        "fieldmapping",
        "_fmap",
        "fmap",
        "phase",
        "magnitude",
        "b0rf",
        "b0_map",
        "b0map",
    ),
}

# Keep a pristine copy of the default patterns so the GUI can restore them
DEFAULT_BIDS_PATTERNS = {m: tuple(pats) for m, pats in BIDS_PATTERNS.items()}


def load_sequence_dictionary() -> None:
    """Load user-modified sequence patterns from :data:`SEQ_DICT_FILE`."""
    global BIDS_PATTERNS
    if not SEQ_DICT_FILE.exists():
        return
    try:
        df = pd.read_csv(SEQ_DICT_FILE, sep="\t", keep_default_na=False)
    except Exception:
        return
    patterns: defaultdict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        mod = str(row.get("modality", "")).strip()
        pat = str(row.get("pattern", "")).strip().lower()
        if mod and pat:
            patterns[mod].append(pat)
    if patterns:
        BIDS_PATTERNS = {m: tuple(pats) for m, pats in patterns.items()}


def restore_sequence_dictionary() -> None:
    """Revert :data:`BIDS_PATTERNS` to the bundled defaults."""
    global BIDS_PATTERNS
    BIDS_PATTERNS = {m: tuple(pats) for m, pats in DEFAULT_BIDS_PATTERNS.items()}
    try:
        SEQ_DICT_FILE.unlink()
    except Exception:
        pass


load_sequence_dictionary()

def guess_modality(series: str) -> str:
    """Return first matching fine-grained modality label or 'unknown'.

    Important details:
    - We classify common vendor DWI derivative maps (ADC/FA/TRACEW/ColFA/expADC)
      before general DWI detection to keep them out of the raw `dwi/` tree.
    - Matching is performed on the lower-cased SeriesDescription.
    """
    s = series.lower()

    # Recognize common scanner-generated maps as derivatives
    dwi_derivatives = [
        "_adc", "_fa", "_tracew", "_colfa", "_expadc",
        " adc", " fa", " tracew", " colfa", " expadc",
        "adc", "fa", "tracew", "colfa", "expadc",
    ]
    for deriv in dwi_derivatives:
        if deriv in s:
            return "dwi_derivative"

    for label, pats in BIDS_PATTERNS.items():
        if any(p in s for p in pats):
            return label

    return "unknown"


MAGNITUDE_IMGTYPE = ["ORIGINAL", "PRIMARY", "M", "ND", "NORM"]
PHASE_IMGTYPE = ["ORIGINAL", "PRIMARY", "P", "ND"]

def normalize_image_type(value) -> list:
    """Return ImageType components as a list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, MultiValue)):
        return [str(x).strip() for x in value]
    text = str(value)
    if "\\" in text:
        return [p.strip() for p in text.split("\\")]
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
        return [p.strip().strip("'") for p in text.split(",")]
    return [text] if text else []


def classify_fieldmap_type(img_list: list) -> str:
    """Return 'M' for magnitude, 'P' for phase, '' otherwise."""
    if img_list == MAGNITUDE_IMGTYPE:
        return "M"
    if img_list == PHASE_IMGTYPE:
        return "P"
    return ""


# ----------------------------------------------------------------------
# 2.  Map fine label → top-level BIDS container (anat, func, …)
# ----------------------------------------------------------------------
BIDS_CONTAINER = {
    "T1w":"anat", "T2w":"anat", "FLAIR":"anat",
    "MTw":"anat", "PDw":"anat",
    "scout":"anat", "report":"anat",
    "bold":"func", "SBRef":"func", "physio":"func",
    "dwi":"dwi",
    "dwi_derivative":"derivatives",  # DWI derivatives go to derivatives folder
    "fmap":"fmap",
}
def modality_to_container(mod: str) -> str:
    """Translate T1w → anat, bold → func, etc.; unknown → ''."""
    return BIDS_CONTAINER.get(mod, "")

# session detector (e.g. ses-pre, ses-01) -- case-insensitive
SESSION_RE = re.compile(r"ses-([a-zA-Z0-9]+)", re.IGNORECASE)


# ----------------------------------------------------------------------
# 3.  Main scanner
# ----------------------------------------------------------------------
def scan_dicoms_long(
    root_dir: str,
    output_tsv: Optional[str] = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Walk *root_dir*, read DICOM headers, return long-format DataFrame.

    Parameters
    ----------
    root_dir   : str
        Path with raw DICOMs organised in sub-folders.
    output_tsv : str | None
        If provided, write the TSV to that path.
    n_jobs : int
        Number of parallel workers to use when reading DICOM files.

    Returns
    -------
    pandas.DataFrame
        Inventory as described in module docstring.
    """

    root_dir = Path(root_dir)
    print(f"Scanning DICOM headers under: {root_dir}")

    # in-memory stores
    demo    = {}
    counts     = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    mods       = defaultdict(lambda: defaultdict(dict))
    acq_times  = defaultdict(lambda: defaultdict(dict))
    imgtypes   = defaultdict(lambda: defaultdict(dict))
    sessset = defaultdict(lambda: defaultdict(set))

    # PASS 1: Walk filesystem and collect info in parallel
    file_list = []
    for root, _dirs, files in os.walk(root_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if is_dicom_file(fpath):
                file_list.append(fpath)

    def _read_one(fpath: str):
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
        except Exception as exc:  # pragma: no cover - I/O errors
            print(f"Warning: could not read {fpath}: {exc}")
            return None

        root = os.path.dirname(fpath)
        pn = getattr(ds, "PatientName", None)
        given = pn.given_name.strip() if pn and pn.given_name else ""
        pid = getattr(ds, "PatientID", "").strip()
        subj = given or pid or "UNKNOWN"
        study = (
            getattr(ds, "StudyDescription", None)
            or getattr(ds, "StudyName", None)
            or "n/a"
        )
        study = str(study).strip()
        subj_key = f"{subj}||{study}"
        rel = os.path.relpath(root, root_dir)
        folder = root_dir.name if rel == "." else rel
        series = getattr(ds, "SeriesDescription", "n/a").strip()
        uid = getattr(ds, "SeriesInstanceUID", "")
        raw_img_type = getattr(ds, "ImageType", None)
        img_list = normalize_image_type(raw_img_type)
        img3 = classify_fieldmap_type(img_list)
        if not img3:
            img3 = img_list[2] if len(img_list) >= 3 else ""
        acq_time = str(getattr(ds, "AcquisitionTime", "")).strip()
        m = SESSION_RE.search(series)
        sess_tag = f"ses-{m.group(1)}" if m else None
        demo_dict = dict(
            GivenName=given,
            FamilyName=getattr(pn, "family_name", "").strip(),
            PatientID=pid,
            PatientSex=getattr(ds, "PatientSex", "n/a").strip(),
            PatientAge=getattr(ds, "PatientAge", "n/a").strip(),
            StudyDescription=study,
        )
        return dict(
            subj_key=subj_key,
            folder=folder,
            series=series,
            uid=uid,
            modality=guess_modality(series),
            img3=img3,
            acq_time=acq_time,
            sess_tag=sess_tag,
            demo=demo_dict,
        )

    results = Parallel(n_jobs=n_jobs)(delayed(_read_one)(fp) for fp in file_list)
    for res in results:
        if not res:
            continue
        subj_key = res["subj_key"]
        folder = res["folder"]
        series = res["series"]
        uid = res["uid"]
        key = (series, uid)
        counts[subj_key][folder][key] += 1
        mods[subj_key][folder][key] = res["modality"]
        if key not in imgtypes[subj_key][folder]:
            imgtypes[subj_key][folder][key] = res["img3"]
        if key not in acq_times[subj_key][folder] and res["acq_time"]:
            acq_times[subj_key][folder][key] = res["acq_time"]
        if res["sess_tag"]:
            sessset[subj_key][folder].add(res["sess_tag"])
        if subj_key not in demo:
            demo[subj_key] = res["demo"]

    print(f"Subjects found            : {len(demo)}")
    total_series = sum(len(seq_dict)
                       for subj in counts.values()
                       for folder, seq_dict in subj.items())
    print(f"Unique Series instances   : {total_series}")

    # PASS 2: assign BIDS subject numbers PER STUDY
    study_subjects = defaultdict(set)
    for subj_key in demo:
        subj, stud = subj_key.split("||", 1)
        study_subjects[stud].add(subj)

    bids_map = {}
    for study, subj_set in study_subjects.items():
        for i, sid in enumerate(sorted(subj_set)):
            bids_map[f"{sid}||{study}"] = f"sub-{i+1:03d}"

    print("Assigned BIDS IDs:", bids_map)

    # PASS 3: build DataFrame rows
    rows = []
    for subj_key in sorted(counts):
        first_row = True
        for folder in sorted(counts[subj_key]):

            # decide session label for this folder
            ses_labels = sorted(sessset[subj_key][folder])
            session = ses_labels[0] if len(ses_labels) == 1 else ""

            rep_counter = defaultdict(int)
            for (series, uid), n_files in sorted(counts[subj_key][folder].items()):
                fine_mod = mods[subj_key][folder][(series, uid)]
                img3 = imgtypes[subj_key][folder].get((series, uid), "")
                include = 1
                if fine_mod in {"scout", "report", "physio"} or "physlog" in series.lower():
                    include = 0
                # Do not consider image type when counting scout duplicates
                rep_key = series if fine_mod == "scout" else (series, img3)
                rep_counter[rep_key] += 1
                rows.append({
                    "subject"       : demo[subj_key]["GivenName"] if first_row else "",
                    "BIDS_name"     : bids_map[subj_key],
                    "session"       : session,
                    "source_folder" : folder,
                    "include"       : include,
                    "sequence"      : series,
                    "series_uid"    : uid,
                    "rep"           : rep_counter[rep_key] if rep_counter[rep_key] > 1 else "",
                    "image_type"    : img3,
                    "acq_time"      : acq_times[subj_key][folder].get((series, uid), ""),
                    "modality"      : fine_mod,
                    "modality_bids" : modality_to_container(fine_mod),
                    "n_files"       : n_files,
                    **demo[subj_key],                                # demographics
                })
                first_row = False

    # Final column order
    columns = [
        "subject", "BIDS_name", "session", "source_folder",
        "include", "sequence", "series_uid", "rep", "acq_time",
        "image_type", "modality", "modality_bids", "n_files",
        "GivenName", "FamilyName", "PatientID",
        "PatientSex", "PatientAge", "StudyDescription",
    ]
    df = pd.DataFrame(rows, columns=columns)

    # Collapse magnitude/phase rows for fieldmaps
    fmap_mask = df.modality == "fmap"
    if fmap_mask.any():
        base_cols = [
            "BIDS_name",
            "session",
            "source_folder",
            "sequence",
        ]
        # Use acquisition time rounded to the minute to merge magnitude and
        # phase series from the same fieldmap even if their timestamps differ
        # by a few seconds.
        fmap_df = df[fmap_mask].copy()
        fmap_df["acq_group"] = fmap_df["acq_time"].apply(lambda t: str(t)[:4])
        group_cols = base_cols + ["acq_group"]
        fmap_df["uid_list"] = fmap_df["series_uid"]
        # keep all UIDs within each group so both magnitude and phase series
        # are converted; they will be joined with '|' below
        fmap_df["img_set"] = fmap_df["image_type"]
        fmap_df = (
            fmap_df.groupby(group_cols, as_index=False)
            .agg(
                {
                    "subject": "first",
                    "BIDS_name": "first",
                    "session": "first",
                    "source_folder": "first",
                    "include": "max",
                    "sequence": "first",
                    "uid_list": lambda x: "|".join(sorted(set(str(v) for v in x))),
                    "img_set": lambda x: "".join(sorted(set(str(v) for v in x))),
                    "acq_time": "first",
                    "modality": "first",
                    "modality_bids": "first",
                    "n_files": "sum",
                    "GivenName": "first",
                    "FamilyName": "first",
                    "PatientID": "first",
                    "PatientSex": "first",
                    "PatientAge": "first",
                    "StudyDescription": "first",
                }
            )
        )
        fmap_df.rename(columns={"uid_list": "series_uid", "img_set": "image_type"}, inplace=True)
        fmap_df.drop(columns=["acq_group"], inplace=True)
        sort_cols = base_cols + ["acq_time"]
        fmap_df.sort_values(sort_cols, inplace=True)
        fmap_df["rep"] = fmap_df.groupby(base_cols).cumcount() + 1
        repeat_mask = fmap_df.groupby(base_cols)["rep"].transform("count") > 1
        fmap_df.loc[~repeat_mask, "rep"] = ""

        df = pd.concat([df[~fmap_mask], fmap_df], ignore_index=True, sort=False)

    df.sort_values(["StudyDescription", "BIDS_name"], inplace=True)

    # ------------------------------------------------------------------
    # Proposed BIDS names
    # ------------------------------------------------------------------
    if load_bids_schema and SeriesInfo and build_preview_names:
        try:
            schema = load_bids_schema(DEFAULT_SCHEMA_DIR)
            rows = []
            idxs = []
            for i, row in df.iterrows():
                subj = str(row.get("BIDS_name", ""))
                subj = subj[4:] if subj.lower().startswith("sub-") else subj
                session = row.get("session") or None
                modality = str(row.get("modality") or "")
                sequence = str(row.get("sequence") or "")
                rep = row.get("rep") or 1
                rows.append(SeriesInfo(subj, session, modality, sequence, int(rep or 1), {}))
                idxs.append(i)
            for (series, dt, base), idx in zip(build_preview_names(rows, schema), idxs):
                df.loc[idx, "proposed_datatype"] = dt
                df.loc[idx, "proposed_basename"] = base
                ext = ".tsv" if base.endswith("_physio") else ".nii.gz"
                df.loc[idx, "Proposed BIDS name"] = f"{dt}/{base}{ext}" if base else ""
        except Exception:
            # Preview generation is best-effort; fall back silently if anything
            # goes wrong so the inventory can still be written.
            pass

    # optional TSV export
    if output_tsv:
        df.to_csv(output_tsv, sep="\t", index=False)
        print(f"Inventory written to: {output_tsv}")

    return df


# ----------------------------------------------------------------------
# Command-line test
# ----------------------------------------------------------------------
def main() -> None:
    """Command line interface for :func:`scan_dicoms_long`."""

    import argparse

    parser = argparse.ArgumentParser(description="Generate TSV inventory for a DICOM folder")
    parser.add_argument("dicom_dir", help="Path to the directory containing DICOM files")
    parser.add_argument("output_tsv", help="Destination TSV file")
    parser.add_argument(
        "--jobs",
        type=int,
        # Use ~80% of available CPUs to provide a sensible default while
        # leaving some resources free for the rest of the system.
        default=max(1, round((os.cpu_count() or 1) * 0.8)),
        help="Number of parallel workers to use",
    )
    args = parser.parse_args()

    table = scan_dicoms_long(args.dicom_dir, args.output_tsv, n_jobs=args.jobs)
    print("\nPreview (first 10 rows):\n")
    print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
