#!/usr/bin/env python3
"""
dicom_inventory.py — fully-commented, no-emoji version
------------------------------------------------------

Creates a long-format TSV describing every DICOM series in *root_dir*.

Why you want this
-----------------
* Lets you review **all** SeriesDescriptions, subjects, sessions and file counts
  before converting anything.
* Column `include` defaults to 1 except for scout/report/physlog sequences,
  which start at 0 so they are skipped by default.
* Generated table is the single source of truth you feed into a helper script
  that writes the HeuDiConv heuristic.

Output columns (ordered as they appear)
---------------------------------------
subject        – GivenName shown only on the first row of each subject block
BIDS_name      – auto-assigned `sub-001`, `sub-002`, …
session        – `ses-<label>` if exactly one unique session tag is present in
                 that folder, otherwise blank
source_folder  – first directory under *root_dir* that contained the DICOM
include        – defaults to 1 but scout/report/physlog rows start at 0
sequence       – original SeriesDescription
modality       – fine label inferred from patterns (T1w, bold, dwi, …)
modality_bids  – top-level container (anat, func, dwi, fmap) derived from
                 *modality*
n_files        – number of *.dcm files* with that SeriesDescription
GivenName … StudyDescription – demographics copied from the first header seen
"""

import os
import re
from collections import defaultdict
from typing import Optional

import pandas as pd
import pydicom


# ----------------------------------------------------------------------
# 1.  Patterns: SeriesDescription → fine-grained modality label
#    (order matters: first match wins)
# ----------------------------------------------------------------------
BIDS_PATTERNS = {
    # anatomy
    "T1w"    : ("t1w", "mprage", "tfl3d"),
    "T2w"    : ("t2w", "space", "tse"),
    "FLAIR"  : ("flair",),
    "scout"  : ("localizer", "scout"),
    "report" : ("phoenixzipreport", "phoenix document", ".pdf", "report"),
    "refscan": ("type-ref", "reference", "refscan"),
    # functional
    "bold"   : ("fmri", "bold", "task-"),
    "SBRef"  : ("sbref",),
    # diffusion
    "dwi"    : ("dti", "dwi", "diff"),
    # field maps
    "fmap"   : ("gre_field", "fieldmapping", "_fmap", "phase", "magnitude"),
    # misc (kept for completeness)
    "physio" : ("physiolog", "physio", "pulse", "resp"),
}

def guess_modality(series: str) -> str:
    """Return first matching fine label; otherwise 'unknown'."""
    s = series.lower()
    for label, pats in BIDS_PATTERNS.items():
        if any(p in s for p in pats):
            return label
    return "unknown"


# ----------------------------------------------------------------------
# 2.  Map fine label → top-level BIDS container (anat, func, …)
# ----------------------------------------------------------------------
BIDS_CONTAINER = {
    "T1w":"anat", "T2w":"anat", "FLAIR":"anat",
    "scout":"anat", "report":"anat", "refscan":"anat",
    "bold":"func", "SBRef":"func",
    "dwi":"dwi",
    "fmap":"fmap",
}
def modality_to_container(mod: str) -> str:
    """Translate T1w → anat, bold → func, etc.; unknown → ''."""
    return BIDS_CONTAINER.get(mod, "")

# session detector (e.g. ses-pre, ses-01)
SESSION_RE = re.compile(r"ses-([a-zA-Z0-9]+)")


# ----------------------------------------------------------------------
# 3.  Main scanner
# ----------------------------------------------------------------------
def scan_dicoms_long(root_dir: str,
                     output_tsv: Optional[str] = None) -> pd.DataFrame:
    """
    Walk *root_dir*, read DICOM headers, return long-format DataFrame.

    Parameters
    ----------
    root_dir   : str
        Path with raw DICOMs organised in sub-folders.
    output_tsv : str | None
        If provided, write the TSV to that path.

    Returns
    -------
    pandas.DataFrame
        Inventory as described in module docstring.
    """

    print(f"Scanning DICOM headers under: {root_dir}")

    # in-memory stores
    demo    = {}  # subj_id → demographics dictionary
    counts  = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    mods    = defaultdict(lambda: defaultdict(dict))
    sessset = defaultdict(lambda: defaultdict(set))

    # PASS 1: Walk filesystem and collect info
    for root, _dirs, files in os.walk(root_dir):
        for fname in files:
            if not fname.lower().endswith(".dcm"):
                continue
            fpath = os.path.join(root, fname)

            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            except Exception as exc:
                print(f"Warning: could not read {fpath}: {exc}")
                continue

            # ---- subject id  (GivenName > PatientID > 'UNKNOWN')
            pn    = getattr(ds, "PatientName", None)
            given = pn.given_name.strip()  if pn and pn.given_name  else ""
            pid   = getattr(ds, "PatientID", "").strip()
            subj  = given or pid or "UNKNOWN"

            # ---- source folder  (first dir under root)
            rel = os.path.relpath(root, root_dir)
            folder = rel.split(os.sep)[0] if rel != "." else ""

            series = getattr(ds, "SeriesDescription", "n/a").strip()
            counts[subj][folder][series] += 1
            mods[subj][folder][series] = guess_modality(series)

            # collect session tags
            m = SESSION_RE.search(series.lower())
            if m:
                sessset[subj][folder].add(f"ses-{m.group(1)}")

            # store demographics once per subject
            if subj not in demo:
                demo[subj] = dict(
                    GivenName        = given,
                    FamilyName       = getattr(pn, "family_name", "").strip(),
                    PatientID        = pid,
                    PatientSex       = getattr(ds, "PatientSex", "n/a").strip(),
                    PatientAge       = getattr(ds, "PatientAge", "n/a").strip(),
                    StudyDescription = getattr(ds, "StudyDescription", "n/a").strip(),
                )

    print(f"Subjects found            : {len(demo)}")
    total_series = sum(len(seq_dict)
                       for subj in counts.values()
                       for folder, seq_dict in subj.items())
    print(f"Unique SeriesDescriptions : {total_series}")

    # PASS 2: assign BIDS subject numbers
    bids_map = {sid: f"sub-{i+1:03d}" for i, sid in enumerate(sorted(demo))}
    print("Assigned BIDS IDs:", bids_map)

    # PASS 3: build DataFrame rows
    rows = []
    for subj in sorted(counts):
        first_row = True
        for folder in sorted(counts[subj]):

            # decide session label for this folder
            ses_labels = sorted(sessset[subj][folder])
            session = ses_labels[0] if len(ses_labels) == 1 else ""

            for series, n_files in sorted(counts[subj][folder].items()):
                fine_mod = mods[subj][folder][series]
                include = 1
                if fine_mod in {"scout", "report"} or "physlog" in series.lower():
                    include = 0
                rows.append({
                    "subject"       : demo[subj]["GivenName"] if first_row else "",
                    "BIDS_name"     : bids_map[subj],
                    "session"       : session,
                    "source_folder" : folder,
                    "include"       : include,
                    "sequence"      : series,
                    "modality"      : fine_mod,
                    "modality_bids" : modality_to_container(fine_mod),
                    "n_files"       : n_files,
                    **demo[subj],                                # demographics
                })
                first_row = False

    # Final column order
    columns = [
        "subject", "BIDS_name", "session", "source_folder",
        "include", "sequence", "modality", "modality_bids", "n_files",
        "GivenName", "FamilyName", "PatientID",
        "PatientSex", "PatientAge", "StudyDescription",
    ]
    df = pd.DataFrame(rows, columns=columns)

    # optional TSV export
    if output_tsv:
        df.to_csv(output_tsv, sep="\t", index=False)
        print(f"Inventory written to: {output_tsv}")

    return df


# ----------------------------------------------------------------------
# Command-line test
# ----------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate TSV inventory for a DICOM folder")
    parser.add_argument("dicom_dir", help="Path to the directory containing DICOM files")
    parser.add_argument("output_tsv", help="Destination TSV file")
    args = parser.parse_args()

    table = scan_dicoms_long(args.dicom_dir, args.output_tsv)
    print("\nPreview (first 10 rows):\n")
    print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
