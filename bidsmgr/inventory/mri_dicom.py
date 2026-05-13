"""MRI DICOM scanner — port of BIDS-Manager v0.2.5 ``dicom_inventory.py``.

Walks a DICOM tree with ``pydicom`` + ``joblib`` parallelism and emits a
long-format ``pandas.DataFrame``. One row per (subject, source folder,
SeriesDescription, SeriesInstanceUID).

Preserves the v0.2.5 22-column TSV contract (improvement_plan.md §4):

    subject, BIDS_name, session, source_folder,
    include, sequence, series_uid, rep, acq_time,
    image_type, modality, modality_bids, n_files,
    GivenName, FamilyName, PatientID,
    PatientSex, PatientAge, StudyDescription,
    proposed_datatype, proposed_basename, Proposed BIDS name

The ``modality`` and ``modality_bids`` columns are filled by the legacy
regex-dictionary classifier (``classifier.sequence_dict.guess_modality``).
``proposed_*`` columns are populated downstream by the CLI orchestrator
after running the BidsGuess classifier (improvement_plan.md M1).

The scanner intentionally does **not** invoke the classifier chain itself
— that orchestration lives in ``cli/scan.py`` so the inventory step can
be reused as-is by the GUI worker layer (architecture.md §12 rule 3,
no Pipeline orchestrator).
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import pydicom
from joblib import Parallel, delayed
from pydicom.multival import MultiValue

from ..classifier.sequence_dict import (
    SKIP_MODALITIES,
    guess_modality,
    modality_to_container,
    normalize_study_name,
)
from .subject_identity import IdentityTuple, cluster_subjects, normalize_tuple


DICOM_EXTS = (".dcm", ".ima")

MAGNITUDE_IMGTYPE = ["ORIGINAL", "PRIMARY", "M", "ND", "NORM"]
PHASE_IMGTYPE = ["ORIGINAL", "PRIMARY", "P", "ND"]

SESSION_RE = re.compile(r"ses-([a-zA-Z0-9]+)", re.IGNORECASE)


# v0.2.5 22-column TSV contract — keep this list in sync with the docstring.
TSV_COLUMNS: tuple[str, ...] = (
    "subject", "BIDS_name", "session", "source_folder",
    "include", "sequence", "series_uid", "rep", "acq_time",
    "image_type", "modality", "modality_bids", "n_files",
    "GivenName", "FamilyName", "PatientID",
    "PatientSex", "PatientAge", "StudyDescription",
    "proposed_datatype", "proposed_basename", "Proposed BIDS name",
)

# Extended columns added to support longitudinal session inference and
# abort detection. The 22-col contract above is preserved verbatim; these
# come after it.
EXTENDED_COLUMNS: tuple[str, ...] = (
    "study_instance_uid",
    "study_date",
    "study_time",
)

# User-editable BIDS dataset name. Written by ``cli/scan.py``; read by
# ``cli/convert.py`` to partition rows across sibling BIDS roots. The
# converter writes each distinct value to ``<bids_parent>/<dataset>/``.
DATASET_COLUMNS: tuple[str, ...] = ("dataset",)


# The canonical BIDS entity dict per row, JSON-encoded. **Source of
# truth** for the BIDS basename: scanners populate it; ``bidsmgr-rebuild``
# regenerates ``proposed_basename`` and mirror cells from it (or, in
# ``--from columns`` mode, the reverse). The converter reads from this
# column directly, so the row's BIDS name always reflects whatever the
# user last edited here. Format: a JSON object with BIDS entity keys —
# ``{"sub": "001", "ses": "pre", "task": "rest", "run": "1", ...}``.
BIDS_ENTITIES_COLUMNS: tuple[str, ...] = ("entities",)


def is_dicom_file(path: str) -> bool:
    """Return True if ``path`` looks like a DICOM file.

    Files with a known extension are accepted immediately; extensionless
    files are accepted if they carry the standard ``DICM`` marker at byte 128.
    """
    name = Path(path).name.lower()
    if name.endswith(DICOM_EXTS):
        return True
    if "." in name:
        return False
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def normalize_image_type(value) -> list[str]:
    """Coerce a DICOM ``ImageType`` value into a list of strings."""
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


def classify_fieldmap_type(img_list: list[str]) -> str:
    if img_list == MAGNITUDE_IMGTYPE:
        return "M"
    if img_list == PHASE_IMGTYPE:
        return "P"
    return ""


def _read_one(fpath: str, root_dir: Path) -> Optional[dict]:
    """Read a single DICOM header; return summary dict or ``None`` on failure."""
    try:
        ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
    except Exception as exc:  # pragma: no cover - I/O paths
        print(f"Warning: could not read {fpath}: {exc}")
        return None

    file_root = os.path.dirname(fpath)
    pn = getattr(ds, "PatientName", None)
    given = pn.given_name.strip() if pn and pn.given_name else ""
    family = pn.family_name.strip() if pn and pn.family_name else ""
    pid = str(getattr(ds, "PatientID", "")).strip()
    birth = str(getattr(ds, "PatientBirthDate", "")).strip()
    subj = given or pid or "UNKNOWN"
    study = (
        getattr(ds, "StudyDescription", None)
        or getattr(ds, "StudyName", None)
        or "n/a"
    )
    study = normalize_study_name(study)

    # Subject identity (architecture.md §4.1) — three-component key
    # ``pid||given||family``. The actual subject clustering (joining records
    # that share any non-placeholder field) happens after the parallel scan
    # in ``scan_dicoms_long`` via
    # :func:`bidsmgr.inventory.subject_identity.cluster_subjects`.
    # ``PatientBirthDate`` is intentionally excluded — anonymisation
    # pipelines mutate it inconsistently across visits.
    identity_key = "||".join((pid, given, family))
    if not identity_key.strip("|"):
        # Last resort: fall back to ``(subj, study)`` so anonymised datasets
        # without ANY identifier don't collapse all rows into one subject.
        identity_key = f"{subj}||{study}"

    rel = os.path.relpath(file_root, root_dir)
    folder = root_dir.name if rel == "." else rel
    series = str(getattr(ds, "SeriesDescription", "n/a")).strip()
    uid = str(getattr(ds, "SeriesInstanceUID", ""))
    img_list = normalize_image_type(getattr(ds, "ImageType", None))
    img3 = classify_fieldmap_type(img_list)
    if not img3:
        img3 = img_list[2] if len(img_list) >= 3 else ""
    acq_time = str(getattr(ds, "AcquisitionTime", "")).strip()

    # Study-level identifiers — used for longitudinal session inference
    # (architecture.md §4.1). Distinct StudyInstanceUID + StudyDate per
    # subject indicates separate imaging visits.
    study_uid = str(getattr(ds, "StudyInstanceUID", "")).strip()
    study_date = str(getattr(ds, "StudyDate", "")).strip()
    study_time = str(getattr(ds, "StudyTime", "")).strip()

    m = SESSION_RE.search(series)
    sess_tag = f"ses-{m.group(1)}" if m else None

    family_name = ""
    if pn is not None:
        family_name = (pn.family_name or "").strip()

    return {
        "subj_key": identity_key,
        "study_key": study,
        "folder": folder,
        "file_dir": file_root,
        "file_path": fpath,
        "series": series,
        "uid": uid,
        "modality": guess_modality(series),
        "img3": img3,
        "acq_time": acq_time,
        "sess_tag": sess_tag,
        "study_uid": study_uid,
        "study_date": study_date,
        "study_time": study_time,
        "demo": {
            "GivenName": given,
            "FamilyName": family_name,
            "PatientID": pid,
            "PatientSex": str(getattr(ds, "PatientSex", "n/a")).strip(),
            "PatientAge": str(getattr(ds, "PatientAge", "n/a")).strip(),
            "StudyDescription": study,
        },
    }


def scan_dicoms_long(
    root_dir: str | os.PathLike,
    output_tsv: Optional[str | os.PathLike] = None,
    n_jobs: int = 1,
    dataset: Optional[str] = None,
) -> pd.DataFrame:
    """Walk ``root_dir`` and return a long-format inventory DataFrame.

    Parameters
    ----------
    root_dir
        Directory containing raw DICOMs (any depth).
    output_tsv
        If provided, write the inventory TSV to that path.
    n_jobs
        Number of parallel workers used to read DICOM headers.
    dataset
        User-supplied BIDS dataset slug stamped into every row's
        ``dataset`` column. The converter uses this to partition subjects
        across sibling BIDS roots. ``None`` leaves the column empty (the
        CLI fills in a default before this function is called).
    """

    root_dir = Path(root_dir)
    print(f"Scanning DICOM headers under: {root_dir}")

    counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    mods: dict = defaultdict(lambda: defaultdict(dict))
    acq_times: dict = defaultdict(lambda: defaultdict(dict))
    imgtypes: dict = defaultdict(lambda: defaultdict(dict))
    sessset: dict = defaultdict(lambda: defaultdict(set))
    file_dirs: dict = defaultdict(lambda: defaultdict(dict))
    # study-level metadata (per series): subj_key -> folder -> (series,uid) -> study tuple
    study_uids: dict = defaultdict(lambda: defaultdict(dict))
    # all distinct study tuples seen for each subject (for session inference).
    subject_studies: dict = defaultdict(set)
    # Per-SeriesInstanceUID list of source DICOM file paths. Used by
    # ``probe_convert`` to symlink just one series's files into a per-row
    # work directory before invoking dcm2niix.
    files_by_uid: dict[str, list[str]] = defaultdict(list)
    demo: dict = {}

    file_list: list[str] = []
    for root, _dirs, files in os.walk(root_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if is_dicom_file(fpath):
                file_list.append(fpath)

    results = Parallel(n_jobs=n_jobs)(delayed(_read_one)(fp, root_dir) for fp in file_list)
    for res in results:
        if not res:
            continue
        subj_key = res["subj_key"]
        folder = res["folder"]
        key = (res["series"], res["uid"])
        counts[subj_key][folder][key] += 1
        mods[subj_key][folder][key] = res["modality"]
        if key not in imgtypes[subj_key][folder]:
            imgtypes[subj_key][folder][key] = res["img3"]
        if key not in acq_times[subj_key][folder] and res["acq_time"]:
            acq_times[subj_key][folder][key] = res["acq_time"]
        if key not in file_dirs[subj_key][folder]:
            file_dirs[subj_key][folder][key] = res["file_dir"]
        # Track every DICOM file path per UID for later per-series probe.
        uid_str = res["uid"]
        if uid_str:
            files_by_uid[uid_str].append(res["file_path"])
        if res["sess_tag"]:
            sessset[subj_key][folder].add(res["sess_tag"])
        study_tuple = (
            res.get("study_uid", "") or "",
            res.get("study_date", "") or "",
            res.get("study_time", "") or "",
        )
        if key not in study_uids[subj_key][folder]:
            study_uids[subj_key][folder][key] = study_tuple
        # Track only studies that have a real UID; ignore blank ones.
        if study_tuple[0]:
            subject_studies[subj_key].add(study_tuple)
        demo.setdefault(subj_key, res["demo"])

    print(f"Subjects found            : {len(demo)}")
    total_series = sum(
        len(seq_dict)
        for subj in counts.values()
        for seq_dict in subj.values()
    )
    print(f"Unique Series instances   : {total_series}")

    # Subject identity clustering (architecture.md §4.1).
    # The per-record identity key is ``pid||given||family``. Across visits
    # any of these fields can be inconsistent (anonymisation rewrites,
    # operator typos, swapped given/family). Build a union-find over the
    # unique identity tuples: link tuples that share any non-placeholder
    # field; each connected component becomes one BIDS subject. See
    # :mod:`bidsmgr.inventory.subject_identity`.
    identity_to_tuple: dict[str, IdentityTuple] = {}
    for k in demo:
        if "||" in k:
            parts = k.split("||")
            if len(parts) == 3:
                identity_to_tuple[k] = normalize_tuple(*parts)
                continue
        # Fallback path (no PID/name): treat the raw key as a placeholder PID.
        identity_to_tuple[k] = normalize_tuple(k, "", "")

    cluster_root_for_tuple = cluster_subjects(set(identity_to_tuple.values()))

    cluster_roots = sorted(set(cluster_root_for_tuple.values()))
    cluster_to_id: dict[IdentityTuple, str] = {
        root: f"sub-{i + 1:03d}" for i, root in enumerate(cluster_roots)
    }

    bids_map: dict[str, str] = {}
    for k, t in identity_to_tuple.items():
        bids_map[k] = cluster_to_id[cluster_root_for_tuple[t]]

    # Session inference operates per *cluster* (one physical subject), not
    # per identity_key. Two visits with different anonymised PatientIDs
    # that the union-find merged still need to produce ses-1 / ses-2.
    studies_per_cluster: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for k, study_set in subject_studies.items():
        bids_id = bids_map.get(k)
        if not bids_id:
            continue
        studies_per_cluster[bids_id].update(study_set)

    inferred_session: dict[str, dict[tuple[str, str, str], str]] = {}
    for bids_id, study_set in studies_per_cluster.items():
        if len(study_set) <= 1:
            continue
        ordered = sorted(study_set, key=lambda t: (t[1], t[2], t[0]))
        inferred_session[bids_id] = {
            tup: f"ses-{idx + 1:d}" for idx, tup in enumerate(ordered)
        }

    # Build rows.
    rows: list[dict] = []
    for subj_key in sorted(counts):
        given_name = demo[subj_key]["GivenName"]
        for folder in sorted(counts[subj_key]):
            ses_labels = sorted(sessset[subj_key][folder])
            session = ses_labels[0] if len(ses_labels) == 1 else ""
            rep_counter: dict = defaultdict(int)
            for (series, uid), n_files in sorted(counts[subj_key][folder].items()):
                fine_mod = mods[subj_key][folder][(series, uid)]
                img3 = imgtypes[subj_key][folder].get((series, uid), "")
                include = 0 if fine_mod in SKIP_MODALITIES else 1
                rep_key = series if fine_mod == "scout" else (series, img3)
                rep_counter[rep_key] += 1
                study_tuple = study_uids[subj_key][folder].get((series, uid), ("", "", ""))

                # Session: path-derived (single ses-X tag in folder) wins,
                # otherwise fall back to inferred longitudinal label.
                row_session = session
                if not row_session:
                    bids_id = bids_map[subj_key]
                    inferred = inferred_session.get(bids_id, {}).get(study_tuple)
                    if inferred:
                        row_session = inferred

                rows.append({
                    "subject": given_name,
                    "BIDS_name": bids_map[subj_key],
                    "session": row_session,
                    "source_folder": folder,
                    "include": include,
                    "sequence": series,
                    "series_uid": uid,
                    "rep": rep_counter[rep_key] if rep_counter[rep_key] > 1 else "",
                    "image_type": img3,
                    "acq_time": acq_times[subj_key][folder].get((series, uid), ""),
                    "modality": fine_mod,
                    "modality_bids": modality_to_container(fine_mod),
                    "n_files": n_files,
                    "study_instance_uid": study_tuple[0],
                    "study_date": study_tuple[1],
                    "study_time": study_tuple[2],
                    "_source_dir": file_dirs[subj_key][folder].get((series, uid), ""),
                    **demo[subj_key],
                })

    df = pd.DataFrame(rows)

    # Collapse magnitude/phase rows for fieldmaps (v0.2.5 behaviour).
    if not df.empty:
        df = _collapse_fieldmap_rows(df)
        df = _assign_chronological_rep(df)

    # Add proposed_* columns as empty (filled by CLI orchestrator after BidsGuess).
    for col in ("proposed_datatype", "proposed_basename", "Proposed BIDS name"):
        if col not in df.columns:
            df[col] = ""

    # Stamp the dataset slug across every row. ``None`` leaves it blank;
    # the CLI orchestrator resolves a default and forwards it here.
    df["dataset"] = dataset or ""

    if not df.empty:
        df.sort_values(["BIDS_name", "subject", "session", "acq_time"], inplace=True)

    # Stash the per-UID file map on the DataFrame so callers (the CLI's
    # probe_convert pass) can find the source DICOMs of each detected
    # sequence without re-walking the disk. ``DataFrame.attrs`` survives
    # standard pandas ops (slicing, copy, to_csv).
    df.attrs["files_by_uid"] = {k: list(v) for k, v in files_by_uid.items()}

    visible_columns = (
        [c for c in TSV_COLUMNS if c in df.columns]
        + [c for c in DATASET_COLUMNS if c in df.columns]
        + [c for c in EXTENDED_COLUMNS if c in df.columns]
    )

    if output_tsv:
        df.to_csv(output_tsv, sep="\t", index=False, columns=visible_columns)
        print(f"Inventory written to: {output_tsv}")

    return df


def _assign_chronological_rep(df: pd.DataFrame) -> pd.DataFrame:
    """Populate ``rep`` with the chronological position within each
    ``(BIDS_name, session, sequence, image_type)`` group.

    Within a group (same subject, same session, same SeriesDescription,
    same image_type), rows are ordered by ``acq_time`` (then ``series_uid``
    as deterministic tiebreaker), and assigned ``1, 2, 3, …``. Singleton
    groups receive an empty string so the column makes the repetition
    visually obvious without polluting one-off acquisitions.
    """

    if df.empty:
        return df

    df = df.copy()
    keys = ["BIDS_name", "session", "sequence", "image_type"]
    sort_keys = keys + ["acq_time", "series_uid"]
    # Stable sort by acquisition time inside each group.
    df.sort_values(sort_keys, inplace=True, kind="stable")
    counts = df.groupby(keys).cumcount() + 1
    sizes = df.groupby(keys)["sequence"].transform("size")
    rep_series = counts.where(sizes > 1, "")
    df["rep"] = rep_series.astype(object)
    return df


def _collapse_fieldmap_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Merge magnitude/phase fieldmap rows the way v0.2.5 did.

    Same ``(BIDS_name, session, source_folder, sequence)`` grouped by
    acquisition-time minute → joined ``series_uid`` (``|``-separated) and
    summed ``n_files``. Run-numbering happens via the ``rep`` column.
    """

    fmap_mask = df["modality"] == "fmap"
    if not fmap_mask.any():
        return df

    base_cols = ["BIDS_name", "session", "source_folder", "sequence"]
    fmap_df = df[fmap_mask].copy()
    fmap_df["acq_group"] = fmap_df["acq_time"].apply(lambda t: str(t)[:4])

    group_cols = base_cols + ["acq_group"]
    fmap_df["uid_list"] = fmap_df["series_uid"]
    fmap_df["img_set"] = fmap_df["image_type"]
    agg_spec = {
        "subject": "first",
        "BIDS_name": "first",
        "session": "first",
        "source_folder": "first",
        "include": "max",
        "sequence": "first",
        "uid_list": lambda x: "|".join(sorted({str(v) for v in x})),
        "img_set": lambda x: "".join(sorted({str(v) for v in x})),
        "acq_time": "first",
        "modality": "first",
        "modality_bids": "first",
        "n_files": "sum",
        "study_instance_uid": "first",
        "study_date": "first",
        "study_time": "first",
        "_source_dir": "first",
        "GivenName": "first",
        "FamilyName": "first",
        "PatientID": "first",
        "PatientSex": "first",
        "PatientAge": "first",
        "StudyDescription": "first",
    }
    fmap_df = fmap_df.groupby(group_cols, as_index=False).agg(agg_spec)
    fmap_df.rename(columns={"uid_list": "series_uid", "img_set": "image_type"}, inplace=True)
    fmap_df.drop(columns=["acq_group"], inplace=True)
    fmap_df.sort_values(base_cols + ["acq_time"], inplace=True)
    fmap_df["rep"] = (fmap_df.groupby(base_cols).cumcount() + 1).astype(object)
    repeat_mask = fmap_df.groupby(base_cols)["rep"].transform("count") > 1
    fmap_df.loc[~repeat_mask, "rep"] = ""

    return pd.concat([df[~fmap_mask], fmap_df], ignore_index=True, sort=False)


__all__ = [
    "TSV_COLUMNS",
    "EXTENDED_COLUMNS",
    "DICOM_EXTS",
    "is_dicom_file",
    "normalize_image_type",
    "classify_fieldmap_type",
    "scan_dicoms_long",
]
