"""Schema-driven post-conversion BIDS metadata generator.

Operates on a converted BIDS dataset on disk (``<bids_root>/sub-*/...``)
and emits the dataset-level files BIDS expects but that dcm2niix doesn't
write:

* ``dataset_description.json`` (REQUIRED) — append/merge with what the
  converter already wrote (preserves ``GeneratedBy`` history).
* ``participants.tsv`` + ``participants.json`` (REQUIRED if >1 subject).
* ``README`` + ``CHANGES`` — minimal scaffolds, never overwrite.
* ``sub-*[_ses-*]_scans.tsv`` per subject/session (RECOMMENDED).
* Sidecar audit + filename-derivable fills (``TaskName`` from
  ``_task-<label>``).

Audit is schema-driven via :func:`bidsmgr.schema.required_sidecar_fields`
/ :func:`bidsmgr.schema.recommended_sidecar_fields` (replacing v0.2.5's
hardcoded ``_REQUIRED_SIDECAR_FIELDS`` table).

Reference: architecture.md §7 tail; v0.2.5
``BIDS-Manager/bids_manager/bids_metadata_engine.py`` (port source).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

import bidsmgr
from .. import schema as schema_mod
from .types import DatasetMetadata, MetadataReport, SidecarFill, TodoFill

# The literal placeholder value written by ``--fill-todos`` for every
# missing required / recommended field. Deliberately just the string
# ``"TODO"`` (not ``TODO_<fieldname>``) — the user wants a clean marker
# they can grep for, fill, and remove.
_TODO_VALUE = "TODO"

log = logging.getLogger(__name__)


_BIDS_DATATYPE_NAMES: tuple[str, ...] = (
    "anat", "func", "dwi", "fmap", "perf",
    "meg", "eeg", "ieeg", "beh", "pet", "micr", "nirs",
)

_TASKNAME_RE = re.compile(r"_task-([A-Za-z0-9]+)")


# Some bidsschematools field names carry a ``__<rule>`` suffix used
# internally to disambiguate context-dependent rules (e.g. fmap/phase1
# requires ``EchoTime`` but the schema reports it as ``EchoTime__fmap``).
# The actual JSON field is the prefix.
def _canonical_field_name(name: str) -> str:
    return name.split("__", 1)[0]


# (datatype, suffix) → tuple of mutually-exclusive required-field
# alternatives. If ANY field in a tuple is present in the sidecar, all
# fields in that tuple are considered satisfied. The schema engine
# doesn't yet model mutual exclusivity, so we encode the few cases the
# spec defines (BIDS 1.10).
_REQUIRED_ALTERNATIVES: dict[tuple[str, str], tuple[tuple[str, ...], ...]] = {
    # func/bold (and related): RepetitionTime XOR VolumeTiming.
    ("func", "bold"):  (("RepetitionTime", "VolumeTiming"),),
    ("func", "sbref"): (("RepetitionTime", "VolumeTiming"),),
    ("func", "cbv"):   (("RepetitionTime", "VolumeTiming"),),
    ("func", "phase"): (("RepetitionTime", "VolumeTiming"),),
}


# Recommended fields for ``dataset_description.json`` per BIDS 1.10.
# These aren't surfaced by ``bidsmgr.schema.recommended_sidecar_fields``
# (which keys off (datatype, suffix)), so we list them here for the
# audit and ``--fill-todos`` pass.
_DATASET_DESCRIPTION_RECOMMENDED: tuple[str, ...] = (
    "License", "Authors", "Acknowledgements", "HowToAcknowledge",
    "Funding", "EthicsApprovals", "ReferencesAndLinks", "DatasetDOI",
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_metadata(
    bids_root: Path,
    *,
    inventory_tsv: Optional[Path] = None,
    dataset_meta: Optional[DatasetMetadata] = None,
    fill_todos: bool = False,
    write_report: bool = True,
    generator_label: str = "bidsmgr",
) -> MetadataReport:
    """Generate dataset-level metadata for the BIDS root at ``bids_root``.

    Parameters
    ----------
    bids_root
        A BIDS dataset root (the directory containing ``sub-*/`` and
        ``dataset_description.json``).
    inventory_tsv
        Optional inventory TSV produced by ``bidsmgr-scan``. Used only
        to enrich ``participants.tsv`` with demographics; the engine
        works without it (cells default to ``"n/a"``).
    dataset_meta
        Caller-supplied fields for ``dataset_description.json``. The
        ``Name`` defaults to ``bids_root.name`` if not provided.
    fill_todos
        When ``True``, every missing required + recommended field across
        every sidecar (and the recommended fields of
        ``dataset_description.json``) gets the literal string ``"TODO"``
        written. Existing values are never overwritten. The fill is
        recorded in ``report.todo_fills`` and in the JSON report.
    write_report
        When ``True`` (default), write the full ``MetadataReport`` to
        ``<bids_root>/.bidsmgr/metadata_report.json`` after every run.
    generator_label
        ``GeneratedBy.Name`` to record. Defaults to ``"bidsmgr"``.

    Returns
    -------
    MetadataReport
        Summary of what was written, what sidecar fills were applied,
        which sidecars are missing required/recommended fields, and
        which fields got TODO placeholders inserted.
    """
    bids_root = Path(bids_root)
    if not bids_root.is_dir():
        raise FileNotFoundError(f"BIDS root not found: {bids_root}")

    meta = dataset_meta or DatasetMetadata(name=bids_root.name)
    if meta.name == "Untitled BIDS Dataset":
        # Caller supplied an empty DatasetMetadata; use the directory name
        # so dataset_description.json reads sensibly.
        meta = meta.model_copy(update={"name": bids_root.name})

    report = MetadataReport(
        bidsmgr_version=str(getattr(bidsmgr, "__version__", "0.0.0")),
        bids_root=bids_root,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    _write_dataset_description(bids_root, meta, generator_label, report)
    _write_participants(bids_root, inventory_tsv, report)
    _write_readme(bids_root, meta.name, report)
    _write_changes(bids_root, report)
    _refresh_scans_tsv(bids_root, report)
    _fill_and_audit_sidecars(bids_root, report, fill_todos=fill_todos)
    _audit_dataset_description(bids_root, report, fill_todos=fill_todos)

    if write_report:
        _write_metadata_report(bids_root, report)

    return report


# ---------------------------------------------------------------------------
# dataset_description.json
# ---------------------------------------------------------------------------


def _write_dataset_description(
    bids_root: Path,
    meta: DatasetMetadata,
    generator_label: str,
    report: MetadataReport,
) -> None:
    """Write/merge ``dataset_description.json``.

    Merge semantics:

    * ``Name`` / ``BIDSVersion`` / ``DatasetType`` from ``meta`` always win.
    * Optional fields from ``meta`` (License, Authors, …) win when supplied.
    * ``GeneratedBy`` from disk is **preserved** (the converter already
      appended an entry); a new bidsmgr metadata entry is appended.
    * Any other keys already present (e.g. user-edited fields) are
      preserved verbatim.
    """
    out = bids_root / "dataset_description.json"
    existing: dict = {}
    if out.exists():
        try:
            data = json.loads(out.read_text())
            if isinstance(data, dict):
                existing = data
        except (OSError, json.JSONDecodeError) as exc:
            report.warnings.append(
                f"could not parse existing dataset_description.json ({exc!r}); "
                "rewriting from scratch"
            )

    merged: dict[str, object] = dict(existing)
    merged["Name"] = meta.name
    merged["BIDSVersion"] = meta.bids_version
    merged["DatasetType"] = meta.dataset_type

    if meta.license:
        merged["License"] = meta.license
    if meta.authors:
        merged["Authors"] = list(meta.authors)
    if meta.acknowledgements:
        merged["Acknowledgements"] = meta.acknowledgements
    if meta.how_to_acknowledge:
        merged["HowToAcknowledge"] = meta.how_to_acknowledge
    if meta.funding:
        merged["Funding"] = list(meta.funding)
    if meta.ethics_approvals:
        merged["EthicsApprovals"] = list(meta.ethics_approvals)
    if meta.references_and_links:
        merged["ReferencesAndLinks"] = list(meta.references_and_links)
    if meta.dataset_doi:
        merged["DatasetDOI"] = meta.dataset_doi
    if meta.source_datasets:
        merged["SourceDatasets"] = list(meta.source_datasets)

    # GeneratedBy: preserve everything already there (the converter wrote
    # one entry per convert run), append the metadata-engine entry once.
    # Dedupe by (Name, Version, Description) so re-running the metadata
    # engine doesn't pile up identical entries on each run.
    existing_gb = merged.get("GeneratedBy")
    gen_by: list = list(existing_gb) if isinstance(existing_gb, list) else []
    new_entry = {
        "Name": generator_label,
        "Version": str(getattr(bidsmgr, "__version__", "0.0.0")),
        "Description": "metadata engine",
    }
    if not any(
        isinstance(e, dict)
        and e.get("Name") == new_entry["Name"]
        and e.get("Version") == new_entry["Version"]
        and e.get("Description") == new_entry["Description"]
        for e in gen_by
    ):
        gen_by.append(new_entry)
    merged["GeneratedBy"] = gen_by

    out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    report.files_written.append(out)


# ---------------------------------------------------------------------------
# participants.tsv + participants.json
# ---------------------------------------------------------------------------


_PARTICIPANT_COLUMNS: tuple[str, ...] = (
    "participant_id", "age", "sex", "given_name", "family_name", "patient_id",
)


_PARTICIPANT_DESCRIPTIONS: dict[str, dict[str, object]] = {
    "participant_id": {"Description": "Unique participant identifier"},
    "age": {"Description": "Age of the participant", "Units": "years"},
    "sex": {
        "Description": "Biological sex",
        "Levels": {"M": "male", "F": "female", "O": "other", "n/a": "not available"},
    },
    "given_name": {"Description": "Subject given name (kept for internal traceability)"},
    "family_name": {"Description": "Subject family name (kept for internal traceability)"},
    "patient_id": {"Description": "Original DICOM PatientID"},
}


def _write_participants(
    bids_root: Path, inventory_tsv: Optional[Path], report: MetadataReport,
) -> None:
    subjects = sorted(p.name for p in bids_root.glob("sub-*") if p.is_dir())
    if not subjects:
        return

    demo_lookup = _load_demographics_from_inventory(inventory_tsv, report)

    rows: list[dict[str, str]] = []
    for sid in subjects:
        row: dict[str, str] = {"participant_id": sid}
        extra = demo_lookup.get(sid, {})
        for col in ("age", "sex", "given_name", "family_name", "patient_id"):
            v = extra.get(col, "")
            row[col] = v if v else "n/a"
        rows.append(row)

    df_new = pd.DataFrame(rows, columns=list(_PARTICIPANT_COLUMNS))

    out_tsv = bids_root / "participants.tsv"
    df_existing: Optional[pd.DataFrame] = None
    if out_tsv.exists():
        try:
            df_existing = pd.read_csv(
                out_tsv, sep="\t", dtype=str, keep_default_na=False,
            )
        except OSError as exc:
            report.warnings.append(f"could not read {out_tsv}: {exc}")
            df_existing = None

    df_out, merged = _merge_participants(df_existing, df_new)

    # When building from scratch, drop columns that are entirely "n/a";
    # when merging, keep every column the upstream tool wrote (those are
    # part of the dataset's schema even if our enrichment didn't fill them).
    if not merged:
        keep = ["participant_id"] + [
            c for c in df_out.columns
            if c != "participant_id" and (df_out[c].astype(str) != "n/a").any()
        ]
        df_out = df_out[keep]

    df_out.to_csv(out_tsv, sep="\t", index=False)
    report.files_written.append(out_tsv)

    out_json = bids_root / "participants.json"
    json_payload = {
        col: _PARTICIPANT_DESCRIPTIONS[col]
        for col in df_out.columns
        if col in _PARTICIPANT_DESCRIPTIONS
    }
    out_json.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")
    report.files_written.append(out_json)


def _load_demographics_from_inventory(
    inventory_tsv: Optional[Path], report: MetadataReport,
) -> dict[str, dict[str, str]]:
    """Read the inventory TSV and group demographics by ``BIDS_name``.

    bidsmgr-scan writes ``GivenName`` / ``FamilyName`` / ``PatientID`` /
    ``PatientAge`` / ``PatientSex`` columns (PascalCase) — see
    ``inventory/mri_dicom.py:TSV_COLUMNS``.
    """
    if inventory_tsv is None:
        return {}
    inventory_tsv = Path(inventory_tsv)
    if not inventory_tsv.exists():
        report.warnings.append(f"inventory TSV not found: {inventory_tsv}")
        return {}

    try:
        # dtype=str keeps "001" as a string; keep_default_na=False keeps
        # blank cells as "" instead of NaN.
        df = pd.read_csv(
            inventory_tsv, sep="\t", dtype=str, keep_default_na=False,
        )
    except OSError as exc:
        report.warnings.append(f"could not read inventory TSV {inventory_tsv}: {exc}")
        return {}

    if df.empty or "BIDS_name" not in df.columns:
        return {}

    lookup: dict[str, dict[str, str]] = {}
    for bids_name, sub_df in df.groupby("BIDS_name"):
        bids_name = str(bids_name).strip()
        if not bids_name:
            continue
        pid = bids_name if bids_name.startswith("sub-") else f"sub-{bids_name}"
        head = sub_df.iloc[0]
        lookup[pid] = {
            "given_name": str(head.get("GivenName", "") or ""),
            "family_name": str(head.get("FamilyName", "") or ""),
            "patient_id": str(head.get("PatientID", "") or ""),
            "age": str(head.get("PatientAge", "") or head.get("age", "") or ""),
            "sex": str(head.get("PatientSex", "") or head.get("sex", "") or ""),
        }
    return lookup


def _merge_participants(
    df_existing: Optional[pd.DataFrame],
    df_new: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """Merge fresh participants data with an existing participants.tsv.

    Returns ``(merged_df, was_merge)``. ``was_merge=False`` when there
    was no existing file to merge with. Merge rule: keep the existing
    file's columns and values verbatim; only fill cells that are blank
    or ``"n/a"`` with new data.
    """
    if df_existing is None or "participant_id" not in df_existing.columns:
        return df_new, False

    merged = df_existing.set_index("participant_id")
    new_indexed = df_new.set_index("participant_id")

    for sid in new_indexed.index:
        if sid not in merged.index:
            merged.loc[sid] = "n/a"
    for col in new_indexed.columns:
        if col not in merged.columns:
            merged[col] = "n/a"
        for sid in new_indexed.index:
            incoming = new_indexed.at[sid, col]
            if incoming and incoming != "n/a":
                current = merged.at[sid, col] if sid in merged.index else ""
                if not current or current == "n/a":
                    merged.at[sid, col] = incoming

    return merged.reset_index(), True


# ---------------------------------------------------------------------------
# README + CHANGES
# ---------------------------------------------------------------------------


def _write_readme(bids_root: Path, name: str, report: MetadataReport) -> None:
    """Write a minimal README scaffold. Never overwrites a hand-edited file."""
    out = bids_root / "README"
    if out.exists():
        return
    body = (
        f"# {name}\n\n"
        "This BIDS dataset was generated by bidsmgr.\n\n"
        "Edit this README to describe acquisition details, study design,\n"
        "and any deviations from the standard BIDS layout.\n"
    )
    out.write_text(body, encoding="utf-8")
    report.files_written.append(out)


def _write_changes(bids_root: Path, report: MetadataReport) -> None:
    """Seed CHANGES with version 1.0.0 entry. Never overwrites."""
    out = bids_root / "CHANGES"
    if out.exists():
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out.write_text(f"1.0.0 {ts}\n  - Initial release.\n", encoding="utf-8")
    report.files_written.append(out)


# ---------------------------------------------------------------------------
# *_scans.tsv
# ---------------------------------------------------------------------------


def _refresh_scans_tsv(bids_root: Path, report: MetadataReport) -> None:
    """Generate one ``*_scans.tsv`` per subject (or subject/session).

    Lists every NIfTI under the subject's tree with its ``acq_time``
    parsed from the JSON sidecar's ``AcquisitionDateTime`` /
    ``AcquisitionTime``. Missing acq times are recorded as ``"n/a"`` per
    BIDS convention. Regenerated on every run (no merge).
    """
    for subj_dir in sorted(bids_root.glob("sub-*")):
        if not subj_dir.is_dir():
            continue
        ses_dirs = [s for s in sorted(subj_dir.glob("ses-*")) if s.is_dir()]
        roots = ses_dirs or [subj_dir]
        for root in roots:
            rows: list[dict[str, str]] = []
            for nii in sorted(root.rglob("*.nii*")):
                if not nii.is_file():
                    continue
                rel = nii.relative_to(root).as_posix()
                json_path = _matching_json(nii)
                acq_time = "n/a"
                if json_path.exists():
                    try:
                        meta = json.loads(json_path.read_text())
                        adt = meta.get("AcquisitionDateTime") or meta.get("AcquisitionTime")
                        if isinstance(adt, str) and adt:
                            acq_time = adt
                    except (OSError, json.JSONDecodeError):
                        pass
                rows.append({"filename": rel, "acq_time": acq_time})
            if not rows:
                continue
            ses_part = f"_{root.name}" if root.name.startswith("ses-") else ""
            out = root / f"{subj_dir.name}{ses_part}_scans.tsv"
            pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
            report.files_written.append(out)


# ---------------------------------------------------------------------------
# Sidecar fill + audit
# ---------------------------------------------------------------------------


def _fill_and_audit_sidecars(
    bids_root: Path,
    report: MetadataReport,
    *,
    fill_todos: bool = False,
) -> None:
    """Walk every sidecar JSON, fill derivable fields, audit required ones.

    Fills are non-destructive: existing keys are never overwritten.
    ``TaskName`` (from the ``_task-<label>`` token in the filename) is
    filled when missing. The audit then checks required + recommended
    fields per the schema engine and records anything still missing.

    When ``fill_todos=True``, every still-missing required + recommended
    field gets the literal string ``"TODO"`` written into the file. The
    audit messages still report what *was* missing (the report shows
    what got TODO'd in ``report.todo_fills``), but the file ends the
    run with no field absent.
    """
    for json_path in sorted(bids_root.rglob("*.json")):
        if json_path.name in ("dataset_description.json", "participants.json"):
            continue
        # Skip files at the dataset root (CHANGES, README — they're not JSON).
        if json_path.parent == bids_root:
            continue

        datatype, suffix = _infer_datatype_suffix(json_path, bids_root)
        if not datatype or not suffix:
            continue

        try:
            data = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue

        # Filename-derivable fills (TaskName).
        fills: dict[str, object] = {}
        if datatype == "func" and suffix in {"bold", "sbref", "cbv", "phase"}:
            m = _TASKNAME_RE.search(json_path.name)
            if m and "TaskName" not in data:
                fills["TaskName"] = m.group(1)

        # Required field audit.
        try:
            required = schema_mod.required_sidecar_fields(datatype, suffix)
        except (KeyError, ValueError, AttributeError):
            required = []
        required_names = {_canonical_field_name(f.name) for f in required}
        # Mutual exclusivity: if any alternative is already present,
        # drop the whole alternative group from required_names.
        for alternatives in _REQUIRED_ALTERNATIVES.get((datatype, suffix), ()):
            if any(alt in data or alt in fills for alt in alternatives):
                required_names -= set(alternatives)

        rel = json_path.relative_to(bids_root)
        missing_req: list[str] = sorted(
            n for n in required_names if n not in data and n not in fills
        )
        for name in missing_req:
            report.missing_required.append(f"{rel}: missing {name!r}")

        # Recommended field audit.
        try:
            recommended = schema_mod.recommended_sidecar_fields(datatype, suffix)
        except (KeyError, ValueError, AttributeError):
            recommended = []
        recommended_names: list[str] = []
        seen_rec: set[str] = set()
        for field_info in recommended:
            name = _canonical_field_name(field_info.name)
            if name in seen_rec:
                continue
            seen_rec.add(name)
            recommended_names.append(name)
        missing_rec: list[str] = [
            n for n in recommended_names if n not in data and n not in fills
        ]
        for name in missing_rec:
            report.missing_recommended.append(
                f"{rel}: missing recommended {name!r}"
            )

        # Apply --fill-todos for everything still missing.
        todo_added: list[str] = []
        if fill_todos:
            for name in missing_req + missing_rec:
                if name not in data and name not in fills:
                    fills[name] = _TODO_VALUE
                    todo_added.append(name)

        # Write back if anything changed.
        if fills:
            data.update(fills)
            try:
                json_path.write_text(
                    json.dumps(data, indent=2) + "\n", encoding="utf-8",
                )
            except OSError as exc:
                report.warnings.append(f"could not write {json_path}: {exc}")
                continue

            # Record filename-derived fills (TaskName) separately from TODOs.
            non_todo = {k: v for k, v in fills.items() if v != _TODO_VALUE}
            if non_todo:
                report.sidecar_fills.append(
                    SidecarFill(sidecar=json_path, fields=non_todo),
                )
            if todo_added:
                report.todo_fills.append(
                    TodoFill(sidecar=json_path, fields=sorted(todo_added)),
                )


def _audit_dataset_description(
    bids_root: Path,
    report: MetadataReport,
    *,
    fill_todos: bool = False,
) -> None:
    """Audit ``dataset_description.json`` recommended fields.

    The schema engine doesn't expose dataset-level recommended fields
    (its API is keyed by ``(datatype, suffix)``), so we use the small
    BIDS 1.10 list in ``_DATASET_DESCRIPTION_RECOMMENDED``. With
    ``fill_todos=True``, missing recommended fields get the literal
    ``"TODO"`` string written.
    """
    path = bids_root / "dataset_description.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return

    rel = path.name
    missing: list[str] = [
        f for f in _DATASET_DESCRIPTION_RECOMMENDED if f not in data
    ]
    for name in missing:
        report.missing_recommended.append(f"{rel}: missing recommended {name!r}")

    if not (fill_todos and missing):
        return

    for name in missing:
        data[name] = _TODO_VALUE
    try:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        report.todo_fills.append(TodoFill(sidecar=path, fields=sorted(missing)))
    except OSError as exc:
        report.warnings.append(f"could not write {path}: {exc}")


def _write_metadata_report(bids_root: Path, report: MetadataReport) -> None:
    """Persist the full ``MetadataReport`` next to the dataset.

    Always written to ``<bids_root>/.bidsmgr/metadata_report.json``.
    Pydantic handles the JSON encoding (Path → str) via ``model_dump``.
    The report records itself in ``files_written`` so downstream
    consumers (the GUI in particular) can pick it up.
    """
    out_dir = bids_root / ".bidsmgr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "metadata_report.json"
    # Self-record before serializing so the on-disk report includes its
    # own path (chicken-and-egg avoided by appending first).
    report.files_written.append(out)
    payload = report.model_dump(mode="json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _infer_datatype_suffix(
    json_path: Path, bids_root: Path,
) -> tuple[str, str]:
    """Infer ``(datatype, suffix)`` from the sidecar's path + filename.

    Path looks like ``<bids_root>/sub-X[/ses-Y]/<datatype>/sub-X..._<suffix>.json``.
    """
    try:
        rel_parts = json_path.relative_to(bids_root).parts
    except ValueError:
        return "", ""

    datatype = ""
    for part in rel_parts:
        if part in _BIDS_DATATYPE_NAMES:
            datatype = part
            break

    stem = json_path.name
    if stem.endswith(".json"):
        stem = stem[: -len(".json")]
    # The BIDS suffix is the last underscore-delimited token that is NOT
    # an entity (entity tokens contain a hyphen, suffixes don't).
    suffix = ""
    for tok in reversed(stem.split("_")):
        if "-" not in tok:
            suffix = tok
            break
    return datatype, suffix


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _matching_json(image: Path) -> Path:
    """Return the JSON sidecar path for a NIfTI image."""
    if image.suffix == ".gz" and image.name.endswith(".nii.gz"):
        return image.with_suffix("").with_suffix(".json")
    return image.with_suffix(".json")


__all__ = ["run_metadata"]
