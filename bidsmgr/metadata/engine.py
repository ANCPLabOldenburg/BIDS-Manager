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
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import bidsmgr
from .. import schema as schema_mod
from .demographics import (
    load_participants_table,
    normalize_age,
    merge_demographics,
    normalize_handedness,
    normalize_sex,
)
from .phenotype import load_sidecar_dictionary, write_phenotype
from .types import DatasetMetadata, MetadataReport, SidecarFill, TodoFill
from ..recording_meta import load_spec, scaffold_sidecar_path
from ..editor.rename import PATH_FIELDS

# The literal placeholder value written by ``--fill-todos`` for every
# missing required / recommended field. Deliberately just the string
# ``"TODO"`` (not ``TODO_<fieldname>``) — the user wants a clean marker
# they can grep for, fill, and remove.
_TODO_VALUE = "TODO"

# What BIDS itself writes when a field was asked and has no answer. Valid
# wherever the schema declares it, unlike the TODO marker.
_NA_VALUE = "n/a"

# Sentinel: this field gets no placeholder at all.
NO_TODO = object()

# Fields that LINK files by an identifier: the same placeholder on both
# sides IS a link. "TODO" in the B0FieldIdentifier of two phase images made
# sdcflows estimate a field from them, and "TODO" in a run's B0FieldSource
# then pointed the run at it (or, with no such field, at nothing, so
# fMRIPrep corrected the run with no fieldmap at all and never fell back to
# IntendedFor). Absent is the honest state: the run has no fieldmap.
_LINK_FIELDS = frozenset({"B0FieldIdentifier", "B0FieldSource"})

# Fields that NAME another file. "TODO" there is a reference to a file that
# does not exist, which the validator reports as an error
# (ASSOCIATED_EMPTY_ROOM on every MEG recording at the recommended scope), so
# a gap in one stays a gap. The list is the one rename keeps up to date.
_PATH_FIELDS = frozenset(PATH_FIELDS)

# How much of what the standard declares a fill should mark. Nested: each
# includes the ones before it.
FILL_NONE = "none"
FILL_REQUIRED = "required"
FILL_RECOMMENDED = "recommended"
FILL_OPTIONAL = "optional"
FILL_SCOPES = (FILL_NONE, FILL_REQUIRED, FILL_RECOMMENDED, FILL_OPTIONAL)
FILL_SCOPE_LABELS = {
    FILL_NONE: "None: leave every gap absent",
    FILL_REQUIRED: "Required fields only",
    FILL_RECOMMENDED: "Required and recommended (default)",
    FILL_OPTIONAL: "Everything the standard declares, including optional",
}


def todo_value_for(field) -> object:
    """The placeholder to write for a field, or ``NO_TODO`` for none.

    A placeholder marks a gap. It must not itself BE a violation, or the
    dataset gains an error for a field that was merely absent, which is worse
    than the gap it marks. So the marker is chosen from what the schema says
    the field accepts, and there is more than one:

    * a free-text string field takes ``"TODO"``, which is greppable and
      obviously not a value;
    * an array of strings takes ``["TODO"]``;
    * a field that accepts the literal ``"n/a"`` takes ``"n/a"``, which is how
      BIDS itself spells "this was asked and there is no answer". That covers
      a large class the old rule skipped outright, because the schema writes
      those as ``anyOf`` (number or ``"n/a"``) and the plain ``type`` is
      empty;
    * a controlled vocabulary that INCLUDES ``"n/a"`` takes it, for the same
      reason.

    What is left is a field where no honest marker exists: a number, a
    boolean, an enum of real choices. Inventing one would be writing a value
    nobody stated. Those stay absent and stay in the missing-field report,
    and :func:`unmarkable_fields` explains which and why, so the gap is
    visible rather than silent.

    ``field`` is a :class:`~bidsmgr.schema.FieldInfo`.
    """
    accepts = tuple(getattr(field, "accepts", ()) or ())
    enum = tuple(getattr(field, "enum", ()) or ())
    item_type = getattr(field, "item_type", "")

    name = canonical_field_name(str(getattr(field, "name", "")))
    if name in _LINK_FIELDS or name in _PATH_FIELDS:
        return NO_TODO
    if getattr(field, "accepts_na", False) or "n/a" in enum:
        return _NA_VALUE
    if enum:
        # A real vocabulary. Nothing in it means "unanswered".
        return NO_TODO
    # Free text by the schema's own answer, which a format (a time, a URI, a
    # HED version) rules out: "TODO" there is a validation error, not a gap.
    if getattr(field, "accepts_free_text", False):
        return _TODO_VALUE
    if "array" in accepts and item_type == "string" and not getattr(field, "format", ""):
        return [_TODO_VALUE]
    # An array whose item type the schema does not state is NOT assumed to
    # hold strings. That assumption is what wrote ``["TODO"]`` into EchoTime,
    # a number-or-array-of-numbers field, turning a gap into a type error.
    return NO_TODO


def unmarkable_reason(field) -> str:
    """Why this field can take no placeholder, for the report. Empty if it can.

    Said out loud because the alternative is a fill that quietly covers two
    thirds of what is missing and reports itself as complete.
    """
    if todo_value_for(field) is not NO_TODO:
        return ""
    name = canonical_field_name(str(getattr(field, "name", "")))
    if name in _LINK_FIELDS:
        return "links files by its value: a placeholder would link them"
    if name in _PATH_FIELDS:
        return "names another file: a placeholder would name one that does not exist"
    fmt = getattr(field, "format", "")
    if fmt:
        return f"must be written as a {fmt.replace('_', ' ')}, which a placeholder is not"
    enum = tuple(getattr(field, "enum", ()) or ())
    if enum:
        shown = ", ".join(str(v) for v in enum[:4])
        more = " ..." if len(enum) > 4 else ""
        return f"only accepts {shown}{more}"
    accepts = tuple(getattr(field, "accepts", ()) or ())
    if accepts:
        return f"accepts only {' or '.join(accepts)}, which has no marker"
    return "the schema declares no type for it"


def scope_levels(scope: str) -> frozenset[str]:
    """Which requirement levels a fill scope covers.

    The scopes nest, because that is how a user thinks about it: required is
    the floor, recommended adds what analyses usually need, optional adds
    everything else the standard declares. Deprecated is in none of them: a
    placeholder in a field BIDS is retiring is work nobody should do.
    """
    if scope == FILL_REQUIRED:
        return frozenset({"required"})
    if scope == FILL_RECOMMENDED:
        return frozenset({"required", "recommended"})
    if scope == FILL_OPTIONAL:
        return frozenset({"required", "recommended", "optional"})
    return frozenset()


log = logging.getLogger(__name__)


def _datatype_names() -> frozenset[str]:
    """The datatype folders the standard defines, read from it.

    A hand-kept list stood here and predated ``mrs``, ``emg`` and ``motion``,
    so every spectroscopy sidecar was skipped: no placeholder, no template
    answer, no level in the form.
    """
    return frozenset(schema_mod.list_datatypes())

_TASKNAME_RE = re.compile(r"_task-([A-Za-z0-9]+)")


# Some bidsschematools field names carry a ``__<rule>`` suffix used
# internally to disambiguate context-dependent rules (e.g. fmap/phase1
# requires ``EchoTime`` but the schema reports it as ``EchoTime__fmap``).
# The actual JSON field is the prefix.
def canonical_field_name(name: str) -> str:
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


# What a CITATION.cff states for the dataset. With one present, the schema's
# checks want these out of dataset_description.json: Authors as an error,
# the rest as a warning.
_CITATION_FIELDS = frozenset(
    {"Authors", "License", "HowToAcknowledge", "ReferencesAndLinks"}
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_metadata(
    bids_root: Path,
    *,
    inventory_tsv: Optional[Path] = None,
    dataset_meta: Optional[DatasetMetadata] = None,
    fill_scope: Optional[str] = None,
    write_report: bool = True,
    generator_label: str = "bidsmgr",
    participants_file: Optional[Path] = None,
    phenotype_files: Optional[list[Path]] = None,
    tables: bool = True,
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
    fill_scope
        Which placeholders the dataset ends this run with. ``None`` (the
        default) leaves them alone: none written, none taken back. A scope
        makes them match it: ``"required"``, ``"recommended"`` or
        ``"optional"`` (they nest) give every missing field at those levels a
        placeholder, and a ``"TODO"`` an earlier, wider run wrote into a field
        outside them is taken back; ``"none"`` takes back every one. Without
        that, narrowing the scope did nothing to a dataset already marked.
        Existing values are NEVER overwritten, and an ``"n/a"`` is never
        taken back, since it cannot be told from one a converter or a person
        wrote. Recorded in ``report.todo_fills`` and ``report.todo_removed``;
        a field that can take no placeholder is in ``report.unmarkable``
        with the reason.
    write_report
        When ``True`` (default), write the full ``MetadataReport`` to
        ``<bids_root>/.bidsmgr/metadata_report.json`` after every run.
    generator_label
        ``GeneratedBy.Name`` to record. Defaults to ``"bidsmgr"``.
    participants_file
        Optional participants spreadsheet (TSV/CSV/XLSX/ODS) keyed by
        ``participant_id``. Its demographic columns (``age`` / ``sex`` /
        ``handedness``) override the inventory-derived values.
    phenotype_files
        Optional list of phenotype measure tables. Each becomes
        ``phenotype/<measure>.tsv`` + ``.json``.
    tables
        Write the ``*_scans.tsv`` and ``*_sessions.tsv`` tables
        (``metadata.tables``; the standard makes both optional, so this
        can be turned off). Existing tables are merged, never replaced.

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

    # The dataset's own description, from the scaffold when this run was not
    # told one directly. Without this the metadata step that runs itself after
    # a conversion had no authors, no licence and no funding to write, which is
    # why every dataset came out reporting NO_AUTHORS.
    dataset_meta = _dataset_meta_from_scaffold(inventory_tsv, dataset_meta)
    meta = dataset_meta or DatasetMetadata()
    # Resolve the dataset Name without clobbering one already on disk. An
    # explicit caller-supplied Name wins; otherwise preserve an existing
    # dataset_description.json Name (set by ``bidsmgr create`` or hand-edited);
    # else fall back to the folder name. A metadata run with no ``--name`` must
    # not overwrite a Name the user already chose.
    if meta.name in (None, "", "Untitled BIDS Dataset"):
        resolved = _existing_dataset_name(bids_root) or bids_root.name
        meta = meta.model_copy(update={"name": resolved})

    report = MetadataReport(
        bidsmgr_version=str(getattr(bidsmgr, "__version__", "0.0.0")),
        bids_root=bids_root,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    # Auto-discover phenotype tables + participants spreadsheet from the
    # inventory's recording-metadata scaffold when not explicitly provided (the
    # GUI dataset-metadata dialog stores them there).
    if phenotype_files is None and inventory_tsv is not None:
        phenotype_files = _phenotype_files_from_scaffold(inventory_tsv)
    if participants_file is None and inventory_tsv is not None:
        participants_file = _participants_file_from_scaffold(inventory_tsv)

    _write_dataset_description(
        bids_root, meta, generator_label, report,
        extra_fields=_dataset_extras_from_scaffold(inventory_tsv),
    )
    # Everything the user stated about these files: the per-sequence templates,
    # the per-recording answers, the cells typed into the table, and the dataset
    # block every datatype declares. One pass, one chain, and it lives here
    # rather than in convert so that verb stays a conversion.
    n_stated = _apply_stated_metadata(bids_root, inventory_tsv)
    if n_stated:
        log.info("applied stated metadata to %d sidecar(s)", n_stated)
    _write_participants(bids_root, inventory_tsv, report, participants_file=participants_file)
    write_phenotype(bids_root, phenotype_files, report)
    _write_readme(bids_root, meta.name, report)
    _write_changes(bids_root, report)
    if tables:
        _refresh_scans_tsv(bids_root, report, inventory_tsv)
    # One resolved scope for the whole run, so the two passes below cannot
    # disagree about what was asked for. It used to be a switch AND a scope,
    # and a scope passed with the switch off still marked: that is how the
    # GUI wrote placeholders with "mark missing fields" unticked.
    # The tree has changed since the levels were last read (a conversion, a
    # citation file), and two of them depend on it.
    schema_mod.forget_dataset()
    if fill_scope is not None and fill_scope not in FILL_SCOPES:
        raise ValueError(
            f"invalid fill_scope {fill_scope!r}; expected one of {FILL_SCOPES}"
        )
    levels = None if fill_scope is None else scope_levels(fill_scope)
    report.fill_scope = fill_scope or "unchanged"

    _fill_and_audit_sidecars(bids_root, report, levels=levels)
    _audit_dataset_description(bids_root, report, levels=levels)

    if write_report:
        _write_metadata_report(bids_root, report)

    return report


# ---------------------------------------------------------------------------
# dataset_description.json
# ---------------------------------------------------------------------------


def _existing_dataset_name(bids_root: Path) -> Optional[str]:
    """Return the ``Name`` from an existing ``dataset_description.json``, if any.

    Used so a metadata run with no explicit ``--name`` preserves a Name the user
    already set (via ``bidsmgr create`` or by hand) instead of overwriting it.
    """
    p = bids_root / "dataset_description.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    name = data.get("Name") if isinstance(data, dict) else None
    return name if isinstance(name, str) and name.strip() else None


def describe_dataset(
    existing: dict,
    meta: DatasetMetadata,
    generator_label: str = "bidsmgr",
    extra_fields: Optional[dict] = None,
) -> dict:
    """``dataset_description.json`` as the metadata step leaves it.

    Merge semantics:

    * ``Name`` / ``BIDSVersion`` / ``DatasetType`` from ``meta`` always win.
    * Optional fields from ``meta`` (License, Authors, …) win when supplied.
    * ``GeneratedBy`` from ``existing`` is **preserved** (the converter
      already appended an entry); a new bidsmgr metadata entry is appended.
    * Any other keys already present (e.g. user-edited fields) are
      preserved verbatim.

    Pure, so the BIDS preview shows the description this step writes.
    """
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

    # Fields the user stated that DatasetMetadata does not model. It has one
    # attribute per CLI flag; the schema declares more, and the metadata form
    # offers all of them. Without this they reached the file at conversion and
    # were then absent from a metadata-only run, which is worse than either.
    for key, value in (extra_fields or {}).items():
        if value not in (None, "", [], {}):
            merged[key] = value

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
    return merged


def _write_dataset_description(
    bids_root: Path,
    meta: DatasetMetadata,
    generator_label: str,
    report: MetadataReport,
    extra_fields: Optional[dict] = None,
) -> None:
    """Write/merge ``dataset_description.json`` (see :func:`describe_dataset`)."""
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
    merged = describe_dataset(existing, meta, generator_label, extra_fields)
    out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    report.files_written.append(out)


# ---------------------------------------------------------------------------
# participants.tsv + participants.json
# ---------------------------------------------------------------------------


_PARTICIPANT_COLUMNS: tuple[str, ...] = (
    "participant_id", "age", "sex", "handedness",
    "given_name", "family_name", "patient_id",
)


PARTICIPANT_DESCRIPTIONS: dict[str, dict[str, object]] = {
    "participant_id": {"Description": "Unique participant identifier"},
    "age": {"Description": "Age of the participant", "Units": "years"},
    "sex": {
        "Description": "Biological sex",
        "Levels": {"M": "male", "F": "female", "O": "other", "n/a": "not available"},
    },
    "handedness": {
        "Description": "Handedness of the participant",
        "Levels": {"R": "right", "L": "left", "A": "ambidextrous", "n/a": "not available"},
    },
    "given_name": {"Description": "Subject given name (kept for internal traceability)"},
    "family_name": {"Description": "Subject family name (kept for internal traceability)"},
    "patient_id": {"Description": "Original DICOM PatientID"},
}


def stated_dataset_fields(spec) -> dict:
    """What the dataset metadata form states about the dataset, as
    :class:`DatasetMetadata` keywords: only the fields it answers."""
    stored = getattr(spec, "dataset_description", None)
    if stored is None:
        return {}
    return {
        key: value
        for key, value in stored.model_dump().items()
        if value not in (None, "", [], {})
    }


def _dataset_meta_from_scaffold(
    inventory_tsv: Optional[Path], supplied: Optional[DatasetMetadata],
) -> Optional[DatasetMetadata]:
    """Merge the scaffold's dataset_description block under what the caller gave.

    Precedence is caller, then scaffold, then whatever is already on disk. An
    explicit ``--author`` on this run therefore wins over the stored answer,
    and the stored answer wins over nothing at all, which is what the automatic
    post-convert metadata run used to have.

    Only fields the scaffold actually states are merged, so a half-filled block
    contributes its half rather than blanking the rest.
    """
    if inventory_tsv is None:
        return supplied
    try:
        scaffold = scaffold_sidecar_path(inventory_tsv)
        if not scaffold.exists():
            return supplied
        stated = stated_dataset_fields(load_spec(scaffold))
    except Exception:
        return supplied
    if not stated:
        return supplied
    if supplied is None:
        return DatasetMetadata(**stated)

    # The caller's own answers stay on top of the stored ones.
    given = {
        field: value
        for field, value in supplied.model_dump().items()
        if value not in (None, "", [], {})
        and not (field == "name" and value == "Untitled BIDS Dataset")
    }
    return DatasetMetadata(**{**stated, **given})


def _apply_stated_metadata(bids_root: Path, inventory_tsv: Optional[Path]) -> int:
    """Write the metadata template's answers into the sidecars that take them."""
    if inventory_tsv is None:
        return 0
    from ..fixups.sidecar_schema import apply_stated_metadata

    try:
        scaffold = scaffold_sidecar_path(inventory_tsv)
        if not scaffold.exists():
            return 0
        spec = load_spec(scaffold)
        inventory = pd.read_csv(inventory_tsv, sep="\t", dtype=str).fillna("")
    except (OSError, ValueError):
        return 0
    return apply_stated_metadata(bids_root, spec, inventory)


def _dataset_extras_from_scaffold(inventory_tsv: Optional[Path]) -> dict:
    """The stated dataset_description fields DatasetMetadata has no attribute for."""
    if inventory_tsv is None:
        return {}
    try:
        scaffold = scaffold_sidecar_path(inventory_tsv)
        if not scaffold.exists():
            return {}
        return dict(load_spec(scaffold).dataset_description.extra or {})
    except Exception:
        return {}


def _phenotype_files_from_scaffold(inventory_tsv: Path) -> Optional[list[Path]]:
    """Read phenotype table paths from the inventory's recording-meta scaffold.

    The GUI dataset-metadata dialog stores phenotype file paths in
    ``<inventory>.tsv.recording_meta.json``. Paths are resolved relative to the
    inventory directory when not absolute. Returns ``None`` when there is no
    scaffold or it lists no phenotype files.
    """
    try:
        scaffold = scaffold_sidecar_path(inventory_tsv)
        if not scaffold.exists():
            return None
        spec = load_spec(scaffold)
    except Exception:
        return None
    if not spec.phenotype_files:
        return None
    base = Path(inventory_tsv).parent
    out: list[Path] = []
    for p in spec.phenotype_files:
        pp = Path(p)
        out.append(pp if pp.is_absolute() else base / pp)
    return out or None


def _participants_file_from_scaffold(inventory_tsv: Path) -> Optional[Path]:
    """Read the participants spreadsheet path from the recording-meta scaffold.

    The GUI dataset-metadata dialog stores it in
    ``<inventory>.tsv.recording_meta.json``. Resolved relative to the inventory
    directory when not absolute. Returns ``None`` when unset.
    """
    try:
        scaffold = scaffold_sidecar_path(inventory_tsv)
        if not scaffold.exists():
            return None
        spec = load_spec(scaffold)
    except Exception:
        return None
    raw = (spec.participants_file or "").strip()
    if not raw:
        return None
    pp = Path(raw)
    return pp if pp.is_absolute() else Path(inventory_tsv).parent / pp


def participant_rows(
    subjects: list[str], demo_lookup: dict[str, dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """``participants.tsv``'s columns and one row per subject, before any merge.

    The known demographics are normalised; any other column the participants
    spreadsheet carries is kept verbatim, in first-seen order. Pure, so the
    BIDS preview shows the table this step writes.
    """
    known = set(_PARTICIPANT_COLUMNS)
    extra_cols: list[str] = []
    for sid in subjects:
        for col in demo_lookup.get(sid, {}):
            if col and col not in known and col not in extra_cols:
                extra_cols.append(col)

    rows: list[dict[str, str]] = []
    for sid in subjects:
        row: dict[str, str] = {"participant_id": sid}
        extra = demo_lookup.get(sid, {})
        for col in ("age", "sex", "handedness", "given_name", "family_name", "patient_id"):
            v = extra.get(col, "")
            if col == "sex":
                v = normalize_sex(v)
            elif col == "handedness":
                v = normalize_handedness(v)
            elif col == "age":
                v = normalize_age(v)
            row[col] = v if v else "n/a"
        for col in extra_cols:
            v = str(extra.get(col, "") or "").strip()
            row[col] = v if v else "n/a"
        rows.append(row)
    return list(_PARTICIPANT_COLUMNS) + extra_cols, rows


def participants_json(columns: list[str], codebook: Optional[dict] = None) -> dict:
    """``participants.json`` for these columns: the known ones described by
    BIDS Manager, an extra one by the user's codebook, else by its name."""
    payload: dict = {}
    for col in columns:
        if col in PARTICIPANT_DESCRIPTIONS:
            payload[col] = PARTICIPANT_DESCRIPTIONS[col]
        elif col == "participant_id":
            continue
        else:
            entry = (codebook or {}).get(str(col))
            payload[col] = entry if isinstance(entry, dict) and entry \
                else {"Description": str(col)}
    return payload


def _write_participants(
    bids_root: Path,
    inventory_tsv: Optional[Path],
    report: MetadataReport,
    participants_file: Optional[Path] = None,
) -> None:
    subjects = sorted(p.name for p in bids_root.glob("sub-*") if p.is_dir())
    if not subjects:
        return

    demo_lookup = _load_demographics_from_inventory(inventory_tsv, report)
    participant_codebook: dict = {}
    if participants_file is not None:
        # A user-supplied participants spreadsheet is the authoritative
        # demographic source: its non-empty cells override the inventory.
        demo_lookup = merge_demographics(
            demo_lookup, load_participants_table(Path(participants_file)),
        )
        # Optional sibling JSON codebook (descriptions / levels / units for the
        # spreadsheet's columns) -> participants.json.
        participant_codebook = load_sidecar_dictionary(Path(participants_file))

    # Any column the participants spreadsheet carries beyond the known
    # demographics is preserved verbatim: it flows into participants.tsv and is
    # described in participants.json. Order is the first-seen order across
    # subjects for a stable, readable table.
    columns, rows = participant_rows(subjects, demo_lookup)
    df_new = pd.DataFrame(rows, columns=columns)

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
        else:
            # Repair an age left by an earlier run, or by another tool, in
            # DICOM's "065Y" form. BIDS wants a number, and the merge below
            # prefers what is already on disk, so without this a bad value
            # survives every rerun. "n/a" is a legitimate answer and stays.
            if df_existing is not None and "age" in df_existing.columns:
                df_existing["age"] = [
                    value if str(value).strip() in ("", "n/a")
                    else (normalize_age(value) or "n/a")
                    for value in df_existing["age"]
                ]

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
    json_payload = participants_json(list(df_out.columns), participant_codebook)
    out_json.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")
    report.files_written.append(out_json)


def _load_demographics_from_inventory(
    inventory_tsv: Optional[Path], report: MetadataReport,
) -> dict[str, dict[str, str]]:
    """Read the inventory TSV and group demographics by ``participant_id``.

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

    return demographics_by_participant(df)


def demographics_by_participant(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """The inventory's demographics, one entry per ``participant_id``, read
    from each participant's first row. Pure, for the preview as well."""
    if df.empty or "participant_id" not in df.columns:
        return {}

    lookup: dict[str, dict[str, str]] = {}
    for participant, sub_df in df.groupby("participant_id"):
        participant = str(participant).strip()
        if not participant:
            continue
        pid = participant if participant.startswith("sub-") else f"sub-{participant}"
        head = sub_df.iloc[0]
        lookup[pid] = {
            "given_name": str(head.get("GivenName", "") or ""),
            "family_name": str(head.get("FamilyName", "") or ""),
            "patient_id": str(head.get("PatientID", "") or ""),
            # DICOM writes an age as "065Y"; BIDS wants a number of years.
            "age": normalize_age(
                head.get("PatientAge", "") or head.get("age", "") or ""
            ),
            "sex": str(head.get("PatientSex", "") or head.get("sex", "") or ""),
            "handedness": str(head.get("Handedness", "") or head.get("handedness", "") or ""),
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

    Columns are added before rows, which is not cosmetic: a dataset with no
    demographics gets a ``participants.tsv`` holding nothing but
    ``participant_id``, and adding a row to a frame with no columns raises. In
    other words, doing it the other way round made the SECOND conversion into
    the plainest possible dataset fail outright.
    """
    if df_existing is None or "participant_id" not in df_existing.columns:
        return df_new, False

    merged = df_existing.set_index("participant_id")
    new_indexed = df_new.set_index("participant_id")

    for col in new_indexed.columns:
        if col not in merged.columns:
            merged[col] = "n/a"
    incoming_subjects = [s for s in new_indexed.index if s not in merged.index]
    if incoming_subjects:
        merged = pd.concat([
            merged,
            pd.DataFrame(
                "n/a", index=pd.Index(incoming_subjects, name=merged.index.name),
                columns=merged.columns,
            ),
        ])
    for col in new_indexed.columns:
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


def readme_text(name: str) -> str:
    """The README scaffold the metadata step writes (the BIDS preview shows
    the same text)."""
    return (
        f"# {name}\n\n"
        "This BIDS dataset was generated by bidsmgr.\n\n"
        "Edit this README to describe acquisition details, study design,\n"
        "and any deviations from the standard BIDS layout.\n"
    )


def _write_readme(bids_root: Path, name: str, report: MetadataReport) -> None:
    """Write a minimal README scaffold. Never overwrites a hand-edited file."""
    out = bids_root / "README"
    if out.exists():
        return
    out.write_text(readme_text(name), encoding="utf-8")
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


def _refresh_scans_tsv(bids_root: Path, report: MetadataReport,
                       inventory_tsv: Optional[Path] = None) -> None:
    """The ``*_scans.tsv`` and ``*_sessions.tsv`` tables (``metadata.tables``).

    Existing tables are MERGED, never replaced: mne-bids writes the scans
    table during conversion with a real ``acq_time`` read from the
    recording, and a user's rows and columns are theirs. What the scan
    inventory knows about times fills a recording whose sidecar has none.
    """
    from .tables import plan_tables, times_from_inventory, write_tables

    times = {}
    if inventory_tsv is not None and Path(inventory_tsv).is_file():
        try:
            times = times_from_inventory(pd.read_csv(
                inventory_tsv, sep="\t", dtype=str, keep_default_na=False))
        except (OSError, ValueError) as exc:
            log.warning("could not read times from %s: %s", inventory_tsv, exc)
    report.files_written.extend(write_tables(plan_tables(bids_root, times=times)))


# ---------------------------------------------------------------------------
# Sidecar fill + audit
# ---------------------------------------------------------------------------


def _is_todo_marker(value: object) -> bool:
    """True for a marker this engine writes and may therefore take back.

    ``"n/a"`` is deliberately not one: a converter writes it as a real answer
    (MNE-BIDS for an unset line frequency) and so may a person, and nothing in
    the file says which wrote it.
    """
    return value == _TODO_VALUE or value == [_TODO_VALUE]


def _take_back(
    data: dict, specs: dict, levels: frozenset[str],
    *, held_elsewhere: frozenset[str] = frozenset(),
) -> list[str]:
    """Remove every TODO marker this run would not have written.

    ``specs`` maps the fields the standard declares for the file to their
    spec. A marker stays only where its field's level is in ``levels`` AND it
    is the marker the field takes now: an earlier run that marked a wider
    scope, or wrote ``"TODO"`` into a field that names a file, is undone.
    ``held_elsewhere`` fields go regardless. An undeclared key is left alone:
    this engine never writes one, so a TODO there is somebody's own note.
    """
    removed: list[str] = []
    for name in [k for k, v in data.items() if _is_todo_marker(v)]:
        spec = specs.get(name)
        if name not in held_elsewhere:
            if spec is None:
                continue
            if (schema_mod.effective_level(spec) in levels
                    and todo_value_for(spec) == data[name]):
                continue
        del data[name]
        removed.append(name)
    return removed


@dataclass
class SidecarMarks:
    """What the metadata step did to one sidecar's fields (see :func:`mark_sidecar`)."""

    #: Fields derived from the filename (``TaskName``), name -> value.
    derived: dict[str, object] = field(default_factory=dict)
    #: Placeholders written, name -> value.
    todo_added: dict[str, object] = field(default_factory=dict)
    #: Placeholders taken back (outside the scope, or a marker the field no
    #: longer takes).
    todo_removed: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    missing_recommended: list[str] = field(default_factory=list)
    #: ``(field, why it takes no placeholder)``.
    unmarkable: list[tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.derived or self.todo_added or self.todo_removed)


def mark_sidecar(
    data: dict,
    name: str,
    datatype: str,
    suffix: str,
    levels: Optional[frozenset[str]],
    *,
    bids_root: Optional[Path] = None,
) -> SidecarMarks:
    """What the metadata step does to one sidecar, on its fields in memory.

    ``data`` is changed in place: ``TaskName`` from the filename where the
    file is a run that needs one, the placeholders ``levels`` asks for, and
    the placeholders outside them taken back (``None`` leaves placeholders
    alone). ``name`` is the file's name, whose entities settle which fields
    apply (a T1w with no task takes no TaskName); ``bids_root`` settles the
    rules that depend on the dataset.

    One definition for the step that writes files and for the BIDS preview,
    which runs it on the sidecar it predicts, so the preview cannot show a
    placeholder the step would not write.
    """
    marks = SidecarMarks()

    # Filename-derivable fills (TaskName).
    if datatype == "func" and suffix in {"bold", "sbref", "cbv", "phase"}:
        m = _TASKNAME_RE.search(name)
        if m and "TaskName" not in data:
            marks.derived["TaskName"] = m.group(1)

    try:
        declared_specs = schema_mod.sidecar_fields(
            datatype, suffix, bids_root, entities=schema_mod.name_entities(name),
        )
    except (KeyError, ValueError, AttributeError):
        declared_specs = []
    by_level: dict[str, list] = {}
    for spec in declared_specs:
        by_level.setdefault(schema_mod.effective_level(spec), []).append(spec)

    # Take back what an earlier, wider scope marked, before the audit and the
    # fill, so a narrower scope is what the file ends up carrying.
    if levels is not None and any(_is_todo_marker(v) for v in data.values()):
        marks.todo_removed = _take_back(data, {
            canonical_field_name(spec.name): spec for spec in declared_specs
        }, levels)

    known = set(data) | set(marks.derived)

    # Required field audit. A speculative requirement (one whose rule may not
    # describe this file) is already counted as optional by effective_level.
    # Merely CONDITIONAL requirements stay: RepetitionTime is excused only
    # when VolumeTiming is present, which _REQUIRED_ALTERNATIVES handles, so a
    # bold run missing both is still flagged.
    required = by_level.get("required", [])
    required_names = {canonical_field_name(f.name) for f in required}
    for alternatives in _REQUIRED_ALTERNATIVES.get((datatype, suffix), ()):
        if any(alt in known for alt in alternatives):
            required_names -= set(alternatives)
    marks.missing_required = sorted(n for n in required_names if n not in known)

    # Recommended field audit.
    recommended = by_level.get("recommended", [])
    seen_rec: set[str] = set()
    for field_info in recommended:
        rec_name = canonical_field_name(field_info.name)
        if rec_name not in seen_rec:
            seen_rec.add(rec_name)
            if rec_name not in known:
                marks.missing_recommended.append(rec_name)

    # The placeholder fill, as far as the caller asked for.
    if levels:
        optional = by_level.get("optional", []) if "optional" in levels else []
        by_name: dict[str, object] = {}
        for spec in list(required) + list(recommended) + list(optional):
            by_name.setdefault(canonical_field_name(spec.name), spec)
        wanted: list[str] = []
        if "required" in levels:
            wanted += marks.missing_required
        if "recommended" in levels:
            wanted += marks.missing_recommended
        if "optional" in levels:
            wanted += [canonical_field_name(f.name) for f in optional
                       if canonical_field_name(f.name) not in known]
        for field_name in wanted:
            if field_name in known or field_name in marks.todo_added:
                continue
            spec = by_name.get(field_name)
            if spec is None:
                continue
            value = todo_value_for(spec)
            if value is NO_TODO:
                # Not a silent skip: say which field and why, so a fill that
                # covers two thirds does not report itself complete.
                marks.unmarkable.append((field_name, unmarkable_reason(spec)))
                continue
            marks.todo_added[field_name] = value

    data.update(marks.derived)
    data.update(marks.todo_added)
    return marks


def _fill_and_audit_sidecars(
    bids_root: Path,
    report: MetadataReport,
    *,
    levels: Optional[frozenset[str]] = None,
) -> None:
    """Walk every sidecar JSON, run :func:`mark_sidecar` on it, record and write.

    Fills are non-destructive: existing keys are never overwritten. The
    audit messages still report what *was* missing (the report shows what
    got TODO'd in ``report.todo_fills``).
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

        marks = mark_sidecar(data, json_path.name, datatype, suffix, levels,
                             bids_root=bids_root)
        rel = json_path.relative_to(bids_root)
        for name in marks.missing_required:
            report.missing_required.append(f"{rel}: missing {name!r}")
        for name in marks.missing_recommended:
            report.missing_recommended.append(f"{rel}: missing recommended {name!r}")
        for name, why in marks.unmarkable:
            report.unmarkable.append(f"{rel}: {name!r} takes no placeholder ({why})")
        if not marks.changed:
            continue
        try:
            json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            report.warnings.append(f"could not write {json_path}: {exc}")
            continue
        # Filename-derived fills (TaskName) are recorded apart from TODOs.
        if marks.derived:
            report.sidecar_fills.append(SidecarFill(sidecar=json_path, fields=marks.derived))
        if marks.todo_added:
            report.todo_fills.append(
                TodoFill(sidecar=json_path, fields=sorted(marks.todo_added)))
        if marks.todo_removed:
            report.todo_removed.append(
                TodoFill(sidecar=json_path, fields=sorted(marks.todo_removed)))


def mark_dataset_description(
    data: dict,
    levels: Optional[frozenset[str]],
    *,
    bids_root: Optional[Path] = None,
    citation: bool = False,
) -> SidecarMarks:
    """What the metadata step does to ``dataset_description.json``'s fields.

    Every level comes from the schema, asked with the dataset in hand, because
    two of them depend on it: ``Authors`` is recommended only while there is
    no ``CITATION.cff``. A hand-kept list used to add six optional fields
    (Funding, EthicsApprovals, ...) to the recommended scope, which is how a
    "required and recommended" run left placeholders the Editor shows under
    Optional. ``levels`` as in :func:`mark_sidecar`; ``citation`` says the
    dataset has a ``CITATION.cff``. ``data`` is changed in place.
    """
    marks = SidecarMarks()
    try:
        specs = {f.name: f for f in schema_mod.dataset_description_fields(bids_root)}
    except Exception:  # noqa: BLE001 - the audit must still run
        specs = {}
    marks.missing_recommended = [
        name for name, spec in specs.items()
        if name not in data and schema_mod.effective_level(spec) == "recommended"
    ]
    if levels is None:
        return marks

    # With a citation file these belong in it, and the schema's own checks
    # (SingleSourceAuthors, SingleSourceCitationFields) report them here: a
    # placeholder would add the very finding the citation file removed.
    held_elsewhere = _CITATION_FIELDS if citation else frozenset()
    marks.todo_removed = _take_back(data, specs, levels, held_elsewhere=held_elsewhere)
    for name, spec in sorted(specs.items()):
        if (name in data or name in held_elsewhere
                or schema_mod.effective_level(spec) not in levels):
            continue
        value = todo_value_for(spec)
        if value is NO_TODO:
            marks.unmarkable.append((name, unmarkable_reason(spec)))
            continue
        marks.todo_added[name] = value
    data.update(marks.todo_added)
    return marks


def _audit_dataset_description(
    bids_root: Path,
    report: MetadataReport,
    *,
    levels: Optional[frozenset[str]] = None,
) -> None:
    """Run :func:`mark_dataset_description` on the dataset's own, record, write."""
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
    marks = mark_dataset_description(
        data, levels, bids_root=bids_root,
        citation=(bids_root / "CITATION.cff").exists(),
    )
    for name in marks.missing_recommended:
        report.missing_recommended.append(f"{rel}: missing recommended {name!r}")
    for name, why in marks.unmarkable:
        report.unmarkable.append(f"{rel}: {name!r} takes no placeholder ({why})")
    if not marks.changed:
        return
    try:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        report.warnings.append(f"could not write {path}: {exc}")
        return
    if marks.todo_added:
        report.todo_fills.append(TodoFill(sidecar=path, fields=sorted(marks.todo_added)))
    if marks.todo_removed:
        report.todo_removed.append(TodoFill(sidecar=path, fields=sorted(marks.todo_removed)))


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
        if part in _datatype_names():
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


__all__ = ["run_metadata"]
