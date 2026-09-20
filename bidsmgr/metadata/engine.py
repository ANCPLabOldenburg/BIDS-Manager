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

# The literal placeholder value written by ``--fill-todos`` for every
# missing required / recommended field. Deliberately just the string
# ``"TODO"`` (not ``TODO_<fieldname>``) — the user wants a clean marker
# they can grep for, fill, and remove.
_TODO_VALUE = "TODO"

# What BIDS itself writes when a field was asked and has no answer. Valid
# wherever the schema declares it, unlike the TODO marker.
_NA_VALUE = "n/a"

# Sentinel: this field gets no placeholder at all.
_NO_TODO = object()

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


def _todo_value_for(field) -> object:
    """The placeholder to write for a field, or ``_NO_TODO`` for none.

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
    declared = getattr(field, "type", "")

    if getattr(field, "accepts_na", False) or "n/a" in enum:
        return _NA_VALUE
    if enum:
        # A real vocabulary. Nothing in it means "unanswered".
        return _NO_TODO
    if getattr(field, "accepts_free_text", False) or declared == "string":
        return _TODO_VALUE
    if "array" in accepts and item_type == "string":
        return [_TODO_VALUE]
    # An array whose item type the schema does not state is NOT assumed to
    # hold strings. That assumption is what wrote ``["TODO"]`` into EchoTime,
    # a number-or-array-of-numbers field, turning a gap into a type error.
    return _NO_TODO


def unmarkable_reason(field) -> str:
    """Why this field can take no placeholder, for the report. Empty if it can.

    Said out loud because the alternative is a fill that quietly covers two
    thirds of what is missing and reports itself as complete.
    """
    if _todo_value_for(field) is not _NO_TODO:
        return ""
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
    fill_scope: Optional[str] = None,
    write_report: bool = True,
    generator_label: str = "bidsmgr",
    participants_file: Optional[Path] = None,
    phenotype_files: Optional[list[Path]] = None,
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
        Turn the placeholder fill on. What it covers is ``fill_scope``.
        Existing values are NEVER overwritten, at any scope. The fill is
        recorded in ``report.todo_fills`` and in the JSON report, and every
        field that could take no placeholder is recorded in
        ``report.unmarkable`` with the reason.
    fill_scope
        How much of what the standard declares to mark:
        ``"required"``, ``"recommended"`` (the default, and what
        ``fill_todos=True`` meant before this existed) or ``"optional"``.
        The scopes nest. ``"none"`` is the same as ``fill_todos=False``.
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
    _refresh_scans_tsv(bids_root, report)
    # One resolved scope for the whole run, so the two passes below cannot
    # disagree about what was asked for.
    scope = fill_scope or (FILL_RECOMMENDED if fill_todos else FILL_NONE)
    if scope not in FILL_SCOPES:
        raise ValueError(
            f"invalid fill_scope {scope!r}; expected one of {FILL_SCOPES}"
        )
    if not fill_todos and fill_scope is None:
        scope = FILL_NONE
    levels = scope_levels(scope)
    report.fill_scope = scope

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


def _write_dataset_description(
    bids_root: Path,
    meta: DatasetMetadata,
    generator_label: str,
    report: MetadataReport,
    extra_fields: Optional[dict] = None,
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

    out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    report.files_written.append(out)


# ---------------------------------------------------------------------------
# participants.tsv + participants.json
# ---------------------------------------------------------------------------


_PARTICIPANT_COLUMNS: tuple[str, ...] = (
    "participant_id", "age", "sex", "handedness",
    "given_name", "family_name", "patient_id",
)


_PARTICIPANT_DESCRIPTIONS: dict[str, dict[str, object]] = {
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
        stored = load_spec(scaffold).dataset_description
    except Exception:
        return supplied

    stated = {
        field: value
        for field, value in stored.model_dump().items()
        if value not in (None, "", [], {})
    }
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

    df_new = pd.DataFrame(rows, columns=list(_PARTICIPANT_COLUMNS) + extra_cols)

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
    json_payload: dict = {}
    for col in df_out.columns:
        if col in _PARTICIPANT_DESCRIPTIONS:
            json_payload[col] = _PARTICIPANT_DESCRIPTIONS[col]
        elif col == "participant_id":
            continue
        else:
            # Extra spreadsheet column: use the user's codebook entry if any,
            # else a minimal Description = the column name.
            entry = participant_codebook.get(str(col))
            json_payload[col] = (
                entry if isinstance(entry, dict) and entry
                else {"Description": str(col)}
            )
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


# Files that live in a datatype folder but are NOT the recording. BIDS is
# explicit that the scans file describes "each neural recording file", so the
# tables and sidecars that accompany one do not get a row.
_SCANS_COMPANION_SUFFIXES: frozenset[str] = frozenset({
    "events", "channels", "electrodes", "coordsystem", "headshape", "markers",
    "photo", "physio", "stim", "blood", "scans", "sessions", "participants",
})

# Extensions that are part of a recording rather than a recording themselves.
# A BrainVision recording is a .vhdr header plus a .eeg binary and a .vmrk
# marker file, and an EEGLAB one is a .set plus a .fdt: the header is the
# recording and gets the row, the rest would be duplicates of it. ``.eeg``
# here is always the BrainVision binary, because the other format that uses
# the extension (Nihon Kohden) is not BIDS-native and cannot appear in a
# converted dataset. ``.dat``, ``.mrk`` and ``.pos`` are the MEG calibration,
# marker and headshape side files.
_SCANS_COMPANION_EXTS: frozenset[str] = frozenset({
    ".json", ".tsv", ".gz", ".vmrk", ".eeg", ".fdt", ".dat", ".mrk", ".pos",
})

# MEG system files mne-bids writes beside the data, named by entity rather
# than by suffix: acq-calibration_meg.dat and acq-crosstalk_meg.fif. The
# second carries a .fif extension, which is a real recording extension, so
# only the entity tells them apart.
_SCANS_SYSTEM_ENTITIES: tuple[str, ...] = ("acq-calibration", "acq-crosstalk")

# Recordings that are a DIRECTORY: CTF and EGI. One row each, and the walk
# must not descend into them or every internal file would get a row.
_SCANS_DIR_EXTS: frozenset[str] = frozenset({".ds", ".mff"})


def _is_recording_file(path: Path) -> bool:
    """Does this path get a row in ``*_scans.tsv``?

    True for the data files themselves, of every modality, and false for the
    sidecars, tables and format side files that accompany them.
    """
    name = path.name
    if name.endswith(".tsv.gz") or name.endswith(".nii.gz"):
        ext = ".tsv.gz" if name.endswith(".tsv.gz") else ".nii.gz"
    else:
        ext = path.suffix.lower()
    if ext == ".nii.gz":
        return True
    if ext in _SCANS_COMPANION_EXTS or ext == ".tsv.gz":
        return False
    if any(token in name for token in _SCANS_SYSTEM_ENTITIES):
        return False
    # The BIDS suffix is the last underscore-delimited token that carries no
    # hyphen; entity tokens all do.
    stem = name.split(".")[0]
    for token in reversed(stem.split("_")):
        if "-" not in token:
            return token not in _SCANS_COMPANION_SUFFIXES
    return False


def _scans_rows_for(root: Path) -> list[dict[str, str]]:
    """One row per recording under ``root``, in path order.

    Walks by hand rather than with ``rglob`` so a folder-shaped recording
    (CTF ``.ds``, EGI ``.mff``) is taken as a single entry and not descended
    into.
    """
    rows: list[dict[str, str]] = []

    def walk(folder: Path) -> None:
        for entry in sorted(folder.iterdir(), key=lambda p: p.name):
            if entry.is_dir():
                if entry.suffix.lower() in _SCANS_DIR_EXTS:
                    rows.append({
                        "filename": entry.relative_to(root).as_posix(),
                        "acq_time": _acq_time_for(entry),
                    })
                else:
                    walk(entry)
            elif entry.is_file() and _is_recording_file(entry):
                rows.append({
                    "filename": entry.relative_to(root).as_posix(),
                    "acq_time": _acq_time_for(entry),
                })

    walk(root)
    return rows


def _acq_time_for(recording: Path) -> str:
    """``acq_time`` from the recording's own sidecar, or ``"n/a"``.

    Only the MRI sidecars carry an acquisition time we can read this way.
    EEG and MEG get theirs from mne-bids, which reads ``meas_date`` off the
    recording at conversion and writes it into the scans table; that value is
    preserved by the merge rather than recomputed here, because the sidecar
    does not hold it.
    """
    json_path = _matching_json(recording)
    if not json_path.exists():
        return _NA_VALUE
    try:
        meta = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return _NA_VALUE
    adt = meta.get("AcquisitionDateTime") or meta.get("AcquisitionTime")
    return adt if isinstance(adt, str) and adt else _NA_VALUE


def _refresh_scans_tsv(bids_root: Path, report: MetadataReport) -> None:
    """Generate one ``*_scans.tsv`` per subject (or subject/session).

    Lists EVERY recording, of every modality, not just the NIfTIs. The
    previous version globbed ``*.nii*``, which had two consequences on any
    dataset holding EEG, MEG, iEEG or ECAT PET. A subject with no NIfTI at all
    produced no scans table, and a mixed session produced one that listed the
    images and silently omitted the recordings beside them.

    Worse, mne-bids WRITES this table during conversion, with a real
    ``acq_time`` taken from the recording's ``meas_date``. Regenerating it
    from the NIfTIs deleted those rows and the only acquisition times the
    dataset had for its recordings. So the fresh rows are MERGED into what is
    already there rather than replacing it: existing rows keep their values
    and their extra columns (mne-bids adds ``source`` when asked), rows for
    recordings that are new get appended, and nothing that was measured at
    conversion time is thrown away.
    """
    from .preserve import merge_scans_table

    for subj_dir in sorted(bids_root.glob("sub-*")):
        if not subj_dir.is_dir():
            continue
        ses_dirs = [s for s in sorted(subj_dir.glob("ses-*")) if s.is_dir()]
        roots = ses_dirs or [subj_dir]
        for root in roots:
            rows = _scans_rows_for(root)
            if not rows:
                continue
            ses_part = f"_{root.name}" if root.name.startswith("ses-") else ""
            out = root / f"{subj_dir.name}{ses_part}_scans.tsv"

            existing: list[dict[str, str]] = []
            if out.is_file():
                try:
                    existing = pd.read_csv(
                        out, sep="\t", dtype=str, keep_default_na=False,
                    ).to_dict("records")
                except (OSError, ValueError) as exc:
                    log.warning("could not read %s: %s", out, exc)
            # Rows whose file is gone are dropped: a stale filename is an
            # error the validator reports (SCANS_FILENAME_NOT_MATCH_DATASET),
            # and after a rename the old name is exactly that.
            present = {row["filename"] for row in rows}
            existing = [r for r in existing if r.get("filename", "") in present]

            fields, merged = merge_scans_table(existing, rows)
            pd.DataFrame(merged, columns=fields).to_csv(
                out, sep="\t", index=False,
            )
            report.files_written.append(out)


# ---------------------------------------------------------------------------
# Sidecar fill + audit
# ---------------------------------------------------------------------------


def _fill_and_audit_sidecars(
    bids_root: Path,
    report: MetadataReport,
    *,
    levels: frozenset[str] = frozenset(),
) -> None:
    """Walk every sidecar JSON, fill derivable fields, audit required ones.

    Fills are non-destructive: existing keys are never overwritten.
    ``TaskName`` (from the ``_task-<label>`` token in the filename) is
    filled when missing. The audit then checks required + recommended
    fields per the schema engine and records anything still missing.

    ``levels`` says which requirement levels get a placeholder written into
    the file for fields that are still missing. Empty means none. The
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
        # Speculative requirements are excluded. The schema requires
        # SkullStripped only of a derivative and Units only of a phase image;
        # neither is knowable from the datatype and suffix, and demanding them
        # of an ordinary raw scan reports a violation that is not one. Merely
        # CONDITIONAL requirements stay: RepetitionTime is excused only when
        # VolumeTiming is present, which _REQUIRED_ALTERNATIVES below handles,
        # so a bold run missing both is still flagged.
        required_names = {
            _canonical_field_name(f.name) for f in required if not f.speculative
        }
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

        # Apply the placeholder fill, as far as the caller asked for.
        todo_added: list[str] = []
        if levels:
            # Optional is only looked up when it is actually wanted: it is the
            # biggest group by far and reading it costs on every sidecar.
            optional = []
            if "optional" in levels:
                try:
                    optional = schema_mod.optional_sidecar_fields(
                        datatype, suffix,
                    )
                except (KeyError, ValueError, AttributeError):
                    optional = []
            by_name: dict[str, object] = {}
            for spec in list(required) + list(recommended) + list(optional):
                by_name.setdefault(_canonical_field_name(spec.name), spec)

            wanted: list[str] = []
            if "required" in levels:
                wanted += missing_req
            if "recommended" in levels:
                wanted += missing_rec
            if "optional" in levels:
                wanted += [
                    _canonical_field_name(f.name) for f in optional
                    if _canonical_field_name(f.name) not in data
                ]

            for name in wanted:
                if name in data or name in fills:
                    continue
                spec = by_name.get(name)
                if spec is None:
                    continue
                value = _todo_value_for(spec)
                if value is _NO_TODO:
                    # Not a silent skip: say which field and why, so a fill
                    # that covers two thirds does not report itself complete.
                    report.unmarkable.append(
                        f"{rel}: {name!r} takes no placeholder "
                        f"({unmarkable_reason(spec)})"
                    )
                    continue
                fills[name] = value
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
            # Keyed on what we actually recorded as a TODO, not on the value:
            # an array placeholder is ``["TODO"]``, which no scalar comparison
            # would catch, and it would then be reported as a real fill.
            non_todo = {k: v for k, v in fills.items() if k not in set(todo_added)}
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
    levels: frozenset[str] = frozenset(),
) -> None:
    """Audit ``dataset_description.json`` recommended fields.

    The schema engine doesn't expose dataset-level recommended fields
    (its API is keyed by ``(datatype, suffix)``), so we use the small
    BIDS 1.10 list in ``_DATASET_DESCRIPTION_RECOMMENDED``. With
    ``levels`` says which requirement levels get a placeholder written for
    fields that are still missing. Empty means none.
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
    # What the STANDARD declares for this file, at the levels asked for,
    # rather than the hand-kept recommended list this used to consult. The
    # list was right about the recommended ones and silent about everything
    # else, so an "optional" scope could never have reached them.
    try:
        specs = {
            f.name: f for f in schema_mod.dataset_description_fields(bids_root)
        }
    except Exception:  # noqa: BLE001 - the audit must still run
        specs = {}

    missing: list[str] = [
        f for f in _DATASET_DESCRIPTION_RECOMMENDED if f not in data
    ]
    for name in missing:
        report.missing_recommended.append(f"{rel}: missing recommended {name!r}")

    if not levels:
        return

    wanted = [
        name for name, spec in specs.items()
        if name not in data and getattr(spec, "level", "") in levels
    ]
    # The hand-kept recommended names stay in play even if the schema places
    # them elsewhere, so nothing this file used to fill stops being filled.
    if "recommended" in levels:
        wanted += [n for n in missing if n not in wanted]

    filled: list[str] = []
    for name in sorted(set(wanted)):
        spec = specs.get(name)
        if spec is None:
            continue
        value = _todo_value_for(spec)
        if value is _NO_TODO:
            report.unmarkable.append(
                f"{rel}: {name!r} takes no placeholder "
                f"({unmarkable_reason(spec)})"
            )
            continue
        data[name] = value
        filled.append(name)
    if not filled:
        return
    try:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        report.todo_fills.append(TodoFill(sidecar=path, fields=sorted(filled)))
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
