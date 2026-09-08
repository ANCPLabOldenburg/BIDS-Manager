"""BIDS Manager-native validation supplements (Qt-free).

Validation findings come from the standalone :mod:`bidsval` package. These are
the two things bidsval does not produce that BIDS Manager's Editor still needs:

* :func:`sidecar_fields_for` - the schema-aware *form* data the sidecar editor
  renders (one row per schema-defined field at a ``(datatype, suffix)`` plus any
  extra keys the file carries). This is form scaffolding, not a finding; it
  drives the Editor's sidecar form and the HTML report's schema-audit table.
* :func:`todo_issues_for` - TODO-placeholder findings. A BIDS Manager
  convention: the metadata engine writes the literal string ``"TODO"`` into
  missing recommended fields, and the Editor flags those so the user fills them
  in. bidsval (correctly) does not know about this convention.

Both are derived straight from :mod:`bidsmgr.schema`, the same keystone the rest
of the package reads from. No Qt here (architectural rule 2).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from bidsval import schema as bidsval_schema

from .. import schema as schema_mod
from .types import FieldLevel, Issue, Severity, SidecarField

# Recognised BIDS datatype directory names, used to infer a file's datatype from
# its path the same way the old in-house validator did.
_BIDS_DATATYPE_NAMES: tuple[str, ...] = (
    "anat", "func", "dwi", "fmap", "perf",
    "meg", "eeg", "ieeg", "beh", "pet", "micr", "nirs",
)

# Multi-part extensions stripped (longest first) to recover the BIDS suffix.
_KNOWN_EXTS: tuple[str, ...] = (
    ".nii.gz", ".tsv.gz", ".nii", ".json", ".tsv", ".bval", ".bvec",
    ".edf", ".bdf", ".set", ".vhdr", ".fif", ".fif.gz",
)


# Data-file extensions whose metadata lives in a sibling ``.json`` sidecar.
# Longest-first so multi-part extensions strip correctly. Used to find the
# editable sidecar for a finding attached to a data file.
_DATA_EXTS: tuple[str, ...] = (
    ".nii.gz", ".nii", ".fif.gz", ".fif", ".edf", ".bdf", ".gdf", ".set",
    ".vhdr", ".cnt", ".con", ".sqd", ".kdf", ".ds", ".mff", ".mef", ".nwb",
)


def _canonical(name: str) -> str:
    """``EchoTime__fmap`` -> ``EchoTime`` (strip the schema's context suffix)."""
    return name.split("__", 1)[0]


def sibling_json(path: Path) -> Optional[Path]:
    """The ``.json`` sidecar path for a data file (``*_bold.nii.gz`` ->
    ``*_bold.json``), or ``None`` if ``path`` is not a recognised data file.

    Purely a name transform - the caller checks whether the sidecar exists.
    """
    name = path.name
    lower = name.lower()
    for ext in _DATA_EXTS:
        if lower.endswith(ext):
            return path.with_name(name[: -len(ext)] + ".json")
    return None


def value_kind(value: object) -> str:
    """Classify a parsed JSON value for the sidecar form's value column."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value == "TODO":
        return "todo"
    return "string"


def infer_datatype_suffix(
    path: Path, bids_root: Path,
) -> tuple[Optional[str], Optional[str]]:
    """Best-effort ``(datatype, suffix)`` for a file from its path + name.

    Datatype is the first recognised datatype directory on the path; suffix is
    the last underscore-separated token of the basename with no ``-`` (so it is
    not an entity). Mirrors the old in-house validator so downstream consumers
    (HTML report typed annotation, sidecar form) see the same values.
    """
    try:
        rel_parts = path.relative_to(bids_root).parts
    except ValueError:
        rel_parts = path.parts

    datatype: Optional[str] = None
    for part in rel_parts:
        if part in _BIDS_DATATYPE_NAMES:
            datatype = part
            break

    stem = path.name
    for ext in _KNOWN_EXTS:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    suffix: Optional[str] = None
    for tok in reversed(stem.split("_")):
        if tok and "-" not in tok:
            suffix = tok
            break
    return datatype, suffix


@lru_cache(maxsize=1)
def _sole_datatype_by_suffix() -> dict[str, str]:
    """Suffixes only one datatype can carry, mapped to that datatype.

    99 of the schema's 111 suffixes are unambiguous: only ``func`` has ``bold``,
    only ``anat`` has ``T1w``. The other 12 are the cross-modality companions
    (``events``, ``channels``, ``physio``), which several datatypes share and
    which therefore cannot be settled by name.
    """
    owners: dict[str, set[str]] = {}
    try:
        for datatype in schema_mod.list_datatypes():
            for suffix in schema_mod.list_suffixes(datatype):
                owners.setdefault(suffix, set()).add(datatype)
    except Exception:
        return {}
    return {s: next(iter(d)) for s, d in owners.items() if len(d) == 1}


def _resolve_datatype(datatype: Optional[str], suffix: Optional[str]) -> Optional[str]:
    """Fill in a datatype the path could not supply.

    An INHERITED sidecar sits above the datatype directories: a top-level
    ``task-rest_bold.json`` applies to every func/bold run in the dataset, but
    nothing in its path says ``func``, so it was typed as nothing and every one
    of its fields fell back to optional. The suffix settles it whenever only one
    datatype claims that suffix.
    """
    if datatype or not suffix:
        return datatype
    return _sole_datatype_by_suffix().get(suffix)


def _shown_level(level: FieldLevel, spec) -> FieldLevel:
    """Do not badge a speculative requirement as one.

    A field is speculative when the rule demanding it may not describe this
    file at all. ``SkullStripped`` is required of DERIVATIVES and ``Genetics``
    only of a dataset that ships a ``genetic_info.json``; neither is knowable
    from the file in front of us. Listing them is right, the form should show
    everything the standard allows, but painting them RED as required is a
    demand the standard is not making, and it sends the user hunting for a
    field they must not add.
    """
    if getattr(spec, "speculative", False) and level in (
        FieldLevel.REQUIRED,
        FieldLevel.RECOMMENDED,
    ):
        return FieldLevel.OPTIONAL
    return level


def _load_json_object(fp: Path) -> Optional[dict]:
    """Parse ``fp`` as JSON, returning the object or ``None`` (unreadable / not
    an object). Errors are bidsval's job to report; here we just need the data."""
    try:
        data = json.loads(Path(fp).read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def sidecar_fields_for(
    fp: Path,
    datatype: Optional[str],
    suffix: Optional[str],
    bids_root: Optional[Path] = None,
) -> list[SidecarField]:
    """Build the schema-aware sidecar form rows for a JSON file.

    Returns one :class:`SidecarField` per schema-defined field at
    ``(datatype, suffix)`` (required + recommended + optional + deprecated),
    each flagged present/missing against the file's actual contents, plus any
    extra keys the file carries (as OPTIONAL). Returns ``[]`` for non-JSON
    files or untyped sidecars (the form falls back to a plain on-disk read).
    """
    if not str(fp).lower().endswith(".json"):
        return []
    datatype = _resolve_datatype(datatype, suffix)
    if not (datatype and suffix):
        # A top-level dataset file has no datatype and no suffix, so it cannot
        # be addressed the same way, but it does have declared fields and they
        # were previously invented from finding severity. Ask the schema.
        if Path(fp).name == "dataset_description.json":
            return _dataset_description_fields_for(fp, bids_root)
        return []
    data = _load_json_object(fp)
    if data is None:
        return []

    seen: set[str] = set()
    fields: list[SidecarField] = []

    def _add_level(level: FieldLevel, getter) -> None:
        try:
            schema_fields = getter(datatype, suffix, bids_root)
        except (KeyError, ValueError, AttributeError):
            schema_fields = []
        for fi in schema_fields or []:
            name = _canonical(fi.name)
            if name in seen:
                continue
            seen.add(name)
            present = name in data
            value = data.get(name) if present else None
            fields.append(SidecarField(
                level=_shown_level(level, fi), name=name, value=value, present=present,
                value_kind=value_kind(value) if present else "missing",
                description=getattr(fi, "description", None),
            ))

    _add_level(FieldLevel.REQUIRED, schema_mod.required_sidecar_fields)
    _add_level(FieldLevel.RECOMMENDED, schema_mod.recommended_sidecar_fields)
    _add_level(FieldLevel.OPTIONAL, schema_mod.optional_sidecar_fields)
    _add_level(FieldLevel.DEPRECATED, schema_mod.deprecated_sidecar_fields)

    # Keys present in the file but absent from every schema list -> OPTIONAL.
    for k, v in data.items():
        if k in seen:
            continue
        seen.add(k)
        fields.append(SidecarField(
            level=FieldLevel.OPTIONAL, name=k, value=v, present=True,
            value_kind=value_kind(v),
        ))
    return fields


_BIDSVAL_LEVEL_TO_FIELD_LEVEL = {
    "required": FieldLevel.REQUIRED,
    "recommended": FieldLevel.RECOMMENDED,
    "optional": FieldLevel.OPTIONAL,
    "deprecated": FieldLevel.DEPRECATED,
    # BIDS Manager's form has no "prohibited" bar; such a field is best shown as
    # an ordinary row the user can see and delete.
    "prohibited": FieldLevel.OPTIONAL,
}


def schema_level_for(
    field: str,
    datatype: Optional[str],
    suffix: Optional[str],
    *,
    dataset_description: bool = False,
    bids_root: Optional[Path] = None,
) -> Optional[FieldLevel]:
    """The level the standard declares for ``field``, if it says anything.

    Returns ``None`` when the schema has nothing to say. A caller must then
    treat the field as OPTIONAL rather than inferring something stronger from
    how loudly the validator complained: severity does not carry the level back,
    since optional and prohibited are both silent.
    """
    datatype = _resolve_datatype(datatype, suffix)
    try:
        if dataset_description:
            specs = bidsval_schema.dataset_description_fields(dataset_root=bids_root)
        elif datatype and suffix:
            specs = bidsval_schema.sidecar_fields(
                datatype, suffix, dataset_root=bids_root,
            )
        else:
            return None
    except Exception:
        return None
    canonical = _canonical(field)
    for spec in specs:
        if _canonical(spec.name) == canonical:
            return _shown_level(
                _BIDSVAL_LEVEL_TO_FIELD_LEVEL.get(spec.level, FieldLevel.OPTIONAL),
                spec,
            )
    return None


def _dataset_description_fields_for(
    fp: Path, bids_root: Optional[Path] = None,
) -> list[SidecarField]:
    """Form rows for ``dataset_description.json``.

    Its levels are widely mis-stated because they are easy to guess wrong:
    ``Name`` and ``BIDSVersion`` are required, ``License`` and ``DatasetType``
    recommended, and ``Authors``, ``Funding``, ``EthicsApprovals`` and
    ``ReferencesAndLinks`` are merely optional.
    """
    data = _load_json_object(fp)
    if data is None:
        return []
    try:
        specs = bidsval_schema.dataset_description_fields(
            dataset_root=bids_root or Path(fp).parent,
        )
    except Exception:
        specs = []

    seen: set[str] = set()
    fields: list[SidecarField] = []
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        present = spec.name in data
        value = data.get(spec.name) if present else None
        fields.append(SidecarField(
            level=_shown_level(
                _BIDSVAL_LEVEL_TO_FIELD_LEVEL.get(spec.level, FieldLevel.OPTIONAL),
                spec,
            ),
            name=spec.name, value=value, present=present,
            value_kind=value_kind(value) if present else "missing",
            description=spec.description or None,
        ))
    for key, value in data.items():
        if key in seen:
            continue
        seen.add(key)
        fields.append(SidecarField(
            level=FieldLevel.OPTIONAL, name=key, value=value, present=True,
            value_kind=value_kind(value),
        ))
    return fields


def todo_issues_for(fp: Path) -> list[Issue]:
    """Flag every field whose value is the literal placeholder ``"TODO"``.

    Works on any JSON object (typed sidecar or top-level file such as
    ``dataset_description.json``). The fix carries the field name so the
    Editor's fix button jumps straight to that row in the sidecar form.
    """
    if not str(fp).lower().endswith(".json"):
        return []
    data = _load_json_object(fp)
    if data is None:
        return []
    out: list[Issue] = []
    for key, value in data.items():
        if value == "TODO":
            out.append(Issue(
                severity=Severity.WARN,
                rule_id="bidsmgr.todo_placeholder",
                message=f"field {key!r} contains a TODO placeholder",
                field=key,
                fix_label="Set a real value",
                fix_action="set_field",
            ))
    return out


__all__ = [
    "value_kind",
    "infer_datatype_suffix",
    "schema_level_for",
    "sidecar_fields_for",
    "todo_issues_for",
    "sibling_json",
]
