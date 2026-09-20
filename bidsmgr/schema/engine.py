"""Schema rules engine, the keystone API.

Reference: architecture.md §3.

Functions here read the BIDS schema and translate it into a small,
strongly-typed surface. Every other layer of the package imports from here;
this module imports nothing else from ``bidsmgr``.

The schema is the source of truth. We do not hand-curate rules.

Which sidecar fields the standard declares for a datatype/suffix, and at what
requirement level, is read from ``bidsval.schema`` rather than derived here.
That is a fact about BIDS, and facts about BIDS get ONE implementation, so that
a tool which fills metadata and a tool which checks it cannot drift apart. The
rest of this module still reads ``bidsschematools`` directly for vocabulary and
filename construction.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

from ..vendor.bidsval import schema as bidsval_schema

from .loader import active_version, get_schema, register_cache
from .types import Datatype, Entity, EntityFormat, EntityInfo, FieldInfo, Suffix


# -- listing --------------------------------------------------------------


def list_datatypes() -> list[Datatype]:
    """Return all canonical datatypes (anat, func, dwi, eeg, …)."""
    return list(get_schema().objects.datatypes.keys())


def list_suffixes(datatype: Optional[Datatype] = None) -> list[Suffix]:
    """Return suffixes valid for ``datatype`` (or all suffixes if omitted)."""
    schema = get_schema()
    if datatype is None:
        return list(schema.objects.suffixes.keys())
    seen: list[Suffix] = []
    seen_set: set[str] = set()
    for group in _datatype_groups(datatype):
        for s in group.get("suffixes", []) or []:
            if s not in seen_set:
                seen_set.add(s)
                seen.append(s)
    return seen


def list_extensions(datatype: Datatype, suffix: Suffix) -> list[str]:
    """Return file extensions valid for ``(datatype, suffix)``."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for group in _datatype_groups(datatype):
        if suffix not in (group.get("suffixes") or []):
            continue
        for ext in group.get("extensions", []) or []:
            if ext not in seen_set:
                seen_set.add(ext)
                seen.append(ext)
    return seen


# -- entities -------------------------------------------------------------


def entity_order() -> list[Entity]:
    """Return the canonical filename order for all entities."""
    return list(get_schema().rules.entities)


@lru_cache(maxsize=1)
def _entity_index_lookup() -> dict[str, int]:
    return {e: i for i, e in enumerate(entity_order())}


@lru_cache(maxsize=1)
def entity_keys() -> tuple[str, ...]:
    """Every entity's SHORT key, in canonical filename order.

    The schema keys entities by their long name (``subject``) while filenames
    carry the short one (``sub``). Anything that reads or writes a name needs
    the short form, and needs it in this order, so it is derived here once
    rather than re-listed by each caller.
    """
    schema = get_schema()
    out: list[str] = []
    for long_name in entity_order():
        raw = schema.objects.entities.get(long_name) or {}
        key = str(raw.get("name", long_name))
        if key:
            out.append(key)
    return tuple(out)


@lru_cache(maxsize=1)
def _key_to_long() -> dict[str, str]:
    schema = get_schema()
    out: dict[str, str] = {}
    for long_name in entity_order():
        raw = schema.objects.entities.get(long_name) or {}
        out[str(raw.get("name", long_name))] = long_name
    return out


def entity_key_info(key: str) -> EntityInfo:
    """Schema metadata for an entity named by its SHORT key (``"sub"``).

    :func:`entity_info` takes the long name. Callers that start from a
    filename have the short one, and translating in each of them is how the
    two drift apart.
    """
    long_name = _key_to_long().get(key)
    if long_name is None:
        raise KeyError(f"Unknown entity key: {key!r}")
    return entity_info(long_name)


@lru_cache(maxsize=1)
def directory_entity_keys() -> tuple[str, ...]:
    """The entities that name a DIRECTORY as well as appearing in filenames.

    ``sub`` and ``ses`` today, read from ``rules.directories.raw`` rather than
    assumed, because renaming one of these has to move a folder and renaming
    any other must not.
    """
    schema = get_schema()
    raw = schema.rules.directories.get("raw") or {}
    out: list[str] = []
    for name, spec in raw.items():
        if name == "root" or not hasattr(spec, "get"):
            continue
        long_name = spec.get("entity")
        if not long_name:
            continue
        info = schema.objects.entities.get(long_name) or {}
        key = str(info.get("name", long_name))
        if key not in out:
            out.append(key)
    # Canonical order, so a caller iterating them moves the outermost first.
    order = {k: i for i, k in enumerate(entity_keys())}
    return tuple(sorted(out, key=lambda k: order.get(k, len(order))))


@lru_cache(maxsize=1)
def opaque_directories() -> tuple[str, ...]:
    """Top-level directories whose contents are not the raw dataset's.

    ``derivatives``, ``sourcedata``, ``code`` and friends are marked
    ``opaque`` in the schema, which is exactly the property a bulk operation
    needs: what is inside was produced by something else and renaming into it
    silently breaks provenance.
    """
    schema = get_schema()
    raw = schema.rules.directories.get("raw") or {}
    out = [
        str(spec.get("name", name))
        for name, spec in raw.items()
        if name != "root" and hasattr(spec, "get") and spec.get("opaque")
    ]
    return tuple(sorted(out))


def datatypes_with_suffix(suffix: Suffix) -> tuple[Datatype, ...]:
    """Every datatype the standard allows ``suffix`` in.

    Used instead of a hand-kept list of "datatypes that have events" or
    "datatypes that have channels". Those lists were both wrong: they predated
    ``emg`` and ``motion``, so two whole modalities were quietly skipped by
    every check that consulted them.
    """
    return tuple(
        dt for dt in list_datatypes() if suffix in list_suffixes(dt)
    )


def required_entities(datatype: Datatype, suffix: Suffix) -> list[Entity]:
    return _entities_with_kind(datatype, suffix, "required")


def optional_entities(datatype: Datatype, suffix: Suffix) -> list[Entity]:
    return _entities_with_kind(datatype, suffix, "optional")


def deprecated_entities(datatype: Datatype, suffix: Suffix) -> list[Entity]:
    return _entities_with_kind(datatype, suffix, "deprecated")


def allowed_entities(datatype: Datatype, suffix: Suffix) -> list[Entity]:
    """Union of required + optional + deprecated entities for the (datatype, suffix)."""
    union: dict[str, None] = {}
    for kind in ("required", "optional", "deprecated"):
        for e in _entities_with_kind(datatype, suffix, kind):
            union.setdefault(e, None)
    return list(union.keys())


def entity_info(entity: Entity) -> EntityInfo:
    """Return schema metadata for ``entity`` (e.g. ``"task"``)."""
    schema = get_schema()
    if entity not in schema.objects.entities:
        raise KeyError(f"Unknown entity: {entity!r}")
    raw = schema.objects.entities[entity]
    fmt_name = str(raw.get("format", "label"))
    fmt_pattern = _format_pattern(fmt_name)
    return EntityInfo(
        key=entity,
        name=str(raw.get("name", entity)),
        display_name=str(raw.get("display_name", entity)),
        format=EntityFormat(name=fmt_name, pattern=fmt_pattern),
        description=str(raw.get("description", "")),
    )


def entity_format(entity: Entity) -> EntityFormat:
    return entity_info(entity).format


@lru_cache(maxsize=64)
def _format_pattern(format_name: str) -> str:
    schema = get_schema()
    fmt = schema.objects.formats.get(format_name) if hasattr(schema.objects.formats, "get") else None
    if fmt and "pattern" in fmt:
        return str(fmt["pattern"])
    # Sensible defaults if the schema doesn't expose the format we asked for.
    if format_name == "label":
        return r"[0-9a-zA-Z]+"
    if format_name == "index":
        return r"[0-9]+"
    return r".+"


# -- sidecar fields -------------------------------------------------------


def required_sidecar_fields(
    datatype: Datatype, suffix: Suffix, bids_root: Optional[Path] = None,
) -> list[FieldInfo]:
    return _sidecar_fields(datatype, suffix, levels={"required"}, bids_root=bids_root)


def recommended_sidecar_fields(
    datatype: Datatype, suffix: Suffix, bids_root: Optional[Path] = None,
) -> list[FieldInfo]:
    return _sidecar_fields(datatype, suffix, levels={"recommended"}, bids_root=bids_root)


def optional_sidecar_fields(
    datatype: Datatype, suffix: Suffix, bids_root: Optional[Path] = None,
) -> list[FieldInfo]:
    return _sidecar_fields(datatype, suffix, levels={"optional"}, bids_root=bids_root)


def deprecated_sidecar_fields(
    datatype: Datatype, suffix: Suffix, bids_root: Optional[Path] = None,
) -> list[FieldInfo]:
    return _sidecar_fields(datatype, suffix, levels={"deprecated"}, bids_root=bids_root)


def sidecar_fields(
    datatype: Datatype, suffix: Suffix, bids_root: Optional[Path] = None,
) -> list[FieldInfo]:
    """Every field BIDS declares for this kind of file, at every level.

    The per-level accessors above answer "which fields are required here";
    this answers "what may this file carry at all", which is what a form
    building a whole section needs.
    """
    return _sidecar_fields(
        datatype, suffix,
        levels={"required", "recommended", "optional", "deprecated"},
        bids_root=bids_root,
    )


def dataset_description_fields(bids_root: Optional[Path] = None) -> list[FieldInfo]:
    """Every field BIDS declares for ``dataset_description.json``.

    A top-level dataset file has no datatype and no suffix, so it cannot be
    addressed through :func:`required_sidecar_fields` and friends. Levels here
    are easy to get wrong and were: ``Name`` and ``BIDSVersion`` are required,
    ``License`` and ``DatasetType`` recommended, and ``Authors``, ``Funding``,
    ``EthicsApprovals`` and ``ReferencesAndLinks`` merely optional.
    """
    try:
        specs = bidsval_schema.dataset_description_fields(
            dataset_root=bids_root, schema=active_version(),
        )
    except Exception:
        return []
    return [
        FieldInfo(
            name=spec.name,
            display_name=spec.display_name or spec.name,
            description=spec.description,
            type=spec.type,
            item_type=spec.item_type,
            enum=tuple(spec.enum),
            accepts=tuple(getattr(spec, "accepts", ()) or ()),
            accepts_free_text=bool(getattr(spec, "accepts_free_text", False)),
            accepts_na=bool(getattr(spec, "accepts_na", False)),
            level=spec.level,
            unit=spec.unit,
            required=spec.is_required,
            conditional=spec.conditional,
            speculative=spec.speculative,
        )
        for spec in specs
    ]


def field_applies(field_name: str, datatype: Datatype, suffix: Suffix) -> bool:
    """True when BIDS declares ``field_name`` for this kind of file.

    The question a table column or a form section asks before showing itself:
    does ``PowerLineFrequency`` mean anything for an anatomical scan (no), or
    ``TracerName`` for a MEG recording (no).
    """
    try:
        return bool(bidsval_schema.field_applies(
            field_name, datatype, suffix, schema=active_version(),
        ))
    except Exception:
        return False


@lru_cache(maxsize=256)
def _declared_type(field_name: str, datatype: str, suffix: str) -> str:
    """What shape the standard declares for this field, in this kind of file.

    Read through :func:`sidecar_fields`, which is bidsval's interpretation and
    the one the forms use. The local ``objects.metadata`` reader disagrees about
    a field the schema types with ``anyOf``: it reports PowerLineFrequency as a
    string, when the standard accepts a number or the word "n/a" and rejects
    the string "60".
    """
    if datatype:
        for spec in sidecar_fields(datatype, suffix or datatype):
            if spec.name == field_name:
                return spec.type
    try:
        return field_metadata(field_name).type
    except (KeyError, ValueError, OSError):
        return ""


def coerce(field_name: str, value, datatype: str = "", suffix: str = ""):
    """Put ``value`` into the shape BIDS declares for ``field_name``.

    Everything arrives as text somewhere. The inventory is a TSV, so a mains
    frequency typed into the table is the string "60"; a combo box hands back
    whatever was typed. BIDS declares PowerLineFrequency a number, and writing
    "60" into a sidecar fails validation for a field that was answered right.

    Only a value that genuinely converts is converted. "n/a" stays "n/a", which
    several of these fields accept, and a field the schema types as a string is
    left alone even when it looks like a number: a run label of "01" is not 1.
    """
    if value is None or isinstance(value, (bool, int, float, list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return value
    declared = _declared_type(field_name, datatype, suffix)

    if declared == "boolean":
        low = text.lower()
        return True if low == "true" else False if low == "false" else value
    if declared == "integer":
        try:
            return int(text)
        except ValueError:
            return value
    if declared in ("number", ""):
        # An empty type means anyOf: the field takes more than one shape, and
        # what the user typed decides which.
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return value
    return value


def field_metadata(field_name: str) -> FieldInfo:
    schema = get_schema()
    raw = schema.objects.metadata.get(field_name) if hasattr(schema.objects.metadata, "get") else None
    if raw is None:
        raise KeyError(f"Unknown metadata field: {field_name!r}")
    items = raw.get("items")
    return FieldInfo(
        name=str(raw.get("name", field_name)),
        display_name=str(raw.get("display_name", field_name)),
        description=str(raw.get("description", "")),
        type=str(raw.get("type", "string")),
        # An array says what its items are, and a caller repairing types needs
        # that: "array" alone cannot tell a list of frame times from a list of
        # parameter names.
        item_type=str(items.get("type", "")) if items is not None else "",
    )


# -- name building --------------------------------------------------------


def build_basename(
    entities: Mapping[str, str],
    datatype: Datatype,
    suffix: Suffix,
    extension: str = "",
) -> str:
    """Build a BIDS basename from entities, in the schema's canonical order.

    ``entities`` keys are entity *long* names (``"subject"``, ``"task"``,
    ``"run"``); they are emitted using their schema-defined short form
    (``"sub"``, ``"task"``, ``"run"``). Unknown entity keys raise.

    The basename always ends with ``_<suffix>[<extension>]``; ``sub-`` is
    required. ``extension`` may be empty (e.g. for previewing).
    """

    if not suffix:
        raise ValueError("suffix is required to build a BIDS basename")

    schema = get_schema()
    parts: list[str] = []
    order = entity_order()
    seen: set[str] = set()
    for ent in order:
        if ent not in entities:
            continue
        seen.add(ent)
        value = entities[ent]
        if value in (None, ""):
            continue
        if ent not in schema.objects.entities:
            raise KeyError(f"Unknown entity: {ent!r}")
        short = str(schema.objects.entities[ent].get("name", ent))
        parts.append(f"{short}-{value}")

    leftover = [k for k in entities if k not in seen]
    if leftover:
        raise KeyError(f"Unknown or unordered entities: {leftover!r}")

    if "subject" not in entities or not entities["subject"]:
        raise ValueError("'subject' entity is required")

    parts.append(suffix)
    base = "_".join(parts)
    return f"{base}{extension}" if extension else base


def build_relative_path(
    entities: Mapping[str, str],
    datatype: Datatype,
    suffix: Suffix,
    extension: str = "",
) -> Path:
    """Return the path of the file relative to a BIDS root."""
    if "subject" not in entities or not entities["subject"]:
        raise ValueError("'subject' entity is required")

    rel = Path(f"sub-{entities['subject']}")
    if entities.get("session"):
        rel = rel / f"ses-{entities['session']}"
    rel = rel / datatype
    return rel / build_basename(entities, datatype, suffix, extension)


# -- internals ------------------------------------------------------------


@lru_cache(maxsize=64)
def _datatype_groups(datatype: Datatype) -> tuple[Mapping, ...]:
    """Return raw rule-group dicts that apply to ``datatype``.

    Memoised: the schema does not change within a process, and the inventory
    table asks this once per cell per repaint. Walking the bidsschematools
    Namespace each time cost a viewport over a second.

    The result is a tuple because it is shared between callers; treat the
    dicts inside it as read-only.

    Includes both:

    * Direct rule groups under ``rules.files.raw.<datatype>`` (e.g.
      ``func.func`` for bold/sbref/cbv).
    * Cross-cutting groups under sibling top keys (e.g. ``rules.files.raw.task``
      for physio/physioevents/stim, ``rules.files.raw.events``) whose
      ``datatypes`` list mentions ``datatype``. This is how BIDS 1.11+ models
      suffixes that live in modality directories without being part of the
      modality's "primary" rule group.
    """

    schema = get_schema()
    raw = schema.rules.files.raw
    out: list[Mapping] = []

    if datatype in raw:
        groups = raw[datatype]
        for k in groups.keys():
            out.append(_namespace_to_dict(groups[k]))

    for top_key in raw.keys():
        if top_key == datatype:
            continue
        sibling = raw[top_key]
        for grp_name in sibling.keys():
            grp = _namespace_to_dict(sibling[grp_name])
            datatypes = grp.get("datatypes") or []
            if datatype in datatypes:
                out.append(grp)

    return tuple(out)


def _entities_with_kind(datatype: Datatype, suffix: Suffix, kind: str) -> list[Entity]:
    """Return entities at requirement level ``kind`` for ``(datatype, suffix)``."""
    seen: list[Entity] = []
    seen_set: set[str] = set()
    for group in _datatype_groups(datatype):
        if suffix not in (group.get("suffixes") or []):
            continue
        ent_block = group.get("entities") or {}
        for ent, level in ent_block.items():
            if str(level) == kind and ent not in seen_set:
                seen_set.add(ent)
                seen.append(ent)
    # Re-order by canonical filename order.
    order = _entity_index_lookup()
    seen.sort(key=lambda e: order.get(e, 1_000_000))
    return seen


def _sidecar_fields(
    datatype: Datatype,
    suffix: Suffix,
    levels: set[str],
    bids_root: Optional[Path] = None,
) -> list[FieldInfo]:
    """Ask bidsval which fields the standard declares, and at what level.

    Which fields apply to a datatype/suffix is a fact about BIDS, so it is
    interpreted once, in bidsval, and read here.

    This used to be derived locally by reading ``rules.sidecars.<datatype>``,
    which is not how BIDS files those rules. They are keyed by selector
    expression, and the large MRI groups select on ``modality == "mri"``, so
    ``rules.sidecars.mri.MRIHardware`` never mentions ``anat``. The local
    version therefore found NOTHING for an anatomical scan, which is why the
    Editor's sidecar form was empty for one.
    """
    try:
        specs = bidsval_schema.sidecar_fields(
            datatype, suffix, dataset_root=bids_root, schema=active_version(),
        )
    except Exception:
        # An unknown datatype/suffix is a question, not a crash: callers audit
        # rows whose classification the user is still editing.
        return []
    return [
        FieldInfo(
            name=spec.name,
            display_name=spec.display_name or spec.name,
            description=spec.description,
            type=spec.type,
            item_type=spec.item_type,
            enum=tuple(spec.enum),
            accepts=tuple(getattr(spec, "accepts", ()) or ()),
            accepts_free_text=bool(getattr(spec, "accepts_free_text", False)),
            accepts_na=bool(getattr(spec, "accepts_na", False)),
            level=spec.level,
            unit=spec.unit,
            required=spec.is_required,
            conditional=spec.conditional,
            speculative=spec.speculative,
        )
        for spec in specs
        if spec.level in levels
    ]


def _namespace_to_dict(ns) -> Mapping:
    """``bidsschematools.types.namespace.Namespace`` is mapping-like — wrap defensively."""
    if isinstance(ns, dict):
        return ns
    if hasattr(ns, "keys"):
        return {k: ns[k] for k in ns.keys()}
    return {}


__all__ = [
    "list_datatypes",
    "list_suffixes",
    "list_extensions",
    "entity_order",
    "required_entities",
    "optional_entities",
    "deprecated_entities",
    "allowed_entities",
    "entity_info",
    "entity_format",
    "required_sidecar_fields",
    "recommended_sidecar_fields",
    "optional_sidecar_fields",
    "deprecated_sidecar_fields",
    "dataset_description_fields",
    "sidecar_fields",
    "field_applies",
    "field_metadata",
    "build_basename",
    "build_relative_path",
]


# The memoised lookups above hold answers about one BIDS version. Switching
# version has to drop them, and listing them at the bottom rather than at each
# definition keeps that list in one readable place.
for _cached in (_entity_index_lookup, _format_pattern, _datatype_groups, _declared_type):
    register_cache(_cached)
