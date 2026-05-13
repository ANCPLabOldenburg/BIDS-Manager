"""Schema rules engine — the keystone API.

Reference: architecture.md §3.

Functions here read the BIDS schema (via ``bidsschematools``) and translate
it into a small, strongly-typed surface. Every other layer of the package
imports from here; this module imports nothing else from ``bidsmgr``.

The schema is the source of truth. We do not hand-curate rules.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .loader import get_schema
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


def required_sidecar_fields(datatype: Datatype, suffix: Suffix) -> list[FieldInfo]:
    return _sidecar_fields(datatype, suffix, levels={"required"})


def recommended_sidecar_fields(datatype: Datatype, suffix: Suffix) -> list[FieldInfo]:
    return _sidecar_fields(datatype, suffix, levels={"recommended"})


def optional_sidecar_fields(datatype: Datatype, suffix: Suffix) -> list[FieldInfo]:
    return _sidecar_fields(datatype, suffix, levels={"optional"})


def deprecated_sidecar_fields(datatype: Datatype, suffix: Suffix) -> list[FieldInfo]:
    return _sidecar_fields(datatype, suffix, levels={"deprecated"})


def field_metadata(field_name: str) -> FieldInfo:
    schema = get_schema()
    raw = schema.objects.metadata.get(field_name) if hasattr(schema.objects.metadata, "get") else None
    if raw is None:
        raise KeyError(f"Unknown metadata field: {field_name!r}")
    return FieldInfo(
        name=str(raw.get("name", field_name)),
        display_name=str(raw.get("display_name", field_name)),
        description=str(raw.get("description", "")),
        type=str(raw.get("type", "string")),
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


def _datatype_groups(datatype: Datatype) -> list[Mapping]:
    """Return raw rule-group dicts that apply to ``datatype``.

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

    return out


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


def _sidecar_fields(datatype: Datatype, suffix: Suffix, levels: set[str]) -> list[FieldInfo]:
    schema = get_schema()
    sidecars = schema.rules.sidecars.get(datatype) if hasattr(schema.rules.sidecars, "get") else None
    if not sidecars:
        return []
    out: list[FieldInfo] = []
    seen: set[str] = set()
    for _group_name, group in sidecars.items():
        applies = group.get("selectors") or []
        # Coarse selector check: sidecar group must mention this suffix in fields.
        # bidsschematools selectors are jsonpath-ish; for now we just check
        # ``suffix == "..."`` substring as a fast path. Errors on the side of
        # over-inclusion, which is fine for required-field auditing.
        if applies and not _selectors_match_suffix(applies, suffix):
            continue
        fields_block = group.get("fields") or {}
        for fname, info in fields_block.items():
            level = _field_level(info)
            if level not in levels:
                continue
            if fname in seen:
                continue
            seen.add(fname)
            out.append(
                FieldInfo(
                    name=fname,
                    display_name=str(_meta_get(fname, "display_name", fname)),
                    description=str(_meta_get(fname, "description", "")),
                    type=str(_meta_get(fname, "type", "string")),
                    required=(level == "required"),
                )
            )
    return out


def _field_level(info) -> str:
    if isinstance(info, str):
        return info
    if hasattr(info, "get"):
        lvl = info.get("level")
        if isinstance(lvl, str):
            return lvl
    return "optional"


def _selectors_match_suffix(selectors: Sequence, suffix: Suffix) -> bool:
    needle = f'== "{suffix}"'
    for sel in selectors:
        if isinstance(sel, str) and (needle in sel or f'"{suffix}"' in sel):
            return True
    return False


def _meta_get(field_name: str, key: str, default):
    schema = get_schema()
    info = schema.objects.metadata.get(field_name) if hasattr(schema.objects.metadata, "get") else None
    if info is None:
        return default
    return info.get(key, default)


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
    "field_metadata",
    "build_basename",
    "build_relative_path",
]
