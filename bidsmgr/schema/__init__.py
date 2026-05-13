"""Schema rules engine — the keystone of the package.

Wraps ``bidsschematools`` (canonical BIDS schema source) and exposes a
strongly-typed API every other layer reads from. This module imports nothing
from elsewhere in ``bidsmgr``; everything imports from here. See
architecture.md §0–§3 for the design rationale.

Public API:

* ``list_datatypes``, ``list_suffixes``, ``list_extensions``
* ``required_entities``, ``optional_entities``, ``deprecated_entities``,
  ``allowed_entities``, ``entity_info``, ``entity_format``, ``entity_order``
* ``required_sidecar_fields``, ``recommended_sidecar_fields``,
  ``optional_sidecar_fields``, ``deprecated_sidecar_fields``, ``field_metadata``
* ``build_basename``, ``build_relative_path``
* ``validate_entity_set``, ``validate_basename``, ``validate_dataset``
* Loader: ``get_schema``, ``schema_version``, ``bids_version``
* Types: ``Datatype``, ``Suffix``, ``Entity``, ``EntityFormat``, ``EntityInfo``,
  ``FieldInfo``, ``Severity``, ``Scope``, ``ValidationVerdict``
"""

from __future__ import annotations

from .engine import (
    allowed_entities,
    build_basename,
    build_relative_path,
    deprecated_entities,
    deprecated_sidecar_fields,
    entity_format,
    entity_info,
    entity_order,
    field_metadata,
    list_datatypes,
    list_extensions,
    list_suffixes,
    optional_entities,
    optional_sidecar_fields,
    recommended_sidecar_fields,
    required_entities,
    required_sidecar_fields,
)
from .loader import bids_version, get_schema, schema_version
from .types import (
    Datatype,
    Entity,
    EntityFormat,
    EntityInfo,
    FieldInfo,
    Scope,
    Severity,
    Suffix,
    ValidationVerdict,
)
from .validation import validate_basename, validate_dataset, validate_entity_set

__all__ = [
    # listing
    "list_datatypes",
    "list_suffixes",
    "list_extensions",
    # entities
    "entity_order",
    "required_entities",
    "optional_entities",
    "deprecated_entities",
    "allowed_entities",
    "entity_info",
    "entity_format",
    # sidecar fields
    "required_sidecar_fields",
    "recommended_sidecar_fields",
    "optional_sidecar_fields",
    "deprecated_sidecar_fields",
    "field_metadata",
    # name building
    "build_basename",
    "build_relative_path",
    # validation
    "validate_entity_set",
    "validate_basename",
    "validate_dataset",
    # loader
    "get_schema",
    "schema_version",
    "bids_version",
    # types
    "Datatype",
    "Suffix",
    "Entity",
    "EntityFormat",
    "EntityInfo",
    "FieldInfo",
    "Severity",
    "Scope",
    "ValidationVerdict",
]
