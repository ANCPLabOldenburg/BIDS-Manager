"""Schema rules engine — the keystone of the package.

Wraps ``bidsschematools`` (canonical BIDS schema source) and exposes a
strongly-typed API every other layer reads from. This module imports nothing
from elsewhere in ``bidsmgr``; everything imports from here. See
architecture.md §0–§3 for the design rationale.

Public API:

* ``list_datatypes``, ``list_suffixes``, ``list_extensions``
* ``required_entities``, ``optional_entities``, ``deprecated_entities``,
  ``allowed_entities``, ``entity_info``, ``entity_format``, ``entity_order``,
  ``entity_keys``, ``entity_key_info``, ``directory_entity_keys``
* ``opaque_directories``, ``datatypes_with_suffix``
* ``required_sidecar_fields``, ``recommended_sidecar_fields``,
  ``optional_sidecar_fields``, ``deprecated_sidecar_fields``,
  ``dataset_description_fields``, ``field_metadata``
* ``build_basename``, ``build_relative_path``
* ``validate_entity_set``, ``validate_basename``, ``validate_dataset``
* Loader: ``get_schema``, ``schema_version``, ``bids_version``
* Types: ``Datatype``, ``Suffix``, ``Entity``, ``EntityFormat``, ``EntityInfo``,
  ``FieldInfo``, ``Severity``, ``Scope``, ``ValidationVerdict``
"""

from __future__ import annotations

from .engine import (
    coerce,
    allowed_entities,
    datatypes_with_suffix,
    directory_entity_keys,
    build_basename,
    build_relative_path,
    deprecated_entities,
    dataset_description_fields,
    deprecated_sidecar_fields,
    entity_format,
    entity_info,
    entity_key_info,
    entity_keys,
    entity_order,
    field_applies,
    field_metadata,
    list_datatypes,
    list_extensions,
    list_suffixes,
    opaque_directories,
    optional_entities,
    optional_sidecar_fields,
    recommended_sidecar_fields,
    required_entities,
    required_sidecar_fields,
    sidecar_fields,
)
from .loader import (
    active_version,
    available_versions,
    bids_version,
    get_schema,
    schema_version,
    set_active_version,
)
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
    "coerce",
    "active_version",
    "available_versions",
    "set_active_version",
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
    "entity_keys",
    "entity_key_info",
    "directory_entity_keys",
    # layout
    "opaque_directories",
    "datatypes_with_suffix",
    # sidecar fields
    "required_sidecar_fields",
    "recommended_sidecar_fields",
    "optional_sidecar_fields",
    "deprecated_sidecar_fields",
    "dataset_description_fields",
    "sidecar_fields",
    "field_applies",
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
