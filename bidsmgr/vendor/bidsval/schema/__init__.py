"""Schema loading and version selection.

This subpackage is the single place that knows how to *find* a BIDS schema, and
the single place that answers what the schema *says*. The validator is one
consumer of that; an editor, a metadata form or a conversion template is
another, and they need the same answers before any file exists. Exposing them
here is what keeps a downstream tool from re-deriving the rules and drifting.

The
rest of the validator receives one resolved schema object and reads all
vocabulary and rules from it, so it never branches on the BIDS version. That
isolation is what lets a user point the validator at any schema (the bundled
default, a specific version, or a local/forked schema) and have everything
downstream work unchanged.
"""

from __future__ import annotations

from .fields import (
    FieldSpec,
    dataset_description_fields,
    field_applies,
    invalidate_dataset_cache,
    sidecar_fields,
)
from .introspect import (
    datatypes,
    entity_pattern,
    extensions,
    metadata_by_name,
    modality_for,
    short_to_long,
    suffixes,
)
from .resolve import (
    DEFAULT_VERSION,
    SchemaNotAvailable,
    SchemaSelector,
    available_versions,
    bids_version,
    resolve,
    schema_version,
)

__all__ = [
    # introspection: what the standard declares, for callers that need to
    # render a form or a template rather than validate a dataset
    "FieldSpec",
    "sidecar_fields",
    "dataset_description_fields",
    "field_applies",
    "invalidate_dataset_cache",
    "datatypes",
    "suffixes",
    "extensions",
    "modality_for",
    "entity_pattern",
    "short_to_long",
    "metadata_by_name",
    # schema loading
    "resolve",
    "available_versions",
    "schema_version",
    "bids_version",
    "DEFAULT_VERSION",
    "SchemaNotAvailable",
    "SchemaSelector",
]
