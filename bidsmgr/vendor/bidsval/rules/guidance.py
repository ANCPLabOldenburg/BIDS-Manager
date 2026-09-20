"""Human guidance for findings: what a missing or wrong piece of metadata should
contain, with a concrete example, read from the schema.

Every finding bidsval reports should tell the user how to fix it, not only what is
wrong. These helpers turn a schema field/column/entity definition into a short,
example-bearing sentence that is stored on an :class:`~bidsval.issues.Issue` as its
``suggestion``. Examples are synthesised from the definition itself (its type,
allowed values, item type, and unit), so they stay correct as the schema changes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from bidsschematools.types.namespace import Namespace

from ..schema import introspect


def _first_sentence(text: Any) -> str:
    """The first sentence of a (possibly multi-line, markdown) description."""
    flat = " ".join(str(text or "").split())
    if not flat:
        return ""
    for end in (". ", "; "):
        if end in flat:
            return flat.split(end, 1)[0].strip(" .;") + "."
    return flat if flat.endswith(".") else flat + "."


def _example_value(definition: Mapping[str, Any]) -> Any:
    """A representative example value for a metadata/column definition.

    Built from the definition so it matches the real constraints: an allowed value
    if the field is an enumeration, otherwise a value of the declared type.
    """
    if not isinstance(definition, Mapping):
        return "value"
    enum = definition.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    alternatives = definition.get("anyOf")
    if isinstance(alternatives, list):
        for alt in alternatives:
            if isinstance(alt, Mapping):
                return _example_value(alt)
    declared = definition.get("type")
    if declared == "array":
        items = definition.get("items")
        inner = _example_value(items) if isinstance(items, Mapping) else "value"
        return [inner]
    if declared == "object":
        return {"key": "value"}
    if declared == "number":
        return 1.0
    if declared == "integer":
        return 1
    if declared == "boolean":
        return True
    if declared == "string":
        return "text"
    return "value"


def _snippet(name: str, value: Any) -> str:
    """A copy-pasteable JSON snippet, e.g. ``{"RepetitionTime": 1.0}``."""
    return json.dumps({name: value})


def _metadata_definition(schema: Namespace, field_name: str) -> Mapping[str, Any]:
    definitions = introspect.metadata_by_name(schema).get(field_name) or []
    return definitions[0] if definitions else {}


def field_guidance(schema: Namespace, field_name: str) -> str:
    """What a (missing) metadata field should contain, with an example."""
    definition = _metadata_definition(schema, field_name)
    parts: list[str] = []
    description = _first_sentence(definition.get("description")) if definition else ""
    if description:
        parts.append(description)
    unit = definition.get("unit") if isinstance(definition, Mapping) else None
    if unit:
        parts.append(f"Unit: {unit}.")
    example = _snippet(field_name, _example_value(definition))
    parts.append(f"Add it to the JSON, for example {example}.")
    return " ".join(parts)


def value_guidance(field_name: str, definition: Mapping[str, Any]) -> str:
    """How to correct a present field whose value is the wrong type/shape."""
    example = _snippet(field_name, _example_value(definition))
    description = _first_sentence(definition.get("description")) if definition else ""
    fix = f"Use a value of the correct type, for example {example}."
    return f"{description} {fix}".strip()


def column_guidance(schema: Namespace, column_name: str) -> str:
    """What a (missing) TSV column holds, and how to add it."""
    definition: Mapping[str, Any] = {}
    for candidate in schema["objects"].get("columns", {}).values():
        if str(candidate.get("name", "")) == column_name:
            definition = candidate
            break
    description = _first_sentence(definition.get("description")) if definition else ""
    add = (
        f"Add a '{column_name}' column to the TSV header row "
        "(use 'n/a' for any rows that have no value)."
    )
    return f"{description} {add}".strip()


def entity_guidance(short_name: str, pattern: str) -> str:
    """How an entity's value must be formatted."""
    return (
        f"The value of '{short_name}-' must match the pattern /{pattern}/. "
        f"For example, '{short_name}-01' if labels are zero-padded numbers."
    )
