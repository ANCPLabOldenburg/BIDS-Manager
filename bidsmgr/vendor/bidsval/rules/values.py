"""Validate a metadata value against its schema definition.

Sidecar and dataset_description fields are not only required-or-not; each has a
type (and sometimes an enum, numeric bounds, or item type) in
``schema.objects.metadata``. This checks a present value against that definition
and returns human-readable problems (empty list means valid), e.g. ``Authors``
given as a string yields ``["must be array"]``.

A pragmatic subset of JSON Schema is implemented - the constructs the BIDS
metadata definitions actually use: ``type``, ``enum``, ``anyOf``, ``items``, and
numeric / array-length bounds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def validate_value(value: Any, definition: Mapping[str, Any]) -> list[str]:
    """Return a list of problems for ``value`` vs ``definition`` (empty = valid)."""
    if not isinstance(definition, Mapping):
        return []

    # anyOf: valid if the value satisfies at least one alternative.
    alternatives = definition.get("anyOf")
    if isinstance(alternatives, list):
        if any(not validate_value(value, alt) for alt in alternatives if isinstance(alt, Mapping)):
            return []
        labels = [_type_label(alt.get("type")) for alt in alternatives if isinstance(alt, Mapping)]
        joined = " or ".join(label for label in labels if label)
        return [f"must be {joined}" if joined else "has an invalid value"]

    expected = definition.get("type")
    if expected and not _type_matches(value, expected):
        # A wrong type is the primary, most useful message.
        return [f"must be {_type_label(expected)}"]

    problems: list[str] = []

    enum = definition.get("enum")
    if enum is not None and value not in enum:
        problems.append(f"must be one of {enum}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        problems += _numeric_problems(value, definition)

    if isinstance(value, list):
        problems += _array_problems(value, definition)

    return problems


def _numeric_problems(value: float, definition: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    if "minimum" in definition and value < definition["minimum"]:
        out.append(f"must be >= {definition['minimum']}")
    if "maximum" in definition and value > definition["maximum"]:
        out.append(f"must be <= {definition['maximum']}")
    if "exclusiveMinimum" in definition and value <= definition["exclusiveMinimum"]:
        out.append(f"must be > {definition['exclusiveMinimum']}")
    if "exclusiveMaximum" in definition and value >= definition["exclusiveMaximum"]:
        out.append(f"must be < {definition['exclusiveMaximum']}")
    return out


def _array_problems(value: list[Any], definition: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    if "minItems" in definition and len(value) < definition["minItems"]:
        out.append(f"must have at least {definition['minItems']} item(s)")
    if "maxItems" in definition and len(value) > definition["maxItems"]:
        out.append(f"must have at most {definition['maxItems']} item(s)")
    items = definition.get("items")
    if isinstance(items, Mapping):
        for element in value:
            element_problems = validate_value(element, items)
            if element_problems:
                out.append(f"items {element_problems[0]}")
                break
    return out


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True  # unknown type keyword: do not flag


def _type_label(expected: Any) -> str:
    return str(expected) if expected else ""
