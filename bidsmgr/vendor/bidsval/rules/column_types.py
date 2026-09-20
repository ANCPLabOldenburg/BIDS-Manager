"""Value-type checking for TSV columns, ported from the reference validator.

A column's allowed values come from its schema definition (a format pattern, an enum
of levels, numeric bounds), which a sidecar may refine. This module:

* computes the effective value "signature" for a column (schema, optionally refined
  by the sidecar);
* flags a sidecar that redefines the type incompatibly (``TSV_COLUMN_TYPE_REDEFINED``);
* checks each cell against the signature (``TSV_VALUE_INCORRECT_TYPE``).

It mirrors ``schema/tables.ts`` in the reference (including its loose multi-format
pattern join), so bidsval's value checks stay a subset of the reference's, never a
false positive.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class Signature:
    formats: list[str]
    pattern: str | None = None
    units: str | None = None
    levels: list[str] | None = None
    maximum: float | None = None
    minimum: float | None = None


@dataclass
class Spec:
    pattern: re.Pattern[str]
    levels: list[str] | None
    maximum: float | None
    minimum: float | None


_FORMAT_TYPE = {"integer": "number", "number": "number", "boolean": "boolean"}


def _format_to_type(fmt: str) -> str:
    return _FORMAT_TYPE.get(fmt, "string")


def _get_formats(obj: Mapping[str, Any]) -> list[str]:
    any_of = obj.get("anyOf")
    if isinstance(any_of, list):
        out: list[str] = []
        for alt in any_of:
            if isinstance(alt, Mapping):
                out += _get_formats(alt)
        return out or ["string"]
    return [str(obj.get("format") or obj.get("type") or "string")]


def _extract_schema(obj: Mapping[str, Any]) -> Signature:
    enum = obj.get("enum")
    return Signature(
        formats=_get_formats(obj),
        pattern=obj.get("pattern"),
        units=obj.get("unit"),
        levels=[str(v) for v in enum] if isinstance(enum, list) else None,
        maximum=obj.get("maximum"),
        minimum=obj.get("minimum"),
    )


def _extract_definition(definition: Mapping[str, Any]) -> Signature:
    fmt = definition.get("Format") or ("number" if definition.get("Units") else "string")
    levels = definition.get("Levels")
    return Signature(
        formats=[str(fmt)],
        units=definition.get("Units"),
        levels=[str(k) for k in levels] if isinstance(levels, Mapping) else None,
        maximum=definition.get("Maximum"),
        minimum=definition.get("Minimum"),
    )


def _refine(base: Signature, override: Signature) -> tuple[Signature, str | None]:
    """Combine a schema signature with a sidecar override, or report a conflict.

    Returns ``(signature, error)``: on conflict ``error`` is set and the *base*
    (schema) signature is returned, matching the reference's fall-back behaviour.
    """
    base_types = [_format_to_type(f) for f in base.formats]
    if _format_to_type(override.formats[0]) not in base_types:
        return base, f'format "{override.formats[0]}" must be {" or ".join(base.formats)}'
    effective_levels = override.levels if override.levels is not None else base.levels
    if base.levels is not None and override.levels is not None:
        if not all(v in base.levels for v in effective_levels or []):
            return base, "the redefined levels are not a subset of the schema's levels"
    effective_units = override.units if override.units is not None else base.units
    if base.units is not None and effective_units != base.units:
        return base, f'unit "{effective_units}" must be "{base.units}"'
    effective_min = override.minimum if override.minimum is not None else base.minimum
    if base.minimum is not None and effective_min is not None and effective_min < base.minimum:
        return base, f"minimum {effective_min} is below the schema minimum {base.minimum}"
    effective_max = override.maximum if override.maximum is not None else base.maximum
    if base.maximum is not None and effective_max is not None and effective_max > base.maximum:
        return base, f"maximum {effective_max} is above the schema maximum {base.maximum}"
    return (
        Signature(
            formats=override.formats,
            pattern=base.pattern,
            units=effective_units,
            levels=effective_levels,
            maximum=effective_max,
            minimum=effective_min,
        ),
        None,
    )


def value_signature(
    column_object: Mapping[str, Any], sidecar_def: Any
) -> tuple[Signature, str | None]:
    """The effective signature for a column, and a redefinition error if any."""
    if "definition" in column_object:
        # A "conventional" column: fully overridable by the sidecar.
        source = sidecar_def if isinstance(sidecar_def, Mapping) else column_object["definition"]
        return _extract_definition(source if isinstance(source, Mapping) else {}), None
    base = _extract_schema(column_object)
    if isinstance(sidecar_def, Mapping):
        return _refine(base, _extract_definition(sidecar_def))
    return base, None


def is_trivial(signature: Signature) -> bool:
    """A signature that would accept any value (free-text), so checking is skipped."""
    return (
        signature.levels is None
        and signature.maximum is None
        and signature.minimum is None
        and (
            (signature.pattern is None and signature.formats[:1] == ["string"])
            or signature.pattern == ".*"
        )
    )


def compile_spec(signature: Signature, formats: Mapping[str, Any]) -> Spec:
    if signature.pattern is not None:
        pattern = str(signature.pattern)
    else:
        # Loose join (matches the reference's `^a|b$`); never stricter than it.
        parts = [str(formats.get(f, {}).get("pattern", ".*")) for f in signature.formats]
        pattern = "|".join(parts) if parts else ".*"
    try:
        compiled = re.compile(f"^{pattern}$")
    except re.error:
        compiled = re.compile(".*")  # an unparseable pattern must never false-positive
    return Spec(
        pattern=compiled,
        levels=signature.levels,
        maximum=signature.maximum,
        minimum=signature.minimum,
    )


def check_value(value: str, spec: Spec) -> bool:
    if value == "n/a":
        return True
    if not spec.pattern.match(value):
        return False
    if spec.levels is not None and value not in spec.levels:
        return False
    if spec.maximum is not None or spec.minimum is not None:
        try:
            number = float(value)
        except ValueError:
            return False
        if spec.maximum is not None and number > spec.maximum:
            return False
        if spec.minimum is not None and number < spec.minimum:
            return False
    return True
