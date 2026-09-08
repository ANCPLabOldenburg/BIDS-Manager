"""Strongly-typed primitives returned by the schema engine.

Reference: architecture.md §2.4, §3.

These types are pure data — no I/O methods — and the only schema-related
classes any other layer should consume. ``Datatype``, ``Suffix``, ``Entity``
are kept as plain ``str`` aliases because the canonical list comes from
``bidsschematools`` at runtime; we don't freeze it into a Python enum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

# Plain string aliases — the schema is the source of truth, not a static enum.
Datatype = str
Suffix = str
Entity = str


class Severity(str, Enum):
    """Severity of a single :class:`ValidationVerdict`."""

    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Scope(str, Enum):
    """Where a verdict applies (architecture.md §6)."""

    ENTITY = "entity"
    BASENAME = "basename"
    SIDECAR = "sidecar"
    FILE = "file"
    DATASET = "dataset"


@dataclass(frozen=True)
class EntityFormat:
    """Entity value-format constraints sourced from ``objects.formats``."""

    name: str  # 'label' | 'index' | ...
    pattern: str  # regex


@dataclass(frozen=True)
class EntityInfo:
    """Schema-level metadata for a single entity."""

    key: str  # canonical key, e.g. 'subject', 'task'
    name: str  # filename short form, e.g. 'sub', 'task'
    display_name: str
    format: EntityFormat
    description: str = ""


@dataclass(frozen=True)
class FieldInfo:
    """Sidecar field metadata from ``objects.metadata``."""

    name: str
    display_name: str
    description: str
    # The JSON Schema type, EMPTY when the schema declares none (EchoTime and
    # FlipAngle are anyOf number-or-array). Do not substitute "string" for an
    # empty one: a caller that writes a string into such a field produces a
    # type error, which is how "TODO" ended up in numeric fields.
    type: str
    # For an array field, the JSON type of its ELEMENTS. "array" alone does not
    # say what may go in it, and the schema is specific: Authors holds strings,
    # FrameDuration numbers, SourceDatasets objects.
    item_type: str = ""
    # The controlled vocabulary, empty when the field is free text. A field with
    # an enum accepts nothing outside it, placeholders included.
    enum: tuple = ()
    # The BIDS requirement level verbatim: required / recommended / optional /
    # deprecated / prohibited. ``required`` below is the boolean shorthand;
    # a form needs the word, to badge and to order by.
    level: str = ""
    # The unit the schema declares, empty when it declares none. A form shows
    # it beside the box so nobody has to guess whether a dose is in MBq or Bq.
    unit: str = ""
    required: bool = False
    # True when the level or the rule's applicability depends on something a
    # datatype and suffix cannot settle. Show such a field, but present it as
    # possible rather than certain.
    conditional: bool = False
    # The narrower flag, and the one that decides whether a field may be
    # DEMANDED: the rule it came from may not describe this file at all.
    # RepetitionTime on a bold run is conditional (the schema excuses it when
    # VolumeTiming is present) but not speculative, so a missing value is a real
    # violation. SkullStripped is required of derivatives, and nothing here says
    # this dataset is one, so demanding it of a raw scan invents a violation.
    speculative: bool = False


@dataclass
class ValidationVerdict:
    """Single result from any of the schema validators (architecture.md §2.4)."""

    severity: Severity
    scope: Scope
    rule_id: str
    message: str
    suggestion: Optional[str] = None
    autofix: Optional[Callable[[], None]] = field(default=None, repr=False)

    @property
    def is_ok(self) -> bool:
        return self.severity is Severity.OK


__all__ = [
    "Datatype",
    "Suffix",
    "Entity",
    "Severity",
    "Scope",
    "EntityFormat",
    "EntityInfo",
    "FieldInfo",
    "ValidationVerdict",
]
