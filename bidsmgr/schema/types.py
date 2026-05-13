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
    type: str  # JSON Schema type
    required: bool = False


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
