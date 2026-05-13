"""Pure-data types for the validator.

Designed to be the single source of truth for both the CLI summary
output and the GUI editor view (``inspector_proto/proto.py`` is the
visual reference). The shape mirrors what the prototype's three
panes already render:

* Tree badge dots (one severity per file) → ``FileVerdict.severity``.
* Schema-aware sidecar form (one row per JSON field with its level)
  → ``FileVerdict.sidecar_fields``.
* Issues panel grouped by scope (this file / this folder / dataset)
  → ``FileVerdict.issues`` + ``ValidationReport.folder_issues`` +
  ``ValidationReport.dataset_issues``.

No Qt imports here (architectural rule 2). Pydantic v2 only.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Severity(str, Enum):
    """Tree-badge color in the prototype.

    Worth-attention ordering: ``OK < WARN < ERR``. The ``rollup`` helper
    on :class:`ValidationReport` uses this to compute dataset-wide
    severity from per-file severities.
    """
    OK = "ok"
    WARN = "warn"
    ERR = "err"


class FieldLevel(str, Enum):
    """Schema-defined level of a sidecar field.

    Drives the prototype's left-bar colour and stylings (``SidecarRow``
    in ``proto.py`` line 414). REQUIRED gets a red bar; RECOMMENDED an
    amber one; OPTIONAL grey; DEPRECATED struck-through grey.
    """
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    DEPRECATED = "deprecated"


class Issue(BaseModel):
    """One validator finding.

    Fits the GUI's ``ValMessage`` widget (``proto.py`` line 463) which
    renders ``severity``, an uppercase rule label, the message body,
    and an optional fix button. ``fix_action`` is an opaque token the
    GUI maps to a handler; the validator never executes fixes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    severity: Severity
    rule_id: str
    message: str
    field: Optional[str] = None  # JSON key when applicable (or BIDS entity)
    fix_label: Optional[str] = None
    fix_action: Optional[str] = None


class SidecarField(BaseModel):
    """One row of the GUI's schema-aware sidecar form.

    Represents every schema-defined field at this ``(datatype, suffix)``
    PLUS every field actually present in the JSON (so the form shows
    extras the user added). The ``level`` for unknown fields is
    ``OPTIONAL``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    level: FieldLevel
    name: str
    value: Optional[Any] = None       # parsed JSON value; None if missing
    present: bool = False
    value_kind: str = "missing"       # "string"|"number"|"array"|"object"|"bool"|"null"|"todo"|"missing"
    description: Optional[str] = None  # tooltip text from the schema


class FileVerdict(BaseModel):
    """Per-file rollup. One per actual file in the BIDS tree.

    The GUI's left-pane tree badge consumes ``severity`` directly. The
    center pane's sidecar form consumes ``sidecar_fields`` (only present
    for JSON sidecars; empty for ``.nii.gz``/``.tsv``/etc).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path                 # relative to bids_root
    severity: Severity = Severity.OK
    datatype: Optional[str] = None
    suffix: Optional[str] = None
    issues: list[Issue] = Field(default_factory=list)
    sidecar_fields: list[SidecarField] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Top-level result. Written to ``<bids_root>/.bidsmgr/validation_report.json``.

    The CLI prints a flattened summary; the GUI binds this directly to
    its three panes — no reshape, no glue code beyond severity-to-paint
    mapping.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bids_root: Optional[Path] = None
    bidsmgr_version: str = ""
    bids_version: str = ""
    generated_at: Optional[str] = None
    severity: Severity = Severity.OK
    counts: dict[str, int] = Field(
        default_factory=lambda: {"ok": 0, "warn": 0, "err": 0}
    )
    files: list[FileVerdict] = Field(default_factory=list)
    folder_issues: dict[str, list[Issue]] = Field(default_factory=dict)
    dataset_issues: list[Issue] = Field(default_factory=list)


def rollup_severity(severities: list[Severity]) -> Severity:
    """Pick the highest severity from a list. ``OK`` if empty."""
    if any(s is Severity.ERR for s in severities):
        return Severity.ERR
    if any(s is Severity.WARN for s in severities):
        return Severity.WARN
    return Severity.OK


__all__ = [
    "FieldLevel", "FileVerdict", "Issue", "Severity",
    "SidecarField", "ValidationReport", "rollup_severity",
]
