"""Provenance side-table — records the source of every value.

Reference: architecture.md §10. Distinct from the event log in two ways:

1. **Granularity**: events are at the row/cell level; provenance is at
   the (row_id, field) level — every entity, every sidecar field, every
   inferred column has its source recorded.

2. **Semantics**: events are "what happened" (action), provenance is
   "where did this come from" (origin). They overlap (a ``UserSetEntity``
   event implies a ``user`` provenance) but provenance also covers
   pipeline values that are not user-edits (DICOM tag readouts,
   classifier decisions, fixup outputs).

On-disk format: a single JSON object. Not append-only because each
(row_id, field) holds *one* source — the most recent writer wins. The
event log retains the full history; provenance is the latest-source
snapshot the GUI binds to its "where did this come from?" right-click.

A side-table file is rewritten in full on every save. With the
architecture target of < 50k inventory rows × ~20 fields each, a 1 MB
JSON file is the worst case — well within the < 2s load budget.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .types import utc_now


# Conventional source-string prefixes — kept as constants so producers
# and consumers stay in sync. The GUI's "where did this come from?"
# tooltip parses the prefix to decide which icon/colour to render.
SOURCE_USER = "user"
SOURCE_DICOM = "dicom"           # "dicom:(0018,0080)" → DICOM tag readout
SOURCE_CLASSIFIER = "classifier" # "classifier:dcm2niix_bidsguess"
SOURCE_REGEX = "regex"           # "regex:task-([a-z]+)"
SOURCE_INFER = "infer"           # "infer:meas_date_session_clustering"
SOURCE_FIXUP = "fixup"           # "fixup:fieldmaps"
SOURCE_SCHEMA = "schema"         # "schema:required_field_default"
SOURCE_AUTO_NUMBER = "auto_number"


@dataclass(frozen=True)
class ProvenanceEntry:
    """One (row_id, field) → source record.

    ``set_at`` is an ISO 8601 UTC timestamp. ``source`` is a free-form
    string, conventionally prefixed by one of the ``SOURCE_*`` constants
    above (e.g. ``"dicom:(0018,0080)"``, ``"classifier:dcm2niix_bidsguess"``).
    """

    row_id: str
    field: str
    source: str
    set_at: str


class ProvenanceMap:
    """In-memory ``(row_id, field) -> ProvenanceEntry`` map.

    Construct empty, populate via :meth:`set`, persist with
    :meth:`save`, reload with :meth:`load`. ``set`` for an existing key
    overwrites the entry — the event log retains the previous source
    for audit purposes.
    """

    def __init__(self) -> None:
        # Keyed by row_id to keep typical access patterns (one row's
        # fields at a time) localised. Inner dict maps field → entry.
        self._by_row: dict[str, dict[str, ProvenanceEntry]] = {}

    def set(
        self,
        row_id: str,
        field: str,
        source: str,
        *,
        set_at: Optional[str] = None,
    ) -> ProvenanceEntry:
        """Record (or replace) the source for ``(row_id, field)``.

        ``set_at`` defaults to now (UTC). Returns the stored entry for
        convenience.
        """
        entry = ProvenanceEntry(
            row_id=row_id,
            field=field,
            source=source,
            set_at=set_at or utc_now(),
        )
        self._by_row.setdefault(row_id, {})[field] = entry
        return entry

    def get(self, row_id: str, field: str) -> Optional[ProvenanceEntry]:
        """Return the entry for ``(row_id, field)`` or ``None`` if unknown."""
        return self._by_row.get(row_id, {}).get(field)

    def for_row(self, row_id: str) -> dict[str, ProvenanceEntry]:
        """All recorded entries for one row. Empty dict if none."""
        return dict(self._by_row.get(row_id, {}))

    def __len__(self) -> int:
        return sum(len(fields) for fields in self._by_row.values())

    def __contains__(self, key: tuple[str, str]) -> bool:
        row_id, field = key
        return field in self._by_row.get(row_id, {})

    def save(self, path: Path) -> None:
        """Write the map to ``path`` as JSON.

        Atomic via temp-file rename so a crash mid-write does not
        corrupt the provenance file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            row_id: {
                field: {
                    "source": entry.source,
                    "set_at": entry.set_at,
                }
                for field, entry in fields.items()
            }
            for row_id, fields in self._by_row.items()
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ProvenanceMap":
        """Load a previously-saved map from disk.

        Absent file → empty map (so a project bundle that has never
        recorded provenance just yields an empty map, not a crash).
        """
        m = cls()
        path = Path(path)
        if not path.exists():
            return m
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row_id, fields in payload.items():
            for field, raw in fields.items():
                m._by_row.setdefault(row_id, {})[field] = ProvenanceEntry(
                    row_id=row_id,
                    field=field,
                    source=raw["source"],
                    set_at=raw["set_at"],
                )
        return m


__all__ = [
    "SOURCE_AUTO_NUMBER",
    "SOURCE_CLASSIFIER",
    "SOURCE_DICOM",
    "SOURCE_FIXUP",
    "SOURCE_INFER",
    "SOURCE_REGEX",
    "SOURCE_SCHEMA",
    "SOURCE_USER",
    "ProvenanceEntry",
    "ProvenanceMap",
]
