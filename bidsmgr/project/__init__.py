"""Event-sourced project files + provenance.

Reference: architecture.md §9, §10.

A project is an append-only event log on disk (plus a small provenance
side-table). Every state change is one event. Replaying the log produces
the current state. Undo = drop the last event. Audit = read the log.

Public API:

* :class:`Project` — open or create a ``*.bidsmgr`` bundle directory.
* :class:`ProjectState` — the result of replaying an event log.
* Event types: :class:`ProjectCreated`, :class:`ScanImported`,
  :class:`UserSetEntity`, :class:`UserSetCell`,
  :class:`UserToggleInclude`, :class:`TodoAcknowledged`,
  :class:`StageCompleted`.
* :class:`ProvenanceMap` + :class:`ProvenanceEntry` and ``SOURCE_*``
  prefix constants.
* :func:`replay`, :func:`parse_event`, :func:`dump_event`,
  :func:`utc_now`, :func:`todo_key`.

Nothing in this subpackage imports Qt (architectural rule 2).
"""

from .log import EventLog, EventLogCorrupt
from .project import (
    EVENTS_FILENAME,
    META_FILENAME,
    PROVENANCE_FILENAME,
    Project,
    ProjectError,
)
from .provenance import (
    SOURCE_AUTO_NUMBER,
    SOURCE_CLASSIFIER,
    SOURCE_DICOM,
    SOURCE_FIXUP,
    SOURCE_INFER,
    SOURCE_REGEX,
    SOURCE_SCHEMA,
    SOURCE_USER,
    ProvenanceEntry,
    ProvenanceMap,
)
from .replay import replay
from .types import (
    Event,
    ProjectCreated,
    ProjectState,
    ScanImported,
    StageCompleted,
    TodoAcknowledged,
    UserSetCell,
    UserSetEntity,
    UserToggleInclude,
    dump_event,
    parse_event,
    todo_key,
    utc_now,
)

__all__ = [
    # Top-level
    "Project",
    "ProjectError",
    "ProjectState",
    # Bundle layout constants
    "EVENTS_FILENAME",
    "META_FILENAME",
    "PROVENANCE_FILENAME",
    # Event types
    "Event",
    "ProjectCreated",
    "ScanImported",
    "StageCompleted",
    "TodoAcknowledged",
    "UserSetCell",
    "UserSetEntity",
    "UserToggleInclude",
    # Event helpers
    "dump_event",
    "parse_event",
    "replay",
    "todo_key",
    "utc_now",
    # Log
    "EventLog",
    "EventLogCorrupt",
    # Provenance
    "ProvenanceEntry",
    "ProvenanceMap",
    "SOURCE_AUTO_NUMBER",
    "SOURCE_CLASSIFIER",
    "SOURCE_DICOM",
    "SOURCE_FIXUP",
    "SOURCE_INFER",
    "SOURCE_REGEX",
    "SOURCE_SCHEMA",
    "SOURCE_USER",
]
