"""Pure-data types for the event-sourced project file.

Reference: architecture.md §9 (event-sourced project file) and §10
(provenance side-table).

A project is an append-only log of immutable events. Each event records
one user action or pipeline outcome. Replaying the log produces the
current :class:`ProjectState`. Undo = drop the last event and replay.

Events are a discriminated union (Pydantic v2 ``Field(discriminator=...)``)
so deserialising the JSONL log validates the shape per-type.

No Qt imports here (architectural rule 2). No I/O — see ``log.py``,
``replay.py``, ``project.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string (millisecond precision).

    Used as the default for every event ``ts`` field. Centralised so the
    format stays stable across writers and so tests can monkey-patch it
    if they need deterministic timestamps.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class _EventBase(BaseModel):
    """Common fields every event carries.

    ``v`` is the event-format version. Bump only on breaking changes to
    the on-disk shape; replay code branches on it.

    ``ts`` is a UTC ISO 8601 timestamp. Stored as a string (not
    ``datetime``) so the JSONL log is human-readable without a custom
    decoder.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    v: Literal[1] = 1
    ts: str = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------


class ProjectCreated(_EventBase):
    """First event in every log. Captures versions and an optional name.

    The bidsmgr version is recorded so a future replay can detect logs
    written by an older codebase and either replay them anyway or warn.
    The schema version is recorded for the same reason against the BIDS
    schema bundled with bidsmgr at write time.
    """

    type: Literal["project_created"] = "project_created"
    bidsmgr_version: str
    bids_schema_version: str
    name: Optional[str] = None
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# Inventory events
# ---------------------------------------------------------------------------


class ScanImported(_EventBase):
    """An inventory TSV was loaded into the project.

    Carries the absolute path of the TSV at import time plus the ordered
    list of row ids that were imported. Re-running ``ScanImported`` for a
    new TSV is allowed — the replay overwrites the previous ``row_ids``.

    The TSV itself is *not* embedded. Projects reference the inventory
    TSV by path; sharing a project means sharing both the bundle and the
    TSV (and ideally the DICOM root, or the BIDS root if conversion has
    happened).
    """

    type: Literal["scan_imported"] = "scan_imported"
    inventory_tsv: str
    row_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# User-edit events (the GUI's main producer)
# ---------------------------------------------------------------------------


class UserSetEntity(_EventBase):
    """A user changed one BIDS entity for one row.

    ``value=None`` deletes the entity from the row. ``previous`` is
    optional — the GUI can record it for human-readable undo display
    (e.g. "Undo: task rest → motor") but replay does not need it.
    """

    type: Literal["user_set_entity"] = "user_set_entity"
    row_id: str
    entity: str
    value: Optional[str]
    previous: Optional[str] = None


class UserSetCell(_EventBase):
    """A user changed a non-entity TSV cell (e.g. ``BIDS_name``, ``session``).

    Distinct from :class:`UserSetEntity` because entities have their own
    rebuild semantics (they regenerate ``proposed_basename``); arbitrary
    cells do not.
    """

    type: Literal["user_set_cell"] = "user_set_cell"
    row_id: str
    column: str
    value: str
    previous: str = ""


class UserToggleInclude(_EventBase):
    """A user flipped the ``include`` flag for one row.

    Modelled as its own event (rather than a :class:`UserSetCell` with
    ``column="include"``) because it is by far the most common GUI
    interaction and we want it to read clearly in the audit log.
    """

    type: Literal["user_toggle_include"] = "user_toggle_include"
    row_id: str
    include: bool


class TodoAcknowledged(_EventBase):
    """A user reviewed a TODO placeholder and chose to keep it.

    The metadata engine inserts literal ``"TODO"`` strings for missing
    recommended sidecar fields. The GUI surfaces those for review; once
    the user accepts that a field will stay TODO (perhaps because the
    information is not available), an ack event is recorded so the GUI
    stops nagging on subsequent opens.

    ``scope`` is one of ``"dataset"``, ``"file:<relpath>"``, or
    ``"row:<row_id>"``. ``field`` is the sidecar key the TODO sits in.
    """

    type: Literal["todo_acknowledged"] = "todo_acknowledged"
    scope: str
    field: str
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline-stage events
# ---------------------------------------------------------------------------


class StageCompleted(_EventBase):
    """A bidsmgr CLI verb (or the GUI equivalent) finished.

    ``stage`` is one of ``"scan"``, ``"rebuild"``, ``"convert"``,
    ``"metadata"``, ``"validate"``. ``summary`` is a small free-form
    dict — counts, paths, durations. Full per-row results are NOT
    embedded; they live in their own files (the TSV, the validation
    report, the metadata report). The event is a pointer + a verdict,
    not a snapshot.
    """

    type: Literal["stage_completed"] = "stage_completed"
    stage: str
    success: bool = True
    summary: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Discriminated union — used for typed deserialisation
# ---------------------------------------------------------------------------


Event = Annotated[
    Union[
        ProjectCreated,
        ScanImported,
        UserSetEntity,
        UserSetCell,
        UserToggleInclude,
        TodoAcknowledged,
        StageCompleted,
    ],
    Field(discriminator="type"),
]


_EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)


def parse_event(record: dict) -> Event:
    """Validate ``record`` against the event union and return the typed model.

    Raises ``pydantic.ValidationError`` if ``record["type"]`` is unknown
    or fields don't match the chosen variant. Used by both the log
    reader and tests.
    """
    return _EVENT_ADAPTER.validate_python(record)


def dump_event(event: Event) -> dict:
    """Serialise ``event`` to a JSON-safe dict.

    Uses the discriminated-union adapter so the ``type`` discriminator
    is always present and every field is JSON-encodable (Pydantic v2
    handles enum / datetime conversion automatically).
    """
    return _EVENT_ADAPTER.dump_python(event, mode="json")


# ---------------------------------------------------------------------------
# ProjectState — the result of replaying an event log
# ---------------------------------------------------------------------------


class ProjectState(BaseModel):
    """Snapshot of project state derived from the event log.

    This is what the GUI binds to. Every field is a pure overlay on top
    of the inventory TSV — replay never embeds the TSV itself. To
    materialise the working DataFrame, load the TSV, then apply the
    overrides in :attr:`entity_overrides` / :attr:`cell_overrides` /
    :attr:`include_overrides`.

    ``acknowledged_todos`` is a set of ``"<scope>|<field>"`` keys. The
    canonical key is :func:`todo_key`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    created_at: Optional[str] = None
    bidsmgr_version: str = ""
    bids_schema_version: str = ""
    name: Optional[str] = None
    description: Optional[str] = None

    inventory_tsv: Optional[str] = None
    row_ids: tuple[str, ...] = ()

    entity_overrides: dict[str, dict[str, Optional[str]]] = Field(default_factory=dict)
    cell_overrides: dict[str, dict[str, str]] = Field(default_factory=dict)
    include_overrides: dict[str, bool] = Field(default_factory=dict)
    acknowledged_todos: set[str] = Field(default_factory=set)

    # Most recent ``StageCompleted`` per stage name. Keeps the audit
    # trail short (full history still in the log).
    last_stage: dict[str, "StageCompleted"] = Field(default_factory=dict)

    event_count: int = 0


def todo_key(scope: str, field: str) -> str:
    """Canonical key used in :attr:`ProjectState.acknowledged_todos`."""
    return f"{scope}|{field}"


__all__ = [
    "Event",
    "ProjectCreated",
    "ProjectState",
    "ScanImported",
    "StageCompleted",
    "TodoAcknowledged",
    "UserSetCell",
    "UserSetEntity",
    "UserToggleInclude",
    "dump_event",
    "parse_event",
    "todo_key",
    "utc_now",
]
