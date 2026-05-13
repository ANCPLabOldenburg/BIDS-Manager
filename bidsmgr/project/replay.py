"""Replay an event log into a :class:`ProjectState`.

Reference: architecture.md §9. ``replay`` is a pure function — same
events in, same state out, no I/O. The :class:`ProjectState` is
intentionally a *thin overlay* on top of the inventory TSV; replay
never embeds row contents. The GUI loads the TSV separately and
applies the overlay to produce its working DataFrame.

Order of application is the chronological order of the log. Later
events overwrite earlier ones for the same key:

* ``UserSetEntity``  → ``state.entity_overrides[row_id][entity] = value``
* ``UserSetCell``    → ``state.cell_overrides[row_id][column] = value``
* ``UserToggleInclude`` → ``state.include_overrides[row_id] = include``
* ``TodoAcknowledged`` → adds ``todo_key(scope, field)`` to the set.

Stage events are kept as a small "most recent per stage" dict so the
GUI can render "last validated 5 minutes ago" without scanning the log.
"""

from __future__ import annotations

from typing import Iterable

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
    todo_key,
)


def replay(events: Iterable[Event]) -> ProjectState:
    """Fold ``events`` into a :class:`ProjectState`.

    The first event in a well-formed log is always
    :class:`ProjectCreated`; if it is missing we still build a state but
    the lifecycle fields stay at their defaults. We do not raise on a
    missing-create event because partial logs (e.g. truncated for testing
    or split into time-windowed slices) should still be replayable.
    """

    state = ProjectState()
    count = 0

    entity_overrides: dict[str, dict[str, "str | None"]] = {}
    cell_overrides: dict[str, dict[str, str]] = {}
    include_overrides: dict[str, bool] = {}
    acks: set[str] = set()
    last_stage: dict[str, StageCompleted] = {}

    created_at: "str | None" = None
    bidsmgr_version = ""
    bids_schema_version = ""
    name: "str | None" = None
    description: "str | None" = None
    inventory_tsv: "str | None" = None
    row_ids: tuple[str, ...] = ()

    for ev in events:
        count += 1
        if isinstance(ev, ProjectCreated):
            created_at = ev.ts
            bidsmgr_version = ev.bidsmgr_version
            bids_schema_version = ev.bids_schema_version
            name = ev.name
            description = ev.description
        elif isinstance(ev, ScanImported):
            inventory_tsv = ev.inventory_tsv
            row_ids = ev.row_ids
        elif isinstance(ev, UserSetEntity):
            entity_overrides.setdefault(ev.row_id, {})[ev.entity] = ev.value
        elif isinstance(ev, UserSetCell):
            cell_overrides.setdefault(ev.row_id, {})[ev.column] = ev.value
        elif isinstance(ev, UserToggleInclude):
            include_overrides[ev.row_id] = ev.include
        elif isinstance(ev, TodoAcknowledged):
            acks.add(todo_key(ev.scope, ev.field))
        elif isinstance(ev, StageCompleted):
            last_stage[ev.stage] = ev
        # Future event types: unknown variants would have failed at
        # parse_event(); reaching here for an unknown type is impossible.

    return ProjectState(
        created_at=created_at,
        bidsmgr_version=bidsmgr_version,
        bids_schema_version=bids_schema_version,
        name=name,
        description=description,
        inventory_tsv=inventory_tsv,
        row_ids=row_ids,
        entity_overrides=entity_overrides,
        cell_overrides=cell_overrides,
        include_overrides=include_overrides,
        acknowledged_todos=acks,
        last_stage=last_stage,
        event_count=count,
    )


__all__ = ["replay"]
