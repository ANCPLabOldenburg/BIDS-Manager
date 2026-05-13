"""Unit tests for ``bidsmgr.project.replay`` — events → ProjectState."""

from __future__ import annotations

from bidsmgr.project.replay import replay
from bidsmgr.project.types import (
    ProjectCreated,
    ScanImported,
    StageCompleted,
    TodoAcknowledged,
    UserSetCell,
    UserSetEntity,
    UserToggleInclude,
    todo_key,
)


def test_empty_log_produces_default_state() -> None:
    state = replay([])
    assert state.event_count == 0
    assert state.created_at is None
    assert state.row_ids == ()
    assert state.entity_overrides == {}
    assert state.acknowledged_todos == set()


def test_project_created_populates_lifecycle_fields() -> None:
    ev = ProjectCreated(
        bidsmgr_version="0.0.1",
        bids_schema_version="1.11.1",
        name="study",
        description="desc",
    )
    state = replay([ev])
    assert state.created_at == ev.ts
    assert state.bidsmgr_version == "0.0.1"
    assert state.bids_schema_version == "1.11.1"
    assert state.name == "study"
    assert state.description == "desc"
    assert state.event_count == 1


def test_scan_imported_sets_row_ids() -> None:
    state = replay([
        ScanImported(inventory_tsv="/tmp/inv.tsv", row_ids=("r1", "r2", "r3")),
    ])
    assert state.inventory_tsv == "/tmp/inv.tsv"
    assert state.row_ids == ("r1", "r2", "r3")


def test_later_scan_imported_overrides_earlier_one() -> None:
    state = replay([
        ScanImported(inventory_tsv="/old.tsv", row_ids=("a",)),
        ScanImported(inventory_tsv="/new.tsv", row_ids=("x", "y")),
    ])
    assert state.inventory_tsv == "/new.tsv"
    assert state.row_ids == ("x", "y")


def test_user_set_entity_accumulates_per_row() -> None:
    state = replay([
        UserSetEntity(row_id="r1", entity="task", value="rest"),
        UserSetEntity(row_id="r1", entity="run", value="1"),
        UserSetEntity(row_id="r2", entity="task", value="motor"),
    ])
    assert state.entity_overrides["r1"] == {"task": "rest", "run": "1"}
    assert state.entity_overrides["r2"] == {"task": "motor"}


def test_user_set_entity_overwrite_keeps_last_value() -> None:
    state = replay([
        UserSetEntity(row_id="r1", entity="task", value="rest"),
        UserSetEntity(row_id="r1", entity="task", value="motor"),
    ])
    assert state.entity_overrides["r1"]["task"] == "motor"


def test_user_set_entity_delete_is_preserved_as_none() -> None:
    # value=None records the deletion intent; the consumer applies it.
    state = replay([
        UserSetEntity(row_id="r1", entity="task", value="rest"),
        UserSetEntity(row_id="r1", entity="task", value=None),
    ])
    assert state.entity_overrides["r1"]["task"] is None


def test_user_set_cell_accumulates() -> None:
    state = replay([
        UserSetCell(row_id="r1", column="BIDS_name", value="002"),
        UserSetCell(row_id="r1", column="session", value="pre"),
    ])
    assert state.cell_overrides["r1"] == {"BIDS_name": "002", "session": "pre"}


def test_user_toggle_include_keeps_last_value() -> None:
    state = replay([
        UserToggleInclude(row_id="r1", include=False),
        UserToggleInclude(row_id="r1", include=True),
        UserToggleInclude(row_id="r2", include=False),
    ])
    assert state.include_overrides == {"r1": True, "r2": False}


def test_todo_acknowledged_uses_canonical_key() -> None:
    state = replay([
        TodoAcknowledged(scope="dataset", field="Authors"),
        TodoAcknowledged(scope="row:r1", field="TaskName"),
    ])
    assert todo_key("dataset", "Authors") in state.acknowledged_todos
    assert todo_key("row:r1", "TaskName") in state.acknowledged_todos
    assert len(state.acknowledged_todos) == 2


def test_stage_completed_keeps_only_most_recent_per_stage() -> None:
    state = replay([
        StageCompleted(stage="scan", summary={"rows": 5}),
        StageCompleted(stage="scan", summary={"rows": 10}),
        StageCompleted(stage="convert", summary={"subjects": 2}),
    ])
    assert state.last_stage["scan"].summary == {"rows": 10}
    assert state.last_stage["convert"].summary == {"subjects": 2}


def test_event_count_reflects_total_processed() -> None:
    state = replay([
        ProjectCreated(bidsmgr_version="0", bids_schema_version="1"),
        UserSetEntity(row_id="r1", entity="task", value="rest"),
        UserSetEntity(row_id="r1", entity="task", value="motor"),
        UserToggleInclude(row_id="r1", include=False),
    ])
    assert state.event_count == 4


def test_replay_is_pure_function() -> None:
    # Running replay twice on the same input must produce equal output —
    # there must be no hidden mutable state in the module.
    events = [
        ProjectCreated(bidsmgr_version="0", bids_schema_version="1"),
        UserSetEntity(row_id="r1", entity="task", value="rest"),
    ]
    a = replay(events)
    b = replay(events)
    assert a == b
