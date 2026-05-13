"""Unit tests for ``bidsmgr.project.types`` — event union + parse/dump.

The event types are a Pydantic discriminated union. These tests cover:

* Roundtrip serialisation for every variant.
* Rejection of unknown ``type`` values.
* Rejection of extra fields (``extra="forbid"``).
* The default-``ts`` factory produces ISO 8601 UTC strings.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bidsmgr.project.types import (
    ProjectCreated,
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


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


def _roundtrip(event):
    record = dump_event(event)
    parsed = parse_event(record)
    assert type(parsed) is type(event)
    assert parsed == event
    return record


def test_project_created_roundtrips() -> None:
    _roundtrip(ProjectCreated(
        bidsmgr_version="0.0.1",
        bids_schema_version="1.11.1",
        name="study",
        description="desc",
    ))


def test_scan_imported_roundtrips() -> None:
    _roundtrip(ScanImported(
        inventory_tsv="/tmp/inv.tsv",
        row_ids=("r1", "r2", "r3"),
    ))


def test_user_set_entity_roundtrips_with_delete() -> None:
    # value=None means "delete this entity from the row"
    _roundtrip(UserSetEntity(row_id="r1", entity="task", value=None))


def test_user_set_cell_roundtrips() -> None:
    _roundtrip(UserSetCell(
        row_id="r1", column="BIDS_name", value="002", previous="",
    ))


def test_user_toggle_include_roundtrips() -> None:
    _roundtrip(UserToggleInclude(row_id="r1", include=False))


def test_todo_acknowledged_roundtrips() -> None:
    _roundtrip(TodoAcknowledged(
        scope="row:r1", field="TaskName", reason="known to lab",
    ))


def test_stage_completed_roundtrips() -> None:
    _roundtrip(StageCompleted(
        stage="convert",
        success=True,
        summary={"subjects": 10, "errors": 0, "duration_s": 12.5},
    ))


# ---------------------------------------------------------------------------
# Discriminator + validation
# ---------------------------------------------------------------------------


def test_parse_event_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        parse_event({"v": 1, "type": "not_a_real_event", "ts": utc_now()})


def test_parse_event_rejects_extra_fields() -> None:
    record = {
        "v": 1, "type": "user_toggle_include", "ts": utc_now(),
        "row_id": "r1", "include": True,
        "rogue_field": "nope",
    }
    with pytest.raises(ValidationError):
        parse_event(record)


def test_dump_event_includes_discriminator() -> None:
    ev = UserToggleInclude(row_id="r1", include=True)
    record = dump_event(ev)
    assert record["type"] == "user_toggle_include"
    assert record["v"] == 1
    assert "ts" in record


def test_events_are_immutable() -> None:
    ev = UserSetEntity(row_id="r1", entity="task", value="rest")
    with pytest.raises(ValidationError):
        # frozen=True → assignment raises
        ev.value = "motor"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_utc_now_format() -> None:
    ts = utc_now()
    # Expected: 2026-05-11T08:52:27.108Z
    assert ts.endswith("Z")
    assert "T" in ts
    # Sortable lexicographically (a basic property the GUI relies on).
    later = utc_now()
    assert later >= ts


def test_todo_key_canonical_form() -> None:
    assert todo_key("dataset", "Authors") == "dataset|Authors"
    assert todo_key("file:sub-01/anat/sub-01_T1w.json", "EchoTime") == (
        "file:sub-01/anat/sub-01_T1w.json|EchoTime"
    )
