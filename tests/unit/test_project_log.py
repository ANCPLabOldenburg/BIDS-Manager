"""Unit tests for ``bidsmgr.project.log`` — append-only JSONL event log."""

from __future__ import annotations

from pathlib import Path

import pytest

from bidsmgr.project.log import EventLog, EventLogCorrupt
from bidsmgr.project.types import (
    ProjectCreated,
    UserSetEntity,
    UserToggleInclude,
    dump_event,
)


def _seed(path: Path) -> EventLog:
    log = EventLog(path)
    log.append(ProjectCreated(bidsmgr_version="0.0.1", bids_schema_version="1.11.1"))
    log.append(UserSetEntity(row_id="r1", entity="task", value="rest"))
    log.append(UserToggleInclude(row_id="r1", include=False))
    return log


def test_append_creates_parent_directory(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "nested" / "deeper" / "events.jsonl")
    log.append(ProjectCreated(bidsmgr_version="0", bids_schema_version="1"))
    assert log.path.exists()


def test_absent_log_is_empty(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    assert len(log) == 0
    assert list(log) == []


def test_append_then_iterate_in_order(tmp_path: Path) -> None:
    log = _seed(tmp_path / "events.jsonl")
    events = list(log)
    assert len(events) == 3
    assert isinstance(events[0], ProjectCreated)
    assert isinstance(events[1], UserSetEntity)
    assert isinstance(events[2], UserToggleInclude)
    assert events[1].value == "rest"


def test_len_matches_appended_count(tmp_path: Path) -> None:
    log = _seed(tmp_path / "events.jsonl")
    assert len(log) == 3


def test_one_line_per_event(tmp_path: Path) -> None:
    log = _seed(tmp_path / "events.jsonl")
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    # Each line is one self-contained JSON object.
    import json
    for line in lines:
        json.loads(line)


def test_iter_rejects_invalid_json(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    log.append(ProjectCreated(bidsmgr_version="0", bids_schema_version="1"))
    # Manually append a corrupt line.
    with log.path.open("a", encoding="utf-8") as fh:
        fh.write("not valid json at all\n")
    with pytest.raises(EventLogCorrupt):
        list(log)


def test_iter_rejects_schema_violations(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    log.append(ProjectCreated(bidsmgr_version="0", bids_schema_version="1"))
    # Hand-craft a record whose ``type`` is unknown.
    import json
    with log.path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"v": 1, "type": "nope", "ts": "2026-01-01T00:00:00Z"}) + "\n")
    with pytest.raises(EventLogCorrupt):
        list(log)


def test_blank_lines_are_tolerated(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    log.append(ProjectCreated(bidsmgr_version="0", bids_schema_version="1"))
    # Append a stray blank line (mimics a half-flushed write recovered as empty).
    with log.path.open("a", encoding="utf-8") as fh:
        fh.write("\n")
    log.append(UserToggleInclude(row_id="r1", include=True))
    assert len(list(log)) == 2


def test_truncate_last_removes_one_event(tmp_path: Path) -> None:
    log = _seed(tmp_path / "events.jsonl")
    assert log.truncate_last() is True
    events = list(log)
    assert len(events) == 2
    assert isinstance(events[-1], UserSetEntity)


def test_truncate_last_on_empty_log_is_noop(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    assert log.truncate_last() is False


def test_truncate_last_on_single_event_empties_log(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    log.append(ProjectCreated(bidsmgr_version="0", bids_schema_version="1"))
    assert log.truncate_last() is True
    assert len(log) == 0
    assert list(log) == []


def test_truncate_last_is_atomic(tmp_path: Path) -> None:
    log = _seed(tmp_path / "events.jsonl")
    log.truncate_last()
    # The temp file used for atomic rewrite should not be left behind.
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []
