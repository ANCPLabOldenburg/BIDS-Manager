"""Unit tests for ``bidsmgr.project.project.Project`` — bundle lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest

import bidsmgr
from bidsmgr.project import (
    EVENTS_FILENAME,
    META_FILENAME,
    PROVENANCE_FILENAME,
    Project,
    ProjectError,
    ProvenanceMap,
    ScanImported,
    StageCompleted,
    UserSetEntity,
    UserToggleInclude,
    SOURCE_USER,
)


def test_create_makes_bundle_with_expected_layout(tmp_path: Path) -> None:
    root = tmp_path / "demo.bidsmgr"
    proj = Project.create(root, name="demo")
    assert root.is_dir()
    assert (root / EVENTS_FILENAME).exists()
    assert (root / META_FILENAME).exists()
    # provenance.json is created lazily on first save.
    assert not (root / PROVENANCE_FILENAME).exists()
    assert proj.state().name == "demo"


def test_create_refuses_to_clobber_existing_path(tmp_path: Path) -> None:
    root = tmp_path / "demo.bidsmgr"
    root.mkdir()
    with pytest.raises(ProjectError):
        Project.create(root)


def test_create_records_project_created_event(tmp_path: Path) -> None:
    proj = Project.create(tmp_path / "demo.bidsmgr", name="demo", description="d")
    state = proj.state()
    assert state.event_count == 1
    assert state.name == "demo"
    assert state.description == "d"
    assert state.bidsmgr_version == bidsmgr.__version__


def test_open_returns_same_state_as_create(tmp_path: Path) -> None:
    root = tmp_path / "demo.bidsmgr"
    p1 = Project.create(root, name="demo")
    p1.append(ScanImported(inventory_tsv="/tmp/inv.tsv", row_ids=("r1",)))
    p1.append(UserSetEntity(row_id="r1", entity="task", value="rest"))

    p2 = Project.open(root)
    state = p2.state()
    assert state.row_ids == ("r1",)
    assert state.entity_overrides["r1"]["task"] == "rest"
    assert state.event_count == 3  # create + scan + edit


def test_open_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ProjectError):
        Project.open(tmp_path / "does_not_exist.bidsmgr")


def test_open_rejects_directory_without_events_log(tmp_path: Path) -> None:
    root = tmp_path / "empty.bidsmgr"
    root.mkdir()
    with pytest.raises(ProjectError):
        Project.open(root)


def test_undo_last_drops_one_event(tmp_path: Path) -> None:
    proj = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    proj.append(UserToggleInclude(row_id="r1", include=False))
    proj.append(UserToggleInclude(row_id="r1", include=True))
    before = proj.state()
    assert proj.undo_last() is True
    after = proj.state()
    assert after.event_count == before.event_count - 1
    # After undo, r1's include reflects the earlier toggle (False).
    assert after.include_overrides["r1"] is False


def test_provenance_starts_empty(tmp_path: Path) -> None:
    proj = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    assert len(proj.provenance()) == 0


def test_provenance_roundtrip(tmp_path: Path) -> None:
    proj = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    m = ProvenanceMap()
    m.set("r1", "task", SOURCE_USER)
    proj.save_provenance(m)

    reopened = Project.open(tmp_path / "demo.bidsmgr")
    loaded = reopened.provenance()
    assert len(loaded) == 1
    assert loaded.get("r1", "task").source == SOURCE_USER


def test_meta_returns_bundle_metadata(tmp_path: Path) -> None:
    proj = Project.create(tmp_path / "demo.bidsmgr", name="demo", description="d")
    meta = proj.meta()
    assert meta["name"] == "demo"
    assert meta["description"] == "d"
    assert meta["bidsmgr_version"] == bidsmgr.__version__
    assert "created_at" in meta
    assert "bids_schema_version" in meta


def test_state_records_last_stage(tmp_path: Path) -> None:
    proj = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    proj.append(StageCompleted(stage="scan", summary={"rows": 5}))
    proj.append(StageCompleted(stage="convert", success=False,
                                summary={"errors": 2}))
    state = proj.state()
    assert state.last_stage["scan"].summary == {"rows": 5}
    assert state.last_stage["convert"].success is False
