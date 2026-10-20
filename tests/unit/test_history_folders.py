"""Undo and redo put folders back as well as files.

Before, the history recorded files only: undoing a subject rename moved every
file back and left the renamed subject's folders standing empty, and an
undone merge left the merged-away subject missing. Folders an operation
creates are now recorded and removed again on undo, folders it removes are
recorded and put back, and the Editor can delete an empty folder at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import remove as rm
from bidsmgr.editor import rename as rn
from bidsmgr.project.operations import (
    begin_operation,
    redo_last,
    undo_last,
)


def _tree(root: Path) -> list[str]:
    """Every file AND folder, the tool's own state aside."""
    return sorted(p.relative_to(root).as_posix() + ("/" if p.is_dir() else "")
                  for p in root.rglob("*") if ".bidsmgr" not in p.parts)


def _subject(root: Path, label: str, *, session: str = "") -> None:
    base = root / f"sub-{label}" / (f"ses-{session}" if session else "") / "anat"
    base.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{label}" + (f"_ses-{session}" if session else "") + "_T1w"
    (base / f"{stem}.nii.gz").write_bytes(b"\x1f\x8b fake")
    (base / f"{stem}.json").write_text(json.dumps({"RepetitionTime": 2.3}))


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    (root / "participants.tsv").write_text("participant_id\nsub-001\nsub-003\n")
    _subject(root, "001")
    _subject(root, "003")
    return root


def test_undoing_a_subject_rename_leaves_no_empty_folder(ds: Path) -> None:
    before = _tree(ds)
    rn.apply_rename(ds, rn.plan_rename(ds, "sub", "001", "002"))
    assert (ds / "sub-002").is_dir() and not (ds / "sub-001").exists()
    undo_last(ds)
    assert _tree(ds) == before, "sub-002/ and its folders went with the files"
    redo_last(ds)
    assert (ds / "sub-002" / "anat").is_dir() and not (ds / "sub-001").exists()
    undo_last(ds)
    assert _tree(ds) == before


def test_undoing_a_merge_gives_the_merged_subject_back(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    (root / "participants.tsv").write_text("participant_id\nsub-001\nsub-003\n")
    _subject(root, "001", session="pre")
    _subject(root, "003", session="post")
    before = _tree(root)
    rn.apply_rename(root, rn.plan_rename(root, "sub", "003", "001", fuse=True))
    assert not (root / "sub-003").exists(), "merged away"
    assert (root / "sub-001" / "ses-post" / "anat").is_dir()
    undo_last(root)
    assert _tree(root) == before, "sub-003 is back, ses-post is gone from sub-001"
    redo_last(root)
    assert not (root / "sub-003").exists()


def test_deleting_a_subject_and_undoing_it_restores_its_folders(ds: Path) -> None:
    (ds / "sub-003" / "func").mkdir()            # a folder with nothing in it
    before = _tree(ds)
    plan = rm.plan_delete(ds, [ds / "sub-003"])
    rm.apply_delete(ds, plan)
    assert not (ds / "sub-003").exists(), "an empty subfolder no longer keeps it standing"
    undo_last(ds)
    assert _tree(ds) == before
    redo_last(ds)
    assert not (ds / "sub-003").exists()


def test_an_empty_folder_can_be_deleted(ds: Path) -> None:
    empty = ds / "sub-001" / "dwi"
    empty.mkdir()
    plan = rm.plan_delete(ds, [empty])
    assert not plan.is_empty and plan.only_folders
    assert plan.verb() == "Delete the empty folder sub-001/dwi"
    rm.apply_delete(ds, plan)
    assert not empty.exists()
    undo_last(ds)
    assert empty.is_dir()


def test_a_folder_holding_only_system_droppings_counts_as_empty(ds: Path) -> None:
    folder = ds / "sub-001" / "beh"
    folder.mkdir()
    (folder / ".DS_Store").write_bytes(b"\x00\x00")
    plan = rm.plan_delete(ds, [folder])
    rm.apply_delete(ds, plan)
    assert not folder.exists()
    undo_last(ds)
    assert (folder / ".DS_Store").read_bytes() == b"\x00\x00", "put back as it was"


def test_several_empty_folders_in_one_step(ds: Path) -> None:
    folders = [ds / "sub-001" / "dwi", ds / "sub-003" / "fmap", ds / "sub-003" / "fmap" / "x"]
    for f in folders:
        f.mkdir(parents=True)
    plan = rm.plan_delete(ds, folders[:2])
    assert plan.verb() == "Delete 3 empty folders"
    rm.apply_delete(ds, plan)
    assert not any(f.exists() for f in folders)


def test_a_created_folder_someone_has_used_since_is_kept(ds: Path) -> None:
    with begin_operation(ds, "Add a note") as op:
        op.write_text(ds / "sub-001" / "notes" / "a.txt", "x")
    (ds / "sub-001" / "notes" / "mine.txt").write_text("added by hand")
    undo_last(ds)
    assert (ds / "sub-001" / "notes" / "mine.txt").exists()
    assert not (ds / "sub-001" / "notes" / "a.txt").exists()


def test_an_older_record_is_undone_without_leaving_empty_folders(ds: Path) -> None:
    """Recorded before folders were: the folders its undo empties go too."""
    log = ds / ".bidsmgr" / "editor" / "operations.log"
    rn.apply_rename(ds, rn.plan_rename(ds, "sub", "001", "002"))
    records = [json.loads(line) for line in log.read_text().splitlines()]
    for record in records:
        record.pop("dirs", None)
        record["children"] = [c for c in record["children"]
                              if c["kind"] not in ("mkdir", "rmdir")]
    log.write_text("".join(json.dumps(r) + "\n" for r in records))
    undo_last(ds)
    assert not (ds / "sub-002").exists()
    assert (ds / "sub-001" / "anat" / "sub-001_T1w.json").exists()


def test_folders_are_not_counted_or_checked_as_files(ds: Path) -> None:
    from bidsmgr.project.operations import peek_undo

    with begin_operation(ds, "Write one file in new folders") as op:
        op.write_text(ds / "sub-009" / "anat" / "x.json", "{}")
    step = peek_undo(ds, check=True)
    assert step.files == ("sub-009/anat/x.json",) or step.files == (
        str(Path("sub-009/anat/x.json")),)
    assert step.changed_since == ()


def test_deleting_a_session_drops_its_sessions_row(tmp_path: Path) -> None:
    root = tmp_path / "s"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _subject(root, "001", session="pre")
    _subject(root, "001", session="post")
    table = root / "sub-001" / "sub-001_sessions.tsv"
    table.write_text("session_id\tacq_time\nses-pre\tn/a\nses-post\tn/a\n")
    plan = rm.plan_delete(root, [root / "sub-001" / "ses-post"])
    assert [(t.name, s) for t, s in plan.session_rows] == [("sub-001_sessions.tsv", "ses-post")]
    assert "1 sessions row(s)" in plan.summary()
    rm.apply_delete(root, plan)
    assert table.read_text() == "session_id\tacq_time\nses-pre\tn/a\n"
    undo_last(root)
    assert "ses-post" in table.read_text()
