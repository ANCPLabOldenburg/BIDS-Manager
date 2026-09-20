"""Renaming some of a subject's files makes a SECOND SUBJECT, not a folder.

The reported case: one subject, rename a few of its files to ``sub-002``. Four
things were wrong, and each of them left the dataset in a state a tool would
misread:

* ``sub-002`` got no ``*_scans.tsv`` at all, so a subject with recordings had
  nothing describing them;
* ``sub-001``'s table kept the moved row, and rewrote it to a path that exists
  under NEITHER subject, so a working reference became a dangling one;
* ``sub-001/anat/`` was left empty, which every tool listing datatypes reads as
  "this subject has anatomy";
* ``participants.tsv`` never gained a row, so the dataset claimed one subject
  and contained two.

A subject is more than a directory. These tests are about the difference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import rename as rn
from bidsmgr.editor.tsv_edit import read_table
from bidsmgr.project.operations import read_log, undo_last


@pytest.fixture
def one_subject(tmp_path: Path) -> Path:
    """One subject, two datatypes, a scans table and a participants row."""
    root = tmp_path / "ds"
    anat = root / "sub-001" / "anat"
    func = root / "sub-001" / "func"
    anat.mkdir(parents=True)
    func.mkdir(parents=True)
    (anat / "sub-001_T1w.nii.gz").write_text("t1")
    (anat / "sub-001_T1w.json").write_text(json.dumps({"EchoTime": 0.03}))
    (func / "sub-001_task-rest_bold.nii.gz").write_text("bold")
    (func / "sub-001_task-rest_bold.json").write_text(
        json.dumps({"TaskName": "rest"})
    )
    (root / "sub-001" / "sub-001_scans.tsv").write_text(
        "filename\tacq_time\n"
        "anat/sub-001_T1w.nii.gz\t2026-01-01T09:00:00\n"
        "func/sub-001_task-rest_bold.nii.gz\t2026-01-01T09:20:00\n"
    )
    (root / "participants.tsv").write_text(
        "participant_id\tage\tsex\nsub-001\t31\tF\n"
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    return root


def _split(root: Path, keep: str = "anat") -> tuple[int, list[str]]:
    """Move the files under ``keep`` to sub-002, leave the rest."""
    plan = rn.plan_rename(root, "sub", "001", "002")
    chosen = {k for k in plan.file_keys(root) if f"/{keep}/" in k}
    assert chosen, "the fixture should have files there"
    return rn.apply_rename(root, plan, only=chosen)


def _scans(subject: Path):
    path = subject / f"{subject.name}_scans.tsv"
    return read_table(path) if path.exists() else None


# ---------------------------------------------------------------------------
# The new subject gets what makes it a subject
# ---------------------------------------------------------------------------


def test_the_new_subject_gets_a_scans_table(one_subject: Path) -> None:
    """It had none. A subject with recordings and no table is the thing a
    split used to leave behind."""
    _split(one_subject)
    table = _scans(one_subject / "sub-002")
    assert table is not None, "sub-002 has no scans table"
    assert table.column("filename") == ["anat/sub-002_T1w.nii.gz"]


def test_the_moved_row_is_rewritten_to_its_new_path(one_subject: Path) -> None:
    """The filename column is relative to the SUBJECT, so a row does not
    merely change label when it moves, it changes meaning."""
    _split(one_subject)
    subject = one_subject / "sub-002"
    for name in _scans(subject).column("filename"):
        assert (subject / name).exists(), name


def test_the_old_table_keeps_only_what_stayed(one_subject: Path) -> None:
    _split(one_subject)
    subject = one_subject / "sub-001"
    names = _scans(subject).column("filename")
    assert names == ["func/sub-001_task-rest_bold.nii.gz"]
    for name in names:
        assert (subject / name).exists(), name


def test_no_row_anywhere_points_at_a_file_that_is_not_there(
    one_subject: Path,
) -> None:
    """The property, checked against the tree rather than against strings."""
    _split(one_subject)
    for subject in sorted(one_subject.glob("sub-*")):
        table = _scans(subject)
        if table is None:
            continue
        for name in table.column("filename"):
            assert (subject / name).exists(), f"{subject.name}: {name}"


def test_the_new_subject_gets_a_participants_row(one_subject: Path) -> None:
    """The dataset claimed one subject and contained two."""
    _split(one_subject)
    table = read_table(one_subject / "participants.tsv")
    assert table.column("participant_id") == ["sub-001", "sub-002"]


def test_the_new_row_carries_what_was_known(one_subject: Path) -> None:
    """The source's values are the only information that exists about the new
    subject. Copying them and letting the user correct beats leaving blanks."""
    _split(one_subject)
    table = read_table(one_subject / "participants.tsv")
    rows = {r[0]: dict(zip(table.header, r)) for r in table.rows}
    assert rows["sub-002"]["age"] == "31"
    assert rows["sub-002"]["sex"] == "F"


def test_a_participants_row_is_never_duplicated(one_subject: Path) -> None:
    (one_subject / "participants.tsv").write_text(
        "participant_id\tage\tsex\nsub-001\t31\tF\nsub-002\t44\tM\n"
    )
    _split(one_subject)
    table = read_table(one_subject / "participants.tsv")
    assert table.column("participant_id") == ["sub-001", "sub-002"]
    rows = {r[0]: dict(zip(table.header, r)) for r in table.rows}
    assert rows["sub-002"]["age"] == "44", "an existing row is not overwritten"


# ---------------------------------------------------------------------------
# What the old subject is left with
# ---------------------------------------------------------------------------


def test_the_folder_the_files_left_is_removed(one_subject: Path) -> None:
    """An empty anat/ reads as "this subject has anatomy" to every tool that
    lists datatypes."""
    _split(one_subject)
    assert not (one_subject / "sub-001" / "anat").exists()
    assert (one_subject / "sub-001" / "func").is_dir(), "the rest stays"


def test_a_folder_that_still_holds_something_is_kept(
    one_subject: Path,
) -> None:
    """Pruning is about folders the move emptied, not about tidying. A file
    left behind IN a folder the move touched keeps that folder."""
    extra = one_subject / "sub-001" / "anat" / "sub-001_T2w.nii.gz"
    extra.write_text("t2")

    plan = rn.plan_rename(one_subject, "sub", "001", "002")
    chosen = {k for k in plan.file_keys(one_subject) if "T1w" in k}
    touched, errors = rn.apply_rename(one_subject, plan, only=chosen)
    assert not errors

    assert extra.exists(), "a file nobody moved must not be swept up"
    assert (one_subject / "sub-001" / "anat").is_dir()


def test_moving_everything_is_still_a_plain_rename(one_subject: Path) -> None:
    """Not a split: there is no second subject, so no table is created and no
    participants row is added."""
    plan = rn.plan_rename(one_subject, "sub", "001", "002")
    rn.apply_rename(one_subject, plan)
    assert not (one_subject / "sub-001").exists()
    assert _scans(one_subject / "sub-002") is not None
    table = read_table(one_subject / "participants.tsv")
    assert table.column("participant_id") == ["sub-002"]


def test_the_old_table_goes_when_every_row_leaves(one_subject: Path) -> None:
    """An empty table is not a table."""
    plan = rn.plan_rename(one_subject, "sub", "001", "002")
    chosen = {
        k for k in plan.file_keys(one_subject)
        if "/anat/" in k or "/func/" in k
    }
    rn.apply_rename(one_subject, plan, only=chosen)
    assert not (one_subject / "sub-001" / "sub-001_scans.tsv").exists()
    assert _scans(one_subject / "sub-002") is not None


# ---------------------------------------------------------------------------
# Splitting into a subject that already exists
# ---------------------------------------------------------------------------


def test_rows_are_appended_to_an_existing_table(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    for sub, datatype, stem in (
        ("sub-001", "anat", "sub-001_T1w"),
        ("sub-001", "func", "sub-001_task-rest_bold"),
        ("sub-002", "dwi", "sub-002_dwi"),
    ):
        folder = root / sub / datatype
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{stem}.nii.gz").write_text("x")
    (root / "sub-001" / "sub-001_scans.tsv").write_text(
        "filename\tacq_time\n"
        "anat/sub-001_T1w.nii.gz\tA\n"
        "func/sub-001_task-rest_bold.nii.gz\tB\n"
    )
    (root / "sub-002" / "sub-002_scans.tsv").write_text(
        "filename\tacq_time\ndwi/sub-002_dwi.nii.gz\tC\n"
    )
    (root / "participants.tsv").write_text(
        "participant_id\nsub-001\nsub-002\n"
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )

    plan = rn.plan_rename(root, "sub", "001", "002", fuse=True)
    chosen = {k for k in plan.file_keys(root) if "/anat/" in k}
    touched, errors = rn.apply_rename(root, plan, only=chosen)
    assert not errors

    names = _scans(root / "sub-002").column("filename")
    assert "dwi/sub-002_dwi.nii.gz" in names, "what was there stays"
    assert "anat/sub-002_T1w.nii.gz" in names, "what arrived is added"
    for name in names:
        assert (root / "sub-002" / name).exists()


# ---------------------------------------------------------------------------
# It is one operation
# ---------------------------------------------------------------------------


def test_a_split_is_a_single_undo(one_subject: Path) -> None:
    """It moves files, rewrites two tables, creates a third and deletes a
    folder. Half of that applied would be worse than none of it."""
    before = {
        str(p.relative_to(one_subject)): p.read_bytes()
        for p in sorted(one_subject.rglob("*"))
        if p.is_file() and ".bidsmgr" not in p.parts
    }
    _split(one_subject)
    assert len(read_log(one_subject)) == 1

    undo_last(one_subject)
    after = {
        str(p.relative_to(one_subject)): p.read_bytes()
        for p in sorted(one_subject.rglob("*"))
        if p.is_file() and ".bidsmgr" not in p.parts
    }
    assert after == before


def test_undo_puts_the_emptied_folder_back(one_subject: Path) -> None:
    _split(one_subject)
    undo_last(one_subject)
    assert (one_subject / "sub-001" / "anat" / "sub-001_T1w.nii.gz").exists()


# ---------------------------------------------------------------------------
# Edge cases that must not crash or corrupt
# ---------------------------------------------------------------------------


def test_a_dataset_with_no_scans_table_splits_anyway(
    one_subject: Path,
) -> None:
    (one_subject / "sub-001" / "sub-001_scans.tsv").unlink()
    touched, errors = _split(one_subject)
    assert not errors
    assert (one_subject / "sub-002" / "anat"
            / "sub-002_T1w.nii.gz").exists()
    assert _scans(one_subject / "sub-002") is None, "none to move, none made"


def test_a_dataset_with_no_participants_table_splits_anyway(
    one_subject: Path,
) -> None:
    (one_subject / "participants.tsv").unlink()
    touched, errors = _split(one_subject)
    assert not errors
    assert not (one_subject / "participants.tsv").exists()


def test_a_scans_table_without_a_filename_column_is_left_alone(
    one_subject: Path,
) -> None:
    """Without the column that identifies a row there is no safe edit."""
    odd = one_subject / "sub-001" / "sub-001_scans.tsv"
    odd.write_text("something_else\tacq_time\nx\tA\n")
    touched, errors = _split(one_subject)
    assert not errors
    assert odd.read_text() == "something_else\tacq_time\nx\tA\n"


def test_renaming_a_session_is_not_a_subject_split(one_subject: Path) -> None:
    """Only a subject rename makes a second subject. A session rename that
    moves some files stays inside the subject it started in."""
    session = one_subject / "sub-001" / "ses-pre" / "anat"
    session.mkdir(parents=True)
    (session / "sub-001_ses-pre_T1w.nii.gz").write_text("x")
    plan = rn.plan_rename(one_subject, "ses", "pre", "post")
    touched, errors = rn.apply_rename(one_subject, plan)
    assert not errors
    assert (one_subject / "sub-001" / "ses-post").is_dir()
    assert read_table(
        one_subject / "participants.tsv"
    ).column("participant_id") == ["sub-001"]


# ---------------------------------------------------------------------------
# Sessions, and collisions at the destination
# ---------------------------------------------------------------------------


@pytest.fixture
def two_sessions(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    for session in ("ses-pre", "ses-post"):
        folder = root / "sub-001" / session / "anat"
        folder.mkdir(parents=True)
        (folder / f"sub-001_{session}_T1w.nii.gz").write_text("x")
    (root / "sub-001" / "sub-001_scans.tsv").write_text(
        "filename\tacq_time\n"
        "ses-pre/anat/sub-001_ses-pre_T1w.nii.gz\tA\n"
        "ses-post/anat/sub-001_ses-post_T1w.nii.gz\tB\n"
    )
    (root / "participants.tsv").write_text("participant_id\nsub-001\n")
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    return root


def test_a_split_carries_the_session_structure(two_sessions: Path) -> None:
    """Splitting one session out of a subject keeps it a session, and the
    scans paths stay relative to the subject they now belong to."""
    plan = rn.plan_rename(two_sessions, "sub", "001", "002")
    chosen = {k for k in plan.file_keys(two_sessions) if "ses-pre" in k}
    touched, errors = rn.apply_rename(two_sessions, plan, only=chosen)
    assert not errors

    assert (two_sessions / "sub-002" / "ses-pre" / "anat"
            / "sub-002_ses-pre_T1w.nii.gz").exists()
    assert (two_sessions / "sub-001" / "ses-post" / "anat"
            / "sub-001_ses-post_T1w.nii.gz").exists()
    assert _scans(two_sessions / "sub-002").column("filename") == [
        "ses-pre/anat/sub-002_ses-pre_T1w.nii.gz"
    ]
    assert _scans(two_sessions / "sub-001").column("filename") == [
        "ses-post/anat/sub-001_ses-post_T1w.nii.gz"
    ]


def test_the_emptied_session_folder_goes_too(two_sessions: Path) -> None:
    """Not just the datatype folder: the session above it is empty as well,
    and an empty ses-pre/ says the subject has a session it does not."""
    plan = rn.plan_rename(two_sessions, "sub", "001", "002")
    chosen = {k for k in plan.file_keys(two_sessions) if "ses-pre" in k}
    rn.apply_rename(two_sessions, plan, only=chosen)
    assert not (two_sessions / "sub-001" / "ses-pre").exists()
    assert (two_sessions / "sub-001" / "ses-post").is_dir()


def test_a_collision_at_the_destination_refuses_the_split(
    two_sessions: Path,
) -> None:
    """Two recordings claiming one name is a decision for a person. The plan
    must say which file, not just that something is wrong."""
    existing = two_sessions / "sub-002" / "ses-pre" / "anat"
    existing.mkdir(parents=True)
    (existing / "sub-002_ses-pre_T1w.nii.gz").write_text("already here")

    plan = rn.plan_rename(two_sessions, "sub", "001", "002", fuse=True)
    assert plan.conflicts
    assert "sub-002_ses-pre_T1w.nii.gz" in plan.conflicts[0]
    with pytest.raises(rn.RenameError):
        rn.apply_rename(two_sessions, plan)
    assert (existing / "sub-002_ses-pre_T1w.nii.gz").read_text() == \
        "already here"
