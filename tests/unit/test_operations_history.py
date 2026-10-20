"""The Editor's history: every operation is ONE step of undo, and an undo can
be redone.

Before 1.5.1 the window offered no way to undo an operation at all, and its
Undo button walked back the open file's edits one at a time. A change made
to many files at once (a rename, a fix applied everywhere, a deletion) must
go back together, and coming back must be possible too.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bidsmgr.project.operations import (
    ORIGINALS_DIR,
    REDO_LOG,
    begin_operation,
    peek_redo,
    peek_undo,
    read_log,
    redo_last,
    undo_last,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    ds = tmp_path / "ds"
    (ds / "sub-01" / "anat").mkdir(parents=True)
    (ds / "a.json").write_text('{"x": 1}\n')
    (ds / "b.json").write_text('{"y": 2}\n')
    (ds / "gone.tsv").write_text("h\nv\n")
    (ds / "sub-01" / "anat" / "old.txt").write_text("moving\n")
    return ds


def _state(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text()
            for p in sorted(root.rglob("*"))
            if p.is_file() and ".bidsmgr" not in p.parts}


def _everything(root: Path, label: str = "Everything at once") -> None:
    """One operation of every kind: write, create, delete and rename."""
    with begin_operation(root, label) as op:
        op.write_json(root / "a.json", {"x": 10})
        op.write_json(root / "b.json", {"y": 20})
        op.write_text(root / "new.txt", "created\n")
        op.delete(root / "gone.tsv")
        op.rename(root / "sub-01" / "anat" / "old.txt", root / "sub-01" / "anat" / "new.txt")


def test_a_change_to_many_files_is_undone_in_one_step(root: Path) -> None:
    before = _state(root)
    _everything(root)
    assert _state(root) != before
    undo_last(root)
    assert _state(root) == before
    assert read_log(root) == []


def test_an_undo_can_be_redone_and_undone_again(root: Path) -> None:
    before = _state(root)
    _everything(root)
    after = _state(root)

    undo_last(root)
    assert _state(root) == before
    assert peek_redo(root).label == "Everything at once"

    redo_last(root)
    assert _state(root) == after
    assert peek_redo(root) is None
    assert peek_undo(root).label == "Everything at once", "redone, it can be undone again"

    undo_last(root)
    assert _state(root) == before


def test_several_steps_walk_back_and_forth_in_order(root: Path) -> None:
    states = [_state(root)]
    for value in (1, 2, 3):
        with begin_operation(root, f"Set x to {value}") as op:
            op.write_json(root / "a.json", {"x": value})
        states.append(_state(root))
    for expected in reversed(states[:-1]):
        undo_last(root)
        assert _state(root) == expected
    assert undo_last(root) is None
    for expected in states[1:]:
        redo_last(root)
        assert _state(root) == expected
    assert redo_last(root) is None


def test_a_new_change_ends_what_can_be_redone(root: Path) -> None:
    _everything(root)
    undo_last(root)
    kept = [r["op_id"] for r in json.loads("[" + ",".join(
        (root / REDO_LOG).read_text().splitlines()) + "]")]
    with begin_operation(root, "Something new") as op:
        op.write_json(root / "b.json", {"y": 3})
    assert peek_redo(root) is None
    assert not (root / REDO_LOG).exists()
    for op_id in kept:
        assert not (root / ORIGINALS_DIR / op_id).exists(), "the copies kept to redo go too"


def test_what_undo_and_redo_would_do_is_named_with_its_files(root: Path) -> None:
    assert peek_undo(root) is None and peek_redo(root) is None
    _everything(root, "Rename and tidy")
    step = peek_undo(root)
    assert step.label == "Rename and tidy"
    assert set(step.files) == {"a.json", "b.json", "new.txt", "gone.tsv",
                               "sub-01/anat/new.txt", "sub-01/anat/old.txt"}


def test_a_file_changed_since_the_step_is_named_before_undoing(root: Path) -> None:
    with begin_operation(root, "Edit a") as op:
        op.write_json(root / "a.json", {"x": 5})
    assert peek_undo(root, check=True).changed_since == ()
    (root / "a.json").write_text('{"x": "by hand"}\n')
    assert peek_undo(root, check=True).changed_since == ("a.json",)


def test_undoing_two_steps_on_one_file_raises_no_false_alarm(root: Path) -> None:
    """The second undo finds the file as the first step left it: restored
    files keep their old modification time."""
    for value in (1, 2):
        with begin_operation(root, f"Set x to {value}") as op:
            op.write_json(root / "a.json", {"x": value})
    undo_last(root)
    assert peek_undo(root, check=True).changed_since == ()


def test_the_newest_step_is_read_from_a_long_log(root: Path) -> None:
    for value in range(40):
        with begin_operation(root, f"Step {value}") as op:
            op.write_json(root / "a.json", {"x": value, "pad": "z" * 3000})
    assert peek_undo(root).label == "Step 39"
    # One record longer than the 64 KB the reader takes at a time.
    with begin_operation(root, "Many files") as op:
        for i in range(1500):
            op.write_text(root / f"f{i:04d}_with_a_rather_long_name.txt", "x")
    assert peek_undo(root).label == "Many files"
    assert len(peek_undo(root).files) == 1500


def test_a_failed_undo_changes_nothing(root: Path, monkeypatch) -> None:
    _everything(root)
    after = _state(root)
    import bidsmgr.project.operations as ops

    real = ops.Operation.rename

    def broken(self, src, dst):
        raise OSError("disk said no")

    monkeypatch.setattr(ops.Operation, "rename", broken)
    with pytest.raises(OSError):
        undo_last(root)
    monkeypatch.setattr(ops.Operation, "rename", real)
    assert _state(root) == after, "all or nothing, like any other operation"
    assert peek_undo(root).label == "Everything at once", "still there to undo"


@pytest.mark.skipif(os.name == "nt", reason="POSIX modification times")
def test_restored_files_get_their_old_modification_time(root: Path) -> None:
    os.utime(root / "a.json", ns=(1_000_000_000, 1_000_000_000))
    with begin_operation(root, "Edit a") as op:
        op.write_json(root / "a.json", {"x": 2})
    undo_last(root)
    assert (root / "a.json").stat().st_mtime_ns == 1_000_000_000
