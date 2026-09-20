"""Renaming a folder the GUI is watching, which Windows refuses.

``QFileSystemWatcher`` is ``FindFirstChangeNotification`` on Windows, which
holds an open handle on every directory it watches, and an open handle makes
``MoveFile`` fail with ``ERROR_ACCESS_DENIED``. A subject rename then moved
every file correctly, failed only on the call that would have moved the folder,
and left the emptied original behind — the reported symptom.

The behavioural assertions here are Windows-only because the behaviour is: on
POSIX a watched directory renames fine and there is nothing to prove.

Each platform asserts the contract that actually holds there. On Windows the
block releases the watches and restores the paths that survived; off Windows
it is a deliberate no-op and the watches are untouched. Those are different
promises, and a test written for one of them fails on the other: the
"restores what still exists" test used to run everywhere and failed on macOS,
where the block does nothing so a deleted path simply stays in
``directories()``.

What runs on every platform is the end a caller cares about, which is that a
watched directory renames inside the block. On Windows that passes because of
the release, off Windows because the platform never objected, and it is the
same assertion either way.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PyQt6.QtCore import QFileSystemWatcher
from PyQt6.QtWidgets import QWidget

from bidsmgr.gui.fs_watch import live_watchers, watchers_released

pytestmark = pytest.mark.gui

windows_only = pytest.mark.skipif(
    os.name != "nt", reason="only Windows holds a handle on a watched directory",
)


def _tree(tmp_path: Path) -> Path:
    """A miniature of the shape that broke: sub-X/ses-pre/<datatype>."""
    leaf = tmp_path / "sub-old" / "ses-pre" / "eeg"
    leaf.mkdir(parents=True)
    (leaf / "sub-old_ses-pre_task-rest_eeg.edf").write_bytes(b"")
    return leaf


@windows_only
def test_a_watched_directory_cannot_be_renamed(qtbot, tmp_path):
    """The defect itself. If this ever stops failing, the fix is unnecessary."""
    leaf = _tree(tmp_path)
    subject = leaf.parent.parent

    host = QWidget()
    qtbot.addWidget(host)
    watcher = QFileSystemWatcher(host)
    watcher.addPaths([str(leaf), str(leaf.parent), str(subject)])

    with pytest.raises(PermissionError) as excinfo:
        subject.rename(subject.with_name("sub-new"))
    assert getattr(excinfo.value, "winerror", None) == 5
    assert subject.exists(), "the rename failed, so the old folder is still there"


@windows_only
def test_a_watch_on_a_descendant_alone_also_blocks_the_parent(qtbot, tmp_path):
    """Why the dataset-folder rename needs the same treatment.

    Nothing watches the project root itself; the panes watch the datatype
    folders under it. That is still enough to pin the root in place.
    """
    leaf = _tree(tmp_path)
    subject = leaf.parent.parent

    host = QWidget()
    qtbot.addWidget(host)
    watcher = QFileSystemWatcher(host)
    watcher.addPaths([str(leaf)])          # the leaf only

    with pytest.raises(PermissionError):
        subject.rename(subject.with_name("sub-new"))


@windows_only
def test_released_watches_let_the_folder_move(qtbot, tmp_path):
    """The fix, through the real context manager."""
    leaf = _tree(tmp_path)
    subject = leaf.parent.parent

    host = QWidget()
    host.show()                            # top-level, so live_watchers finds it
    qtbot.addWidget(host)
    watcher = QFileSystemWatcher(host)
    watcher.addPaths([str(leaf), str(leaf.parent), str(subject)])
    assert watcher in live_watchers()

    target = subject.with_name("sub-new")
    with watchers_released():
        subject.rename(target)

    assert target.is_dir()
    assert not subject.exists(), "no empty original left behind"


@windows_only
def test_the_block_restores_what_still_exists(qtbot, tmp_path):
    """Windows contract: released inside, restored after, minus the dead.

    Re-adding a path that no longer exists is how a watcher accumulates dead
    entries and starts logging about them, and the normal case here is that
    some of the released paths were just renamed away.

    Windows-only, and it has to be. Off Windows ``watchers_released`` is a
    deliberate no-op, so nothing is released and nothing is restored, and
    "restored, minus the dead" is not a claim that can hold. This test used to
    run everywhere and failed on macOS for that reason: with the block doing
    nothing, ``QFileSystemWatcher`` simply kept the rmdir'd path in
    ``directories()``. See ``test_off_windows_the_block_leaves_watches_alone``
    for the contract that DOES hold there.
    """
    going = tmp_path / "sub-old" / "ses-pre" / "eeg"
    going.mkdir(parents=True)
    keeper = tmp_path / "keep-me"
    keeper.mkdir()

    host = QWidget()
    host.show()
    qtbot.addWidget(host)
    watcher = QFileSystemWatcher(host)
    watcher.addPaths([str(going), str(keeper)])

    with watchers_released():
        assert watcher.directories() == [], "watches must be off inside"
        going.rmdir()                      # this one goes away

    restored = set(watcher.directories())
    assert str(keeper) in restored
    assert str(going) not in restored


@pytest.mark.skipif(
    os.name == "nt", reason="Windows releases the watches; that is tested above",
)
def test_off_windows_the_block_leaves_watches_alone(qtbot, tmp_path):
    """The POSIX contract, which is that the block changes nothing.

    inotify and FSEvents watch an inode without holding it open, so a watched
    directory renames fine and there is nothing to release. Dropping the
    watches anyway would mean rebuilding watches the platform never needed
    dropped, so the block returns immediately. Asserted rather than assumed,
    because "no-op" is a promise the rename path relies on.
    """
    watched = tmp_path / "sub-old" / "ses-pre" / "eeg"
    watched.mkdir(parents=True)

    host = QWidget()
    host.show()
    qtbot.addWidget(host)
    watcher = QFileSystemWatcher(host)
    watcher.addPaths([str(watched)])

    with watchers_released():
        assert watcher.directories() == [str(watched)], "nothing to release"
    assert watcher.directories() == [str(watched)], "and nothing to restore"


def test_a_watched_directory_renames_fine_off_windows(qtbot, tmp_path):
    """The reason the block is Windows-only, stated as a test.

    Runs everywhere: on Windows it passes because the rename happens inside
    the block, and off Windows because the platform never objected.
    """
    leaf = _tree(tmp_path)
    subject = leaf.parent.parent

    host = QWidget()
    qtbot.addWidget(host)
    watcher = QFileSystemWatcher(host)
    watcher.addPaths([str(leaf), str(subject)])

    with watchers_released():
        subject.rename(subject.with_name("sub-new"))

    assert not subject.exists(), "no empty original left behind"
    assert subject.with_name("sub-new").is_dir()


def test_it_is_a_no_op_without_an_application(monkeypatch):
    """Never raises, whatever the GUI state — a rename must not depend on it."""
    monkeypatch.setattr(
        "bidsmgr.gui.fs_watch.QApplication.instance", staticmethod(lambda: None),
    )
    assert live_watchers() == []
    with watchers_released():
        pass
