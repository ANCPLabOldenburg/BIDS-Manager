"""The Editor's tree must not hand a watch to the converter's staging dir.

A watch is an open handle on the directory. Conversion stages each subject in
``<bids_root>/.tmp_bidsmgr/sub-<id>/`` and commits it with one ``os.rename``,
and on Windows a directory cannot be moved while any handle is open anywhere
inside it. So a watch landing there failed EVERY commit with ``WinError 5``,
for as long as the pane lived — not a race, a standing block.

It only happened with "Show hidden files" on, because ``is_hidden_name``
conflates two different things: dotfiles, which that setting is meant to
reveal, and the machinery directories of :data:`_SKIP_DIRS`, which the tree
has never wanted to expose. Turning the setting on therefore also started
sweeping — and watching — ``.tmp_bidsmgr``.

The fix withholds the handle, not the folder: the machinery is still drawn
when the user asks to see hidden files, it simply never gets a watch. These
tests pin both halves of that, since dropping the folder from the tree would
be a different (and unwanted) change.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bidsmgr.gui.widgets.bids_tree_pane import _sweep

pytestmark = pytest.mark.gui

windows_only = pytest.mark.skipif(
    os.name != "nt", reason="only Windows refuses to move a watched directory"
)


def _dataset(tmp_path: Path) -> Path:
    """A dataset with a committed subject, a staged one, and a .bidsmgr/."""
    root = tmp_path / "ds_19"
    (root / "sub-002" / "anat").mkdir(parents=True)
    (root / "sub-002" / "anat" / "sub-002_T1w.nii.gz").write_bytes(b"\0" * 64)
    staging = root / ".tmp_bidsmgr" / "sub-001"
    for datatype in ("anat", "func"):
        (staging / datatype).mkdir(parents=True)
        (staging / datatype / f"sub-001_{datatype}.nii.gz").write_bytes(b"\0" * 64)
    (root / ".bidsmgr" / "errors").mkdir(parents=True)
    (root / ".bidsmgr" / "errors" / "sub-001.json").write_text("{}")
    return root


def _swept(root: Path, *, show_hidden: bool) -> tuple[list[str], dict]:
    dirs: list[str] = [str(root)]
    counts: dict[str, tuple[int, int, int]] = {}
    _sweep(root, depth=0, dirs=dirs, counts=counts, show_hidden=show_hidden)
    return dirs, counts


def _rel(root: Path, paths) -> set[str]:
    return {Path(p).relative_to(root).as_posix() for p in paths}


@pytest.mark.parametrize("show_hidden", [True, False])
def test_no_watch_ever_lands_inside_the_staging_tree(tmp_path: Path, show_hidden: bool):
    root = _dataset(tmp_path)

    dirs, _ = _swept(root, show_hidden=show_hidden)

    inside = [d for d in dirs if ".tmp_bidsmgr" in d]
    assert inside == [], f"these would block every commit: {inside}"


@pytest.mark.parametrize("show_hidden", [True, False])
def test_the_machinery_directory_itself_is_not_watched(tmp_path: Path, show_hidden: bool):
    """``.bidsmgr/`` too: a watch there is an open handle for no live data."""
    root = _dataset(tmp_path)

    dirs, _ = _swept(root, show_hidden=show_hidden)

    assert not [d for d in dirs if ".bidsmgr" in Path(d).name]


def test_the_real_dataset_directories_are_still_watched(tmp_path: Path):
    """The fix must not cost the live refresh it exists for."""
    root = _dataset(tmp_path)

    dirs, _ = _swept(root, show_hidden=True)

    assert _rel(root, dirs) == {".", "sub-002", "sub-002/anat"}


def test_show_hidden_still_reveals_the_machinery_in_the_tree(tmp_path: Path):
    """Withhold the handle, not the folder: ``counts`` is what gets drawn."""
    root = _dataset(tmp_path)

    _, counts = _swept(root, show_hidden=True)

    drawn = _rel(root, counts)
    assert {".tmp_bidsmgr", ".tmp_bidsmgr/sub-001", ".bidsmgr", ".bidsmgr/errors"} <= drawn


def test_show_hidden_off_hides_the_machinery_as_before(tmp_path: Path):
    root = _dataset(tmp_path)

    _, counts = _swept(root, show_hidden=False)

    assert _rel(root, counts) == {".", "sub-002", "sub-002/anat"}


@windows_only
def test_the_commit_succeeds_with_the_pane_watching_everything_it_asked_for(
    tmp_path: Path, qtbot,
):
    """End to end: the swept watch set is live, and the commit still lands.

    This is the reported failure. Before the fix the same rename raised
    ``PermissionError: [WinError 5]`` here, every single time.
    """
    from PyQt6.QtCore import QFileSystemWatcher

    root = _dataset(tmp_path)
    dirs, _ = _swept(root, show_hidden=True)
    watcher = QFileSystemWatcher()
    watcher.addPaths(dirs)
    try:
        os.rename(root / ".tmp_bidsmgr" / "sub-001", root / "sub-001")
    finally:
        watcher.removePaths(watcher.directories())

    assert (root / "sub-001" / "anat" / "sub-001_anat.nii.gz").is_file()
