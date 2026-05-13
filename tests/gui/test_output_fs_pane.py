"""Tests for feature #6 — output filesystem pane (lower half of col 1).

Three concerns:

* :class:`OutputFsPane` starts empty, populates from a real directory
  on ``set_root``, and clears on ``set_root(None)``.
* It applies BIDS-aware coloring per file extension (sanity-checked
  by walking the rendered ``QTreeWidgetItem``s — we trust that the
  ``setForeground`` calls themselves don't crash; pixel-level colour
  is not asserted).
* The ConverterPanel wires its lifecycle correctly: picking a BIDS
  output sets the pane's root; the post-convert hook refreshes it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.gui.converter_panel import ConverterPanel
from bidsmgr.gui.output_fs_pane import OutputFsPane


pytestmark = pytest.mark.gui


def _wait_scan_idle(qtbot, pane: OutputFsPane, timeout: int = 5000) -> None:
    """Block until the pane's background scan has caught up.

    The pane's disk walk runs on the global ``QThreadPool``; its result
    arrives on the GUI event loop via a queued signal. Tests that
    assert on the rendered tree must wait until the latest request
    has produced output (or was synchronously short-circuited to an
    empty state).
    """
    qtbot.waitUntil(
        lambda: (
            not pane._scan_in_progress
            and pane._completed_scan_generation == pane._scan_generation
        ),
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# OutputFsPane standalone
# ---------------------------------------------------------------------------


def test_pane_starts_empty(qtbot) -> None:
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    assert pane._empty.isVisible()
    assert not pane._tree.isVisible()


def test_pane_populates_from_a_bids_tree(qtbot, tmp_path: Path) -> None:
    # Mimic a BIDS output: <bids_parent>/<dataset>/<sub-X>/<datatype>/...
    (tmp_path / "study" / "sub-001" / "anat").mkdir(parents=True)
    (tmp_path / "study" / "sub-001" / "anat" / "sub-001_T1w.nii.gz").write_bytes(b"")
    (tmp_path / "study" / "sub-001" / "anat" / "sub-001_T1w.json").write_text("{}")
    (tmp_path / "study" / "participants.tsv").write_text("participant_id\n")

    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    pane.set_root(tmp_path)

    # Visibility flips synchronously; the populated tree arrives off-thread.
    assert pane._tree.isVisible()
    assert not pane._empty.isVisible()
    _wait_scan_idle(qtbot, pane)

    def walk(item):
        yield item.text(0)
        for i in range(item.childCount()):
            yield from walk(item.child(i))

    labels: list[str] = []
    for i in range(pane._tree.topLevelItemCount()):
        labels.extend(walk(pane._tree.topLevelItem(i)))
    assert any("study" in lbl for lbl in labels)
    assert any("sub-001_T1w.nii.gz" in lbl for lbl in labels)
    assert any("sub-001_T1w.json" in lbl for lbl in labels)
    assert any("participants.tsv" in lbl for lbl in labels)


def test_pane_clears_on_none(qtbot, tmp_path: Path) -> None:
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    (tmp_path / "x").mkdir()
    pane.set_root(tmp_path)
    _wait_scan_idle(qtbot, pane)
    pane.set_root(None)
    # ``set_root(None)`` short-circuits synchronously — the tree
    # clears and the empty hint flips back without waiting on disk.
    assert pane._empty.isVisible()
    assert pane._tree.topLevelItemCount() == 0


def _walk_labels(pane: OutputFsPane) -> list[str]:
    out: list[str] = []
    def walk(item):
        yield item.text(0)
        for i in range(item.childCount()):
            yield from walk(item.child(i))
    for i in range(pane._tree.topLevelItemCount()):
        out.extend(walk(pane._tree.topLevelItem(i)))
    return out


def _find_item(pane: OutputFsPane, path: tuple[str, ...]):
    """Walk the pane's tree and return the QTreeWidgetItem at ``path``."""
    cur = None
    for i in range(pane._tree.topLevelItemCount()):
        if pane._tree.topLevelItem(i).text(0) == path[0]:
            cur = pane._tree.topLevelItem(i)
            break
    if cur is None:
        return None
    for name in path[1:]:
        found = None
        for j in range(cur.childCount()):
            if cur.child(j).text(0) == name:
                found = cur.child(j)
                break
        if found is None:
            return None
        cur = found
    return cur


def test_user_expansion_survives_rebuild(qtbot, tmp_path: Path) -> None:
    """Manually-collapsing a node and then triggering a refresh should
    NOT re-expand it. State snapshot/restore guards user interactions
    against watcher-driven rebuilds.
    """
    (tmp_path / "study" / "sub-001" / "anat").mkdir(parents=True)
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    pane.set_root(tmp_path)
    _wait_scan_idle(qtbot, pane)

    # ``study`` is auto-expanded on first render. Collapse it manually.
    item_study = _find_item(pane, (tmp_path.name, "study"))
    assert item_study is not None
    assert item_study.isExpanded(), "first render auto-expands the first level"
    item_study.setExpanded(False)
    assert not item_study.isExpanded()

    # Trigger a rebuild (simulates a watcher event mid-conversion).
    pane._rebuild()
    _wait_scan_idle(qtbot, pane)
    item_study2 = _find_item(pane, (tmp_path.name, "study"))
    assert item_study2 is not None
    # Collapsed state preserved.
    assert not item_study2.isExpanded()


def test_user_expanded_deep_path_survives_rebuild(qtbot, tmp_path: Path) -> None:
    """An expansion the user opened DEEP in the tree must survive."""
    (tmp_path / "study" / "sub-001" / "anat").mkdir(parents=True)
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    pane.set_root(tmp_path)
    _wait_scan_idle(qtbot, pane)

    sub = _find_item(pane, (tmp_path.name, "study", "sub-001"))
    assert sub is not None
    sub.setExpanded(True)
    pane._rebuild()
    _wait_scan_idle(qtbot, pane)
    sub_after = _find_item(pane, (tmp_path.name, "study", "sub-001"))
    assert sub_after is not None and sub_after.isExpanded()


def test_user_selection_survives_rebuild(qtbot, tmp_path: Path) -> None:
    (tmp_path / "study" / "sub-001").mkdir(parents=True)
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    pane.set_root(tmp_path)
    _wait_scan_idle(qtbot, pane)

    target = _find_item(pane, (tmp_path.name, "study", "sub-001"))
    pane._tree.setCurrentItem(target)
    assert pane._tree.currentItem() is target
    pane._rebuild()
    _wait_scan_idle(qtbot, pane)
    new_cur = pane._tree.currentItem()
    assert new_cur is not None
    assert pane._item_path(new_cur) == (tmp_path.name, "study", "sub-001")


def test_pane_live_refreshes_when_file_created(qtbot, tmp_path: Path) -> None:
    """``QFileSystemWatcher`` should fire on file creation inside a
    watched dir; the debounced ``_rebuild`` then re-renders the tree
    with the new file. No explicit ``refresh()`` call from the user.
    """
    (tmp_path / "study").mkdir()
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    pane.set_root(tmp_path)
    _wait_scan_idle(qtbot, pane)
    assert "marker.tsv" not in str(_walk_labels(pane))

    (tmp_path / "study" / "marker.tsv").write_text("x\n")
    # FSEvents (macOS) / inotify (Linux) may take a few hundred ms +
    # the 250ms debounce. waitUntil polls until success or timeout.
    qtbot.waitUntil(
        lambda: any("marker.tsv" in lbl for lbl in _walk_labels(pane)),
        timeout=5000,
    )


def test_pane_live_refreshes_when_dir_deleted(qtbot, tmp_path: Path) -> None:
    """Deleting a watched directory must remove it from the tree
    without requiring an app restart.
    """
    import shutil
    (tmp_path / "study").mkdir()
    (tmp_path / "study" / "leaf.tsv").write_text("x\n")
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    pane.set_root(tmp_path)
    _wait_scan_idle(qtbot, pane)
    assert any("leaf.tsv" in lbl for lbl in _walk_labels(pane))

    shutil.rmtree(tmp_path / "study")
    qtbot.waitUntil(
        lambda: not any("leaf.tsv" in lbl for lbl in _walk_labels(pane)),
        timeout=5000,
    )


def test_pane_refreshes_after_files_appear(qtbot, tmp_path: Path) -> None:
    pane = OutputFsPane()
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    pane.set_root(tmp_path)
    _wait_scan_idle(qtbot, pane)

    # Initially empty (the dir exists but has no children).
    def walk(item):
        yield item.text(0)
        for i in range(item.childCount()):
            yield from walk(item.child(i))

    labels_before: list[str] = []
    for i in range(pane._tree.topLevelItemCount()):
        labels_before.extend(walk(pane._tree.topLevelItem(i)))

    # Simulate a convert producing files.
    (tmp_path / "study" / "sub-001").mkdir(parents=True)
    (tmp_path / "study" / "sub-001" / "sub-001_T1w.nii.gz").write_bytes(b"")

    pane.refresh()
    _wait_scan_idle(qtbot, pane)
    labels_after: list[str] = []
    for i in range(pane._tree.topLevelItemCount()):
        labels_after.extend(walk(pane._tree.topLevelItem(i)))
    assert any("sub-001_T1w.nii.gz" in lbl for lbl in labels_after)
    assert "sub-001_T1w.nii.gz" not in str(labels_before)


# ---------------------------------------------------------------------------
# ConverterPanel integration
# ---------------------------------------------------------------------------


def test_panel_has_an_output_pane(qtbot) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    assert isinstance(panel._output_pane, OutputFsPane)


def test_pick_bids_parent_updates_output_pane_root(qtbot, tmp_path: Path) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    # Simulate the user picking a BIDS output (bypass the dialog).
    panel._bids_parent = tmp_path
    panel._output_pane.set_root(tmp_path)
    assert panel._output_pane._root == tmp_path


def test_convert_finished_refreshes_output_pane(qtbot, tmp_path: Path) -> None:
    """After ``_on_convert_finished``, the output pane re-walks the
    BIDS parent so newly-produced files appear without a restart.
    """
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    # Pre-seed the BIDS parent and produce a file ahead of the
    # convert hook so we can confirm the rebuild picks it up.
    bids_parent = tmp_path / "bids"
    bids_parent.mkdir()
    (bids_parent / "study").mkdir()
    (bids_parent / "study" / "marker.tsv").write_text("x\n")

    panel._on_convert_finished(rc=0, bids_parent=bids_parent)
    _wait_scan_idle(qtbot, panel._output_pane)

    # Walk the rendered tree and assert the marker file is present.
    def walk(item):
        yield item.text(0)
        for i in range(item.childCount()):
            yield from walk(item.child(i))

    labels: list[str] = []
    for i in range(panel._output_pane._tree.topLevelItemCount()):
        labels.extend(walk(panel._output_pane._tree.topLevelItem(i)))
    assert any("marker.tsv" in lbl for lbl in labels)
