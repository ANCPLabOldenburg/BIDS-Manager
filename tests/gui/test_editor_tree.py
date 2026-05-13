"""Tests for the BIDS tree pane + EditorPanel open-root flow (M6 Step 2).

Builds a tiny BIDS-shaped tree on disk and asserts:

* The tree is populated when ``set_root`` is called, with rows in the
  expected nesting and per-row palette tokens.
* Hidden folders (``.bidsmgr``) and junk dirs (``.tmp_bidsmgr``) are skipped.
* Folder-shaped recordings (``.ds``, ``.mff``) appear as leaves —
  their contents are NOT recursed into.
* ``EditorPanel._set_root`` updates the path bar, stores the root via
  :class:`AppSettings`, and re-loads it on a fresh panel.
* The empty-state hint is visible before any root is loaded; the tree
  takes over once a root is opened.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bidsmgr.gui.app_settings import AppSettings
from bidsmgr.gui.editor_panel import EditorPanel
from bidsmgr.gui.widgets.bids_tree_pane import (
    COLOR_TOKEN_ROLE,
    PATH_ROLE,
    BidsTreePane,
)


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bids_root(tmp_path: Path) -> Path:
    """Build a small but realistic BIDS-shaped tree on disk."""
    root = tmp_path / "Studyname"
    # MRI side.
    anat = root / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_ses-01_T1w.nii.gz").write_bytes(b"")
    (anat / "sub-01_ses-01_T1w.json").write_text("{}")
    (root / "participants.tsv").write_text("participant_id\nsub-01\n")
    # MEG folder-recording (CTF). Contents should not appear in the tree.
    meg = root / "sub-01" / "ses-01" / "meg"
    meg.mkdir(parents=True)
    ds = meg / "sub-01_ses-01_task-rest_meg.ds"
    ds.mkdir()
    (ds / "marker.txt").write_text("inside the .ds folder")
    # Junk that must be skipped.
    (root / ".bidsmgr").mkdir()
    (root / ".bidsmgr" / "events.jsonl").write_text("")
    (root / ".tmp_bidsmgr").mkdir()
    return root


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_items(tree) -> list[tuple[int, str, str | None]]:
    """Return (depth, label, color_token) for every visible row."""
    out: list[tuple[int, str, str | None]] = []

    def visit(item, depth: int) -> None:
        out.append((
            depth,
            item.text(0),
            item.data(0, COLOR_TOKEN_ROLE),
        ))
        for i in range(item.childCount()):
            visit(item.child(i), depth + 1)

    for i in range(tree.topLevelItemCount()):
        visit(tree.topLevelItem(i), 0)
    return out


# ---------------------------------------------------------------------------
# BidsTreePane
# ---------------------------------------------------------------------------


def test_empty_state_hint_before_root(qapp) -> None:
    pane = BidsTreePane()
    assert pane.root() is None
    assert pane._stack.currentIndex() == 0  # hint
    # Tree has no items yet.
    assert pane._tree.topLevelItemCount() == 0


def test_set_root_populates_tree(qapp, bids_root: Path) -> None:
    pane = BidsTreePane()
    pane.set_root(bids_root)
    qapp.processEvents()

    assert pane.root() == bids_root
    assert pane._stack.currentIndex() == 1  # tree visible
    assert pane._tree.topLevelItemCount() == 1

    items = _collect_items(pane._tree)
    labels = {label for _, label, _ in items}

    assert "Studyname" in labels
    assert "sub-01" in labels
    assert "ses-01" in labels
    assert "anat" in labels
    assert "sub-01_ses-01_T1w.nii.gz" in labels
    assert "sub-01_ses-01_T1w.json" in labels
    assert "participants.tsv" in labels


def test_hidden_and_junk_dirs_skipped(qapp, bids_root: Path) -> None:
    pane = BidsTreePane()
    pane.set_root(bids_root)
    labels = {label for _, label, _ in _collect_items(pane._tree)}
    assert ".bidsmgr" not in labels
    assert ".tmp_bidsmgr" not in labels


def test_folder_recording_collapses_to_leaf(qapp, bids_root: Path) -> None:
    pane = BidsTreePane()
    pane.set_root(bids_root)

    # Find the ``.ds`` directory in the tree.
    ds_items = []

    def find(item) -> None:
        if item.text(0).endswith(".ds"):
            ds_items.append(item)
        for i in range(item.childCount()):
            find(item.child(i))

    for i in range(pane._tree.topLevelItemCount()):
        find(pane._tree.topLevelItem(i))

    assert len(ds_items) == 1
    ds_item = ds_items[0]
    # The folder-recording has no children — we do NOT descend into it.
    assert ds_item.childCount() == 0
    # And it's colored like a recording (``text``), not like a dir.
    assert ds_item.data(0, COLOR_TOKEN_ROLE) == "text"


def test_color_tokens_match_kinds(qapp, bids_root: Path) -> None:
    pane = BidsTreePane()
    pane.set_root(bids_root)
    items = _collect_items(pane._tree)
    by_label = {label: token for _, label, token in items}
    # Subject / session / datatype directories are accent.
    assert by_label["sub-01"] == "accent"
    assert by_label["ses-01"] == "accent"
    assert by_label["anat"] == "accent"
    # File kinds.
    assert by_label["sub-01_ses-01_T1w.nii.gz"] == "text"
    assert by_label["sub-01_ses-01_T1w.json"] == "purple"
    assert by_label["participants.tsv"] == "teal"


def test_file_selected_signal_emits_path(qapp, bids_root: Path, qtbot) -> None:
    pane = BidsTreePane()
    pane.set_root(bids_root)

    # Locate the T1w.json item to click.
    target = None

    def find(item) -> None:
        nonlocal target
        if item.text(0) == "sub-01_ses-01_T1w.json":
            target = item
        for i in range(item.childCount()):
            find(item.child(i))

    for i in range(pane._tree.topLevelItemCount()):
        find(pane._tree.topLevelItem(i))
    assert target is not None

    with qtbot.waitSignal(pane.file_selected, timeout=500) as sig:
        pane._tree.setCurrentItem(target)

    emitted_path = sig.args[0]
    assert emitted_path.name == "sub-01_ses-01_T1w.json"
    assert str(emitted_path).startswith(str(bids_root))
    # And the PATH_ROLE matches.
    assert target.data(0, PATH_ROLE) == str(emitted_path)


def test_set_root_none_clears(qapp, bids_root: Path) -> None:
    pane = BidsTreePane()
    pane.set_root(bids_root)
    assert pane._stack.currentIndex() == 1
    pane.set_root(None)
    assert pane.root() is None
    assert pane._stack.currentIndex() == 0
    assert pane._tree.topLevelItemCount() == 0


# ---------------------------------------------------------------------------
# EditorPanel
# ---------------------------------------------------------------------------


def test_editor_panel_loads_persisted_root_on_construction(
    qapp, isolated_settings, bids_root: Path,
) -> None:
    AppSettings.remember_editor_bids_root(bids_root)

    panel = EditorPanel()
    qapp.processEvents()

    assert panel.current_root() == bids_root
    # The path bar reflects the root.
    assert str(bids_root) in panel._path_bar.value()


def test_editor_panel_set_root_persists(
    qapp, isolated_settings, bids_root: Path,
) -> None:
    panel = EditorPanel()
    assert panel.current_root() is None  # no setting yet

    # Drive the open flow directly (we don't want a real file dialog).
    panel._set_root(bids_root, persist=True)
    qapp.processEvents()

    assert panel.current_root() == bids_root
    assert AppSettings.load().editor_bids_root == str(bids_root)


def test_editor_panel_open_button_exists_and_enabled(qapp) -> None:
    panel = EditorPanel()
    assert panel._open_btn.isEnabled()
    # Validate buttons stay disabled — wired in later steps.
    assert not panel._validate_file_btn.isEnabled()
    assert not panel._validate_folder_btn.isEnabled()
    assert not panel._validate_dataset_btn.isEnabled()
