"""The ignore list and the tree's Ignore in validation: granular, written at
once as one Undo, and the same matcher as the validator."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from PyQt6.QtCore import QItemSelectionModel, Qt

from bidsmgr.gui.widgets.bidsignore_pane import BidsIgnorePane
from bidsmgr.project.operations import peek_undo, undo_last

pytestmark = pytest.mark.gui

FILES = [
    "sub-01/anat/sub-01_T1w.nii.gz",
    "sub-01/anat/sub-01_T1w.json",
    "sub-01/anat/notes.txt",
    "sub-02/anat/sub-02_T1w.nii.gz",
    "sub-02/anat/notes.txt",
    "extra/a.txt",
    "sourcedata/1.dcm",
]


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    for rel in FILES:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if rel.endswith(".json") else "x")
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    return root


def _pane(qtbot, root: Path) -> BidsIgnorePane:
    pane = BidsIgnorePane()
    qtbot.addWidget(pane)
    pane.resize(1100, 700)
    pane.show()
    pane.set_root(root)
    return pane


def _select(pane: BidsIgnorePane, *paths: str) -> None:
    model, sel = pane._model, pane._files.selectionModel()
    sel.clearSelection()
    for row in range(model.rowCount()):
        if model.path_at(row) in paths:
            sel.select(model.index(row, 0), QItemSelectionModel.SelectionFlag.Select
                       | QItemSelectionModel.SelectionFlag.Rows)


def _choices(pane: BidsIgnorePane, *, stop: bool = False) -> dict[str, object]:
    menu = pane._menu_for(pane._selected(), stop=stop)
    assert menu is not None
    out = {a.text().split("\t")[0]: a for a in menu.actions()}
    return out


def test_picking_a_file_ignores_that_file(qtbot, ds: Path) -> None:
    pane = _pane(qtbot, ds)
    assert pane._model.total() == 7, "sourcedata/ is named, not listed"
    _select(pane, "sub-01/anat/notes.txt")
    assert pane._ignore_btn.isEnabled() and not pane._stop_btn.isEnabled()
    choices = _choices(pane)
    assert list(choices)[:3] == ["This file", "The whole folder sub-01/anat/",
                                 "The same name in every subject"]
    with qtbot.waitSignal(pane.changed) as said:
        choices["This file"].trigger()
    assert "/sub-01/anat/notes.txt" in said.args[0]
    assert (ds / ".bidsignore").read_text() == "/sub-01/anat/notes.txt\n"
    assert pane._model.why("sub-01/anat/notes.txt") == "/sub-01/anat/notes.txt"
    assert pane._model.why("sub-01/anat/sub-01_T1w.nii.gz") is None, "not the subject"
    assert peek_undo(ds).label == "Ignore in validation: /sub-01/anat/notes.txt"


def test_stop_ignoring_removes_the_line(qtbot, ds: Path) -> None:
    (ds / ".bidsignore").write_text("*.txt\n")
    pane = _pane(qtbot, ds)
    _select(pane, "extra/a.txt")
    assert pane._stop_btn.isEnabled() and not pane._ignore_btn.isEnabled()
    choices = _choices(pane, stop=True)
    assert [t for t in choices if t] == ["Stop ignoring this file",
                                         "Remove *.txt, which hides 3 files"]
    with qtbot.waitSignal(pane.changed):
        choices["Remove *.txt, which hides 3 files"].trigger()
    assert (ds / ".bidsignore").read_text() == "\n"
    undo_last(ds)
    assert (ds / ".bidsignore").read_text() == "*.txt\n"


def test_several_files_stop_being_ignored_at_once(qtbot, ds: Path) -> None:
    pane = _pane(qtbot, ds)
    picked = ("sub-01/anat/notes.txt", "sub-02/anat/notes.txt", "extra/a.txt")
    _select(pane, *picked)
    _choices(pane)["These 3 files"].trigger()
    assert all(pane._model.why(p) for p in picked)
    _select(pane, *picked)
    choices = _choices(pane, stop=True)
    assert list(choices) == ["Stop ignoring these 3 files"]
    with qtbot.waitSignal(pane.changed) as said:
        choices["Stop ignoring these 3 files"].trigger()
    assert said.args[0].startswith("Validated again: 3 files")
    assert not any(pane._model.why(p) for p in picked)
    assert peek_undo(ds).label == "Stop ignoring 3 files"


def test_several_lines_are_removed_in_one_step(qtbot, ds: Path) -> None:
    (ds / ".bidsignore").write_text("*.txt\n/extra/\n/sub-02/\n")
    pane = _pane(qtbot, ds)
    pane._patterns.topLevelItem(0).setSelected(True)
    pane._patterns.topLevelItem(2).setSelected(True)
    assert pane._remove_btn.text() == "Remove 2 lines"
    assert pane._focus_label.text() == "What these 2 lines match: 4 files"
    pane._remove_btn.click()
    assert (ds / ".bidsignore").read_text() == "/extra/\n"
    undo_last(ds)
    assert (ds / ".bidsignore").read_text() == "*.txt\n/extra/\n/sub-02/\n"


def test_a_line_shows_what_it_hides(qtbot, ds: Path) -> None:
    (ds / ".bidsignore").write_text("# ours\n*.txt\n/nothing/\nsourcedata/\n")
    pane = _pane(qtbot, ds)
    rows = {pane._patterns.topLevelItem(i).text(0): pane._patterns.topLevelItem(i).text(1)
            for i in range(pane._patterns.topLevelItemCount())}
    assert rows == {"# ours": "", "*.txt": "3 files", "/nothing/": "matches nothing",
                    "sourcedata/": "already ignored"}
    pane._patterns.setCurrentItem(pane._patterns.topLevelItem(1))
    assert pane._focus_box.isVisible()
    assert [pane._model.path_at(r) for r in range(pane._model.rowCount())] == [
        "extra/a.txt", "sub-01/anat/notes.txt", "sub-02/anat/notes.txt"]
    pane._focus_clear.click()
    assert pane._model.rowCount() == 7


def test_the_selected_line_is_removed_by_the_button_and_the_key(qtbot, ds: Path) -> None:
    (ds / ".bidsignore").write_text("*.txt\n/extra/\n")
    pane = _pane(qtbot, ds)
    pane._patterns.setCurrentItem(pane._patterns.topLevelItem(0))
    pane._remove_btn.click()
    assert (ds / ".bidsignore").read_text() == "/extra/\n"
    pane._patterns.setFocus()
    pane._patterns.setCurrentItem(pane._patterns.topLevelItem(0))
    qtbot.keyClick(pane._patterns, Qt.Key.Key_Backspace)
    assert (ds / ".bidsignore").read_text() == "\n"


def test_a_typed_pattern_is_previewed_then_added(qtbot, ds: Path) -> None:
    pane = _pane(qtbot, ds)
    pane._new_pattern.setText("*.txt")
    assert pane._preview.text().startswith("Would hide 3 files")
    pane._new_pattern.setText("*.nope")
    assert "usually a typo" in pane._preview.text()
    pane._new_pattern.setText("/extra/")
    qtbot.keyClick(pane._new_pattern, Qt.Key.Key_Return)
    assert (ds / ".bidsignore").read_text() == "/extra/\n"
    assert pane._new_pattern.text() == ""


def test_focus_alone_does_not_filter_the_files(qtbot, ds: Path) -> None:
    """Focus makes the first line current without selecting it; that used to
    show only its files the moment the list opened."""
    (ds / ".bidsignore").write_text("/extra/\n")
    pane = _pane(qtbot, ds)
    pane._patterns.setCurrentItem(pane._patterns.topLevelItem(0),
                                  0, QItemSelectionModel.SelectionFlag.NoUpdate)
    assert not pane._focus_box.isVisible() and pane._model.rowCount() == 7


def test_the_files_filter_by_state_and_by_name(qtbot, ds: Path) -> None:
    (ds / ".bidsignore").write_text("/extra/\n")
    pane = _pane(qtbot, ds)
    pane._on_mode("ignored")
    assert [pane._model.path_at(r) for r in range(pane._model.rowCount())] == ["extra/a.txt"]
    pane._on_mode("validated")
    assert pane._model.rowCount() == 6
    pane._on_mode("all")
    pane._search.setText("notes")
    qtbot.waitUntil(lambda: pane._model.rowCount() == 2, timeout=2_000)
    assert pane._count.text() == "2 of 7 files"


def test_a_view_only_folder_shows_and_never_changes(qtbot, ds: Path) -> None:
    pane = _pane(qtbot, ds)
    pane.set_read_only(True)
    for w in (pane._ignore_btn, pane._stop_btn, pane._add_btn, pane._remove_btn,
              pane._new_pattern):
        assert not w.isVisible()
    assert not pane._add(["/extra/"]) and not (ds / ".bidsignore").exists()


def test_twenty_thousand_files_stay_fast(qtbot, tmp_path: Path) -> None:
    root = tmp_path / "big"
    for s in range(100):
        anat = root / f"sub-{s:03d}" / "anat"
        anat.mkdir(parents=True)
        for i in range(200):
            (anat / f"sub-{s:03d}_run-{i:03d}_T1w.json").write_text("")
    (root / ".bidsignore").write_text("*_run-00*\n")
    pane = BidsIgnorePane()
    qtbot.addWidget(pane)
    pane.show()
    start = time.perf_counter()
    pane.set_root(root)
    opened = time.perf_counter() - start
    assert pane._model.total() == 20_000
    start = time.perf_counter()
    pane._search.setText("sub-050")
    pane._debounce.stop()
    pane._apply_filter(needle="sub-050")
    searched = time.perf_counter() - start
    assert pane._model.rowCount() == 200
    assert opened < 3.0 and searched < 0.5, (opened, searched)


# -- the tree and the Editor ----------------------------------------------------


def _editor(qtbot, root: Path):
    from bidsmgr.gui.editor_panel import EditorPanel

    panel = EditorPanel()
    qtbot.addWidget(panel)
    panel.resize(1400, 900)
    panel.show()
    panel._tree_pane.set_root(root)
    return panel


def _unfold(tree) -> None:
    for _ in range(4):
        tree.expand_next_level()


def _item(tree, path: Path):
    from bidsmgr.gui.widgets.bids_tree_pane import PATH_ROLE

    found = []

    def visit(item):
        if item.data(0, PATH_ROLE) == str(path):
            found.append(item)
        for i in range(item.childCount()):
            visit(item.child(i))

    widget = tree._tree
    for i in range(widget.topLevelItemCount()):
        visit(widget.topLevelItem(i))
    return found[0] if found else None


def test_the_tree_dims_what_validation_ignores(qtbot, ds: Path) -> None:
    from bidsmgr.gui.widgets.bids_tree_pane import IGNORED_ROLE

    (ds / ".bidsignore").write_text("/extra/\n")
    panel = _editor(qtbot, ds)
    tree = panel._tree_pane
    _unfold(tree)
    extra = _item(tree, ds / "extra")
    assert extra.data(0, IGNORED_ROLE) == "/extra/" and extra.font(0).italic()
    assert "Not validated" in extra.toolTip(0)
    assert _item(tree, ds / "sourcedata").data(0, IGNORED_ROLE) == "sourcedata/"
    assert _item(tree, ds / "sub-01").data(0, IGNORED_ROLE) is None


def test_the_right_click_ignores_one_file_and_takes_it_back(qtbot, ds: Path,
                                                           monkeypatch) -> None:
    from PyQt6.QtWidgets import QMenu

    from bidsmgr.gui.widgets.bids_tree_pane import IGNORED_ROLE

    panel = _editor(qtbot, ds)
    tree = panel._tree_pane
    notes = ds / "sub-01" / "anat" / "notes.txt"
    _unfold(tree)
    item = _item(tree, notes)
    tree._tree.setCurrentItem(item)
    shown: list[QMenu] = []
    monkeypatch.setattr(QMenu, "exec", lambda self, *a, **k: shown.append(self))
    tree._on_show_context_menu(tree._tree.visualItemRect(item).center())
    menu = shown[-1]
    sub = next(a.menu() for a in menu.actions() if a.text() == "Ignore in validation")
    texts = [a.text() for a in sub.actions() if a.text()]
    assert texts[0] == "This file\t1 file"
    assert texts[-1] == "Ignored files..."
    sub.actions()[0].trigger()
    assert (ds / ".bidsignore").read_text() == "/sub-01/anat/notes.txt\n"
    _unfold(tree)
    assert _item(tree, notes).data(0, IGNORED_ROLE) == "/sub-01/anat/notes.txt"
    assert _item(tree, ds / "sub-01" / "anat" / "sub-01_T1w.json").data(0, IGNORED_ROLE) \
        is None, "only the file"

    shown.clear()
    item = _item(tree, notes)
    tree._tree.setCurrentItem(item)
    tree._on_show_context_menu(tree._tree.visualItemRect(item).center())
    stop = next(a for a in shown[-1].actions() if a.text() == "Stop ignoring this file")
    stop.trigger()
    assert (ds / ".bidsignore").read_text() == "\n"


def test_the_right_click_stops_ignoring_a_multi_selection(qtbot, ds: Path,
                                                         monkeypatch) -> None:
    from PyQt6.QtWidgets import QMenu

    from bidsmgr.gui.widgets.bids_tree_pane import IGNORED_ROLE

    (ds / ".bidsignore").write_text("/sub-01/anat/notes.txt\n/sub-02/anat/notes.txt\n")
    panel = _editor(qtbot, ds)
    tree = panel._tree_pane
    _unfold(tree)
    one = _item(tree, ds / "sub-01/anat/notes.txt")
    two = _item(tree, ds / "sub-02/anat/notes.txt")
    tree._tree.setCurrentItem(one)
    two.setSelected(True)
    assert len(tree.selected_paths()) == 2
    shown: list[QMenu] = []
    monkeypatch.setattr(QMenu, "exec", lambda self, *a, **k: shown.append(self))
    tree._on_show_context_menu(tree._tree.visualItemRect(two).center())
    stop = next(a for a in shown[-1].actions() if a.text() == "Stop ignoring these 2 files")
    stop.trigger()
    assert (ds / ".bidsignore").read_text() == "\n"
    _unfold(tree)
    assert _item(tree, ds / "sub-01/anat/notes.txt").data(0, IGNORED_ROLE) is None
    assert _item(tree, ds / "sub-02/anat/notes.txt").data(0, IGNORED_ROLE) is None


def test_tools_opens_the_list_without_hidden_files(qtbot, ds: Path) -> None:
    panel = _editor(qtbot, ds)
    labels = [a.text() for a in panel._tools_menu.actions()]
    assert labels.index("Ignored files...") == labels.index("Fix ups...") + 1
    panel._ignore_action.trigger()
    assert panel._center_stack.currentWidget() is panel._bidsignore_pane
    assert panel._bidsignore_pane._model.total() == 7
