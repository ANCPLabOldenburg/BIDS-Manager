"""The Editor's Undo and Redo reach every change, one step each.

Before 1.5.1 the toolbar's Undo only walked back the open file's edits, one
at a time, and forgot them on the next click; a change made to many files at
once (a rename, a fix applied everywhere, a deletion) could not be undone
from the window at all, though the window said it could. Now: the open
file's unsaved edits first, then the dataset's history, one step per change
however many files it touched, with a question first when it touches several
or when one was changed since.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QMessageBox

from bidsmgr.gui.editor_panel import EditorPanel
from bidsmgr.project.operations import begin_operation, peek_redo, peek_undo

pytestmark = pytest.mark.gui


def _dataset(root: Path) -> Path:
    for sub in ("01", "02", "03"):
        folder = root / f"sub-{sub}" / "anat"
        folder.mkdir(parents=True)
        (folder / f"sub-{sub}_T1w.json").write_text(json.dumps({"RepetitionTime": 2.3}))
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Study", "BIDSVersion": "1.10.0"}))
    (root / "participants.tsv").write_text("participant_id\tage\nsub-01\t30\n")
    return root


def _sidecars(root: Path) -> list[Path]:
    return sorted(root.glob("sub-*/anat/*_T1w.json"))


def _fill_everywhere(root: Path, value: float) -> None:
    """What a Tools fix does: one change to many files."""
    with begin_operation(root, "Fill RepetitionTime in 3 files") as op:
        for path in _sidecars(root):
            op.write_json(path, {"RepetitionTime": value})


def _values(root: Path) -> list:
    return [json.loads(p.read_text())["RepetitionTime"] for p in _sidecars(root)]


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    return _dataset(tmp_path / "ds")


@pytest.fixture
def panel(qtbot, ds: Path) -> EditorPanel:
    p = EditorPanel()
    qtbot.addWidget(p)
    p.set_project_root(ds)
    p._sidecar_form.set_autosave(False)
    return p


@pytest.fixture
def confirmed(monkeypatch, panel: EditorPanel) -> list:
    """Answer the question a multi-file step asks with yes, and keep it."""
    asked = []
    monkeypatch.setattr(panel, "_confirm_step",
                        lambda which, step: asked.append((which, step)) or True)
    return asked


def test_a_change_to_many_files_is_one_undo_and_one_redo(panel, ds, confirmed) -> None:
    _fill_everywhere(ds, 9.9)
    panel._sync_undo_redo()
    assert panel._undo_btn.isEnabled()
    assert panel._undo_btn.toolTip().startswith("Undo: Fill RepetitionTime in 3 files (")
    assert "(3 files)" not in panel._undo_btn.toolTip(), "the label says it already"

    panel._undo_btn.click()
    assert _values(ds) == [2.3, 2.3, 2.3], "every file in one step"
    assert len(confirmed) == 1, "several files: asked first"
    assert panel._redo_btn.isEnabled()
    assert "Redo: Fill RepetitionTime in 3 files" in panel._redo_btn.toolTip()

    panel._redo_btn.click()
    assert _values(ds) == [9.9, 9.9, 9.9]
    assert not panel._redo_btn.isEnabled()


def test_several_changes_walk_back_one_step_each(panel, ds, confirmed) -> None:
    for value in (4.0, 5.0):
        _fill_everywhere(ds, value)
    panel._sync_undo_redo()
    panel._undo_btn.click()
    assert _values(ds) == [4.0, 4.0, 4.0]
    panel._undo_btn.click()
    assert _values(ds) == [2.3, 2.3, 2.3]
    assert not panel._undo_btn.isEnabled()


def test_a_file_changed_since_is_named_before_anything_is_replaced(
        panel, ds, monkeypatch) -> None:
    _fill_everywhere(ds, 9.9)
    edited = _sidecars(ds)[1]
    edited.write_text(json.dumps({"RepetitionTime": 7.0}))
    shown = {}

    def answer(box):
        shown["text"] = box.informativeText()
        shown["default"] = box.defaultButton().text()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", answer)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: box.defaultButton())
    panel._sync_undo_redo()
    panel._undo_btn.click()
    assert "1 of them changed after it" in shown["text"]
    assert edited.name in shown["text"]
    assert shown["default"] != "Undo", "with changes at stake, Cancel is the default"
    assert json.loads(edited.read_text())["RepetitionTime"] == 7.0, "cancelled: untouched"


def test_one_file_saved_is_undone_without_a_question(panel, ds, monkeypatch) -> None:
    monkeypatch.setattr(panel, "_confirm_step",
                        lambda *_a: pytest.fail("one file: no question"))
    path = _sidecars(ds)[0]
    with begin_operation(ds, f"Edit {path.name}") as op:
        op.write_json(path, {"RepetitionTime": 1.0})
    panel._sync_undo_redo()
    panel._undo_btn.click()
    assert json.loads(path.read_text())["RepetitionTime"] == 2.3


def test_a_saved_table_can_be_undone_after_looking_elsewhere(panel, ds, qtbot) -> None:
    table = ds / "participants.tsv"
    panel._on_file_selected(table)
    pane = panel._tsv_viewer
    qtbot.waitUntil(lambda: pane._model.rowCount() == 1, timeout=10_000)
    pane._model.setData(pane._model.index(0, 1), "31", Qt.ItemDataRole.EditRole)
    assert pane.save()
    assert "31" in table.read_text()
    panel._on_file_selected(_sidecars(ds)[0])
    panel._sync_undo_redo()
    assert "Undo: Edit participants.tsv" in panel._undo_btn.toolTip()
    panel._undo_btn.click()
    assert table.read_text() == "participant_id\tage\nsub-01\t30\n"


def test_unsaved_edits_come_first(panel, ds, confirmed) -> None:
    _fill_everywhere(ds, 9.9)
    path = _sidecars(ds)[0]
    panel._on_file_selected(path)
    form = panel._sidecar_form
    form._on_field_committed("RepetitionTime", 3.3, "number")
    assert form.is_dirty()
    assert "not saved yet" in panel._undo_btn.toolTip()
    panel._undo_btn.click()
    assert not form.is_dirty(), "the unsaved edit went back"
    assert _values(ds) == [9.9, 9.9, 9.9], "and the dataset step is still there"
    assert peek_undo(ds).label == "Fill RepetitionTime in 3 files"
    assert "Undo: Fill RepetitionTime" in panel._undo_btn.toolTip()


def test_after_undo_the_open_file_shows_what_is_on_disk(panel, ds, confirmed) -> None:
    path = _sidecars(ds)[0]
    _fill_everywhere(ds, 9.9)
    panel._on_file_selected(path)
    assert panel._sidecar_form._json_cache["RepetitionTime"] == 9.9
    panel._sync_undo_redo()
    panel._undo_btn.click()
    assert panel._sidecar_form._current_file == path
    assert panel._sidecar_form._json_cache["RepetitionTime"] == 2.3


def test_a_folder_open_for_viewing_has_no_history(qtbot, tmp_path) -> None:
    folder = tmp_path / "plain"
    folder.mkdir()
    (folder / "notes.txt").write_text("x")
    p = EditorPanel()
    qtbot.addWidget(p)
    assert p.open_folder(folder) == "view"
    p._sync_undo_redo()
    assert not p._undo_btn.isEnabled() and not p._redo_btn.isEnabled()


def test_the_undo_key_works_anywhere_in_the_editor(panel, ds, qtbot, confirmed) -> None:
    from PyQt6.QtTest import QTest

    _fill_everywhere(ds, 9.9)
    panel.show()
    panel.activateWindow()
    qtbot.waitActive(panel)              # shortcuts fire in the active window
    panel._tree_pane.setFocus()
    qtbot.wait(10)                       # let the focus settle before the key
    QTest.keySequence(panel._tree_pane, QKeySequence(QKeySequence.StandardKey.Undo))
    assert _values(ds) == [2.3, 2.3, 2.3]
    assert peek_redo(ds) is not None


def test_a_step_whose_label_has_no_count_gets_one(panel, ds, confirmed) -> None:
    with begin_operation(ds, "Rename subject 01 to 001") as op:
        for path in _sidecars(ds):
            op.write_json(path, {"RepetitionTime": 1.0})
    panel._sync_undo_redo()
    assert "Undo: Rename subject 01 to 001 (3 files)" in panel._undo_btn.toolTip()


# ---------------------------------------------------------------------------
# The history menu: right-click on Undo or Redo, like a browser's back button
# ---------------------------------------------------------------------------


def _three_steps(ds: Path) -> None:
    for value in (4.0, 5.0, 6.0):
        _fill_everywhere(ds, value)
        # Labels that say which step is which.
        log = ds / ".bidsmgr" / "editor" / "operations.log"
        lines = log.read_text().splitlines()
        record = json.loads(lines[-1])
        record["label"] = f"Set {value}"
        lines[-1] = json.dumps(record)
        log.write_text("\n".join(lines) + "\n")


def test_the_history_lists_every_step_newest_first(panel, ds) -> None:
    from PyQt6.QtWidgets import QMenu

    _three_steps(ds)
    menu = QMenu()
    actions = panel.history_menu_actions("undo", menu)
    assert [a.text().split("\t")[0] for a in actions] == [
        "Set 6.0 (3 files)", "Set 5.0 (3 files)", "Set 4.0 (3 files)"]
    assert all("\t" in a.text() for a in actions), "with the time on the right"


def test_picking_a_step_goes_back_to_it_in_one_question(panel, ds, monkeypatch) -> None:
    from PyQt6.QtWidgets import QMenu

    _three_steps(ds)
    asked = []
    monkeypatch.setattr(panel, "_confirm_steps",
                        lambda which, steps, changed: asked.append(len(steps)) or True)
    menu = QMenu()
    panel.history_menu_actions("undo", menu)[1].trigger()      # "Set 5.0"
    assert asked == [2]
    assert _values(ds) == [4.0, 4.0, 4.0], "back to before Set 5.0"
    redo_menu = QMenu()
    redo = panel.history_menu_actions("redo", redo_menu)
    assert [a.text().split("\t")[0] for a in redo] == ["Set 5.0 (3 files)", "Set 6.0 (3 files)"]
    redo[1].trigger()
    assert _values(ds) == [6.0, 6.0, 6.0], "forward again, both steps"


def test_the_history_waits_for_unsaved_edits(panel, ds) -> None:
    from PyQt6.QtWidgets import QMenu

    _three_steps(ds)
    panel._on_file_selected(_sidecars(ds)[0])
    panel._sidecar_form._on_field_committed("RepetitionTime", 1.0, "number")
    menu = QMenu()
    assert panel.history_menu_actions("undo", menu) == []
    assert menu.actions()[0].text().startswith("Save or revert")
    assert not menu.actions()[0].isEnabled()


def test_the_buttons_say_the_history_is_one_right_click_away(panel, ds) -> None:
    _fill_everywhere(ds, 9.9)
    panel._sync_undo_redo()
    assert "Right-click for the whole history" in panel._undo_btn.toolTip()
