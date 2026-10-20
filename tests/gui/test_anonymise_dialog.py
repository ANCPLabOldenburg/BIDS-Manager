"""Tools, Anonymise metadata: what by purpose, where by subject and session,
every change shown, then one step of the history (undone and redone as one)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from PyQt6.QtCore import QItemSelectionModel, Qt
from PyQt6.QtWidgets import QMessageBox

from bidsmgr.gui.anonymise_dialog import AnonymiseDialog
from bidsmgr.project.operations import ORIGINALS_DIR, peek_undo, redo_last, undo_last

pytestmark = pytest.mark.gui


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    for sub, when in (("sub-01", "2025-05-26T11:14:30"), ("sub-02", "2025-06-01T09:00:00")):
        anat = root / sub / "ses-1" / "anat"
        anat.mkdir(parents=True)
        (anat / f"{sub}_ses-1_T1w.json").write_text(json.dumps({
            "AcquisitionDateTime": when, "PatientName": f"Doe^{sub}",
            "InstitutionAddress": "Main Street 1", "DeviceSerialNumber": "66001",
            "ProtocolName": "t1 Jane", "EchoTime": 0.003}))
    (root / "participants.tsv").write_text("participant_id\tgiven_name\tage\n"
                                           "sub-01\tJane\t93\nsub-02\tJoe\t30\n")
    (root / "dataset_description.json").write_text('{"Name": "d", "BIDSVersion": "1.10.0"}')
    (root / "sourcedata").mkdir()
    return root


def _settled(dlg: AnonymiseDialog) -> bool:
    return not dlg.busy() and not dlg._settle.isActive() and dlg._plan is not None


def _open(qtbot, root: Path) -> AnonymiseDialog:
    dlg = AnonymiseDialog(root)
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitUntil(lambda: _settled(dlg), timeout=20_000)
    return dlg


def _wait(qtbot, dlg: AnonymiseDialog) -> None:
    qtbot.waitUntil(lambda: _settled(dlg), timeout=20_000)


def _yes(monkeypatch) -> list:
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: asked.append(a[2]) or QMessageBox.StandardButton.Ok)
    return asked


def _titles(dlg: AnonymiseDialog) -> list[str]:
    model = dlg._list.model
    return [model.group_at(model.index(i, 0)).title for i in range(model.rowCount())]


def _files(dlg: AnonymiseDialog) -> set[str]:
    return {dlg._plan.rel(f.path) for f in dlg._plan.files}


def test_every_change_is_shown_by_kind_before_anything_is_written(qtbot, ds: Path) -> None:
    before = {p: p.read_bytes() for p in ds.rglob("*") if p.is_file()}
    dlg = _open(qtbot, ds)
    assert _titles(dlg)[:4] == ["Names and IDs", "Ages", "Dates and times", "The site"]
    assert dlg._cards["identity"].count.text() == "3", "2 PatientName, 1 column"
    assert dlg._cards["photos"].count.text() == "0"
    assert dlg._stat_files.number.text() == "3" and dlg._ok.text() == "Anonymise 3 files"
    assert dlg._preset.buttons["recommended"].isChecked()
    assert {p: p.read_bytes() for p in before} == before, "nothing written"


def test_the_other_views(qtbot, ds: Path) -> None:
    dlg = _open(qtbot, ds)
    dlg._view.buttons["file"].click()
    assert set(_titles(dlg)) == {"participants.tsv", "sub-01/ses-1/anat/sub-01_ses-1_T1w.json",
                                 "sub-02/ses-1/anat/sub-02_ses-1_T1w.json"}
    dlg._view.buttons["review"].click()
    assert _titles(dlg) == ["ProtocolName"] and dlg._view.buttons["review"].text() == \
        "To read by hand (1)"
    dlg._view.buttons["cannot"].click()
    assert _titles(dlg) == ["Left as they are"]
    row = dlg._list.model.row_at(dlg._list.model.index(0, 0, dlg._list.model.index(0, 0)))
    assert row.what == "sourcedata/"


def test_a_preset_sets_the_categories_and_any_change_makes_it_custom(qtbot, ds: Path) -> None:
    dlg = _open(qtbot, ds)
    dlg._preset.buttons["strict"].click()
    _wait(qtbot, dlg)
    assert dlg._cards["dates"].options.buttons["midnight"].isChecked()
    assert dlg._cards["site"].options.buttons["all"].isChecked()
    assert dlg.policy().name == "strict"
    dlg._cards["identity"].box.setChecked(False)
    _wait(qtbot, dlg)
    assert dlg._preset.buttons["custom"].isChecked() and dlg.policy().name == "custom"
    assert dlg._cards["identity"].count.text() == "off"
    assert "Names and IDs" not in _titles(dlg)
    assert not any(c.category == "identity" for f in dlg._plan.files for c in f.changes)
    dlg._cards["identity"].box.setChecked(True)
    _wait(qtbot, dlg)
    assert dlg._preset.buttons["strict"].isChecked(), "back to the preset it matches"


def test_an_option_replans(qtbot, ds: Path) -> None:
    dlg = _open(qtbot, ds)
    dlg._cards["dates"].options.buttons["midnight"].click()
    _wait(qtbot, dlg)
    assert dlg.policy().times == "midnight"
    afters = [c.after for f in dlg._plan.files for c in f.changes if c.category == "dates"]
    assert afters and all("T00:00:00" in a for a in afters if "T" in a)


def test_the_scope_limits_what_changes(qtbot, ds: Path) -> None:
    dlg = _open(qtbot, ds)
    tree = dlg._scope.tree
    sub02 = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
                 if tree.topLevelItem(i).text(0) == "sub-02")
    sub02.setCheckState(0, Qt.CheckState.Unchecked)
    _wait(qtbot, dlg)
    assert dlg._where.text() == "Anonymising 1 subject and the dataset files."
    assert _files(dlg) == {"participants.tsv", "sub-01/ses-1/anat/sub-01_ses-1_T1w.json"}
    whats = {c.what for f in dlg._plan.files for c in f.changes}
    assert "given_name of sub-01" in whats, "a row blanked, the column kept for sub-02"
    dlg._scope._set_all(False)
    qtbot.waitUntil(lambda: not dlg.busy() and not dlg._settle.isActive(), timeout=20_000)
    assert dlg._plan is None
    assert dlg._where.text().startswith("Nothing ticked") and not dlg._ok.isEnabled()


def test_a_file_left_out_is_not_written(qtbot, ds: Path, monkeypatch) -> None:
    _yes(monkeypatch)
    dlg = _open(qtbot, ds)
    dlg._view.buttons["file"].click()
    model = dlg._list.model
    target = "sub-02/ses-1/anat/sub-02_ses-1_T1w.json"
    row = next(i for i in range(model.rowCount())
               if model.group_at(model.index(i, 0)).key == target)
    dlg._list.view.selectionModel().select(
        model.index(row, 0), QItemSelectionModel.SelectionFlag.ClearAndSelect)
    qtbot.keyClick(dlg._list.view, Qt.Key.Key_Space)
    assert target in model.left_out and dlg._ok.text() == "Anonymise 2 files"
    assert "1 left out" in dlg._status.text()
    original = (ds / target).read_bytes()
    dlg._on_apply()
    qtbot.waitUntil(lambda: dlg.written() > 0, timeout=20_000)
    assert (ds / target).read_bytes() == original


def test_anonymising_is_one_step_undone_and_redone(qtbot, ds: Path, monkeypatch) -> None:
    asked = _yes(monkeypatch)
    before = {p: p.read_bytes() for p in ds.rglob("*")
              if p.is_file() and ".bidsmgr" not in p.parts}
    dlg = _open(qtbot, ds)
    dlg._on_apply()
    qtbot.waitUntil(lambda: dlg.written() > 0, timeout=20_000)
    assert "Undo takes it back, Redo makes it again" in asked[0]
    assert peek_undo(ds).label == "Anonymise metadata (Recommended)"
    after = {p: p.read_bytes() for p in ds.rglob("*")
             if p.is_file() and ".bidsmgr" not in p.parts}
    side = json.loads((ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.json").read_text())
    assert "PatientName" not in side and side["AcquisitionDateTime"] < "1926"
    undo_last(ds)
    assert {p: p.read_bytes() for p in before} == before
    redo_last(ds)
    assert {p: p.read_bytes() for p in after} == after


def test_a_scoped_custom_run_says_so_in_the_history(qtbot, ds: Path, monkeypatch) -> None:
    _yes(monkeypatch)
    dlg = _open(qtbot, ds)
    dlg._cards["uids"].box.setChecked(False)
    tree = dlg._scope.tree
    next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
         if tree.topLevelItem(i).text(0) == "sub-02").setCheckState(0, Qt.CheckState.Unchecked)
    _wait(qtbot, dlg)
    dlg._on_apply()
    qtbot.waitUntil(lambda: dlg.written() > 0, timeout=20_000)
    assert peek_undo(ds).label == ("Anonymise metadata (Custom, 1 subject and the "
                                   "dataset files)")


def test_forgetting_the_originals_asks_and_deletes_them(qtbot, ds: Path, monkeypatch) -> None:
    asked = _yes(monkeypatch)
    dlg = _open(qtbot, ds)
    dlg._forget.setChecked(True)
    dlg._on_apply()
    qtbot.waitUntil(lambda: dlg.written() > 0, timeout=20_000)
    assert "DELETED" in asked[0]
    assert not (ds / ORIGINALS_DIR).exists()
    assert peek_undo(ds) is None, "nothing left to undo, as it said"


def test_a_long_list_stays_fast(qtbot, tmp_path: Path) -> None:
    root = tmp_path / "big"
    for s in range(60):
        anat = root / f"sub-{s:03d}" / "anat"
        anat.mkdir(parents=True)
        for r in range(40):
            (anat / f"sub-{s:03d}_run-{r:02d}_T1w.json").write_text(json.dumps({
                "AcquisitionDateTime": "2025-05-26T11:14:30", "PatientName": "X",
                "DeviceSerialNumber": "1", "SeriesInstanceUID": f"1.2.{s}.{r}"}))
    (root / "dataset_description.json").write_text('{"Name": "d", "BIDSVersion": "1.10.0"}')
    dlg = AnonymiseDialog(root)
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitUntil(lambda: _settled(dlg), timeout=60_000)
    assert dlg._plan.n_changes >= 4 * 2400
    start = time.perf_counter()
    dlg._view.buttons["file"].click()
    dlg._view.buttons["kind"].click()
    switched = time.perf_counter() - start
    assert switched < 1.5, switched


def test_the_editor_offers_it_with_the_face_tools(qtbot) -> None:
    from bidsmgr.gui.editor_panel import EditorPanel

    panel = EditorPanel()
    qtbot.addWidget(panel)
    labels = [a.text() for a in panel._tools_menu.actions()]
    assert labels.index("Anonymise metadata...") == labels.index("Put the face back...") + 1


def test_it_opens_on_what_was_right_clicked(qtbot, ds: Path, monkeypatch) -> None:
    from PyQt6.QtWidgets import QMenu

    from bidsmgr.anonymise import Scope
    from bidsmgr.gui.editor_panel import EditorPanel
    from bidsmgr.gui.widgets.bids_tree_pane import PATH_ROLE

    scope = Scope(subjects=frozenset({"sub-02"}))
    dlg = AnonymiseDialog(ds, scope=scope)
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitUntil(lambda: _settled(dlg), timeout=20_000)
    assert dlg.scope() == scope
    assert _files(dlg) == {"participants.tsv", "sub-02/ses-1/anat/sub-02_ses-1_T1w.json"}

    panel = EditorPanel()
    qtbot.addWidget(panel)
    panel._tree_pane.set_root(ds)
    tree = panel._tree_pane
    item = next(tree._tree.topLevelItem(0).child(i)
                for i in range(tree._tree.topLevelItem(0).childCount())
                if tree._tree.topLevelItem(0).child(i).data(0, PATH_ROLE).endswith("sub-02"))
    shown: list[QMenu] = []
    monkeypatch.setattr(QMenu, "exec", lambda self, *a, **k: shown.append(self))
    opened = []

    def fake_exec(dialog):              # the Editor opens it modally
        opened.append(dialog._initial_scope)
        dialog.deleteLater()
        return 0

    monkeypatch.setattr(AnonymiseDialog, "exec", fake_exec)
    tree._tree.setCurrentItem(item)
    tree._on_show_context_menu(tree._tree.visualItemRect(item).center())
    next(a for a in shown[-1].actions() if a.text() == "Anonymise metadata...").trigger()
    assert opened == [Scope(subjects=frozenset({"sub-02"}))]
