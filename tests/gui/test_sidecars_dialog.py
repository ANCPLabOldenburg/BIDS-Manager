"""Sidecars from the recordings: the dialog, the tree's right-click and the
Editor's handler.

The engine has its own tests; these pin what a user meets. The plan is made on
a thread and shown recording by recording with where every field was read,
unticking a recording or a folder leaves it out, a right-click on a recording
with a sidecar opens on "add to existing", the right-click offers the words
that fit what was clicked, and writing is one Undo.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QMenu, QTreeWidgetItemIterator

from bidsmgr.editor.sidecars import sidecar_path
from bidsmgr.gui.sidecars_dialog import SidecarsDialog
from bidsmgr.gui.widgets.sidecar_views import short_name
from bidsmgr.project.operations import peek_undo, undo_last

from .conftest import open_every_folder

pytestmark = pytest.mark.gui


def _nifti(path: Path, shape=(4, 4, 3), tr=2.0, descrip=b"TE=30;Time=1;phase=1") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.int16), np.eye(4))
    img.header["pixdim"][4] = tr
    img.header.set_xyzt_units("mm", "sec")
    img.header["descrip"] = descrip
    nib.save(img, str(path))
    return path


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _nifti(root / "sub-01/ses-1/func/sub-01_ses-1_task-rest_bold.nii.gz", (4, 4, 3, 6), tr=0.8)
    _nifti(root / "sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz", tr=2.3)
    t2 = _nifti(root / "sub-01/ses-1/anat/sub-01_ses-1_T2w.nii.gz", tr=3.2)
    sidecar_path(t2).write_text(json.dumps({"RepetitionTime": 3.2}))
    return root


def _open(qtbot, root: Path, targets=None, **kwargs) -> SidecarsDialog:
    dlg = SidecarsDialog(root, targets, fill_scope="none", **kwargs)
    qtbot.addWidget(dlg)
    qtbot.waitUntil(lambda: not dlg.planning() and dlg.plan() is not None, timeout=30_000)
    return dlg


def _replanned(qtbot, dlg: SidecarsDialog, generation: int) -> None:
    """Wait for a plan newer than ``generation`` to land. Not for planning()
    to turn on: on a small dataset it turns on and off between two polls."""
    qtbot.waitUntil(lambda: dlg._generation > generation and not dlg.planning(),
                    timeout=30_000)


def test_the_missing_sidecars_are_shown_with_their_fields(qtbot, ds) -> None:
    dlg = _open(qtbot, ds)
    folders = {dlg._tree.topLevelItem(i).text(0) for i in range(dlg._tree.topLevelItemCount())}
    assert folders == {"sub-01/ses-1/func", "sub-01/ses-1/anat"}
    names = {item.text(0): item.text(1) for item in dlg._tree.recordings()}
    assert names == {"task-rest_bold.nii.gz": "3, new", "T1w.nii.gz": "2, new"}
    assert dlg._stat_new.number.text() == "2"
    assert dlg._stat_fields.number.text() == "5"
    assert dlg._ok.text() == "Write 2 sidecars"
    assert "1 recording here already has its sidecar" in dlg._status.text()

    # The selected recording's fields, each with where it was read.
    bold = next(i for i in dlg._tree.recordings() if i.text(0) == "task-rest_bold.nii.gz")
    dlg._tree.setCurrentItem(bold)
    bold.setSelected(True)
    rows = {}
    head = dlg._fields.topLevelItem(0)
    assert head.text(0).startswith("Written")
    for j in range(head.childCount()):
        rows[head.child(j).text(0)] = (head.child(j).text(1), head.child(j).text(2))
    assert rows["RepetitionTime"] == ("0.8", "File header")
    assert rows["TaskName"] == ("rest", "File name")
    assert dlg._detail_read.text() == "Read from the NIfTI header."
    assert dlg._fields.visual_rect_of("EchoTime") is not None


def test_unticked_recordings_are_not_written(qtbot, ds) -> None:
    dlg = _open(qtbot, ds)
    anat = next(dlg._tree.topLevelItem(i) for i in range(dlg._tree.topLevelItemCount())
                if dlg._tree.topLevelItem(i).text(0).endswith("anat"))
    anat.setCheckState(0, Qt.CheckState.Unchecked)
    assert [Path(p.recording).name for p in dlg._tree.chosen()] == [
        "sub-01_ses-1_task-rest_bold.nii.gz"]
    assert dlg._ok.text() == "Write 1 sidecar"
    dlg._tree.set_all(False)
    assert dlg._tree.chosen() == [] and not dlg._ok.isEnabled()
    dlg._tree.set_all(True)
    bold = next(i for i in dlg._tree.recordings() if i.text(0) == "task-rest_bold.nii.gz")
    bold.setCheckState(0, Qt.CheckState.Unchecked)
    func = bold.parent()
    assert func.checkState(0) == Qt.CheckState.Unchecked

    dlg._on_write()
    t1 = ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.json"
    assert [p.name for p in dlg.written()] == [t1.name]
    assert json.loads(t1.read_text()) == {"RepetitionTime": 2.3, "EchoTime": 0.03}
    assert not (ds / "sub-01/ses-1/func/sub-01_ses-1_task-rest_bold.json").exists()


def test_writing_is_one_undoable_step(qtbot, ds) -> None:
    dlg = _open(qtbot, ds)
    dlg._on_write()
    assert len(dlg.written()) == 2
    assert peek_undo(ds).label == "Write 2 sidecars from the recordings"
    undo_last(ds)
    assert all(not p.exists() for p in dlg.written())


def test_a_recording_with_a_sidecar_opens_on_adding_to_it(qtbot, ds) -> None:
    t2 = ds / "sub-01/ses-1/anat/sub-01_ses-1_T2w.nii.gz"
    dlg = _open(qtbot, ds, [t2])
    assert dlg._which.current() == "complete"
    (only,) = dlg._tree.recordings()
    assert only.text(1) == "1 added"          # EchoTime; TR is kept
    assert dlg._detail_path.text().startswith("Adds to ")
    # "Missing ones" leaves it alone and says so.
    dlg._which.set_current("missing")
    dlg._replan()
    qtbot.waitUntil(lambda: not dlg.planning(), timeout=30_000)
    assert dlg._tree.recordings() == []
    assert "has its sidecar" in dlg._status.text() or "Also add" in dlg._status.text()


def test_the_placeholder_choice_follows_the_setting_and_replans(qtbot, ds) -> None:
    dlg = _open(qtbot, ds, [ds / "sub-01/ses-1/func"])
    before = int(dlg._stat_fields.number.text())
    generation = dlg._generation
    dlg._scope.setCurrentIndex(dlg._scope.findData("recommended"))
    _replanned(qtbot, dlg, generation)
    assert int(dlg._stat_fields.number.text()) > before
    origins = {f.origin for p in dlg.plan().files for f in p.added}
    assert "placeholder" in origins


def test_the_source_box_is_off_where_nothing_was_converted(qtbot, ds) -> None:
    dlg = _open(qtbot, ds)
    assert not dlg._use_source.isEnabled() and not dlg._use_source.isChecked()


def test_short_names_drop_what_the_folder_says() -> None:
    assert short_name("sub-01/ses-1/func/sub-01_ses-1_task-rest_bold.nii.gz") == \
        "task-rest_bold.nii.gz"
    assert short_name("sub-01/anat/sub-01_T1w.nii.gz") == "T1w.nii.gz"
    assert short_name("sub-01/anat/sub-02_T1w.nii.gz") == "sub-02_T1w.nii.gz"


# ---------------------------------------------------------------------------
# The tree's right-click
# ---------------------------------------------------------------------------


@pytest.fixture
def pane(qtbot, ds):
    from bidsmgr.gui.widgets.bids_tree_pane import BidsTreePane

    widget = BidsTreePane()
    qtbot.addWidget(widget)
    widget.set_root(ds)
    return widget


def _menu(pane, monkeypatch, name: str) -> dict:
    from bidsmgr.gui.widgets.bids_tree_pane import PATH_ROLE

    open_every_folder(pane._tree)
    it = QTreeWidgetItemIterator(pane._tree)
    item = None
    while it.value():
        value = it.value().data(0, PATH_ROLE)
        if value and Path(value).name == name:
            item = it.value()
            break
        it += 1
    assert item is not None, name
    monkeypatch.setattr(pane._tree, "itemAt", lambda _pos: item)
    captured: dict = {}

    def _exec(self, *args, **kwargs):
        captured.update({a.text(): a for a in self.actions() if a.text()})
        return None

    monkeypatch.setattr(QMenu, "exec", _exec)
    pane._on_show_context_menu(QPoint(1, 1))
    return captured


def test_the_right_click_offers_the_words_that_fit(pane, monkeypatch) -> None:
    assert "Write its sidecar from the recording..." in _menu(
        pane, monkeypatch, "sub-01_ses-1_T1w.nii.gz")
    assert "Complete its sidecar from the recording..." in _menu(
        pane, monkeypatch, "sub-01_ses-1_T2w.nii.gz")
    assert "Complete this sidecar from its recording..." in _menu(
        pane, monkeypatch, "sub-01_ses-1_T2w.json")
    labels = _menu(pane, monkeypatch, "anat")
    assert "Write missing sidecars here from the recordings..." in labels
    # Under the same heading as the other repairs.
    keys = list(labels)
    assert keys.index("CHECK AND REPAIR") < keys.index(
        "Write missing sidecars here from the recordings...")


def test_the_right_click_announces_what_and_how(pane, monkeypatch) -> None:
    got = []
    pane.sidecars_requested.connect(lambda paths, complete: got.append((paths, complete)))
    _menu(pane, monkeypatch, "sub-01_ses-1_T2w.nii.gz")[
        "Complete its sidecar from the recording..."].trigger()
    _menu(pane, monkeypatch, "sub-01_ses-1_T1w.nii.gz")[
        "Write its sidecar from the recording..."].trigger()
    assert [(Path(p[0]).name, c) for p, c in got] == [
        ("sub-01_ses-1_T2w.nii.gz", True), ("sub-01_ses-1_T1w.nii.gz", False)]


def test_a_dataset_description_offers_nothing(pane, monkeypatch) -> None:
    labels = _menu(pane, monkeypatch, "dataset_description.json")
    assert not any("sidecar" in label.lower() for label in labels)


# ---------------------------------------------------------------------------
# The Editor
# ---------------------------------------------------------------------------


def test_the_editor_writes_and_refreshes(qtbot, ds, monkeypatch) -> None:
    from bidsmgr.gui.editor_panel import EditorPanel

    panel = EditorPanel()
    qtbot.addWidget(panel)
    panel._tree_pane.set_root(ds)
    opened = []

    def fake_exec(dlg):
        qtbot.waitUntil(lambda: not dlg.planning() and dlg.plan() is not None,
                        timeout=30_000)
        opened.append((dlg._which.current(), [Path(t).name for t in
                                              dlg._scope_bar.targets()]))
        dlg._on_write()
        return dlg.result()

    monkeypatch.setattr(SidecarsDialog, "exec", fake_exec)
    t1 = ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz"
    panel._tree_pane.sidecars_requested.emit([t1], False)
    assert opened == [("missing", ["sub-01_ses-1_T1w.nii.gz"])]
    assert sidecar_path(t1).exists()
    assert "sidecar" in peek_undo(ds).label


def test_no_window_appears_but_the_dialog(qtbot, ds) -> None:
    """GUI_LOOK_AND_FEEL 4.26: a spinner or a pill shown before it is placed
    is a window of its own. Planning, choosing and reading a sidecar open
    none."""
    from PyQt6.QtWidgets import QApplication

    dlg = SidecarsDialog(ds, None, fill_scope="none")
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitUntil(lambda: not dlg.planning() and dlg.plan() is not None, timeout=30_000)
    dlg._which.set_current("complete")
    dlg._replan()
    qtbot.waitUntil(lambda: not dlg.planning(), timeout=30_000)
    for item in dlg._tree.recordings():
        dlg._tree.setCurrentItem(item)
        item.setSelected(True)
    stray = [w for w in QApplication.topLevelWidgets()
             if w.isVisible() and w is not dlg]
    assert stray == [], [type(w).__name__ for w in stray]


# ---------------------------------------------------------------------------
# Files not named the BIDS way (2026-10-21)
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed(ds: Path) -> Path:
    _nifti(ds / "sub-01/ses-1/anat/T1.nii.gz")
    _nifti(ds / "sub-01/extra/scan.nii.gz", (4, 4, 3, 6), tr=0.8)
    _nifti(ds / "derivatives/prep/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz")
    (ds / "notes").mkdir()
    (ds / "notes/report.pdf").write_bytes(b"%PDF-1.4")
    return ds


@pytest.fixture
def mixed_pane(qtbot, mixed):
    from bidsmgr.gui.widgets.bids_tree_pane import BidsTreePane

    widget = BidsTreePane()
    qtbot.addWidget(widget)
    widget.set_root(mixed)
    return widget


@pytest.mark.parametrize("name", ["T1.nii.gz", "scan.nii.gz", "sub-01_desc-preproc_T1w.nii.gz"])
def test_a_file_not_named_the_bids_way_is_offered_too(mixed_pane, monkeypatch, name) -> None:
    labels = _menu(mixed_pane, monkeypatch, name)
    assert "Write its sidecar from the recording..." in labels
    assert labels["Write its sidecar from the recording..."].isEnabled()


def test_a_document_is_not_offered(mixed_pane, monkeypatch) -> None:
    labels = _menu(mixed_pane, monkeypatch, "report.pdf")
    assert not any("sidecar" in label.lower() for label in labels)


def test_view_only_shows_the_entry_greyed_and_says_why(mixed_pane, monkeypatch) -> None:
    mixed_pane.read_only = True
    action = _menu(mixed_pane, monkeypatch, "T1.nii.gz")["Write its sidecar from the recording..."]
    assert not action.isEnabled()
    assert "view only" in action.toolTip()


def test_a_non_bids_file_can_be_read_as_another_kind(qtbot, mixed) -> None:
    scan = mixed / "sub-01/extra/scan.nii.gz"
    dlg = _open(qtbot, mixed, [scan])
    (item,) = dlg._tree.recordings()
    assert item.font(0).italic()
    assert "Not a BIDS name or place" in item.toolTip(0)
    assert dlg._kind_row.isVisibleTo(dlg)
    assert (dlg._kind_datatype.currentData(), dlg._kind_suffix.currentData()) == ("func", "bold")
    assert "guessed" in dlg._kind_why.text()

    generation = dlg._generation
    dlg._kind_datatype.setCurrentIndex(dlg._kind_datatype.findData("dwi"))
    dlg._on_kind_datatype()
    _replanned(qtbot, dlg, generation)
    (plan,) = dlg.plan().files
    assert (plan.datatype, plan.suffix, plan.kind.how) == ("dwi", "dwi", "chosen")
    assert dlg._tree.selected_plan().recording == "sub-01/extra/scan.nii.gz"
    assert dlg._kind_why.text() == "your choice"


def test_a_bids_name_shows_no_kind_choice(qtbot, ds) -> None:
    dlg = _open(qtbot, ds, [ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz"])
    assert not dlg._kind_row.isVisibleTo(dlg)
    (item,) = dlg._tree.recordings()
    assert not item.font(0).italic()


def test_a_rejected_value_is_listed_with_why(qtbot) -> None:
    from bidsmgr.editor.sidecars import Kind, Rejected, SidecarPlan
    from bidsmgr.gui.widgets.sidecar_views import FieldList

    plan = SidecarPlan("sub-01/anat/a.nii.gz", "sub-01/anat/a.json", "anat", "T1w",
                       exists=False, kind=Kind("anat", "T1w", "bids"),
                       rejected=[Rejected("FlipAngle", -5, "source", "FlipAngle must be > 0")])
    view = FieldList()
    qtbot.addWidget(view)
    view.show_plan(plan)
    head = view.topLevelItem(0)
    assert head.text(0).startswith("Found, not written")
    row = head.child(0)
    assert (row.text(0), row.text(1), row.text(2)) == ("FlipAngle", "-5", "not valid")
    assert "must be > 0" in row.toolTip(2)
