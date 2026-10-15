"""The DICOM header pane: the Editor's view of a DICOM, the Converter's
DICOM window, the comparison and the series shown for a look."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from bidsmgr.gui.editor_panel import EditorPanel
from bidsmgr.gui.widgets.dicom_pane import (
    ELEMENT_ROLE, HIDDEN, READ_FOR, VALUE, DicomPane, DicomWindow,
)
from tests.fixtures.dicoms import write_mr_series

pytestmark = pytest.mark.gui


def _wait(qtbot, pane: DicomPane) -> None:
    qtbot.waitUntil(lambda: pane.header() is not None or
                    pane._stack.currentWidget() is pane._message_page, timeout=10000)


def _rows(pane: DicomPane) -> list:
    """Every element row shown, depth first."""
    out = []

    def walk(parent) -> None:
        for r in range(pane.proxy.rowCount(parent)):
            idx = pane.proxy.index(r, 0, parent)
            e = pane.proxy.data(idx, ELEMENT_ROLE)
            if e is not None:
                out.append((e, pane.proxy.data(idx.siblingAtColumn(VALUE))))
            walk(idx)

    from PyQt6.QtCore import QModelIndex

    walk(QModelIndex())
    return out


def _enhanced(path: Path, frames: int = 3) -> Path:
    import pydicom
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    ds = pydicom.dcmread(str(path))
    items = []
    for i in range(frames):
        item = Dataset()
        pos = Dataset()
        pos.ImagePositionPatient = [0.0, 0.0, float(i)]
        item.PlanePositionSequence = Sequence([pos])
        content = Dataset()
        content.TemporalPositionIndex = i + 1
        item.FrameContentSequence = Sequence([content])
        items.append(item)
    ds.PerFrameFunctionalGroupsSequence = Sequence(items)
    ds.save_as(str(path))
    return path


@pytest.fixture
def series(tmp_path) -> list[Path]:
    return write_mr_series(tmp_path / "raw" / "s3", "1.2.3.4", description="bold_run1")


@pytest.fixture
def pane(qtbot) -> DicomPane:
    p = DicomPane()
    qtbot.addWidget(p)
    p.resize(900, 600)
    return p


def test_the_header_is_read_on_a_thread_and_summarised(pane, qtbot, series) -> None:
    pane.set_file(series[0])
    assert pane._stack.currentWidget() is pane._pending, "reading shows where it lands"
    _wait(qtbot, pane)
    assert pane._stack.currentWidget() is pane.tree
    facts = pane.facts_text()
    assert facts[0] == "MR"
    assert "Series 3 · bold_run1" in facts
    assert any(f.endswith("identifying values") for f in facts)
    assert pane._title.text().lower() == series[0].name.lower()
    # Modules at the top, open; the Patient module first.
    first = pane.proxy.index(0, 0)
    assert pane.proxy.data(first) == "Patient" and pane.tree.isExpanded(first)


def test_identifying_values_are_hidden_until_asked(pane, qtbot, series) -> None:
    pane.set_file(series[0])
    _wait(qtbot, pane)
    shown = {e.keyword: v for e, v in _rows(pane)}
    assert shown["PatientName"] == HIDDEN
    assert shown["SeriesDescription"] == "bold_run1"
    # A hidden value cannot be found by searching for it.
    pane.set_search("Test^Subject")
    assert _rows(pane) == []
    pane.identifying_toggle.setChecked(True)
    assert [v for _e, v in _rows(pane)] == ["Test^Subject"]


def test_search_finds_by_name_keyword_tag_and_value(pane, qtbot, series) -> None:
    pane.set_file(series[0])
    _wait(qtbot, pane)
    for text in ("series description", "SeriesDescription", "(0008,103E)", "bold_run1"):
        pane.set_search(text)
        assert "SeriesDescription" in {e.keyword for e, _v in _rows(pane)}, text
    pane.set_search("no such thing anywhere")
    assert _rows(pane) == []
    pane.set_search("")
    assert len(_rows(pane)) > 20


def test_what_bids_manager_reads(pane, qtbot, series) -> None:
    pane.set_file(series[0])
    _wait(qtbot, pane)
    assert pane.tree.isColumnHidden(READ_FOR)
    pane.read_toggle.setChecked(True)
    rows = _rows(pane)
    assert rows and all(e.read_for for e, _v in rows)
    assert "SeriesInstanceUID" in {e.keyword for e, _v in rows}
    assert not pane.tree.isColumnHidden(READ_FOR)


def test_an_enhanced_file_lists_its_frames(pane, qtbot, series) -> None:
    pane.set_file(_enhanced(series[0]))
    _wait(qtbot, pane)
    assert pane.frames_pill.isVisibleTo(pane) and pane.frames_pill.text() == "Frames · 3"
    pane.show_frames()
    assert pane._stack.currentWidget() is pane.frames
    assert pane.frames_model.rowCount() == 3
    assert not pane.search.isVisibleTo(pane), "the search is for the header"
    # A plain file has no frames view at all.
    pane.set_file(series[1])
    _wait(qtbot, pane)
    assert not pane.frames_pill.isVisibleTo(pane)
    assert pane._stack.currentWidget() is pane.tree


def test_copies_leave_identifying_values_out(pane, qtbot, series, tmp_path) -> None:
    pane.set_file(series[0])
    _wait(qtbot, pane)
    assert "Test^Subject" not in pane.text_copy()
    target = tmp_path / "header.json"
    pane.save_copy(target, "json")
    assert "Test^Subject" not in target.read_text(encoding="utf-8")
    pane._include_identifying_in_copies = True
    assert "Test^Subject" in pane.text_copy()


def test_copying_the_chosen_rows(pane, qtbot, series) -> None:
    from PyQt6.QtGui import QGuiApplication

    pane.set_file(series[0])
    _wait(qtbot, pane)
    pane.set_search("SeriesDescription")
    for r in range(pane.proxy.rowCount()):
        module = pane.proxy.index(r, 0)
        pane.tree.selectionModel().select(
            pane.proxy.index(0, 0, module),
            pane.tree.selectionModel().SelectionFlag.Select
            | pane.tree.selectionModel().SelectionFlag.Rows)
    pane.copy_selection()
    assert QGuiApplication.clipboard().text() == "Series Description: bold_run1"


def test_a_file_that_cannot_be_read_says_so(pane, qtbot, tmp_path) -> None:
    gone = tmp_path / "moved_away.dcm"         # moved between the click and the read
    pane.set_file(gone)
    _wait(qtbot, pane)
    assert pane.header() is None
    assert pane._stack.currentWidget() is pane._message_page
    assert "could not be read" in pane.message.text()
    assert not pane.compare_button.isEnabled() and not pane.image_button.isEnabled()


def test_comparing_two_series(pane, qtbot, series, tmp_path) -> None:
    import pydicom

    other = write_mr_series(tmp_path / "raw" / "s4", "1.2.3.5", description="bold_run2")[0]
    ds = pydicom.dcmread(str(other))
    ds.RepetitionTime = 2000
    ds.save_as(str(other))
    pane.set_file(series[0])
    _wait(qtbot, pane)
    dlg = pane.compare_with(other)
    qtbot.addWidget(dlg)
    qtbot.waitUntil(lambda: not dlg.is_pending(), timeout=10000)
    where = {d.where for d in dlg.differences()}
    assert "Series Description" in where
    assert "SOP Instance UID" not in where, "what differs between any two images is hidden"
    dlg.routine_toggle.setChecked(False)
    assert "SOP Instance UID" in {d.where for d in dlg.differences()}
    dlg.search.setText("description")
    assert {d.where for d in dlg.differences()} == {"Series Description"}
    assert "difference" in dlg.summary.text()


def test_the_editor_opens_a_dicom_by_its_content(qtbot, series, tmp_path) -> None:
    panel = EditorPanel()
    qtbot.addWidget(panel)
    raw = tmp_path / "raw"
    bare = raw / "s3" / "IM0001"
    shutil.copy(series[0], bare)
    assert panel.open_folder(raw) == "view"
    panel._on_file_selected(bare)
    assert panel._center_stack.currentWidget() is panel._dicom_pane
    _wait(qtbot, panel._dicom_pane)
    assert panel._dicom_pane.facts_text()[0] == "MR"
    # Choosing another file clears it.
    (raw / "notes.txt").write_text("x\n", encoding="utf-8")
    panel._on_file_selected(raw / "notes.txt")
    assert panel._dicom_pane.current_file() is None


def test_the_converter_window_is_reused(qtbot, series) -> None:
    win = DicomWindow()
    qtbot.addWidget(win)
    win.show_file(series[0])
    _wait(qtbot, win.pane)
    assert win.isVisible() and series[0].name in win.windowTitle()
    win.show_file(series[1])
    qtbot.waitUntil(lambda: win.pane.current_file() == series[1], timeout=5000)
    assert series[1].name in win.windowTitle()


def test_the_image_opens_on_the_click_and_fills(pane, qtbot, series) -> None:
    pytest.importorskip("dcm2niix")
    pane.set_file(series[0])
    _wait(qtbot, pane)
    pane.show_image()
    dlg = pane.preview_dialog
    qtbot.addWidget(dlg)
    assert dlg.isVisible() and dlg.is_pending(), "the window opens on the click"
    assert not pane.image_button.isEnabled(), "one at a time"
    qtbot.waitUntil(lambda: not dlg.is_pending(), timeout=60000)
    assert dlg.files and dlg.files[0].name.endswith((".nii.gz", ".nii"))
    assert pane.image_button.isEnabled()
    dlg.close()


def _converter(qtbot, tmp_path: Path, uid: str):
    import json

    import pandas as pd

    from bidsmgr.cli.create import open_or_create_workspace
    from bidsmgr.gui.converter_panel import ConverterPanel
    from bidsmgr.project import Project, ScanImported, workspace

    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    vdir = workspace.allocate_version_dir(root, "raw")
    vproj = Project.create(vdir, name=root.name)
    df = pd.DataFrame([{
        "participant_id": "sub-001", "session": "", "include": 1, "modality": "mri",
        "datatype": "func", "bids_name": "sub-001_task-rest_bold", "issues": "",
        "bids_guess_suffix": "bold", "bids_guess_confidence": "0.9",
        "bids_guess_skip": False,
        "sequence": "bold_run1", "entities": json.dumps({"subject": "001", "task": "rest"}),
        "task": "rest", "run": "", "series_uid": uid, "dataset": "ds",
        "source_file": "", "source_folder": "s3"}])
    df.to_csv(workspace.version_inventory(vdir), sep="\t", index=False)
    workspace.write_version_meta(vdir, source_label="raw", raw_root=str(tmp_path / "raw"),
                                 status="curating")
    vproj.append(ScanImported(inventory_tsv=str(workspace.version_inventory(vdir)),
                              row_ids=(uid,), raw_root=None))
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)
    return panel


def test_the_converter_shows_a_rows_header_and_a_raw_files(qtbot, series, tmp_path) -> None:
    from bidsmgr.gui.viz.menus import popup_menu
    from bidsmgr.inventory.sources import ScanSources

    panel = _converter(qtbot, tmp_path, "1.2.3.4")
    raw = tmp_path / "raw"
    panel._sources = ScanSources(raw, raw, {"1.2.3.4": [str(f) for f in series]})
    menu = popup_menu(panel)
    action = panel._dicom_header_action(menu, 0)
    assert action is not None and action.text() == "Show the DICOM header"
    action.trigger()
    win = panel._dicom_window
    qtbot.addWidget(win)
    _wait(qtbot, win.pane)
    assert win.pane.current_file() == series[0]

    # A double-click on a DICOM in the raw data tree opens it in the same window.
    panel._raw_pane.set_root(raw)
    tree = panel._raw_pane._tree
    folder = tree.topLevelItem(0).child(0)
    folder.setExpanded(True)
    item = next(folder.child(i) for i in range(folder.childCount())
                if folder.child(i).text(0) == series[1].name)
    tree.itemDoubleClicked.emit(item, 0)
    assert panel._dicom_window is win
    qtbot.waitUntil(lambda: win.pane.current_file() == series[1], timeout=5000)
