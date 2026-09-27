"""Tests for the TSV viewer + file-kind routing (M6 Step 6a.3).

The Editor's center pane is now a :class:`QStackedWidget` that swaps
viewers based on the clicked file's extension:

* ``.tsv`` / ``.tsv.gz`` → :class:`TsvViewerPane` (read-only table).
* everything else (JSON sidecars, NIfTI, EEG/MEG, …) →
  :class:`SidecarFormPane` (which shows its existing form for JSON
  and a "no sidecar form for this file type" hint for other kinds).
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt

from bidsmgr.gui.editor_panel import EditorPanel
from bidsmgr.gui.widgets.tsv_viewer_pane import TsvViewerPane, _read_tsv


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Model helpers (the table model is a lazy QAbstractTableModel now, not a
# QStandardItemModel, so cells are read/written via the model index API).
# ---------------------------------------------------------------------------
def _cell(pane, r, c):
    return pane._model.data(pane._model.index(r, c))


def _set_cell(pane, r, c, value):
    pane._model.setData(pane._model.index(r, c), value)


def _hdr(pane, i):
    return pane._model.headerData(i, Qt.Orientation.Horizontal)



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bids_root(tmp_path: Path) -> Path:
    root = tmp_path / "Studyname"
    anat = root / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_ses-01_T1w.nii.gz").write_bytes(b"")
    (anat / "sub-01_ses-01_T1w.json").write_text(
        '{"Manufacturer": "Siemens"}'
    )
    func = root / "sub-01" / "ses-01" / "func"
    func.mkdir(parents=True)
    (func / "sub-01_ses-01_task-rest_events.tsv").write_text(
        "onset\tduration\ttrial_type\n"
        "0\t1.0\tcue\n"
        "1.5\t0.5\ttarget\n"
        "2.0\t1.0\trest\n"
    )
    # A gzipped TSV too.
    gz_path = func / "sub-01_ses-01_task-rest_physio.tsv.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        f.write("trigger\tcardiac\n")
        f.write("0\t0.5\n")
        f.write("1\t0.6\n")
    (root / "participants.tsv").write_text(
        "participant_id\tage\nsub-01\t27\n"
    )
    return root


# ---------------------------------------------------------------------------
# Helpers - TSV loading is threaded (NIfTI pattern), so tests must let the
# worker finish before asserting on the populated model. Polling on the
# pane's ``_loader`` is race-free (set to None in the load callbacks).
# ---------------------------------------------------------------------------
def _load(pane, path, root, qapp, timeout_ms: int = 10000) -> None:
    from PyQt6.QtCore import QElapsedTimer

    pane.set_file(path, root)
    t = QElapsedTimer()
    t.start()
    while pane._loader is not None and t.elapsed() < timeout_ms:
        qapp.processEvents()
    qapp.processEvents()


def _wait_loaded(viewer, qapp, timeout_ms: int = 10000) -> None:
    from PyQt6.QtCore import QElapsedTimer

    t = QElapsedTimer()
    t.start()
    while viewer._loader is not None and t.elapsed() < timeout_ms:
        qapp.processEvents()
    qapp.processEvents()


# ---------------------------------------------------------------------------
# _read_tsv helper
# ---------------------------------------------------------------------------


def test_read_tsv_plain(qtbot, tmp_path: Path) -> None:
    p = tmp_path / "x.tsv"
    p.write_text("a\tb\n1\t2\n3\t4\n")
    header, rows, total = _read_tsv(p)
    assert header == ["a", "b"]
    assert rows == [["1", "2"], ["3", "4"]]
    assert total == 2


def test_read_tsv_gzipped(qtbot, tmp_path: Path) -> None:
    p = tmp_path / "x.tsv.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write("a\tb\n1\t2\n")
    header, rows, _ = _read_tsv(p)
    assert header == ["a", "b"]
    assert rows == [["1", "2"]]


def test_read_tsv_empty_returns_no_header(qtbot, tmp_path: Path) -> None:
    p = tmp_path / "empty.tsv"
    p.write_text("")
    header, rows, total = _read_tsv(p)
    assert header == []
    assert rows == []
    assert total == 0


def test_read_tsv_handles_unreadable_file(qtbot, tmp_path: Path) -> None:
    header, rows, _ = _read_tsv(tmp_path / "does-not-exist.tsv")
    assert header == []
    assert rows == []


def test_read_tsv_caps_preview_rows(qtbot, tmp_path: Path) -> None:
    p = tmp_path / "big.tsv"
    buf = io.StringIO()
    buf.write("a\n")
    for i in range(7):
        buf.write(f"{i}\n")
    p.write_text(buf.getvalue())
    header, rows, total = _read_tsv(p, max_rows=3)
    assert header == ["a"]
    assert len(rows) == 3
    assert total == 7  # disk had 7 data rows


# ---------------------------------------------------------------------------
# TsvViewerPane basics
# ---------------------------------------------------------------------------


def test_pane_starts_with_empty_hint(qapp, qtbot) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    assert pane.current_file() is None
    assert pane._stack.currentIndex() == 0  # hint visible
    assert pane._model.rowCount() == 0


def test_set_file_populates_table(qapp, qtbot, bids_root: Path) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    _load(pane, events, bids_root, qapp)
    assert pane.current_file() == events
    assert pane._stack.currentIndex() == 1  # table visible
    assert pane._model.columnCount() == 3
    assert pane._model.rowCount() == 3
    # Column headers come from the first line of the file.
    headers = [
        _hdr(pane, i)
        for i in range(pane._model.columnCount())
    ]
    assert headers == ["onset", "duration", "trial_type"]
    # First data row.
    assert _cell(pane, 0, 0) == "0"
    assert _cell(pane, 0, 2) == "cue"


def test_set_file_handles_tsv_gz(qapp, qtbot, bids_root: Path) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    gz = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_physio.tsv.gz"
    )
    # .gz tables always load on a background thread (uncompressed size is
    # opaque), so wait for the parse to land before asserting.
    with qtbot.waitSignal(pane.loaded, timeout=10000):
        pane.set_file(gz, bids_root)
    assert pane._model.columnCount() == 2
    assert pane._model.rowCount() == 2


def test_set_file_none_clears(qapp, qtbot, bids_root: Path) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    _load(pane, events, bids_root, qapp)
    pane.set_file(None, None)
    assert pane.current_file() is None
    assert pane._stack.currentIndex() == 0
    assert pane._model.rowCount() == 0


def test_footer_summary_reports_dimensions(qapp, qtbot, bids_root: Path) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    _load(pane, events, bids_root, qapp)
    text = pane._footer_summary.text()
    assert "3 rows" in text and "3 columns" in text


def test_table_cells_are_editable(qapp, qtbot, bids_root: Path) -> None:
    from PyQt6.QtWidgets import QAbstractItemView
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    _load(pane, events, bids_root, qapp)
    triggers = pane._table.editTriggers()
    # Cells respond to double-click and F2.
    assert bool(triggers & QAbstractItemView.EditTrigger.DoubleClicked)
    assert bool(triggers & QAbstractItemView.EditTrigger.EditKeyPressed)


def test_edit_cell_marks_pane_dirty(
    qapp, qtbot, bids_root: Path,
) -> None:
    """Editing a cell flips the dirty state and enables Save/Revert.
    Disk is untouched until Save."""
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    _load(pane, events, bids_root, qapp)
    assert not pane.is_dirty()
    assert not pane._save_btn.isEnabled()

    _set_cell(pane, 0, 2, "custom_trial")
    qapp.processEvents()

    assert pane.is_dirty()
    assert pane._save_btn.isEnabled()
    assert pane._revert_btn.isEnabled()
    # Disk untouched.
    on_disk = events.read_text()
    assert "custom_trial" not in on_disk


def test_save_flushes_cells_to_disk(
    qapp, qtbot, bids_root: Path,
) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    _load(pane, events, bids_root, qapp)
    _set_cell(pane, 0, 2, "custom_trial")
    qapp.processEvents()

    with qtbot.waitSignal(pane.file_saved, timeout=500):
        pane.save()

    # Disk reflects the edit.
    on_disk = events.read_text()
    assert "custom_trial" in on_disk
    # Header still intact.
    first_line = on_disk.splitlines()[0]
    assert first_line == "onset\tduration\ttrial_type"
    # Pane is clean.
    assert not pane.is_dirty()
    assert not pane._save_btn.isEnabled()


def test_add_row_appends_blank_row(qapp, qtbot, bids_root: Path) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    _load(pane, events, bids_root, qapp)
    before = pane._model.rowCount()
    pane._on_add_row()
    assert pane._model.rowCount() == before + 1
    # Newest row has the same column count and is empty.
    last = before
    assert _cell(pane, last, 0) == ""
    assert pane.is_dirty()


def test_delete_row_removes_selected_row(qapp, qtbot, bids_root: Path) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    _load(pane, events, bids_root, qapp)
    before = pane._model.rowCount()
    # Select the first row.
    pane._table.setCurrentIndex(pane._model.index(0, 0))
    pane._on_delete_row()
    assert pane._model.rowCount() == before - 1
    assert pane.is_dirty()


def test_add_column_appends_with_user_name(
    qapp, qtbot, bids_root: Path, monkeypatch,
) -> None:
    """``Add column`` pops a QInputDialog for the name; the new column
    appends to the right with the user-provided header."""
    from PyQt6.QtWidgets import QInputDialog

    monkeypatch.setattr(
        QInputDialog,
        "getText",
        lambda *args, **kwargs: ("extra", True),
    )

    pane = TsvViewerPane()

    qtbot.addWidget(pane)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    _load(pane, events, bids_root, qapp)
    before_cols = pane._model.columnCount()
    pane._on_add_column()

    assert pane._model.columnCount() == before_cols + 1
    new_idx = before_cols
    assert _hdr(pane, new_idx) == "extra"
    # Every existing row got a blank cell in the new column.
    for r in range(pane._model.rowCount()):
        assert _cell(pane, r, new_idx) == ""
    assert pane.is_dirty()


def test_add_column_cancel_is_noop(
    qapp, qtbot, bids_root: Path, monkeypatch,
) -> None:
    from PyQt6.QtWidgets import QInputDialog

    monkeypatch.setattr(
        QInputDialog,
        "getText",
        lambda *args, **kwargs: ("", False),  # user clicked Cancel
    )

    pane = TsvViewerPane()

    qtbot.addWidget(pane)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    _load(pane, events, bids_root, qapp)
    before = pane._model.columnCount()
    pane._on_add_column()
    assert pane._model.columnCount() == before
    assert not pane.is_dirty()


def test_delete_column_removes_selected_column(qapp, qtbot, bids_root: Path) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    _load(pane, events, bids_root, qapp)
    before = pane._model.columnCount()
    pane._table.setCurrentIndex(pane._model.index(0, 1))  # duration column
    pane._on_delete_column()
    assert pane._model.columnCount() == before - 1
    # The remaining headers no longer include "duration".
    remaining = [
        _hdr(pane, i)
        for i in range(pane._model.columnCount())
    ]
    assert "duration" not in remaining
    assert pane.is_dirty()


def test_revert_reloads_from_disk(qapp, qtbot, bids_root: Path) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    _load(pane, events, bids_root, qapp)
    pane._on_add_row()
    _set_cell(pane, pane._model.rowCount() - 1, 0, "99")
    qapp.processEvents()
    assert pane.is_dirty()

    pane.revert()  # revert re-reads from disk on the worker thread
    _wait_loaded(pane, qapp)

    assert not pane.is_dirty()
    # Disk unchanged (we never saved), and the in-memory row count
    # reflects only the on-disk rows.
    on_disk = events.read_text()
    assert "99" not in on_disk
    assert pane._model.rowCount() == 3  # original row count


def test_save_handles_tsv_gz(qapp, qtbot, bids_root: Path) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    gz = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_physio.tsv.gz"
    )
    with qtbot.waitSignal(pane.loaded, timeout=10000):
        pane.set_file(gz, bids_root)
    _set_cell(pane, 0, 0, "99")
    qapp.processEvents()

    with qtbot.waitSignal(pane.file_saved, timeout=500):
        pane.save()

    # Re-read via gzip to confirm.
    header, rows, _ = _read_tsv(gz)
    assert rows[0][0] == "99"


def test_save_failed_signal_on_io_error(
    qapp, qtbot, bids_root: Path, monkeypatch,
) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    _load(pane, events, bids_root, qapp)
    _set_cell(pane, 0, 0, "X")
    qapp.processEvents()

    def _explode(self_, path):
        raise OSError("disk full")

    monkeypatch.setattr(type(pane), "_write_model_to_disk", _explode)
    with qtbot.waitSignal(pane.save_failed, timeout=500) as sig:
        ok = pane.save()
    assert ok is False
    _, msg = sig.args
    assert "disk full" in msg
    # Pane stays dirty because the save failed.
    assert pane.is_dirty()


def test_save_with_no_dirty_state_is_noop(qapp, qtbot, bids_root: Path) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    _load(pane, events, bids_root, qapp)
    mtime_before = events.stat().st_mtime_ns
    assert pane.save() is True
    assert events.stat().st_mtime_ns == mtime_before


def test_switching_files_discards_unsaved_edits(
    qapp, qtbot, bids_root: Path,
) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    _load(pane, events, bids_root, qapp)
    pane._on_add_row()
    assert pane.is_dirty()

    participants = bids_root / "participants.tsv"
    _load(pane, participants, bids_root, qapp)
    # Loading a different file resets dirty state.
    assert not pane.is_dirty()
    # Original file unchanged on disk.
    assert "99" not in events.read_text()


def test_large_tsv_reports_total_and_caps(qapp, qtbot, tmp_path: Path) -> None:
    """A big TSV reads its real total but caps the preview - and (with the
    pandas C parser) loads on the worker without freezing the GUI."""
    from bidsmgr.gui.widgets.tsv_viewer_pane import _MAX_PREVIEW_ROWS

    p = tmp_path / "big.tsv"
    extra = 2500
    with open(p, "w") as f:
        f.write("a\tb\n")
        for i in range(_MAX_PREVIEW_ROWS + extra):
            f.write(f"{i}\t{i * 2}\n")
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    _load(pane, p, tmp_path, qapp)
    assert pane._model.rowCount() == _MAX_PREVIEW_ROWS
    assert str(_MAX_PREVIEW_ROWS + extra) in pane._footer_summary.text()


def test_ragged_row_is_padded_to_header_width(qapp, qtbot, tmp_path: Path) -> None:
    p = tmp_path / "ragged.tsv"
    p.write_text("a\tb\tc\n1\t2\n3\t4\t5\n")
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    _load(pane, p, tmp_path, qapp)
    # Header has 3 columns; first data row has 2 cells but is padded.
    assert pane._model.columnCount() == 3
    assert pane._model.rowCount() == 2
    assert _cell(pane, 0, 0) == "1"
    assert _cell(pane, 0, 1) == "2"
    assert _cell(pane, 0, 2) == ""


# ---------------------------------------------------------------------------
# EditorPanel routing
# ---------------------------------------------------------------------------


def test_clicking_tsv_swaps_center_pane_to_table(
    qapp, qtbot, isolated_settings, bids_root: Path,
) -> None:
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)

    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    panel._on_file_selected(events)
    _wait_loaded(panel._tsv_viewer, qapp)

    # Center stack now shows the TSV viewer.
    assert panel._center_stack.currentWidget() is panel._tsv_viewer
    assert panel._tsv_viewer.current_file() == events
    # Sidecar pane was reset.
    assert panel._sidecar_form.current_file() is None


def test_clicking_json_swaps_back_to_sidecar(
    qapp, qtbot, isolated_settings, bids_root: Path,
) -> None:
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)

    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    panel._on_file_selected(events)
    _wait_loaded(panel._tsv_viewer, qapp)
    assert panel._center_stack.currentWidget() is panel._tsv_viewer

    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    panel._on_file_selected(json_path)
    qapp.processEvents()
    assert panel._center_stack.currentWidget() is panel._sidecar_form
    assert panel._sidecar_form.current_file() == json_path
    # TSV viewer was reset.
    assert panel._tsv_viewer.current_file() is None


def test_clicking_directory_returns_to_sidecar_view(
    qapp, qtbot, isolated_settings, bids_root: Path,
) -> None:
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)

    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    panel._on_file_selected(events)
    _wait_loaded(panel._tsv_viewer, qapp)
    assert panel._center_stack.currentWidget() is panel._tsv_viewer

    anat = bids_root / "sub-01" / "ses-01" / "anat"
    panel._on_file_selected(anat)
    qapp.processEvents()
    assert panel._center_stack.currentWidget() is panel._sidecar_form
    assert panel._sidecar_form.current_file() is None


def test_root_swap_clears_tsv_viewer(
    qapp, qtbot, isolated_settings, bids_root: Path, tmp_path: Path,
) -> None:
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)
    events = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_events.tsv"
    )
    panel._on_file_selected(events)
    _wait_loaded(panel._tsv_viewer, qapp)
    assert panel._tsv_viewer.current_file() == events

    other = tmp_path / "Other"
    other.mkdir()
    panel._set_root(other, persist=False)
    assert panel._tsv_viewer.current_file() is None
    assert panel._center_stack.currentWidget() is panel._sidecar_form


def test_tsv_undo_redo(qapp, qtbot, bids_root: Path) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    events = bids_root / "sub-01" / "ses-01" / "func" / "sub-01_ses-01_task-x_events.tsv"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text("onset\tduration\n0\t1\n", encoding="utf-8")
    _load(pane, events, bids_root, qapp)
    assert not pane.can_undo()  # nothing to undo at load

    _set_cell(pane, 0, 0, "5")  # edit a cell
    assert pane.can_undo() and pane.is_dirty()
    assert _cell(pane, 0, 0) == "5"

    pane.undo()
    assert _cell(pane, 0, 0) == "0"
    assert not pane.is_dirty() and pane.can_redo()

    pane.redo()
    assert _cell(pane, 0, 0) == "5"
    assert pane.is_dirty()


def test_tsv_new_edit_clears_redo(qapp, qtbot, bids_root: Path) -> None:
    pane = TsvViewerPane()
    qtbot.addWidget(pane)
    events = bids_root / "e.tsv"
    events.write_text("a\tb\n1\t2\n", encoding="utf-8")
    _load(pane, events, bids_root, qapp)
    _set_cell(pane, 0, 0, "x")
    pane.undo()
    assert pane.can_redo()
    _set_cell(pane, 0, 1, "y")  # a fresh edit diverges the timeline
    assert not pane.can_redo()
