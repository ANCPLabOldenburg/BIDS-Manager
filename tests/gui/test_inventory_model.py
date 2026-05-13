"""Tests for ``bidsmgr.gui.models.InventoryTableModel``.

Two layers:

* **Pure model tests** — exercise ``rowCount`` / ``columnCount`` /
  ``data`` / ``setData`` against an in-memory DataFrame. No view, no
  delegate, no QApplication required beyond what pytest-qt sets up.
* **Integration test** — render the model through a real ``QTableView``
  with the extracted delegates. Confirms the role roundtrip (model
  publishes data → delegate reads it from the index) works end to end.

Marked ``gui`` so they run under ``QT_QPA_PLATFORM=offscreen`` in the
same gate as the widget smoke tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PyQt6.QtCore import QModelIndex, Qt
from PyQt6.QtWidgets import QTableView

from bidsmgr.gui.delegates import (
    PAYLOAD_ROLE,
    ROW_STATE_ROLE,
    CellTextDelegate,
    CheckboxDelegate,
    StatusDelegate,
)
from bidsmgr.gui.models import COLUMNS, InventoryTableModel
from bidsmgr.project import (
    Project,
    ScanImported,
    UserSetCell,
    UserToggleInclude,
)


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ok_row(**overrides) -> dict:
    """A valid MRI row that should pass through with row_state=''."""
    base = {
        "BIDS_name": "sub-001",
        "session": "ses-pre",
        "include": 1,
        "modality": "mri",
        "modality_bids": "anat",
        "sequence": "t1w_mprage",
        "series_uid": "1.2.3.4",
        "proposed_datatype": "anat",
        "proposed_basename": "sub-001_ses-pre_T1w",
        "Proposed BIDS name": "anat/sub-001_ses-pre_T1w.nii.gz",
        "bids_guess_classifier": "dcm2niix_bidsguess",
        "bids_guess_datatype": "anat",
        "bids_guess_suffix": "T1w",
        "bids_guess_confidence": "0.97",
        "bids_guess_skip": False,
        "proposed_issues": "",
        "entities": json.dumps(
            {"subject": "001", "session": "pre"},
            sort_keys=True,
        ),
        "task": "",
        "run": "",
        "source_file": "",
    }
    base.update(overrides)
    return base


def _func_row(**overrides) -> dict:
    return _ok_row(
        modality_bids="func",
        proposed_datatype="func",
        proposed_basename="sub-001_ses-pre_task-rest_bold",
        Proposed_BIDS_name="func/sub-001_ses-pre_task-rest_bold.nii.gz",
        bids_guess_datatype="func",
        bids_guess_suffix="bold",
        bids_guess_confidence="0.97",
        entities=json.dumps(
            {"subject": "001", "session": "pre", "task": "rest"},
            sort_keys=True,
        ),
        task="rest",
        **overrides,
    )


def _physio_row(**overrides) -> dict:
    return _ok_row(
        modality_bids="physio",
        proposed_datatype="func",
        proposed_basename="sub-002_ses-post_task-mb_physio",
        bids_guess_suffix="physio",
        bids_guess_confidence="0.99",
        series_uid="5.6.7.8",
        BIDS_name="sub-002",
        session="ses-post",
        task="mb",
        **overrides,
    )


def _eeg_row(**overrides) -> dict:
    return _ok_row(
        modality="eeg",
        modality_bids="eeg",
        proposed_datatype="eeg",
        proposed_basename="sub-001_task-rest_eeg",
        bids_guess_datatype="eeg",
        bids_guess_suffix="eeg",
        source_file="/data/raw/sub-001/eeg/sub-001_task-rest_eeg.edf",
        series_uid="",
        entities=json.dumps(
            {"subject": "001", "task": "rest"}, sort_keys=True,
        ),
        task="rest",
        **overrides,
    )


def _err_row(**overrides) -> dict:
    return _ok_row(
        proposed_basename="",
        proposed_datatype="",
        proposed_issues="task entity required for func/bold",
        bids_guess_confidence="0.61",
        bids_guess_suffix="bold",
        series_uid="9.10.11.12",
        BIDS_name="sub-003",
        session="",
        **overrides,
    )


def _skip_row(**overrides) -> dict:
    return _ok_row(
        include=0,
        bids_guess_skip=True,
        proposed_basename="localizer_20ch_head-coil",
        proposed_datatype="",
        bids_guess_confidence="",
        series_uid="13.14.15",
        **overrides,
    )


def make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_row_and_column_counts() -> None:
    df = make_df([_ok_row(), _func_row()])
    model = InventoryTableModel(df)
    assert model.rowCount() == 2
    assert model.columnCount() == len(COLUMNS)


def test_header_labels_match_spec() -> None:
    df = make_df([_ok_row()])
    model = InventoryTableModel(df)
    for col, spec in enumerate(COLUMNS):
        assert model.headerData(col, Qt.Orientation.Horizontal) == spec.header


def test_flags_only_editable_columns_carry_edit_flag() -> None:
    df = make_df([_ok_row()])
    model = InventoryTableModel(df)
    for col, spec in enumerate(COLUMNS):
        idx = model.index(0, col)
        editable = bool(model.flags(idx) & Qt.ItemFlag.ItemIsEditable)
        assert editable == spec.editable, f"column {spec.key!r} editable mismatch"


# ---------------------------------------------------------------------------
# Display values
# ---------------------------------------------------------------------------


def test_id_strips_sub_prefix() -> None:
    df = make_df([_ok_row(BIDS_name="sub-042")])
    model = InventoryTableModel(df)
    col = next(i for i, c in enumerate(COLUMNS) if c.key == "id")
    assert model.data(model.index(0, col), Qt.ItemDataRole.DisplayRole) == "042"


def test_session_strips_ses_prefix_and_dashes_empty() -> None:
    df = make_df([_ok_row(session="ses-pre"), _ok_row(session="")])
    model = InventoryTableModel(df)
    col = next(i for i, c in enumerate(COLUMNS) if c.key == "ses")
    assert model.data(model.index(0, col), Qt.ItemDataRole.DisplayRole) == "pre"
    assert model.data(model.index(1, col), Qt.ItemDataRole.DisplayRole) == "—"


def test_confidence_formatted_without_leading_zero() -> None:
    df = make_df([_ok_row(bids_guess_confidence="0.97"),
                  _ok_row(bids_guess_confidence="0.71")])
    model = InventoryTableModel(df)
    col = next(i for i, c in enumerate(COLUMNS) if c.key == "conf")
    assert model.data(model.index(0, col), Qt.ItemDataRole.DisplayRole) == ".97"
    assert model.data(model.index(1, col), Qt.ItemDataRole.DisplayRole) == ".71"


def test_backend_derivation() -> None:
    df = make_df([_ok_row(), _physio_row(), _eeg_row()])
    model = InventoryTableModel(df)
    col = next(i for i, c in enumerate(COLUMNS) if c.key == "backend")
    assert model.data(model.index(0, col), Qt.ItemDataRole.DisplayRole) == "dcm2niix"
    assert model.data(model.index(1, col), Qt.ItemDataRole.DisplayRole) == "bidsphysio"
    assert model.data(model.index(2, col), Qt.ItemDataRole.DisplayRole) == "mne-bids"


# ---------------------------------------------------------------------------
# Row state / status badge
# ---------------------------------------------------------------------------


def test_row_state_for_valid_row_is_empty() -> None:
    df = make_df([_ok_row()])
    model = InventoryTableModel(df)
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == ""


def test_row_state_err_for_missing_basename() -> None:
    df = make_df([_err_row()])
    model = InventoryTableModel(df)
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == "err"


def test_row_state_skip_for_excluded_rows() -> None:
    df = make_df([_skip_row()])
    model = InventoryTableModel(df)
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == "skip"


def test_status_kind_phys_for_physio_rows() -> None:
    df = make_df([_physio_row()])
    model = InventoryTableModel(df)
    status_col = next(i for i, c in enumerate(COLUMNS) if c.key == "status")
    assert model.data(model.index(0, status_col), PAYLOAD_ROLE) == "phys"


def test_status_kind_ok_for_valid_mri_row() -> None:
    df = make_df([_ok_row()])
    model = InventoryTableModel(df)
    status_col = next(i for i, c in enumerate(COLUMNS) if c.key == "status")
    assert model.data(model.index(0, status_col), PAYLOAD_ROLE) == "ok"


def test_include_payload_reads_dataframe() -> None:
    df = make_df([_ok_row(include=1), _ok_row(include=0)])
    model = InventoryTableModel(df)
    inc_col = next(i for i, c in enumerate(COLUMNS) if c.key == "include")
    assert model.data(model.index(0, inc_col), PAYLOAD_ROLE) is True
    assert model.data(model.index(1, inc_col), PAYLOAD_ROLE) is False


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------


def test_setdata_rejects_non_editable_column() -> None:
    df = make_df([_ok_row()])
    model = InventoryTableModel(df)
    id_col = next(i for i, c in enumerate(COLUMNS) if c.key == "id")
    assert model.setData(model.index(0, id_col), "999") is False


def test_setdata_on_task_rebuilds_basename(qtbot) -> None:
    df = make_df([_func_row()])
    model = InventoryTableModel(df)
    task_col = next(i for i, c in enumerate(COLUMNS) if c.key == "task")
    bn_col = next(i for i, c in enumerate(COLUMNS) if c.key == "basename")
    # Edit task=rest → task=motor; basename should regenerate.
    with qtbot.waitSignal(model.dataChanged, timeout=500):
        assert model.setData(model.index(0, task_col), "motor") is True
    new_bn = model.data(model.index(0, bn_col), Qt.ItemDataRole.DisplayRole)
    assert "task-motor" in new_bn
    assert "task-rest" not in new_bn


def test_setdata_same_value_is_noop() -> None:
    df = make_df([_func_row()])
    model = InventoryTableModel(df)
    task_col = next(i for i, c in enumerate(COLUMNS) if c.key == "task")
    assert model.setData(model.index(0, task_col), "rest") is False


def test_toggling_include_changes_row_state_to_skip() -> None:
    df = make_df([_ok_row()])
    model = InventoryTableModel(df)
    inc_col = next(i for i, c in enumerate(COLUMNS) if c.key == "include")
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == ""
    assert model.setData(model.index(0, inc_col), False) is True
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == "skip"


# ---------------------------------------------------------------------------
# Project integration
# ---------------------------------------------------------------------------


def test_edit_appends_user_set_cell_event_to_project(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    project.append(ScanImported(inventory_tsv="/x", row_ids=("1.2.3.4",)))

    df = make_df([_func_row()])
    model = InventoryTableModel(df, project=project)

    task_col = next(i for i, c in enumerate(COLUMNS) if c.key == "task")
    model.setData(model.index(0, task_col), "motor")

    events = list(project.log)
    set_cell_events = [e for e in events if isinstance(e, UserSetCell)]
    assert len(set_cell_events) == 1
    ev = set_cell_events[0]
    assert ev.row_id == "1.2.3.4"
    assert ev.column == "task"
    assert ev.value == "motor"
    assert ev.previous == "rest"


def test_include_toggle_appends_user_toggle_include_event(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    df = make_df([_ok_row()])
    model = InventoryTableModel(df, project=project)

    inc_col = next(i for i, c in enumerate(COLUMNS) if c.key == "include")
    model.setData(model.index(0, inc_col), False)

    toggles = [e for e in project.log if isinstance(e, UserToggleInclude)]
    assert len(toggles) == 1
    assert toggles[0].row_id == "1.2.3.4"
    assert toggles[0].include is False


def test_project_overlay_applied_on_load(tmp_path: Path) -> None:
    """Reopening a project should reapply previous cell + include edits."""
    project = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    project.append(ScanImported(inventory_tsv="/x", row_ids=("1.2.3.4",)))
    project.append(UserSetCell(row_id="1.2.3.4", column="task", value="motor", previous="rest"))
    project.append(UserToggleInclude(row_id="1.2.3.4", include=False))

    # Fresh model from the original DataFrame — overlay should apply.
    df = make_df([_func_row()])
    model = InventoryTableModel(df, project=project)

    task_col = next(i for i, c in enumerate(COLUMNS) if c.key == "task")
    bn_col = next(i for i, c in enumerate(COLUMNS) if c.key == "basename")
    inc_col = next(i for i, c in enumerate(COLUMNS) if c.key == "include")
    assert model.data(model.index(0, task_col), Qt.ItemDataRole.DisplayRole) == "motor"
    assert "task-motor" in model.data(model.index(0, bn_col), Qt.ItemDataRole.DisplayRole)
    assert model.data(model.index(0, inc_col), PAYLOAD_ROLE) is False
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == "skip"


def test_overlay_with_no_matching_row_id_is_safe(tmp_path: Path) -> None:
    """A project that references rows not present in the TSV is ignored."""
    project = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    project.append(UserSetCell(row_id="absent-row", column="task", value="x"))
    df = make_df([_func_row()])
    model = InventoryTableModel(df, project=project)  # must not raise


# ---------------------------------------------------------------------------
# Integration: model inside a real QTableView
# ---------------------------------------------------------------------------


def test_checkbox_delegate_toggle_on_click(qtbot) -> None:
    """A left-mouse click on the include column flips the cell."""
    from PyQt6.QtCore import QEvent, QPoint, Qt
    from PyQt6.QtGui import QMouseEvent

    df = make_df([_ok_row(include=1)])
    model = InventoryTableModel(df)
    view = QTableView()
    view.setModel(model)
    qtbot.addWidget(view)

    inc_col = next(i for i, c in enumerate(COLUMNS) if c.key == "include")
    view.setItemDelegateForColumn(inc_col, CheckboxDelegate(view))
    view.resize(800, 100)
    view.show()
    qtbot.waitExposed(view)

    cell_rect = view.visualRect(model.index(0, inc_col))
    center = cell_rect.center()
    # Simulate a click on the cell's viewport. ``mouseClick`` uses the
    # widget's coordinate space; ``visualRect`` is already in that space.
    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=center)
    assert model.data(model.index(0, inc_col), PAYLOAD_ROLE) is False

    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=center)
    assert model.data(model.index(0, inc_col), PAYLOAD_ROLE) is True


def test_model_renders_in_real_qtableview(qtbot) -> None:
    df = make_df([_ok_row(), _func_row(), _physio_row(), _err_row(), _skip_row()])
    model = InventoryTableModel(df)

    view = QTableView()
    view.setModel(model)
    qtbot.addWidget(view)

    # Wire delegates per spec — same as the eventual ConverterView will.
    for col, spec in enumerate(COLUMNS):
        if spec.role == "checkbox":
            view.setItemDelegateForColumn(col, CheckboxDelegate(view))
        elif spec.role == "status":
            view.setItemDelegateForColumn(col, StatusDelegate(view))
        else:
            view.setItemDelegateForColumn(col, CellTextDelegate(spec.role, view))

    view.resize(900, 200)
    view.show()
    qtbot.waitExposed(view)
    # If we got here, the model + delegates rendered with no Qt errors.
    assert view.model() is model
    assert view.model().rowCount() == 5
