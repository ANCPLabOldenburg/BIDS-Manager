"""Tests for the Manage-columns dialog + column resize behaviour."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from PyQt6.QtWidgets import QHeaderView

from bidsmgr.gui.column_manager_dialog import ColumnManagerDialog
from bidsmgr.gui.converter_panel import ConverterPanel
from bidsmgr.gui.models import COLUMN_DESCRIPTIONS, COLUMNS, MANDATORY_COLUMN_KEYS

pytestmark = pytest.mark.gui


def test_every_column_has_a_description() -> None:
    missing = [c.key for c in COLUMNS if not COLUMN_DESCRIPTIONS.get(c.key)]
    assert missing == [], f"columns without a description: {missing}"


def test_dialog_result_keeps_mandatory_visible(qtbot) -> None:
    current = {c.key: c.default_visible for c in COLUMNS}
    dlg = ColumnManagerDialog(current, None)
    qtbot.addWidget(dlg)
    # Try to hide everything; mandatory columns must stay on.
    dlg._set_all(False)
    res = dlg.result_visibility()
    for key in MANDATORY_COLUMN_KEYS:
        assert res[key] is True


def test_dialog_select_all_and_defaults(qtbot) -> None:
    current = {c.key: c.default_visible for c in COLUMNS}
    dlg = ColumnManagerDialog(current, None)
    qtbot.addWidget(dlg)
    dlg._set_all(True)
    res = dlg.result_visibility()
    assert all(res[c.key] for c in COLUMNS)  # everything on
    dlg._restore_defaults()
    res = dlg.result_visibility()
    for c in COLUMNS:
        if c.key in MANDATORY_COLUMN_KEYS:
            assert res[c.key] is True
        else:
            assert res[c.key] == c.default_visible


def _df() -> pd.DataFrame:
    return pd.DataFrame([{
        "participant_id": "sub-001", "session": "", "include": 1, "modality": "mri",
        "datatype": "anat", "bids_name": "sub-001_T1w",
        "bids_path": "sub-001_T1w", "bids_guess_suffix": "T1w",
        "bids_guess_confidence": "0.9", "bids_guess_skip": False,
        "issues": "", "entities": json.dumps({"subject": "001"}),
        "task": "", "run": "", "series_uid": "1.1", "PatientID": "P1",
        "n_files": "100", "dataset": "D",
    }])


def test_set_columns_visible_applies_and_forces_mandatory(qtbot, isolated_settings) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel.load_inventory(_df(), None)

    mapping = {c.key: c.default_visible for c in COLUMNS}
    mapping["PatientID"] = True   # reveal a hidden column
    mapping["id"] = False         # attempt to hide a mandatory column
    panel.set_columns_visible(mapping)

    idx_pid = next(i for i, c in enumerate(COLUMNS) if c.key == "PatientID")
    idx_id = next(i for i, c in enumerate(COLUMNS) if c.key == "id")
    assert not panel._table.isColumnHidden(idx_pid)   # revealed
    assert not panel._table.isColumnHidden(idx_id)    # mandatory stayed visible
    assert panel._table.columnWidth(idx_pid) > 0      # revealed with a width


def test_studydescription_is_hidden_toggleable_column(qtbot, isolated_settings) -> None:
    # Registered, read-only, hidden by default, and described - so it appears in
    # Manage columns as an off-by-default toggle the user can switch on.
    spec = next((c for c in COLUMNS if c.key == "StudyDescription"), None)
    assert spec is not None
    assert spec.default_visible is False
    assert spec.editable is False
    assert COLUMN_DESCRIPTIONS.get("StudyDescription")
    assert "StudyDescription" not in MANDATORY_COLUMN_KEYS

    # The user can reveal it from Manage columns, and it shows the value.
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    df = _df()
    df["StudyDescription"] = "PPMI"
    panel.load_inventory(df, None)
    mapping = {c.key: c.default_visible for c in COLUMNS}
    mapping["StudyDescription"] = True
    panel.set_columns_visible(mapping)
    idx = next(i for i, c in enumerate(COLUMNS) if c.key == "StudyDescription")
    assert not panel._table.isColumnHidden(idx)


def test_all_columns_are_user_resizable(qtbot) -> None:
    """Every column is Interactive (independently resizable); the table uses
    stretch-last-section + horizontal scroll instead of a greedy stretch
    column, so even the predicted-basename column can be shrunk / widened."""
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel.load_inventory(_df(), None)
    header = panel._table.horizontalHeader()
    for col in range(len(COLUMNS)):
        assert header.sectionResizeMode(col) == QHeaderView.ResizeMode.Interactive
    assert header.stretchLastSection() is True
    assert header.sectionsMovable() is True


def test_column_order_persists_and_restores(qtbot, isolated_settings) -> None:
    """Dragging a header section persists the order; a fresh panel restores it."""
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel.load_inventory(_df(), None)
    header = panel._table.horizontalHeader()
    task_logical = next(i for i, c in enumerate(COLUMNS) if c.key == "task")
    header.moveSection(header.visualIndex(task_logical), 3)

    panel2 = ConverterPanel()
    qtbot.addWidget(panel2)
    panel2.load_inventory(_df(), None)
    assert panel2._table.horizontalHeader().visualIndex(task_logical) == 3


def test_double_click_autofit_does_not_crash(qtbot) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel.load_inventory(_df(), None)
    for col, spec in enumerate(COLUMNS):
        panel._table_resize_column_to_contents(col)  # incl. the stretch col
