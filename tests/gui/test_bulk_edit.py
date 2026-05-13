"""Tests for feature #5 — multi-row bulk edit.

Covers three layers:

* ``InventoryTableModel.bulk_set`` dispatches subject/datatype/suffix/
  mirror-cell writes through the right per-row API so entities +
  basenames stay consistent.
* ``BulkEditDialog`` reads from the combo + line edit, calls
  ``bulk_set``, and reports the count of rows changed.
* ``ConverterPanel`` enables the toolbar button only when ≥ 2 rows
  are selected.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from PyQt6.QtCore import QItemSelection, QItemSelectionModel

from bidsmgr.gui.bulk_edit_dialog import BulkEditDialog
from bidsmgr.gui.converter_panel import ConverterPanel
from bidsmgr.gui.models import COLUMNS, InventoryTableModel


pytestmark = pytest.mark.gui


def _row(**overrides) -> dict:
    base = {
        "BIDS_name": "sub-001",
        "session": "ses-pre",
        "include": 1,
        "modality": "mri",
        "proposed_datatype": "func",
        "proposed_basename": "sub-001_ses-pre_task-rest_bold",
        "Proposed BIDS name": "sub-001_ses-pre_task-rest_bold",
        "bids_guess_classifier": "dcm2niix_bidsguess",
        "bids_guess_datatype": "func",
        "bids_guess_suffix": "bold",
        "bids_guess_confidence": "0.97",
        "bids_guess_skip": False,
        "proposed_issues": "",
        "entities": json.dumps(
            {"subject": "001", "session": "pre", "task": "rest"},
            sort_keys=True,
        ),
        "task": "rest",
        "run": "",
        "source_file": "",
        "series_uid": "1.2.3.4",
        "dataset": "study",
    }
    base.update(overrides)
    return base


def make_df(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model dispatcher
# ---------------------------------------------------------------------------


def test_bulk_set_subject_updates_bids_name_and_basename() -> None:
    df = make_df([
        _row(series_uid="1"),
        _row(BIDS_name="sub-002", series_uid="2"),
        _row(BIDS_name="sub-003", series_uid="3"),
    ])
    model = InventoryTableModel(df)
    n = model.bulk_set([0, 1, 2], "id", "099")
    assert n == 3
    out = model.dataframe()
    assert (out["BIDS_name"] == "sub-099").all()
    # The basename column reflects the new subject across all rows.
    assert out["proposed_basename"].str.startswith("sub-099").all()
    # Entities JSON was updated too.
    for ent in out["entities"]:
        assert json.loads(ent)["subject"] == "099"


def test_bulk_set_dataset_applies_to_every_row() -> None:
    df = make_df([_row(series_uid=str(i)) for i in range(3)])
    model = InventoryTableModel(df)
    n = model.bulk_set([0, 1, 2], "dataset", "other_study")
    assert n == 3
    assert (model.dataframe()["dataset"] == "other_study").all()


def test_bulk_set_task_rebuilds_basename() -> None:
    df = make_df([_row(series_uid="1"), _row(series_uid="2")])
    model = InventoryTableModel(df)
    n = model.bulk_set([0, 1], "task", "motor")
    assert n == 2
    out = model.dataframe()
    assert (out["task"] == "motor").all()
    assert out["proposed_basename"].str.contains("task-motor").all()


def test_bulk_set_datatype_preserves_per_row_suffix() -> None:
    """When changing datatype, the original suffix for each row is kept."""
    df = make_df([
        _row(proposed_datatype="func", bids_guess_suffix="bold", series_uid="1"),
        _row(proposed_datatype="anat", bids_guess_suffix="T1w", series_uid="2",
             proposed_basename="sub-001_ses-pre_T1w"),
    ])
    model = InventoryTableModel(df)
    # Change both to "func" — the second row's suffix stays "T1w".
    n = model.bulk_set([0, 1], "datatype", "func")
    assert n >= 1  # at least the second row changed datatype
    out = model.dataframe()
    assert (out["proposed_datatype"] == "func").all()
    # Suffix not touched.
    assert out.at[0, "bids_guess_suffix"] == "bold"
    assert out.at[1, "bids_guess_suffix"] == "T1w"


def test_bulk_set_returns_change_count_excluding_noops() -> None:
    df = make_df([
        _row(task="rest", series_uid="1"),
        _row(task="rest", series_uid="2"),  # already "rest" — no-op
    ])
    model = InventoryTableModel(df)
    n = model.bulk_set([0, 1], "task", "rest")
    assert n == 0


def test_bulk_set_rejects_unknown_column_key() -> None:
    df = make_df([_row()])
    model = InventoryTableModel(df)
    assert model.bulk_set([0], "nope", "anything") == 0


# ---------------------------------------------------------------------------
# BulkEditDialog
# ---------------------------------------------------------------------------


def test_dialog_apply_writes_through_bulk_set(qtbot) -> None:
    df = make_df([_row(series_uid=str(i)) for i in range(3)])
    model = InventoryTableModel(df)

    dlg = BulkEditDialog(model, rows=[0, 1, 2])
    qtbot.addWidget(dlg)
    # Pick the dataset column from the dropdown (find by user data).
    for i in range(dlg._col_combo.count()):
        if dlg._col_combo.itemData(i) == "dataset":
            dlg._col_combo.setCurrentIndex(i)
            break
    dlg._value_edit.setText("other_study")
    dlg._on_apply()

    assert dlg.changed_count() == 3
    assert (model.dataframe()["dataset"] == "other_study").all()


def test_dialog_blank_value_is_a_noop(qtbot) -> None:
    df = make_df([_row()])
    model = InventoryTableModel(df)
    dlg = BulkEditDialog(model, rows=[0])
    qtbot.addWidget(dlg)
    dlg._value_edit.setText("   ")
    dlg._on_apply()
    # Dialog stays open (not accepted); nothing changed.
    assert dlg.changed_count() == 0


def test_dialog_datatype_uses_combo_with_schema_values(qtbot) -> None:
    df = make_df([_row()])
    model = InventoryTableModel(df)
    dlg = BulkEditDialog(model, rows=[0])
    qtbot.addWidget(dlg)
    # Select datatype column.
    for i in range(dlg._col_combo.count()):
        if dlg._col_combo.itemData(i) == "datatype":
            dlg._col_combo.setCurrentIndex(i)
            break
    # The combo is now visible; line edit is hidden.
    assert dlg._value_combo.isVisibleTo(dlg)
    assert not dlg._value_edit.isVisibleTo(dlg)
    # Combo populated with at least a few canonical datatypes.
    items = [dlg._value_combo.itemText(i) for i in range(dlg._value_combo.count())]
    assert "anat" in items
    assert "func" in items


# ---------------------------------------------------------------------------
# ConverterPanel — selection-driven enable
# ---------------------------------------------------------------------------


def test_bulk_btn_disabled_by_default(qtbot, tmp_path) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    df = make_df([_row(series_uid=str(i)) for i in range(3)])
    panel.load_inventory(df, output_tsv=tmp_path / "inv.tsv")
    # Just one row default-selected by load_inventory.
    assert panel._bulk_btn.isEnabled() is False


def test_bulk_btn_enables_with_multi_selection(qtbot, tmp_path) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    df = make_df([_row(series_uid=str(i)) for i in range(3)])
    panel.load_inventory(df, output_tsv=tmp_path / "inv.tsv")

    sel = panel._table.selectionModel()
    # Programmatically select rows 0 and 2.
    for row in (0, 2):
        idx = panel._model.index(row, 0)
        sel.select(
            QItemSelection(idx, panel._model.index(row, panel._model.columnCount() - 1)),
            QItemSelectionModel.SelectionFlag.Select,
        )
    assert sorted(panel._selected_rows()) == [0, 2]
    assert panel._bulk_btn.isEnabled() is True
