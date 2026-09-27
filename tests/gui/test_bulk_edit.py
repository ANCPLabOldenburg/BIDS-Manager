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
        "participant_id": "sub-001",
        "session": "ses-pre",
        "include": 1,
        "modality": "mri",
        "datatype": "func",
        "bids_name": "sub-001_ses-pre_task-rest_bold",
        "bids_path": "sub-001_ses-pre_task-rest_bold",
        "bids_guess_classifier": "dcm2niix_bidsguess",
        "bids_guess_datatype": "func",
        "bids_guess_suffix": "bold",
        "bids_guess_confidence": "0.97",
        "bids_guess_skip": False,
        "issues": "",
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
        _row(participant_id="sub-002", series_uid="2"),
        _row(participant_id="sub-003", series_uid="3"),
    ])
    model = InventoryTableModel(df)
    n = model.bulk_set([0, 1, 2], "id", "099")
    assert n == 3
    out = model.dataframe()
    assert (out["participant_id"] == "sub-099").all()
    # The basename column reflects the new subject across all rows.
    assert out["bids_name"].str.startswith("sub-099").all()
    # Entities JSON was updated too.
    for ent in out["entities"]:
        assert json.loads(ent)["subject"] == "099"


def test_bulk_set_dataset_is_blocked() -> None:
    # The dataset column is owned by the project / locked output folder and is
    # read-only; bulk-editing it must be a no-op (nothing changed).
    df = make_df([_row(series_uid=str(i)) for i in range(3)])
    model = InventoryTableModel(df)
    assert "dataset" not in model.BULK_EDITABLE_KEYS
    n = model.bulk_set([0, 1, 2], "dataset", "other_study")
    assert n == 0
    assert (model.dataframe()["dataset"] == "study").all()  # unchanged


def test_bulk_set_task_rebuilds_basename() -> None:
    df = make_df([_row(series_uid="1"), _row(series_uid="2")])
    model = InventoryTableModel(df)
    n = model.bulk_set([0, 1], "task", "motor")
    assert n == 2
    out = model.dataframe()
    assert (out["task"] == "motor").all()
    assert out["bids_name"].str.contains("task-motor").all()


def test_bulk_set_datatype_preserves_per_row_suffix() -> None:
    """When changing datatype, the original suffix for each row is kept."""
    df = make_df([
        _row(datatype="func", bids_guess_suffix="bold", series_uid="1"),
        _row(datatype="anat", bids_guess_suffix="T1w", series_uid="2",
             bids_name="sub-001_ses-pre_T1w"),
    ])
    model = InventoryTableModel(df)
    # Change both to "func" — the second row's suffix stays "T1w".
    n = model.bulk_set([0, 1], "datatype", "func")
    assert n >= 1  # at least the second row changed datatype
    out = model.dataframe()
    assert (out["datatype"] == "func").all()
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
    # Pick the task column from the dropdown (find by user data). The dataset
    # column is intentionally not offered (read-only / project-owned).
    for i in range(dlg._col_combo.count()):
        if dlg._col_combo.itemData(i) == "task":
            dlg._col_combo.setCurrentIndex(i)
            break
    dlg._value_edit.setText("memory")
    dlg._on_apply()

    assert dlg.changed_count() == 3
    assert (model.dataframe()["task"] == "memory").all()


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


# ---------------------------------------------------------------------------
# Constrained-choice columns -> dropdown-only (never free-typed)
# ---------------------------------------------------------------------------


def _select_column(dlg, key) -> None:
    for i in range(dlg._col_combo.count()):
        if dlg._col_combo.itemData(i) == key:
            dlg._col_combo.setCurrentIndex(i)
            break
    dlg._on_column_changed(dlg._col_combo.currentIndex())


def test_dialog_line_freq_is_fixed_dropdown(qtbot) -> None:
    df = make_df([_row(datatype="eeg", series_uid="", source_file="a.edf",
                       line_freq="", bids_name="sub-001_task-rest_eeg")])
    model = InventoryTableModel(df)
    dlg = BulkEditDialog(model, rows=[0])
    qtbot.addWidget(dlg)
    _select_column(dlg, "line_freq")
    assert dlg._value_is_combo is True
    assert dlg._value_combo.isEditable() is False  # dropdown-only
    items = [dlg._value_combo.itemText(i) for i in range(dlg._value_combo.count())]
    assert items == ["50", "60"]


def test_dialog_montage_is_dropdown(qtbot) -> None:
    df = make_df([_row(datatype="eeg", series_uid="", source_file="a.edf",
                       montage="", bids_name="sub-001_task-rest_eeg")])
    model = InventoryTableModel(df)
    dlg = BulkEditDialog(model, rows=[0])
    qtbot.addWidget(dlg)
    _select_column(dlg, "montage")
    assert dlg._value_is_combo is True
    assert dlg._value_combo.isEditable() is False
    assert dlg._value_combo.count() > 0


def test_dialog_apply_combo_value(qtbot) -> None:
    df = make_df([
        _row(datatype="eeg", series_uid="", source_file="a.edf",
             line_freq="", bids_name="sub-001_task-rest_eeg"),
        _row(participant_id="sub-002", datatype="eeg", series_uid="",
             source_file="b.edf", line_freq="",
             bids_name="sub-002_task-rest_eeg"),
    ])
    model = InventoryTableModel(df)
    dlg = BulkEditDialog(model, rows=[0, 1])
    qtbot.addWidget(dlg)
    _select_column(dlg, "line_freq")
    dlg._value_combo.setCurrentText("60")
    dlg._on_apply()
    assert dlg.changed_count() == 2
    assert (model.dataframe()["line_freq"] == "60").all()


# ---------------------------------------------------------------------------
# Targeting: the selection is where it starts, not what it does
#
# It used to write into every selected row, full stop, so "change task-rest
# but leave the localizers alone" meant going back to the table and
# re-selecting. The Editor's rename tool had the better model: find the rows
# that say a particular thing, and change those.


def _mixed_model() -> InventoryTableModel:
    """Four rows: three say task-rest, one says task-nback."""
    return InventoryTableModel(make_df([
        _row(series_uid="1"),
        _row(series_uid="2"),
        _row(series_uid="3"),
        _row(series_uid="4", task="nback",
             entities=json.dumps(
                 {"subject": "001", "session": "pre", "task": "nback"},
                 sort_keys=True,
             ),
             bids_name="sub-001_ses-pre_task-nback_bold"),
    ]))


def _pick_column(dlg: BulkEditDialog, key: str) -> None:
    dlg._col_combo.setCurrentIndex(dlg._col_combo.findData(key))


def _set_value(dlg: BulkEditDialog, value: str) -> None:
    if dlg._value_is_combo:
        dlg._value_combo.setCurrentText(value)
    else:
        dlg._value_edit.setText(value)
    dlg._replan_timer.stop()
    dlg._replan()


def _preview(dlg: BulkEditDialog) -> list[tuple[str, str, str]]:
    return [
        (dlg._preview.topLevelItem(i).text(0),
         dlg._preview.topLevelItem(i).text(1),
         dlg._preview.topLevelItem(i).text(2))
        for i in range(dlg._preview.topLevelItemCount())
    ]


class TestTheModelCanBeRead:
    def test_bulk_value_reads_what_bulk_set_writes(self):
        model = _mixed_model()
        assert model.bulk_value(0, "task") == "rest"
        model.bulk_set([0], "task", "x")
        assert model.bulk_value(0, "task") == "x"

    def test_it_reads_an_entity_with_no_column(self):
        model = _mixed_model()
        model.bulk_set([0], "entity:acq", "fm2")
        assert model.bulk_value(0, "entity:acq") == "fm2"
        assert model.bulk_value(1, "entity:acq") == ""

    def test_id_reads_the_subject_entity(self):
        assert _mixed_model().bulk_value(0, "id") == "001"

    def test_a_row_is_named_by_its_bids_name(self):
        assert _mixed_model().row_label(0) == "sub-001_ses-pre_task-rest_bold"


class TestNarrowingByValue:
    def test_every_value_in_use_is_offered_with_its_count(self, qtbot):
        model = _mixed_model()
        dlg = BulkEditDialog(model, [0, 1, 2, 3])
        qtbot.addWidget(dlg)
        _pick_column(dlg, "task")
        offered = [
            dlg._target_combo.itemText(i)
            for i in range(dlg._target_combo.count())
        ]
        assert "every selected row (4)" in offered
        assert "only the rows that say rest (3)" in offered
        assert "only the rows that say nback (1)" in offered

    def test_picking_one_confines_the_change(self, qtbot):
        model = _mixed_model()
        dlg = BulkEditDialog(model, [0, 1, 2, 3])
        qtbot.addWidget(dlg)
        _pick_column(dlg, "task")
        dlg._target_combo.setCurrentIndex(dlg._target_combo.findData("rest"))
        _set_value(dlg, "restingstate")
        dlg._on_apply()

        tasks = list(model.dataframe()["task"])
        assert tasks == ["restingstate", "restingstate", "restingstate", "nback"]

    def test_rows_that_say_nothing_are_their_own_choice(self, qtbot):
        """Blank is a value people want to fill, and an empty string in a
        dropdown is invisible."""
        model = InventoryTableModel(make_df([
            _row(series_uid="1"),
            _row(series_uid="2", run="", task="rest"),
        ]))
        dlg = BulkEditDialog(model, [0, 1])
        qtbot.addWidget(dlg)
        _pick_column(dlg, "run")
        offered = [
            dlg._target_combo.itemText(i)
            for i in range(dlg._target_combo.count())
        ]
        assert any("say nothing (2)" in o for o in offered)

    def test_everything_is_the_default(self, qtbot):
        """Opening it and pressing Apply must behave as it always did."""
        model = _mixed_model()
        dlg = BulkEditDialog(model, [0, 1, 2, 3])
        qtbot.addWidget(dlg)
        _pick_column(dlg, "task")
        _set_value(dlg, "x")
        dlg._on_apply()
        assert list(model.dataframe()["task"]) == ["x"] * 4


class TestThePreview:
    def test_it_says_what_each_row_says_now_and_would_say(self, qtbot):
        model = _mixed_model()
        dlg = BulkEditDialog(model, [0, 3])
        qtbot.addWidget(dlg)
        _pick_column(dlg, "task")
        _set_value(dlg, "x")
        assert _preview(dlg) == [
            ("sub-001_ses-pre_task-rest_bold", "rest", "x"),
            ("sub-001_ses-pre_task-nback_bold", "nback", "x"),
        ]

    def test_a_row_that_already_says_it_is_not_listed(self, qtbot):
        """It would be a no-op, and a preview full of no-ops hides the
        rows that do change."""
        model = _mixed_model()
        dlg = BulkEditDialog(model, [0, 1, 2, 3])
        qtbot.addWidget(dlg)
        _pick_column(dlg, "task")
        _set_value(dlg, "rest")
        assert [row[0] for row in _preview(dlg)] == [
            "sub-001_ses-pre_task-nback_bold"
        ]

    def test_unticking_a_row_leaves_it_alone(self, qtbot):
        """The case no filter can express: these three, but not that one."""
        from PyQt6.QtCore import Qt

        model = _mixed_model()
        dlg = BulkEditDialog(model, [0, 1, 2, 3])
        qtbot.addWidget(dlg)
        _pick_column(dlg, "task")
        _set_value(dlg, "x")
        dlg._preview.topLevelItem(1).setCheckState(0, Qt.CheckState.Unchecked)
        dlg._on_apply()

        tasks = list(model.dataframe()["task"])
        assert tasks.count("x") == 3
        assert tasks[1] == "rest"

    def test_apply_is_dead_with_nothing_ticked(self, qtbot):
        from PyQt6.QtCore import Qt

        model = _mixed_model()
        dlg = BulkEditDialog(model, [0, 1, 2, 3])
        qtbot.addWidget(dlg)
        _pick_column(dlg, "task")
        _set_value(dlg, "x")
        dlg._set_all(Qt.CheckState.Unchecked)
        assert not dlg._apply_btn.isEnabled()
        dlg._on_apply()
        assert list(model.dataframe()["task"]) == ["rest"] * 3 + ["nback"]

    def test_the_button_says_how_many(self, qtbot):
        model = _mixed_model()
        dlg = BulkEditDialog(model, [0, 1, 2, 3])
        qtbot.addWidget(dlg)
        _pick_column(dlg, "task")
        _set_value(dlg, "x")
        assert "4" in dlg._apply_btn.text()

    def test_a_value_typed_inside_the_debounce_still_counts(self, qtbot):
        """Typing and pressing Apply within the delay would otherwise write
        the plan from BEFORE the value was typed."""
        model = _mixed_model()
        dlg = BulkEditDialog(model, [0, 1, 2, 3])
        qtbot.addWidget(dlg)
        _pick_column(dlg, "task")
        if dlg._value_is_combo:
            dlg._value_combo.setCurrentText("typed")
        else:
            dlg._value_edit.setText("typed")
        assert dlg._replan_timer.isActive()
        dlg._on_apply()
        assert list(model.dataframe()["task"]) == ["typed"] * 4
