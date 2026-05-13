"""Tests for the M5.7 side panes (RawFsPane + FilterPane).

Both are bound to the same :class:`InventoryTableModel` the table uses,
so the asserts ride the same code path the real app does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PyQt6.QtCore import Qt

from bidsmgr.gui.filter_pane import FilterPane
from bidsmgr.gui.models import COLUMNS, InventoryTableModel
from bidsmgr.gui.raw_fs_pane import RawFsPane


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _eeg_row(**overrides) -> dict:
    base = {
        "BIDS_name": "sub-001",
        "session": "",
        "include": 1,
        "modality": "eeg",
        "modality_bids": "eeg",
        "sequence": "",
        "series_uid": "",
        "proposed_datatype": "eeg",
        "proposed_basename": "sub-001_task-rest_eeg",
        "Proposed BIDS name": "sub-001_task-rest_eeg",
        "bids_guess_classifier": "mne",
        "bids_guess_datatype": "eeg",
        "bids_guess_suffix": "eeg",
        "bids_guess_confidence": "0.97",
        "bids_guess_skip": False,
        "proposed_issues": "",
        "entities": json.dumps({"subject": "001", "task": "rest"}, sort_keys=True),
        "task": "rest",
        "run": "",
        "source_file": "",
        "dataset": "Demo",
    }
    base.update(overrides)
    return base


def _func_row(**overrides) -> dict:
    base = {
        "BIDS_name": "sub-001",
        "session": "ses-pre",
        "include": 1,
        "modality": "mri",
        "modality_bids": "func",
        "sequence": "bold_rest",
        "series_uid": "1.2.3.4",
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
            {"subject": "001", "session": "pre", "task": "rest"}, sort_keys=True,
        ),
        "task": "rest",
        "run": "",
        "source_file": "",
        "dataset": "Demo",
    }
    base.update(overrides)
    return base


def make_df(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RawFsPane
# ---------------------------------------------------------------------------


def test_raw_fs_pane_starts_empty(qtbot) -> None:
    pane = RawFsPane()
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    assert pane._empty.isVisible()
    assert not pane._tree.isVisible()


def test_raw_fs_pane_populates_from_root(qtbot, tmp_path: Path) -> None:
    # Build a tiny directory tree.
    (tmp_path / "sub-001" / "ses-pre" / "anat").mkdir(parents=True)
    (tmp_path / "sub-001" / "ses-pre" / "anat" / "T1w.nii.gz").write_bytes(b"")
    (tmp_path / "sub-002").mkdir()
    (tmp_path / "sub-002" / "info.txt").write_text("hi")

    pane = RawFsPane()
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    pane.set_root(tmp_path)

    assert pane._tree.isVisible()
    assert not pane._empty.isVisible()

    def walk(item):
        yield item.text(0)
        for i in range(item.childCount()):
            yield from walk(item.child(i))

    labels: list[str] = []
    for i in range(pane._tree.topLevelItemCount()):
        labels.extend(walk(pane._tree.topLevelItem(i)))
    assert any("sub-001" in lbl for lbl in labels)
    assert any("sub-002" in lbl for lbl in labels)
    assert any("info.txt" in lbl for lbl in labels)


def test_raw_fs_pane_clears_on_none(qtbot, tmp_path: Path) -> None:
    pane = RawFsPane()
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    pane.set_root(tmp_path)
    pane.set_root(None)
    assert pane._empty.isVisible()
    assert pane._tree.topLevelItemCount() == 0


def test_raw_fs_pane_skips_dot_directories(qtbot, tmp_path: Path) -> None:
    (tmp_path / ".bidsmgr").mkdir()
    (tmp_path / "real_subject").mkdir()
    pane = RawFsPane()
    qtbot.addWidget(pane)
    pane.set_root(tmp_path)

    def walk(item):
        yield item.text(0)
        for i in range(item.childCount()):
            yield from walk(item.child(i))

    labels: list[str] = []
    for i in range(pane._tree.topLevelItemCount()):
        labels.extend(walk(pane._tree.topLevelItem(i)))
    assert any("real_subject" in l for l in labels)
    assert not any(".bidsmgr" in l for l in labels)


# ---------------------------------------------------------------------------
# FilterPane
# ---------------------------------------------------------------------------


def test_filter_pane_starts_empty(qtbot) -> None:
    pane = FilterPane()
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    assert pane._empty.isVisible()
    assert not pane._tree.isVisible()


def test_filter_pane_builds_tree_from_model(qtbot) -> None:
    df = make_df([
        _func_row(),
        _func_row(BIDS_name="sub-002", session="ses-post",
                  proposed_basename="sub-002_ses-post_task-rest_bold",
                  series_uid="9.9.9"),
        _func_row(BIDS_name="sub-002", session="ses-post",
                  proposed_datatype="anat", bids_guess_suffix="T1w",
                  proposed_basename="sub-002_ses-post_T1w",
                  series_uid="8.8.8", task=""),
    ])
    model = InventoryTableModel(df)
    pane = FilterPane()
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    pane.bind_model(model)

    assert pane._tree.isVisible()
    # Root has one dataset; expand and verify the dataset has both subjects.
    assert pane._tree.topLevelItemCount() == 1
    ds_node = pane._tree.topLevelItem(0)
    sub_labels = [ds_node.child(i).text(0) for i in range(ds_node.childCount())]
    assert any("sub-001" in lbl for lbl in sub_labels)
    assert any("sub-002" in lbl for lbl in sub_labels)


def test_filter_pane_unchecking_propagates_to_model(qtbot) -> None:
    df = make_df([_func_row(), _func_row(series_uid="9.9.9")])
    model = InventoryTableModel(df)
    pane = FilterPane()
    qtbot.addWidget(pane)
    pane.bind_model(model)

    ds_node = pane._tree.topLevelItem(0)
    # Uncheck the dataset node — Qt cascades to every leaf.
    ds_node.setCheckState(0, Qt.CheckState.Unchecked)

    inc_col = next(i for i, c in enumerate(COLUMNS) if c.key == "include")
    # Every row should now be excluded.
    assert model.data(model.index(0, inc_col), Qt.ItemDataRole.UserRole) is False
    assert model.data(model.index(1, inc_col), Qt.ItemDataRole.UserRole) is False


def test_filter_pane_partial_state_when_rows_mixed(qtbot) -> None:
    df = make_df([
        _func_row(include=1),
        _func_row(include=0, series_uid="9.9.9"),
    ])
    model = InventoryTableModel(df)
    pane = FilterPane()
    qtbot.addWidget(pane)
    pane.bind_model(model)

    # New shape: ds → sub → ses → datatype (parent) → sequence (leaves).
    # The datatype parent has two child sequences with disagreeing
    # include flags → tri-state partial via Qt's ItemIsAutoTristate.
    ds = pane._tree.topLevelItem(0)
    sub = ds.child(0)
    ses = sub.child(0)
    datatype_parent = ses.child(0)
    # Datatype parent has 2 sequence-leaf children.
    assert datatype_parent.childCount() == 2
    assert datatype_parent.checkState(0) == Qt.CheckState.PartiallyChecked


def test_filter_pane_per_sequence_leaves_show_basenames(qtbot) -> None:
    """Each sequence under a datatype is its own leaf labeled by
    ``proposed_basename`` (or fallback)."""
    df = make_df([
        _func_row(proposed_basename="sub-001_ses-pre_task-rest_bold", series_uid="1.1"),
        _func_row(proposed_basename="sub-001_ses-pre_task-mb_bold",   series_uid="2.2"),
    ])
    model = InventoryTableModel(df)
    pane = FilterPane()
    qtbot.addWidget(pane)
    pane.bind_model(model)

    ds = pane._tree.topLevelItem(0)
    sub = ds.child(0)
    ses = sub.child(0)
    dt = ses.child(0)
    labels = [dt.child(i).text(0) for i in range(dt.childCount())]
    assert "sub-001_ses-pre_task-rest_bold" in labels
    assert "sub-001_ses-pre_task-mb_bold" in labels


def test_filter_pane_unchecking_one_sequence_only_toggles_that_row(qtbot) -> None:
    """Unticking a single sequence leaf must only affect THAT row's
    include flag — not the whole datatype group.
    """
    df = make_df([
        _func_row(proposed_basename="sub-001_ses-pre_task-rest_bold",
                  series_uid="1.1"),
        _func_row(proposed_basename="sub-001_ses-pre_task-mb_bold",
                  series_uid="2.2"),
    ])
    model = InventoryTableModel(df)
    pane = FilterPane()
    qtbot.addWidget(pane)
    pane.bind_model(model)

    ds = pane._tree.topLevelItem(0)
    sub = ds.child(0)
    ses = sub.child(0)
    dt = ses.child(0)

    # Find the leaf for the first basename and uncheck it.
    target = None
    for i in range(dt.childCount()):
        if dt.child(i).text(0) == "sub-001_ses-pre_task-rest_bold":
            target = dt.child(i)
            break
    assert target is not None
    target.setCheckState(0, Qt.CheckState.Unchecked)

    # Row 0 now excluded; row 1 still included.
    assert model._read_include(0) is False
    assert model._read_include(1) is True
