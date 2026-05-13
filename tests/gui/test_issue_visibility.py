"""Tests for feature #3 — issue visibility.

Three concerns:

* :class:`Chip` is clickable when explicitly opted in.
* :class:`IssuesDialog` lists exactly the rows of a given severity
  and emits ``row_selected`` with the inventory row index on
  double-click.
* :class:`PropertiesPanel` renders one ``ValMessage`` per
  scanner-detected issue on the selected row's ``proposed_issues``.
"""

from __future__ import annotations

import json
from typing import Iterable

import pandas as pd
import pytest
from PyQt6.QtCore import Qt

from bidsmgr.gui.converter_panel import ConverterPanel
from bidsmgr.gui.issues_dialog import IssuesDialog
from bidsmgr.gui.models import COLUMNS, InventoryTableModel
from bidsmgr.gui.properties_panel import PropertiesPanel
from bidsmgr.gui.widgets import Chip, ValMessage


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Fixture rows
# ---------------------------------------------------------------------------


def _func_row(**overrides) -> dict:
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
    }
    base.update(overrides)
    return base


def _warn_row(**overrides) -> dict:
    # Generic non-err-token issue so ``_derive_row_state`` keeps this
    # at ``warn``. The model's err-token list contains
    # ``suspected_abort``/``required``/``build_basename``/``missing``.
    base = dict(
        proposed_issues="rerouted to fmap/epi: smaller than DWI peer",
        series_uid="2.2.2",
        BIDS_name="sub-002",
    )
    base.update(overrides)
    return _func_row(**base)


def _err_row(**overrides) -> dict:
    return _func_row(
        proposed_basename="",
        proposed_datatype="",
        proposed_issues="task entity required for func/bold",
        series_uid="9.9.9",
        BIDS_name="sub-003",
        **overrides,
    )


def _skip_row(**overrides) -> dict:
    return _func_row(
        include=0,
        proposed_issues="",
        series_uid="8.8.8",
        BIDS_name="sub-004",
        **overrides,
    )


def make_df(rows: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# Chip
# ---------------------------------------------------------------------------


def test_chip_clicked_signal_only_fires_when_set_clickable(qtbot) -> None:
    chip = Chip("3 warnings", "warn")
    qtbot.addWidget(chip)
    chip.resize(80, 24)
    chip.show()
    qtbot.waitExposed(chip)

    received: list = []
    chip.clicked.connect(lambda: received.append(True))

    # Without set_clickable(True) the chip is passive.
    qtbot.mouseClick(chip, Qt.MouseButton.LeftButton)
    assert received == []

    chip.set_clickable(True)
    qtbot.mouseClick(chip, Qt.MouseButton.LeftButton)
    assert received == [True]


# ---------------------------------------------------------------------------
# IssuesDialog
# ---------------------------------------------------------------------------


def test_issues_dialog_lists_only_rows_of_requested_severity(qtbot) -> None:
    from bidsmgr.gui.issues_dialog import _RowCard
    df = make_df([_func_row(), _warn_row(), _err_row(), _skip_row()])
    model = InventoryTableModel(df)
    dlg = IssuesDialog(model, severity="warn")
    qtbot.addWidget(dlg)

    # One card per matching row; read each card's title button.
    cards = dlg.findChildren(_RowCard)
    titles = [c._title_btn.text() for c in cards]
    assert any("sub-002" in t for t in titles)
    assert not any("sub-003" in t for t in titles)  # err
    assert not any("sub-004" in t for t in titles)  # skip


def test_issues_dialog_shows_none_when_no_matches(qtbot) -> None:
    from bidsmgr.gui.issues_dialog import _RowCard
    from PyQt6.QtWidgets import QLabel
    df = make_df([_func_row()])
    model = InventoryTableModel(df)
    dlg = IssuesDialog(model, severity="err")
    qtbot.addWidget(dlg)
    assert dlg.findChildren(_RowCard) == []
    # The "(no matching rows)" hint label is present.
    hints = [
        lbl for lbl in dlg.findChildren(QLabel)
        if lbl.text() == "(no matching rows)"
    ]
    assert hints, "empty state hint should be visible"


def test_issues_dialog_emits_row_selected_on_activation(qtbot) -> None:
    from bidsmgr.gui.issues_dialog import _RowCard
    df = make_df([_func_row(), _warn_row()])
    model = InventoryTableModel(df)
    dlg = IssuesDialog(model, severity="warn")
    qtbot.addWidget(dlg)

    captured: list[int] = []
    dlg.row_selected.connect(captured.append)
    cards = dlg.findChildren(_RowCard)
    assert len(cards) == 1
    # Activating the card's title button fires the signal.
    cards[0]._title_btn.click()
    assert captured == [1]  # row 1 is the warn row (row 0 is the ok one)


def test_issues_dialog_renders_one_valmessage_per_pipe_separated_issue(qtbot) -> None:
    from bidsmgr.gui.issues_dialog import _RowCard
    df = make_df([_warn_row(
        proposed_issues="rerouted to fmap/epi: foo | fmap multi-output: bar",
    )])
    model = InventoryTableModel(df)
    dlg = IssuesDialog(model, severity="warn")
    qtbot.addWidget(dlg)
    cards = dlg.findChildren(_RowCard)
    assert len(cards) == 1
    # Each issue is its own ValMessage inside the card.
    assert len(cards[0].findChildren(ValMessage)) == 2


# ---------------------------------------------------------------------------
# PropertiesPanel — row issues section
# ---------------------------------------------------------------------------


def test_properties_panel_shows_one_valmessage_per_issue(qtbot) -> None:
    df = make_df([_warn_row(
        proposed_issues="rerouted to fmap/epi: first reason | fmap multi-output: second reason",
    )])
    model = InventoryTableModel(df)
    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    panel.bind_model(model)
    panel.set_selected_row(0)

    # The scanner-issue ValMessages live in the panel's body. Each one
    # is a QFrame whose body label contains the issue text.
    from PyQt6.QtWidgets import QLabel
    vmsgs = panel.findChildren(ValMessage)
    bodies: list[str] = []
    for vm in vmsgs:
        for lbl in vm.findChildren(QLabel):
            bodies.append(lbl.text())
    # Both issues surfaced (anywhere in the rendered text).
    assert any("rerouted to fmap" in t for t in bodies)
    assert any("fmap multi-output" in t for t in bodies)


def test_properties_panel_omits_issue_section_when_no_issues(qtbot) -> None:
    df = make_df([_func_row()])
    model = InventoryTableModel(df)
    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    panel.bind_model(model)
    panel.set_selected_row(0)

    # Only the schema-validation ValMessage should be present; no
    # scanner-issue messages because proposed_issues is empty.
    from PyQt6.QtWidgets import QLabel
    vmsgs = panel.findChildren(ValMessage)
    bodies = []
    for vm in vmsgs:
        for lbl in vm.findChildren(QLabel):
            bodies.append(lbl.text())
    assert not any("rerouted to fmap" in t for t in bodies)


# ---------------------------------------------------------------------------
# ConverterPanel — chip click opens the dialog
# ---------------------------------------------------------------------------


def test_warn_chip_is_clickable(qtbot) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    assert panel._chip_warn._clickable is True
    assert panel._chip_err._clickable is True
    assert panel._chip_skip._clickable is True
    # The "valid" chip stays passive — nothing to drill into.
    assert panel._chip_valid._clickable is False


def test_open_issues_dialog_is_a_noop_without_a_model(qtbot) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    # Must not raise / open anything when there's no model yet.
    panel._open_issues_dialog("warn")


def test_jump_to_row_selects_the_table_row(qtbot, tmp_path) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    df = make_df([_func_row(), _warn_row()])
    panel.load_inventory(df, output_tsv=tmp_path / "inv.tsv")
    panel.show()
    qtbot.waitExposed(panel)

    panel._jump_to_row(1)
    assert panel._table.currentIndex().row() == 1
