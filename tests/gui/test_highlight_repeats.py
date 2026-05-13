"""Tests for feature #4 — "Highlight aborts" toolbar toggle.

Three concerns:

* ``InventoryTableModel.is_row_aborted`` flags only the rows the
  scanner marked as ``suspected_abort`` (planned / trivial / isolated
  rows stay un-highlighted).
* ``set_highlight_aborts`` publishes the role across the table and
  is reflected by ``data(idx, HIGHLIGHT_ROLE)``.
* The ``ConverterPanel`` toolbar toggle propagates to the model and
  persists across instances via ``QSettings``.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from bidsmgr.gui.app_settings import AppSettings
from bidsmgr.gui.converter_panel import ConverterPanel
from bidsmgr.gui.delegates import HIGHLIGHT_ROLE
from bidsmgr.gui.models import InventoryTableModel


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
        "repetition_type": "",
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


def make_df(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model classification — only ``suspected_abort`` is treated as aborted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind,expected", [
    ("suspected_abort",  True),   # the only kind that counts
    ("planned",          False),  # legitimate repeats stay un-highlighted
    ("trivial",          False),  # derivative outputs are not aborts
    ("isolated",         False),
    ("",                 False),
])
def test_is_row_aborted_only_matches_suspected_abort(kind: str, expected: bool) -> None:
    df = make_df([_row(repetition_type=kind)])
    model = InventoryTableModel(df)
    assert model.is_row_aborted(0) is expected


def test_highlight_role_off_by_default() -> None:
    df = make_df([_row(repetition_type="suspected_abort")])
    model = InventoryTableModel(df)
    # Even though row 0 is an abort, the role is False until enabled.
    assert model.data(model.index(0, 0), HIGHLIGHT_ROLE) is False


def test_set_highlight_aborts_emits_role_only_for_aborts(qtbot) -> None:
    df = make_df([
        _row(repetition_type="suspected_abort", series_uid="1.1"),
        _row(repetition_type="planned",         series_uid="2.2"),
        _row(repetition_type="isolated",        series_uid="3.3"),
        _row(repetition_type="suspected_abort", series_uid="4.4"),
    ])
    model = InventoryTableModel(df)
    with qtbot.waitSignal(model.dataChanged, timeout=500):
        model.set_highlight_aborts(True)
    assert model.data(model.index(0, 0), HIGHLIGHT_ROLE) is True   # abort
    assert model.data(model.index(1, 0), HIGHLIGHT_ROLE) is False  # planned
    assert model.data(model.index(2, 0), HIGHLIGHT_ROLE) is False  # isolated
    assert model.data(model.index(3, 0), HIGHLIGHT_ROLE) is True   # abort


def test_set_highlight_aborts_idempotent_no_emit(qtbot) -> None:
    df = make_df([_row(repetition_type="suspected_abort")])
    model = InventoryTableModel(df)
    model.set_highlight_aborts(True)
    received: list = []
    model.dataChanged.connect(lambda *_: received.append(True))
    model.set_highlight_aborts(True)
    assert received == []


# ---------------------------------------------------------------------------
# ConverterPanel + persistence
# ---------------------------------------------------------------------------


def test_toolbar_toggle_propagates_to_model(isolated_settings, qtbot, tmp_path) -> None:
    df = make_df([_row(repetition_type="suspected_abort")])
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel.load_inventory(df, output_tsv=tmp_path / "inv.tsv")
    assert panel.model() is not None
    assert panel.model().highlight_aborts() is False

    panel._aborts_btn.setChecked(True)
    assert panel.model().highlight_aborts() is True

    panel._aborts_btn.setChecked(False)
    assert panel.model().highlight_aborts() is False


def test_toolbar_toggle_state_persists_across_panels(isolated_settings, qtbot) -> None:
    """Toggling the highlight button writes to QSettings; a second
    ConverterPanel reads the same value at construction time.
    """
    AppSettings.remember_highlight_aborts(True)
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    assert panel._aborts_btn.isChecked() is True

    panel._aborts_btn.setChecked(False)
    assert AppSettings.load().highlight_aborts is False


def test_load_inventory_applies_persisted_toggle(isolated_settings, qtbot, tmp_path) -> None:
    AppSettings.remember_highlight_aborts(True)
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    df = make_df([_row(repetition_type="suspected_abort")])
    panel.load_inventory(df, output_tsv=tmp_path / "inv.tsv")
    # Model picks up the toolbar state on attach.
    assert panel.model().highlight_aborts() is True
    assert panel.model().data(panel.model().index(0, 0), HIGHLIGHT_ROLE) is True


def test_planned_rows_are_not_highlighted_even_when_toggle_on(qtbot, tmp_path) -> None:
    """Regression guard for the original misclassification — planned
    repeats (legitimate run-N acquisitions) must stay un-highlighted
    even when the toolbar toggle is on.
    """
    df = make_df([_row(repetition_type="planned")])
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel.load_inventory(df, output_tsv=tmp_path / "inv.tsv")
    panel._aborts_btn.setChecked(True)
    assert panel.model().data(panel.model().index(0, 0), HIGHLIGHT_ROLE) is False
