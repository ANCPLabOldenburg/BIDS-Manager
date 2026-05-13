"""Tests for ``bidsmgr.gui.properties_panel.PropertiesPanel``.

Covers:

* ``InventoryTableModel.set_entity`` / ``set_datatype_suffix`` /
  ``entities()`` / ``datatype_suffix()`` — the API the panel uses.
* The panel's render: form rows for every allowed entity, required
  markers, validation messages, predicted-path preview.
* Bidirectional sync: editing a field in the panel updates the
  model's basename; editing a mirror cell on the model updates the
  panel's form.
* Project integration: Properties-panel edits append
  :class:`UserSetEntity` events.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PyQt6.QtCore import Qt

from bidsmgr.gui.models import COLUMNS, InventoryTableModel
from bidsmgr.gui.properties_panel import PropertiesPanel
from bidsmgr.project import Project, ScanImported, UserSetCell, UserSetEntity


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


# ---------------------------------------------------------------------------
# Model API for the panel
# ---------------------------------------------------------------------------


def test_entities_reads_parsed_json() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row()]))
    assert m.entities(0) == {"subject": "001", "session": "pre", "task": "rest"}


def test_entities_returns_empty_for_malformed_json() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row(entities="{not valid")]))
    assert m.entities(0) == {}


def test_datatype_suffix_reads_columns() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row()]))
    assert m.datatype_suffix(0) == ("func", "bold")


def test_set_entity_adds_to_basename() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row()]))
    assert m.set_entity(0, "acquisition", "mprage") is True
    bn = m.dataframe().at[0, "proposed_basename"]
    assert "acq-mprage" in bn


def test_set_entity_removes_when_value_blank() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row()]))
    m.set_entity(0, "acquisition", "mprage")
    assert "acq-mprage" in m.dataframe().at[0, "proposed_basename"]
    m.set_entity(0, "acquisition", "")
    assert "acq-" not in m.dataframe().at[0, "proposed_basename"]


def test_set_entity_subject_updates_bids_name_mirror() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row()]))
    m.set_entity(0, "subject", "042")
    assert m.dataframe().at[0, "BIDS_name"] == "sub-042"
    assert m.dataframe().at[0, "proposed_basename"].startswith("sub-042_")


def test_set_entity_no_op_when_value_unchanged() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row()]))
    assert m.set_entity(0, "task", "rest") is False


def test_set_datatype_suffix_rebuilds() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row()]))
    m.set_datatype_suffix(0, "anat", "T1w")
    assert m.datatype_suffix(0) == ("anat", "T1w")


def test_set_entity_appends_user_set_entity_event(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    project.append(ScanImported(inventory_tsv="/x", row_ids=("1.2.3.4",)))
    m = InventoryTableModel(pd.DataFrame([_func_row()]), project=project)

    m.set_entity(0, "acquisition", "mprage")

    events = [e for e in project.log if isinstance(e, UserSetEntity)]
    assert len(events) == 1
    assert events[0].entity == "acquisition"
    assert events[0].value == "mprage"
    assert events[0].previous is None  # was absent before


# ---------------------------------------------------------------------------
# PropertiesPanel render
# ---------------------------------------------------------------------------


def _build_panel_with_model(qtbot, row: dict = None) -> tuple[PropertiesPanel, InventoryTableModel]:
    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    m = InventoryTableModel(pd.DataFrame([row or _func_row()]))
    panel.bind_model(m)
    panel.set_selected_row(0)
    return panel, m


def test_panel_renders_entity_rows_for_selected_row(qtbot) -> None:
    panel, _m = _build_panel_with_model(qtbot)
    names = [r.entity_name for r in panel._entity_rows]
    # func/bold's allowed entities, in display order.
    assert names[0] == "subject"
    assert "task" in names
    assert "acquisition" in names
    assert "run" in names


def test_panel_shows_current_entity_values(qtbot) -> None:
    panel, _m = _build_panel_with_model(qtbot)
    by_name = {r.entity_name: r.edit.text() for r in panel._entity_rows}
    assert by_name["subject"] == "001"
    assert by_name["task"] == "rest"
    assert by_name["session"] == "pre"
    assert by_name["acquisition"] == ""


def test_panel_committing_field_updates_model_basename(qtbot) -> None:
    panel, m = _build_panel_with_model(qtbot)
    # Simulate user typing into the acquisition field and tabbing out.
    acq_row = next(r for r in panel._entity_rows if r.entity_name == "acquisition")
    acq_row.edit.setText("mprage")
    panel._on_entity_committed("acquisition", "mprage")
    assert "acq-mprage" in m.dataframe().at[0, "proposed_basename"]


def test_panel_blank_when_no_row_selected(qtbot) -> None:
    panel, _m = _build_panel_with_model(qtbot)
    panel.set_selected_row(None)
    assert panel._entity_rows == []


def test_panel_model_data_changed_rerenders(qtbot) -> None:
    panel, m = _build_panel_with_model(qtbot)
    # Edit the model directly (simulating a table-cell edit on the
    # task column). The panel must reflect the new value after
    # ``dataChanged`` fires.
    task_col = next(i for i, c in enumerate(COLUMNS) if c.key == "task")
    m.setData(m.index(0, task_col), "motor")
    by_name = {r.entity_name: r.edit.text() for r in panel._entity_rows}
    assert by_name["task"] == "motor"


def test_panel_project_integration(qtbot, tmp_path: Path) -> None:
    project = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    project.append(ScanImported(inventory_tsv="/x", row_ids=("1.2.3.4",)))
    m = InventoryTableModel(pd.DataFrame([_func_row()]), project=project)

    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    panel.bind_model(m)
    panel.set_project(project)
    panel.set_selected_row(0)

    panel._on_entity_committed("acquisition", "mprage")
    events = [e for e in project.log if isinstance(e, UserSetEntity)]
    assert any(e.entity == "acquisition" and e.value == "mprage" for e in events)
