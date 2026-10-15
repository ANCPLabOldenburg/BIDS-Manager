"""GUI smoke tests for the Settings 'Scan rules' tab + rules persistence."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidget

from bidsmgr.gui.app_settings import AppSettings
from bidsmgr.gui.settings_dialog import SettingsDialog

pytestmark = pytest.mark.gui


def test_dialog_has_scan_rules_tab(qtbot) -> None:
    dlg = SettingsDialog(AppSettings(), None)
    qtbot.addWidget(dlg)
    assert "Scan rules" in dlg.page_titles()
    # Editable tables present.
    assert isinstance(dlg._excl_table, QTableWidget)
    assert isinstance(dlg._hint_table, QTableWidget)


def test_builtin_criteria_table_populated_and_readonly(qtbot) -> None:
    dlg = SettingsDialog(AppSettings(), None)
    qtbot.addWidget(dlg)
    t = QTableWidget(0, 4)
    SettingsDialog._populate_builtin_criteria(t)
    assert t.rowCount() > 10           # SEQUENCE_HINTS + DWI + task patterns
    # Read-only: items lack the editable flag.
    it = t.item(0, 0)
    assert not (it.flags() & Qt.ItemFlag.ItemIsEditable)


def test_read_scan_rules_valid(qtbot) -> None:
    dlg = SettingsDialog(AppSettings(), None)
    qtbot.addWidget(dlg)
    dlg._add_exclusion_row("calibration", "sequence", "substring")
    dlg._add_hint_row("lab_t1, my_mprage", "anat", "T1w", "", "substring", True)
    hints, excl, err = dlg._read_scan_rules()
    assert err is None
    assert excl == [{"pattern": "calibration", "target": "sequence", "match_mode": "substring"}]
    assert hints[0]["datatype"] == "anat" and hints[0]["suffix"] == "T1w" and hints[0]["force"] is True
    assert hints[0]["patterns"] == ["lab_t1", "my_mprage"]


def test_hint_datatype_suffix_are_constrained_dropdowns(qtbot) -> None:
    """Datatype + suffix are dropdowns of valid BIDS labels (no free text);
    suffix re-fills when the datatype changes, and 'derivatives' is absent."""
    from PyQt6.QtWidgets import QComboBox
    dlg = SettingsDialog(AppSettings(), None)
    qtbot.addWidget(dlg)
    dlg._add_hint_row("seq", "anat", "T1w", "", "substring", False)
    dt_cb = dlg._hint_table.cellWidget(0, 1)
    sfx_cb = dlg._hint_table.cellWidget(0, 2)
    assert isinstance(dt_cb, QComboBox) and isinstance(sfx_cb, QComboBox)
    dt_items = [dt_cb.itemText(i) for i in range(dt_cb.count())]
    assert "derivatives" not in dt_items
    assert "anat" in dt_items and "func" in dt_items
    # anat suffixes include T1w, not bold.
    anat_suffixes = [sfx_cb.itemText(i) for i in range(sfx_cb.count())]
    assert "T1w" in anat_suffixes and "bold" not in anat_suffixes
    # Switching the datatype re-fills the suffix list.
    dt_cb.setCurrentText("func")
    func_suffixes = [sfx_cb.itemText(i) for i in range(sfx_cb.count())]
    assert "bold" in func_suffixes and "T1w" not in func_suffixes


def test_persistence_round_trip(qtbot, isolated_settings) -> None:
    s = AppSettings.load()
    s.user_hints = [{"patterns": ["lab_t1"], "datatype": "anat", "suffix": "T1w", "force": True}]
    s.scan_exclusions = [{"pattern": "calib", "target": "sequence", "match_mode": "substring"}]
    s.save()
    s2 = AppSettings.load()
    assert s2.user_hints == s.user_hints
    assert s2.scan_exclusions == s.scan_exclusions
    # Boundary converters produce the engine dataclasses.
    hints = s2.to_user_hints()
    assert hints[0].force is True and hints[0].datatype == "anat"
    # Defaults are empty (Restore-defaults path).
    assert AppSettings().user_hints == [] and AppSettings().scan_exclusions == []


def test_load_into_widgets_then_restore_defaults(qtbot, isolated_settings) -> None:
    s = AppSettings.load()
    s.user_hints = [{"patterns": ["a"], "datatype": "anat", "suffix": "T1w"}]
    s.scan_exclusions = [{"pattern": "x", "target": "path", "match_mode": "substring"}]
    dlg = SettingsDialog(s, None)
    qtbot.addWidget(dlg)
    assert dlg._hint_table.rowCount() == 1
    assert dlg._excl_table.rowCount() == 1
    dlg._on_restore_defaults()
    assert dlg._hint_table.rowCount() == 0
    assert dlg._excl_table.rowCount() == 0
