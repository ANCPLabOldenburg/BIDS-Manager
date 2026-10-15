"""Floating panels stay where the user put them, a right-click menu stays
open, and the Log has one header.

The defect behind the first two (2026-10-12): Properties built each entity
row's remove button with no parent and made it visible before adding it to
the row, so for a moment it was a top-level window. That window took the
focus from a floating Inspection or Filter window on every click (macOS then
raised the main window over it), and closed a right-click menu opened on a
row not yet selected (19 of 29 rows). Re-docking a side panel likewise left
its unused fold caret as a 28 px window.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QApplication

pytestmark = pytest.mark.gui


def _rows(n: int = 12) -> pd.DataFrame:
    import json

    rows = []
    for i in range(n):
        rows.append({
            "participant_id": f"sub-00{1 + i % 3}", "session": "", "include": 1,
            "modality": "mri", "datatype": "func", "bids_guess_suffix": "bold",
            "bids_name": f"sub-00{1 + i % 3}_task-t{i}_run-1_bold",
            "bids_guess_confidence": "0.9", "bids_guess_skip": False, "issues": "",
            "sequence": f"bold_{i}", "series_uid": f"1.{i}", "dataset": "ds",
            "source_file": "", "source_folder": "s", "acq_time": f"10:{i:02d}:00",
            "task": f"t{i}", "run": "1",
            "entities": json.dumps({"subject": f"00{1 + i % 3}", "task": f"t{i}", "run": "1"}),
        })
    return pd.DataFrame(rows)


class _WindowSpy(QObject):
    """Records every window shown."""

    def __init__(self) -> None:
        super().__init__()
        self.shown: list[str] = []

    def eventFilter(self, obj, event):  # noqa: N802 - Qt signature
        if event.type() == QEvent.Type.Show and obj.isWidgetType() and obj.isWindow():
            self.shown.append(f"{type(obj).__name__}#{obj.objectName()}")
        return False


@pytest.fixture
def window(qtbot, qapp):
    from bidsmgr.gui.main_window import MainWindow
    from bidsmgr.gui.theme_manager import ThemeManager

    win = MainWindow(ThemeManager(qapp))
    qtbot.addWidget(win)
    win.resize(1400, 900)
    win.show()
    qtbot.waitExposed(win)
    win._apply_active_view("converter", persist=False)
    win.converter.load_inventory(_rows())
    qtbot.wait(50)
    return win


def _visible_windows() -> list[str]:
    return sorted(f"{type(w).__name__}#{w.objectName()}"
                  for w in QApplication.topLevelWidgets() if w.isVisible())


def _settle(qtbot, ms: int = 120) -> None:
    end = time.time() + ms / 1000
    while time.time() < end:
        qtbot.wait(10)


@pytest.mark.parametrize("frame_name", ["_inspection_unit", "_filter_frame"])
def test_a_floating_panel_keeps_the_focus_through_clicks(qtbot, qapp, window, frame_name) -> None:
    from PyQt6.QtTest import QTest

    conv = window.converter
    frame = getattr(conv, frame_name)
    frame.detach()
    floating = frame._detached
    qtbot.addWidget(floating)
    qtbot.waitExposed(floating)
    floating.activateWindow()
    _settle(qtbot)
    spy = _WindowSpy()
    qapp.installEventFilter(spy)
    try:
        for row in (1, 4, 7, 10):
            if frame_name == "_inspection_unit":
                rect = conv._table.visualRect(conv._model.index(row, 3))
                QTest.mouseClick(conv._table.viewport(), Qt.MouseButton.LeftButton,
                                 Qt.KeyboardModifier.NoModifier, rect.center())
            else:
                tree = conv._filter_pane._tree
                item = conv._filter_pane._leaf_for_row(row)
                QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton,
                                 Qt.KeyboardModifier.NoModifier,
                                 tree.visualItemRect(item).center())
            _settle(qtbot)
            assert qapp.activeWindow() is floating, f"row {row}: the floating window lost the focus"
    finally:
        qapp.removeEventFilter(spy)
    assert spy.shown == [], f"windows appeared while clicking: {spy.shown}"
    frame.reattach()
    _settle(qtbot)
    assert _visible_windows() == ["MainWindow#"], "re-docking leaves nothing behind"


def test_every_panel_docks_back_without_a_stray_window(qtbot, window) -> None:
    conv = window.converter
    for frame in (conv._filter_frame, conv._inspection_unit, conv._log_frame):
        frame.detach()
        qtbot.addWidget(frame._detached)
        _settle(qtbot, 60)
        frame.reattach()
        _settle(qtbot, 60)
        assert _visible_windows() == ["MainWindow#"], frame._title


def test_a_right_click_menu_stays_open_on_a_row_not_selected(qtbot, window) -> None:
    conv = window.converter
    conv._jump_to_row(0)
    conv._properties.flush()
    for row in range(1, 8):
        rect = conv._table.visualRect(conv._model.index(row, 3))
        conv._on_table_menu(rect.center())
        menu = conv._table_menu
        qtbot.addWidget(menu)
        _settle(qtbot, 200)
        assert menu.isVisible(), f"row {row}: the menu closed by itself"
        assert conv._properties._row == row, "Properties follows the right-click"
        menu.close()


def test_the_log_has_one_header_and_its_buttons_travel_with_it(qtbot, window) -> None:
    from PyQt6.QtWidgets import QTabWidget

    conv = window.converter
    frame = conv._log_frame
    assert frame.findChild(QTabWidget) is None, "no tab bar repeating its name"
    buttons = list(conv._log_buttons.values())
    assert all(frame._bar.isAncestorOf(b) for b in buttons)
    frame.detach()
    floating = frame._detached
    qtbot.addWidget(floating)
    _settle(qtbot, 60)
    assert all(floating.isAncestorOf(b) and b.isVisible() for b in buttons), "they float along"
    frame.reattach()
    _settle(qtbot, 60)
    assert all(frame._bar.isAncestorOf(b) for b in buttons), "and come back to the header"


def test_lines_qt_draws_follow_the_theme(qapp) -> None:
    """A document-mode tab bar's base and frame bevels read Light/Mid/Dark,
    which were Qt's defaults (pure white in a dark theme)."""
    from PyQt6.QtGui import QPalette

    from bidsmgr.gui.theme_manager import CUR, ThemeManager

    manager = ThemeManager(qapp)
    manager.apply("graphite")
    border = CUR()["border"].lower()
    palette = qapp.palette()
    for role in (QPalette.ColorRole.Light, QPalette.ColorRole.Mid, QPalette.ColorRole.Dark):
        assert palette.color(role).name() == border


def test_a_pane_rebuilt_twice_in_a_row_opens_no_window(qtbot, qapp, tmp_path) -> None:
    """The Validation pane rebuilt its sections twice in one turn (opening a
    file sets the file, then the folder); the first set, detached for
    deletion, still had a show queued from being added, and its hint opened
    as a window for a moment, taking the focus like the remove button."""
    from bidsmgr.gui.widgets.validation_pane import ValidationPane

    pane = ValidationPane()
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    spy = _WindowSpy()
    qapp.installEventFilter(spy)
    try:
        for name in ("a", "b", "c"):
            pane.set_current_file(tmp_path / f"sub-01_{name}_T1w.nii.gz", tmp_path)
            pane.set_report(None)
        _settle(qtbot)
    finally:
        qapp.removeEventFilter(spy)
    assert spy.shown == [], f"windows appeared while rebuilding: {spy.shown}"
