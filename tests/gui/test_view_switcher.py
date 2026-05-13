"""Tests for the Converter/Editor view switcher (M6 Step 1).

The top header now hosts two view pills and ``MainWindow`` swaps a
``QStackedWidget`` between :class:`ConverterPanel` and
:class:`EditorPanel`. The active view is persisted via ``QSettings`` so
the user lands on the same pane next launch.
"""

from __future__ import annotations

import pytest

from bidsmgr.gui.app_settings import AppSettings
from bidsmgr.gui.converter_panel import ConverterPanel
from bidsmgr.gui.editor_panel import EditorPanel
from bidsmgr.gui.main_window import MainWindow
from bidsmgr.gui.theme_manager import ThemeManager


pytestmark = pytest.mark.gui


def test_stack_hosts_both_panels(qapp, isolated_settings) -> None:
    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qapp.processEvents()

    assert win.stack.count() == 2
    assert isinstance(win.stack.widget(0), ConverterPanel)
    assert isinstance(win.stack.widget(1), EditorPanel)
    # Default is Converter when no setting persisted.
    assert win.stack.currentIndex() == 0
    assert win._header._converter_btn.isChecked()
    assert not win._header._editor_btn.isChecked()


def test_view_change_signal_swaps_stack_and_persists(
    qapp, isolated_settings,
) -> None:
    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qapp.processEvents()

    # Simulate clicking the Editor pill.
    win._header._editor_btn.setChecked(True)
    win._header._pill_group.idClicked.emit(1)
    qapp.processEvents()

    assert win.stack.currentIndex() == 1
    assert AppSettings.load().active_view == "editor"

    # And back.
    win._header._converter_btn.setChecked(True)
    win._header._pill_group.idClicked.emit(0)
    qapp.processEvents()

    assert win.stack.currentIndex() == 0
    assert AppSettings.load().active_view == "converter"


def test_persisted_active_view_restores_on_construction(
    qapp, isolated_settings,
) -> None:
    AppSettings.remember_active_view("editor")

    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qapp.processEvents()

    assert win.stack.currentIndex() == 1
    assert win._header._editor_btn.isChecked()


def test_theme_cascade_reaches_editor_panel(
    qapp, isolated_settings, monkeypatch,
) -> None:
    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qapp.processEvents()

    calls: list[dict] = []
    original = win.editor.repaint_for_palette

    def _spy(pal: dict) -> None:
        calls.append(pal)
        return original(pal)

    monkeypatch.setattr(win.editor, "repaint_for_palette", _spy)

    theme.toggle()  # dark -> light
    theme.toggle()  # light -> dark

    assert len(calls) == 2
