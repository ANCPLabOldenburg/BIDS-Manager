"""Tests for the inspection-pane footer + About dialog.

Two unrelated UI tweaks landed together so they share a test file:

* Highlight aborts + Bulk edit buttons moved from the converter
  toolbar to a new footer strip at the bottom of the inspection
  pane (closer to the table they act on).
* Clicking the BIDS-Manager logo or wordmark in the top header
  pops an About / Authorship dialog with the bidsmgr intro plus a
  faithful port of the original BIDS-Manager v0.2.5 Authorship
  dialog (lab logo, author bios, acknowledgements, links).
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QPushButton

from bidsmgr.gui.about_dialog import AboutDialog
from bidsmgr.gui.converter_panel import ConverterPanel
from bidsmgr.gui.main_window import _ClickableLabel, _TopHeader
from bidsmgr.gui.theme_manager import ThemeManager


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Inspection-pane footer relocation
# ---------------------------------------------------------------------------


def test_aborts_and_bulk_buttons_live_inside_inspection_pane(
    qapp, isolated_settings,
) -> None:
    """Both buttons must be parented under the inspection pane (not
    the toolbar) so they sit next to the table they act on."""
    panel = ConverterPanel()
    # Walk up the parent chain from each button to confirm it lands
    # inside the inspection stack's pane, not the toolbar QFrame.
    for btn in (panel._aborts_btn, panel._bulk_btn):
        parent = btn.parent()
        # Direct parent is the footer frame, with objectName
        # "inspection-footer".
        assert parent.objectName() == "inspection-footer"


def test_aborts_toggle_still_routes_to_model(
    qapp, isolated_settings,
) -> None:
    """The toggled signal still flips ``_on_aborts_toggled`` — moving
    the button between layouts didn't break the connection."""
    panel = ConverterPanel()
    # Without a loaded model the toggle just persists the setting.
    panel._aborts_btn.setChecked(True)
    from bidsmgr.gui.app_settings import AppSettings
    assert AppSettings.load().highlight_aborts is True


def test_bulk_button_still_starts_disabled(qapp, isolated_settings) -> None:
    panel = ConverterPanel()
    assert not panel._bulk_btn.isEnabled()


# ---------------------------------------------------------------------------
# AboutDialog
# ---------------------------------------------------------------------------


def _click(label: _ClickableLabel) -> None:
    """Synthesise a left-click release on a ClickableLabel."""
    pos = QPoint(label.width() // 2, label.height() // 2)
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        pos.toPointF(),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    label.mouseReleaseEvent(event)


def test_clickable_label_emits_clicked(qapp, qtbot) -> None:
    lbl = _ClickableLabel("hello")
    lbl.resize(50, 20)
    with qtbot.waitSignal(lbl.clicked, timeout=500):
        _click(lbl)


def test_top_header_logo_is_clickable(qapp, qtbot, isolated_settings) -> None:
    """The brand logo emits ``about_requested`` when clicked."""
    theme = ThemeManager(qapp)
    header = _TopHeader(theme)
    qapp.processEvents()
    header._logo.resize(28, 24)
    with qtbot.waitSignal(header.about_requested, timeout=500):
        _click(header._logo)


def test_top_header_wordmark_is_clickable(
    qapp, qtbot, isolated_settings,
) -> None:
    """The wordmark also emits ``about_requested``."""
    theme = ThemeManager(qapp)
    header = _TopHeader(theme)
    qapp.processEvents()
    wordmark = next(
        c for c in header.findChildren(_ClickableLabel)
        if c.objectName() == "brand-name"
    )
    wordmark.resize(120, 24)
    with qtbot.waitSignal(header.about_requested, timeout=500):
        _click(wordmark)


def test_about_dialog_builds_without_crashing(qapp) -> None:
    """The About dialog constructs cleanly with the bundled assets."""
    dlg = AboutDialog()
    # The window title is set and the dialog has content sections.
    assert dlg.windowTitle() == "About BIDS-Manager"
    # Title label exists with the right object name.
    title_labels = [
        lbl for lbl in dlg.findChildren(type(dlg).__bases__[0])
        if False  # placeholder; not used
    ]
    del title_labels
    from PyQt6.QtWidgets import QLabel
    titles = [
        lbl for lbl in dlg.findChildren(QLabel)
        if lbl.objectName() == "about-title"
    ]
    assert titles and titles[0].text() == "BIDS-Manager"


def test_about_dialog_contains_author_names(qapp) -> None:
    """Bios from the original BIDS-Manager AuthorshipDialog are present."""
    from PyQt6.QtWidgets import QLabel
    dlg = AboutDialog()
    bio_html = " ".join(
        lbl.text()
        for lbl in dlg.findChildren(QLabel)
        if lbl.objectName() == "about-author-bio"
    )
    assert "Karel López Vilaret" in bio_html
    assert "Jochem Rieger" in bio_html
    assert "Applied Neurocognitive Psychology" in bio_html


def test_about_dialog_contains_acknowledgements(qapp) -> None:
    from PyQt6.QtWidgets import QLabel
    dlg = AboutDialog()
    ack_labels = [
        lbl for lbl in dlg.findChildren(QLabel)
        if lbl.objectName() == "about-ack"
    ]
    assert ack_labels
    txt = ack_labels[0].text()
    # A few of the names from the original list.
    assert "Bosch-Bayard" in txt
    assert "Tina Schmitt" in txt


def test_about_dialog_shows_version(qapp) -> None:
    import bidsmgr
    from PyQt6.QtWidgets import QLabel
    dlg = AboutDialog()
    versions = [
        lbl for lbl in dlg.findChildren(QLabel)
        if lbl.objectName() == "about-version"
    ]
    assert versions
    assert bidsmgr.__version__ in versions[0].text()


def test_main_window_wires_about_request_to_dialog(
    qapp, isolated_settings, monkeypatch,
) -> None:
    """End-to-end: MainWindow connects ``about_requested`` to
    ``_show_about_dialog`` which builds + exec's an AboutDialog."""
    from bidsmgr.gui.main_window import MainWindow

    opened: list[AboutDialog] = []

    def _spy_exec(self):
        opened.append(self)
        return AboutDialog.DialogCode.Accepted

    monkeypatch.setattr(AboutDialog, "exec", _spy_exec)

    theme = ThemeManager(qapp)
    win = MainWindow(theme)
    win._header.about_requested.emit()
    qapp.processEvents()

    assert opened, "About dialog was not constructed"
