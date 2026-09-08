"""Runtime text must never resize the window.

Paths, log lines and tracebacks are all handed to the GUI at runtime and
can be arbitrarily long. A plain ``QLabel`` reports its full text width
as its *minimum* size hint, which walks up the layout tree and forces Qt
to widen the window -- the converter used to jump wider the moment a
scan started on a deeply nested raw folder.

These tests lock the invariant: no displayed string, at any length, may
change what the layout demands.

Marked ``gui``; run headless with ``QT_QPA_PLATFORM=offscreen``.
"""

from __future__ import annotations

import pytest

from bidsmgr.gui.converter_panel import ConverterPanel
from bidsmgr.gui.main_window import MainWindow
from bidsmgr.gui.theme_manager import ThemeManager
from bidsmgr.gui.widgets import BusySpinner, ElidedLabel

pytestmark = pytest.mark.gui


# A realistically deep raw path, plus a pathological repeat of it.
LONG_PATH = (
    "/Users/someone/Development/datasets/BIDS_Manager/raw_data/MRI/"
    "neuroimaging_unit_new/sub-OL_0001/ses-pre/anat/00001.dcm"
)
ABSURD = LONG_PATH * 8


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


def test_elided_label_never_claims_horizontal_room(qtbot) -> None:
    label = ElidedLabel("short")
    qtbot.addWidget(label)

    assert label.minimumSizeHint().width() == 0
    label.setText(ABSURD)
    assert label.minimumSizeHint().width() == 0
    # Height still comes from the font, so the row does not collapse.
    assert label.minimumSizeHint().height() > 0


def test_elided_label_keeps_the_full_string(qtbot) -> None:
    """Elision is a paint-time concern; the text itself is not truncated."""
    label = ElidedLabel()
    qtbot.addWidget(label)

    label.setText(LONG_PATH)
    assert label.text() == LONG_PATH
    # Hovering recovers what the elision dropped on screen.
    assert label.toolTip() == LONG_PATH


def test_elided_label_paints_at_any_width(qtbot) -> None:
    """The custom paintEvent must survive widths far below the text."""
    label = ElidedLabel(ABSURD)
    qtbot.addWidget(label)
    label.show()

    for width in (1, 40, 300, 2000):
        label.resize(width, 20)
        label.repaint()  # would raise if the elision maths went wrong


# ---------------------------------------------------------------------------
# The spinner
# ---------------------------------------------------------------------------


def test_busy_spinner_width_is_message_independent(qtbot) -> None:
    spinner = BusySpinner()
    qtbot.addWidget(spinner)

    spinner.set_busy(True, message="Scanning…")
    short = spinner.minimumSizeHint().width()
    spinner.set_message(ABSURD)
    assert spinner.minimumSizeHint().width() == short


def test_busy_spinner_claims_no_width_when_it_appears(qtbot) -> None:
    """Going busy must not widen the toolbar it sits in either."""
    spinner = BusySpinner()
    qtbot.addWidget(spinner)

    assert spinner.minimumSizeHint().width() == 0
    spinner.set_busy(True, message=LONG_PATH)
    assert spinner.minimumSizeHint().width() == 0


# ---------------------------------------------------------------------------
# Wired into the real views
# ---------------------------------------------------------------------------


def test_converter_toolbar_width_survives_a_log_firehose(qtbot) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)

    toolbar = panel._spinner.parentWidget()
    panel._spinner.set_busy(True, message="Scanning…")
    before = toolbar.minimumSizeHint().width()

    # Worker log lines are mirrored into the spinner one by one.
    for line in (LONG_PATH, ABSURD, f"Converting {ABSURD}"):
        panel._on_log_message(line)
        assert toolbar.minimumSizeHint().width() == before


def test_main_window_minimum_survives_long_status_messages(
    qapp, qtbot, isolated_settings
) -> None:
    theme = ThemeManager(qapp)
    theme.apply("dark")
    window = MainWindow(theme)
    qtbot.addWidget(window)
    window.show()
    qapp.processEvents()

    before = window.minimumSizeHint().width()
    for message in (LONG_PATH, ABSURD, f"Traceback:\n{ABSURD}"):
        window._set_status(message)
        assert window.minimumSizeHint().width() == before
    # And the first line still reaches the label in full.
    assert window._status_text.text() == "Traceback:"
