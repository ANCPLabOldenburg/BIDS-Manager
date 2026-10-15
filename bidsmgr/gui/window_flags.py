"""Window buttons for windows we open ourselves.

A bare ``QDialog`` gets only a close button on Linux and Windows
(CROSS_PLATFORM_RULES 5.3), so a window people work in (the Logs, a DICOM
header, a preview) could not be maximised or tiled there. This gives it the
buttons a normal window has, on every OS.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget


def give_window_buttons(window: QWidget) -> None:
    """Minimise, maximise and close, like any top-level window."""
    window.setWindowFlags(
        Qt.WindowType.Window
        | Qt.WindowType.WindowTitleHint
        | Qt.WindowType.WindowSystemMenuHint
        | Qt.WindowType.WindowMinimizeButtonHint
        | Qt.WindowType.WindowMaximizeButtonHint
        | Qt.WindowType.WindowCloseButtonHint
    )


__all__ = ["give_window_buttons"]
