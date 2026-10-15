"""The search field the viewers share: the DICOM header and the sidecars.

One widget, so finding something works the same wherever it is done: the
same prompt width, a magnifier saying what the box is for, a clear button,
Esc to clear what was typed, and the platform's Find key (Ctrl+F, Cmd+F on
a Mac) to reach it from anywhere in the pane.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import QLineEdit, QWidget


class SearchField(QLineEdit):
    """A themed line edit for a search, with room to show its prompt.

    A bare ``QLineEdit`` asks for 17 characters, which cuts a prompt like
    "Find a name, keyword, tag or value" in half; this asks for more and can
    still shrink when the pane is dragged narrow.
    """

    def __init__(self, placeholder: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("search-field")
        self.setPlaceholderText(placeholder)
        self.setClearButtonEnabled(True)
        self._glyph = QAction(self)
        self.addAction(self._glyph, QLineEdit.ActionPosition.LeadingPosition)
        self.refresh_icon()

    def refresh_icon(self) -> None:
        """Re-tint the magnifier in the theme's colours (on a theme change)."""
        from .. import icons

        self._glyph.setIcon(icons.icon("search"))

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        from ..viz import fonts

        hint = super().sizeHint()
        return QSize(max(hint.width(), fonts.px(250)), hint.height())

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Esc clears a search, which is what it means in every search box. An
        # empty box lets it through, so Esc still closes a dialog.
        if event.key() == Qt.Key.Key_Escape and self.text():
            self.clear()
            event.accept()
            return
        super().keyPressEvent(event)


def find_shortcut(owner: QWidget, field: SearchField) -> QShortcut:
    """The platform's Find key, anywhere in ``owner``, focuses ``field``."""
    shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Find), owner)
    shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)

    def _focus() -> None:
        if field.isVisible() and field.isEnabled():
            field.setFocus()
            field.selectAll()

    shortcut.activated.connect(_focus)
    return shortcut


__all__ = ["SearchField", "find_shortcut"]
