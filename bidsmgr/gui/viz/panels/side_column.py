"""A view with its controls column on the right, as every viewer has it.

The content (traces, a spectrum) with a painted "Advanced controls" tab on its right
edge, and the column itself in a splitter beyond the tab: the tab belongs to
the content's side, so it is never dragged away, and the column opens where
the tab says. The image viewer builds the same arrangement itself (its
column can also be lent to a comparison); the signal and spectrum viewers
share this one.
"""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QScrollArea, QWidget

from .side_tab import SideTab
from ...widgets.splitter import Splitter

#: The share of the width the column takes when it opens.
COLUMN_FRACTION = 0.26


class SideColumn:
    """``content`` and ``controls`` side by side; ``widget`` is what to place."""

    def __init__(self, content: QWidget, controls: QWidget, *,
                 remember: Callable[[bool], None], changed: Callable[[], None]) -> None:
        self._remember = remember
        self._changed = changed
        split = Splitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.setChildrenCollapsible(False)
        left = QWidget()
        row = QHBoxLayout(left)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(content, 1)
        self.tab = SideTab("Advanced controls", "controls")
        self.tab.clicked.connect(lambda: self.set_open(not self.is_open()))
        row.addWidget(self.tab)
        split.addWidget(left)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("viz-side-scroll")
        self.scroll.viewport().setObjectName("viz-side-viewport")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidget(controls)
        # Never narrower than its controls: with no horizontal scroll bar a
        # narrower column cut them off at its edge.
        self.scroll.setMinimumWidth(controls.minimumSizeHint().width()
                                    + self.scroll.verticalScrollBar().sizeHint().width() + 2)
        self.scroll.setVisible(False)
        split.addWidget(self.scroll)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        self.splitter = split
        self.widget = split

    def is_open(self) -> bool:
        return not self.scroll.isHidden()

    def set_open(self, on: bool, *, remember: bool = True) -> None:
        was = self.is_open()
        self.scroll.setVisible(bool(on))
        self.tab.set_open(bool(on))
        if on and not was:
            total = self.splitter.width() or 900
            column = max(self.scroll.minimumWidth(), int(total * COLUMN_FRACTION))
            self.splitter.setSizes([max(total - column, 1), column])
        if remember:
            self._remember(bool(on))
        self._changed()

    def show_section(self, section: QWidget) -> None:
        """Open the column and bring ``section`` into view."""
        self.set_open(True)
        self.scroll.ensureWidgetVisible(section, 0, 0)


__all__ = ["COLUMN_FRACTION", "SideColumn"]
