"""Splitters whose handles say they can be dragged.

Every splitter in the GUI is a :class:`Splitter`. Its handle paints itself (the
stylesheet cannot draw a grip): at rest, three small dots centred along it and,
between two panes in a card, the hairline that divides them; under the mouse or
while dragged, the handle fills with the theme's accent wash and the dots and the
line take the accent. The way an editor such as VS Code shows a sash, plus a
grip so the handle is found without hovering (user request, 2026-10-13).

A handle is at least :data:`GRIP_PX` wide, scaled with the font, so the grip
has room and the target is easier to hit than the 1 px hairlines were. Colours
come from the palette at paint time, so a theme change is a repaint.
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import QSplitter, QSplitterHandle

from ..theme_manager import CUR, scaled_px
from .primitives import qcolor

#: The narrowest handle, at font scale 1.0.
GRIP_PX = 6
#: Dot diameter and the distance between dot centres, at font scale 1.0.
_DOT_PX = 3
_STEP_PX = 6


class GripHandle(QSplitterHandle):
    """A splitter handle with a centred three-dot grip and a hover tint."""

    def __init__(self, orientation: Qt.Orientation, parent: QSplitter) -> None:
        super().__init__(orientation, parent)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._hover = False
        self._pressed = False

    # -- state ------------------------------------------------------------------

    def is_active(self) -> bool:
        return self._hover or self._pressed

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._pressed = True
        self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._pressed = False
        self.update()
        super().mouseReleaseEvent(event)

    # -- painting -----------------------------------------------------------------

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        pal = CUR()
        rect = QRectF(self.rect())
        if rect.width() < 1 or rect.height() < 1:
            return
        across = self.orientation() == Qt.Orientation.Horizontal   # panes side by side
        between_cards = self.splitter().objectName() == "card-split"
        active = self.is_active()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if active:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(qcolor(pal["accent_bg"]))
            radius = min(rect.width(), rect.height()) / 2
            painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        dot = max(2.0, scaled_px(_DOT_PX) * 1.0)
        step = max(dot + 2, scaled_px(_STEP_PX) * 1.0)
        thickness = rect.width() if across else rect.height()
        dot = min(dot, max(thickness - 2, 2.0))
        centre = rect.center()
        if not between_cards:
            # The dividing hairline, broken around the grip so it does not
            # strike through the dots.
            line = QPen(qcolor(pal["accent"] if active else pal["subtle"]))
            line.setWidthF(1.0)
            painter.setPen(line)
            gap = step + dot
            if across:
                x = centre.x()
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, centre.y() - gap))
                painter.drawLine(QPointF(x, centre.y() + gap), QPointF(x, rect.bottom()))
            else:
                y = centre.y()
                painter.drawLine(QPointF(rect.left(), y), QPointF(centre.x() - gap, y))
                painter.drawLine(QPointF(centre.x() + gap, y), QPointF(rect.right(), y))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qcolor(pal["accent"] if active else pal["muted"]))
        for k in (-1, 0, 1):
            c = QPointF(centre.x(), centre.y() + k * step) if across else \
                QPointF(centre.x() + k * step, centre.y())
            painter.drawEllipse(c, dot / 2, dot / 2)
        painter.end()


class Splitter(QSplitter):
    """A ``QSplitter`` whose handles carry a grip and a hover tint."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        super().setHandleWidth(scaled_px(GRIP_PX))

    def createHandle(self) -> QSplitterHandle:  # noqa: N802 - Qt naming
        return GripHandle(self.orientation(), self)

    def setHandleWidth(self, width: int) -> None:  # noqa: N802 - Qt naming
        """Never narrower than the grip needs."""
        super().setHandleWidth(max(int(width), scaled_px(GRIP_PX)))


__all__ = ["GRIP_PX", "GripHandle", "Splitter"]
