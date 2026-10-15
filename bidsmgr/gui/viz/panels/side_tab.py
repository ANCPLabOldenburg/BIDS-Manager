"""The handle of a controls column: a tab on the edge, always in view.

A "Controls" button among a toolbar's buttons was easy to miss, and once the
column was closed nothing on screen said there was one. This is the drawer's
handle: a slim tab on the right edge of the images, the column's name written
down it with its icon, an arrow saying which way it opens. Click to open,
click again to close (or its key, Ctrl+I; Cmd+I on a Mac).

ONE painted widget, in the theme's colours, rounded on its outer side.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ....viz import actions as A
from ....viz import keynames
from ....viz.theme import parse_colour


def _live_keys(action_id: str) -> list[str]:
    """The keys ``action_id`` has in the user's keymap (stored spelling)."""
    from ..bridge import SettingsHub

    keymap = A.effective_keymap(SettingsHub.instance().settings.keymap)
    return keymap.get(action_id, [])


def _qcolor(value: str, alpha=None) -> QColor:
    c = QColor(*parse_colour(value))
    if alpha is not None:
        c.setAlpha(alpha)
    return c


class SideTab(QWidget):
    """A vertical tab: the column's name, its icon and an arrow."""

    clicked = pyqtSignal()
    WIDTH = 24

    def __init__(self, text: str = "Advanced controls", icon: str = "controls",
                 parent=None, *, action_id: str = "view.inspector") -> None:
        super().__init__(parent)
        self.text = text
        self.icon_name = icon
        #: The action the tab stands for; its key is named in the tooltip.
        self.action_id = action_id
        #: Whether the column it opens is open (the arrow points to close).
        self.open = False
        self.setFixedWidth(self.WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setObjectName("viz-side-tab")
        self._sync_tip()

    def set_open(self, on: bool) -> None:
        if self.open != bool(on):
            self.open = bool(on)
            self._sync_tip()
            self.update()

    def _sync_tip(self) -> None:
        keys = ", ".join(keynames.keys(_live_keys(self.action_id)))
        key = f" ({keys})" if keys else ""
        self.setToolTip(f"{'Close' if self.open else 'Open'} the {self.text.lower()}"
                        f"{key}: every control, grouped by purpose")

    def event(self, event) -> bool:  # noqa: N802 - Qt signature
        if event.type() == QEvent.Type.ToolTip:
            self._sync_tip()            # the key as bound NOW, as this OS names it
        return super().event(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def paintEvent(self, _event) -> None:  # noqa: N802
        from .. import bridge
        from ... import icons

        theme = bridge.ThemeHub.instance().theme
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        hover = self.underMouse()
        r = QRectF(self.rect()).adjusted(1.0, 2.0, 0.0, -2.0)
        path = QPainterPath()
        path.addRoundedRect(r, 6.0, 6.0)
        p.fillPath(path, _qcolor(theme.token("surface3" if hover else "surface2", "#161b22")))
        p.setPen(_qcolor(theme.token("border", "#21262d")))
        p.drawPath(path)
        cx = r.center().x()
        ink = _qcolor(theme.text, 255 if hover else 215)
        from .. import fonts

        font = fonts.font(11, bold=True)
        font.setCapitalization(QFont.Capitalization.AllUppercase)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.0)
        from PyQt6.QtGui import QFontMetricsF

        text_len = QFontMetricsF(font).horizontalAdvance(self.text.upper())
        # Arrow, icon and name as ONE group, centred on the tab's height
        # (pinned to the top when the tab is too short to centre it).
        group = 10.0 + 12.0 + 16.0 + 10.0 + text_len
        top = max(r.top() + 6.0, r.center().y() - group / 2.0)
        # The arrow: pointing left (open it out) or right (put it away).
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(ink)
        y = top + 5.0
        if self.open:
            tri = [QPointF(cx - 3, y - 5), QPointF(cx - 3, y + 5), QPointF(cx + 3, y)]
        else:
            tri = [QPointF(cx + 3, y - 5), QPointF(cx + 3, y + 5), QPointF(cx - 3, y)]
        p.drawPolygon(QPolygonF(tri))
        pix = icons.icon(self.icon_name, theme.accent).pixmap(16, 16)
        p.drawPixmap(int(cx - 8), int(y + 12), pix)
        # The name, written down the tab.
        p.setFont(font)
        p.setPen(ink)
        p.save()
        p.translate(cx, y + 5.0 + 12.0 + 16.0 + 10.0)
        p.rotate(90)
        p.drawText(QRectF(0, -9, text_len + 4.0, 18),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text)
        p.restore()
        p.end()


__all__ = ["SideTab"]
