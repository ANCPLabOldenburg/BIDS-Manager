"""Delegate that paints the BIDS dataset tree (Editor view, left pane).

Each tree row carries an optional severity badge published at
``Qt.ItemDataRole.UserRole + 2``:

* ``"ok"``   → small green dot
* ``"warn"`` → small amber dot
* ``"err"``  → small red dot
* ``None``   → no dot (e.g. directory rows)

The base text + selection paint comes from ``QStyledItemDelegate``'s
default implementation. Lift-and-shift from ``inspector_proto/proto.py``
lines 386-408.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem

from ..theme_manager import CUR


BADGE_ROLE: int = Qt.ItemDataRole.UserRole + 2

_BADGE_TOKEN: dict[str, str] = {
    "ok":   "success",
    "warn": "warning",
    "err":  "error",
}


class BidsTreeDelegate(QStyledItemDelegate):
    """Paints a small validation badge to the right of each tree row."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        super().paint(painter, option, index)
        badge = index.data(BADGE_ROLE)
        if not badge:
            return
        token = _BADGE_TOKEN.get(badge)
        if token is None:
            return
        pal = CUR()
        painter.save()
        size = 8
        margin = 12
        bx = option.rect.right() - margin
        by = option.rect.center().y()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(pal[token]))
        painter.drawEllipse(bx - size // 2, by - size // 2, size, size)
        painter.restore()


__all__ = ["BADGE_ROLE", "BidsTreeDelegate"]
