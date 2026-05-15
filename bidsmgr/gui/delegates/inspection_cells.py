"""Delegates for the Converter view's Inspection table.

Three classes, lift-and-shifted from ``inspector_proto/proto.py``
lines 292-381:

* :class:`StatusDelegate`    — first column. Paints the row's status
  badge (``ok``/``warn``/``err``/``phys``/``skip``/``info``).
* :class:`CheckboxDelegate`  — the ``include`` column. Paints a small
  rounded checkbox; click handling lives in the model (the delegate
  doesn't open an editor — the model toggles on click).
* :class:`CellTextDelegate`  — every text column. Knows how to render
  ``mono``, ``basename``, ``conf`` (color by confidence value), and
  ``plain`` cells, plus the strike-through for skipped rows on the
  ``basename`` column.

All three call :func:`paint_row_state` before drawing so the row tint
shows through regardless of which column repaints first.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QRect, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem

from ..theme_manager import CUR
from ..widgets.status_badge import badge_paint
from .row_state import (
    HIGHLIGHT_ROLE,
    ROW_STATE_ROLE,
    paint_highlight,
    paint_row_state,
)


# Role index used to publish the status kind ("ok"/"warn"/...) on the
# Status column and the boolean checked-state on the Include column.
PAYLOAD_ROLE: int = Qt.ItemDataRole.UserRole


class StatusDelegate(QStyledItemDelegate):
    """Status-badge column.

    Reads the row state from :data:`ROW_STATE_ROLE` and the badge kind
    from :data:`PAYLOAD_ROLE`.
    """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        paint_row_state(painter, option, index.data(ROW_STATE_ROLE))
        paint_highlight(painter, option, bool(index.data(HIGHLIGHT_ROLE)))
        kind = index.data(PAYLOAD_ROLE) or "ok"
        painter.save()
        badge_paint(painter, option.rect, kind, 16)
        painter.restore()


class CheckboxDelegate(QStyledItemDelegate):
    """Include-column checkbox.

    Paints the visual + handles clicks. The delegate overrides
    :meth:`editorEvent` so a single click on the cell toggles the bool
    via ``model.setData`` without going through ``QAbstractItemView``'s
    edit-trigger plumbing (otherwise a user would have to select the
    row first, then click again).

    Reads the row state from :data:`ROW_STATE_ROLE` and the bool from
    :data:`PAYLOAD_ROLE`.
    """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        paint_row_state(painter, option, index.data(ROW_STATE_ROLE))
        paint_highlight(painter, option, bool(index.data(HIGHLIGHT_ROLE)))
        pal = CUR()
        checked = bool(index.data(PAYLOAD_ROLE))

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        size = 13
        cx, cy = option.rect.center().x(), option.rect.center().y()
        rect = QRect(cx - size // 2, cy - size // 2, size, size)
        path = QPainterPath()
        path.addRoundedRect(rect.x(), rect.y(), rect.width(), rect.height(), 3, 3)

        if checked:
            painter.fillPath(path, QColor(pal["accent"]))
            painter.setPen(QColor(pal["primary_btn_text"]))
            f = QFont(painter.font())
            f.setBold(True)
            # Pixel size so the checkmark stays the same logical size on
            # macOS (72 dpi) and Linux / Windows (96 dpi). Point sizes
            # made the glyph render ~35% larger on the latter platforms.
            f.setPixelSize(9)
            painter.setFont(f)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "✓")
        else:
            painter.setPen(QPen(QColor(pal["border"]), 1))
            painter.setBrush(QColor(pal["bg"]))
            painter.drawPath(path)

        painter.restore()

    def editorEvent(self, event, model, option, index):  # noqa: N802 — Qt naming
        """Toggle the cell on a left-mouse release anywhere in it.

        Returning ``True`` tells the view we consumed the event so it
        doesn't fall through to a default edit-trigger handler that
        would try to open a text editor on the checkbox cell.
        """
        if event.type() != QEvent.Type.MouseButtonRelease:
            return False
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if not (index.flags() & Qt.ItemFlag.ItemIsEnabled):
            return False
        current = bool(index.data(PAYLOAD_ROLE))
        model.setData(index, not current, Qt.ItemDataRole.EditRole)
        return True


class CellTextDelegate(QStyledItemDelegate):
    """Generic text-cell delegate with role-based formatting.

    ``role`` selects how the cell is rendered:

    * ``"plain"``     — sans-serif text (default).
    * ``"mono"``      — monospace text (used for IDs, sessions, runs).
    * ``"basename"``  — dim text + strikethrough on skipped rows + tint
      to ``pal["error"]`` on error rows.
    * ``"conf"``      — color by numeric confidence: ≥0.9 green,
      ≥0.75 amber, lower red. Non-numeric → muted.

    The model can also opt into universal treatments by writing the
    sentinel value ``"—"`` (em-dash) for missing fields — these paint
    in ``pal["muted"]`` regardless of role.
    """

    _MONO_FAMILIES = ["SF Mono", "Menlo", "Monaco", "Consolas", "monospace"]

    def __init__(self, role: str = "plain", parent=None) -> None:
        super().__init__(parent)
        self._role = role

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        pal = CUR()
        row_state = index.data(ROW_STATE_ROLE) or ""
        paint_row_state(painter, option, row_state)
        paint_highlight(painter, option, bool(index.data(HIGHLIGHT_ROLE)))

        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        if not text:
            return

        painter.save()
        f = QFont(painter.font())
        if self._role in ("mono", "basename", "conf"):
            f.setFamilies(self._MONO_FAMILIES)
        # Pixel size, not point size: ``setPointSize`` is DPI-relative
        # (11 pt is ~11 px on macOS at 72 dpi but ~15 px on Linux /
        # Windows at 96 dpi). Pixel size matches the QSS rules in
        # ``theme.qss`` so the painted cell text lines up with the
        # widgets around the table on every platform.
        f.setPixelSize(11)
        painter.setFont(f)

        color = QColor(pal["text"])
        if text == "—":
            color = QColor(pal["muted"])
        elif self._role == "basename":
            color = QColor(pal["dim"])
            if row_state == "err":
                color = QColor(pal["error"])
        elif self._role == "conf":
            try:
                v = float(text)
                if v >= 0.9:
                    color = QColor(pal["success"])
                elif v >= 0.75:
                    color = QColor(pal["warning"])
                else:
                    color = QColor(pal["error"])
            except ValueError:
                color = QColor(pal["muted"])
        elif text == "missing":
            color = QColor(pal["error"])

        if row_state == "skip":
            color = QColor(pal["muted"])

        painter.setPen(color)
        r = option.rect.adjusted(8, 0, -8, 0)
        fm = QFontMetrics(f)
        elided = fm.elidedText(text, Qt.TextElideMode.ElideRight, r.width())
        painter.drawText(
            r,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided,
        )

        # Strikethrough for skipped basenames — drawn by hand because
        # QFont.setStrikeOut would also strike the elision ellipsis,
        # which looks off at narrow widths.
        if row_state == "skip" and self._role == "basename":
            br = fm.boundingRect(elided)
            y = r.center().y()
            painter.setPen(QPen(QColor(pal["muted"]), 1))
            painter.drawLine(r.left(), y, r.left() + min(br.width(), r.width()), y)

        painter.restore()


__all__ = [
    "CellTextDelegate",
    "CheckboxDelegate",
    "PAYLOAD_ROLE",
    "StatusDelegate",
]
