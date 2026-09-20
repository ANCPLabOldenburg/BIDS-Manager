"""Delegate that paints the BIDS dataset tree (Editor view, left pane).

Three things sit to the right of a row's name, in this order:

* **what a folder holds** (``3 ses, 47 files``), from :data:`COUNT_ROLE`, so a
  subject can be read without expanding it;
* **how many findings it has**, from :data:`ISSUE_ROLE` as ``(errors,
  warnings)``, painted as small counted pills;
* a green pill carrying a tick when a file was checked and is clean. Same
  shape as the counts, because it answers the same question; a bare dot beside
  two counted pills read as a different kind of thing.

The counts replaced a coloured dot. A dot says "something is wrong in here"
and stops; on a folder holding four hundred files that is the beginning of a
search rather than an answer. The number says how much, so a user can tell a
subject with one recommended field missing from a subject with ninety.

**Every size here scales with the UI font.** The previous version asked the
option font for ``pointSizeF()``, which returns ``-1`` when the font was set
in pixels, as the stylesheet sets it, so the count was clamped to a fixed 7pt
and did not respond to the font-scale setting at all. Sizes now come from
:func:`~bidsmgr.gui.theme_manager.scaled_px`, the same source the stylesheet
uses.
"""

from __future__ import annotations

from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem

from ..theme_manager import CUR, scaled_px

# Worst severity on the row, or rolled up from a folder's descendants.
BADGE_ROLE: int = Qt.ItemDataRole.UserRole + 2
# What a folder holds, as text: "3 ses, 47 files".
COUNT_ROLE: int = Qt.ItemDataRole.UserRole + 4
# (errors, warnings) on this row, or summed over a folder's descendants.
ISSUE_ROLE: int = Qt.ItemDataRole.UserRole + 5

_BADGE_TOKEN: dict[str, str] = {
    "ok":   "success",
    "warn": "warning",
    "err":  "error",
}

# Gap between the row's right edge and the first thing painted in it.
_EDGE = 8
# Gap between two painted chips.
_GAP = 5

# What a clean row shows instead of a number. A tick rather than a count,
# because "clean" has no quantity.
_TICK = "\u2713"


def _small_font(option: QStyleOptionViewItem, delta: int = -1) -> QFont:
    """A font one step below the row's, in pixels so it tracks the scale."""
    font = QFont(option.font)
    font.setPixelSize(max(scaled_px(11 + delta), 7))
    return font


class BidsTreeDelegate(QStyledItemDelegate):
    """Paints the folder summary and finding counts to the right of a row."""

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        """Leave room for a comfortable row at any font scale.

        Without this the row keeps the height Qt computed for the unscaled
        font and the pills are clipped top and bottom when the user turns the
        scale up.
        """
        size = super().sizeHint(option, index)
        size.setHeight(max(size.height(), scaled_px(22)))
        return size

    def initStyleOption(  # noqa: N802 - Qt signature
        self, option: QStyleOptionViewItem, index,
    ) -> None:
        """Elide the name against the space the chips take, not the row edge.

        Done HERE rather than in :meth:`paint` because the base ``paint``
        calls this again on its own copy of the option, which threw away an
        elision applied beforehand and drew long filenames straight through
        the counts. The rect itself is deliberately not narrowed: the
        selection bar has to span the whole row.
        """
        super().initStyleOption(option, index)
        if not option.text or option.rect.width() <= 0:
            return
        reserved = self._reserved_width(option, index)
        if not reserved:
            return
        available = self._text_width(option) - reserved
        if available <= 0:
            return
        option.text = option.fontMetrics.elidedText(
            option.text, Qt.TextElideMode.ElideRight, available,
        )

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        super().paint(painter, option, index)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Right to left: findings first, so the counts stay in a straight
        # column down the tree and the folder summary flows around them.
        right = option.rect.right() - _EDGE
        right = self._paint_issues(painter, option, index, right)
        right = self._paint_ok(painter, option, index, right)
        self._paint_count(painter, option, index, right)
        painter.restore()

    # ------------------------------------------------------------------

    @staticmethod
    def _text_width(opt: QStyleOptionViewItem) -> int:
        """How much of the row the name may use, past the icon."""
        width = opt.rect.width()
        if not opt.icon.isNull():
            width -= opt.decorationSize.width() + scaled_px(8)
        return max(width, 0)

    def _reserved_width(self, opt: QStyleOptionViewItem, index) -> int:
        """How much room the right-hand chips need on this row."""
        total = 0
        counts = index.data(ISSUE_ROLE)
        metrics = self._small_metrics(opt)
        if counts:
            try:
                values = (int(counts[0]), int(counts[1]))
            except (TypeError, ValueError, IndexError):
                values = ()
            for count in values:
                if count > 0:
                    text = str(count) if count < 1000 else "999+"
                    total += (
                        metrics.horizontalAdvance(text)
                        + 2 * scaled_px(5) + _GAP
                    )
        elif index.data(BADGE_ROLE) == "ok":
            height = metrics.height() + scaled_px(2)
            total += max(
                height, metrics.horizontalAdvance(_TICK) + 2 * scaled_px(5),
            ) + _GAP
        summary = index.data(COUNT_ROLE)
        if summary:
            total += metrics.horizontalAdvance(str(summary)) + _GAP
        return total + _EDGE if total else 0

    @staticmethod
    def _small_metrics(opt: QStyleOptionViewItem):
        from PyQt6.QtGui import QFontMetrics

        return QFontMetrics(_small_font(opt))

    def _paint_issues(
        self, painter: QPainter, option: QStyleOptionViewItem, index,
        right: int,
    ) -> int:
        """Counted pills for errors and warnings. Errors rightmost."""
        counts = index.data(ISSUE_ROLE)
        if not counts:
            return right
        try:
            errors, warnings = int(counts[0]), int(counts[1])
        except (TypeError, ValueError, IndexError):
            return right

        pal = CUR()
        font = _small_font(option)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        pad_x = scaled_px(5)
        height = metrics.height() + scaled_px(2)
        top = option.rect.center().y() - height // 2

        for count, token in ((errors, "error"), (warnings, "warning")):
            if count <= 0:
                continue
            text = str(count) if count < 1000 else "999+"
            width = metrics.horizontalAdvance(text) + 2 * pad_x
            rect = QRect(right - width, top, width, height)
            colour = QColor(pal[token])
            fill = QColor(colour)
            # A tint rather than the full colour: a tree of solid red blocks
            # is unreadable, and the number is the message, not the alarm.
            fill.setAlpha(56)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            radius = height / 2.0
            painter.drawRoundedRect(rect, radius, radius)
            painter.setPen(colour)
            painter.drawText(
                rect,
                int(Qt.AlignmentFlag.AlignCenter),
                text,
            )
            right = rect.left() - _GAP
        return right

    def _paint_ok(
        self, painter: QPainter, option: QStyleOptionViewItem, index,
        right: int,
    ) -> int:
        """A green pill with a tick, for a row that was checked and is clean.

        The same shape the counts use, because they answer the same question
        and a bare dot beside two counted pills read as a different kind of
        thing. "Clean" has no quantity, so the pill carries a tick instead of
        a number.
        """
        if index.data(BADGE_ROLE) != "ok" or index.data(ISSUE_ROLE):
            return right
        pal = CUR()
        painter.setFont(_small_font(option))
        metrics = painter.fontMetrics()
        height = metrics.height() + scaled_px(2)
        width = max(height, metrics.horizontalAdvance(_TICK) + 2 * scaled_px(5))
        top = option.rect.center().y() - height // 2
        rect = QRect(right - width, top, width, height)

        colour = QColor(pal[_BADGE_TOKEN["ok"]])
        fill = QColor(colour)
        fill.setAlpha(56)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        radius = height / 2.0
        painter.drawRoundedRect(rect, radius, radius)
        painter.setPen(colour)
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), _TICK)
        return rect.left() - _GAP

    def _paint_count(
        self, painter: QPainter, option: QStyleOptionViewItem, index,
        right: int,
    ) -> None:
        """What a folder holds, dimmed, to the left of any finding counts."""
        text = index.data(COUNT_ROLE)
        if not text:
            return
        painter.setFont(_small_font(option))
        painter.setPen(QColor(CUR()["muted"]))
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(str(text))
        left = right - width
        # Never paint over the name. A deep tree at a large font scale runs
        # out of room, and a summary drawn on top of the thing it summarises
        # is worse than no summary.
        if left <= option.rect.left():
            return
        painter.drawText(
            QRect(left, option.rect.top(), width, option.rect.height()),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            str(text),
        )


__all__ = ["BADGE_ROLE", "COUNT_ROLE", "ISSUE_ROLE", "BidsTreeDelegate"]
