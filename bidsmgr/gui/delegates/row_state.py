"""Row-state painting helper shared by every cell delegate.

A row's *state* (``selected``, ``warn``, ``err``, ``skip``, or empty)
tints the whole row's background with a subtle alpha so the user sees
state at the row level, not just in one cell. Each delegate calls
:func:`paint_row_state` first thing in its ``paint`` to ensure the tint
is applied beneath the text/badge/checkbox.

A second overlay — :data:`HIGHLIGHT_ROLE` / :func:`paint_highlight` —
paints a purple tint on top of the row-state colour for rows that the
user has opted into highlighting (e.g. the "Highlight repeats" toggle
in the Converter toolbar).

Both states are stored on every cell of the row (the model is
responsible for keeping cells in sync).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QStyleOptionViewItem

from ..theme_manager import CUR, rgba


# The role index used by the model to publish per-row state.
ROW_STATE_ROLE: int = Qt.ItemDataRole.UserRole + 1
# Bool role: ``True`` paints a purple tint on top of the row-state tint.
HIGHLIGHT_ROLE: int = Qt.ItemDataRole.UserRole + 3


def paint_row_state(
    painter: QPainter,
    option: QStyleOptionViewItem,
    row_state: Optional[str],
) -> None:
    """Tint the cell's background according to ``row_state``.

    Recognised values: ``"selected"``, ``"warn"``, ``"err"``, ``"skip"``.
    Anything else (including ``None`` / ``""``) is a no-op so the model
    can omit the role for non-special rows.
    """
    if not row_state:
        return
    pal = CUR()
    if row_state == "selected":
        painter.fillRect(option.rect, rgba(pal["accent"], 0.16))
    elif row_state == "warn":
        painter.fillRect(option.rect, rgba(pal["warning"], 0.06))
    elif row_state == "err":
        painter.fillRect(option.rect, rgba(pal["error"], 0.06))
    elif row_state == "skip":
        painter.fillRect(option.rect, QColor(pal["bg"]))


def paint_highlight(
    painter: QPainter,
    option: QStyleOptionViewItem,
    highlight: bool,
) -> None:
    """Overlay a subtle purple tint when ``highlight`` is ``True``.

    Layered ON TOP of :func:`paint_row_state` so a warn row that's
    also repeated still shows its warning tint — purple just adds a
    second wash. Alpha kept low so text stays readable.
    """
    if not highlight:
        return
    pal = CUR()
    painter.fillRect(option.rect, rgba(pal["purple"], 0.14))


__all__ = [
    "HIGHLIGHT_ROLE",
    "ROW_STATE_ROLE",
    "paint_highlight",
    "paint_row_state",
]
