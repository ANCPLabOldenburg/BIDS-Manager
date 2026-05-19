"""Status badge widget + the underlying paint helper.

A status badge is a small circular pill with a single-character glyph
inside, used to show severity at a glance (``ok``, ``warn``, ``err``,
``phys``, ``skip``, ``info``). Lift-and-shift from
``inspector_proto/proto.py`` lines 232-274.

The paint helper :func:`badge_paint` is exposed separately because both
the standalone :class:`StatusBadge` widget AND the table delegates
(``StatusDelegate``) need to render the exact same glyph + circle in
their own paint cycles.
"""

from __future__ import annotations

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import QLabel

from ..theme_manager import CUR, rgba, scaled_px


# Single-character glyph for each severity. Matches the prototype exactly.
KIND_CHAR: dict[str, str] = {
    "ok":   "✓",
    "warn": "!",
    "err":  "✕",
    "phys": "P",
    "skip": "−",
    "info": "i",
}

# Background color: which palette token to tint, and at what alpha.
KIND_BG_TOKEN: dict[str, tuple[str, float]] = {
    "ok":   ("success", 0.18),
    "warn": ("warning", 0.18),
    "err":  ("error",   0.18),
    "phys": ("accent",  0.18),
    "skip": ("muted",   0.20),
    "info": ("accent",  0.18),
}

# Foreground (glyph) color token.
KIND_FG_TOKEN: dict[str, str] = {
    "ok": "success", "warn": "warning", "err": "error",
    "phys": "accent", "skip": "dim", "info": "accent",
}


def badge_paint(painter: QPainter, rect: QRect, kind: str, size: int = 16) -> None:
    """Paint a single status badge inside ``rect``.

    Used directly by :class:`StatusBadge` and by ``StatusDelegate`` to
    keep the cell-painted badge pixel-identical to the standalone
    widget. The painter's render hint is set internally; callers do not
    need to enable antialiasing.
    """
    pal = CUR()
    bg_tok, alpha = KIND_BG_TOKEN.get(kind, ("muted", 0.20))
    fg_tok = KIND_FG_TOKEN.get(kind, "dim")

    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    cx, cy = rect.center().x(), rect.center().y()
    r = QRect(cx - size // 2, cy - size // 2, size, size)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(rgba(pal[bg_tok], alpha))
    painter.drawEllipse(r)

    painter.setPen(QColor(pal[fg_tok]))
    f = QFont(painter.font())
    f.setBold(True)
    # Pixel size so the badge glyph renders at the same logical size on
    # macOS (72 dpi) and Linux / Windows (96 dpi). ``scaled_px`` honours
    # the user's "Font scale" preference set via Settings.
    f.setPixelSize(scaled_px(9))
    painter.setFont(f)
    painter.drawText(r, Qt.AlignmentFlag.AlignCenter, KIND_CHAR.get(kind, "?"))


class StatusBadge(QLabel):
    """An 18×18 widget that paints one status badge of the given ``kind``.

    Re-render is automatic on every ``paintEvent``, so a theme change
    is reflected on the next repaint — no listener wiring required as
    long as the parent triggers a repaint (the top-level palette listener
    in ``MainWindow`` calls ``viewport().update()`` on contained widgets).
    """

    def __init__(self, kind: str = "ok", parent=None) -> None:
        super().__init__(parent)
        self._kind = kind
        self.setFixedSize(18, 18)

    @property
    def kind(self) -> str:
        return self._kind

    def set_kind(self, kind: str) -> None:
        """Switch the badge to a different severity and request a repaint."""
        if kind != self._kind:
            self._kind = kind
            self.update()

    def paintEvent(self, event):  # noqa: N802 — Qt naming convention
        p = QPainter(self)
        badge_paint(p, self.rect(), self._kind, 16)
        p.end()


__all__ = [
    "KIND_BG_TOKEN",
    "KIND_CHAR",
    "KIND_FG_TOKEN",
    "StatusBadge",
    "badge_paint",
]
