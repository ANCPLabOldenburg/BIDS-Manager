"""Busy indicator drawn as a comet tracing the infinity (∞) curve.

The glyph is a custom-painted ``QWidget`` that animates a head dot
plus a trail of fading dots along a Bernoulli lemniscate (a figure-8
lying on its side). The trail uses the active theme's ``accent``
colour so dark↔light swaps follow automatically — no listener wiring
required.

Public API (unchanged from the previous braille-glyph version)::

    spinner = BusySpinner()
    spinner.set_busy(True, message="Scanning…")
    # … later
    spinner.set_busy(False)
"""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, QSize, QTimer, QPointF
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QHBoxLayout, QWidget

from ..theme_manager import CUR
from .primitives import ElidedLabel


# Glyph footprint. Wider than tall to read clearly as ∞ rather than a
# generic blob; tuned to sit in the toolbar without nudging neighbours.
_GLYPH_W = 44
_GLYPH_H = 20

# Comet shape: one bright head followed by a tail of dimmer/smaller dots.
_DOT_COUNT = 8
_TRAIL_STEP = 0.045   # how far apart (fraction of a lap) consecutive dots sit
_LAP_MS = 1400        # time for the head to complete one full ∞ trip
_TICK_MS = 33         # ~30 fps — smooth motion, modest CPU

# Bernoulli lemniscate y-range is ±0.5/√2 ≈ ±0.3536 of its x amplitude.
# Multiply ``y_raw`` by this constant to renormalise to ±1 so the
# vertical extent of the curve fills the widget height like the x
# extent fills its width.
_Y_NORM = 1.0 / (0.5 / math.sqrt(2.0))


class _Lemniscate(QWidget):
    """Custom-painted ∞-shaped dot trail.

    Owns no timer — the parent :class:`BusySpinner` calls
    :meth:`advance` from its tick. That keeps animation pause/resume
    centralised on visibility changes and avoids two timers racing.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("busy-spinner-glyph")
        self.setFixedSize(_GLYPH_W, _GLYPH_H)
        self._phase = 0.0

    def reset(self) -> None:
        self._phase = 0.0
        self.update()

    def advance(self, dt_ms: int) -> None:
        self._phase = (self._phase + dt_ms / _LAP_MS) % 1.0
        self.update()

    def paintEvent(self, _evt) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)

        accent_hex = CUR().get("accent", "#58a6ff")
        base = QColor(accent_hex)

        w = self.width()
        h = self.height()
        margin = 2.5
        ax = (w / 2.0) - margin
        ay = (h / 2.0) - margin
        cx = w / 2.0
        cy = h / 2.0

        # Draw oldest tail dot first (lowest alpha) so the brighter head
        # dot lands on top of any overlap near the lemniscate crossing.
        for i in range(_DOT_COUNT):
            offset = (_DOT_COUNT - 1 - i) * _TRAIL_STEP
            t = ((self._phase - offset) % 1.0) * 2.0 * math.pi
            sin_t = math.sin(t)
            cos_t = math.cos(t)
            denom = 1.0 + sin_t * sin_t
            x = cx + ax * cos_t / denom
            y = cy + ay * (sin_t * cos_t / denom) * _Y_NORM

            frac = (i + 1) / _DOT_COUNT
            col = QColor(base)
            col.setAlphaF(0.12 + 0.88 * frac)
            radius = 1.0 + 1.7 * frac
            p.setBrush(col)
            p.drawEllipse(QPointF(x, y), radius, radius)


class BusySpinner(QWidget):
    """``[∞-glyph] message`` busy indicator.

    Invisible by default; call :meth:`set_busy(True, "…")` to show it
    and ``set_busy(False)`` to hide. Multiple sequential operations can
    safely call ``set_busy`` repeatedly — the timer is owned by the
    widget and resets on each transition.

    The message is an :class:`~bidsmgr.gui.widgets.primitives.ElidedLabel`,
    so callers may pass a full path or log line of any length: it is
    elided to the room the toolbar has and never resizes the window.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("busy-spinner")
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        self._glyph = _Lemniscate(self)
        h.addWidget(self._glyph)

        # Elided, not plain: the message carries runtime paths and log
        # lines, and a plain QLabel would raise the toolbar's layout
        # minimum by the full text width and grow the window with it.
        self._message = ElidedLabel("")
        self._message.setObjectName("busy-spinner-message")
        h.addWidget(self._message, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._tick)

        # Start hidden — the layout reserves no space when not busy.
        self.setVisible(False)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 — Qt naming
        """Claim no horizontal room, so becoming busy never resizes the window.

        The glyph is a fixed 44px and the message elides on its own, but
        the pair still raised the toolbar's layout minimum the moment the
        spinner appeared — enough to widen a window already sitting at
        its minimum. A cramped toolbar now clips the glyph instead.
        """
        return QSize(0, super().minimumSizeHint().height())

    # ------------------------------------------------------------------
    def set_busy(self, busy: bool, *, message: str = "") -> None:
        """Show / hide the spinner and update its message."""
        if busy:
            self._message.setText(message)
            self._glyph.reset()
            self.setVisible(True)
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()
            self.setVisible(False)
            self._message.setText("")

    def set_message(self, message: str) -> None:
        """Update only the trailing message without restarting the timer."""
        self._message.setText(message)

    def is_busy(self) -> bool:
        return self._timer.isActive()

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        self._glyph.advance(_TICK_MS)


__all__ = ["BusySpinner"]
