"""Number controls for every panel of the viewer: a slider WITH its number.

Two widgets, used wherever a value is set:

* :class:`NumberControl`: a slider and a number field kept in step, with the
  unit, double-click to go back to the default, and a linear or logarithmic
  slider.
* :class:`RangeControl`: a window (low and high) as two handles on a bar that
  draws the image's histogram, plus the two numbers. Drag a handle to move
  one end, drag between them to slide the window.

Both are built to sit in scroll areas and long panels, so the mouse wheel
moves a value ONLY when the control has focus: otherwise scrolling a panel
would change whatever passed under the pointer. A drag is reported through
``pressed`` / ``released`` so the panel can make it ONE undo step
(:meth:`bidsmgr.viz.store.SceneStore.begin_gesture`).

Painted, not composed of child widgets per tick or bar (guard 8e).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PyQt6.QtCore import QEvent, QLocale, QObject, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QAbstractSpinBox, QDoubleSpinBox, QHBoxLayout, QSizePolicy, QSlider, QVBoxLayout,
    QWidget,
)

from .bridge import ThemeHub

_SLIDER_STEPS = 1000


class _WheelOnlyWhenFocused(QObject):
    """Wheel events reach a control only when it has focus; otherwise they
    go to the scroll area around it."""

    def eventFilter(self, obj, event):  # noqa: N802 - Qt signature
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return False


_WHEEL_GUARD: Optional[_WheelOnlyWhenFocused] = None


def _guard_wheel(widget: QWidget) -> None:
    global _WHEEL_GUARD
    if _WHEEL_GUARD is None:
        _WHEEL_GUARD = _WheelOnlyWhenFocused()
    widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    widget.installEventFilter(_WHEEL_GUARD)


def decimals_for(step: float) -> int:
    """Decimals that show a step of ``step`` (0.05 -> 2, 1 -> 0)."""
    if step <= 0 or step >= 1:
        return 0
    return min(6, int(math.ceil(-math.log10(step) - 1e-9)))


class _Spin(QDoubleSpinBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setKeyboardTracking(False)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setObjectName("viz-number")
        # One number format everywhere, whatever the system's: a decimal
        # point and no thousands separator ("1,000" reads as one or a
        # thousand depending on who reads it).
        locale = QLocale.c()
        locale.setNumberOptions(QLocale.NumberOption.OmitGroupSeparator)
        self.setLocale(locale)
        _guard_wheel(self)


class NumberControl(QWidget):
    """A slider and a number field for one value."""

    #: The user changed the value (never emitted by :meth:`set_value`).
    value_changed = pyqtSignal(float)
    #: A slider drag started / ended (one undo step in between).
    pressed = pyqtSignal()
    released = pyqtSignal()

    def __init__(self, lo: float, hi: float, *, step: float = 0.01, unit: str = "",
                 default: Optional[float] = None, log: bool = False,
                 decimals: Optional[int] = None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("viz-number-control")
        self._lo, self._hi = float(lo), float(hi)
        self._log = bool(log and lo > 0)
        self._default = default
        self._value = float(default if default is not None else lo)
        self._syncing = False
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, _SLIDER_STEPS)
        self.slider.setMinimumWidth(80)
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        _guard_wheel(self.slider)
        self.slider.valueChanged.connect(self._from_slider)
        self.slider.sliderPressed.connect(self.pressed.emit)
        self.slider.sliderReleased.connect(self.released.emit)
        self.slider.installEventFilter(self)
        row.addWidget(self.slider, 1)
        self.spin = _Spin()
        self.spin.setRange(self._lo, self._hi)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals if decimals is not None else decimals_for(step))
        if unit:
            self.spin.setSuffix(f" {unit}")
        # Wide enough for its numbers at the app's font size.
        from . import fonts

        self.spin.setFixedWidth(fonts.px(78 if not unit else 92))
        self.spin.valueChanged.connect(self._from_spin)
        row.addWidget(self.spin)
        if default is not None:
            self.setToolTip(f"Double-click the slider for the default ({default:g}).")
        self.set_value(self._value)

    # -- mapping -------------------------------------------------------
    def _to_pos(self, v: float) -> int:
        lo, hi = self._lo, self._hi
        if hi <= lo:
            return 0
        v = min(max(v, lo), hi)
        if self._log:
            f = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
        else:
            f = (v - lo) / (hi - lo)
        return int(round(f * _SLIDER_STEPS))

    def _from_pos(self, pos: int) -> float:
        f = pos / _SLIDER_STEPS
        lo, hi = self._lo, self._hi
        if self._log:
            return math.exp(math.log(lo) + f * (math.log(hi) - math.log(lo)))
        return lo + f * (hi - lo)

    # -- events ---------------------------------------------------------
    def eventFilter(self, obj, event):  # noqa: N802
        if (obj is self.slider and event.type() == QEvent.Type.MouseButtonDblClick
                and self._default is not None):
            self._commit(float(self._default))
            return True
        return False

    def _from_slider(self, pos: int) -> None:
        if self._syncing:
            return
        v = self._from_pos(pos)
        # Snap to the field's precision, so the number shown is the number set.
        v = round(v, self.spin.decimals())
        self._commit(v, slider=False)

    def _from_spin(self, v: float) -> None:
        if self._syncing:
            return
        self._commit(float(v), spin=False)

    def _commit(self, v: float, *, slider: bool = True, spin: bool = True) -> None:
        v = min(max(float(v), self._lo), self._hi)
        if v == self._value:
            return
        self._value = v
        self._syncing = True
        try:
            if slider:
                self.slider.setValue(self._to_pos(v))
            if spin:
                self.spin.setValue(v)
        finally:
            self._syncing = False
        self.value_changed.emit(v)

    # -- public ---------------------------------------------------------
    def value(self) -> float:
        return self._value

    def minimum(self) -> float:
        return self._lo

    def maximum(self) -> float:
        return self._hi

    def type_value(self, v: float) -> None:
        """Enter ``v`` as a user would, in the number field (tests, scripts)."""
        self.spin.setValue(float(v))

    def set_value(self, v: float) -> None:
        """Show ``v`` without reporting a change."""
        v = min(max(float(v), self._lo), self._hi)
        self._value = v
        self._syncing = True
        try:
            self.slider.setValue(self._to_pos(v))
            self.spin.setValue(v)
        finally:
            self._syncing = False

    def set_range(self, lo: float, hi: float) -> None:
        self._lo, self._hi = float(lo), float(max(hi, lo))
        self._log = self._log and self._lo > 0
        self.spin.setRange(self._lo, self._hi)
        self.set_value(self._value)


# ---------------------------------------------------------------------------
# A window over a histogram
# ---------------------------------------------------------------------------


class _RangeBar(QWidget):
    """The painted bar: histogram, the window band, two handles."""

    HANDLE = 7

    def __init__(self, owner: "RangeControl") -> None:
        super().__init__(owner)
        self.owner = owner
        self.setMinimumHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(30)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self._drag: Optional[str] = None
        self._grab = 0.0
        self._start = (0.0, 0.0)

    def _x(self, v: float) -> float:
        lo, hi = self.owner.domain
        w = max(self.width() - 2 * self.HANDLE, 1)
        return self.HANDLE + (v - lo) / max(hi - lo, 1e-12) * w

    def _v(self, x: float) -> float:
        lo, hi = self.owner.domain
        w = max(self.width() - 2 * self.HANDLE, 1)
        return lo + (x - self.HANDLE) / w * (hi - lo)

    def paintEvent(self, _event) -> None:  # noqa: N802
        theme = ThemeHub.instance().theme
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(theme.token("input_border", "#586069")), 1))
        p.setBrush(QColor(theme.token("surface3", "#1c2128")))
        p.drawRoundedRect(r, 5, 5)
        lo, hi = self.owner.window
        x0, x1 = self._x(lo), self._x(hi)
        band = QColor(theme.accent)
        band.setAlpha(46)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(band)
        p.drawRect(QRectF(min(x0, x1), r.top() + 1, abs(x1 - x0), r.height() - 2))
        hist = self.owner.hist
        if hist is not None:
            counts, edges = hist
            c = np.log1p(np.asarray(counts, dtype=float))
            top = float(c.max()) or 1.0
            path = QPainterPath()
            base = r.bottom() - 2
            height = r.height() - 6
            path.moveTo(self._x(float(edges[0])), base)
            for k, value in enumerate(c):
                path.lineTo(self._x(float(edges[k])), base - height * value / top)
                path.lineTo(self._x(float(edges[k + 1])), base - height * value / top)
            path.lineTo(self._x(float(edges[-1])), base)
            path.closeSubpath()
            fill = QColor(theme.dim)
            fill.setAlpha(110)
            p.setBrush(fill)
            p.drawPath(path)
        handle = QColor(theme.accent)
        for x in (x0, x1):
            p.setPen(QPen(QColor(theme.token("bg", "#0a0e13")), 1))
            p.setBrush(handle)
            p.drawRoundedRect(QRectF(x - 2.5, r.top() + 2, 5, r.height() - 4), 2, 2)
        p.end()

    def _hit(self, x: float) -> str:
        lo, hi = self.owner.window
        x0, x1 = self._x(lo), self._x(hi)
        if abs(x - x0) <= self.HANDLE and abs(x - x0) <= abs(x - x1):
            return "lo"
        if abs(x - x1) <= self.HANDLE:
            return "hi"
        if min(x0, x1) < x < max(x0, x1):
            return "band"
        return "lo" if abs(x - x0) < abs(x - x1) else "hi"

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        self._drag = self._hit(x)
        self._grab = self._v(x)
        self._start = self.owner.window
        self.owner.pressed.emit()
        if self._drag in ("lo", "hi"):
            self._move(x)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag is None:
            hit = self._hit(event.position().x())
            self.setCursor(Qt.CursorShape.OpenHandCursor if hit == "band"
                           else Qt.CursorShape.SizeHorCursor)
            return
        self._move(event.position().x())

    def _move(self, x: float) -> None:
        v = self._v(x)
        lo, hi = self._start
        if self._drag == "lo":
            self.owner._commit(min(v, self.owner.window[1] - self.owner.min_width),
                               self.owner.window[1])
        elif self._drag == "hi":
            self.owner._commit(self.owner.window[0],
                               max(v, self.owner.window[0] + self.owner.min_width))
        elif self._drag == "band":
            d = v - self._grab
            self.owner._commit(lo + d, hi + d)

    def mouseReleaseEvent(self, _event) -> None:  # noqa: N802
        if self._drag is not None:
            self._drag = None
            self.owner.released.emit()

    def mouseDoubleClickEvent(self, _event) -> None:  # noqa: N802
        self.owner.reset_requested.emit()


class RangeControl(QWidget):
    """A window: two handles over the histogram and the two numbers."""

    range_changed = pyqtSignal(float, float)
    pressed = pyqtSignal()
    released = pyqtSignal()
    #: Double-click on the bar: the host decides what "default" is (the
    #: robust range of the image).
    reset_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("viz-range-control")
        self.domain = (0.0, 1.0)
        self.window = (0.0, 1.0)
        self.hist = None
        self.min_width = 1e-6
        self._syncing = False
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        self.bar = _RangeBar(self)
        self.bar.setToolTip("Drag a handle to move one end of the window, drag "
                            "between them to slide it; double-click for the "
                            "image's own range. Behind: the histogram (log).")
        v.addWidget(self.bar)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.lo_spin, self.hi_spin = _Spin(), _Spin()
        from . import fonts

        for spin in (self.lo_spin, self.hi_spin):
            spin.setRange(-1e12, 1e12)
            spin.setFixedWidth(fonts.px(90))
        self.lo_spin.valueChanged.connect(lambda x: self._from_spin(0, x))
        self.hi_spin.valueChanged.connect(lambda x: self._from_spin(1, x))
        row.addWidget(self.lo_spin)
        row.addStretch(1)
        row.addWidget(self.hi_spin)
        v.addLayout(row)

    def _from_spin(self, end: int, x: float) -> None:
        if self._syncing:
            return
        w = list(self.window)
        w[end] = float(x)
        if w[1] > w[0]:
            self._commit(w[0], w[1])

    def _commit(self, lo: float, hi: float) -> None:
        if (lo, hi) == self.window:
            return
        self.set_window(lo, hi)
        self.range_changed.emit(float(lo), float(hi))

    def set_data(self, domain: tuple[float, float], hist=None) -> None:
        """The values the image takes (the bar's ends) and its histogram."""
        lo, hi = float(domain[0]), float(domain[1])
        if not hi > lo:
            hi = lo + 1.0
        self.domain = (lo, hi)
        self.hist = hist
        self.min_width = (hi - lo) * 1e-4
        self._extend()
        self.bar.update()

    def _extend(self) -> None:
        # A window beyond the data stays reachable: the bar grows to show it.
        lo, hi = self.domain
        wlo, whi = self.window
        self.domain = (min(lo, wlo), max(hi, whi))

    def set_window(self, lo: float, hi: float) -> None:
        """Show a window without reporting a change."""
        self.window = (float(lo), float(hi))
        self._extend()
        span = abs(self.domain[1] - self.domain[0])
        decimals = 0 if span >= 1000 else 1 if span >= 100 else 2 if span >= 1 else 4
        self._syncing = True
        try:
            for spin, value in ((self.lo_spin, lo), (self.hi_spin, hi)):
                spin.setDecimals(decimals)
                spin.setSingleStep(max(span / 100.0, 10.0 ** -decimals))
                spin.setValue(float(value))
        finally:
            self._syncing = False
        self.bar.update()


__all__ = ["NumberControl", "RangeControl", "decimals_for"]
