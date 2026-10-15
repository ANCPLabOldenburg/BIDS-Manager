"""The overview bar under the traces: the whole recording, and where you are.

What replaced the plain time slider, after mne-qt-browser's overview and the
navigators of the acquisition packages: one painted strip showing, end to
end,

* how ACTIVE the recording is, bin by bin (``viz.compute.overview``, on a
  worker), so an artefact, a dropout or a flat stretch shows before anyone
  pages to it;
* the stretches with no samples, hatched;
* every event as a tick in its label's colour, and every span marked bad as
  a red band;
* the time cursor, when one is placed;
* the window on screen as a rectangle. Click anywhere to centre the window
  there, drag to move it, wheel to page.

ONE widget, painted (guard 8e): a few thousand events cost one pass of
``drawLine``, not a few thousand items. Rounded like every other surface of
the app.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QToolTip, QWidget

from ....viz.commands import signal as sigcmd
from ....viz.compute.overview import activity
from ....viz.theme import parse_colour
from ..bridge import connect_while_alive
from ..context import ViewerContext

#: Events painted at most; beyond it they are thinned evenly.
MAX_TICKS = 4000


def _qcolor(value: str, alpha: Optional[int] = None) -> QColor:
    c = QColor(*parse_colour(value))
    if alpha is not None:
        c.setAlpha(alpha)
    return c


class OverviewBar(QWidget):
    """The whole recording in one strip; the visible window as a rectangle."""

    HEIGHT = 30
    RADIUS = 5.0
    #: The band along the top that carries the event ticks.
    TICK_PX = 6.0
    #: The view rectangle is never drawn narrower than this, or a ten
    #: second window over an hour-long recording could not be found.
    MIN_VIEW_PX = 6.0

    def __init__(self, ctx: ViewerContext, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setObjectName("viz-overview")
        self.setFixedHeight(self.HEIGHT)
        self.setMinimumWidth(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._profile: Optional[np.ndarray] = None
        self._gaps: Optional[np.ndarray] = None
        self._profile_of: Optional[int] = None
        self._generation = 0
        self._events_key: Optional[tuple] = None
        self._events: list = []
        self._labels: list[str] = []
        # Seconds between the pointer and the window's start while dragging.
        self._grab: Optional[float] = None
        self._drag_cost = 0.0
        self._drag_last = 0.0
        self._wheel_acc = 0.0
        ctx.qstore.changed.connect(self._on_changed)
        ctx.jobs.done.connect(self._on_job_done)
        connect_while_alive(ctx.theme_hub.changed, self, lambda w, _t: w.update())
        connect_while_alive(ctx.settings_hub.changed, self, lambda w, _s: w.update())

    # ------------------------------------------------------------------
    @property
    def source(self):
        return sigcmd.source(self.ctx.store)

    def profile(self) -> Optional[np.ndarray]:
        """The activity profile drawn (None until the worker has it)."""
        return self._profile

    def _on_changed(self, paths) -> None:
        if any(p == "scene" or p.startswith("sources") for p in paths):
            self.refresh()
        elif any(p.startswith("traces") or p.startswith("cursor") for p in paths):
            self.update()

    def refresh(self) -> None:
        """Start the activity profile of the source now open, once per
        source (a resampled recording is a new one)."""
        src = self.source
        if src is None:
            self.ctx.jobs.cancel("overview")
            self._profile = self._gaps = None
            self._profile_of = None
            self.update()
            return
        if self._profile_of != id(src):
            self._profile = self._gaps = None
            self._profile_of = id(src)
            self._generation += 1
            self.ctx.jobs.start("overview", self._generation, activity, src)
        self.update()

    def _on_job_done(self, tag: str, generation: int, result) -> None:
        if tag != "overview" or generation != self._generation:
            return
        self._profile = np.asarray(result["profile"], dtype=float)
        self._gaps = np.asarray(result["gaps"], dtype=bool)
        self.update()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _inner(self) -> QRectF:
        return QRectF(self.rect()).adjusted(3.0, 3.0, -3.0, -3.0)

    def _x_of(self, t: float, duration: float) -> float:
        inner = self._inner()
        return inner.left() + (t / duration if duration > 0 else 0.0) * inner.width()

    def _t_of(self, x: float, duration: float) -> float:
        inner = self._inner()
        f = (x - inner.left()) / max(inner.width(), 1.0)
        return min(max(f, 0.0), 1.0) * duration

    def view_rect(self) -> Optional[QRectF]:
        """Where the window on screen is drawn (tests, hit testing)."""
        src = self.source
        if src is None or src.duration <= 0:
            return None
        tr = self.ctx.scene.traces
        inner = self._inner()
        x0 = self._x_of(tr.t0, src.duration)
        x1 = self._x_of(min(tr.t0 + tr.width, src.duration), src.duration)
        if x1 - x0 < self.MIN_VIEW_PX:
            mid = (x0 + x1) / 2.0
            x0, x1 = mid - self.MIN_VIEW_PX / 2.0, mid + self.MIN_VIEW_PX / 2.0
        return QRectF(x0, inner.top(), x1 - x0, inner.height())

    def _event_list(self, src) -> list:
        tr = self.ctx.scene.traces
        key = (id(src), tr.event_source)
        if key != self._events_key:
            events = src.events(tr.event_source)
            if len(events) > MAX_TICKS:
                events = events[:: len(events) // MAX_TICKS]
            self._events = events
            self._labels = sorted({e.label for e in events})
            self._events_key = key
        return self._events

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:  # noqa: N802
        theme = self.ctx.theme
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        outline = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        shape = QPainterPath()
        shape.addRoundedRect(outline, self.RADIUS, self.RADIUS)
        p.fillPath(shape, _qcolor(theme.token("surface2", "#161b22")))
        p.save()
        p.setClipPath(shape)
        src = self.source
        if src is not None and src.duration > 0:
            self._paint_content(p, src)
        p.restore()
        p.setPen(QPen(_qcolor(theme.token("border", "#21262d")), 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(shape)
        p.end()

    def _paint_content(self, p: QPainter, src) -> None:
        theme = self.ctx.theme
        inner = self._inner()
        dur = src.duration
        offset = src.start_time
        # Events take a thin band along the top; the activity profile has
        # the rest. A thousand ticks over the full height buried it.
        band = QRectF(inner.left(), inner.top(), inner.width(), self.TICK_PX)
        area = QRectF(inner.left(), band.bottom() + 1.0, inner.width(),
                      inner.bottom() - band.bottom() - 1.0)
        prof = self._profile
        if prof is not None and prof.size:
            n = prof.size
            xs = area.left() + (np.arange(n) + 0.5) / n * area.width()
            ys = area.bottom() - np.clip(prof, 0.0, 1.0) * area.height()
            pts = [QPointF(area.left(), area.bottom())]
            pts += [QPointF(float(x), float(y)) for x, y in zip(xs, ys)]
            pts.append(QPointF(area.right(), area.bottom()))
            p.setPen(QPen(_qcolor(theme.dim, 170), 1.0))
            p.setBrush(_qcolor(theme.dim, 70))
            p.drawPolygon(QPolygonF(pts))
            gaps = self._gaps
            if gaps is not None and gaps.any():
                brush = QBrush(_qcolor(theme.token("muted", "#656d76"), 170),
                               Qt.BrushStyle.BDiagPattern)
                p.setPen(Qt.PenStyle.NoPen)
                w = area.width() / n
                for k in np.flatnonzero(gaps):
                    p.fillRect(QRectF(area.left() + k * w, area.top(), max(w, 1.0),
                                      area.height()), brush)
        # Bad segments: the review's (marked in this session, or the
        # recording's own), as red bands the full height.
        bad = _qcolor(theme.token("error", "#f85149"), 80)
        p.setPen(Qt.PenStyle.NoPen)
        for sp in sigcmd.bad_spans(self.ctx.store):
            x0 = self._x_of(sp.onset - offset, dur)
            x1 = self._x_of(sp.onset - offset + max(sp.duration, 0.0), dur)
            p.fillRect(QRectF(x0, inner.top(), max(x1 - x0, 1.5), inner.height()), bad)
        events = self._event_list(src)
        if events:
            ts = self.ctx.settings.traces
            by_colour: dict[str, list] = {}
            for e in events:
                if getattr(e, "kind", "") == "bad":
                    continue
                colour = ts.event_color or theme.series(self._labels.index(e.label))
                by_colour.setdefault(colour, []).append(e.onset - offset)
            for colour, onsets in by_colour.items():
                p.setPen(QPen(_qcolor(colour, 230), 1.0))
                for t in onsets:
                    if 0.0 <= t <= dur:
                        x = self._x_of(t, dur)
                        p.drawLine(QPointF(x, band.top()), QPointF(x, band.bottom()))
        # The time cursor.
        cursor = self.ctx.scene.cursor.time
        accent = _qcolor(theme.accent)
        if cursor is not None and 0.0 <= cursor - offset <= dur:
            x = self._x_of(cursor - offset, dur)
            p.setPen(QPen(_qcolor(theme.text), 1.0))
            p.drawLine(QPointF(x, inner.top()), QPointF(x, inner.bottom()))
        # The window on screen.
        rect = self.view_rect()
        if rect is not None:
            p.setPen(QPen(accent, 1.5))
            p.setBrush(_qcolor(theme.accent, 45))
            p.drawRoundedRect(rect, 3.0, 3.0)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _go(self, t0: float, *, finish: bool = False) -> None:
        """Move the window, paced by the cost of the last redraw (the same
        rule a drag on the traces follows), always on the release."""
        now = time.perf_counter()
        if not finish and now - self._drag_last < max(2.0 * self._drag_cost, 1.0 / 60.0):
            return
        self.ctx.run("time.set", t0=float(t0))
        self.ctx.qstore.flush()
        self._drag_cost = time.perf_counter() - now
        self._drag_last = time.perf_counter()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        src = self.source
        if src is None or src.duration <= 0 or event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        tr = self.ctx.scene.traces
        x = event.position().x()
        t = self._t_of(x, src.duration)
        rect = self.view_rect()
        if rect is not None and rect.left() <= x <= rect.right():
            self._grab = t - tr.t0
        else:
            # Outside the window: centre it on the click, then drag from there.
            self._grab = tr.width / 2.0
            self._go(t - self._grab, finish=True)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        src = self.source
        if src is None or src.duration <= 0:
            return
        x = event.position().x()
        t = self._t_of(x, src.duration)
        if self._grab is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._go(t - self._grab)
            return
        rect = self.view_rect()
        inside = rect is not None and rect.left() <= x <= rect.right()
        self.setCursor(Qt.CursorShape.OpenHandCursor if inside
                       else Qt.CursorShape.PointingHandCursor)
        QToolTip.showText(event.globalPosition().toPoint(), self._tip(src, t), self)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        src = self.source
        if self._grab is not None and src is not None:
            t = self._t_of(event.position().x(), src.duration)
            self._go(t - self._grab, finish=True)
        self._grab = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def wheelEvent(self, event) -> None:  # noqa: N802
        ad, pd = event.angleDelta(), event.pixelDelta()
        use_pixel = not pd.isNull()
        dx = pd.x() if use_pixel else ad.x()
        dy = pd.y() if use_pixel else ad.y()
        delta = dx if abs(dx) > abs(dy) else dy
        self._wheel_acc += delta
        thresh = 40.0 if use_pixel else 120.0
        steps = int(self._wheel_acc / thresh)
        self._wheel_acc -= steps * thresh
        if steps:
            self.ctx.run("time.page", n=-float(steps))
        event.accept()

    def _tip(self, src, t: float) -> str:
        text = f"{t + src.start_time:.1f} s"
        if self._gaps is not None and self._gaps.size:
            k = min(int(t / src.duration * self._gaps.size), self._gaps.size - 1)
            if self._gaps[k]:
                text += ": no samples here"
        return text + "\nClick to go there, drag the window, wheel to page"


__all__ = ["OverviewBar"]
