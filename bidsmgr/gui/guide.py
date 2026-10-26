"""The guide: what would you like to do, and then one control at a time.

Switched on and off in the header, left of Settings (:class:`GuideSwitch`,
remembered under ``ui/guided_mode``). While it is on, the guide ASKS what the
user wants to do in the view they are in (:class:`GuideQuestion`: when the app
opens, when the user comes to a view, after a task is finished, and from the
card's All tasks), then walks them through the task they chose: a ring around
the control to use (painted by :class:`GuideOverlay`) and a card beside it
(:class:`GuideCard`) saying what to do, with Previous, Next and All tasks. A
step also moves on by itself when what it asks for happens.

The pieces:

- :mod:`guide_steps` holds the tasks and their steps, decided from facts each
  view reports (``guide_facts``), naming controls each view publishes
  (``guide_targets``); no Qt.
- The overlay, the card, the question and the small Guide button left when the
  user says Not now (:class:`GuideRest`) are CHILDREN of the central widget,
  never windows (GUI_LOOK_AND_FEEL 4.26). The overlay is transparent to the
  mouse; the card, the question and the button are styled by the stylesheet,
  so they follow the theme and the font size.
- :class:`Guide` reads the window four times a second while it is on and
  changes something only when the step, the control's place or the card's
  size changed; the overlay repaints only the ring's region.

Hidden while a dialog is open: the user is busy with it, and a ring around a
control they cannot reach would be noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from PyQt6.QtCore import (
    QEasingCurve,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    QVariantAnimation,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .guide_steps import Step, Task, find_task, should_advance, shown, tasks
from .viz import fonts

#: How often the window is read while the guide is on.
POLL_MS = 250
#: The key of the question's "Continue" answer.
CONTINUE = "__continue__"


def _palette() -> dict:
    from .theme_manager import CUR

    return CUR()


def _colour(token: str, alpha: float = 1.0) -> QColor:
    from .widgets.primitives import qcolor

    c = QColor(qcolor(_palette().get(token, "#888888")))
    c.setAlphaF(max(0.0, min(1.0, c.alphaF() * alpha)))
    return c


def _area(r: QRectF) -> float:
    return max(0.0, r.width()) * max(0.0, r.height())


def _blend(a: QColor, b: QColor, t: float) -> QColor:
    return QColor.fromRgbF(
        a.redF() + (b.redF() - a.redF()) * t,
        a.greenF() + (b.greenF() - a.greenF()) * t,
        a.blueF() + (b.blueF() - a.blueF()) * t,
        a.alphaF() + (b.alphaF() - a.alphaF()) * t,
    )


def _repolish(widget: QWidget) -> None:
    """Re-read the stylesheet after a property it selects on changed."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def _is_guide(widget: QWidget) -> bool:
    return widget.objectName().startswith("guide-")


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


class GuideSwitch(QAbstractButton):
    """A labelled on/off switch, painted: a track with a knob, then "Guide".

    Painted rather than styled so the knob is round and the track a pill at
    every font size and on every platform (the reason ``Chip`` paints its
    pill too). Colours are read at paint time, so a theme change needs only
    an ``update``; sizes come from ``fonts.px``, so a font-size change needs
    only ``updateGeometry``.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("guide-switch")
        self.setCheckable(True)
        self.setText("Guide")
        self.setAccessibleName("Guide")
        self.setToolTip("Guide: asks what you want to do, then shows the way, one "
                        "step at a time")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # Reached with Tab and toggled with Space; a click leaves no outline.
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._knob = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_knob)
        self.toggled.connect(self._slide)

    def set_on(self, on: bool) -> None:
        """Set without the slide and without ``toggled``'s side effects."""
        self.blockSignals(True)
        self.setChecked(bool(on))
        self.blockSignals(False)
        self._anim.stop()
        self._knob = 1.0 if on else 0.0
        self.update()

    def _slide(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._knob)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def _on_knob(self, value) -> None:
        self._knob = float(value)
        self.update()

    def _font(self) -> QFont:
        f = fonts.font(12)
        f.setWeight(QFont.Weight.DemiBold if self.isChecked() else QFont.Weight.Medium)
        return f

    def _geometry(self) -> tuple[int, int, int, int]:
        """``(padding, track width, track height, gap)`` at the font scale."""
        return fonts.px(8), fonts.px(28), fonts.px(16), fonts.px(7)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        pad, track_w, track_h, gap = self._geometry()
        f = fonts.font(12)
        f.setWeight(QFont.Weight.DemiBold)    # room for the bolder "on" label
        fm = QFontMetrics(f)
        return QSize(pad + track_w + gap + fm.horizontalAdvance(self.text()) + pad,
                     max(fonts.px(28), fm.height() + fonts.px(10)))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return self.sizeHint()

    def repaint_for_palette(self) -> None:
        self.updateGeometry()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        pad, track_w, track_h, gap = self._geometry()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        hovered = self.underMouse()
        if hovered or self.isDown():
            p.setPen(QPen(_colour("border"), 1))
            p.setBrush(_colour("surface2" if self.isDown() else "surface3"))
            p.drawRoundedRect(r, 6, 6)
        if self.hasFocus() and not hovered:
            p.setPen(QPen(_colour("accent"), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r, 6, 6)

        t = self._knob
        track = QRectF(pad, (self.height() - track_h) / 2.0, track_w, track_h)
        p.setPen(QPen(_blend(_colour("input_border"), _colour("accent"), t), 1))
        p.setBrush(_blend(_colour("surface3"), _colour("accent"), t))
        p.drawRoundedRect(track, track_h / 2.0, track_h / 2.0)
        inset = max(2.0, track_h * 0.15)
        d = track_h - 2 * inset
        x = track.left() + inset + (track_w - 2 * inset - d) * t
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_blend(_colour("dim"), _colour("primary_btn_text"), t))
        p.drawEllipse(QRectF(x, track.top() + inset, d, d))

        p.setFont(self._font())
        p.setPen(_colour("text") if (self.isChecked() or hovered) else _colour("dim"))
        text = QRectF(track.right() + gap, 0, self.width() - track.right() - gap, self.height())
        p.drawText(text, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                   self.text())


# ---------------------------------------------------------------------------
# The overlay: the ring, and the card's arrow
# ---------------------------------------------------------------------------


class GuideOverlay(QWidget):
    """The ring around the step's control and the arrow from the card to it,
    painted; and where the card goes (:meth:`place`).

    Repaints only the ring's and the arrow's region: the overlay covers the
    whole window, and an ``update()`` of all of it repaints every table and
    tree under it.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("guide-overlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.hide()
        self._step: Optional[Step] = None
        self._target: Optional[QRectF] = None
        self._anchor = QRectF()
        self._keep_clear: tuple = ()
        self._card = QRectF()
        self._side: Optional[str] = None
        self._tip = QPointF()
        self._entrance = 1.0
        self._dirty = QRect()
        self._enter = QVariantAnimation(self)
        self._enter.setDuration(600)
        self._enter.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._enter.setStartValue(0.0)
        self._enter.setEndValue(1.0)
        self._enter.valueChanged.connect(self._on_entrance)

    # -- what is shown ------------------------------------------------------

    @property
    def step(self) -> Optional[Step]:
        return self._step

    @property
    def target_rect(self) -> Optional[QRectF]:
        return self._target

    @property
    def card_rect(self) -> QRectF:
        return QRectF(self._card)

    def clear(self) -> None:
        self._enter.stop()
        self._step = None
        self._target = None
        self._card = QRectF()
        self.hide()

    def place(self, step: Step, target: Optional[QRect], anchor: QRect,
              keep_clear: tuple, card: QSize) -> QRect:
        """Ring ``target`` (overlay coordinates; ``None`` when the control is
        not on screen) and return where a card of size ``card`` goes: beside
        it, covering least of what the user may need and none of
        ``keep_clear``; at the top of ``anchor`` when there is no control."""
        new_target = QRectF(target) if target is not None else None
        clear = tuple(QRectF(r) for r in keep_clear)
        new_step = self._step is None or step.key != self._step.key
        same = (not new_step and step == self._step and new_target == self._target
                and QRectF(anchor) == self._anchor and clear == self._keep_clear
                and self._card.size() == QRectF(QPointF(), card.toSizeF()).size()
                and self.isVisible())
        if same:
            return self._card.toAlignedRect()
        self._step = step
        self._target = new_target
        self._anchor = QRectF(anchor)
        self._keep_clear = clear
        self._card, self._side, self._tip = self._place(float(card.width()),
                                                         float(card.height()))
        if new_step:
            self._enter.stop()
            if step.busy:
                # The user just pressed this control and the app is starting
                # its work: shown at once, so no frame is painted while that
                # work begins (measured: an entrance there added a 200 ms
                # pause to the start of a conversion).
                self._entrance = 1.0
            else:
                self._entrance = 0.0
                self._enter.start()
        if not self.isVisible():
            self.show()
        self._repaint()
        return self._card.toAlignedRect()

    # -- placing the card ---------------------------------------------------

    #: What the card should not cover, by kind, with a weight: a button or a
    #: field the user may need next counts fully, a text a quarter, a list or
    #: table (large, mostly content) a little.
    _WEIGHTS = ((QAbstractButton, 1.0), (QLineEdit, 1.0), (QComboBox, 1.0),
                (QAbstractSpinBox, 1.0), (QAbstractItemView, 0.15), (QLabel, 0.25))

    def _obstacles(self, roi: QRectF) -> list[tuple[QRectF, float]]:
        """What lies in ``roi`` that the card should not cover, with
        weights, plus what the step's text sends the user to (four times).

        Walks only the VISIBLE widgets that reach into ``roi``, pruning hidden
        branches (the other views of the window) and branches outside it, and
        stops at a control; the guide's own widgets are not obstacles.
        Measured: ``findChildren`` over the whole window took up to 470 ms
        while a conversion filled the panels.
        """
        found: list[tuple[QRectF, float]] = [(r, 4.0) for r in self._keep_clear]
        host = self.parentWidget()
        if host is None:
            return found
        stack = [(w, QRectF(w.geometry()), QRectF(host.rect()))
                 for w in host.children()
                 if isinstance(w, QWidget) and not w.isHidden() and not _is_guide(w)]
        while stack:
            widget, rect, clip = stack.pop()
            seen = rect.intersected(clip)
            if seen.isEmpty() or not seen.intersects(roi):
                continue
            weight = next((wt for kind, wt in self._WEIGHTS if isinstance(widget, kind)), 0.0)
            if weight:
                found.append((seen, weight))
                continue                      # a control: nothing inside it counts
            for child in widget.children():
                if isinstance(child, QWidget) and not child.isHidden() \
                        and not child.isWindow():
                    stack.append((child, QRectF(child.geometry()).translated(rect.topLeft()),
                                  seen))
        return found

    def _place(self, w: float, h: float) -> tuple[QRectF, Optional[str], QPointF]:
        """Beside the control, where the card covers the least the user may
        need (preferring below, then above, right, left); inside it, with no
        arrow, when it fills the room; at the top of the view when it is off
        screen."""
        gap = fonts.px(14)
        m = fonts.px(8)
        area = QRectF(self.rect()).adjusted(m, m, -m, -m)

        def clamp(v: float, lo: float, hi: float) -> float:
            return max(lo, min(v, hi)) if hi >= lo else lo

        t = self._target
        if t is None:
            a = self._anchor if not self._anchor.isEmpty() else area
            x = clamp(a.center().x() - w / 2, area.left(), area.right() - w)
            return QRectF(x, clamp(a.top() + fonts.px(16), area.top(), area.bottom() - h),
                          w, h), None, QPointF()
        xs = (t.center().x() - w / 2, t.left(), t.right() - w)
        ys = (t.center().y() - h / 2, t.top())
        candidates = []
        for side, order in (("below", 0), ("above", 1), ("right", 2), ("left", 3)):
            for i, along in enumerate(xs if side in ("below", "above") else ys):
                if side == "below":
                    rect = QRectF(clamp(along, area.left(), area.right() - w),
                                  t.bottom() + gap, w, h)
                elif side == "above":
                    rect = QRectF(clamp(along, area.left(), area.right() - w),
                                  t.top() - gap - h, w, h)
                elif side == "right":
                    rect = QRectF(t.right() + gap,
                                  clamp(along, area.top(), area.bottom() - h), w, h)
                else:
                    rect = QRectF(t.left() - gap - w,
                                  clamp(along, area.top(), area.bottom() - h), w, h)
                if area.contains(rect):
                    candidates.append((side, rect, order * 3 + i))
        if candidates:
            roi = QRectF(candidates[0][1])
            for _side, rect, _rank in candidates[1:]:
                roi = roi.united(rect)
            obstacles = [(r, wt) for r, wt in self._obstacles(roi)
                         if not t.contains(r) and not r.contains(t)]

            def cost(candidate) -> float:
                _side, rect, rank = candidate
                covered = sum(wt * _area(rect.intersected(r)) for r, wt in obstacles)
                return covered + rank      # equal cover: the preferred side
            side, rect, _rank = min(candidates, key=cost)
            tip = {
                "below": QPointF(t.center().x(), t.bottom() + fonts.px(5)),
                "above": QPointF(t.center().x(), t.top() - fonts.px(5)),
                "right": QPointF(t.right() + fonts.px(5), t.center().y()),
                "left": QPointF(t.left() - fonts.px(5), t.center().y()),
            }[side]
            return rect, side, tip
        inside = t.intersected(area)
        x = clamp(inside.center().x() - w / 2, area.left(), area.right() - w)
        y = clamp(inside.top() + fonts.px(24), area.top(), area.bottom() - h)
        return QRectF(x, y, w, h), None, QPointF()

    # -- painting -----------------------------------------------------------

    def _region(self) -> QRect:
        grow = fonts.px(24)
        r = QRectF()
        if self._target is not None:
            r = self._target.adjusted(-grow, -grow, grow, grow)
        if self._side is not None:
            pad = fonts.px(12)
            r = r.united(QRectF(self._tip, self._tip).adjusted(-pad, -pad, pad, pad))
            r = r.united(self._arrow_base().boundingRect().adjusted(-2, -2, 2, 2))
        return r.toAlignedRect()

    def _repaint(self) -> None:
        region = self._region()
        self.update(region.united(self._dirty) if not self._dirty.isNull() else region)
        self._dirty = region

    def _on_entrance(self, value) -> None:
        # About twenty frames, not the animation clock's sixty: each one
        # repaints whatever lies under the ring.
        value = float(value)
        if value < 1.0 and value - self._entrance < 0.05:
            return
        self._entrance = value
        self._repaint()

    def _accent(self) -> str:
        return "success" if self._step is not None and self._step.done else "accent"

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._step is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._target is not None:
            self._paint_ring(p, self._entrance)
        if self._side is not None:
            self._paint_arrow(p)

    def _paint_ring(self, p: QPainter, e: float) -> None:
        grow = fonts.px(4) + (1.0 - e) * fonts.px(12)
        # Kept inside the window: a card at its edge keeps its whole ring.
        edge = fonts.px(10) / 2.0 + 1
        r = self._target.adjusted(-grow, -grow, grow, grow).intersected(
            QRectF(self.rect()).adjusted(edge, edge, -edge, -edge))
        radius = min(float(fonts.px(10)), r.height() / 2.0)
        accent = self._accent()
        p.setBrush(Qt.BrushStyle.NoBrush)
        # A soft glow, then a band in the canvas colour that keeps the line
        # off whatever is under it, then the line (GUI_LOOK_AND_FEEL 4.6).
        p.setPen(QPen(_colour(accent, 0.22 * e), fonts.px(10)))
        p.drawRoundedRect(r, radius, radius)
        p.setPen(QPen(_colour("bg", 0.85 * e), fonts.px(4)))
        p.drawRoundedRect(r, radius, radius)
        line = QPen(_colour(accent, e), max(2, fonts.px(2)))
        if self._step.busy:
            line.setStyle(Qt.PenStyle.DashLine)
        p.setPen(line)
        p.drawRoundedRect(r, radius, radius)

    def _arrow_base(self) -> QPolygonF:
        """The arrow from the card's edge to the control: base on the edge,
        point at the control."""
        b = self._card
        half = fonts.px(7)
        inset = fonts.px(10) + half
        tip = self._tip
        if self._side in ("below", "above"):
            x = max(b.left() + inset, min(tip.x(), b.right() - inset))
            edge = b.top() if self._side == "below" else b.bottom()
            return QPolygonF([QPointF(x - half, edge), QPointF(x, tip.y()),
                              QPointF(x + half, edge)])
        y = max(b.top() + inset, min(tip.y(), b.bottom() - inset))
        edge = b.left() if self._side == "right" else b.right()
        return QPolygonF([QPointF(edge, y - half), QPointF(tip.x(), y),
                          QPointF(edge, y + half)])

    def _paint_arrow(self, p: QPainter) -> None:
        """Filled in the card's colour over the card's own outline where they
        meet, then its two slanted sides in the outline's colour: card and
        arrow read as one shape."""
        poly = self._arrow_base()
        a, tip, c = (QPointF(poly[i]) for i in range(3))   # copies: poly is temporary
        inward = {"below": QPointF(0, 1.5), "above": QPointF(0, -1.5),
                  "right": QPointF(1.5, 0), "left": QPointF(-1.5, 0)}[self._side]
        body = QPainterPath()
        body.addPolygon(QPolygonF([a + inward, tip, c + inward]))
        body.closeSubpath()
        p.fillPath(body, _colour("surface"))
        p.setPen(QPen(_colour(self._accent()), 1.2))
        p.drawLine(a, tip)
        p.drawLine(tip, c)


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


class GuideCard(QFrame):
    """The step's card beside its control: the task and the step's place in
    it, what to do and why, and the way around: All tasks (back to the
    question), Previous, Next (Finish on the last step)."""

    previous = pyqtSignal()
    next = pyqtSignal()
    all_tasks = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("guide-card")
        self.hide()
        self._step: Optional[Step] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(fonts.px(14), fonts.px(12), fonts.px(14), fonts.px(10))
        lay.setSpacing(fonts.px(4))
        self._kicker = QLabel(self)
        self._kicker.setObjectName("guide-card-kicker")
        self._kicker.setWordWrap(True)
        self._title = QLabel(self)
        self._title.setObjectName("guide-card-title")
        self._title.setWordWrap(True)
        self._body = QLabel(self)
        self._body.setObjectName("guide-card-body")
        self._body.setWordWrap(True)
        for label in (self._kicker, self._title, self._body):
            lay.addWidget(label)
        rule = QFrame(self)
        rule.setObjectName("guide-card-rule")
        rule.setFixedHeight(1)
        lay.addSpacing(fonts.px(4))
        lay.addWidget(rule)
        foot = QHBoxLayout()
        foot.setContentsMargins(0, fonts.px(2), 0, 0)
        foot.setSpacing(fonts.px(6))
        self._tasks_btn = self._button("All tasks", "guide-tasks",
                                       "Back to the list of things to do here")
        self._prev_btn = self._button("‹  Previous", "guide-prev", "The step before this one")
        self._next_btn = self._button("Next  ›", "guide-next", "The step after this one")
        self._tasks_btn.clicked.connect(self.all_tasks.emit)
        self._prev_btn.clicked.connect(self.previous.emit)
        self._next_btn.clicked.connect(self.next.emit)
        foot.addWidget(self._tasks_btn)
        foot.addStretch(1)
        foot.addWidget(self._prev_btn)
        foot.addWidget(self._next_btn)
        lay.addLayout(foot)

    def _button(self, text: str, name: str, tip: str) -> QPushButton:
        btn = QPushButton(text, self)
        btn.setObjectName(name)
        btn.setToolTip(tip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn

    @property
    def step(self) -> Optional[Step]:
        return self._step

    def set_step(self, step: Step, *, first: bool, last: bool) -> None:
        """Show ``step``; Previous is off on the first, Next reads Finish on
        the last."""
        if step == self._step and self._prev_btn.isEnabled() == (not first) \
                and (self._next_btn.text() == "Finish") == last:
            return
        self._step = step
        self._kicker.setText(step.heading.upper())
        self._kicker.setVisible(bool(step.heading))
        self._title.setText(step.title)
        self._body.setText(step.text)
        self._prev_btn.setEnabled(not first)
        self._next_btn.setText("Finish" if last else "Next  ›")
        self._next_btn.setToolTip("End this task and choose another" if last
                                  else "The step after this one")
        if bool(self.property("done")) != step.done:
            self.setProperty("done", step.done)
            self._kicker.setProperty("done", step.done)
            _repolish(self)
            _repolish(self._kicker)

    def size_for(self, width: int) -> QSize:
        """The card's size at ``width``: its labels wrap to it."""
        self.setFixedWidth(width)
        lay = self.layout()
        lay.activate()
        height = lay.totalHeightForWidth(width) if lay.hasHeightForWidth() \
            else lay.totalSizeHint().height()
        return QSize(width, height)


# ---------------------------------------------------------------------------
# The question
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Answer:
    """One answer the question offers."""

    key: str
    label: str
    detail: str
    recommended: bool = False


class _ChoiceRow(QFrame):
    """One answer: its label, a line on what it leads to, "Next step" on the
    path's own. A frame rather than a button so the two lines wrap and the
    card's layout measures them; clicked, or Return/Space with the focus."""

    clicked = pyqtSignal()

    def __init__(self, answer: Answer, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("guide-choice")
        self.setProperty("recommended", bool(answer.recommended))
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName(answer.label)
        self.setToolTip(answer.detail)
        self.key = answer.key
        lay = QVBoxLayout(self)
        lay.setContentsMargins(fonts.px(12), fonts.px(8), fonts.px(12), fonts.px(8))
        lay.setSpacing(fonts.px(2))
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(fonts.px(8))
        title = QLabel(answer.label, self)
        title.setObjectName("guide-choice-title")
        title.setWordWrap(True)
        top.addWidget(title, 1)
        labels = [title]
        if answer.recommended:
            tag = QLabel("Next step" if answer.key != CONTINUE else "In progress", self)
            tag.setObjectName("guide-choice-tag")
            top.addWidget(tag, 0, Qt.AlignmentFlag.AlignTop)
            labels.append(tag)
        lay.addLayout(top)
        detail = QLabel(answer.detail, self)
        detail.setObjectName("guide-choice-detail")
        detail.setWordWrap(True)
        lay.addWidget(detail)
        lay.addStretch(1)
        labels.append(detail)
        for label in labels:
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Taken here: a frame ignores a press, which then travels up to the
        # window (where it read as a click elsewhere) and the release never
        # comes back to this row.
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.MouseButton.LeftButton \
                and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(event)


class GuideQuestion(QFrame):
    """The question "What would you like to do?", in a card over the views.

    One column of answers, or two when there are more than four; the answers
    scroll when the window is too low for them. The path's own next step is
    tinted and holds the focus (Return takes it). An answer emits ``chosen``;
    Not now, Escape or a click anywhere else emits ``dismissed``, and that
    click still does what it does.
    """

    chosen = pyqtSignal(str)
    dismissed = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("guide-question")
        self.hide()
        self._columns = 1
        self._closing = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(fonts.px(16), fonts.px(14), fonts.px(16), fonts.px(12))
        lay.setSpacing(fonts.px(8))
        kicker = QLabel("GUIDE", self)
        kicker.setObjectName("guide-question-kicker")
        self._title = QLabel("What would you like to do?", self)
        self._title.setObjectName("guide-question-title")
        self._title.setWordWrap(True)
        lay.addWidget(kicker)
        lay.addWidget(self._title)
        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("guide-question-scroll")
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._rows_widget = QWidget()
        self._rows_widget.setObjectName("guide-question-rows")
        self._rows = QGridLayout(self._rows_widget)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(fonts.px(6))
        self._scroll.setWidget(self._rows_widget)
        self._scroll.viewport().setAutoFillBackground(False)
        lay.addWidget(self._scroll)
        foot = QHBoxLayout()
        foot.setContentsMargins(0, 0, 0, 0)
        foot.addStretch(1)
        later = QPushButton("Not now", self)
        later.setObjectName("guide-dismiss")
        later.setCursor(Qt.CursorShape.PointingHandCursor)
        later.setToolTip("Close; the Guide button in the corner brings this back")
        later.clicked.connect(self.dismissed.emit)
        foot.addWidget(later)
        lay.addLayout(foot)

    def choices(self) -> list[str]:
        return [row.key for row in self._rows_widget.findChildren(_ChoiceRow)
                if not row.isHidden()]

    @property
    def title(self) -> str:
        return self._title.text()

    def ask(self, answers: Sequence[Answer], anchor: QRect,
            title: str = "What would you like to do?") -> None:
        """Show ``answers`` over the top of ``anchor`` (the views' area)."""
        self._title.setText(title)
        while self._rows.count():
            row = self._rows.takeAt(0).widget()
            if row is not None:
                row.hide()                    # before it loses its parent (4.26)
                row.setParent(None)
                row.deleteLater()
        self._columns = 2 if len(answers) > 4 else 1
        for i, answer in enumerate(answers):
            row = _ChoiceRow(answer, self._rows_widget)
            row.clicked.connect(lambda key=answer.key: self.chosen.emit(key))
            self._rows.addWidget(row, i // self._columns, i % self._columns)
            # Shown NOW: added to a question already on screen, Qt would show
            # it on the next turn of the event loop, and a hidden row is left
            # out of the measure below (the question opened with no answers).
            # It has its parent, so it is never a window (4.26).
            row.show()
        for c in range(self._columns):
            self._rows.setColumnStretch(c, 1)
        self.place(anchor)
        self.raise_()
        if not self.isVisible():
            self.show()
            QApplication.instance().installEventFilter(self)
        # The focus on the path's own next step, so Return takes it and the
        # focus outline never marks another answer as the one to pick.
        rows = self._rows_widget.findChildren(_ChoiceRow)
        pick = next((r for r in rows if r.property("recommended")), rows[0] if rows else None)
        if pick is not None:
            pick.setFocus(Qt.FocusReason.OtherFocusReason)

    def place(self, anchor: QRect) -> None:
        """Centred over the views, a little below their top; the answers
        scroll when the views are too low for them."""
        margin = fonts.px(16)
        wanted = fonts.px(640) if self._columns == 2 else fonts.px(420)
        width = max(fonts.px(260), min(wanted, anchor.width() - 2 * margin))
        self.setFixedWidth(width)
        lay = self.layout()
        inner = width - lay.contentsMargins().left() - lay.contentsMargins().right()
        rows = self._rows_widget.layout()
        rows.activate()
        rows_h = rows.totalHeightForWidth(inner) if rows.hasHeightForWidth() \
            else rows.totalSizeHint().height()
        # Everything but the answers, measured with the answers' area empty.
        self._scroll.setFixedHeight(0)
        lay.activate()
        chrome = lay.totalHeightForWidth(width) if lay.hasHeightForWidth() \
            else lay.totalSizeHint().height()
        room = max(fonts.px(120), anchor.height() - 2 * margin - chrome)
        self._scroll.setFixedHeight(min(rows_h, room))
        height = chrome + min(rows_h, room)
        x = anchor.left() + (anchor.width() - width) // 2
        y = anchor.top() + max(margin, int((anchor.height() - height) * 0.2))
        self.setGeometry(x, y, width, height)

    def close_question(self) -> None:
        if self.isVisible():
            QApplication.instance().removeEventFilter(self)
            self.hide()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() == Qt.Key.Key_Escape:
            self.dismissed.emit()
            return
        super().keyPressEvent(event)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt naming
        # Decided by WHERE the press is, not by which widget sees it: an
        # ignored press travels up through the window's widgets, and each of
        # them is "elsewhere".
        if event.type() == QEvent.Type.MouseButtonPress and self.isVisible() \
                and isinstance(obj, QWidget) and not self._closing:
            at = self.mapFromGlobal(event.globalPosition().toPoint())
            if not self.rect().contains(at) and not self._on_guide(event):
                self._closing = True          # one press can be seen several times
                QTimer.singleShot(0, self._dismiss)
        return False

    def _dismiss(self) -> None:
        self._closing = False
        if self.isVisible():
            self.dismissed.emit()

    def _on_guide(self, event) -> bool:
        """A press on another piece of the guide (its corner button)."""
        widget = QApplication.widgetAt(event.globalPosition().toPoint())
        while widget is not None:
            if _is_guide(widget):
                return True
            widget = widget.parentWidget()
        return False


class GuideRest(QPushButton):
    """The small Guide button left in the views' corner when the user says
    Not now: one click and the question is back."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__("Guide: what next?", parent)
        self.setObjectName("guide-rest")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Show what the guide can walk you through here")
        self.hide()

    def place(self, anchor: QRect) -> None:
        self.adjustSize()
        margin = fonts.px(14)
        self.move(anchor.right() - self.width() - margin,
                  anchor.bottom() - self.height() - margin)


# ---------------------------------------------------------------------------
# The controller
# ---------------------------------------------------------------------------


@dataclass
class _Run:
    """A task in progress in one view: which, which step, and the facts when
    that step was shown (what moves it on is what changed since)."""

    key: str
    index: int
    then: dict = field(default_factory=dict)


class Guide(QObject):
    """Reads the window, asks, walks the chosen task, places the pieces.

    ``window`` provides ``guide_view()``, ``guide_facts()``, ``guide_target(
    name)`` and a ``stack`` (the views' area), and its ``converter`` emits
    ``scan_finished`` and ``convert_finished``.
    """

    def __init__(self, window) -> None:
        super().__init__(window)
        self._window = window
        self._host = window.centralWidget()
        self._card = GuideCard(self._host)
        self._overlay = GuideOverlay(self._host)
        self._question = GuideQuestion(self._host)
        self._rest = GuideRest(self._host)
        self._card.next.connect(self._on_next)
        self._card.previous.connect(self._on_previous)
        self._card.all_tasks.connect(self._ask_again)
        self._question.chosen.connect(self._on_chosen)
        self._question.dismissed.connect(self._on_dismissed)
        self._rest.clicked.connect(self._ask_again)
        self._host.installEventFilter(self)
        self._on = False
        self._converted_root: Optional[str] = None
        self._conversions = 0
        self._subjects: dict[str, bool] = {}
        #: The view the guide last saw; coming to another one asks.
        self._view: Optional[str] = None
        #: Per view, the task in progress.
        self._runs: dict[str, _Run] = {}
        #: Views where the user said Not now (until they come back).
        self._resting: set[str] = set()
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self.refresh)
        converter = window.converter
        converter.convert_finished.connect(self._on_convert_finished)
        converter.scan_finished.connect(self._on_scan_finished)
        from .viz.bridge import connect_while_alive

        connect_while_alive(QApplication.instance().focusChanged, self,
                            lambda guide, *_: guide._soon())

    # -- what is on -----------------------------------------------------------

    @property
    def overlay(self) -> GuideOverlay:
        return self._overlay

    @property
    def card(self) -> GuideCard:
        return self._card

    @property
    def question(self) -> GuideQuestion:
        return self._question

    @property
    def rest(self) -> GuideRest:
        return self._rest

    @property
    def enabled(self) -> bool:
        return self._on

    def run(self, view: str) -> Optional[tuple[str, int]]:
        """``(task key, step index)`` in progress in ``view``, if any."""
        run = self._runs.get(view)
        return (run.key, run.index) if run is not None else None

    def set_enabled(self, on: bool) -> None:
        self._on = bool(on)
        self._view = None                 # switched on: the view is entered afresh
        self._runs.clear()
        self._resting.clear()
        if self._on:
            self._timer.start()
            self.refresh()
        else:
            self._timer.stop()
            self._hide_all()

    def repaint_for_palette(self) -> None:
        """The theme or the font size changed: measure and place again."""
        if self._question.isVisible():
            self._question.place(self._anchor())
        self._overlay.clear()                       # placed again on the next refresh
        self._soon()

    def _hide_all(self) -> None:
        self._question.close_question()
        self._card.hide()
        self._overlay.clear()
        self._rest.hide()

    def _anchor(self) -> QRect:
        stack = self._window.stack
        return QRect(stack.mapTo(self._host, QPoint(0, 0)), stack.size())

    # -- the question ---------------------------------------------------------

    def ask(self, title: str = "What would you like to do?") -> None:
        """Ask what to do in the current view; a task in progress there is
        offered first, to continue."""
        if not self._on:
            return
        view = self._window.guide_view()
        facts = self._facts(view)
        offered = tasks(view, facts)
        answers: list[Answer] = []
        run = self._runs.get(view)
        task = find_task(view, run.key, facts) if run is not None else None
        if task is not None:
            step = shown(task, min(run.index, len(task.steps) - 1), facts)
            answers.append(Answer(CONTINUE, f"Continue: {task.label}",
                                  (f"{step.counter}: " if step.counter else "") + step.title,
                                  True))
        answers += [Answer(t.key, t.label, t.detail, t.recommended and task is None)
                    for t in offered]
        self._resting.discard(view)
        self._card.hide()
        self._overlay.clear()
        self._rest.hide()
        self._question.ask(answers, self._anchor(), title)

    def _ask_again(self, *_args) -> None:
        """All tasks, or the corner button (``clicked`` carries a bool)."""
        self.ask()

    def _on_chosen(self, key: str) -> None:
        view = self._window.guide_view()
        self._question.close_question()
        if key != CONTINUE:
            self._runs[view] = _Run(key, 0, self._facts(view))
        self.refresh()

    def _on_dismissed(self) -> None:
        self._question.close_question()
        self._resting.add(self._window.guide_view())
        self.refresh()

    # -- moving through a task --------------------------------------------------

    def _task(self, view: str, facts: dict) -> Optional[Task]:
        run = self._runs.get(view)
        return find_task(view, run.key, facts) if run is not None else None

    def _on_next(self) -> None:
        view = self._window.guide_view()
        facts = self._facts(view)
        task, run = self._task(view, facts), self._runs.get(view)
        if task is None or run is None:
            return
        if run.index + 1 >= len(task.steps):
            self._finish(view)
            return
        self._runs[view] = _Run(run.key, run.index + 1, facts)
        self.refresh()

    def _on_previous(self) -> None:
        view = self._window.guide_view()
        run = self._runs.get(view)
        if run is None or run.index == 0:
            return
        self._runs[view] = _Run(run.key, run.index - 1, self._facts(view))
        self.refresh()

    def _finish(self, view: str) -> None:
        self._runs.pop(view, None)
        if view == self._window.guide_view():
            self.ask("Done. What would you like to do next?")

    def _move_on(self, view: str, facts: dict, *, announce: bool = True) -> None:
        """A step whose thing happened moves on by itself; the last one
        finishes its task (and, with ``announce``, asks what next)."""
        run = self._runs.get(view)
        task = self._task(view, facts)
        if run is None:
            return
        if task is None:
            self._runs.pop(view, None)          # no longer offered here
            return
        index = min(run.index, len(task.steps) - 1)
        while should_advance(task, index, run.then, facts):
            if index + 1 >= len(task.steps):
                self._runs.pop(view, None)
                if announce and view == self._window.guide_view() and self._view == view:
                    self.ask("Done. What would you like to do next?")
                return
            index += 1
            run = _Run(run.key, index, facts)
            self._runs[view] = run

    # -- what the guide tracks itself ----------------------------------------

    def _on_convert_finished(self, rc: int, _parent) -> None:
        self._conversions += 1
        if int(rc) == 0:
            root = self._window.converter.guide_facts().get("_root")
            self._converted_root = str(root) if root is not None else None
        self._subjects.clear()
        self._soon()

    def _on_scan_finished(self, *_args) -> None:
        self._converted_root = None
        self._soon()

    def _has_subjects(self, root) -> bool:
        key = str(root)
        if key not in self._subjects:
            try:
                from pathlib import Path

                self._subjects[key] = any(
                    p.name.startswith("sub-") and p.is_dir() for p in Path(root).iterdir())
            except OSError:
                self._subjects[key] = False
        return self._subjects[key]

    def _facts(self, view: str) -> dict:
        facts = dict(self._window.guide_facts())
        root = facts.pop("_root", None)
        if view == "converter":
            facts["converted"] = root is not None and str(root) == self._converted_root
            facts["conversions"] = self._conversions
        if view in ("converter", "editor") and root is not None:
            facts["subjects"] = self._has_subjects(root)
        return facts

    # -- placing ---------------------------------------------------------------

    def _soon(self) -> None:
        if self._on:
            QTimer.singleShot(0, self.refresh)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt naming
        if obj is self._host and event.type() == QEvent.Type.Resize and self._on:
            self._overlay.setGeometry(self._host.rect())
            if self._question.isVisible():
                self._question.place(self._anchor())
            self._soon()
        return False

    def _rect_in_host(self, widget: Optional[QWidget]) -> Optional[QRect]:
        """Where ``widget`` shows in the host, clipped by every parent (a
        scrolled list shows part of what it holds); ``None`` when it is not
        on screen in this window.

        Not ``visibleRegion``: that leaves out what an opaque child paints
        (a scroll area's viewport), and ringed only the top of a panel."""
        host = self._host
        if widget is None or not widget.isVisible() or widget.window() is not self._window:
            return None
        rect = QRect(widget.mapTo(host, QPoint(0, 0)), widget.size())
        parent = widget.parentWidget()
        while parent is not None and parent is not host:
            rect = rect.intersected(QRect(parent.mapTo(host, QPoint(0, 0)), parent.size()))
            parent = parent.parentWidget()
        rect = rect.intersected(host.rect())
        return None if rect.isEmpty() else rect

    def refresh(self) -> None:
        if not self._on:
            return
        window = self._window
        if not window.isVisible() or window.isMinimized() \
                or QApplication.activeModalWidget() is not None:
            self._card.hide()
            self._overlay.clear()
            return
        view = window.guide_view()
        facts = self._facts(view)
        if view != self._view:
            # Come to a view: a task in progress here may have been finished
            # meanwhile; then ask what to do (one thing only: just start it).
            self._view = view
            self._resting.discard(view)
            self._move_on(view, facts, announce=False)
            offered = tasks(view, facts)
            if len(offered) >= 2 or (offered and view in self._runs):
                self.ask()
            elif offered:
                self._question.close_question()
                self._runs[view] = _Run(offered[0].key, 0, facts)
            else:
                self._question.close_question()
        if self._question.isVisible():
            return
        self._move_on(view, facts)
        if self._question.isVisible():
            return
        task = self._task(view, facts)
        if task is None or view in self._resting:
            self._card.hide()
            self._overlay.clear()
            if tasks(view, facts):
                self._rest.place(self._anchor())
                self._on_top(self._rest)
                self._rest.show()
            else:
                self._rest.hide()
            return
        self._rest.hide()
        run = self._runs[view]
        index = min(run.index, len(task.steps) - 1)
        step = shown(task, index, facts)
        self._card.set_step(step, first=index == 0, last=index == len(task.steps) - 1)
        if self._overlay.geometry() != self._host.rect():
            self._overlay.setGeometry(self._host.rect())
        target = self._rect_in_host(window.guide_target(step.target))
        clear = [r for r in (self._rect_in_host(window.guide_target(name))
                             for name in step.keep_clear) if r is not None]
        size = self._card.size_for(fonts.px(320))
        where = self._overlay.place(step, target, self._anchor(), tuple(clear), size)
        if self._card.geometry() != where:
            self._card.setGeometry(where)
        self._on_top(self._card, self._overlay)   # the arrow over the card's edge
        if not self._card.isVisible():
            self._card.show()

    def _on_top(self, *widgets: QWidget) -> None:
        """Stack ``widgets`` last among the host's children, in this order,
        ONLY when they are not already: a raise that reorders repaints the
        widget's whole rectangle, and the overlay's is the whole window
        (measured: raising it on every check put 150 to 220 ms pauses into a
        conversion)."""
        kids = [w for w in self._host.children() if isinstance(w, QWidget)]
        if kids[-len(widgets):] != list(widgets):
            for widget in widgets:
                widget.raise_()


__all__ = ["CONTINUE", "Answer", "Guide", "GuideCard", "GuideOverlay", "GuideQuestion",
           "GuideRest", "GuideSwitch", "POLL_MS"]
