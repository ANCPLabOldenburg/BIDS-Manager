"""What's new, as short animations: one scene per new feature.

The What's new dialog (``gui/whats_new.py``) plays these one after the other
beside the list of what the release brings. Each scene is a function of time
painted on a 16 x 10 stage, in the theme's colours and the app's fonts, and
shows the feature with the app's own words and values: the guide's real steps,
the Editor's menu entries, the anonymiser's real output for a sidecar, the
columns a CMRR physiology log is written as. A scene never shows what the app
does not do.

Everything is painted (no widgets), so a frame costs a few dozen rectangles,
lines and words. Text goes through ``gui/viz/fonts`` and so follows the
font-size setting; a stage drawn smaller than its text was laid out for
shrinks the text with it rather than letting it overflow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional, Sequence

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)

#: The stage, in units: everything is placed on this grid.
W, H = 16.0, 10.0
#: Pixels per unit, at font scale 1.0, that the text sizes are chosen for.
UNIT_PX = 34

_LEFT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
_RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
_CENTRE = Qt.AlignmentFlag.AlignCenter


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def ease(s: float) -> float:
    """Ease in and out (cubic), clipped to 0..1."""
    s = min(1.0, max(0.0, s))
    return 4 * s ** 3 if s < 0.5 else 1 - (-2 * s + 2) ** 3 / 2


def phase(t: float, start: float, end: float) -> float:
    """How far ``t`` is from ``start`` to ``end``, eased, 0..1."""
    return ease((t - start) / max(1e-6, end - start))


def lerp(a: float, b: float, s: float) -> float:
    return a + (b - a) * s


def swap(s: float) -> tuple[float, float]:
    """How visible the old and the new text are while one replaces the
    other: the old fades out, then the new fades in, so the two never
    overlap."""
    return max(0.0, 1 - 2 * s), max(0.0, 2 * s - 1)


def _lerp_box(a: Sequence[float], b: Sequence[float], s: float) -> tuple:
    return tuple(lerp(x, y, s) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# The stage: units to pixels, and the few shapes every scene is made of
# ---------------------------------------------------------------------------


class Stage:
    """A painter on the 16 x 10 grid, with the theme's colours and fonts."""

    def __init__(self, painter: QPainter, rect: QRectF, palette: dict, theme) -> None:
        from .viz import fonts

        self.p = painter
        self.u = min(rect.width() / W, rect.height() / H)
        self.x0 = rect.x() + (rect.width() - W * self.u) / 2
        self.y0 = rect.y() + (rect.height() - H * self.u) / 2
        self.pal = palette
        self.theme = theme
        self.series = [theme.series(i) for i in range(6)]
        self.shrink = min(1.0, self.u / fonts.px(UNIT_PX))
        self._fonts: dict = {}

    # -- geometry ---------------------------------------------------------

    def rect(self, x: float, y: float, w: float, h: float) -> QRectF:
        return QRectF(self.x0 + x * self.u, self.y0 + y * self.u, w * self.u, h * self.u)

    def point(self, x: float, y: float) -> QPointF:
        return QPointF(self.x0 + x * self.u, self.y0 + y * self.u)

    def px(self, units: float) -> float:
        return units * self.u

    # -- colour and type --------------------------------------------------

    def colour(self, name: str, alpha: float = 1.0) -> QColor:
        """A palette token (``"text"``, ``"accent_bg"``), a plot series
        (``"s1"``..``"s6"``) or a literal colour, with ``alpha`` applied."""
        from .widgets.primitives import qcolor

        if name.startswith("s") and name[1:].isdigit():
            value = self.series[int(name[1:]) - 1]
        else:
            value = self.pal.get(name, name)
        c = qcolor(value)
        if not c.isValid():
            c = qcolor(self.pal.get("dim", "#888888"))
        c.setAlphaF(max(0.0, min(1.0, c.alphaF() * alpha)))
        return c

    def font(self, size: int, *, bold: bool = False, mono: bool = False,
             italic: bool = False) -> QFont:
        key = (size, bold, mono, italic, self.shrink)
        cached = self._fonts.get(key)
        if cached is None:
            from .viz import fonts

            cached = fonts.font(size, bold=bold, mono=mono)
            cached.setPixelSize(max(6, round(cached.pixelSize() * self.shrink)))
            cached.setItalic(italic)
            self._fonts[key] = cached
        return cached

    def width(self, text: str, size: int, *, bold: bool = False, mono: bool = False) -> float:
        """How wide ``text`` is, in units."""
        metrics = QFontMetricsF(self.font(size, bold=bold, mono=mono))
        return metrics.horizontalAdvance(text) / self.u

    # -- drawing ----------------------------------------------------------

    def text(self, x: float, y: float, w: float, h: float, text: str, colour: str = "text",
             size: int = 11, *, bold: bool = False, mono: bool = False, italic: bool = False,
             align=_LEFT, alpha: float = 1.0, wrap: bool = False, strike: bool = False,
             clip: bool = False) -> None:
        if alpha <= 0.002 or not text:
            return
        font = self.font(size, bold=bold, mono=mono, italic=italic)
        if strike:
            font = QFont(font)
            font.setStrikeOut(True)
        self.p.setFont(font)
        self.p.setPen(self.colour(colour, alpha))
        box = self.rect(x, y, w, h)
        if wrap:
            self.p.drawText(box, align.value | Qt.TextFlag.TextWordWrap.value, text)
            return
        if clip:            # a growing word is revealed, never cut with an ellipsis
            self.p.save()
            self.p.setClipRect(box)
            self.p.drawText(QRectF(box.x(), box.y(), box.width() + self.px(20), box.height()),
                            align, text)
            self.p.restore()
            return
        shown = QFontMetricsF(font).elidedText(text, Qt.TextElideMode.ElideRight, box.width())
        self.p.drawText(box, align, shown)

    def box(self, x: float, y: float, w: float, h: float, fill: Optional[str] = "surface",
            border: Optional[str] = "border", *, radius: float = 0.16, alpha: float = 1.0,
            line: float = 1.0, dash: bool = False) -> None:
        if alpha <= 0.002:
            return
        if border:
            pen = QPen(self.colour(border, alpha), line)
            if dash:
                pen.setStyle(Qt.PenStyle.DashLine)
            self.p.setPen(pen)
        else:
            self.p.setPen(Qt.PenStyle.NoPen)
        self.p.setBrush(self.colour(fill, alpha) if fill else Qt.BrushStyle.NoBrush)
        r = self.px(radius)
        self.p.drawRoundedRect(self.rect(x, y, w, h), r, r)
        self.p.setBrush(Qt.BrushStyle.NoBrush)

    def shadow(self, x: float, y: float, w: float, h: float, *, alpha: float = 1.0) -> None:
        """A soft shadow under a floating card (a question, a menu)."""
        for i, a in enumerate((0.10, 0.06, 0.035)):
            grow = 0.05 * (i + 1)
            self.box(x - grow, y - grow + 0.08, w + 2 * grow, h + 2 * grow, "#000000", None,
                     radius=0.2 + grow, alpha=a * alpha)

    def chip(self, x: float, y: float, text: str, fg: str, bg: str, *, size: int = 9,
             h: float = 0.46, alpha: float = 1.0, right: bool = False, bold: bool = True) -> float:
        """A rounded label; ``right`` puts its right edge at ``x``. Returns
        its width in units."""
        w = self.width(text, size, bold=bold) + 0.36
        left = x - w if right else x
        self.box(left, y, w, h, bg, None, radius=h / 2, alpha=alpha)
        self.text(left, y, w, h, text, fg, size, bold=bold, align=_CENTRE, alpha=alpha)
        return w

    def line(self, x1: float, y1: float, x2: float, y2: float, colour: str = "subtle", *,
             width: float = 1.0, alpha: float = 1.0) -> None:
        self.p.setPen(QPen(self.colour(colour, alpha), width, Qt.PenStyle.SolidLine,
                           Qt.PenCapStyle.RoundCap))
        self.p.drawLine(self.point(x1, y1), self.point(x2, y2))

    def trace(self, xs: np.ndarray, ys: np.ndarray, colour: str, *, width: float = 1.4,
              alpha: float = 1.0) -> None:
        if len(xs) < 2 or alpha <= 0.002:
            return
        path = QPainterPath(self.point(float(xs[0]), float(ys[0])))
        for x, y in zip(xs[1:].tolist(), ys[1:].tolist()):
            path.lineTo(self.x0 + x * self.u, self.y0 + y * self.u)
        self.p.setPen(QPen(self.colour(colour, alpha), width, Qt.PenStyle.SolidLine,
                           Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        self.p.drawPath(path)

    def dot(self, x: float, y: float, r: float, colour: str, *, alpha: float = 1.0) -> None:
        if alpha <= 0.002:
            return
        self.p.setPen(Qt.PenStyle.NoPen)
        self.p.setBrush(self.colour(colour, alpha))
        self.p.drawEllipse(self.point(x, y), self.px(r), self.px(r))
        self.p.setBrush(Qt.BrushStyle.NoBrush)

    def checkbox(self, x: float, y: float, on: float, *, alpha: float = 1.0,
                 size: float = 0.28) -> None:
        """A tick box at (x, y) (its top-left); ``on`` 0..1 fades the tick."""
        self.box(x, y, size, size, "surface2", "input_border", radius=0.06, alpha=alpha)
        if on > 0:
            self.box(x, y, size, size, "accent", None, radius=0.06, alpha=alpha * on)
            pen = QPen(self.colour("primary_btn_text", alpha * on), max(1.2, self.px(0.045)),
                       Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            self.p.setPen(pen)
            self.p.drawPolyline(QPolygonF([self.point(x + 0.27 * size, y + 0.52 * size),
                                           self.point(x + 0.45 * size, y + 0.70 * size),
                                           self.point(x + 0.76 * size, y + 0.32 * size)]))

    def ring(self, x: float, y: float, w: float, h: float, t: float, *, alpha: float = 1.0) -> None:
        """The guide's ring around a control: an accent outline and a halo
        that breathes."""
        pulse = 0.5 + 0.5 * math.sin(t * 4.2)
        grow = 0.08
        self.box(x - grow - 0.06 * pulse, y - grow - 0.06 * pulse, w + 2 * grow + 0.12 * pulse,
                 h + 2 * grow + 0.12 * pulse, None, "accent", radius=0.24,
                 alpha=alpha * (0.25 + 0.2 * (1 - pulse)), line=self.px(0.1))
        self.box(x - grow, y - grow, w + 2 * grow, h + 2 * grow, None, "accent", radius=0.2,
                 alpha=alpha, line=max(1.5, self.px(0.06)))

    def cursor(self, x: float, y: float, *, press: float = 0.0, alpha: float = 1.0) -> None:
        """A mouse pointer with its tip at (x, y); ``press`` shows a click."""
        if alpha <= 0.002:
            return
        if press > 0:
            self.dot(x, y, 0.12 + 0.32 * press, "accent", alpha=alpha * 0.35 * (1 - press))
        s = 0.5
        shape = [(0, 0), (0, 0.78), (0.2, 0.6), (0.33, 0.9), (0.45, 0.85), (0.32, 0.55),
                 (0.58, 0.55)]
        poly = QPolygonF([self.point(x + s * px_, y + s * py_) for px_, py_ in shape])
        self.p.setPen(QPen(self.colour("bg", alpha), max(1.0, self.px(0.035))))
        self.p.setBrush(self.colour("text", alpha))
        self.p.drawPolygon(poly)
        self.p.setBrush(Qt.BrushStyle.NoBrush)

    def keycap(self, x: float, y: float, key: str, *, press: float = 0.0,
               alpha: float = 1.0, size: int = 11) -> float:
        """A key with ``key`` written on it; returns its width in units."""
        w = max(0.62, self.width(key, size, bold=True) + 0.4)
        h = 0.6
        sink = 0.06 * press
        self.box(x, y + 0.07, w, h, "subtle", None, radius=0.12, alpha=alpha)
        self.box(x, y + sink, w, h, "surface2", "accent" if press > 0.05 else "input_border",
                 radius=0.12, alpha=alpha)
        self.text(x, y + sink, w, h, key, "text", size, bold=True, align=_CENTRE, alpha=alpha)
        return w

    def glyph(self, x: float, y: float, kind: str, colour: str, *, alpha: float = 1.0) -> None:
        """A folder or a file, 0.34 units wide, at (x, y) (its left, centred
        on ``y``)."""
        if kind == "folder":
            self.box(x, y - 0.13, 0.15, 0.08, colour, None, radius=0.03, alpha=alpha)
            self.box(x, y - 0.09, 0.34, 0.24, colour, None, radius=0.04, alpha=alpha)
        else:
            self.box(x + 0.05, y - 0.15, 0.24, 0.3, None, colour, radius=0.04, alpha=alpha,
                     line=max(1.0, self.px(0.035)))

    def tree_row(self, x: float, y: float, depth: int, name: str, kind: str, *, width: float,
                 alpha: float = 1.0, colour: Optional[str] = None, italic: bool = False,
                 size: int = 10, indent: float = 0.4, glyph: bool = True) -> None:
        """One row of a file tree, the way the Editor colours it (``glyph``
        False: the name alone, for a name that changes in place)."""
        tone = colour or {"folder": "text", "image": "tree_image", "sidecar": "tree_sidecar",
                          "table": "tree_table", "recording": "tree_recording"}.get(kind, "tree_other")
        left = x + depth * indent
        if glyph:
            self.glyph(left, y + 0.31, "folder" if kind == "folder" else "file",
                       "tree_folder" if kind == "folder" else tone, alpha=alpha)
        self.text(left + 0.46, y, width - (left - x) - 0.46, 0.62, name, tone, size,
                  italic=italic, alpha=alpha)

    def panel(self, x: float, y: float, w: float, h: float, title: str = "", *,
              alpha: float = 1.0, fill: str = "surface") -> None:
        """A card with a small title, as the app's panels are."""
        self.box(x, y, w, h, fill, "border", alpha=alpha)
        if title:
            self.text(x + 0.25, y + 0.1, w - 0.5, 0.5, title, "dim", 10, bold=True, alpha=alpha)

    def menu(self, x: float, y: float, items: Sequence[str], hover: int, *, alpha: float = 1.0,
             width: float = 3.6, notes: Sequence[str] = ()) -> None:
        """A right-click menu, ``hover`` the highlighted entry (-1: none)."""
        if alpha <= 0.002:
            return
        h = 0.2 + 0.52 * len(items)
        self.shadow(x, y, width, h, alpha=alpha)
        self.box(x, y, width, h, "surface", "border", radius=0.14, alpha=alpha)
        for i, item in enumerate(items):
            row_y = y + 0.1 + 0.52 * i
            if i == hover:
                self.box(x + 0.08, row_y, width - 0.16, 0.52, "accent_bg", None, radius=0.1,
                         alpha=alpha)
            self.text(x + 0.25, row_y, width - 0.5, 0.52, item, "text", 10, alpha=alpha)
            if i < len(notes) and notes[i]:
                self.text(x + 0.25, row_y, width - 0.5, 0.52, notes[i], "dim", 9, align=_RIGHT,
                          alpha=alpha)

    def json_line(self, x: float, y: float, key: str, value: str, *, size: int = 10,
                  alpha: float = 1.0, value_colour: str = "text", strike: bool = False,
                  width: float = 8.0) -> None:
        """``"key": value`` in the code font, key dimmer than its value."""
        head = f'"{key}": '
        self.text(x, y, width, 0.6, head, "dim", size, mono=True, alpha=alpha, strike=strike)
        dx = self.width(head, size, mono=True)
        self.text(x + dx, y, width - dx, 0.6, value, value_colour, size, mono=True,
                  alpha=alpha, strike=strike)


# ---------------------------------------------------------------------------
# The scenes
# ---------------------------------------------------------------------------


def _guide(s: Stage, t: float) -> None:
    """The guide: it asks, the user chooses, and it rings one control after
    another with a card saying what to do there."""
    s.box(0.4, 0.4, 15.2, 9.2, "bg", "border", radius=0.24)
    # The header: views, then the Guide switch and the settings.
    s.line(0.4, 1.2, 15.6, 1.2, "subtle")
    s.dot(0.95, 0.8, 0.17, "accent")
    s.text(1.25, 0.5, 3, 0.6, "BIDS Manager", "text", 10, bold=True)
    views = ("Home", "Converter", "Editor")
    widths = [s.width(v, 10) + 0.5 for v in views]
    left = 8.0 - sum(widths) / 2
    s.box(left - 0.06, 0.53, sum(widths) + 0.12, 0.54, "surface2", None, radius=0.27)
    for name, w in zip(views, widths):
        if name == "Converter":
            s.box(left, 0.58, w, 0.44, "surface3", None, radius=0.22)
        s.text(left, 0.58, w, 0.44, name, "text" if name == "Converter" else "dim", 10,
               align=_CENTRE)
        left += w
    s.text(11.9, 0.5, 1.3, 0.6, "Guide", "text", 10, align=_RIGHT)
    s.box(13.32, 0.62, 0.64, 0.36, "accent", None, radius=0.18)
    s.dot(13.78, 0.8, 0.13, "primary_btn_text")
    s.dot(14.75, 0.8, 0.16, "surface3")
    s.dot(14.75, 0.8, 0.06, "dim")

    # Structure, the table, Properties, Run.
    left_out = phase(t, 8.85, 9.1)
    s.panel(0.7, 1.5, 3.6, 7.8, "Structure")
    tree = (("sub-001", 0), ("anat", 1), ("func", 1), ("fmap", 1), ("sub-002", 0),
            ("anat", 1), ("func", 1))
    for i, (name, depth) in enumerate(tree):
        y = 2.15 + 0.6 * i
        on = 1.0 - left_out if i == 3 else 1.0
        s.checkbox(0.95 + 0.38 * depth, y + 0.16, on)
        s.text(1.35 + 0.38 * depth, y, 2.6, 0.6, name, "text" if on > 0.5 else "dim", 10)

    s.panel(4.5, 1.5, 7.0, 4.55, "Inspection")
    for x, name in ((5.15, "Subject"), (6.45, "Datatype"), (7.65, "BIDS name")):
        s.text(x, 2.05, 2.5, 0.45, name, "dim", 9, bold=True)
    s.line(4.7, 2.55, 11.3, 2.55, "subtle")
    rows = (("sub-001", "anat", "sub-001_T1w"), ("sub-001", "func", "sub-001_task-rest_bold"),
            ("sub-001", "fmap", "sub-001_phasediff"), ("sub-002", "anat", "sub-002_T1w"),
            ("sub-002", "func", "sub-002_task-rest_bold"))
    chosen = phase(t, 4.35, 4.5)
    for i, row in enumerate(rows):
        y = 2.65 + 0.6 * i
        if i == 1 and chosen > 0:
            s.box(4.65, y + 0.03, 6.7, 0.54, "accent_bg", None, radius=0.1, alpha=chosen)
        out = left_out if i == 2 else 0.0
        s.checkbox(4.78, y + 0.16, 1.0 - out)
        tone = "dim" if out > 0.5 else "text"
        for x, w, value in zip((5.15, 6.45, 7.65), (1.25, 1.15, 3.7), row):
            s.text(x, y, w, 0.6, value, tone, 10)
    s.panel(4.5, 6.3, 7.0, 3.0, "Log")
    for i, line in enumerate(("Scanning /data/raw_study...", "5 recordings in 2 subjects",
                              "Scan saved as version 1")):
        s.text(4.75, 6.95 + 0.5 * i, 6.5, 0.5, line, "muted", 9, mono=True)

    s.panel(11.8, 1.5, 3.5, 7.8, "Properties")
    details = phase(t, 4.4, 4.9)
    s.text(11.8, 4.6, 3.5, 0.6, "Click a row", "muted", 10, align=_CENTRE, alpha=1 - details)
    for i, (label, value) in enumerate((("Datatype", "func"), ("Suffix", "bold"),
                                        ("task", "rest"))):
        y = 2.2 + 0.72 * i
        s.text(12.05, y, 1.3, 0.55, label, "dim", 9, alpha=details)
        s.box(13.3, y + 0.04, 1.8, 0.48, "surface2", "input_border", radius=0.1, alpha=details)
        s.text(13.45, y + 0.04, 1.6, 0.48, value, "text", 10, alpha=details)
    s.text(12.05, 4.45, 3.0, 0.5, "BIDS name", "dim", 9, alpha=details)
    s.text(12.05, 4.85, 3.1, 0.55, "sub-001_task-rest_bold", "text", 9, alpha=details)
    s.chip(12.05, 5.5, "Valid BIDS", "success", "success_bg", alpha=details)

    # The question, then the three steps of the task chosen.
    q_in, q_out = phase(t, 0.35, 0.8), phase(t, 2.55, 2.95)
    q = q_in * (1 - q_out)
    if q > 0:
        lift = 0.25 * (1 - q_in)
        x, y, w, h = 4.15, 2.25 + lift, 7.7, 5.3
        s.shadow(x, y, w, h, alpha=q)
        s.box(x, y, w, h, "surface", "border", radius=0.22, alpha=q)
        s.text(x + 0.35, y + 0.2, w - 0.7, 0.7, "What would you like to do?", "text", 13,
               bold=True, alpha=q)
        answers = (("Scan new data", "Another raw data folder, scanned into a new table."),
                   ("Check and edit the table",
                    "Correct names, leave out what you do not want."),
                   ("Convert the table", "Write the ticked recordings into the dataset."))
        picked = phase(t, 2.1, 2.25)
        for i, (title, detail) in enumerate(answers):
            ry = y + 1.0 + 1.22 * i
            fill = "surface2"
            s.box(x + 0.3, ry, w - 0.6, 1.08, fill, "accent_border" if i == 1 else "border",
                  radius=0.14, alpha=q)
            if i == 1 and picked > 0:
                s.box(x + 0.3, ry, w - 0.6, 1.08, "accent_bg", None, radius=0.14, alpha=q * picked)
            s.text(x + 0.55, ry + 0.1, w - 1.4, 0.5, title, "text", 11, bold=True, alpha=q)
            s.text(x + 0.55, ry + 0.52, w - 1.1, 0.45, detail, "dim", 9, alpha=q)
            if i == 1:
                s.dot(x + w - 0.62, ry + 0.35, 0.08, "accent", alpha=q)
        s.text(x + 0.3, y + h - 0.6, w - 0.6, 0.45, "Not now", "dim", 9, align=_RIGHT, alpha=q)
        # The pointer comes to the second answer and presses it.
        go = phase(t, 1.0, 2.0)
        cx, cy = lerp(13.4, 8.4, go), lerp(8.6, y + 1.0 + 1.22 + 0.55, go)
        press = max(0.0, min(1.0, (t - 2.1) / 0.35)) if t > 2.1 else 0.0
        s.cursor(cx, cy, press=press, alpha=q_in * (1 - q_out))

    # Steps: (ring box, card box, arrow side, kicker, title, body). The card
    # sits over the log, the panel it hides least of, as the guide places it.
    steps = (
        ((4.5, 1.5, 7.0, 4.55), (4.75, 6.5, 6.5, 2.65), "up", "STEP 1 OF 4",
         "Look through the table", "Untick a row to leave it out. Click a row to see it in "
         "Properties."),
        ((11.8, 1.5, 3.5, 7.8), (4.75, 6.5, 6.5, 2.65), "right", "STEP 2 OF 4",
         "Its details in Properties", "Its datatype, suffix and entities, checked against "
         "BIDS as you type."),
        ((0.7, 1.5, 3.6, 7.8), (4.75, 6.5, 6.5, 2.65), "left", "STEP 3 OF 4",
         "Leave out whole groups", "Untick a subject or a kind of recording to leave all of "
         "it out at once."),
    )
    starts = (3.0, 5.4, 7.3)
    shown = phase(t, 2.95, 3.35)
    if shown > 0:
        k = sum(1 for st in starts if t >= st) - 1
        move = phase(t, starts[k], starts[k] + 0.6) if k > 0 else 1.0
        prev = steps[max(0, k - 1)]
        cur = steps[k]
        ring = _lerp_box(prev[0], cur[0], move)
        card = _lerp_box(prev[1], cur[1], move)
        s.ring(*ring, t, alpha=shown)
        x, y, w, h = card
        s.shadow(x, y, w, h, alpha=shown)
        s.box(x, y, w, h, "surface", "accent_border", radius=0.18, alpha=shown)
        # The arrow from the card to the ring, on the side it faces.
        side = cur[2]
        if side == "up":
            ax = x + w / 2
            poly = QPolygonF([s.point(ax - 0.22, y), s.point(ax, y - 0.3),
                              s.point(ax + 0.22, y)])
        else:
            ay = y + 0.9
            base = x + w if side == "right" else x
            tip = base + (0.3 if side == "right" else -0.3)
            poly = QPolygonF([s.point(base, ay - 0.22), s.point(tip, ay),
                              s.point(base, ay + 0.22)])
        s.p.setPen(Qt.PenStyle.NoPen)
        s.p.setBrush(s.colour("accent", shown))
        s.p.drawPolygon(poly)
        s.p.setBrush(Qt.BrushStyle.NoBrush)
        for j, step in enumerate(steps):
            a = shown * ((1 - move) if j == k - 1 else move if j == k else 0.0)
            if a <= 0:
                continue
            s.text(x + 0.25, y + 0.15, w - 0.5, 0.4, step[3], "dim", 8, bold=True, alpha=a)
            s.text(x + 0.25, y + 0.5, w - 0.5, 0.5, step[4], "text", 11, bold=True, alpha=a)
            s.text(x + 0.25, y + 1.02, w - 0.5, 1.0, step[5], "dim", 9, wrap=True,
                   align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, alpha=a)
        s.chip(x + 0.25, y + h - 0.62, "‹ Previous", "dim", "surface3", size=9, bold=False,
               alpha=shown)
        s.chip(x + w - 0.25, y + h - 0.62, "Next ›", "text", "surface3", size=9, alpha=shown,
               right=True)
        # The pointer clicks a row in step 1, unticks fmap in step 3.
        if 3.6 < t < 5.0:
            go = phase(t, 3.6, 4.3)
            s.cursor(lerp(6.0, 8.3, go), lerp(8.2, 3.5, go),
                     press=max(0.0, min(1.0, (t - 4.35) / 0.35)) if t > 4.35 else 0.0,
                     alpha=min(phase(t, 3.6, 3.8), 1 - phase(t, 4.75, 5.0)))
        if 8.0 < t < 9.9:
            go = phase(t, 8.0, 8.8)
            s.cursor(lerp(3.2, 1.48, go), lerp(8.2, 4.12, go),
                     press=max(0.0, min(1.0, (t - 8.85) / 0.35)) if t > 8.85 else 0.0,
                     alpha=min(phase(t, 8.0, 8.2), 1 - phase(t, 9.6, 9.9)))


def _ecg(x: np.ndarray, gain: float, shift: float) -> np.ndarray:
    ph = np.mod(x / 1.15 + shift, 1.0)
    wave = (0.12 * np.exp(-((ph - 0.18) / 0.035) ** 2) - 0.15 * np.exp(-((ph - 0.29) / 0.012) ** 2)
            + 1.0 * np.exp(-((ph - 0.31) / 0.013) ** 2) - 0.25 * np.exp(-((ph - 0.335) / 0.012) ** 2)
            + 0.28 * np.exp(-((ph - 0.56) / 0.06) ** 2))
    return gain * wave


def _physiology(s: Stage, t: float) -> None:
    """A Siemens CMRR log: four ECG leads, pulse, breathing and the trigger,
    then the three ways of writing it as BIDS tables."""
    s.panel(0.5, 0.6, 9.4, 8.8)
    lanes = (("ECG 1", "s1"), ("ECG 2", "s1"), ("ECG 3", "s1"), ("ECG 4", "s1"),
             ("Pulse", "s2"), ("Respiration", "s3"), ("Trigger", "s4"))
    x0, x1, top, lane_h = 2.75, 9.6, 0.95, 1.18
    reveal = lerp(x0, x1, phase(t, 0.15, 2.8))
    tr = 0.95
    volumes = np.arange(x0 + 0.35, x1, tr)
    for vx in volumes:
        if vx <= reveal:
            s.line(float(vx), top + 0.1, float(vx), top + lane_h * 7 - 0.1, "subtle", alpha=0.8)
    xs = np.linspace(x0, max(x0 + 0.01, reveal), max(2, int((reveal - x0) * 26)))
    gains = (1.0, 0.72, -0.5, 0.88)
    for i, (name, colour) in enumerate(lanes):
        cy = top + lane_h * (i + 0.5)
        s.text(0.75, cy - 0.3, 1.95, 0.6, name, "dim", 10)
        amp = 0.42 * lane_h
        if i < 4:
            ys = cy + 0.2 * amp - amp * _ecg(xs, gains[i], 0.02 * i)
        elif i == 4:
            ph = np.mod(xs / 1.15 - 0.22, 1.0)
            ys = cy + 0.4 * amp - amp * (0.85 * np.exp(-((ph - 0.5) / 0.12) ** 2)
                                         + 0.3 * np.exp(-((ph - 0.74) / 0.08) ** 2))
        elif i == 5:
            ys = cy - 0.75 * amp * np.sin(2 * np.pi * (xs - x0) / 4.6)
        else:
            pulse = np.zeros_like(xs)
            for vx in volumes:
                pulse = np.where((xs >= vx) & (xs < vx + 0.09), 1.0, pulse)
            ys = cy + 0.45 * amp - 0.95 * amp * pulse
        s.trace(xs, ys, colour, width=max(1.2, s.px(0.045)))
    if reveal < x1 - 0.01:
        s.line(reveal, top, reveal, top + lane_h * 7, "dim", alpha=0.6)

    # How it is written.
    s.panel(10.2, 0.6, 5.3, 8.8, "Written as")
    options = ("A table per sampling rate", "One table, interpolated", "One table, as recorded")
    which = 0 if t < 3.6 else 1 if t < 5.9 else 2
    swap = phase(t, (0, 3.6, 5.9)[which], (0, 3.6, 5.9)[which] + 0.35) if which else 1.0
    for i, label in enumerate(options):
        y = 1.25 + 0.6 * i
        s.dot(10.6, y + 0.3, 0.13, "surface2")
        s.p.setPen(QPen(s.colour("input_border"), 1.0))
        s.p.drawEllipse(s.point(10.6, y + 0.3), s.px(0.13), s.px(0.13))
        if i == which:
            s.dot(10.6, y + 0.3, 0.07, "accent", alpha=swap)
        s.text(10.9, y, 4.5, 0.6, label, "text" if i == which else "dim", 10)
    s.line(10.45, 3.2, 15.25, 3.2, "subtle")
    a = swap if which else phase(t, 0.3, 0.8)
    if which == 0:
        tables = (("recording-ecg · 400 Hz", "ecg1 ecg2 ecg3 ecg4 trigger"),
                  ("recording-cardiac · 200 Hz", "cardiac trigger"),
                  ("recording-respiratory · 50 Hz", "respiratory trigger"))
        for i, (head, cols) in enumerate(tables):
            y = 3.45 + 1.9 * i
            s.box(10.45, y, 4.8, 1.65, "surface2", "border", radius=0.12, alpha=a)
            s.text(10.65, y + 0.15, 4.4, 0.55, head, "text", 10, bold=True, alpha=a)
            s.text(10.65, y + 0.75, 4.4, 0.55, cols, "dim", 9, mono=True, alpha=a)
        return
    s.text(10.45, 3.35, 4.8, 0.55, "One table · 400 Hz", "text", 10, bold=True, alpha=a)
    cols = ("ecg1", "cardiac", "respiratory", "trigger")
    xs_col = (10.5, 11.55, 12.75, 14.35)
    for x, name in zip(xs_col, cols):
        s.text(x, 3.95, 1.6, 0.45, name, "dim", 8, mono=True, bold=True, alpha=a)
    s.line(10.45, 4.45, 15.25, 4.45, "subtle", alpha=a)
    ecg = ("1702", "1711", "1735", "1806", "1990", "2302", "2417", "2240")
    if which == 1:
        cardiac = ("2210", "2212", "2214", "2216.5", "2219", "2221", "2223.5", "2226")
        resp = ("2048", "2048.6", "2049.1", "2049.7", "2050.2", "2050.8", "2051.3", "2051.9")
    else:
        cardiac = ("2210", "n/a", "2214", "n/a", "2219", "n/a", "2223.5", "n/a")
        resp = ("2048", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a")
    for r in range(8):
        y = 4.55 + 0.54 * r
        for x, value in zip(xs_col, (ecg[r], cardiac[r], resp[r], "1" if r == 0 else "0")):
            s.text(x, y, 1.6, 0.5, value, "muted" if value == "n/a" else "text", 9, mono=True,
                   alpha=a)


def _dummy_volumes(s: Stage, t: float) -> None:
    """The first three volumes of a run left out: the image, its events and
    its physiology move together, and the sidecar says so."""
    s.text(0.6, 0.25, 9.0, 0.6, "func/sub-01_task-rest_bold.nii.gz", "tree_image", 11)
    s.chip(15.4, 0.32, "TR 2 s", "dim", "surface3", right=True)
    for name, y in (("Volumes", 1.65), ("Events", 3.75), ("Physiology", 5.25)):
        s.text(0.6, y, 2.2, 1.2, name, "dim", 10)
    slot, x_first, n = 1.25, 2.95, 13
    shift = 3 * slot * phase(t, 3.0, 4.6)
    clip = s.rect(2.85, 0.9, 12.55, 6.0)
    s.p.save()
    s.p.setClipRect(clip)
    gone = phase(t, 1.9, 2.9)
    marked = phase(t, 0.8, 1.4)
    for i in range(n):
        x = x_first + i * slot - shift
        a, lift = 1.0, 0.0
        if i < 3:
            a, lift = 1.0 - gone, 0.5 * gone
            if a <= 0:
                continue
        bright = (0.95, 0.78, 0.62)[i] if i < 3 else 0.48
        s.box(x, 1.65 - lift, 1.05, 1.2, "surface2", "warning" if i < 3 and marked > 0 else "border",
              radius=0.12, alpha=a, line=max(1.0, s.px(0.04)) if i < 3 else 1.0)
        tone = QColor(s.colour("surface2"))
        ink = QColor(s.colour("text"))
        mix = QColor.fromRgbF(*(lerp(c1, c2, bright) for c1, c2 in
                                zip(tone.getRgbF()[:3], ink.getRgbF()[:3])))
        # An axial slice: the head, the gap between the hemispheres, the
        # ventricles. The first volumes are brighter: the signal has not
        # settled yet, which is why they are left out.
        s.p.setPen(Qt.PenStyle.NoPen)
        mix.setAlphaF(a)
        s.p.setBrush(mix)
        s.p.drawEllipse(s.rect(x + 0.2, 1.83 - lift, 0.65, 0.84))
        s.p.setBrush(s.colour("surface2", a * 0.85))
        s.p.drawEllipse(s.rect(x + 0.42, 2.12 - lift, 0.09, 0.26))
        s.p.drawEllipse(s.rect(x + 0.54, 2.12 - lift, 0.09, 0.26))
        s.p.setBrush(Qt.BrushStyle.NoBrush)
        s.line(x + 0.525, 1.88 - lift, x + 0.525, 2.06 - lift, "surface2", alpha=a,
               width=max(1.0, s.px(0.03)))
        s.line(x + 0.525, 2.44 - lift, x + 0.525, 2.62 - lift, "surface2", alpha=a,
               width=max(1.0, s.px(0.03)))
    # Events, their onsets counting down with the shift.
    for onset, name in ((7.0, "cue"), (13.0, "go"), (17.0, "cue"), (23.0, "go")):
        x = x_first + onset / 2.0 * slot - shift
        s.line(x, 3.75, x, 4.95, "s2", width=max(1.5, s.px(0.05)))
        shown = onset - 6.0 * phase(t, 3.0, 4.6)
        s.text(x + 0.12, 3.75, 2.0, 0.55, name, "text", 10)
        s.text(x + 0.12, 4.25, 2.0, 0.55, f"{shown:.1f} s", "dim", 9, mono=True)
    xs = np.linspace(2.85, 15.4, 420)
    ph = np.mod((xs + shift) / 1.1, 1.0)
    ys = 5.95 - 0.5 * (0.9 * np.exp(-((ph - 0.5) / 0.12) ** 2) + 0.3 * np.exp(-((ph - 0.74) / 0.08) ** 2))
    s.trace(xs, ys, "s3")
    s.p.restore()
    for i in range(10):
        x = x_first + i * slot + 0.52
        s.text(x - 0.6, 2.95, 1.2, 0.45, f"{2 * i} s", "muted", 8, align=_CENTRE)
    if marked > 0:
        a = marked * (1 - gone)
        s.line(x_first, 1.45, x_first + 3 * slot - 0.2, 1.45, "warning", alpha=a,
               width=max(1.5, s.px(0.05)))
        s.chip(x_first, 0.92, "Dummy volumes: 3", "warning", "warning_bg", alpha=marked)
    card = phase(t, 4.9, 5.6)
    s.box(2.95, 6.95, 12.45, 2.55, "surface", "border", alpha=card)
    s.text(3.25, 7.05, 8.0, 0.55, "sub-01_task-rest_bold.json", "tree_sidecar", 10, alpha=card)
    s.json_line(3.25, 7.65, "NumberOfVolumesDiscardedByUser", "3,", alpha=card, width=11.5)
    s.json_line(3.25, 8.25, "RepetitionTime", "2.0", alpha=card, width=11.5)
    s.text(9.6, 8.25, 5.6, 0.6, "Events: every onset 6.0 s earlier", "dim", 9, align=_RIGHT,
           alpha=card)


def _names(s: Stage, t: float) -> None:
    """Subject labels from the DICOM identity, and entities the sequence
    name decides."""
    s.panel(0.5, 0.6, 8.6, 4.5, "DICOM header")
    # The name field holds a study code, as MRI sites type there: never a
    # person's name in a picture of a tool that keeps names out of datasets.
    fields = (("PatientName", "MOT_017"), ("PatientID", "OL_0007"),
              ("SeriesDescription", "t1_mprage_acq-highres"), ("ProtocolName", "t1_mprage"))
    value_x = 3.75
    for i, (tag, _value) in enumerate(fields):
        s.text(0.8, 1.3 + 0.88 * i, 2.9, 0.6, tag, "dim", 10, mono=True)

    which = 0 if t < 1.0 else 1 if t < 5.6 else 2
    sources = ("Numbers (sub-001, sub-002, ...)", "Patient ID", "Family name")
    labels = ("001", "OL0007", "MOT017")
    notes = ("Numbered: no identity field is read.",
             "OL_0007 becomes OL0007: letters and digits only.",
             "The name field holds the study's code: MOT_017 becomes MOT017.")
    starts = (0.0, 1.0, 5.6)
    flash = 1.0 - phase(t, starts[which], starts[which] + 0.6) if which else 0.0

    s.panel(9.5, 0.6, 6.0, 4.5, "Settings > Scan")
    s.text(9.75, 1.3, 5.5, 0.5, "Subject labels", "dim", 9)
    s.box(9.75, 1.8, 5.5, 0.62, "surface2", "accent" if flash > 0.05 else "input_border",
          radius=0.12)
    s.text(9.95, 1.8, 4.8, 0.62, sources[which], "text", 10)
    s.text(14.7, 1.8, 0.4, 0.62, "▾", "dim", 10, align=_CENTRE)
    entity_on = t >= 3.3
    s.text(9.75, 2.75, 5.5, 0.5, "Entities from the sequence name", "dim", 9)
    s.box(9.75, 3.25, 5.5, 0.62, "surface2",
          "accent" if 3.3 <= t < 3.9 else "input_border", radius=0.12)
    if entity_on:
        s.chip(9.95, 3.33, "acq", "text", "surface3", alpha=phase(t, 3.3, 3.6))
    else:
        s.text(9.95, 3.25, 4.8, 0.62, "None", "muted", 10)

    # The name, built from its parts.
    s.panel(0.5, 5.5, 15.0, 4.0, "BIDS name")
    size = 16
    morph = phase(t, starts[which] + 0.9, starts[which] + 1.5) if which else 1.0
    old_label = labels[which - 1] if which else labels[0]
    new_label = labels[which]
    acq = phase(t, 3.9, 4.6)
    parts_w = [s.width("sub-", size), lerp(s.width(old_label, size, bold=True),
                                           s.width(new_label, size, bold=True), morph),
               s.width("_acq-highres", size) * acq, s.width("_T1w.nii.gz", size)]
    x = 8.0 - sum(parts_w) / 2
    y = 6.55
    s.text(x, y, parts_w[0] + 0.1, 0.9, "sub-", "text", size)
    lx = x + parts_w[0]
    old_a, new_a = swap(morph) if which else (0.0, 1.0)
    s.text(lx, y, parts_w[1], 0.9, old_label, "s1", size, bold=True, alpha=old_a, clip=True)
    s.text(lx, y, parts_w[1], 0.9, new_label, "s1", size, bold=True, alpha=new_a, clip=True)
    ax = lx + parts_w[1]
    if acq > 0:
        s.text(ax, y, parts_w[2], 0.9, "_acq-highres", "s2", size, alpha=acq, clip=True)
    s.text(ax + parts_w[2], y, parts_w[3] + 0.2, 0.9, "_T1w.nii.gz", "text", size)
    note_in = phase(t, starts[which] + 1.1, starts[which] + 1.6) if which else 1.0
    note_out = phase(t, starts[which + 1] - 0.4, starts[which + 1]) if which < 2 else 0.0
    s.text(0.8, 7.75, 14.4, 0.6, notes[which], "dim", 10, align=_CENTRE,
           alpha=note_in * (1 - note_out))

    def link(field: int, start: int, text: str, colour: str, target_x: float, at: float,
             a: float) -> None:
        """Mark ``text`` in a header field and draw a line from it down to the
        part of the name it became, growing from ``at``."""
        if a <= 0:
            return
        before = fields[field][1][:start]
        hx = value_x + s.width(before, 10, mono=True) - 0.03
        hw = s.width(text, 10, mono=True) + 0.06
        hy = 1.3 + 0.88 * field
        s.box(hx, hy + 0.05, hw, 0.5, colour, None, radius=0.1, alpha=0.18 * a)
        s.box(hx, hy + 0.05, hw, 0.5, None, colour, radius=0.1, alpha=a)
        grow = phase(t, at, at + 0.7)
        if grow <= 0:
            return
        a_pt = s.point(hx + hw / 2, hy + 0.58)
        b_pt = s.point(target_x, 6.6)
        path = QPainterPath(a_pt)
        path.cubicTo(QPointF(a_pt.x(), a_pt.y() + s.px(1.2)),
                     QPointF(b_pt.x(), b_pt.y() - s.px(1.2)), b_pt)
        part = QPainterPath(a_pt)
        for k in range(1, int(40 * grow) + 1):
            part.lineTo(path.pointAtPercent(k / 40))
        s.p.setPen(QPen(s.colour(colour, a * 0.9), max(1.4, s.px(0.05)), Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap))
        s.p.drawPath(part)

    label_mid = lx + parts_w[1] / 2
    if which == 1:
        link(1, 0, "OL_0007", "s1", label_mid, 1.2, phase(t, 1.1, 1.4) * (1 - phase(t, 5.2, 5.6)))
    elif which == 2:
        link(0, 0, "MOT_017", "s1", label_mid, 5.8, phase(t, 5.7, 6.0))
    link(2, 10, "acq-highres", "s2", ax + parts_w[2] / 2, 3.5,
         phase(t, 3.4, 3.7) * (1 - 0.6 * phase(t, 5.0, 5.6)))
    # The values over the lines that leave them.
    for i, (_tag, value) in enumerate(fields):
        s.text(value_x, 1.3 + 0.88 * i, 5.2, 0.6, value, "text", 10, mono=True)


#: Where each origin's pill takes its colours, as the Editor's sidecar
#: dialog paints them (``widgets/sidecar_views.ORIGIN_TONES``).
_TONES = {"Source data": ("teal", "teal_bg"), "Your answers": ("purple", "purple_bg"),
          "Placeholder": ("warning", "warning_bg")}
#: The file tree's indent in the sidecars scene, narrower to leave the
#: sidecar room for its longest line and its pill.
_INDENT = 0.34


def _sidecars(s: Stage, t: float) -> None:
    """A missing sidecar written again, each field from where it is known."""
    s.panel(0.5, 0.6, 5.8, 4.9, "Editor")
    rows = (("sub-01", 0, "folder"), ("func", 1, "folder"),
            ("sub-01_task-rest_bold.nii.gz", 2, "image"), ("sub-01_task-rest_bold.json", 2, "sidecar"),
            ("sub-01_task-rest_events.tsv", 2, "table"))
    written = phase(t, 7.0, 7.5)
    for i, (name, depth, kind) in enumerate(rows):
        y = 1.2 + 0.66 * i
        if i == 3:
            if written < 1:
                s.box(0.68 + 2 * _INDENT, y + 0.04, 4.75, 0.56, None, "error", radius=0.1,
                      alpha=1 - written, dash=True)
                s.tree_row(0.75, y, depth, name, kind, width=5.0, colour="muted", italic=True,
                           alpha=1 - written, size=9, indent=_INDENT)
                s.dot(5.98, y + 0.31, 0.08, "error", alpha=1 - written)
            s.tree_row(0.75, y, depth, name, kind, width=5.0, alpha=written, size=9,
                       indent=_INDENT)
            s.dot(5.98, y + 0.31, 0.08, "success", alpha=written)
        else:
            s.tree_row(0.75, y, depth, name, kind, width=5.0, size=9, indent=_INDENT)
    origins = ("Source data", "File header", "Your answers", "Other sidecars", "File name",
               "Placeholder")
    s.panel(0.5, 5.85, 5.8, 3.65, "Read from")
    lines = (("RepetitionTime", "2.0,", 0), ("EchoTime", "0.03,", 0),
             ("PhaseEncodingDirection", '"j-",', 0), ("TaskName", '"rest",', 4),
             ("InstitutionName", '"Oldenburg",', 2), ("B0FieldSource", '"fmap0",', 3),
             ("Instructions", '"TODO"', 5))
    first, gap = 2.0, 0.66
    for i, origin in enumerate(origins):
        y = 6.4 + 0.5 * i
        fg, _bg = _TONES.get(origin, ("dim", "surface3"))
        used = [k for k, line in enumerate(lines) if line[2] == i]
        glow = max((1 - abs(t - (first + gap * k + 0.1)) / 0.45 for k in used), default=0.0)
        if glow > 0:
            s.box(0.65, y + 0.02, 5.5, 0.46, "surface3", None, radius=0.1, alpha=glow)
        s.dot(0.95, y + 0.25, 0.09, fg)
        s.text(1.2, y, 5.6, 0.5, origin, "text", 9)

    s.panel(6.7, 0.6, 8.8, 8.9)
    s.text(6.95, 0.7, 8.3, 0.55, "sub-01_task-rest_bold.json", "tree_sidecar", 10, bold=True)
    s.text(6.95, 1.3, 1.0, 0.6, "{", "dim", 10, mono=True)
    s.text(6.7, 4.5, 8.8, 0.6, "No sidecar yet", "muted", 10, align=_CENTRE,
           alpha=1 - phase(t, 1.7, 2.0))
    for k, (key, value, origin) in enumerate(lines):
        at = first + gap * k
        a = phase(t, at, at + 0.35)
        if a <= 0:
            continue
        y = 1.9 + 0.68 * k
        fly = phase(t, at - 0.45, at)
        if fly < 1:
            sx, sy = 6.1, 6.65 + 0.5 * origin
            fx, fy = lerp(sx, 7.1, fly), lerp(sy, y + 0.3, fly) - 0.8 * math.sin(math.pi * fly)
            s.dot(fx, fy, 0.08, _TONES.get(origins[origin], ("dim",))[0], alpha=1 - fly * 0.3)
        s.json_line(7.25 - 0.3 * (1 - a), y, key, value, size=10, alpha=a, width=6.0,
                    value_colour="warning" if origin == 5 else "text")
        fg, bg = _TONES.get(origins[origin], ("dim", "surface3"))
        s.chip(15.25, y + 0.07, origins[origin], fg, bg, size=8, alpha=a, right=True)
    end_a = phase(t, first + gap * 6, first + gap * 6 + 0.3)
    s.text(6.95, 1.9 + 0.68 * 7, 1.0, 0.6, "}", "dim", 10, mono=True, alpha=end_a)
    s.text(6.95, 8.75, 8.3, 0.55, "Never over a value already there. One step: Undo takes it back.",
           "dim", 9, alpha=written)
    # The right-click that starts it.
    menu = phase(t, 0.7, 0.95) * (1 - phase(t, 1.75, 2.0))
    s.menu(2.7, 3.15, ("Write its sidecar from the recording...",),
           0 if t > 1.2 else -1, alpha=menu, width=s.width("Write its sidecar from the recording...",
                                                           10) + 0.6)
    if t < 2.0:
        go = phase(t, 0.2, 1.2)
        s.cursor(lerp(5.0, 3.6, go), lerp(1.4, 3.5, go),
                 press=max(0.0, min(1.0, (t - 1.45) / 0.3)) if t > 1.45 else 0.0,
                 alpha=1 - phase(t, 1.75, 2.0))


@lru_cache(maxsize=4)
def _slice(kind: str) -> Optional[QImage]:
    """A sagittal slice of the MNI template the quality check ships, the nose
    to the left: the head (``"head"``) or the brain alone (``"brain"``)."""
    try:
        from .star_request import template_volume

        t1 = template_volume("mni2009c_2mm_T1w.nii.gz")
        mask = template_volume("mni2009c_2mm_brain_mask.nii.gz")
    except (OSError, ValueError):
        return None
    mid = t1.shape[0] // 2
    image = t1[mid].T[::-1, ::-1].astype(np.float32)
    if kind == "brain":
        image = image * (mask[mid].T[::-1, ::-1] / 255.0)
    data = np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))
    h, w = data.shape
    return QImage(data.data, w, h, w, QImage.Format.Format_Grayscale8).copy()


def _derivatives(s: Stage, t: float) -> None:
    """An image in derivatives/ skull stripped, the result in its own
    pipeline folder."""
    s.box(0.5, 0.6, 7.4, 8.9, s.theme.background, "border")
    head, brain = _slice("head"), _slice("brain")
    sweep = phase(t, 2.8, 4.6)
    if head is not None and brain is not None:
        box_w = 6.8
        box_h = box_w * head.height() / head.width()
        target = s.rect(0.8, 0.6 + (8.9 - box_h) / 2, box_w, box_h)
        s.p.save()
        s.p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if sweep >= 1:
            s.p.fillRect(target, QColor(s.theme.background))
            s.p.drawImage(target, brain)
        else:
            s.p.drawImage(target, head)
        if 0 < sweep < 1:
            cut = target.top() + target.height() * sweep
            s.p.setClipRect(QRectF(target.left(), target.top(), target.width(),
                                   cut - target.top()))
            s.p.fillRect(target, QColor(s.theme.background))
            s.p.drawImage(target, brain)
            s.p.setClipping(False)
            s.p.setPen(QPen(s.colour("s1", 0.9), max(1.5, s.px(0.05))))
            s.p.drawLine(QPointF(target.left(), cut), QPointF(target.right(), cut))
        s.p.restore()
    label = "desc-preprocbrain" if sweep >= 1 else "desc-preproc"
    s.text(0.75, 8.8, 6.9, 0.55, label, s.theme.canvas_text, 10, bold=True)
    s.text(0.75, 8.8, 6.9, 0.55, "Skull stripped" if sweep >= 1 else "Preprocessed T1w",
           s.theme.canvas_dim, 9, align=_RIGHT)

    s.panel(8.3, 0.6, 7.2, 8.9, "Editor")
    rows = (("derivatives", 0, "folder"), ("fmriprep", 1, "folder"), ("sub-01", 2, "folder"),
            ("anat", 3, "folder"), ("sub-01_desc-preproc_T1w.nii.gz", 4, "image"))
    new = (("bidsmgr-skullstrip", 1, "folder"), ("sub-01", 2, "folder"), ("anat", 3, "folder"),
           ("sub-01_desc-preprocbrain_T1w.nii.gz", 4, "image"))
    picked = phase(t, 0.5, 0.8)
    for i, (name, depth, kind) in enumerate(rows):
        y = 1.3 + 0.62 * i
        if i == 4 and picked > 0:
            s.box(8.45, y + 0.03, 6.9, 0.56, "accent_bg", None, radius=0.1, alpha=picked)
        s.tree_row(8.55, y, depth, name, kind, width=6.8, indent=0.36, size=9)
    for i, (name, depth, kind) in enumerate(new):
        a = phase(t, 4.8 + 0.18 * i, 5.2 + 0.18 * i)
        y = 1.3 + 0.62 * (5 + i)
        if i == 3:
            s.box(8.45, y + 0.03, 6.9, 0.56, "success_bg", None, radius=0.1,
                  alpha=a * (1 - 0.6 * phase(t, 6.4, 7.4)))
        s.tree_row(8.55 + 0.25 * (1 - a), y, depth, name, kind, width=6.8, indent=0.36, size=9,
                   alpha=a)
    menu = phase(t, 1.1, 1.35) * (1 - phase(t, 2.45, 2.7))
    s.menu(10.2, 4.45, ("Remove faces...", "Remove the skull..."), 1 if t > 1.85 else -1,
           alpha=menu, width=3.9)
    if t < 2.8:
        go = phase(t, 0.9, 1.8)
        s.cursor(lerp(12.6, 12.0, go), lerp(4.0, 5.35, go),
                 press=max(0.0, min(1.0, (t - 2.2) / 0.3)) if t > 2.2 else 0.0,
                 alpha=phase(t, 0.6, 0.9) * (1 - phase(t, 2.5, 2.8)))


def _undo(s: Stage, t: float) -> None:
    """Renames and deletions are steps of one history, undone and redone
    with the keyboard."""
    from PyQt6.QtGui import QKeySequence

    native = QKeySequence.SequenceFormat.NativeText
    undo_key = QKeySequence(QKeySequence.StandardKey.Undo).toString(native) or "Ctrl+Z"
    redo_key = QKeySequence(QKeySequence.StandardKey.Redo).toString(native) or "Ctrl+Shift+Z"

    # The rename: on at 1.0, undone at 6.2, redone at 7.8.
    renamed = phase(t, 1.0, 1.6) * (1 - phase(t, 6.2, 6.8)) + phase(t, 7.8, 8.4)
    # The deletion: at 2.4, undone at 4.6.
    deleting = phase(t, 2.9, 3.5) * (1 - phase(t, 4.6, 5.2))
    doomed = phase(t, 2.3, 2.6) * (1 - phase(t, 2.9, 3.2))

    s.panel(0.5, 0.6, 8.6, 8.9, "Editor")
    rows = (("{s}", 0, "folder", 0), ("anat", 1, "folder", 0), ("{s}_T1w.nii.gz", 2, "image", 0),
            ("func", 1, "folder", 1), ("{s}_task-rest_bold.nii.gz", 2, "image", 1),
            ("{s}_task-rest_bold.json", 2, "sidecar", 1), ("{s}_task-rest_events.tsv", 2, "table", 1),
            ("sub-02", 0, "folder", 0), ("anat", 1, "folder", 0), ("sub-02_T1w.nii.gz", 2, "image", 0))
    y = 1.2
    for name, depth, kind, group in rows:
        height = 0.62 * (1 - deleting) if group else 0.62
        if height < 0.05:
            y += height
            continue
        a = (1 - deleting) ** 2 if group else 1.0
        s.p.save()
        s.p.setClipRect(s.rect(0.55, y, 8.5, height))
        if group and doomed > 0:
            s.box(0.65, y + 0.03, 8.3, height - 0.06, "error_bg", None, radius=0.1,
                  alpha=doomed * a)
        if "{s}" in name:
            flash = max(0.0, 1 - abs(t - 1.3) / 0.4, 1 - abs(t - 6.5) / 0.4, 1 - abs(t - 8.1) / 0.4)
            if flash > 0:
                s.box(0.65, y + 0.03, 8.3, 0.56, "accent_bg", None, radius=0.1, alpha=flash)
            old_a, new_a = swap(renamed)
            s.tree_row(0.75, y, depth, "", kind, width=8.0, alpha=a)
            s.tree_row(0.75, y, depth, name.format(s="sub-01"), kind, width=8.0,
                       alpha=a * old_a, glyph=False)
            s.tree_row(0.75, y, depth, name.format(s="sub-101"), kind, width=8.0,
                       alpha=a * new_a, glyph=False)
        else:
            s.tree_row(0.75, y, depth, name, kind, width=8.0, alpha=a)
        s.p.restore()
        y += height

    # The keys pressed.
    presses = ((4.2, 4.6, undo_key, "Undo"), (5.8, 6.2, undo_key, "Undo"),
               (7.4, 7.8, redo_key, "Redo"))
    for appear, press_at, key, verb in presses:
        a = phase(t, appear, appear + 0.25) * (1 - phase(t, press_at + 0.9, press_at + 1.2))
        if a <= 0:
            continue
        press = max(0.0, 1 - abs(t - press_at - 0.1) / 0.18)
        kw = s.width(key, 11, bold=True) + 0.4
        total = max(0.62, kw) + 0.2 + s.width(verb, 10)
        x = 4.8 - total / 2
        w = s.keycap(x, 8.55, key, press=press, alpha=a)
        s.text(x + w + 0.2, 8.55, 2.5, 0.6, verb, "dim", 10, alpha=a)

    s.panel(9.5, 0.6, 6.0, 8.9, "History")
    entries = (("Rename sub-01 to sub-101", 1.4, (6.2, 7.8)),
               ("Delete sub-101/func and its 3 files", 3.3, (4.6, 99.0)))
    for i, (label, at, (undone_at, redone_at)) in enumerate(entries):
        a = phase(t, at, at + 0.35)
        if a <= 0:
            continue
        undone = undone_at <= t < redone_at
        y = 1.35 + 1.05 * i
        s.box(9.7, y, 5.6, 0.9, "surface2", None, radius=0.12, alpha=a)
        s.dot(10.0, y + 0.45, 0.08, "muted" if undone else "accent", alpha=a)
        s.text(10.25, y, 4.95, 0.9, label, "muted" if undone else "text", 9, italic=undone,
               wrap=True, alpha=a)
    s.text(9.75, 8.75, 5.5, 0.6, "Undone steps can be redone", "dim", 9, alpha=phase(t, 4.8, 5.2))


def _anonymise(s: Stage, t: float) -> None:
    """Anonymising by purpose: each choice changes its fields, as the
    anonymiser writes them."""
    s.panel(0.5, 0.6, 5.0, 8.9, "What to change")
    groups = (("THE PARTICIPANTS", ("Names and IDs",)), ("WHEN IT HAPPENED", ("Dates and times",)),
              ("THE SITE AND ITS EQUIPMENT", ("The site", "Devices")))
    ticks = (1.0, 2.7, 4.4, 6.1)
    y = 1.35
    k = 0
    for heading, items in groups:
        s.text(0.75, y, 4.6, 0.45, heading, "dim", 8, bold=True)
        y += 0.5
        for item in items:
            on = phase(t, ticks[k], ticks[k] + 0.2)
            s.checkbox(0.8, y + 0.16, on)
            s.text(1.25, y, 4.1, 0.6, item, "text", 10)
            y += 0.62
            k += 1
        y += 0.35
    s.text(0.75, 8.75, 4.6, 0.6, "Preset: Recommended", "dim", 9)

    s.panel(6.0, 0.6, 9.5, 8.9)
    s.text(6.25, 0.7, 9.0, 0.55, "sub-01_T1w.json", "tree_sidecar", 10, bold=True)
    # (key, before, after (None: removed), when it changes)
    lines = (("PatientName", '"MOT_017"', None, ticks[0]),
             ("PatientID", '"OL_0007"', None, ticks[0]),
             ("AcquisitionDateTime", '"2026-03-14T09:12:00"', '"1905-03-11T09:12:00"', ticks[1]),
             ("InstitutionName", '"University of Oldenburg"', '"University of Oldenburg"', 99.0),
             ("InstitutionAddress", '"Ammerländer Heerstr. 114"', '"n/a"', ticks[2]),
             ("DeviceSerialNumber", '"167023"', '"device-45de5a5d"', ticks[3]),
             ("StationName", '"AWP167023"', '"station-2ef24c50"', ticks[3]),
             ("RepetitionTime", "2.3", "2.3", 99.0))
    y = 1.45
    glyphs = "abcdef0123456789-:T"
    for key, before, after, at in lines:
        change = phase(t, at + 0.3, at + 1.1)
        if after is None:
            height = 0.66 * (1 - phase(t, at + 1.0, at + 1.5))
            if height < 0.04:
                continue
            band = phase(t, at + 0.2, at + 0.4)
            s.box(6.15, y + 0.04, 9.2, height - 0.08, "error_bg", None, radius=0.08,
                  alpha=band * height / 0.66)
            s.json_line(6.35, y, key, before + ",", size=10, strike=change > 0.3,
                        alpha=height / 0.66, width=8.9)
            y += height
            continue
        band = max(0.0, 1 - abs(t - (at + 0.7)) / 0.6)
        if band > 0:
            s.box(6.15, y + 0.04, 9.2, 0.58, "accent_bg", None, radius=0.08, alpha=band)
        if 0 < change < 1:
            n = round(lerp(len(before), len(after), change))
            seed = int(t * 18)
            scrambled = '"' + "".join(glyphs[(seed * 7 + j * 13) % len(glyphs)]
                                      for j in range(max(0, n - 2))) + '"'
            value = scrambled if change < 0.75 else after
        else:
            value = after if change >= 1 else before
        comma = "" if key == "RepetitionTime" else ","
        s.json_line(6.35, y, key, value + comma, size=10, width=8.9)
        y += 0.66
    s.text(6.25, 8.4, 9.0, 0.9, "Dates move back by whole weeks, the same for a participant's "
           "every file: weekdays and intervals stay.", "dim", 9, wrap=True,
           align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, alpha=phase(t, 3.6, 4.0))


def _ignore_and_tables(s: Stage, t: float) -> None:
    """A stray file ignored in one step, and the scans table BIDS asks for."""
    ignored = phase(t, 2.5, 2.9)
    s.panel(0.5, 0.6, 6.5, 8.9, "Editor")
    if ignored < 1:
        s.chip(6.8, 0.68, "1 error", "error", "error_bg", right=True, alpha=1 - ignored)
    s.chip(6.8, 0.68, "No errors", "success", "success_bg", right=True, alpha=ignored)
    rows = (("sub-01", 0, "folder"), ("anat", 1, "folder"), ("sub-01_T1w.nii.gz", 2, "image"),
            ("notes.txt", 2, "other"), ("eeg", 1, "folder"),
            ("sub-01_task-rest_eeg.edf", 2, "recording"), ("func", 1, "folder"),
            ("sub-01_task-rest_bold.nii.gz", 2, "image"))
    for i, (name, depth, kind) in enumerate(rows):
        y = 1.35 + 0.62 * i
        if i == 3:
            if ignored < 1:
                s.tree_row(0.7, y, depth, name, kind, width=5.9, alpha=1 - ignored)
                s.dot(6.6, y + 0.31, 0.08, "error", alpha=1 - ignored)
            s.tree_row(0.7, y, depth, name, kind, width=5.9, colour="muted", italic=True,
                       alpha=ignored)
        else:
            s.tree_row(0.7, y, depth, name, kind, width=5.9)
    table_row = phase(t, 6.4, 6.8)
    s.tree_row(0.7 + 0.2 * (1 - table_row), 1.35 + 0.62 * 8, 1, "sub-01_scans.tsv", "table",
               width=5.9, alpha=table_row)
    s.text(0.75, 8.75, 6.0, 0.6, "Ignored: muted, in italics", "dim", 9, alpha=phase(t, 3.0, 3.4))

    s.panel(7.4, 0.6, 8.1, 8.9)
    tools = phase(t, 3.8, 4.2)
    s.text(7.4, 4.5, 8.1, 0.6, "No scans table yet", "muted", 10, align=_CENTRE,
           alpha=1 - phase(t, 4.6, 4.9))
    s.chip(7.65, 0.7, "Tools  ›  Scans and sessions tables...", "text", "surface3", size=9,
           bold=False, alpha=tools)
    built = phase(t, 4.8, 5.2)
    s.text(7.65, 1.45, 7.6, 0.55, "sub-01_scans.tsv", "tree_table", 10, bold=True, alpha=built)
    s.text(7.7, 2.1, 4.6, 0.5, "filename", "dim", 9, bold=True, alpha=built)
    s.text(12.4, 2.1, 3.0, 0.5, "acq_time", "dim", 9, bold=True, alpha=built)
    s.line(7.65, 2.6, 15.25, 2.6, "subtle", alpha=built)
    for i, (name, when) in enumerate((("anat/sub-01_T1w.nii.gz", "2026-03-14T09:12:00"),
                                      ("eeg/sub-01_task-rest_eeg.edf", "2026-03-14T11:05:42"),
                                      ("func/sub-01_task-rest_bold.nii.gz", "2026-03-14T09:31:10"))):
        a = phase(t, 5.3 + 0.45 * i, 5.7 + 0.45 * i)
        y = 2.7 + 0.62 * i
        s.text(7.7, y, 4.6, 0.6, name, "text", 9, alpha=a)
        s.text(12.4, y, 3.0, 0.6, when, "text", 9, mono=True, alpha=a)
    s.text(7.65, 7.9, 7.6, 1.4, "Every kind of recording is listed, with the time it was "
           "acquired; your own rows and columns are kept.", "dim", 9, wrap=True,
           align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
           alpha=phase(t, 6.6, 7.0))
    # The right-click, drawn last: a menu floats over both panels.
    menu = phase(t, 0.6, 0.85) * (1 - phase(t, 2.35, 2.6))
    s.menu(2.2, 3.85, ("Ignore in validation  ›",), 0 if t > 1.1 else -1, alpha=menu, width=3.7)
    sub = phase(t, 1.3, 1.5) * (1 - phase(t, 2.35, 2.6))
    s.menu(5.85, 3.85, ("This file", "Every .txt file"), 1 if t > 1.75 else -1, alpha=sub,
           width=3.6, notes=("1 file", "1 file"))
    if t < 2.7:
        # To notes.txt (right-click), onto the menu entry, into the submenu.
        go, mid, go2 = phase(t, 0.15, 0.6), phase(t, 0.9, 1.15), phase(t, 1.45, 1.9)
        cx = lerp(lerp(lerp(4.6, 3.0, go), 3.7, mid), 7.4, go2)
        cy = lerp(lerp(lerp(6.0, 3.55, go), 4.2, mid), 4.73, go2)
        s.cursor(cx, cy, press=max(0.0, min(1.0, (t - 2.05) / 0.3)) if t > 2.05 else 0.0,
                 alpha=1 - phase(t, 2.4, 2.7))


# ---------------------------------------------------------------------------
# What the dialog lists
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scene:
    """One feature: its words in the list and under the stage, its animation."""

    key: str
    version: str
    title: str
    text: str
    #: What the animation shows, for a screen reader.
    shows: str
    #: Seconds before the next feature plays.
    duration: float
    paint: Callable[[Stage, float], None]


SCENES: tuple[Scene, ...] = (
    Scene("guide", "1.5.2", "A guide that shows the way",
          "Turn on Guide, beside Settings. It asks what you want to do, then rings each "
          "control in turn and says what to do there, with Previous and Next.",
          "The guide asks what to do, then rings the table, Properties and Structure in turn",
          10.4, _guide),
    Scene("physio", "1.5.2", "Physiology from Siemens CMRR logs",
          "Every ECG lead is its own column and the trigger marks each volume. Write one "
          "table per sampling rate, or one table interpolated or as recorded, cropped to "
          "the run if you like.",
          "Four ECG leads, pulse, breathing and the volume trigger, then the three table "
          "layouts", 8.8, _physiology),
    Scene("dummy", "1.5.2", "Dummy volumes left out",
          "Leave out the first volumes of a functional run. The image, its events and its "
          "physiology move together, and the sidecar records how many were dropped.",
          "Three volumes leave a run; the events and the physiology move with it", 8.0,
          _dummy_volumes),
    Scene("names", "1.5.2", "Names from the scanner",
          "Name subjects from the patient ID, or the code typed as the patient name, "
          "instead of a number, and let a token such as acq-highres in the sequence name "
          "decide an entity.",
          "A subject label taken from the patient ID, then from the study code in the "
          "patient name field, and acq taken from the series description", 9.0, _names),
    Scene("sidecars", "1.5.2", "Sidecars written from the recordings",
          "A missing or incomplete sidecar is written again from the source data, the "
          "file's header, your answers and the files that link to it, never over a value "
          "already there.",
          "A missing sidecar written field by field, each saying where it was read", 9.4,
          _sidecars),
    Scene("derivatives", "1.5.2", "Defacing and skull stripping in derivatives",
          "Images in derivatives can be defaced or skull stripped too. A stripped image "
          "goes to derivatives/bidsmgr-skullstrip, with its description plus brain.",
          "A preprocessed T1w skull stripped into its own pipeline folder", 8.0,
          _derivatives),
    Scene("undo", "1.5.1", "Undo and redo for the dataset",
          "Renames, merges, deletions and repairs are steps of one history. Undo and redo "
          "them with the keyboard, or go back several steps from the menu.",
          "A rename and a deletion, undone and redone with the keyboard", 9.6, _undo),
    Scene("anonymise", "1.5.1", "Anonymise by purpose",
          "Choose what to protect: names and IDs, dates, the site, devices. Sidecars, "
          "tables and headers follow.",
          "Each purpose ticked changes its fields in a sidecar", 8.8, _anonymise),
    Scene("ignore", "1.5.1", "Ignore files, and the tables BIDS asks for",
          "Hide files from validation in one step, and write scans and sessions tables for "
          "every kind of data.",
          "A stray file ignored from its right-click, then a scans table written", 8.6,
          _ignore_and_tables),
)

#: What the release brings beyond the scenes, in one line under the list.
ALSO = ("Also: network shares on Windows for scanning and converting, faster checks on "
        "large datasets, and acquisition values written only into the recording's own "
        "sidecar.")


def paint_scene(painter: QPainter, rect: QRectF, scene: Scene, t: float) -> None:
    """``scene`` at ``t`` seconds on ``rect``, in the current theme."""
    from ..viz.theme import VizTheme
    from .theme_manager import CUR

    palette = CUR()
    stage = Stage(painter, rect, palette, VizTheme.from_palette(palette))
    scene.paint(stage, t)


__all__ = ["ALSO", "H", "SCENES", "Scene", "Stage", "W", "ease", "lerp", "paint_scene", "phase",
           "swap"]
