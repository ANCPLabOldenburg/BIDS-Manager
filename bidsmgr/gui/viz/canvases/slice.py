"""The 2-D slice canvas: one plane of the scene, drawn with QPainter.

No OpenGL, on purpose: a 2-D view has to work on a remote desktop, in a
virtual machine, under software GL and in the offscreen test platform, where
the 3-D render is simply hidden.

Drawing is two steps. The IMAGE (every visible volume layer sampled on the
plane's grid, coloured and composited) is computed only when something it
depends on changes, and cached as a ``QImage`` at the data's own resolution;
QPainter scales it to the screen. Everything drawn on TOP (crosshair, letters,
caption, colour bar, a rubber band) is redrawn every paint and costs nothing.
Moving the crosshair inside a plane therefore never re-samples that plane.

Size on screen is in millimetres (``SliceGrid.pixel_mm``), so anisotropic
voxels are drawn to scale with no special case.

Input goes through the mouse map (:mod:`bidsmgr.viz.inputmap`): this module
knows how to run each TOOL, not which button runs it.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ....viz import colorbar, inputmap, render2d, views
from ....viz.compute import colormaps, geometry
from ..bridge import connect_while_alive
from ..context import ViewerContext
from .. import fonts

log = logging.getLogger(__name__)

#: Wheel deltas that make one step: a mouse notch is 120 eighths of a degree;
#: trackpads stream pixel deltas, whose threshold tames their high rate.
WHEEL_ANGLE_STEP = 120.0
WHEEL_PIXEL_STEP = 40.0
#: A press that moves less than this is a click, not a drag.
CLICK_SLOP = 4

_HALO = QColor(0, 0, 0, 160)


#: Height of one colour bar's strip (title, bar, tick labels).
BAR_PX = 40
#: Space between an orientation letter and the image's edge.
_LETTER_GAP = 3.0


class SliceCanvas(QWidget):
    """One plane of a scene. ``plane=None`` follows the scene's single plane."""

    def __init__(self, ctx: ViewerContext, plane: Optional[str] = None, *,
                 caption: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._fixed_plane = plane
        self._show_caption = caption
        self.setMouseTracking(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(40, 40)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        # -- cache -------------------------------------------------------
        self._image_key: Optional[tuple] = None
        self._image: Optional[QImage] = None
        self._image_buf: Optional[np.ndarray] = None
        self._grid: Optional[geometry.SliceGrid] = None
        self._image_rect = QRectF()
        self._letter_boxes: dict[str, QRectF] = {}
        self._smooth = True
        self._loading = False
        # -- gesture state ---------------------------------------------
        self._tool = "none"
        self._press_pos: Optional[QPointF] = None
        self._last_pos: Optional[QPointF] = None
        self._moved = False
        self._band: Optional[QRectF] = None
        self._wheel_acc: dict = {}
        ctx.qstore.changed.connect(self._on_changed)
        connect_while_alive(ctx.theme_hub.changed, self, lambda w, _t: w.update())
        connect_while_alive(ctx.settings_hub.changed, self, lambda w, _s: w.update())

    def sizeHint(self):  # noqa: N802 - Qt signature
        from PyQt6.QtCore import QSize

        return QSize(320, 320)

    # ------------------------------------------------------------------
    @property
    def plane(self) -> str:
        return self._fixed_plane or self.ctx.scene.plane

    def set_caption(self, on: bool) -> None:
        self._show_caption = bool(on)
        self.update()

    def _on_changed(self, paths) -> None:
        if not self.isVisible():
            return
        for p in paths:
            if p.startswith(("render.", "clips", "graph")):
                continue
            self.update()
            return

    # ------------------------------------------------------------------
    # The image
    # ------------------------------------------------------------------

    def _layers_key(self) -> tuple:
        store = self.ctx.store
        parts = []
        for layer in store.scene.layers:
            if layer.kind != "volume" or not layer.visible:
                continue
            src = views.source_of(store, layer)
            if src is None:
                continue
            t = views.frame_of(store, layer, src)
            parts.append((
                layer.id, t, src.frame_ready(t),
                layer.display.model_dump_json(),
            ))
        return tuple(parts)

    def _ensure_image(self) -> bool:
        """(Re)build the cached image if its inputs changed. False: nothing
        to draw yet (no volume, or its first frame still loading)."""
        store = self.ctx.store
        grid = views.grid(store, self.plane)
        self._grid = grid
        if grid is None:
            self._image = None
            return False
        key = (grid.key(), self._layers_key(), self.ctx.transparent)
        if key == self._image_key and self._image is not None:
            return True
        # The composition is the library's (render2d), so a scripted figure
        # and the screen cannot disagree.
        img = render2d.slice_rgba(store, self.plane, transparent=self.ctx.transparent,
                                  grid=grid)
        if img is None:
            # Nothing visible is not loading; a visible base not read yet is.
            self._loading = any(layer.kind == "volume" and layer.visible
                                for layer in store.scene.layers)
            self._image = None
            self._image_key = None
            return False
        self._loading = False
        self._smooth = img.smooth
        buf = img.rgba
        rows, cols = buf.shape[:2]
        self._image_buf = buf
        self._image = QImage(buf.data, cols, rows, cols * 4, QImage.Format.Format_RGBA8888)
        self._image_key = key
        return True

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def _margins(self) -> tuple[float, float, float, float]:
        """(left, top, right, bottom) reserved around the image: the caption,
        the colour bars and, when they are shown, a band on every side for
        the orientation letters. The image is fitted inside and CLIPPED to
        what is left (``_image_area``), so a letter is never drawn on the
        image, at any size, zoom or pan."""
        side, top, bottom = self._frame()
        if self._letters_fit(side, top, bottom):
            band_w, band_h = self._letter_band()
            side += band_w
            top += band_h
            bottom += band_h
        return side, top, side, bottom

    def _frame(self) -> tuple[float, float, float]:
        """(each side, top, bottom) before the letters: the caption above
        and the colour bars below."""
        top = float(fonts.px(16)) if self._show_caption else 2.0
        bottom = 2.0
        if self.ctx.scene.display.colorbar:
            # Each bar its own strip BELOW the image, never over it.
            bottom += self._bar_px() * max(1, len(colorbar.bars_for(self.ctx.store)))
        return 2.0, top, bottom

    def _letters_fit(self, side: float, top: float, bottom: float) -> bool:
        """Whether the orientation letters are shown: asked for, and the
        canvas big enough that their bands leave the image at least twice
        their size each way (a thumbnail-sized slice drops them rather than
        shrink to a sliver between them)."""
        if not self.ctx.scene.display.labels:
            return False
        band_w, band_h = self._letter_band()
        return (self.width() - 2.0 * side >= 4.0 * band_w
                and self.height() - top - bottom >= 4.0 * band_h)

    def letters_shown(self) -> bool:
        return self._letters_fit(*self._frame())

    @staticmethod
    def _letter_band() -> tuple[float, float]:
        """(width, height) of the band an orientation letter needs beside
        and above or below the image: the widest letter any plane can show,
        so the image does not move when the orientation changes, plus a gap
        on both sides."""
        from PyQt6.QtGui import QFontMetricsF

        fm = QFontMetricsF(fonts.font(fonts.LETTER_PX, bold=True))
        widest = max(fm.horizontalAdvance(ch) for ch in "RLAPSI")
        return widest + 2.0 * _LETTER_GAP, fm.height() + 2.0 * _LETTER_GAP

    def _image_area(self) -> QRectF:
        """The part of the canvas the image may cover: the widget less the
        margins. Zoomed in, the image is larger than this and is cut at its
        edge, which keeps the letters' band clear."""
        left, top, right, bottom = self._margins()
        return QRectF(left, top, max(1.0, self.width() - left - right),
                      max(1.0, self.height() - top - bottom))

    @staticmethod
    def _bar_px() -> float:
        """One colour bar's strip: its title, the bar, its numbers, at the
        app's font size (a fixed 40 px cut larger numbers off)."""
        from PyQt6.QtGui import QFontMetricsF

        th = QFontMetricsF(fonts.font(10)).height()
        return max(float(BAR_PX), 2.0 * th + 13.0)

    def _place(self) -> QRectF:
        """Where the image goes on screen (zoom and pan applied)."""
        grid = self._grid
        if grid is None:
            return QRectF()
        avail = self._image_area()
        w_mm, h_mm = grid.extent_mm
        if w_mm <= 0 or h_mm <= 0:
            return QRectF()
        state = self.ctx.scene.views.get(self.plane)
        zoom = state.zoom if state is not None else 1.0
        pan = state.pan if state is not None else (0.0, 0.0)
        ppm = min(avail.width() / w_mm, avail.height() / h_mm) * zoom
        w_px, h_px = w_mm * ppm, h_mm * ppm
        cx = avail.center().x() + pan[0] * ppm
        cy = avail.center().y() + pan[1] * ppm
        return QRectF(cx - w_px / 2.0, cy - h_px / 2.0, w_px, h_px)

    def pixels_per_mm(self) -> float:
        grid = self._grid
        if grid is None or self._image_rect.width() <= 0:
            return 1.0
        return self._image_rect.width() / max(grid.extent_mm[0], 1e-9)

    def screen_to_grid(self, pos: QPointF) -> Optional[tuple[float, float]]:
        """Screen point to (col, row) of the grid, pixel CENTRES at integers."""
        grid, rect = self._grid, self._image_rect
        if grid is None or rect.width() <= 0 or rect.height() <= 0:
            return None
        rows, cols = grid.shape
        c = (pos.x() - rect.left()) / rect.width() * cols - 0.5
        r = (pos.y() - rect.top()) / rect.height() * rows - 0.5
        return c, r

    def grid_to_screen(self, col: float, row: float) -> QPointF:
        grid, rect = self._grid, self._image_rect
        rows, cols = grid.shape
        return QPointF(rect.left() + (col + 0.5) / cols * rect.width(),
                       rect.top() + (row + 0.5) / rows * rect.height())

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt signature
        p = QPainter(self)
        theme = self.ctx.theme
        self._letter_boxes = {}
        if not self.ctx.transparent:
            p.fillRect(self.rect(), QColor(theme.background))
        try:
            have = self._ensure_image()
        except Exception:  # noqa: BLE001 - a bad frame must not kill the paint loop
            log.exception("could not draw the %s slice", self.plane)
            have = False
        if not have:
            self._image_rect = QRectF()
            if self._loading:
                p.setPen(QColor(theme.canvas_dim))
                p.setFont(fonts.font(12))
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Loading...")
            p.end()
            return
        self._image_rect = self._place()
        # Everything drawn ON the image stays inside its area, so a zoomed
        # image never reaches the letters, the caption or the colour bars.
        p.setClipRect(self._image_area())
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._smooth)
        p.drawImage(self._image_rect, self._image)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self._paint_shapes(p)
        display = self.ctx.scene.display
        if display.crosshair:
            self._paint_crosshair(p)
        p.setClipping(False)
        if self.letters_shown():
            self._paint_labels(p)
        if self._show_caption:
            self._paint_caption(p)
        if display.colorbar:
            self._paint_colorbar(p)
        if self._band is not None:
            pen = QPen(QColor(theme.accent))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(self._band)
        p.end()

    def shape_polygons(self) -> list[tuple]:
        """``(layer, QPolygonF)`` for every shape layer this slice cuts, in
        screen coordinates (empty before the first paint)."""
        found = views.shape_layers(self.ctx.store)
        grid = self._grid
        if not found or grid is None or self._image_rect.width() <= 0:
            return []
        from ....viz.compute.shapes import section

        out = []
        for layer, src in found:
            poly = section(src.box, grid.origin, grid.normal)
            if not len(poly):
                continue
            pts = []
            for w in poly:
                c, r, _d = grid.world_to_pixel(w)
                pts.append(self.grid_to_screen(c, r))
            out.append((layer, QPolygonF(pts)))
        return out

    def _paint_shapes(self, p: QPainter) -> None:
        """Every shape layer (the MRS voxel) as the polygon this slice's
        plane cuts out of its box, in screen space: exact at any zoom and on
        any anatomy, where a sampled box was only as precise as the
        anatomy's pixels (22 mm tall on 2 mm rows, with stepped edges)."""
        polygons = self.shape_polygons()
        if not polygons:
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setClipRect(self._image_rect)
        for layer, polygon in polygons:
            red, green, blue, alpha = views.shape_colour(layer)
            pen = QPen(QColor(red, green, blue, alpha),
                       max(1.0, float(layer.display.outline_px or 2.0)))
            pen.setCosmetic(True)
            pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            p.setPen(pen)
            p.setBrush(QColor(red, green, blue, int(alpha * render2d.SHAPE_FILL)))
            p.drawPolygon(polygon)
        p.restore()

    def _paint_crosshair(self, p: QPainter) -> None:
        grid = self._grid
        world = views.cursor_world(self.ctx.store)
        if grid is None or world is None:
            return
        c, r, _d = grid.world_to_pixel(world)
        if not (np.isfinite(c) and np.isfinite(r)):
            return
        pt = self.grid_to_screen(c, r)
        # Whole pixels: a 1 px line at a fractional position is drawn as two
        # half-strength rows and reads as a pale line instead of the colour.
        rect = self._image_rect.toAlignedRect().toRectF()
        cs = self.ctx.settings.crosshair
        color = QColor(cs.color)
        if not color.isValid():
            color = QColor(self.ctx.theme.crosshair)
        thickness = max(1, int(cs.thickness))
        gap = float(cs.gap)
        sq = max(4.0, min(rect.width(), rect.height()) * 0.025)
        x, y = float(int(pt.x())), float(int(pt.y()))

        def lines(pen: QPen) -> None:
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            if gap > 0:
                p.drawLine(QPointF(x, rect.top()), QPointF(x, y - gap))
                p.drawLine(QPointF(x, y + gap), QPointF(x, rect.bottom()))
                p.drawLine(QPointF(rect.left(), y), QPointF(x - gap, y))
                p.drawLine(QPointF(x + gap, y), QPointF(rect.right(), y))
            else:
                p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
                p.drawRect(QRectF(x - sq / 2, y - sq / 2, sq, sq))

        if thickness >= 2:
            halo = QPen(_HALO)
            halo.setWidthF(thickness + 2)
            lines(halo)
        pen = QPen(color)
        pen.setWidthF(thickness)
        lines(pen)

    def _paint_labels(self, p: QPainter) -> None:
        """The four orientation letters, each in its band OUTSIDE the image
        (``_margins`` reserves it), centred on the part of the image that is
        on screen and set against its edge, or against the edge of the image
        area when the image is zoomed past it. Never on the image, never on
        the caption or the colour bars."""
        grid = self._grid
        if grid is None:
            return
        letters = geometry.plane_labels(self.plane, grid)
        area = self._image_area()
        shown = self._image_rect.intersected(area)
        if shown.isEmpty():
            return
        p.setFont(fonts.font(fonts.LETTER_PX, bold=True))
        fm = p.fontMetrics()
        th = float(fm.height())
        gap = _LETTER_GAP
        cx, cy = shown.center().x(), shown.center().y()
        p.setPen(QColor(self.ctx.theme.label))
        for side in ("left", "right", "top", "bottom"):
            text = letters[side]
            tw = float(fm.horizontalAdvance(text))
            if side == "left":
                box = QRectF(shown.left() - gap - tw, cy - th / 2.0, tw, th)
            elif side == "right":
                box = QRectF(shown.right() + gap, cy - th / 2.0, tw, th)
            elif side == "top":
                box = QRectF(cx - tw / 2.0, shown.top() - gap - th, tw, th)
            else:
                box = QRectF(cx - tw / 2.0, shown.bottom() + gap, tw, th)
            self._letter_boxes[side] = box
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def letter_boxes(self) -> dict[str, QRectF]:
        """Where the orientation letters were last drawn, by side (for
        tests: none of them may touch the image)."""
        return dict(self._letter_boxes)

    def _paint_caption(self, p: QPainter) -> None:
        p.setPen(QColor(self.ctx.theme.caption))
        p.setFont(fonts.font(11))
        p.drawText(QRectF(0, 0, self.width(), float(fonts.px(15))),
                   Qt.AlignmentFlag.AlignCenter, self.plane.capitalize())

    def _paint_colorbar(self, p: QPainter) -> None:
        """One bar per visible scalar layer (``viz.colorbar``), stacked under
        the image: its title, the colour map over the window, round ticks,
        the threshold marked, a two-tailed map's negative half on the left."""
        bars = colorbar.bars_for(self.ctx.store)
        if not bars:
            return
        theme = self.ctx.theme
        # On the black surround in both themes: always light.
        dim = QColor(theme.canvas_dim)
        text = QColor(theme.canvas_text)
        width = max(40.0, min(self.width() - 32.0, 520.0))
        x0 = (self.width() - width) / 2.0
        strip = self._bar_px()
        y = self.height() - strip * len(bars) - 2
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        for spec in bars:
            p.setFont(fonts.font(10))
            p.setPen(text)
            fm = p.fontMetrics()
            th = float(fm.height())
            p.drawText(QRectF(x0, y, width, th), Qt.AlignmentFlag.AlignLeft,
                       fm.elidedText(spec.title, Qt.TextElideMode.ElideMiddle, int(width)))
            bar = QRectF(x0, y + th + 1, width, 7)
            pos = bar
            if spec.negative is not None:
                half = (width - 6) / 2.0
                neg = QRectF(x0, bar.top(), half, bar.height())
                pos = QRectF(x0 + half + 6, bar.top(), half, bar.height())
                self._draw_gradient(p, neg, spec.negative[0], spec.invert, reverse=True)
                p.setPen(dim)
                wide = float(fm.horizontalAdvance("-000000")) + 8
                p.drawText(QRectF(neg.left(), neg.bottom() + 1, wide, th),
                           Qt.AlignmentFlag.AlignLeft, f"-{colorbar.fmt(spec.negative[2], 0.01)}")
                p.drawText(QRectF(neg.right() - wide, neg.bottom() + 1, wide, th),
                           Qt.AlignmentFlag.AlignRight, f"-{colorbar.fmt(spec.negative[1], 0.01)}")
            self._draw_gradient(p, pos, spec.colormap, spec.invert)
            p.setPen(QPen(dim, 1))
            span = (spec.hi - spec.lo) or 1.0
            last_right = -1e9
            for value, label in spec.ticks:
                tx = pos.left() + (value - spec.lo) / span * pos.width()
                p.drawLine(QPointF(tx, pos.bottom()), QPointF(tx, pos.bottom() + 3))
                w = fm.horizontalAdvance(label) + 4
                left = min(max(tx - w / 2, pos.left() - 4), pos.right() - w + 4)
                if left > last_right + 4:
                    p.drawText(QRectF(left, pos.bottom() + 2, w, th),
                               Qt.AlignmentFlag.AlignHCenter, label)
                    last_right = left + w
            if spec.threshold is not None:
                # The threshold: values below are hidden (or faint).
                tx = pos.left() + (spec.threshold - spec.lo) / span * pos.width()
                tri = QPolygonF([QPointF(tx, pos.top() - 1), QPointF(tx - 4, pos.top() - 6),
                                 QPointF(tx + 4, pos.top() - 6)])
                p.setBrush(text)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawPolygon(tri)
                p.setBrush(Qt.BrushStyle.NoBrush)
            y += strip

    def _draw_gradient(self, p: QPainter, rect: QRectF, cmap: str, invert: bool,
                       reverse: bool = False) -> None:
        table = colormaps.lut(cmap, invert)
        if reverse:
            table = table[::-1]
        strip = np.ascontiguousarray(table[None, :, :].repeat(2, axis=0))
        img = QImage(strip.data, 256, 2, 256 * 4, QImage.Format.Format_RGBA8888)
        p.drawImage(rect, img)
        p.setPen(QPen(QColor(self.ctx.theme.grid), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(rect)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _mods(self, event) -> set[str]:
        m = event.modifiers()
        mods = set()
        if m & Qt.KeyboardModifier.ShiftModifier:
            mods.add("shift")
        if m & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            mods.add("ctrl")
        if m & Qt.KeyboardModifier.AltModifier:
            mods.add("alt")
        if "h" in self.ctx.held:
            mods.add("h")
        return mods

    def _tool_for(self, button: str, event) -> str:
        return inputmap.lookup("slice", button, self._mods(event),
                               self.ctx.settings.mousemap)

    def _focus_viewer(self) -> None:
        # Focus the VIEWER, not this canvas, so the viewer's shortcuts (scoped
        # to it and its children) apply after a click or a scroll here.
        w = self.parentWidget()
        while w is not None and not getattr(w, "_is_viz_viewer", False):
            w = w.parentWidget()
        if w is not None:
            w.setFocus(Qt.FocusReason.MouseFocusReason)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._focus_viewer()
        self.ctx.set_active_plane(self.plane)
        button = {Qt.MouseButton.LeftButton: "left", Qt.MouseButton.RightButton: "right",
                  Qt.MouseButton.MiddleButton: "middle"}.get(event.button())
        if button is None:
            return
        self._tool = self._tool_for(button, event)
        self._press_pos = event.position()
        self._last_pos = event.position()
        self._moved = False
        self._press_button = button
        if self._tool == "crosshair":
            self._crosshair_to(event.position())
        elif self._tool == "window_box":
            self._band = QRectF(event.position(), event.position())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._press_pos is None:
            return
        pos = event.position()
        if not self._moved and (pos - self._press_pos).manhattanLength() >= CLICK_SLOP:
            self._moved = True
        last = self._last_pos or pos
        dx, dy = pos.x() - last.x(), pos.y() - last.y()
        self._last_pos = pos
        tool = self._tool
        if tool == "crosshair":
            self._crosshair_to(pos)
        elif tool == "pan":
            ppm = max(self.pixels_per_mm(), 1e-6)
            state = self.ctx.scene.views.get(self.plane)
            zoom = state.zoom if state else 1.0
            self.ctx.run("view.pan", plane=self.plane, dx=dx / ppm * zoom, dy=dy / ppm * zoom)
        elif tool == "zoom":
            self.ctx.run("view.zoom", plane=self.plane, factor=float(np.exp(-dy / 200.0)))
        elif tool == "window":
            # NiiVue / radiology convention: horizontal = width, vertical = level.
            self.ctx.run("window.level_width", d_level=-dy / 300.0, d_width=dx / 300.0)
        elif tool == "window_box" and self._band is not None:
            self._band = QRectF(self._press_pos, pos).normalized()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        tool = self._tool
        if tool == "window_box" and self._band is not None:
            a = self.screen_to_grid(self._band.topLeft())
            b = self.screen_to_grid(self._band.bottomRight())
            self._band = None
            if a is not None and b is not None and self._moved:
                self.ctx.run("window.fit_box", plane=self.plane,
                             c0=a[0], r0=a[1], c1=b[0], r1=b[1])
            self.update()
        elif (tool in ("window", "pan") and getattr(self, "_press_button", "") == "right"
              and not self._moved):
            # A right CLICK (no drag) opens the menu; a right drag ran the tool.
            self._context_menu(event)
        self._press_pos = None
        self._last_pos = None
        self._tool = "none"

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            state = self.ctx.scene.views.get(self.plane)
            if state is not None and state.zoom > 1.0:
                self.ctx.run("view.reset", plane=self.plane)

    def _crosshair_to(self, pos: QPointF) -> None:
        coords = self.screen_to_grid(pos)
        if coords is None or self._grid is None:
            return
        rows, cols = self._grid.shape
        c = float(np.clip(coords[0], 0, cols - 1))
        r = float(np.clip(coords[1], 0, rows - 1))
        w = self._grid.pixel_to_world(c, r)
        # Keep the depth: a click inside a slice moves within that slice.
        world = views.cursor_world(self.ctx.store)
        if world is not None:
            n = self._grid.normal
            w = w + n * float(np.dot(np.asarray(world) - w, n))
        self.ctx.run("cursor.set_world", x=float(w[0]), y=float(w[1]), z=float(w[2]))

    def _context_menu(self, event) -> None:
        if self.ctx.run_action is None:
            return
        from .menus import viewer_context_menu

        viewer_context_menu(self.ctx, self, event.globalPosition().toPoint())

    def wheelEvent(self, event) -> None:  # noqa: N802
        self._focus_viewer()
        self.ctx.set_active_plane(self.plane)
        pd = event.pixelDelta()
        ad = event.angleDelta()
        use_pixel = not pd.isNull()
        dx = pd.x() if use_pixel else ad.x()
        dy = pd.y() if use_pixel else ad.y()
        thresh = WHEEL_PIXEL_STEP if use_pixel else WHEEL_ANGLE_STEP
        # Handle the observable event: a HORIZONTAL scroll arrived. X11 and
        # macOS turn Shift+wheel into one and drop the Shift flag, so asking
        # which device produced it is the bug CROSS_PLATFORM_RULES 5.2 names.
        horizontal = abs(dx) > abs(dy)
        gesture = "hwheel" if horizontal else "wheel"
        tool = inputmap.lookup("slice", gesture, self._mods(event), self.ctx.settings.mousemap)
        delta = dx if horizontal else dy
        if tool == "zoom":
            steps = delta / (thresh * 1.0)
            anchor = self._screen_mm_from_center(event.position())
            self.ctx.run("view.zoom", plane=self.plane, factor=float(1.15 ** steps), about=anchor)
            event.accept()
            return
        steps = self._accumulate((tool, self.plane), delta, thresh)
        if steps:
            if tool == "slice":
                # Scroll up = the previous slice, as the viewer always did.
                self.ctx.run("cursor.step", plane=self.plane, n=int(-steps))
            elif tool == "frame":
                # Scroll down / right = forward in time, as before.
                self.ctx.run("frame.step", n=int(-steps if not horizontal else steps))
        event.accept()

    def _screen_mm_from_center(self, pos: QPointF) -> tuple[float, float]:
        centre = self._image_area().center()
        cx, cy = centre.x(), centre.y()
        ppm = max(self.pixels_per_mm(), 1e-6)
        state = self.ctx.scene.views.get(self.plane)
        zoom = state.zoom if state else 1.0
        return ((pos.x() - cx) / ppm * zoom, (pos.y() - cy) / ppm * zoom)

    def _accumulate(self, key, delta: float, thresh: float) -> int:
        if not delta:
            return 0
        acc = self._wheel_acc.get(key, 0.0) + delta
        steps = int(acc / thresh)
        self._wheel_acc[key] = acc - steps * thresh
        return steps

    # ------------------------------------------------------------------
    def grab_image(self) -> Optional[QImage]:
        """The drawn view as an image (for screenshots and tests)."""
        return self.grab().toImage()


__all__ = ["SliceCanvas"]
