"""The mosaic page: many slices in one figure, from a line of text.

Each tile is a plane at a millimetre position, coloured exactly as the slice
views colour it (same layers, windows and colour maps), so a mosaic is a
figure of what the viewer shows, not a separate rendering of it. Exported
with the screenshot action like any canvas.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ....viz import views
from ....viz.compute import geometry, mosaic
from ....viz.scene import PLANE_AXIS
from ..bridge import connect_while_alive
from ..context import ViewerContext
from .. import fonts

log = logging.getLogger(__name__)


#: A pixel is part of the head when any channel is brighter than this.
CONTENT_LEVEL = 12
#: Room left around the head when the tiles are cropped, as a fraction.
CROP_MARGIN = 0.04


def content_box(rgba: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    """The part of a tile that is not empty field of view, as fractions
    ``(left, top, right, bottom)``; None when the whole tile is empty."""
    lum = rgba[..., :3].max(axis=2)
    rows = np.flatnonzero((lum > CONTENT_LEVEL).any(axis=1))
    cols = np.flatnonzero((lum > CONTENT_LEVEL).any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    h, w = lum.shape
    return (cols[0] / w, rows[0] / h, (cols[-1] + 1) / w, (rows[-1] + 1) / h)


def shared_crop(boxes) -> tuple[float, float, float, float]:
    """One crop for every tile of a plane, so they stay the same size and a
    structure sits at the same place in each: the union of their contents,
    with a margin."""
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return (0.0, 0.0, 1.0, 1.0)
    left = min(b[0] for b in boxes) - CROP_MARGIN
    top = min(b[1] for b in boxes) - CROP_MARGIN
    right = max(b[2] for b in boxes) + CROP_MARGIN
    bottom = max(b[3] for b in boxes) + CROP_MARGIN
    return (max(left, 0.0), max(top, 0.0), min(right, 1.0), min(bottom, 1.0))


class MosaicCanvas(QWidget):
    """Draws the scene's mosaic line."""

    def __init__(self, ctx: ViewerContext, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setMinimumSize(60, 60)
        self._cache: dict = {}
        ctx.qstore.changed.connect(self._on_changed)
        connect_while_alive(ctx.theme_hub.changed, self, lambda w, _t: w.update())

    def _on_changed(self, paths) -> None:
        if self.isVisible() and any(
            not p.startswith(("render.", "clips", "graph", "cursor")) for p in paths
        ):
            self._cache.clear()
            self.update()

    def tile_rows(self, spec=None) -> list:
        """The figure's rows: ``(tile, image, grid, content box)`` each."""
        spec = spec if spec is not None else mosaic.parse(self.ctx.scene.mosaic)
        rows = []
        for row in spec.rows:
            items = []
            for tile in row:
                got = self._tile_image(tile)
                if got is not None:
                    items.append((tile, *got))
            if items:
                rows.append(items)
        return rows

    def crops(self, rows) -> dict[str, tuple[float, float, float, float]]:
        """The crop of each plane in the figure (the whole tile when the
        mosaic is not cropped)."""
        crop = self.ctx.scene.mosaic_build.crop
        by_plane: dict[str, list] = {}
        for items in rows:
            for tile, _img, _grid, box in items:
                by_plane.setdefault(tile.plane, []).append(box)
        return {plane: shared_crop(boxes) if crop else (0.0, 0.0, 1.0, 1.0)
                for plane, boxes in by_plane.items()}

    def _tile_image(self, tile: mosaic.Tile):
        key = (tile.plane, round(tile.mm, 3))
        if key in self._cache:
            return self._cache[key]
        store = self.ctx.store
        layer, src = views.base(store)
        if src is None:
            return None
        world = np.asarray(views.cursor_world(store), dtype=float)
        world[PLANE_AXIS[tile.plane]] = tile.mm
        d = self.ctx.scene.display
        grid = geometry.grid_for(
            tile.plane, affine=src.affine, spatial=src.spatial, zooms=src.zooms3,
            orientation=views.orientation(src), cursor_world=world, space=d.space,
            ras=d.ras, radiological=d.radiological,
        )
        # The slice views' own composition (render2d), so a mosaic is a
        # figure of what the viewer shows; shapes are drawn on top, exactly.
        from ....viz import render2d

        got = render2d.slice_rgba(store, tile.plane, transparent=self.ctx.transparent,
                                  grid=grid)
        if got is None:
            return None
        buf = got.rgba
        img = QImage(buf.data, buf.shape[1], buf.shape[0], buf.shape[1] * 4,
                     QImage.Format.Format_RGBA8888).copy()
        self._cache[key] = (img, grid, content_box(buf))
        return self._cache[key]

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        theme = self.ctx.theme
        if not self.ctx.transparent:
            p.fillRect(self.rect(), QColor(theme.background))
        spec = mosaic.parse(self.ctx.scene.mosaic)
        rows = self.tile_rows(spec)
        if not rows:
            p.setPen(QColor(theme.canvas_dim))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Build a mosaic in the advanced controls (Mosaic section)")
            p.end()
            return
        overlap = spec.overlap
        crops = self.crops(rows)

        def kept_mm(tile, grid):
            left, top, right, bottom = crops[tile.plane]
            return grid.extent_mm[0] * (right - left), grid.extent_mm[1] * (bottom - top)

        # Size in millimetres, then fit the whole figure to the widget.
        row_sizes = []
        for items in rows:
            sizes = [kept_mm(t, g) for t, _i, g, _b in items]
            widths = [w for w, _h in sizes]
            height = max(h for _w, h in sizes)
            total = sum(widths) - overlap * sum(widths[1:])
            row_sizes.append((total, height, widths))
        fig_w = max(w for w, _h, _ws in row_sizes)
        fig_h = sum(h for _w, h, _ws in row_sizes)
        label_h = 14 if spec.labels else 0
        avail_w = max(1.0, self.width() - 8.0)
        avail_h = max(1.0, self.height() - 8.0 - label_h * len(rows))
        ppm = min(avail_w / max(fig_w, 1e-6), avail_h / max(fig_h, 1e-6))
        # Centred both ways: a wide window leaves room at the sides, a tall
        # one above and below.
        y = 4.0 + max(0.0, (avail_h - fig_h * ppm) / 2.0)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        every = [item[:3] for items in rows for item in items]
        for items, (row_w, row_h, widths) in zip(rows, row_sizes):
            x = 4.0 + (avail_w - row_w * ppm) / 2.0
            for k, (tile, img, grid, _box) in enumerate(items):
                w_mm, h_mm = kept_mm(tile, grid)
                rect = QRectF(x, y + (row_h - h_mm) * ppm / 2.0, w_mm * ppm, h_mm * ppm)
                # The whole tile, placed so its kept part lands on ``rect``;
                # everything is drawn against it and clipped to ``rect``.
                left, top, right, bottom = crops[tile.plane]
                full_w = rect.width() / max(right - left, 1e-6)
                full_h = rect.height() / max(bottom - top, 1e-6)
                full = QRectF(rect.left() - left * full_w, rect.top() - top * full_h,
                              full_w, full_h)
                p.save()
                p.setClipRect(rect)
                p.drawImage(full, img)
                self._paint_shapes(p, full, grid)
                if tile.cross:
                    self._paint_cross(p, full, grid, tile, every)
                p.restore()
                if spec.labels:
                    p.setPen(QColor(theme.canvas_dim))
                    p.setFont(fonts.font(10))
                    axis = "xyz"[PLANE_AXIS[tile.plane]]
                    p.drawText(QRectF(rect.left(), rect.bottom() + 1, rect.width(), label_h),
                               Qt.AlignmentFlag.AlignCenter,
                               f"{axis} = {tile.mm:g} mm")
                x += widths[k] * ppm * (1.0 - overlap)
            y += row_h * ppm + label_h
        p.end()

    def _paint_shapes(self, p: QPainter, rect: QRectF, grid) -> None:
        """The MRS voxel (any shape layer) cut by this tile's plane, exact."""
        from PyQt6.QtCore import QPointF
        from PyQt6.QtGui import QPolygonF

        from ....viz import render2d
        from ....viz.compute.shapes import section

        rows, cols = grid.shape
        for layer, src in views.shape_layers(self.ctx.store):
            poly = section(src.box, grid.origin, grid.normal)
            if not len(poly):
                continue
            pts = []
            for w in poly:
                c, r, _d = grid.world_to_pixel(w)
                pts.append(QPointF(rect.left() + (c + 0.5) / cols * rect.width(),
                                   rect.top() + (r + 0.5) / rows * rect.height()))
            red, green, blue, alpha = views.shape_colour(layer)
            pen = QPen(QColor(red, green, blue, alpha), 2.0)
            pen.setCosmetic(True)
            p.save()
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setClipRect(rect, Qt.ClipOperation.IntersectClip)
            p.setPen(pen)
            p.setBrush(QColor(red, green, blue, int(alpha * render2d.SHAPE_FILL)))
            p.drawPolygon(QPolygonF(pts))
            p.restore()

    def _paint_cross(self, p: QPainter, rect: QRectF, grid, tile, items) -> None:
        pen = QPen(QColor(self.ctx.settings.crosshair.color))
        pen.setCosmetic(True)
        p.setPen(pen)
        rows, cols = grid.shape
        for other, _img, _g in items:
            if other.plane == tile.plane:
                continue
            world = np.asarray(grid.origin, dtype=float).copy()
            world[PLANE_AXIS[other.plane]] = other.mm
            c, r, _d = grid.world_to_pixel(world)
            axis = PLANE_AXIS[other.plane]
            if abs(grid.col_vec[axis]) > abs(grid.row_vec[axis]) and np.isfinite(c):
                x = rect.left() + (c + 0.5) / cols * rect.width()
                p.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            elif np.isfinite(r):
                y = rect.top() + (r + 0.5) / rows * rect.height()
                p.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))


class MosaicPage(QWidget):
    """The mosaic, the figure alone: it is built in the controls column's
    Mosaic section (or written there as a line, for the grammar's extras)."""

    def __init__(self, ctx: ViewerContext, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.canvas = MosaicCanvas(ctx)
        lay.addWidget(self.canvas, 1)


__all__ = ["MosaicCanvas", "MosaicPage"]
