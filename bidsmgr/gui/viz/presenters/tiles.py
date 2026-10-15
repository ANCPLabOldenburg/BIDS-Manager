"""One page for every tiled layout: the views the scene's layout asks for,
placed where :func:`bidsmgr.viz.layouts.tile_rects` says.

The six fixed pages this replaces (one plane, three planes, 3-D, three planes
with 3-D, hero) each hard-coded an arrangement, so none could be changed. Here
the arrangement is DATA (``Scene.layout``: row, column, grid or automatic;
which planes, in what order; which view is large and how large) and this page
only places widgets. Changing a layout is a command, so it is undoable, saved
with a view or a scene, and remembered per kind of file.

The 3-D canvas is parented here once, before the window is shown, and only
shown or hidden afterwards: re-parenting a ``QOpenGLWidget`` into a visible
window makes Qt recreate the native window on Windows and Linux.

Children are placed by hand in ``resizeEvent`` (no ``QLayout`` subclass: see
CLAUDE.md on PyQt layout ownership).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtWidgets import QWidget

from ....viz import layouts, views
from ....viz.scene import PLANES
from ..canvases.slice import SliceCanvas

#: Gap between tiles; the hero's is wider, because it is a handle.
GAP = 2
HERO_GAP = 6


class TiledPage(QWidget):
    """The tiled views of the volume viewer."""

    def __init__(self, presenter) -> None:
        super().__init__()
        self.setObjectName("nifti-canvas")
        self._presenter = presenter
        self.ctx = presenter.ctx
        self._slices: dict[str, SliceCanvas] = {}
        self._render: Optional[QWidget] = None
        self.tiles: list[str] = []
        self.hero = ""
        self._rects: list[QRect] = []
        self._drag = False
        self.setMouseTracking(True)

    # -- views -----------------------------------------------------------
    def slice_canvas(self, plane: str) -> SliceCanvas:
        canvas = self._slices.get(plane)
        if canvas is None:
            canvas = SliceCanvas(self.ctx, plane, caption=True)
            canvas.setParent(self)
            canvas.setVisible(False)
            self._slices[plane] = canvas
        return canvas

    def set_render(self, render: QWidget) -> None:
        """Adopt the 3-D canvas (once, before the window shows)."""
        self._render = render
        render.setParent(self)
        render.setVisible(False)

    def widget_for(self, tile: str) -> Optional[QWidget]:
        if tile == layouts.RENDER:
            return self._render
        return self.slice_canvas(tile)

    # -- arrangement -----------------------------------------------------
    def apply(self) -> None:
        """Show the views the scene asks for and place them."""
        scene = self.ctx.scene
        render_ok = self._render is not None and self._presenter.render_allowed()
        self.tiles, self.hero = layouts.tiles_for(scene.mode, scene.plane, scene.layout,
                                                  render_ok)
        wanted = set(self.tiles)
        for plane, canvas in self._slices.items():
            if plane not in wanted:
                canvas.setVisible(False)
        if self._render is not None and layouts.RENDER not in wanted:
            self._render.setVisible(False)
        for tile in self.tiles:
            w = self.widget_for(tile)
            if isinstance(w, SliceCanvas):
                w.set_caption(len(self.tiles) > 1)
        self._place()
        for tile in self.tiles:
            w = self.widget_for(tile)
            if w is not None and not w.isVisible():
                w.setVisible(True)

    def _aspect(self) -> float:
        """Width over height of the planes shown (their median), so the
        automatic arrangement fits the images rather than the boxes."""
        ratios = []
        for tile in self.tiles:
            if tile not in PLANES:
                ratios.append(1.0)
                continue
            grid = views.grid(self.ctx.store, tile)
            if grid is None:
                continue
            w, h = grid.extent_mm
            if w > 0 and h > 0:
                ratios.append(w / h)
        if not ratios:
            return 1.0
        ratios.sort()
        return ratios[len(ratios) // 2]

    def _place(self) -> None:
        lay = self.ctx.scene.layout
        hero = bool(self.hero) and len(self.tiles) > 1
        rects = layouts.tile_rects(
            len(self.tiles), self.width(), self.height(), arrangement=lay.arrangement,
            aspect=self._aspect(), hero=hero, hero_fraction=lay.hero_fraction,
            hero_side=lay.hero_side, gap=HERO_GAP if hero else GAP,
        )
        self._rects = [QRect(int(round(x)), int(round(y)), int(round(w)), int(round(h)))
                       for x, y, w, h in rects]
        for tile, rect in zip(self.tiles, self._rects):
            w = self.widget_for(tile)
            if w is not None:
                w.setGeometry(rect)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._place()

    def rects(self) -> dict[str, QRect]:
        """Where each tile is (tests)."""
        return dict(zip(self.tiles, self._rects))

    # -- the hero's handle -----------------------------------------------
    def _on_handle(self, x: float, y: float) -> bool:
        if not self.hero or len(self._rects) < 2:
            return False
        big = self._rects[0]
        if self.ctx.scene.layout.hero_side == "top":
            return big.bottom() <= y <= big.bottom() + HERO_GAP + 1
        return big.right() <= x <= big.right() + HERO_GAP + 1

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        top = self.ctx.scene.layout.hero_side == "top"
        if self._drag:
            f = pos.y() / max(self.height(), 1) if top else pos.x() / max(self.width(), 1)
            self.ctx.run("layout.set", hero_fraction=float(f))
            return
        if self._on_handle(pos.x(), pos.y()):
            self.setCursor(Qt.CursorShape.SplitVCursor if top else Qt.CursorShape.SplitHCursor)
        else:
            self.unsetCursor()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton and self._on_handle(pos.x(), pos.y()):
            self._drag = True
            self.ctx.store.begin_gesture()

    def mouseReleaseEvent(self, _event) -> None:  # noqa: N802
        if self._drag:
            self._drag = False
            self.ctx.store.end_gesture()


__all__ = ["GAP", "HERO_GAP", "TiledPage"]
