"""Auto-redraw QLabel with click + drag reporting.

Used by :class:`bidsmgr.gui.widgets.nifti_viewer_pane.NiftiViewerPane`
as the canvas for the rendered 2-D slice. The label needs three
behaviours the stock ``QLabel`` doesn't give us:

* re-rasterise the slice when the label is resized (so the pixmap
  follows splitter / window changes without leaving white margins);
* report mouse presses back to the pane so it can move the crosshair
  to the clicked voxel and refresh the voxel-value readout;
* report mouse *drags* (button held while moving) so the user can
  scrub the crosshair continuously instead of having to release and
  re-click for every voxel.

Port of ``_AutoUpdateLabel`` + ``_ImageLabel`` from
``BIDS-Manager/bids_manager/gui.py`` (≈ lines 434-457), with the
drag handler added.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QLabel


class ImageLabel(QLabel):
    """QLabel that calls back on resize, press and drag.

    ``click_fn`` fires on mouse press AND on subsequent
    :meth:`mouseMoveEvent` calls while the left button is held —
    the pane can treat both the same way, since updating the
    crosshair is idempotent.
    """

    def __init__(
        self,
        update_fn: Optional[Callable[[], None]] = None,
        click_fn: Optional[Callable[[QMouseEvent], None]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._update_fn = update_fn
        self._click_fn = click_fn

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        if callable(self._update_fn):
            self._update_fn()

    def mousePressEvent(self, event: QMouseEvent):  # type: ignore[override]
        if (
            callable(self._click_fn)
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._click_fn(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):  # type: ignore[override]
        # Qt only sends move events while a button is held when mouse
        # tracking is off (the default). Restrict to the left button so
        # right-click drags / middle-button pans don't move the
        # crosshair.
        if (
            callable(self._click_fn)
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self._click_fn(event)
        super().mouseMoveEvent(event)


__all__ = ["ImageLabel"]
