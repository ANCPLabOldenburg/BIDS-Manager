"""Round ``QComboBox`` popups and hover tooltips application-wide.

Two things are needed for a combo dropdown to look like the header project
menu (rounded corners, no square frame):

1. The QSS rule ``QComboBox { combobox-popup: 0; }`` (in ``theme.qss``)
   forces Qt's own ``QListView`` popup instead of the native macOS
   ``NSMenu``, which ignores every stylesheet rule. With it the popup view
   honours ``QComboBox QAbstractItemView`` (rounded list + rounded
   selection).

2. The popup still lives in a top-level window with a square OS frame +
   shadow behind the rounded view. This installs an application event
   filter that makes that container frameless + translucent + shadowless
   (the exact recipe the project menu uses), so only the rounded view
   shows. Geometry is captured and restored around the flag change so the
   popup stays anchored under its combo.

The same filter also rounds hover tooltips: the ``QToolTip`` QSS carries a
``border-radius`` but its ``QTipLabel`` window otherwise shows square OS
corners, so it gets the identical frameless + translucent + shadowless
treatment to stay consistent with the rounded GUI.

Defensive: any failure is swallowed so a dropdown or tooltip can never be
broken by this cosmetic pass. Call :func:`install` once, after the
QApplication and theme are set up.
"""

from __future__ import annotations

import logging
import sys

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QRegion

log = logging.getLogger(__name__)

_FLAG = "_bidsmgr_round_popup"

# Painted corner radius of each rounded surface, mirroring the ``border-radius``
# in theme.qss: 6px for ``QToolTip`` but 8px for ``QComboBox QAbstractItemView``
# and ``QMenu#rounded-menu``. They differ, so the mask must use the right base
# per surface (a 6px mask on an 8px popup would leave the 6..8 corner ring
# unclipped and black).
_TOOLTIP_RADIUS = 6
_POPUP_RADIUS = 8

# The mask is specified in logical pixels and Qt rescales it to device pixels.
# On HiDPI / fractionally-scaled Windows and X11 that rescale can land the
# mask's arc a device pixel or two INSIDE the painted border-radius arc,
# leaving a thin ring of the translucent (black) corner unclipped - the
# "still a bit of black" sliver. Cutting the mask a couple of logical pixels
# ROUNDER makes it a strict subset of the painted shape, so every pixel the
# mask keeps is painted and none of the black corner survives. The only cost
# is a cosmetically negligible clip of the outermost corner border.
_MASK_CUSHION = 2

# A translucent window only yields clean rounded corners where the platform
# COMPOSITES it. macOS always does, so the corners there are transparent and
# antialiased. On Windows and on X11/Wayland sessions without a compositor the
# pixels outside the rounded border are left unpainted and come out BLACK —
# the "picky black background" behind the rounded tooltip. Masking cuts those
# pixels out of the window entirely so nothing can show through; the trade-off
# is aliased (hard) edges, which is why macOS keeps the nicer translucent path.
_NEEDS_MASK = sys.platform != "darwin"


class _PopupRounder(QObject):
    """App event filter that rounds combo-popup and tooltip windows."""

    def eventFilter(self, obj, event):  # noqa: N802 - Qt signature
        try:
            etype = event.type()
            # Resize matters as much as Show: Qt REUSES one QTipLabel for every
            # tooltip and just re-texts/resizes it, often without a fresh Show.
            # A mask left at the previous tooltip's size would stop clipping
            # the corners of a narrower one (black corners return) or clip the
            # content of a wider one.
            if etype not in (QEvent.Type.Show, QEvent.Type.Resize):
                return False
            cls = obj.metaObject().className()
            if cls not in ("QComboBoxPrivateContainer", "QTipLabel"):
                return False
            if etype == QEvent.Type.Show and not obj.property(_FLAG):
                if cls == "QComboBoxPrivateContainer":
                    obj.setObjectName("combo-popup")
                # Same recipe for both so the QSS border-radius renders
                # without a square OS frame behind it.
                self._round(obj)
            # Tooltips paint at 6px, combo popups at 8px - mask each to match.
            base = _TOOLTIP_RADIUS if cls == "QTipLabel" else _POPUP_RADIUS
            _mask(obj, base)
        except Exception as exc:  # noqa: BLE001 - never break a popup/tooltip
            log.debug("popup round failed: %s", exc)
        return False

    @staticmethod
    def _round(obj) -> None:
        """Make *obj* a frameless, translucent, shadowless top-level window.

        Geometry is captured and restored around the flag change so the popup
        stays exactly where Qt placed it, then re-shown so the flags apply.
        """
        obj.setProperty(_FLAG, True)
        geo = obj.geometry()
        obj.setWindowFlags(
            obj.windowFlags()
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        obj.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        obj.setGeometry(geo)
        obj.show()  # re-show so the new window flags take effect


def _mask(obj, radius: int) -> None:
    """Clip *obj*'s window to a rounded rectangle a touch rounder than the paint.

    *radius* is the surface's PAINTED ``border-radius``; the mask itself is cut
    at ``radius + _MASK_CUSHION`` so it stays a strict subset of the painted
    shape (see ``_MASK_CUSHION``) and no black corner sliver can survive DPR
    rescaling.

    No-op on macOS, where the translucent window already composites clean
    antialiased corners. Elsewhere this removes the corner pixels from the
    window so an uncomposited desktop cannot paint them black.

    The region is assembled from two overlapping rectangles (a plus/cross
    shape) plus four ``Ellipse`` corner regions. That is the precise way to
    build a rounded-rect ``QRegion``: the earlier ``QPainterPath`` ->
    ``toFillPolygon().toPolygon()`` route flattened the arcs to a coarse,
    integer-rounded polygon, so a few desktop pixels still peeked through the
    corners as black on uncomposited Windows / X11. Rectangles + ellipses give
    exact corners with none of that residue.
    """
    if not _NEEDS_MASK:
        return
    rect = obj.rect()
    w, h = rect.width(), rect.height()
    if w <= 0 or h <= 0:
        return
    x, y = rect.x(), rect.y()
    r = max(0, min(radius + _MASK_CUSHION, w // 2, h // 2))
    if r == 0:
        obj.clearMask()
        return
    d = 2 * r
    Ellipse = QRegion.RegionType.Ellipse
    region = QRegion(x + r, y, w - d, h)             # full-height centre band
    region = region.united(QRegion(x, y + r, w, h - d))  # full-width centre band
    region = region.united(QRegion(x, y, d, d, Ellipse))                  # top-left
    region = region.united(QRegion(x + w - d, y, d, d, Ellipse))          # top-right
    region = region.united(QRegion(x, y + h - d, d, d, Ellipse))          # bottom-left
    region = region.united(QRegion(x + w - d, y + h - d, d, d, Ellipse))  # bottom-right
    obj.setMask(region)


_instance: _PopupRounder | None = None


def install(app) -> _PopupRounder:
    """Install the combo-popup / tooltip rounder on *app* (idempotent)."""
    global _instance
    if _instance is None:
        _instance = _PopupRounder(app)
        app.installEventFilter(_instance)
    return _instance


def round_menu(menu) -> None:
    """Give a ``QMenu`` rounded corners (the project-menu recipe).

    Frameless + translucent + no-shadow so the QSS ``QMenu#rounded-menu``
    border-radius renders without a square OS frame behind it. Use for any
    context / popup menu (e.g. the Welcome recents right-click menu).
    """
    menu.setObjectName("rounded-menu")
    menu.setWindowFlags(
        menu.windowFlags()
        | Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.NoDropShadowWindowHint
    )
    menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    # Same uncomposited-desktop guard as the tooltips: mask once the menu has
    # been laid out, so its corners can't paint black on Windows / X11.
    if _NEEDS_MASK:
        menu.aboutToShow.connect(lambda m=menu: _mask(m, _POPUP_RADIUS))


__all__ = ["install", "round_menu"]
