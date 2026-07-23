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

from PyQt6.QtCore import QEvent, QObject, Qt

log = logging.getLogger(__name__)

_FLAG = "_bidsmgr_round_popup"


class _PopupRounder(QObject):
    """App event filter that rounds combo-popup and tooltip windows."""

    def eventFilter(self, obj, event):  # noqa: N802 - Qt signature
        try:
            if event.type() != QEvent.Type.Show or obj.property(_FLAG):
                return False
            cls = obj.metaObject().className()
            if cls == "QComboBoxPrivateContainer":
                obj.setObjectName("combo-popup")
                self._round(obj)
            elif cls == "QTipLabel":
                # Hover tooltips: same recipe so the QToolTip border-radius
                # renders without square OS corners, matching the rounded GUI.
                self._round(obj)
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


__all__ = ["install", "round_menu"]
