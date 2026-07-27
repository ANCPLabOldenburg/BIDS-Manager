"""Round ``QComboBox`` popups, hover tooltips and menus application-wide.

Every rounded popup in the app uses ONE recipe, the same one the project
switcher menu sets on itself in ``main_window``: make the top-level popup
window **frameless + translucent + shadowless** so the QSS ``border-radius``
shows without a square OS frame or a black background behind it.

The subtlety is *when* the translucent attribute is set. A window only gets a
real alpha channel on Windows and X11 if transparency is requested **before the
OS window is created**; you cannot add an alpha channel to a window that already
exists there (macOS composites every window with alpha regardless, which is why
it never showed the problem). The menus get this for free: we build the
``QMenu`` and set the attribute on it before it is ever shown, so its window is
born translucent.

Qt's own combo dropdown (``QComboBoxPrivateContainer``) and hover tooltip
(``QTipLabel``) are created internally, so an application event filter is the
only way to reach them. The key is to reach them on the ``Polish`` event, which
Qt delivers **before** the native window is created (verified: the widget's
``WA_WState_Created`` attribute is False at Polish and True by Show), not on the
``Show`` event, which fires after. Setting the flags at Polish means the window
is created translucent, with clean transparent corners on every platform, and
needs no clip mask.

An earlier version instead set the attribute at Show (too late off-macOS, so the
corners came out black) and tried to paper over it with a rounded clip mask,
which also mis-clipped tooltips whose size was not laid out yet. Both are gone:
one Polish-time recipe now serves combos, tooltips and menus alike.

The QSS rule ``QComboBox { combobox-popup: 0; }`` (in ``theme.qss``) is still
required so Qt uses its own stylesheet-aware ``QListView`` popup instead of the
native macOS ``NSMenu``.

Defensive: any failure is swallowed so a dropdown or tooltip can never be
broken by this cosmetic pass. Call :func:`install` once, right after the
QApplication is created and before any window is built (so no combo or tooltip
is polished before the filter is watching).
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
            # Polish, NOT Show: Polish is delivered BEFORE the widget's native
            # window is created, so the translucent attribute we set here is in
            # place when the OS creates the window and it gets a real alpha
            # channel (clean corners on Windows / X11). At Show the window
            # already exists and translucency can no longer be added off-macOS.
            if event.type() != QEvent.Type.Polish or obj.property(_FLAG):
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
        """Make *obj* frameless + translucent + shadowless before its window exists.

        Called on Polish, so no native window has been created yet: we just set
        the flags and the attribute and let Qt create the window translucent. No
        geometry capture and no re-show are needed (the widget is not visible
        yet), unlike the old Show-time path.
        """
        obj.setProperty(_FLAG, True)
        obj.setWindowFlags(
            obj.windowFlags()
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        obj.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)


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
    border-radius renders without a square OS frame behind it. Call this right
    after building the menu and before it is shown, so its window is created
    translucent. Use for any context / popup menu (e.g. the Welcome recents
    right-click menu).
    """
    menu.setObjectName("rounded-menu")
    menu.setWindowFlags(
        menu.windowFlags()
        | Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.NoDropShadowWindowHint
    )
    menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


__all__ = ["install", "round_menu"]
