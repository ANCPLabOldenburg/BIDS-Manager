"""Right-click menus of the canvases, built from the action table."""

from __future__ import annotations

from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QMenu, QWidget

#: What a right click on a slice offers, in order; "" is a separator.
SLICE_MENU = (
    "view.axial", "view.coronal", "view.sagittal", "", "view.multi", "view.3d",
    "view.combo", "view.graph", "", "window.robust", "window.full",
    "window.invert", "view.nearest", "", "view.labels", "view.radiological", "view.world",
    "view.colorbar", "view.crosshair", "view.reset", "", "export.screenshot",
    "help.shortcuts",
)


def popup_menu(parent: QWidget) -> QMenu:
    """A rounded popup menu (the app's recipe). Every menu the viewer opens
    is built here, and every submenu through :func:`submenu`, so none can
    come out with square corners."""
    from ..combo_popup import round_menu

    menu = QMenu(parent)
    round_menu(menu)
    return menu


def submenu(menu: QMenu, title: str) -> QMenu:
    """A rounded submenu of ``menu``."""
    from ..combo_popup import add_submenu

    return add_submenu(menu, title)


def viewer_context_menu(ctx, parent: QWidget, at: QPoint) -> None:
    viewer = parent
    while viewer is not None and not getattr(viewer, "_is_viz_viewer", False):
        viewer = viewer.parentWidget()
    if viewer is None:
        return
    manager = viewer.action_manager
    menu = popup_menu(parent)
    for action_id in SLICE_MENU:
        if not action_id:
            menu.addSeparator()
            continue
        act = manager.actions.get(action_id)
        if act is not None and act.isEnabled():
            menu.addAction(act)
    menu.exec(at)


__all__ = ["SLICE_MENU", "popup_menu", "submenu", "viewer_context_menu"]
