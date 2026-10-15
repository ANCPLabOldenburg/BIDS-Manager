"""Building the QApplication the way every BIDS Manager window expects it.

Shared by the full GUI (``bidsmgr``) and the standalone viewer
(``bidsmgr-view``), so the two cannot drift: the OpenGL format the 3-D view
needs, the Fusion style, the brand icon, the bundled typefaces, the persisted
theme and font scale, and the rounded combo popups. Everything here has to happen BEFORE the first
window is built, and most of it before the QApplication itself exists.
"""

from __future__ import annotations

import sys
from typing import Optional


def create_application(theme: Optional[str] = None):
    """Return ``(app, theme_manager)``, configured and themed.

    ``theme`` is ``"dark"`` or ``"light"``; ``None`` restores the theme the
    user last chose.
    """
    # Linux: make sure libxcb-cursor0 is reachable (Qt 6.5+ refuses to load
    # the xcb plugin without it).
    from ..util.qt_platform import prepare as _prepare_qt_platform

    _prepare_qt_platform()

    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    # An OpenGL 3.3 core default surface format BEFORE the QApplication is
    # constructed, so the 3-D ray-caster gets a context its #version 330
    # shaders compile against (macOS's compatibility profile stops at GL 2.1).
    # Shared contexts let the ray-caster survive a detached panel being
    # re-docked. Both must precede the QApplication; harmless for the raster
    # widgets.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    from .viz.canvases.render import request_gl_format

    request_gl_format()

    app = QApplication.instance() or QApplication(sys.argv[:1])
    # QSettings keys these to find the per-user config file on macOS, Linux
    # and Windows; set once, every QSettings() agrees.
    app.setOrganizationName("bidsmgr")
    app.setApplicationName("bidsmgr")
    app.setStyle("Fusion")

    # Title bar / taskbar / alt-tab icon on Linux and Windows. macOS reads
    # its icon from the .app bundle; the call is a no-op there.
    from .app_icon import set_app_icon

    set_app_icon(app)

    # The bundled typefaces before anything is styled: every widget inherits
    # the application font, so it must be Inter before the first one exists.
    from . import typefaces
    from .theme_manager import BASE_FONT_PIXEL_SIZE

    typefaces.load()
    app.setFont(typefaces.ui_font(BASE_FONT_PIXEL_SIZE))

    from .app_settings import AppSettings
    from .theme_manager import ThemeManager

    persisted = AppSettings.load()
    manager = ThemeManager(app, font_scale=persisted.font_scale,
                           appearance=persisted.appearance())
    manager.apply(theme or persisted.theme)

    # Every viewer, in any window or dialog, takes its colours from one hub
    # (pyqtgraph and GL read no QSS). Later toggles are published by the
    # main window, which owns the theme switch.
    from .viz.bridge import ThemeHub

    ThemeHub.instance().publish(manager.palette, manager.name)

    # Round every QComboBox dropdown. Safe no-op if it ever fails.
    from .combo_popup import install as install_combo_popup_rounder

    install_combo_popup_rounder(app)
    return app, manager


__all__ = ["create_application"]
