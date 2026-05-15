"""``QMainWindow`` shell hosting the Inspector layout.

M-CLI scope: a minimal window with just the Converter view + a status
bar + a theme toggle. The Editor view, top header view switcher, and
project menus land in later milestones.

Reference: ``inspector_proto/proto.py`` ``MainWindow``.
"""

from __future__ import annotations

import logging
from typing import Optional

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from ..project import Project
from .converter_panel import ConverterPanel
from .editor_panel import EditorPanel
from .theme_manager import ThemeManager

log = logging.getLogger(__name__)


class _ClickableLabel(QLabel):
    """A QLabel that emits :pyattr:`clicked` on a left-click release.

    Used for the brand logo + wordmark in :class:`_TopHeader` — both
    open the About dialog when clicked. Cursor flips to a pointing
    hand so it reads as an actionable element.
    """

    clicked = pyqtSignal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event):  # noqa: N802 — Qt naming
        if event.button() == Qt.MouseButton.LeftButton \
                and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class _TopHeader(QFrame):
    """Brand header with a Converter/Editor pill switcher and theme toggle.

    Mirrors ``inspector_proto/proto.py``'s ``TopHeader``: brand on the
    left, two checkable view pills, theme toggle on the right.

    The brand logo and wordmark are clickable — emit
    :pyattr:`about_requested` so :class:`MainWindow` can pop the
    :class:`AboutDialog`.
    """

    view_changed = pyqtSignal(str)  # "converter" | "editor"
    about_requested = pyqtSignal()

    def __init__(self, theme: ThemeManager, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("top-header")
        self.setFixedHeight(48)
        h = QHBoxLayout(self)
        h.setContentsMargins(14, 6, 14, 6)
        h.setSpacing(10)

        # Brand logo + name. The bundled PNG ships in
        # ``bidsmgr/gui/assets/logo.png``; we fall back to a gradient-B
        # placeholder if the asset can't be loaded for any reason
        # (e.g. running from a partial source tree). Both the logo and
        # the wordmark are clickable — they pop the About dialog.
        self._logo = _ClickableLabel()
        self._logo.setFixedSize(40, 36)
        self._logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._logo.setToolTip("About BIDS-Manager")
        self._logo.clicked.connect(self.about_requested.emit)
        self._apply_logo_pixmap(theme.palette)
        name = _ClickableLabel("BIDS-Manager")
        name.setObjectName("brand-name")
        name.setToolTip("About BIDS-Manager")
        name.clicked.connect(self.about_requested.emit)
        h.addWidget(self._logo)
        h.addWidget(name)
        h.addSpacing(12)

        # View pills — Converter | Editor. The button group keeps the
        # two checkable buttons mutually exclusive; ``idClicked`` fires
        # whichever was just selected so we can re-emit a typed signal.
        self._converter_btn = QPushButton("Converter")
        self._converter_btn.setObjectName("view-pill")
        self._converter_btn.setCheckable(True)
        self._converter_btn.setChecked(True)
        self._editor_btn = QPushButton("Editor")
        self._editor_btn.setObjectName("view-pill")
        self._editor_btn.setCheckable(True)
        self._pill_group = QButtonGroup(self)
        self._pill_group.setExclusive(True)
        self._pill_group.addButton(self._converter_btn, 0)
        self._pill_group.addButton(self._editor_btn, 1)
        self._pill_group.idClicked.connect(self._on_pill_clicked)
        h.addWidget(self._converter_btn)
        h.addWidget(self._editor_btn)

        h.addStretch(1)

        self._theme = theme
        # Icon-only toggle. ``sun`` glyph in dark mode (click to lighten),
        # ``moon`` glyph in light mode (click to darken). Re-tinted in
        # ``repaint_for_palette`` on every theme swap.
        from . import icons
        self._theme_btn = QPushButton()
        self._theme_btn.setObjectName("theme-toggle")
        self._theme_btn.setToolTip("Toggle light / dark theme")
        self._theme_btn.setFixedSize(32, 28)
        icons.apply_button(
            self._theme_btn,
            "sun" if theme.name == "dark" else "moon",
        )
        self._theme_btn.clicked.connect(self._on_toggle)
        h.addWidget(self._theme_btn)

    def set_active_view(self, view: str) -> None:
        """Programmatically toggle the pills (no signal emitted)."""
        target = self._editor_btn if view == "editor" else self._converter_btn
        target.setChecked(True)

    def _on_pill_clicked(self, idx: int) -> None:
        self.view_changed.emit("editor" if idx == 1 else "converter")

    def _on_toggle(self) -> None:
        from .app_settings import AppSettings
        from . import icons
        new = self._theme.toggle()
        icons.apply_button(self._theme_btn, "sun" if new == "dark" else "moon")
        AppSettings.remember_theme(new)

    def _apply_logo_pixmap(self, pal: dict) -> None:
        """Load the bundled PNG into the logo label.

        The PNG is drawn dark-on-transparent for a light background.
        On a dark theme we invert the RGB channels (keeping alpha) so
        the same artwork reads as light-on-transparent against the
        dark surface. Falls back to a gradient-B if the asset can't
        be loaded.
        """
        png = Path(__file__).parent / "assets" / "logo.png"
        if png.exists():
            img = QImage(str(png))
            if not img.isNull():
                if self._is_dark_theme(pal):
                    # ``InvertRgb`` flips R/G/B; alpha is preserved so
                    # the transparent background stays transparent.
                    img.invertPixels(QImage.InvertMode.InvertRgb)
                pix = QPixmap.fromImage(img)
                self._logo.setPixmap(pix.scaledToHeight(
                    36,
                    Qt.TransformationMode.SmoothTransformation,
                ))
                # Drop any leftover stylesheet from a previous gradient
                # render so the transparent PNG sits flat.
                self._logo.setStyleSheet("")
                self._logo.setText("")
                return
        # Fallback path — keep the GUI usable even without the asset.
        self._logo.setText("B")
        self._logo.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            f"stop:0 {pal['accent']}, stop:1 {pal['purple']});"
            "color: white; border-radius: 6px; font-weight: 700;"
        )

    @staticmethod
    def _is_dark_theme(pal: dict) -> bool:
        """Heuristic: average RGB of ``pal['bg']`` below 128 → dark."""
        bg = pal.get("bg", "#000000").lstrip("#")
        if len(bg) < 6:
            return False
        r = int(bg[0:2], 16)
        g = int(bg[2:4], 16)
        b = int(bg[4:6], 16)
        return (r + g + b) / 3 < 128

    def repaint_for_palette(self, pal: dict) -> None:
        """Reload the logo under the new palette (inverts when dark)."""
        self._apply_logo_pixmap(pal)
        from . import icons
        icons.apply_button(
            self._theme_btn,
            "sun" if self._theme.name == "dark" else "moon",
        )


class MainWindow(QMainWindow):
    """The single application window. Hosts a :class:`ConverterPanel`.

    Constructed with a :class:`ThemeManager` already bound to the
    ``QApplication``; the window does not call ``theme.apply`` itself
    so the caller can pick the initial theme.
    """

    def __init__(
        self,
        theme: ThemeManager,
        project: Optional[Project] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("BIDS-Manager")
        self.resize(1480, 900)

        self._theme = theme
        self._project = project

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self._header = _TopHeader(theme)
        v.addWidget(self._header)

        # Stacked content — Converter and Editor share the window and
        # are swapped via the header pills. The stack keeps both alive
        # so theme listeners stay registered and panel state survives
        # switching back and forth.
        self.stack = QStackedWidget()
        self.converter = ConverterPanel(project=project)
        self.editor = EditorPanel()
        self.stack.addWidget(self.converter)   # index 0 → "converter"
        self.stack.addWidget(self.editor)      # index 1 → "editor"
        v.addWidget(self.stack, 1)

        self._header.view_changed.connect(self._on_view_changed)
        self._header.about_requested.connect(self._show_about_dialog)

        # Restore the user's last view. Pills are syncronised silently
        # so we don't fire a redundant ``view_changed`` on startup.
        from .app_settings import AppSettings
        settings = AppSettings.load()
        self._apply_active_view(settings.active_view, persist=False)

        # Status bar — forwards the Converter's log messages so the
        # user sees scan / convert progress. The bottom-right corner
        # carries the installed version label + "Check updates" button
        # via :func:`update_widgets.attach_update_widgets`.
        sb = QStatusBar()
        sb.setSizeGripEnabled(False)
        self._status_text = QLabel("Ready")
        sb.addWidget(self._status_text, 1)
        self.setStatusBar(sb)

        self.converter.log_message.connect(self._set_status)
        self.editor.log_message.connect(self._set_status)

        # Bottom-right: "vX.Y.Z" + "Check updates". Wrapped in try/except
        # because a broken update path must never prevent the GUI from
        # opening (offline machines, missing certifi, etc.).
        try:
            from .update_widgets import attach_update_widgets, run_startup_check
            attach_update_widgets(sb, self)
            run_startup_check(self)
        except Exception as exc:
            log.warning("update check disabled: %s", exc)

        # Subscribe to palette changes so widgets whose colors are read
        # at construction time (delegate paints, inline ``setStyleSheet``)
        # repaint with the new palette without requiring an app restart.
        theme.add_listener(self._on_palette_changed)

    def apply_theme(self, theme: str) -> None:
        """Switch the live theme. Called by the Settings dialog on save."""
        self._theme.apply(theme)
        # ``apply`` already fires the listener which sets the theme-toggle
        # icon via ``_TopHeader.repaint_for_palette``. Nothing else to do.

    def _on_view_changed(self, view: str) -> None:
        self._apply_active_view(view, persist=True)

    def _show_about_dialog(self) -> None:
        """Pop the About / Authorship dialog from the brand click."""
        from .about_dialog import AboutDialog
        AboutDialog(self).exec()

    def _apply_active_view(self, view: str, *, persist: bool) -> None:
        """Switch the stacked widget and (optionally) persist the choice."""
        if view not in ("converter", "editor"):
            view = "converter"
        self.stack.setCurrentIndex(1 if view == "editor" else 0)
        self._header.set_active_view(view)
        if persist:
            from .app_settings import AppSettings
            AppSettings.remember_active_view(view)

    def _on_palette_changed(self, pal: dict) -> None:
        """Re-render every widget that holds palette-baked styling.

        QSS swap (handled by ``ThemeManager``) takes care of any rule
        keyed on object name / pseudo-state, but a few places still
        bake colors into inline stylesheets at construction time
        (panel headers, hint labels, the brand logo gradient) and
        delegate paints need their viewports invalidated to pick up
        the new palette tokens.
        """
        # Drop the qtawesome icon cache first so every ``apply_button``
        # call in the cascading repaint hooks below re-tints from the
        # new palette instead of returning a cached previous-theme icon.
        from . import icons
        icons.refresh_for_palette(pal)
        # Brand logo gradient — rebuilt from the new palette.
        self._header.repaint_for_palette(pal)
        # Cascade into the Converter and Editor panels.
        if hasattr(self, "converter"):
            self.converter.repaint_for_palette(pal)
        if hasattr(self, "editor"):
            self.editor.repaint_for_palette(pal)
        # Force a viewport repaint on every delegate-driven view in
        # the window so cells / badges / row tints pick up new colors.
        for view in self.findChildren(QAbstractItemView):
            view.viewport().update()
        for tree in self.findChildren(QTreeWidget):
            tree.viewport().update()

    def _set_status(self, text: str) -> None:
        # Truncate long single-line messages so the status bar doesn't
        # blow up the window width on big tracebacks.
        first_line = text.splitlines()[0] if text else ""
        if len(first_line) > 200:
            first_line = first_line[:197] + "…"
        self._status_text.setText(first_line)


__all__ = ["MainWindow"]
