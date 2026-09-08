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

from PyQt6.QtCore import QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPixmap


def _trim_transparent_bbox(img: QImage) -> QImage:
    """Return *img* cropped to its non-transparent bounding box.

    macOS-style app icons (``AppIcon128.png``) ship with ~10 px of
    transparent padding on every side so the artwork sits cleanly
    inside the rounded ``.app`` mask. That padding makes the rendered
    glyph look smaller than a tightly-cropped artwork at the same
    scaled height. Trimming the alpha-channel bbox before scaling
    gives both artworks the same visible footprint.

    Falls back to the original image if it has no alpha channel or
    is entirely transparent.
    """
    if img.isNull():
        return img
    if not img.hasAlphaChannel():
        return img
    if img.format() != QImage.Format.Format_ARGB32:
        img = img.convertToFormat(QImage.Format.Format_ARGB32)
    w, h = img.width(), img.height()
    xmin, ymin, xmax, ymax = w, h, -1, -1
    for y in range(h):
        for x in range(w):
            if (img.pixel(x, y) >> 24) & 0xFF:
                if x < xmin:
                    xmin = x
                if y < ymin:
                    ymin = y
                if x > xmax:
                    xmax = x
                if y > ymax:
                    ymax = y
    if xmax < 0 or ymax < 0:
        return img
    return img.copy(QRect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from ..project import Project
from .converter_panel import ConverterPanel
from .editor_panel import EditorPanel
from .theme_manager import CUR, ThemeManager
from .welcome_panel import WelcomePanel
from .widgets import ElidedLabel

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


class _ProjectMenuRow(QWidget):
    """A two-line row (project name + path) for the project-switcher menu.

    Used both for the current project (display only) and for each recent
    project (clickable -> switch). Emits :pyattr:`clicked` on release when
    clickable. Styling is QSS-driven (object names ``proj-menu-*``).
    """

    clicked = pyqtSignal()

    def __init__(
        self, name: str, path: str, *, current: bool, title: str = "", parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("proj-menu-row-current" if current else "proj-menu-row")
        self._clickable = not current
        # A plain QWidget paints a QSS ``background`` (and its ``:hover``
        # variant) only with WA_StyledBackground; WA_Hover delivers the hover
        # events. Together they give the brighten-on-hover effect the home
        # recents have.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 6, 14, 6)
        lay.setSpacing(1)
        # The project name, and beside it the dataset's own title in a quieter
        # tone when it says something different. One says where the data lives,
        # the other what it is.
        if title:
            n = QLabel(
                f'{name}<span style="color:{CUR()["muted"]}; font-style:italic;">'
                f"&nbsp;&nbsp;{title}</span>"
            )
            n.setTextFormat(Qt.TextFormat.RichText)
        else:
            n = QLabel(name)
        n.setObjectName("proj-menu-name")
        p = QLabel(path)
        p.setObjectName("proj-menu-path")
        lay.addWidget(n)
        lay.addWidget(p)
        if self._clickable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._clickable and event.button() == Qt.MouseButton.LeftButton:
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
    settings_requested = pyqtSignal()
    # A recent project was chosen from the project-switcher dropdown (Path).
    project_switch_requested = pyqtSignal(object)

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
        self._welcome_btn = QPushButton("Home")
        self._welcome_btn.setObjectName("view-pill")
        self._welcome_btn.setCheckable(True)
        self._converter_btn = QPushButton("Converter")
        self._converter_btn.setObjectName("view-pill")
        self._converter_btn.setCheckable(True)
        self._converter_btn.setChecked(True)
        self._editor_btn = QPushButton("Editor")
        self._editor_btn.setObjectName("view-pill")
        self._editor_btn.setCheckable(True)
        self._pill_group = QButtonGroup(self)
        self._pill_group.setExclusive(True)
        self._pill_group.addButton(self._welcome_btn, 2)
        self._pill_group.addButton(self._converter_btn, 0)
        self._pill_group.addButton(self._editor_btn, 1)
        self._pill_group.idClicked.connect(self._on_pill_clicked)
        h.addWidget(self._welcome_btn)
        h.addWidget(self._converter_btn)
        h.addWidget(self._editor_btn)

        # Project switcher (PyCharm-style), set apart from the Editor pill.
        # Shows the current dataset name + a project icon; its dropdown lists
        # the current project (name + path) and recent projects you can switch
        # to. Hidden until a project is open.
        h.addSpacing(18)
        from . import icons as _icons
        self._project_btn = QPushButton("(no project)")
        self._project_btn.setObjectName("project-switcher")
        self._project_btn.setToolTip("Current project. Click to switch.")
        _icons.apply_button(self._project_btn, "project")
        self._project_menu = QMenu(self)
        self._project_menu.setObjectName("project-menu")
        # Frameless + translucent so the QSS border-radius renders rounded
        # corners cleanly (no square OS background / shadow behind them).
        self._project_menu.setWindowFlags(
            self._project_menu.windowFlags()
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self._project_menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._project_menu.aboutToShow.connect(self._rebuild_project_menu)
        self._project_btn.setMenu(self._project_menu)
        self._project_btn.setVisible(False)
        self._active_project_root: Optional[Path] = None
        h.addWidget(self._project_btn)

        h.addStretch(1)

        self._theme = theme
        from . import icons

        # Settings gear, immediately left of the theme toggle. Lives in the
        # header (not the converter toolbar) so it reads as a global control
        # alongside the theme toggle and is reachable from either view.
        self._settings_btn = QPushButton()
        self._settings_btn.setObjectName("theme-toggle")  # same compact icon-button style
        self._settings_btn.setToolTip("Settings")
        self._settings_btn.setFixedSize(32, 28)
        icons.apply_button(self._settings_btn, "settings")
        self._settings_btn.clicked.connect(self.settings_requested.emit)
        h.addWidget(self._settings_btn)

        # Icon-only toggle. ``sun`` glyph in dark mode (click to lighten),
        # ``moon`` glyph in light mode (click to darken). Re-tinted in
        # ``repaint_for_palette`` on every theme swap.
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
        target = {
            "welcome": self._welcome_btn,
            "editor": self._editor_btn,
        }.get(view, self._converter_btn)
        target.setChecked(True)

    def _on_pill_clicked(self, idx: int) -> None:
        view = {2: "welcome", 1: "editor"}.get(idx, "converter")
        self.view_changed.emit(view)

    # ------------------------------------------------------------------
    # Project switcher
    # ------------------------------------------------------------------

    def set_active_project(self, name: str, root: Path) -> None:
        """Show the switcher with ``name`` as the current project.

        The pill carries the project name alone. A QPushButton renders one
        colour and reports its full text as its minimum width, so a dataset
        title in here would both look like part of the name and widen the
        header; it goes in the tooltip and, in two colours, in the menu below.
        """
        from .welcome_panel import _dataset_title

        self._active_project_root = Path(root)
        title = _dataset_title(Path(root))
        self._project_btn.setText(name)
        self._project_btn.setToolTip(
            (f"{name}\n{title}\n{root}" if title else f"{name}\n{root}")
            + "\n\nClick to switch projects."
        )
        self._project_btn.setVisible(True)

    def clear_active_project(self) -> None:
        """Hide the switcher (no project open)."""
        self._active_project_root = None
        self._project_btn.setVisible(False)

    def _rebuild_project_menu(self) -> None:
        """Repopulate the dropdown: current project header + recent projects."""
        from .app_settings import AppSettings
        from .welcome_panel import _dataset_display_name, _dataset_title

        menu = self._project_menu
        menu.clear()

        # Current project header (display only).
        if self._active_project_root is not None:
            cur = self._active_project_root
            header = _ProjectMenuRow(
                _dataset_display_name(cur), str(cur), current=True,
                title=_dataset_title(cur), parent=menu,
            )
            wa = QWidgetAction(menu)
            wa.setDefaultWidget(header)
            wa.setEnabled(False)
            menu.addAction(wa)
            menu.addSeparator()

        # Recent projects (excluding the current one).
        recents = [
            p for p in AppSettings.load().recent_projects
            if self._active_project_root is None
            or Path(p) != self._active_project_root
        ]
        if recents:
            label = QLabel("Recent projects")
            label.setObjectName("proj-menu-section")
            la = QWidgetAction(menu)
            la.setDefaultWidget(label)
            la.setEnabled(False)
            menu.addAction(la)
            for p in recents:
                exists = Path(p).exists()
                name = _dataset_display_name(Path(p)) if exists else Path(p).name
                row = _ProjectMenuRow(
                    name + ("   (missing)" if not exists else ""),
                    str(p), current=False,
                    title=_dataset_title(Path(p)) if exists else "",
                    parent=menu,
                )
                wa = QWidgetAction(menu)
                wa.setDefaultWidget(row)
                wa.setEnabled(exists)
                if exists:
                    row.clicked.connect(
                        lambda _=False, path=Path(p): self._on_recent_chosen(path)
                    )
                menu.addAction(wa)
            menu.addSeparator()

        # Always offer the Home tab (open / create another).
        open_act = menu.addAction("Open or create another… (Home)")
        open_act.triggered.connect(lambda: self.view_changed.emit("welcome"))

    def _on_recent_chosen(self, path: Path) -> None:
        self._project_menu.close()
        self.project_switch_requested.emit(path)

    def _on_toggle(self) -> None:
        from .app_settings import AppSettings
        from . import icons
        new = self._theme.toggle()
        icons.apply_button(self._theme_btn, "sun" if new == "dark" else "moon")
        AppSettings.remember_theme(new)

    def _apply_logo_pixmap(self, pal: dict) -> None:
        """Load the brand artwork chosen by ``AppSettings.header_logo``.

        Two artworks ship with the wheel:

        * ``"default"``: ``assets/logo.png`` — a monochrome mark drawn
          dark-on-transparent for a light background. On a dark theme
          we invert the RGB channels (alpha preserved) so the same
          artwork reads as light-on-transparent against the dark
          surface.
        * ``"app_icon"``: ``assets/macos/AppIcon128.png`` — the
          full-color BIDS-Manager application icon. Rendered as-is on
          both themes (no inversion).

        Falls back to a gradient ``B`` if neither file is readable so
        the GUI stays usable in a partial source tree.
        """
        from .app_settings import AppSettings
        choice = AppSettings.load().header_logo
        assets = Path(__file__).parent / "assets"
        if choice == "app_icon":
            png = assets / "macos" / "AppIcon128.png"
            invert_on_dark = False
        else:
            png = assets / "logo.png"
            invert_on_dark = True

        if png.exists():
            img = QImage(str(png))
            if not img.isNull():
                if invert_on_dark and self._is_dark_theme(pal):
                    # ``InvertRgb`` flips R/G/B; alpha is preserved so
                    # the transparent background stays transparent.
                    img.invertPixels(QImage.InvertMode.InvertRgb)
                if choice == "app_icon":
                    # Trim transparent padding (macOS-bundle icons ship
                    # with ~10 px of inset). Also scale to a slightly
                    # taller pixmap and widen the host label / header so
                    # the full-color icon reads with more presence than
                    # the tightly-cropped monochrome mark.
                    img = _trim_transparent_bbox(img)
                    target_h = 44
                    self._logo.setFixedSize(48, 44)
                    self.setFixedHeight(56)
                else:
                    target_h = 36
                    self._logo.setFixedSize(40, 36)
                    self.setFixedHeight(48)
                pix = QPixmap.fromImage(img)
                self._logo.setPixmap(pix.scaledToHeight(
                    target_h,
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
        icons.apply_button(self._settings_btn, "settings")
        icons.apply_button(self._project_btn, "project")
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
        self.welcome = WelcomePanel()
        self.stack.addWidget(self.converter)   # index 0 → "converter"
        self.stack.addWidget(self.editor)      # index 1 → "editor"
        self.stack.addWidget(self.welcome)     # index 2 → "welcome"
        v.addWidget(self.stack, 1)

        self._header.view_changed.connect(self._on_view_changed)
        self._header.about_requested.connect(self._show_about_dialog)
        self._header.settings_requested.connect(self._open_settings)
        self.welcome.project_opened.connect(self._on_project_opened)
        # Project-switcher dropdown: opening a recent reuses the Welcome
        # open-project flow (which re-emits project_opened -> _on_project_opened).
        self._header.project_switch_requested.connect(self._on_switch_project)

        # Land on the Welcome (Home) tab when no project is bound (the
        # project-first entry point). When a project was passed in (the
        # ``--project`` path) restore the user's last Converter/Editor view.
        from .app_settings import AppSettings
        settings = AppSettings.load()
        if self._project is not None:
            self._apply_active_view(settings.active_view, persist=False)
        else:
            self._apply_active_view("welcome", persist=False)

        # Status bar — forwards the Converter's log messages so the
        # user sees scan / convert progress. The bottom-right corner
        # carries the installed version label + "Check updates" button
        # via :func:`update_widgets.attach_update_widgets`.
        sb = QStatusBar()
        sb.setSizeGripEnabled(False)
        # Elided: worker messages carry full paths and tracebacks, and a
        # plain QLabel would raise the window's layout minimum by the
        # width of whatever it was last handed.
        self._status_text = ElidedLabel("Ready")
        sb.addWidget(self._status_text, 1)
        self.setStatusBar(sb)

        self.converter.log_message.connect(self._set_status)
        self.converter.project_rename_requested.connect(self._rename_project)
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

    def _open_settings(self) -> None:
        """Open the Settings dialog (triggered by the header gear).

        Lives on the window because the gear now sits in the global header
        next to the theme toggle, so it must work from either view. On Save
        we apply the font scale + theme live (the window owns the
        ``ThemeManager``) and let the Converter pick up new scan / convert
        defaults on its next run.
        """
        from .app_settings import AppSettings
        from .settings_dialog import SettingsDialog
        s = AppSettings.load()
        dlg = SettingsDialog(s, self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            # Font scale first, then theme — a single Save can change both
            # and the theme re-apply re-substitutes the scaled QSS template.
            self.apply_font_scale(s.font_scale)
            self.apply_theme(s.theme)
            reload_fn = getattr(self.converter, "reload_app_settings", None)
            if callable(reload_fn):
                reload_fn()

    def apply_theme(self, theme: str) -> None:
        """Switch the live theme. Called by the Settings dialog on save."""
        self._theme.apply(theme)
        # ``apply`` already fires the listener which sets the theme-toggle
        # icon via ``_TopHeader.repaint_for_palette``. Nothing else to do.

    def apply_font_scale(self, scale: float) -> None:
        """Switch the live UI font scale. Called by the Settings dialog
        before ``apply_theme`` so a single Save can change both at once.

        ``ThemeManager.set_font_scale`` re-applies the active theme,
        which re-substitutes the QSS template with the scaled font-size
        values, resizes the QApplication default font, and cascades
        through every panel's ``repaint_for_palette`` so inline-
        stylesheet font sizes (delegate paints, properties panel
        sub-headers) refresh too.
        """
        self._theme.set_font_scale(scale)

    def _on_view_changed(self, view: str) -> None:
        self._apply_active_view(view, persist=True)

    def _show_about_dialog(self) -> None:
        """Pop the About / Authorship dialog from the brand click."""
        from .about_dialog import AboutDialog
        AboutDialog(self).exec()

    def _apply_active_view(self, view: str, *, persist: bool) -> None:
        """Switch the stacked widget and (optionally) persist the choice."""
        if view not in ("welcome", "converter", "editor"):
            view = "converter"
        index = {"welcome": 2, "editor": 1}.get(view, 0)
        self.stack.setCurrentIndex(index)
        self._header.set_active_view(view)
        # Only the working views are persisted; "welcome" is a transient
        # landing, not a remembered preference.
        if persist and view in ("converter", "editor"):
            from .app_settings import AppSettings
            AppSettings.remember_active_view(view)

    def _rename_project(self, root, new_name: str) -> None:
        """Rename the project folder, then reopen it under its new name.

        Asked for from the metadata form, where a user renamed the dataset and
        said they meant the project itself. Doing it here rather than there is
        deliberate: this is a move on disk that invalidates every open handle on
        the old path, so it has to close the project and open the new one, which
        is the window's job.

        Anything else pointing at the old path, a script or a backup, will not
        follow. The dialog says so before the user chooses.
        """
        from PyQt6.QtWidgets import QMessageBox

        root = Path(root)
        target = root.parent / new_name
        if target == root:
            return
        try:
            root.rename(target)
        except OSError as exc:
            QMessageBox.warning(
                self, "Could not rename the project",
                f"{root.name} could not be renamed to {new_name}.\n\n{exc}",
            )
            return

        from .app_settings import AppSettings

        AppSettings.forget_recent_project(root)
        AppSettings.remember_recent_project(target)
        self._set_status(f"Renamed project to {new_name}")

        from ..cli.create import open_or_create_workspace

        try:
            reopened = open_or_create_workspace(target)
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            QMessageBox.warning(
                self, "Renamed, but could not reopen",
                f"The folder is now {target}, but the project could not be "
                f"reopened.\n\n{exc}",
            )
            return
        self._on_project_opened(reopened, target)

    def _on_project_opened(self, project: Project, bids_root: Path) -> None:
        """Bind the project the user created/opened on Welcome and switch in.

        The Converter's output is locked to ``bids_root`` (soft lock) and the
        Editor is pointed at the same root so both views work inside the project.
        """
        self._project = project
        self.converter.set_project(project, Path(bids_root))
        # Point the Editor at the same root so both views work inside the
        # project. ``_set_root`` is the Editor's open-root primitive.
        set_root = getattr(self.editor, "_set_root", None)
        if callable(set_root):
            try:
                set_root(Path(bids_root), persist=False)
            except Exception as exc:  # never block entering the project
                log.warning("editor could not open %s: %s", bids_root, exc)
        # Reveal + label the header project switcher.
        from .welcome_panel import _dataset_display_name
        self._header.set_active_project(
            _dataset_display_name(Path(bids_root)), Path(bids_root),
        )
        self._apply_active_view("converter", persist=True)

    def _on_switch_project(self, path: Path) -> None:
        """A recent project was chosen from the header switcher.

        Reuse the Welcome open-project flow so the bundle is opened/adopted and
        ``project_opened`` re-binds both views (and re-labels the switcher).
        """
        try:
            self.welcome.open_project(Path(path))
        except Exception as exc:
            log.warning("could not switch to project %s: %s", path, exc)

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
        if hasattr(self, "welcome"):
            self.welcome.repaint_for_palette(pal)
        # Force a viewport repaint on every delegate-driven view in
        # the window so cells / badges / row tints pick up new colors.
        for view in self.findChildren(QAbstractItemView):
            view.viewport().update()
        for tree in self.findChildren(QTreeWidget):
            tree.viewport().update()

    def _set_status(self, text: str) -> None:
        # Only the first line: the status bar is one row tall. Its width
        # needs no guarding -- ``ElidedLabel`` clips to the room it has.
        self._status_text.setText(text.splitlines()[0] if text else "")


__all__ = ["MainWindow"]
