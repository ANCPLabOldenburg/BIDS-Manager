"""The viewer: the one widget the BIDS Manager GUI embeds to show data.

A frame (header, a toolbar that wraps, a loading page, a footer with the
readout) around content supplied by a PRESENTER for the kind of data:
volumes now, signals and spectra through the same shell. Everything the
viewer does goes through its :class:`~bidsmgr.viz.store.SceneStore`, so the
host never reaches into a canvas:

* ``set_file(path, root)`` opens a file (``None`` clears);
* ``state()`` / ``apply_state(dict)`` exchange the view as plain data;
* ``run(command, **params)`` does anything a button or a key does;
* ``link(other)`` (:mod:`bidsmgr.gui.viz.link`) keeps two viewers in step.

Keyboard shortcuts are QActions built from the action table with the user's
keymap, scoped to the viewer and its children: they fire once the viewer has
focus (a click or a scroll on an image gives it focus), never while the user
types elsewhere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QStackedLayout, QVBoxLayout,
    QWidget,
)

from ...viz.commands.render import PLANE_COMMANDS
from ..widgets.primitives import ElidedLabel, PaneHeader
from ..widgets.spinner import BusySpinner
from .actions import ActionManager
from .bridge import JobRunner, QtStore, connect_while_alive
from .context import ViewerContext

log = logging.getLogger(__name__)


class Viewer(QWidget):
    """A data viewer for one file at a time."""

    file_changed = pyqtSignal(object)
    #: The whole file is read.
    loaded = pyqtSignal(Path)
    load_failed = pyqtSignal(Path, str)
    #: Something worth a line in the status bar.
    status_message = pyqtSignal(str)
    #: (busy, message): drives the host's busy spinner.
    loading_changed = pyqtSignal(bool, str)
    #: The user closed what was shown (a signal's Close): a host that opened
    #: the viewer as one page of several can go back to the other.
    close_requested = pyqtSignal()
    #: An overlay was added (its layer id).
    overlay_added = pyqtSignal(str)

    _is_viz_viewer = True

    def __init__(self, parent=None, *, kind: str = "volume", header: bool = True,
                 panels: bool = True) -> None:
        super().__init__(parent)
        self.kind = kind
        #: Whether the side panels (the controls column) open. A host that
        #: embeds the viewer only to render a figure passes False.
        self.panels = bool(panels)
        self.setObjectName("pane-dark")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._current_file: Optional[Path] = None
        self._current_root: Optional[Path] = None
        #: True until the first file opens: that one takes the remembered layout.
        self.first_open = True
        self._linking = 0

        self.qstore = QtStore(parent=self)
        self.jobs = JobRunner(self)
        self.ctx = ViewerContext(self.qstore, self.jobs, self)
        self.ctx.gpu_ok = self._probe_gpu() if kind == "volume" else False
        self.action_manager = ActionManager(self, run_command=self._run_command,
                                            run_gui=self._run_gui)
        self.ctx.run_action = self.action_manager.trigger

        self.presenter = self._make_presenter(kind)
        self.action_manager.canvases = getattr(self.presenter, "mouse_canvases", ())

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        if header:
            v.addWidget(PaneHeader(self.presenter.title))
        self._toolbar = self._build_toolbar()
        #: Its place in this layout (after the header), for ``lend_toolbar``.
        self._toolbar_index = v.count()
        v.addWidget(self._toolbar)
        #: Where the toolbar is shown when a host has it (else None).
        self._toolbar_host = None

        self._stack = QStackedLayout()
        self._stack.setContentsMargins(0, 0, 0, 0)
        v.addLayout(self._stack, 1)
        self._empty_hint = QLabel(self.presenter.empty_hint)
        self._empty_hint.setObjectName("pane-hint")
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_hint.setWordWrap(True)
        self._stack.addWidget(self._empty_hint)
        self._stack.addWidget(self.presenter.content)
        self._loading_panel = self._build_loading_panel()
        self._stack.addWidget(self._loading_panel)
        self._stack.setCurrentWidget(self._empty_hint)

        self._footer = QFrame()
        self._footer.setObjectName("sidecar-footer")
        fl = QHBoxLayout(self._footer)
        fl.setContentsMargins(14, 6, 14, 6)
        fl.setSpacing(10)
        # Elided: a footer of plain labels set the pane's minimum width to the
        # length of a deep dataset path.
        self._footer_path = ElidedLabel("")
        self._footer_path.setObjectName("sidecar-footer-path")
        self._readout = ElidedLabel("")
        self._readout.setObjectName("sidecar-footer-summary")
        self._summary = ElidedLabel("")
        self._summary.setObjectName("sidecar-footer-summary")
        fl.addWidget(self._footer_path, 1)
        fl.addWidget(self._readout, 2)
        fl.addWidget(self._summary)
        v.addWidget(self._footer)
        self._toolbar.setVisible(False)
        self._footer.setVisible(False)

        self.qstore.changed.connect(lambda _p: self.refresh_actions())
        self.ctx.readout.connect(self._readout.setText)
        connect_while_alive(self.ctx.settings_hub.changed, self,
                            lambda w, s: w._on_settings(s))
        self.action_manager.apply_keymap(self.ctx.settings.keymap)
        self.refresh_actions()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_gpu() -> bool:
        try:
            from .canvases.render import gpu_available

            return bool(gpu_available())
        except Exception:  # noqa: BLE001 - never block the 2-D viewer
            return False

    def _make_presenter(self, kind: str):
        if kind == "volume":
            from .presenters.volume import VolumePresenter

            return VolumePresenter(self, self.ctx)
        if kind == "signal":
            from .presenters.signal import SignalPresenter

            return SignalPresenter(self, self.ctx)
        if kind == "spectrum":
            from .presenters.spectrum import SpectrumPresenter

            return SpectrumPresenter(self, self.ctx)
        raise ValueError(f"no presenter for {kind!r}")

    def _build_toolbar(self) -> QFrame:
        """One row per presenter spec, each a panel header: the controls
        wrap on the left, and what follows ``"stretch"`` (help, close) sits
        in the upper-right corner as icons, where a window keeps its own.
        A stretch cannot push anything right in a row that wraps, so before
        this the last button trailed the others and, at 1500 px, wrapped
        onto a row of its own."""
        from .panels.panel_header import PanelHeader

        bar = QFrame()
        bar.setObjectName("sidecar-toolbar")
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(14, 6, 10, 6)
        outer.setSpacing(6)
        for spec in self.presenter.toolbar_rows():
            header = PanelHeader(margins=(0, 0, 0, 0), h_spacing=8, v_spacing=6)
            row = header.controls
            outer.addWidget(header)
            corner = []
            for item in spec:
                if item == "stretch":
                    corner = [None]          # what follows goes to the corner
                elif item == "|":
                    row.addSpacing(8)
                elif item.startswith("widget:"):
                    made = self.presenter.make_widget(item.split(":", 1)[1])
                    for w in (made if isinstance(made, list) else [made]):
                        if w is not None:
                            row.addWidget(w)
                elif corner:
                    corner.append(self.action_manager.button(item))
                else:
                    row.addWidget(self.action_manager.button(item))
            header.insert_corner([b for b in corner if b is not None])
        return bar

    def _build_loading_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("pane-dark")
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        v.addStretch(1)
        self._loading_spinner = BusySpinner()
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._loading_spinner)
        row.addStretch(1)
        v.addLayout(row)
        self._loading_label = ElidedLabel("", mode=Qt.TextElideMode.ElideMiddle)
        self._loading_label.setObjectName("pane-hint")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._loading_label)
        v.addStretch(1)
        return panel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def store(self):
        return self.qstore.store

    @property
    def scene(self):
        return self.qstore.store.scene

    def current_file(self) -> Optional[Path]:
        return self._current_file

    def current_root(self) -> Optional[Path]:
        return self._current_root

    def run(self, command_id: str, /, **params: Any):
        return self.qstore.run(command_id, **params)

    def source(self):
        """The open file's source (a ``VolumeSource`` for images), or None."""
        return getattr(self.presenter, "source", None)

    def is_loaded(self) -> bool:
        """Whether the open file has been read completely."""
        src = self.source()
        return src is not None and bool(getattr(src, "fully_loaded", False))

    def page(self) -> str:
        """What the viewer shows: ``"hint"``, ``"loading"`` or ``"content"``."""
        current = self._stack.currentWidget()
        if current is self.presenter.content:
            return "content"
        if current is self._loading_panel:
            return "loading"
        return "hint"

    def hint_text(self) -> str:
        return self._empty_hint.text()

    def readout_text(self) -> str:
        """The footer's position-and-value line."""
        return self._readout.text()

    def summary_text(self) -> str:
        """The footer's description of the file (shape, type, progress)."""
        return self._summary.text()

    def path_text(self) -> str:
        """The footer's dataset-relative path."""
        return self._footer_path.text()

    def toolbar_visible(self) -> bool:
        return self._toolbar.isVisible()

    def trigger(self, action_id: str) -> None:
        """Do what the button, the menu entry or the key for ``action_id`` does."""
        self.action_manager.trigger(action_id)

    def action(self, action_id: str):
        """The QAction of ``action_id`` (enabled and checked state, shortcuts)."""
        return self.action_manager.actions[action_id]

    def button(self, action_id: str):
        """The button bound to ``action_id``, in the toolbar or the content
        (a controls column, a panel), if there is one."""
        for root in (self._toolbar, self.presenter.content):
            for btn in root.findChildren(QWidget):
                if btn.property("viz_action") == action_id:
                    return btn
        return None

    def canvases(self, kind: str = "slice") -> list:
        """The canvases of ``kind`` currently on screen (``slice``, ``render``,
        ``graph``, ``mosaic``), for screenshots, tests and hosts that place
        their own overlays."""
        return self.presenter.canvases(kind)

    def add_overlay(self, path: Path) -> bool:
        """Draw the image at ``path`` over the open one (read on a worker;
        ``overlay_added`` when it is on screen). False when this viewer
        cannot take one (no image open, or not a volume viewer)."""
        add = getattr(self.presenter, "add_overlay", None)
        return bool(add(Path(path))) if add is not None else False

    #: How this viewer moves to another file when asked (next run, a
    #: fieldmap's target): ``fn(path)``. A host that keeps its own selection
    #: (the Editor's tree) sets it, so its tree, sidecar and validation pane
    #: follow; left None, the viewer opens the file itself.
    navigator = None

    def navigate(self, path: Path, root: Optional[Path] = None) -> None:
        """Open ``path`` in place of the file on screen, keeping where the
        crosshair is (the same spot in the next run is the comparison).
        ``root``: the dataset, when this viewer does not know it yet."""
        carry = getattr(self.presenter, "carry_cursor", None)
        if carry is not None:
            carry()
        if self.navigator is not None:
            self.navigator(Path(path))
        else:
            self.set_file(Path(path), root if root is not None else self._current_root)

    def show_header(self, rule: str = ""):
        """Open the header inspector (``rule``: the validator rule whose
        evidence to select). Waits for an image that is still opening."""
        show = getattr(self.presenter, "show_header", None)
        return show(rule) if show is not None else None

    def set_file(self, path: Optional[Path], root: Optional[Path]) -> None:
        """Open a file (``None`` clears). Reading happens on a worker."""
        self._current_file = Path(path) if path is not None else None
        self._current_root = Path(root) if root is not None else None
        if path is None:
            self.clear()
            self.file_changed.emit(None)
            return
        self._loading_label.setText(f"Loading {Path(path).name}...")
        self._loading_spinner.set_busy(True, message="")
        self._toolbar.setVisible(False)
        self._footer.setVisible(False)
        self._stack.setCurrentWidget(self._loading_panel)
        self.loading_changed.emit(True, f"Loading {Path(path).name}")
        self.presenter.load(Path(path), self._current_root)
        self.file_changed.emit(Path(path))

    def clear(self) -> None:
        self.presenter.clear()
        self._loading_spinner.set_busy(False)
        self._toolbar.setVisible(False)
        self._footer.setVisible(False)
        self._empty_hint.setText(self.presenter.empty_hint)
        self._stack.setCurrentWidget(self._empty_hint)
        self.update_footer()
        self.refresh_actions()

    def stop_loading(self, timeout_ms: int = 5000) -> None:
        """Cancel every job and wait: call before the viewer is destroyed."""
        self.presenter.stop()
        self.jobs.stop_all(timeout_ms)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_loading()
        super().closeEvent(event)

    def set_toolbar_visible(self, visible: bool) -> None:
        self._toolbar_wanted = bool(visible)
        self._sync_toolbar()

    def lend_toolbar(self, layout=None) -> None:
        """Show the toolbar in a host's ``layout`` (one toolbar above two
        viewers driven as one), or take it back (``None``)."""
        if layout is self._toolbar_host:
            return
        (self._toolbar_host or self.layout()).removeWidget(self._toolbar)
        if layout is None:
            self.layout().insertWidget(self._toolbar_index, self._toolbar)
        else:
            layout.addWidget(self._toolbar)
        self._toolbar_host = layout
        self._sync_toolbar()

    _toolbar_wanted = True

    def _sync_toolbar(self) -> None:
        """Shown when content is on screen, the host wants it, and the
        presenter has something for it (a signal's metadata card does not)."""
        wanted = getattr(self.presenter, "toolbar_wanted", lambda: True)()
        self._toolbar.setVisible(
            self._toolbar_wanted and wanted and self._current_file is not None
            and self._stack.currentWidget() is self.presenter.content)

    def set_loading_message(self, message: str) -> None:
        """What the loading page says (a series' progress)."""
        if self._loading_label.text() != message:
            self._loading_label.setText(message)

    def show_loading(self, message: str) -> None:
        """The spinner page, for a presenter's second read (Load signal)."""
        self._loading_label.setText(message)
        self._loading_spinner.set_busy(True, message="")
        self._toolbar.setVisible(False)
        self._stack.setCurrentWidget(self._loading_panel)
        self.loading_changed.emit(True, message)

    def show_content(self) -> None:
        self._loading_spinner.set_busy(False)
        self._stack.setCurrentWidget(self.presenter.content)
        self._footer.setVisible(True)
        self._sync_toolbar()
        self.loading_changed.emit(False, "")

    # -- state as plain data ---------------------------------------------

    def state(self) -> dict:
        """How things are shown (not which file): JSON-able."""
        return self.scene.model_dump(mode="json", exclude={"sources", "layers"})

    def apply_state(self, state: Mapping[str, Any]) -> None:
        from .link import apply_view_state

        apply_view_state(self.store, state)

    # ------------------------------------------------------------------
    # Presenter callbacks
    # ------------------------------------------------------------------

    def on_first_frame(self) -> None:
        self._loading_spinner.set_busy(False)
        self._stack.setCurrentWidget(self.presenter.content)
        self._footer.setVisible(True)
        self.presenter.apply_mode()
        self.presenter.apply_graph()
        if self.panels:
            self.presenter.restore_panels()
        self.presenter.sync_widgets()
        self.update_footer()
        self.refresh_actions()
        self._sync_toolbar()

    def on_loaded(self, path: Optional[Path]) -> None:
        if path is None:
            return
        if self._stack.currentWidget() is not self.presenter.content:
            self.on_first_frame()
        self._sync_toolbar()
        self.loading_changed.emit(False, "")
        self.update_footer()
        self.loaded.emit(Path(path))

    def on_load_failed(self, path: Optional[Path], message: str) -> None:
        self._loading_spinner.set_busy(False)
        self.loading_changed.emit(False, "")
        log.warning("Could not load %s: %s", path, message)
        self._toolbar.setVisible(False)
        self._footer.setVisible(False)
        name = path.name if path is not None else "the file"
        self._empty_hint.setText(f"Could not load {name}:\n{message}")
        self._stack.setCurrentWidget(self._empty_hint)
        if path is not None:
            self.load_failed.emit(Path(path), message)

    def update_footer(self) -> None:
        path, root = self._current_file, self._current_root
        if path is None:
            self._footer_path.setText("")
            self._readout.setText("")
            self._summary.setText("")
            return
        text = path.name
        if root is not None:
            try:
                text = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                text = str(path)
        self._footer_path.setText(text)
        self._readout.setText(self.presenter.readout())
        self._summary.setText(self.presenter.summary())

    #: A folder open for viewing: nothing is written into it (no scene, no
    #: QC result, no reviewed channels or segments). The host sets it.
    read_only = False

    def set_read_only(self, on: bool) -> None:
        self.read_only = bool(on)
        self.refresh_actions()

    def refresh_actions(self) -> None:
        ctx = dict(self.presenter.action_context())
        ctx["undo"] = self.store.can_undo
        ctx["redo"] = self.store.can_redo
        ctx["writable"] = not self.read_only
        self.action_manager.update_state(ctx)

    # ------------------------------------------------------------------
    # Commands and GUI actions
    # ------------------------------------------------------------------

    def _run_command(self, command_id: str, **params):
        # The clip keys (Shift+Z, Shift+A, ...) act on the plane the 3-D panel
        # is editing, never silently on the first one.
        if command_id in PLANE_COMMANDS and "index" not in params:
            params["index"] = self.ctx.active_clip
        return self.qstore.run(command_id, **params)

    def _run_gui(self, name: str, params: Mapping[str, Any]) -> None:
        if self.presenter.gui_action(name, params):
            return
        if name == "undo":
            self.store.undo()
        elif name == "redo":
            self.store.redo()
        elif name == "help":
            from .help import show_help

            show_help(self, self.action_manager, self.ctx.settings.mousemap,
                      self.kind, getattr(self.presenter, "mouse_canvases", ()))
        elif name == "palette":
            from .help import CommandPalette

            CommandPalette(self, self.action_manager, self.action_manager.trigger).exec()
        elif name == "screenshot":
            self.save_screenshot()

    def save_screenshot(self, path: Optional[Path] = None, *,
                        scale: float = 1.0, transparent: bool = False) -> Optional[Path]:
        """Save what the viewer shows as a PNG: the figure only, without
        toolbars or editors (the mosaic's line editor is not part of a
        mosaic).

        ``scale`` > 1 draws the 2-D canvases again at that factor rather
        than enlarging a screen grab, so text, lines and edges stay sharp
        in a figure. A view holding the 3-D render is grabbed at screen
        resolution, because a GL surface cannot be redrawn into an image.
        """
        if path is None:
            base = self._current_file.name.split(".")[0] if self._current_file else "viewer"
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Save screenshot", f"{base}.png", "PNG image (*.png)",
            )
            if not chosen:
                return None
            path = Path(chosen)
        image = self.grab_figure(scale=scale, transparent=transparent)
        image.save(str(path), "PNG")
        self.status_message.emit(f"Saved {path.name}")
        return path

    def grab_figure(self, *, scale: float = 1.0, transparent: bool = False):
        """The figure as a QImage (see :meth:`save_screenshot`).

        ``transparent`` leaves out the black surround (and every window
        background), so the image sits on whatever page it is put on.
        """
        from PyQt6.QtGui import QImage, QPainter
        from PyQt6.QtOpenGLWidgets import QOpenGLWidget
        from PyQt6.QtWidgets import QWidget as _QWidget

        widget = self.presenter.figure_widget()
        has_gl = any(w.isVisible() for w in widget.findChildren(QOpenGLWidget))
        if has_gl or isinstance(widget, QOpenGLWidget) or (scale <= 1.0 and not transparent):
            return widget.grab().toImage()
        w = max(1, int(round(widget.width() * scale)))
        h = max(1, int(round(widget.height() * scale)))
        image = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.scale(scale, scale)
        flags = _QWidget.RenderFlag.DrawChildren
        if not transparent:
            flags |= _QWidget.RenderFlag.DrawWindowBackground
        self.ctx.transparent = bool(transparent)
        try:
            from PyQt6.QtCore import QPoint
            from PyQt6.QtGui import QRegion

            widget.render(painter, QPoint(), QRegion(), flags)
        finally:
            self.ctx.transparent = False
            painter.end()
        return image

    # ------------------------------------------------------------------
    # Keys held, settings, theme
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_H and not event.isAutoRepeat():
            self.ctx.held.add("h")
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_H and not event.isAutoRepeat():
            self.ctx.held.discard("h")
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802
        # A key released while another window had focus never arrives here.
        self.ctx.held.clear()
        super().focusOutEvent(event)

    def event(self, event) -> bool:  # noqa: D401
        if event.type() == QEvent.Type.WindowDeactivate:
            self.ctx.held.clear()
        return super().event(event)

    def _on_settings(self, settings) -> None:
        self.action_manager.apply_keymap(settings.keymap)
        self.update()

    def repaint_for_palette(self, pal: dict) -> None:
        """Re-polish QSS-styled children and hand the canvases the palette."""
        from .bridge import ThemeHub

        ThemeHub.instance().publish(pal)
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()
        try:
            from ..converter_panel import _repolish_combo
            from PyQt6.QtWidgets import QComboBox

            for combo in self.findChildren(QComboBox):
                _repolish_combo(combo)
        except Exception:  # noqa: BLE001 - cosmetic
            pass


__all__ = ["Viewer"]
