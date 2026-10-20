"""Volumes (MRI, PET, any NIfTI) in the viewer shell.

The presenter owns what is specific to volumes: the view pages (one plane,
three planes, 3-D, three planes with 3-D, hero, mosaic), the time-course
graph, the toolbar's sliders and contrast controls, loading, and the readout.
The shell (:class:`~bidsmgr.gui.viz.viewer.Viewer`) owns the frame around it:
header, toolbar host, loading page, footer, keys, links.

Loading is two jobs. ``open`` reads the header and the BIDS context; then
``stream`` reads the voxels front to back. The views appear as soon as the
FIRST frame is in, so a 1224-volume run shows within a tenth of a second and
the rest arrives while the user is already looking.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout,
    QLabel, QMenu, QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from ....viz import layouts, views
from ....viz.bids import BidsContext, bids_context
from ....viz.compute import colormaps
from ....viz.data.volume import VolumeSource, memory_budget_bytes, open_volume
from ....viz.scene import (
    Cursor, GraphState, LayoutState, MosaicBuild, Scene, SourceRef, VolumeDisplay,
    VolumeLayer,
)
from ..context import ViewerContext
from ..menus import popup_menu, submenu
from ...widgets.splitter import Splitter

log = logging.getLogger(__name__)

#: How often streaming progress may repaint (seconds).
_PROGRESS_INTERVAL = 0.1

#: ONE row: what you switch while looking (views, slice, volume) and the
#: menus. Everything you set (the look, the view conventions, the layout, the
#: 3-D) is in the controls column, by purpose.
#: One row, by purpose: WHAT is shown (the layout, with the planes inside
#: it, and the time course); WHERE you are (slice, volume, play); what can be
#: DONE (quality control, tools); what is KEPT (save); help. The controls
#: column has its own tab on the right edge of the images.
TOOLBAR_ROWS = (
    ("widget:layout", "view.graph", "|", "widget:slice", "widget:frame", "frame.play",
     "|", "widget:quality", "widget:tools", "widget:save", "stretch", "help.shortcuts"),
)

#: The Layout menu: one plane, then the multi-view layouts.
PLANE_ACTIONS = ("view.axial", "view.coronal", "view.sagittal")
LAYOUT_ACTIONS = ("view.multi", "view.combo", "view.3d", "view.hero", "view.mosaic")
#: What the Layout button says for each mode, and its icon.
LAYOUT_NAMES = {"single": "One plane", "multi": "Multi-planar", "combo": "Planes + 3-D",
                "3d": "3-D", "hero": "Large view", "mosaic": "Mosaic"}
LAYOUT_ICONS = {"single": "layout_single", "multi": "layout_multi", "combo": "layout_combo",
                "3d": "layout_3d", "hero": "layout_hero", "mosaic": "layout_mosaic"}


#: The base layer's look a scene restores.
_SCENE_DISPLAY_KEYS = {"colormap", "colormap_negative", "window", "window_negative", "gamma",
                       "invert", "opacity", "threshold_mode", "outline_px", "interpolation"}


def open_overlay_with_context(path: Path, root: Optional[Path], *, budget_bytes=None,
                              cancel=None):
    """Worker-side: an overlay read whole, and, when it is a series, its own
    run context (repetition time, events, physio) for the graph."""
    from ....viz.overlays import open_overlay

    overlay = open_overlay(path, budget_bytes=budget_bytes, cancel=cancel)
    src = overlay.source
    ctx = None
    if src.is_4d and not src.is_rgb and not src.in_memory:
        try:
            ctx = bids_context(Path(path), root)
        except Exception:  # noqa: BLE001 - the graph works without it
            ctx = None
    return overlay, ctx


def open_with_context(path: Path, root: Optional[Path]) -> tuple[VolumeSource, BidsContext]:
    """Worker-side: the header and the dataset context, nothing heavier."""
    src = open_volume(path)
    ctx = bids_context(path, root)
    return src, ctx


def _caption(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("sidecar-footer-summary")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return lbl


class _SideSplit(Splitter):
    """The images-and-controls splitter, saying when it was resized (the
    controls column gives way when the viewer gets narrow)."""

    resized = pyqtSignal(int, int)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt signature
        super().resizeEvent(event)
        self.resized.emit(int(event.size().width()), int(event.oldSize().width()))


#: Room the images keep when the controls column is open: below it, a
#: narrowing viewer hides the column rather than squeezing the images away.
SIDE_GIVES_WAY_PX = 300


class _GraphWindow(QWidget):
    """The time-course panel in a window of its own. A child of the viewer
    with the Window flag, so it closes with it and stays with its window on
    every platform; closing it puts the panel back."""

    closed = pyqtSignal()

    def __init__(self, viewer) -> None:
        super().__init__(viewer, Qt.WindowType.Window)
        self.setObjectName("pane-dark")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

    def take(self, widget: QWidget) -> None:
        self.layout().addWidget(widget)
        widget.setVisible(True)
        self.show()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt signature
        super().closeEvent(event)
        self.closed.emit()


class VolumePresenter:
    """Volume content for a :class:`Viewer`."""

    kind = "volume"
    title = "NIfTI"
    #: Which mouse tables the help shows.
    mouse_canvases = ("slice", "render")
    empty_hint = "Select a NIfTI (.nii / .nii.gz) file in the BIDS tree to view it."

    def __init__(self, viewer, ctx: ViewerContext) -> None:
        self.viewer = viewer
        self.ctx = ctx
        self.source: Optional[VolumeSource] = None
        self.bids: Optional[BidsContext] = None
        self._generation = 0
        self._last_progress = 0.0
        self._first_frame_shown = False
        #: A series read whole waits for its series-wide window to be shown.
        self._show_when_windowed = False
        #: The window the viewer chose by itself (None once the user picks).
        self._auto_window: Optional[tuple[float, float]] = None
        #: The kind of file on screen (a layout preset id); survives a clear,
        #: so looking at a sidecar and back keeps the arrangement.
        self._preset_id = ""
        self._pending_sizes: dict[str, list[int]] = {}
        self._sizes_timer = QTimer(viewer)
        self._sizes_timer.setSingleShot(True)
        self._sizes_timer.setInterval(400)
        self._sizes_timer.timeout.connect(self._flush_sizes)
        # The arrangement and the look are written after a change settles:
        # a slider drag is one write, not one a tick.
        self._view_timer = QTimer(viewer)
        self._view_timer.setSingleShot(True)
        self._view_timer.setInterval(400)
        self._view_timer.timeout.connect(self._remember_view)
        self._mosaic_page = None
        self._render = None
        self._graph = None
        #: The time course has the whole viewer (the views are hidden).
        self._graph_maximized = False
        #: The time course's own window, while it is in one.
        self._graph_window = None
        #: Each series' run context (repetition time, events, physio), by
        #: source id: the base's, and every 4-D overlay's.
        self._contexts: dict[str, Any] = {}
        #: Each overlay's look as its content called for (restore defaults).
        self._original_looks: dict[str, dict] = {}
        self._graph_context_source: Optional[str] = None
        self._play_timer = QTimer(viewer)
        self._play_timer.timeout.connect(self._play_tick)
        self._remembering = True
        # Overlays read on their own jobs, one tag each, so several can load
        # at once and all are cancelled when the base image changes.
        self._overlay_seq = 0
        self._overlay_tags: list[str] = []
        #: A saved scene's look for each overlay job (display, name, ...).
        self._overlay_looks: dict[str, dict] = {}
        #: Overlays asked for while the base image was still opening.
        self._pending_overlays: list[tuple[Path, dict]] = []
        #: A saved scene being opened: applied once its image is open.
        self._pending_scene: Optional[dict] = None
        #: Its quality maps, computed once the series is read.
        self._pending_qc: list[dict] = []
        #: The crosshair to open the next file at (set by ``carry_cursor``).
        self._carried_cursor = None
        #: The header inspector, asked for while the image was opening (the
        #: rule to select, "" for none).
        self._pending_header: Optional[str] = None
        self.header_dialog = None
        self.content = self._build_content()
        ctx.jobs.done.connect(self._on_job_done)
        ctx.jobs.failed.connect(self._on_job_failed)
        ctx.jobs.progressed.connect(self._on_job_progress)
        ctx.qstore.changed.connect(self._on_changed)
        # Build the 3-D canvas NOW when a GPU exists, before the window is
        # shown: adding the first QOpenGLWidget to a visible window makes Qt
        # recreate the native window on Windows and Linux.
        if ctx.gpu_ok:
            try:
                self._ensure_render()
            except Exception as exc:  # noqa: BLE001 - never block the 2-D viewer
                log.warning("Could not pre-create the 3-D view: %s", exc)
                ctx.gpu_ok = False

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    def _build_content(self) -> QWidget:
        self.vsplit = Splitter(Qt.Orientation.Vertical)
        self.vsplit.setHandleWidth(2)
        self.vsplit.setChildrenCollapsible(False)
        self.pages = QStackedWidget()
        self.pages.setObjectName("nifti-canvas")
        from .tiles import TiledPage

        self.tiles = TiledPage(self)
        self.pages.addWidget(self.tiles)
        self.vsplit.addWidget(self.pages)
        self.graph_host = QWidget()
        lay = QVBoxLayout(self.graph_host)
        lay.setContentsMargins(0, 0, 0, 0)
        self.graph_host.setVisible(False)
        self.vsplit.addWidget(self.graph_host)
        self.vsplit.setStretchFactor(0, 3)
        self.vsplit.setStretchFactor(1, 1)
        self.vsplit.splitterMoved.connect(
            lambda *_a: self.remember_sizes("volume.graph", self.vsplit))
        # The controls column beside images and graph. Built empty; the
        # inspector is created the first time it is opened.
        self.hsplit = _SideSplit(Qt.Orientation.Horizontal)
        self.hsplit.resized.connect(self._on_split_resized)
        #: The column was hidden for lack of room (not by the user).
        self._side_gave_way = False
        self.hsplit.setHandleWidth(2)
        self.hsplit.setChildrenCollapsible(False)
        # The images with the controls column's tab on their right edge:
        # the tab is part of the images' side, so it is never dragged away.
        from ..panels.side_tab import SideTab

        # The quality panel beside the views and the time course (a report
        # is a tall list), or below them: two hosts, one panel.
        self.qsplit = Splitter(Qt.Orientation.Horizontal)
        self.qsplit.setHandleWidth(2)
        self.qsplit.setChildrenCollapsible(False)
        self.qsplit.addWidget(self.vsplit)
        self.quality_right = QWidget()
        QVBoxLayout(self.quality_right).setContentsMargins(0, 0, 0, 0)
        self.quality_right.setVisible(False)
        self.qsplit.addWidget(self.quality_right)
        self.qsplit.setStretchFactor(0, 3)
        self.qsplit.setStretchFactor(1, 1)
        self.qsplit.splitterMoved.connect(
            lambda *_a: self.remember_sizes("volume.quality", self.qsplit))
        self.qouter = Splitter(Qt.Orientation.Vertical)
        self.qouter.setHandleWidth(2)
        self.qouter.setChildrenCollapsible(False)
        self.qouter.addWidget(self.qsplit)
        self.quality_below = QWidget()
        QVBoxLayout(self.quality_below).setContentsMargins(0, 0, 0, 0)
        self.quality_below.setVisible(False)
        self.qouter.addWidget(self.quality_below)
        self.qouter.setStretchFactor(0, 3)
        self.qouter.setStretchFactor(1, 1)
        self.qouter.splitterMoved.connect(
            lambda *_a: self.remember_sizes("volume.quality_below", self.qouter))

        images = QWidget()
        row = QHBoxLayout(images)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.qouter, 1)
        self.side_tab = SideTab("Advanced controls", "controls")
        self.side_tab.clicked.connect(lambda: self.set_inspector(not self.inspector_open()))
        # Into the row BEFORE it is made visible: shown with no parent it is
        # a window for a moment, takes the focus, and on macOS the main
        # window then comes to the front over a preview window that had just
        # opened (GUI_LOOK_AND_FEEL 4.26).
        row.addWidget(self.side_tab)
        self.side_tab.setVisible(bool(getattr(self.viewer, "panels", True)))
        self._tab_row = row
        #: A host showing the column outside these images (two viewers driven
        #: as one keep ONE column, beside both), else None.
        self._host = None
        #: The viewer whose column this one's controls open instead (the
        #: other half of a synced pair), else None.
        self._controls_peer = None
        self.hsplit.addWidget(images)
        self.side = QScrollArea()
        self.side.setObjectName("viz-side-scroll")
        self.side.viewport().setObjectName("viz-side-viewport")
        self.side.setWidgetResizable(True)
        self.side.setFrameShape(QFrame.Shape.NoFrame)
        self.side.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.side.setMinimumWidth(300)
        self.side.setVisible(False)
        self.hsplit.addWidget(self.side)
        self.hsplit.setStretchFactor(0, 1)
        self.hsplit.setStretchFactor(1, 0)
        self.hsplit.splitterMoved.connect(
            lambda *_a: self.remember_sizes("volume.side", self.hsplit))
        self._inspector = None
        return self.hsplit

    def _page(self, mode: str) -> QWidget:
        """The page a mode shows: the tiled page, or the mosaic."""
        if mode != "mosaic":
            return self.tiles
        if self._mosaic_page is None:
            from ..canvases.mosaic import MosaicPage

            self._mosaic_page = MosaicPage(self.ctx)
            self.pages.addWidget(self._mosaic_page)
        return self._mosaic_page

    def _ensure_render(self):
        if self._render is not None:
            return self._render
        from ..canvases.render import RenderCanvas

        self._render = RenderCanvas(self.ctx)
        self.tiles.set_render(self._render)
        return self._render

    def _ensure_graph(self):
        if self._graph is None:
            from ..canvases.graph import TimecourseGraph

            self._graph = TimecourseGraph(self.ctx)
            self._graph.wants_height.connect(self._grow_graph)
            self.graph_host.layout().addWidget(self._graph)
            am = self.viewer.action_manager
            # Icons in the graph's corner: the tooltip names each, with its key.
            self._graph.add_panel_buttons(
                [am.button(a) for a in ("graph.beside", "graph.maximize", "graph.detach")])
            self._graph_context_source = None
            self._sync_graph_context()
        return self._graph

    def _sync_graph_context(self) -> None:
        """Give the graph the run context of the series it plots (the base's,
        or an overlaid BOLD's own), once per change of series."""
        if self._graph is None:
            return
        layer, _src = views.series_layer(self.ctx.store)
        source = layer.source if layer is not None else None
        if source == self._graph_context_source:
            return
        self._graph_context_source = source
        self._graph.set_context(self._contexts.get(source) if source else None)

    def _grow_graph(self, pixels: int) -> None:
        """Give the graph pane ``pixels`` if it has less, never more than
        three fifths of the viewer (the images stay the point)."""
        sizes = self.vsplit.sizes()
        if len(sizes) < 2 or not self.graph_host.isVisible():
            return
        total = sum(sizes)
        want = min(int(pixels), int(total * 0.6))
        if sizes[-1] >= want:
            return
        self.vsplit.setSizes([total - want] + [0] * (len(sizes) - 2) + [want])

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def three_d_capable(self) -> bool:
        src = self.source
        if src is None:
            return False
        if any(d < 2 for d in src.spatial):
            return False
        return not src.is_rgb or src.channels in (3, 4)

    def render_allowed(self) -> bool:
        """Whether the 3-D view can show this image here."""
        return bool(self._render is not None and self.ctx.gpu_ok and self.three_d_capable())

    def allowed_mode(self, mode: str) -> str:
        if mode in ("3d", "combo") and not (self.ctx.gpu_ok and self.three_d_capable()):
            return "multi"
        return mode

    def apply_mode(self) -> None:
        scene = self.ctx.scene
        if self.source is None:
            # Nothing is open (the Editor cleared the viewer for a JSON or a
            # table): the arrangement waits for the next image, which is
            # judged when it opens. Judged here, no image is "3-D capable",
            # so a 3-D layout was rewritten to three planes and the next
            # image of the same kind opened without its 3-D.
            return
        mode = self.allowed_mode(scene.mode)
        if mode != scene.mode:
            # A state asking for 3-D arrives on machines without a GPU as a
            # matter of course (a scene saved on a workstation): fall back to
            # the three planes, never to a single slice.
            scene.mode = mode
        if mode in ("3d", "combo"):
            try:
                self._ensure_render()
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not open the 3-D view: %s", exc)
                scene.mode = mode = "multi"
        page = self._page(mode)
        if page is self.tiles:
            self.tiles.apply()
        self.pages.setCurrentWidget(page)
        # The graph below or beside the views.
        want = (Qt.Orientation.Horizontal if scene.layout.graph == "right"
                else Qt.Orientation.Vertical)
        if self.vsplit.orientation() != want:
            self.vsplit.setOrientation(want)
        if self._render is not None and self._render.isVisible():
            self._render.refresh_volume()

    def apply_graph(self) -> None:
        series, _src = views.series_layer(self.ctx.store)
        # In every layout, the 3-D one included: the time course is the
        # crosshair's voxel, and the crosshair is in the rendering too.
        show = bool(self.ctx.scene.graph_visible and series is not None)
        if show:
            self._ensure_graph()
            if self._graph_window is not None:
                self.graph_host.setVisible(False)
                if self._graph_window.isHidden():
                    self._graph_window.show()
            elif not self.graph_host.isVisible():
                self.graph_host.setVisible(True)
                self.restore_sizes("volume.graph", self.vsplit, [0.7, 0.3])
        else:
            if self._graph_window is not None:
                self.set_graph_detached(False, apply=False)
            self.graph_host.setVisible(False)
        # Maximised: the views give the panel all the room (never with the
        # panel in its own window, which would leave the viewer empty).
        hide_views = bool(show and self._graph_maximized and self._graph_window is None)
        if self.pages.isHidden() != hide_views:
            self.pages.setVisible(not hide_views)

    def set_graph_maximized(self, on: bool) -> None:
        """The time course alone in the viewer, or beside the views again."""
        self._graph_maximized = bool(on)
        self.apply_graph()
        self.viewer.refresh_actions()

    def set_graph_detached(self, on: bool, *, apply: bool = True) -> None:
        """Move the time-course panel to its own window, or back. Its state
        is the scene's either way: closing the window loses nothing."""
        if on and self._graph_window is None:
            graph = self._ensure_graph()
            win = _GraphWindow(self.viewer)
            win.setWindowTitle(f"Time course: {self.source.path.name}"
                               if self.source is not None else "Time course")
            size = self.graph_host.size()
            win.take(graph)
            win.resize(max(size.width(), 640), max(size.height(), 320))
            win.closed.connect(lambda: self.set_graph_detached(False))
            self._graph_window = win
            self._graph_maximized = False
        elif not on and self._graph_window is not None:
            win, self._graph_window = self._graph_window, None
            if self._graph is not None:
                self.graph_host.layout().addWidget(self._graph)
                self._graph.setVisible(True)
            win.closed.disconnect()
            win.close()
            win.deleteLater()
        if apply:
            self.apply_graph()
        self.viewer.refresh_actions()

    # ------------------------------------------------------------------
    # Toolbar widgets
    # ------------------------------------------------------------------

    def toolbar_rows(self):
        return TOOLBAR_ROWS

    def make_widget(self, name: str):
        """A toolbar item by name: one widget, a list of widgets (each its
        own wrap point), or None."""
        return {
            "tools": self._widget_tools,
            "slice": self._widget_slice, "frame": self._widget_frame,
            "save": self._widget_save, "layout": self._widget_layout,
            "quality": self._widget_quality,
        }.get(name, lambda: None)()

    def _widget_layout(self) -> QWidget:
        """What is shown, in ONE menu: a single plane (axial, coronal,
        sagittal) or a layout of several views. The button names what is on
        screen, with its icon; the keys (A, C, S, M, P, D) still switch."""
        from ... import icons

        btn = QPushButton("Layout")
        btn.setObjectName("tb-btn-toggle")
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setIconSize(QSize(16, 16))
        btn.setToolTip("What the viewer shows: one plane, the three planes, the planes "
                       "with the 3-D, the 3-D alone, one large view, or a mosaic of "
                       "slices. More in Controls > Layout and Mosaic.")
        menu = popup_menu(btn)
        menu.setToolTipsVisible(True)
        for action_id in PLANE_ACTIONS:
            menu.addAction(self.viewer.action(action_id))
        menu.addSeparator()
        for action_id in LAYOUT_ACTIONS:
            menu.addAction(self.viewer.action(action_id))
        menu.addSeparator()
        options = menu.addAction("Layout and mosaic options...")
        options.setIcon(icons.icon("controls"))
        options.setToolTip("The arrangement, the planes shown, the large view, the "
                           "mosaic: in the advanced controls")
        options.triggered.connect(lambda: self.open_section("layout"))
        btn.setMenu(menu)
        self.layout_button = btn
        return btn

    def save_mosaic_figure(self, path: Optional[Path] = None, *, scale: Optional[float] = None,
                           transparent: Optional[bool] = None) -> Optional[Path]:
        """The mosaic as a PNG: shown first if it is not, then saved at the
        chosen resolution (asked unless given)."""
        if self.source is None:
            return None
        if self.ctx.scene.mode != "mosaic":
            self.ctx.run("view.mode", mode="mosaic")
            self.ctx.qstore.flush()
        if scale is None or transparent is None:
            from PyQt6.QtWidgets import QDialog

            from ..panels.figure_dialog import FigureDialog

            dlg = FigureDialog("Save the mosaic figure", parent=self.viewer)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return None
            scale, transparent = dlg.values()
        return self.viewer.save_screenshot(path, scale=scale, transparent=transparent)

    def open_section(self, key: str) -> None:
        """Open the controls column at one of its sections."""
        self.set_inspector(True)
        insp = self._inspector
        if insp is None:
            return
        section = insp.section(key)
        if section is not None:
            section.set_open(True)
            QTimer.singleShot(0, lambda: self.side.ensureWidgetVisible(section, 0, 0))

    #: The quality items for a 4-D image, all computed only when chosen.
    QUALITY = ("qc.check", "qc.noise", "", "qc.tsnr", "qc.sd", "qc.mean")

    def _widget_quality(self) -> QWidget:
        """Every quality measure of a series in one place, each computed only
        when chosen."""
        from ... import icons

        btn = QPushButton("QC")
        btn.setObjectName("tb-btn-toggle")
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setIcon(icons.icon("qc"))
        btn.setIconSize(QSize(16, 16))
        self.viewer.action_manager.track_icon(btn, "qc")
        btn.setToolTip("Quality control, computed only when you ask: the quality check of an "
                       "anatomical or diffusion image (measures, findings, the evidence over "
                       "the image), the image windowed to its noise, and for a series temporal "
                       "SNR, standard deviation and mean maps and QC plots under the time "
                       "course.")
        menu = popup_menu(btn)
        menu.setToolTipsVisible(True)
        for action_id in self.QUALITY:
            if action_id:
                menu.addAction(self.viewer.action(action_id))
            else:
                menu.addSeparator()
        menu.addSeparator()
        self._qc_plots_action = menu.addAction("QC plots under the time course")
        self._qc_plots_action.setCheckable(True)
        self._qc_plots_action.setToolTip(
            "Framewise displacement, the motion parameters, DVARS and outlier voxels per "
            "volume (choose more with Plots): opens the time course if it is closed")
        self._qc_plots_action.triggered.connect(self._toggle_qc_plots)
        self._qc_on_open_action = menu.addAction("Run QC when a file opens")
        self._qc_on_open_action.setCheckable(True)
        self._qc_on_open_action.setToolTip(
            "On: QC stays on from one file to the next and is computed as soon as a file "
            "opens (the quality check too, while its panel is open). Off: every file opens "
            "with QC off.")
        self._qc_on_open_action.triggered.connect(
            lambda on: self.ctx.settings_hub.update(
                lambda st: setattr(st.qc, "on_open", bool(on))))

        def sync_menu() -> None:
            self._qc_plots_action.setChecked(bool(self.ctx.scene.graph.qc))
            self._qc_on_open_action.setChecked(bool(self.ctx.settings.qc.on_open))
            # The plots run under a time course: only a series has one.
            series = views.series_layer(self.ctx.store)[1]
            is_series = bool(series is not None and getattr(series, "is_4d", False))
            self._qc_plots_action.setVisible(is_series)
            self._qc_on_open_action.setVisible(is_series or self._quality_kind() is not None)

        menu.aboutToShow.connect(sync_menu)
        btn.setMenu(menu)
        self.quality_button = btn
        return btn

    def _toggle_qc_plots(self, on: bool) -> None:
        if on and not self.ctx.scene.graph_visible:
            self.ctx.run("view.graph", value=True)
        self.ctx.run("graph.set", qc=bool(on))

    def _number_group(self, title: str, tip: str):
        """A titled slider-and-number (slice, volume): typable, and wheel
        only when focused."""
        from ..controls import NumberControl

        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        hdr = QLabel(title)
        hdr.setObjectName("sidecar-footer-summary")
        h.addWidget(hdr)
        nc = NumberControl(0, 1, step=1, decimals=0)
        nc.slider.setMinimumWidth(80)
        nc.slider.setFixedWidth(84)
        # As wide as "1128 / 1128" in its own font: a fixed 96 px pushed the
        # toolbar onto a second row at 1400 px.
        nc.spin.ensurePolished()
        nc.spin.setFixedWidth(nc.spin.fontMetrics().horizontalAdvance("0000 / 0000") + 22)
        nc.setToolTip(tip)
        h.addWidget(nc)
        return box, hdr, nc

    def _widget_slice(self) -> QWidget:
        box, self._slice_hdr, self.slice_control = self._number_group(
            "Slice", "The slice shown; type a number, or drag")
        self.slice_control.value_changed.connect(self._on_slice_control)
        return box

    def _widget_frame(self) -> QWidget:
        box, self._frame_hdr, self.frame_control = self._number_group(
            "Volume", "The volume of the 4-D series shown; type a number, or drag")
        self._frame_group = box
        self.frame_control.value_changed.connect(
            lambda v: self._run_ui("frame.set", frame=int(v)))
        self.frame_control.pressed.connect(self.ctx.store.begin_gesture)
        self.frame_control.released.connect(self.ctx.store.end_gesture)
        return box

    #: The Tools menu, in order ("" is a separator). Each entry is an action,
    #: so it also has a key, a palette entry and a help line.
    TOOLS = ("overlay.add", "header.show", "deface.preview", "",
             "frame.same_shell", "frame.shell", "frame.shell_prev")

    def _widget_tools(self) -> QWidget:
        """What can be done with the open image beyond looking at it."""
        from ... import icons

        btn = QPushButton("Tools")
        btn.setObjectName("tb-btn-toggle")
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setIcon(icons.icon("tools"))
        btn.setIconSize(QSize(16, 16))
        self.viewer.action_manager.track_icon(btn, "tools")
        btn.setToolTip("Add an overlay, inspect the header against the sidecar, preview "
                       "defacing, step through diffusion shells, and go to the same image "
                       "of another run, session or subject")
        menu = popup_menu(btn)
        self._fill_tools_menu(menu)
        # Rebuilt as it opens: the fieldmap entries depend on the image.
        menu.aboutToShow.connect(lambda m=menu: self._fill_tools_menu(m))
        btn.setMenu(menu)
        self.tools_button = btn
        return btn

    def _fill_tools_menu(self, menu: QMenu) -> None:
        menu.clear()
        for action_id in self.TOOLS:
            if action_id:
                menu.addAction(self.viewer.action(action_id))
            else:
                menu.addSeparator()
        menu.addSeparator()
        go = submenu(menu, "Go to")
        for action_id in self.NAVIGATION:
            go.addAction(self.viewer.action(action_id))
        bids = self.bids
        if bids is not None and bids.fieldmaps:
            maps = submenu(menu, "Fieldmaps for this image")
            for fmap in bids.fieldmaps:
                act = maps.addAction(f"Draw {fmap.name} over this")
                act.triggered.connect(lambda _c=False, f=fmap: self.add_overlay(f))
        if bids is not None and bids.datatype == "fmap" and bids.intended_for:
            meant = submenu(menu, "Images this fieldmap is for")
            for target in bids.intended_for:
                act = meant.addAction(f"Open {target.name} with this over it")
                act.setEnabled(target.is_file())
                act.triggered.connect(lambda _c=False, t=target: self.open_under(t))

    #: The "Go to" submenu.
    NAVIGATION = ("nav.run_next", "nav.run_prev", "nav.echo_next", "nav.echo_prev",
                  "nav.ses_next", "nav.ses_prev", "nav.sub_next", "nav.sub_prev")

    def open_under(self, target: Path) -> None:
        """Open ``target`` and draw the image on screen over it (a fieldmap
        over the run it corrects)."""
        if self.source is None:
            return
        overlay = self.source.path
        self.viewer.navigate(Path(target))
        self.viewer.add_overlay(overlay)

    def navigate(self, entity: str, step: int) -> bool:
        """The previous (``step`` < 0) or next file along ``entity``."""
        found = (self.bids.neighbours.get(entity) if self.bids is not None else None)
        if not found:
            return False
        target = found[0] if step < 0 else found[1]
        if target is None:
            return False
        self.viewer.navigate(target)
        return True

    def carry_cursor(self) -> None:
        """Keep the crosshair's world position for the next file opened."""
        world = self.ctx.scene.cursor.world
        self._carried_cursor = tuple(world) if world is not None else None

    def _widget_save(self) -> QWidget:
        """What can be KEPT, each saying what and where: a picture (screenshot,
        mosaic figure), the way the viewer looks under a name (a preset, for
        any image), this image with its overlays and crosshair (a scene, in
        the dataset), the command line that reproduces the view."""
        from ... import icons

        btn = QPushButton("Save")
        btn.setObjectName("tb-btn-toggle")
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setIcon(icons.icon("save"))
        btn.setIconSize(QSize(16, 16))
        self.viewer.action_manager.track_icon(btn, "save")
        btn.setToolTip("Save a picture, a preset of how the viewer looks, a scene of this "
                       "image in the dataset, or the command line of this view. Everything "
                       "else (layouts, the look, options) is kept by itself.")
        menu = popup_menu(btn)
        menu.setToolTipsVisible(True)
        menu.aboutToShow.connect(lambda m=menu: self._fill_save_menu(m))
        self._fill_save_menu(menu)
        btn.setMenu(menu)
        self.save_button = btn
        return btn

    def _fill_save_menu(self, menu: QMenu) -> None:
        from ....viz import scenes

        menu.clear()
        menu.addAction(self.viewer.action("export.screenshot"))
        if "view.mosaic_figure" in self.viewer.action_manager.actions:
            menu.addAction(self.viewer.action("view.mosaic_figure"))
        menu.addSeparator()
        menu.addAction(self.viewer.action("views.save"))
        names = sorted(self.ctx.settings.view_presets, key=str.lower)
        if names:
            presets = submenu(menu, "Apply a preset")
            for name in names:
                act = presets.addAction(name)
                act.triggered.connect(lambda _c=False, n=name: self.apply_view(n))
            delete = submenu(menu, "Delete a preset")
            for name in names:
                act = delete.addAction(name)
                act.triggered.connect(lambda _c=False, n=name: self.delete_view(n))
        menu.addSeparator()
        menu.addAction(self.viewer.action("scene.save"))
        found = scenes.list_scenes(self.scene_root())
        if found:
            sub = submenu(menu, "Open a scene of this dataset")
            for name, path in found:
                act = sub.addAction(name)
                act.triggered.connect(lambda _c=False, p=path: self.open_scene(p))
        menu.addSeparator()
        menu.addAction(self.viewer.action("view.command_line"))
        menu.addSeparator()
        menu.addAction(self.viewer.action("view.restore_defaults"))

    # -- saved views (also the scripting surface) ------------------------

    def _has_series(self) -> bool:
        return views.series_layer(self.ctx.store)[0] is not None

    def save_view(self, name: str) -> None:
        """Save the look as a preset: only what this image has (see
        :mod:`bidsmgr.viz.presets`)."""
        from ....viz import presets

        name = name.strip()
        if not name:
            return
        state = presets.snapshot(self.ctx.scene, has_series=self._has_series())
        self.ctx.settings_hub.update(lambda s: s.view_presets.__setitem__(name, state))
        self.viewer.status_message.emit(f"Saved the view {name!r}")

    def apply_view(self, name: str) -> bool:
        """Apply a preset: the image on screen takes what it has."""
        from ....viz import presets

        state = self.ctx.settings.view_presets.get(name)
        if state is None:
            return False
        self.viewer.apply_state(presets.applicable(state, self.ctx.scene,
                                                   has_series=self._has_series()))
        self.ctx.qstore.flush()
        return True

    def delete_view(self, name: str) -> None:
        if name in self.ctx.settings.view_presets:
            self.ctx.settings_hub.update(lambda s: s.view_presets.pop(name, None))

    # -- scenes: saved WITH the dataset ------------------------------------

    def scene_root(self) -> Optional[Path]:
        from ....viz.bids import dataset_root

        root = self.viewer.current_root()
        if root is None and self.source is not None:
            root = dataset_root(self.source.path)
        return root

    def save_scene(self, name: str) -> Optional[Path]:
        """Save what is on screen as a scene of the dataset."""
        from ....viz import scenes

        name = name.strip()
        root = self.scene_root()
        if not name or root is None or self.source is None or self.viewer.read_only:
            return None
        data = scenes.snapshot(self.ctx.scene, self.ctx.store.sources, root, name)
        path = scenes.save(root, data)
        self.viewer.status_message.emit(
            f"Saved the scene {name!r} in the dataset ({scenes.SCENES.as_posix()})")
        return path

    def open_scene(self, path: Path) -> bool:
        """Open a saved scene: its image, its overlays in their looks, its
        crosshair and layout."""
        from ....viz import scenes

        try:
            data = scenes.load(Path(path))
        except (OSError, ValueError) as exc:
            self.viewer.status_message.emit(f"Could not open the scene: {exc}")
            return False
        root = self.scene_root() or Path(path).parents[3]
        base = scenes.from_rel(root, data["base"])
        if not base.is_file():
            self.viewer.status_message.emit(f"The scene's image is gone: {data['base']}")
            return False
        data["_root"] = str(root)
        self.viewer.navigate(base, root)
        # Set AFTER navigating: opening the image clears what was pending.
        self._pending_scene = data
        return True

    def _apply_scene(self, data: dict) -> None:
        from ....viz import scenes

        from ....viz import presets

        root = Path(data.get("_root") or ".")
        view = dict(data.get("view") or {})
        cursor = (view.pop("cursor", None) or {}).get("world")
        self.viewer.apply_state(presets.applicable(view, self.ctx.scene,
                                                   has_series=self._has_series()))
        base = data.get("base_display")
        if base:
            self.ctx.run("layer.set", **{k: v for k, v in base.items()
                                         if k in _SCENE_DISPLAY_KEYS and v is not None})
        if data.get("frame"):
            self.ctx.run("frame.set", frame=int(data["frame"]))
        if cursor is not None:
            self.ctx.run("cursor.set_world", x=float(cursor[0]), y=float(cursor[1]),
                         z=float(cursor[2]), snap=False)
        gone = []
        for item in data.get("overlays", []):
            look = {"display": item.get("display"), "name": item.get("name", ""),
                    "visible": bool(item.get("visible", True)),
                    "in_3d": bool(item.get("in_3d", True))}
            origin = str(item.get("origin") or "")
            if origin.startswith("qc:"):
                self._pending_qc.append({"which": origin[3:], "look": look})
            elif origin.startswith("deface:"):
                self.preview_defacing(origin[len("deface:"):], **look)
            elif scenes.from_rel(root, item["path"]).is_file():
                self.add_overlay(scenes.from_rel(root, item["path"]), **look)
            else:
                gone.append(item["path"])
        self.ctx.qstore.flush()
        message = f"Opened the scene {data.get('name', '')!r}"
        if gone:
            message += (f"; its overlay{'s are' if len(gone) > 1 else ' is'} gone: "
                        + ", ".join(gone))
        self.viewer.status_message.emit(message)

    def _ask_scene_name(self) -> None:
        from PyQt6.QtWidgets import QInputDialog, QMessageBox

        from ....viz import scenes

        root = self.scene_root()
        if root is None or self.source is None:
            self.viewer.status_message.emit(
                "A scene is saved in a dataset: open an image from one first.")
            return
        name, ok = QInputDialog.getText(self.viewer, "Save as a scene of this dataset",
                                        "Name for this scene:")
        if not ok or not name.strip():
            return
        if scenes.scene_path(root, name.strip()).is_file():
            answer = QMessageBox.question(
                self.viewer, "Replace the scene",
                f"This dataset has a scene called {name.strip()!r}. Replace it?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.save_scene(name)

    def _ask_view_name(self) -> None:
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self.viewer, "Save this view",
                                        "Name for this view:")
        if ok and name.strip():
            if name.strip() in self.ctx.settings.view_presets:
                from PyQt6.QtWidgets import QMessageBox

                answer = QMessageBox.question(
                    self.viewer, "Replace the saved view",
                    f"A view called {name.strip()!r} exists. Replace it?")
                if answer != QMessageBox.StandardButton.Yes:
                    return
            self.save_view(name)

    _syncing_widgets = False

    def _run_ui(self, command_id: str, **params) -> None:
        if not self._syncing_widgets:
            self.ctx.run(command_id, **params)

    def _active_plane(self) -> str:
        scene = self.ctx.scene
        return scene.plane if scene.mode == "single" else self.ctx.active_plane

    def _on_slice_control(self, value: float) -> None:
        if self._syncing_widgets:
            return
        self.ctx.run("cursor.set_slice", plane=self._active_plane(), index=int(value))

    def sync_widgets(self) -> None:
        """Read the scene into the toolbar widgets (no command runs)."""
        store = self.ctx.store
        layer, src = views.base(store)
        self._syncing_widgets = True
        try:
            if hasattr(self, "layout_button"):
                from ... import icons

                mode = self.ctx.scene.mode
                name = (self.ctx.scene.plane.capitalize() if mode == "single"
                        else LAYOUT_NAMES.get(mode, mode))
                if self.layout_button.text() != name:
                    icon_name = LAYOUT_ICONS.get(mode, "layout_multi")
                    self.layout_button.setText(name)
                    self.layout_button.setIcon(icons.icon(icon_name))
                    self.viewer.action_manager.track_icon(self.layout_button, icon_name)
            if hasattr(self, "quality_button"):
                series = views.series_layer(store)[1]
                want = bool((series is not None and getattr(series, "is_4d", False))
                            or self._quality_kind() is not None)
                if self.quality_button.isHidden() == want:    # only on a real change
                    self.quality_button.setVisible(want)
            if hasattr(self, "slice_control"):
                plane = self._active_plane()
                n = views.slice_count(store, plane)
                self._slice_hdr.setText(f"Slice ({plane})" if self.ctx.scene.mode != "single"
                                        else "Slice")
                self.slice_control.set_range(0, max(n - 1, 0))
                self.slice_control.spin.setSuffix(f" / {max(n - 1, 0)}")
                self.slice_control.setEnabled(n > 1)
                self.slice_control.set_value(views.slice_index(store, plane))
            if hasattr(self, "frame_control"):
                s_layer, s_src = views.series_layer(store)
                if s_layer is not None:
                    layer, src = s_layer, s_src
                n = src.n_frames if src is not None else 1
                t = views.frame_of(store, layer, src) if src is not None and layer else 0
                self.frame_control.set_range(0, max(n - 1, 0))
                self.frame_control.spin.setSuffix(f" / {max(n - 1, 0)}")
                self.frame_control.setEnabled(n > 1)
                # A 3-D image has no volumes to step through: the control
                # is not there, rather than greyed (only on a real change).
                if self._frame_group.isHidden() == (n > 1):
                    self._frame_group.setVisible(n > 1)
                self.frame_control.set_value(t)
                title = "Volume"
                bvals = getattr(src, "bvals", None) if src is not None else None
                if bvals is not None and t < len(bvals):
                    title += f"  b={float(bvals[t]):g}"
                self._frame_hdr.setText(title)
        finally:
            self._syncing_widgets = False

    # ------------------------------------------------------------------
    # Context for actions
    # ------------------------------------------------------------------

    def action_context(self) -> dict[str, Any]:
        scene = self.ctx.scene
        src = self.source
        layer = scene.base_layer()
        has = src is not None and layer is not None
        ci = self.ctx.active_clip
        clip = scene.clips[ci] if ci < len(scene.clips) else None
        return {
            "volume": has,
            "volume.4d": views.series_layer(self.ctx.store)[0] is not None,
            "volume.3d": bool(has and self.three_d_capable()),
            "gpu": bool(self.ctx.gpu_ok),
            "slices": bool(has and scene.mode != "3d"),
            "render": bool(has and self.ctx.gpu_ok and scene.mode in ("3d", "combo", "hero")),
            "mode": scene.mode,
            "plane": scene.plane,
            "display.labels": scene.display.labels,
            "display.ras": scene.display.ras,
            "display.radiological": scene.display.radiological,
            "display.colorbar": scene.display.colorbar,
            "display.crosshair": scene.display.crosshair,
            "space": scene.display.space,
            "graph": scene.graph_visible,
            "graph.beside": scene.layout.graph == "right",
            "graph.maximized": bool(self._graph_maximized and scene.graph_visible),
            "graph.detached": self._graph_window is not None,
            "clip.active": bool(clip and clip.active),
            "clip.at_cursor": bool(scene.render.cut_at_cursor),
            "layer.invert": bool(layer.display.invert) if layer else False,
            "layer.nearest": bool(layer and layer.display.interpolation == "nearest"),
            "playing": self.ctx.playing,
            "panel.inspector": self.inspector_open(),
            "volume.loaded": bool(has and src.fully_loaded),
            "volume.dwi": bool(has and src.bvals is not None),
            "qc.checkable": bool(has and src.fully_loaded and self._quality_kind() is not None),
            "qc.noise": bool(self._noise_look is not None),
            "qc.panel": bool(self._quality_open),
            "qc.below": scene.layout.quality == "bottom",
            "qc.maximized": bool(self._quality_max and self._quality_window is None),
            "qc.detached": self._quality_window is not None,
            "volume.deface": bool(has and not src.is_4d and not src.is_rgb
                                  and self._deface_available()),
            **self._navigation_context(),
        }

    def _navigation_context(self) -> dict[str, bool]:
        out = {}
        found = self.bids.neighbours if self.bids is not None else {}
        for entity in ("run", "echo", "ses", "sub"):
            prev, nxt = found.get(entity, (None, None))
            out[f"nav.{entity}.prev"] = prev is not None
            out[f"nav.{entity}.next"] = nxt is not None
        return out

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: Path, root: Optional[Path]) -> None:
        # What was changed on the file on screen is kept under ITS kind
        # before the next file replaces the scene.
        if self._view_timer.isActive():
            self._view_timer.stop()
            self._remember_view()
        self._generation += 1
        self._pending_overlays = []
        self._pending_header = None
        self._pending_scene = None
        self._pending_qc = []
        self.stop_play()
        self._release()
        if self._quality_open and self._quality_panel is not None:
            # Never the last image's result over this one.
            self._quality_panel.set_busy(f"Reading {Path(path).name}")
        self._first_frame_shown = False
        self._show_when_windowed = False
        self._auto_window = None
        self.ctx.jobs.start("open", self._generation, open_with_context, path, root)

    def _release(self) -> None:
        for tag in self._overlay_tags:
            self.ctx.jobs.cancel(tag)
        self._overlay_tags = []
        self._overlay_looks = {}
        self.ctx.jobs.cancel("stream")
        self.ctx.jobs.cancel("open")
        self.ctx.jobs.cancel("render-volume")
        self.ctx.jobs.cancel("series-window")
        self.ctx.jobs.cancel("quality")
        self._noise_look = None
        self._quality_result = None
        self._evidence = {}
        if self.source is not None:
            self.source.release()
        self.source = None
        self.bids = None
        if self._render is not None:
            self._render.clear()

    def clear(self) -> None:
        # A change made just before (a 3-D effect still settling) is kept
        # under the kind of the image leaving, as ``load`` and ``stop`` do.
        if self._view_timer.isActive():
            self._view_timer.stop()
            self._remember_view()
        self._generation += 1
        self._pending_quality = False
        self._pending_overlays = []
        self._pending_header = None
        self._pending_scene = None
        self._pending_qc = []
        self.stop_play()
        self._release()
        if self._quality_panel is not None:
            self._quality_panel.set_empty("", reason="No image is open.")
        store = self.ctx.store
        store.sources.clear()
        keep = self._persistent_scene()
        store.replace_scene(keep)
        self.ctx.qstore.flush()

    def _persistent_scene(self) -> Scene:
        """A fresh scene that keeps the user's view preferences."""
        old = self.ctx.scene
        s = Scene()
        s.mode = old.mode
        s.plane = old.plane
        s.display = old.display.model_copy(deep=True)
        s.graph_visible = old.graph_visible
        s.graph = old.graph.model_copy(deep=True)
        s.render = old.render.model_copy(deep=True)
        s.clips = [c.model_copy() for c in old.clips]
        s.layout = old.layout.model_copy(deep=True)
        s.mosaic = old.mosaic
        s.mosaic_build = old.mosaic_build.model_copy()
        return s

    def initial_scene(self, src: VolumeSource, path: Path) -> Scene:
        """The scene a newly opened file starts in.

        The ARRANGEMENT (mode, plane, graph) follows the kind of file: a
        file of the kind already on screen keeps the current view (you are
        driving); a different kind opens in the arrangement that kind last
        had, else its preset (a BOLD run with its graph, a T1 without). The
        LOOK (gamma, colour map) is the user's default for new images, and
        the display conventions (RAS, radiological, labels) are kept from
        whatever was on screen, or read from the settings for the first
        file a viewer opens.
        """
        settings = self.ctx.settings
        vs = settings.volume
        prefs = self._persistent_scene()
        sid = "vol0"
        prefs.sources = {sid: SourceRef(id=sid, path=str(path), kind="volume")}
        display = VolumeDisplay(gamma=vs.gamma, interpolation=vs.interpolation,
                                colormap=vs.colormap if colormaps.exists(vs.colormap) else "gray")
        prefs.layers = [VolumeLayer(id="base", source=sid, name=path.name, display=display)]
        prefs.cursor = Cursor(world=tuple(float(v) for v in src.center_world()))
        carried, self._carried_cursor = self._carried_cursor, None
        if carried is not None and views.voxel_in(src, np.asarray(carried, dtype=float)):
            # Moved here from another run or subject: the same spot.
            prefs.cursor = Cursor(world=tuple(float(v) for v in carried))
        if self.viewer.first_open:
            prefs.display.labels = vs.labels
            prefs.display.colorbar = vs.colorbar
            prefs.display.ras = vs.ras
            prefs.display.radiological = vs.radiological
            prefs.display.space = vs.space
            prefs.plane = vs.plane
            # The look as last left, in any window: display conventions, the
            # 3-D effect and its parameters, the clip planes.
            from ....viz import memory

            memory.apply_volume_look(prefs, settings.volume_look)
        bids = self.bids
        preset = layouts.match(bids.datatype if bids else "", bids.suffix if bids else "",
                               bool(src.is_4d and not src.is_rgb))
        if self.viewer.first_open or preset.id != self._preset_id:
            default_mode = vs.mode or ("combo" if self.ctx.gpu_ok else "multi")
            view = layouts.opening_view(preset, settings.layout_state.get(preset.id),
                                        default_mode)
            prefs.mode = view["mode"]
            prefs.plane = view.get("plane", prefs.plane)
            prefs.graph_visible = bool(view.get("graph_visible", False))
            if view.get("graph"):
                prefs.graph = GraphState.model_validate(
                    {**prefs.graph.model_dump(), **view["graph"]})
            # The layout this kind was given (arrangement, planes, large view,
            # graph placement) and its mosaic: they were remembered and never
            # read back, so a new window lost every arrangement.
            if isinstance(view.get("layout"), dict):
                try:
                    prefs.layout = LayoutState.model_validate(
                        {**prefs.layout.model_dump(), **view["layout"]})
                except ValueError:
                    pass
            if isinstance(view.get("mosaic"), str) and view["mosaic"]:
                prefs.mosaic = view["mosaic"]
            if isinstance(view.get("mosaic_build"), dict):
                try:
                    prefs.mosaic_build = MosaicBuild.model_validate(view["mosaic_build"])
                except ValueError:
                    pass
        if not settings.qc.on_open:
            # QC is computed when asked, not because the last file had it on.
            prefs.graph.qc = False
        self._preset_id = preset.id
        return prefs

    @property
    def layout_id(self) -> str:
        """The kind of file on screen, as a layout preset id."""
        return self._preset_id

    def _on_job_done(self, tag: str, generation: int, result) -> None:
        if generation != self._generation:
            return
        if tag == "quality":
            self._quality_result = result
            self.show_quality(result)
            return
        if tag in self._overlay_tags:
            self._overlay_tags.remove(tag)
            look = self._overlay_looks.pop(tag, {})
            overlay, run_ctx = result if isinstance(result, tuple) else (result, None)
            n = tag.split(":", 1)[1]
            if run_ctx is not None:
                self._contexts[f"ovl{n}"] = run_ctx
            self._adopt_overlay(n, overlay, **look)
            return
        if tag == "open":
            src, bids = result
            self.source = src
            self.bids = bids
            self._contexts = {"vol0": bids}
            bvals = getattr(bids, "bvals", None)
            if bvals is not None and len(bvals) == src.n_frames_total and src.n_frames > 1:
                # Only when they describe THESE volumes: the inspector says
                # when they do not.
                src.bvals = bvals
            store = self.ctx.store
            store.sources = {"vol0": src}
            self._remembering = False
            try:
                scene = self.initial_scene(src, src.path)
                store.replace_scene(scene)
                self.viewer.first_open = False
                # Unconditionally re-validate the layout for this file.
                scene.mode = self.allowed_mode(scene.mode)
            finally:
                self._remembering = True
            self._graph_context_source = None
            self._sync_graph_context()
            budget = memory_budget_bytes(self.ctx.settings.volume.memory_mb)
            self.ctx.jobs.start("stream", self._generation, src.stream,
                                budget_bytes=budget)
            self.ctx.qstore.flush()
            pending, self._pending_overlays = self._pending_overlays, []
            for path, look in pending:
                self.add_overlay(path, **look)
            if self._pending_header is not None:
                rule, self._pending_header = self._pending_header, None
                self.show_header(rule)
            if self._pending_scene is not None:
                data, self._pending_scene = self._pending_scene, None
                self._apply_scene(data)
        elif tag == "stream":
            src = self.source
            held = self._holding()
            self._auto_window_from_frame()
            self.ctx.store.changed({"sources:vol0"})
            self.ctx.qstore.flush()
            if src is not None and src.is_4d and not src.is_rgb:
                # The whole series, sampled, on a worker: PET's first frames
                # hold almost no counts, so a window taken from frame 0 alone
                # saturates every later one.
                self.ctx.jobs.start("series-window", self._generation, src.series_range)
            if held and src is not None and not src.is_rgb:
                # Shown once its contrast is the series', not frame 0's.
                self._show_when_windowed = True
                self.viewer.set_loading_message(f"Preparing {src.path.name}")
                return
            self._show_read()
        elif tag == "series-window":
            self._apply_series_window(result)
            if self._show_when_windowed:
                self._show_read()

    # ------------------------------------------------------------------
    # The window a volume opens with
    # ------------------------------------------------------------------

    def _auto_window_from_frame(self) -> None:
        """Fill the window ONCE, from the frame on screen.

        ``None`` in the scene means "not chosen yet". Left as ``None``, every
        canvas computed a robust range per frame, so playback recomputed a
        percentile on every step and the brightness flickered from frame to
        frame. Written into the scene, the contrast stays put while frames
        change, and it is a value a linked viewer and a saved scene can copy.
        """
        store = self.ctx.store
        layer, src = views.base(store)
        if layer is None or src is None or src.is_rgb or layer.display.window is not None:
            return
        rng = src.robust_range(views.frame_of(store, layer, src))
        if rng is None:
            return
        layer.display.window = rng
        self._auto_window = rng
        store.changed({f"layer:{layer.id}.display"})

    def _apply_series_window(self, rng) -> None:
        """Widen the opening window to the series, unless the user already
        chose one (any change away from the automatic value counts)."""
        store = self.ctx.store
        layer, _src = views.base(store)
        if layer is None or rng is None or self._auto_window is None:
            return
        if layer.display.window != self._auto_window:
            return
        rng = (float(rng[0]), float(rng[1]))
        if rng == self._auto_window:
            return
        layer.display.window = rng
        self._auto_window = rng
        store.changed({f"layer:{layer.id}.display"})
        self.ctx.qstore.flush()

    def _show_read(self) -> None:
        """The file is read (and a series windowed): on screen, whole."""
        self._show_when_windowed = False
        self._first_frame_shown = True
        pending_qc, self._pending_qc = self._pending_qc, []
        self.viewer.on_loaded(self.source.path if self.source else None)
        for item in pending_qc:
            self.add_quality_map(item["which"], **item["look"])
        if self._pending_quality:
            self._pending_quality = False
            self.check_quality()
        elif self._quality_open:
            self._follow_quality()

    def _on_job_failed(self, tag: str, generation: int, message: str) -> None:
        if generation == self._generation and tag == "quality":
            panel = self._quality_panel
            if panel is not None and self.source is not None:
                panel.set_empty(self.source.path.name,
                                reason=f"The quality check failed: {message}")
            self.viewer.status_message.emit(f"The quality check failed: {message}")
            return
        if (generation == self._generation and tag == "series-window"
                and self._show_when_windowed):
            # The contrast stays frame 0's; the series is shown all the same.
            self._show_read()
            return
        if generation == self._generation and tag in self._overlay_tags:
            self._overlay_tags.remove(tag)
            self._overlay_looks.pop(tag, None)
            self.viewer.loading_changed.emit(False, "")
            self.viewer.status_message.emit(f"Could not add the overlay: {message}")
            return
        if generation != self._generation or tag not in ("open", "stream"):
            return
        path = self.viewer.current_file()
        self._release()
        if self._quality_panel is not None and path is not None:
            self._quality_panel.set_empty(Path(path).name, reason="The image could not be read.")
        self.viewer.on_load_failed(path, message)

    def _holding(self) -> bool:
        """A series stays behind the loading page until it is read whole:
        volumes appearing one by one, the graph growing and the contrast
        changing as they arrive, read as something going wrong."""
        src = self.source
        return bool(src is not None and src.is_4d and src.n_frames > 1
                    and not self._first_frame_shown)

    def _on_job_progress(self, tag: str, generation: int, done: int, total: int) -> None:
        if tag == "quality" and generation == self._generation and self.source is not None:
            if self._quality_panel is not None:
                self._quality_panel.set_progress(
                    f"Checking the quality of {self.source.path.name}: step {done} of {total}")
            return
        if tag != "stream" or generation != self._generation or self.source is None:
            return
        now = time.monotonic()
        if self._holding():
            if done >= total or now - self._last_progress >= _PROGRESS_INTERVAL:
                self._last_progress = now
                n = self.source.n_frames
                got = min(self.source.loaded_frames, n)
                self.viewer.set_loading_message(
                    f"Loading {self.source.path.name}: {got} of {n} volumes")
            return
        first = not self._first_frame_shown and self.source.loaded_frames > 0
        if first:
            self._first_frame_shown = True
            self._auto_window_from_frame()
            self.viewer.on_first_frame()
        if first or done >= total or now - self._last_progress >= _PROGRESS_INTERVAL:
            self._last_progress = now
            self.ctx.store.changed({"sources:vol0"})

    # ------------------------------------------------------------------
    # Scene changes
    # ------------------------------------------------------------------

    def _on_changed(self, paths) -> None:
        if paths & {"mode", "scene", "layout", "plane"}:
            self.apply_mode()
        if "layout" in paths or "scene" in paths:
            self.apply_quality()
        if self._evidence and any(p == "layers" or p.startswith("layer") for p in paths):
            self._sync_evidence_boxes()
        if self._quality_panel is not None and any(
                p in ("graph", "graph_visible") or p.startswith("graph.") for p in paths):
            self._quality_panel.set_plots(bool(self.ctx.scene.graph_visible
                                               and self.ctx.scene.graph.qc))
        if (paths & {"mode", "scene"} or "layers" in paths) and self.ctx.scene.mode == "mosaic":
            # A mosaic fitted to the head on screen (a new file, a new head),
            # unless its line was written by hand.
            self.ctx.run("mosaic.refresh")
        if self._remembering and self.source is not None:
            # A choice (a layout, a plane, a display flag, an effect) is
            # written at once: the Editor is not always closed cleanly. What
            # moves continuously (3-D parameters, the camera, clip planes) is
            # written once it settles.
            if any(p in ("mode", "plane", "graph", "graph_visible", "layout", "mosaic",
                         "render.effect") or p.startswith(("display", "graph."))
                   for p in paths):
                self._view_timer.stop()
                self._remember_view()
            elif any(p.startswith(("render", "clips")) for p in paths):
                self._view_timer.start()
        if any(p in ("graph_visible", "mode", "scene", "layers", "graph") or p.startswith("sources")
               for p in paths):
            self.apply_graph()
            self._sync_graph_context()
        self.sync_widgets()
        self.viewer.update_footer()

    def default_look(self, layer) -> dict:
        """The look ``layer`` opened with: the user's defaults for new images
        for the base, the look its content called for for an overlay."""
        base = self.ctx.scene.base_layer()
        if base is not None and layer.id != base.id and layer.id in self._original_looks:
            return dict(self._original_looks[layer.id])
        vs = self.ctx.settings.volume
        return VolumeDisplay(gamma=vs.gamma, interpolation=vs.interpolation,
                             colormap=vs.colormap if colormaps.exists(vs.colormap)
                             else "gray").model_dump()

    def restore_all_defaults(self, *, confirm: bool = True) -> bool:
        """Every viewer setting as installed (shortcuts, the mouse map and
        saved presets kept), and the view on screen with it."""
        from PyQt6.QtWidgets import QMessageBox

        from ....viz import memory
        from ....viz.scene import RenderState
        from ..link import apply_view_state

        if confirm:
            answer = QMessageBox.question(
                self.viewer, "Restore every viewer default",
                "Put every viewer back as installed: layouts, the display and 3-D look, "
                "clip planes, the crosshair, trace and spectrum options, panel sizes "
                "and the defaults for new images.\n\nYour shortcuts, mouse map and "
                "saved presets are kept. The view on screen changes at once.")
            if answer != QMessageBox.StandardButton.Yes:
                return False
        hub = self.ctx.settings_hub
        hub.replace(memory.defaults(hub.settings))
        if self.source is None:
            return True
        self._remembering = False
        try:
            fresh = Scene()
            preset = layouts.PRESET_BY_ID.get(self._preset_id)
            view = layouts.opening_view(
                preset, None, "combo" if self.ctx.gpu_ok else "multi") if preset else {}
            state = fresh.model_dump(mode="json", include={
                "display", "graph", "layout", "mosaic", "clips"})
            state["render"] = RenderState().model_dump(mode="json")
            state["mode"] = view.get("mode", fresh.mode)
            state["plane"] = fresh.plane
            state["graph_visible"] = bool(view.get("graph_visible", False))
            apply_view_state(self.ctx.store, state)
            base = self.ctx.scene.base_layer()
            if base is not None:
                self.ctx.run("layer.reset", layer=base.id, display=self.default_look(base))
        finally:
            self._remembering = True
        self.set_inspector(self.ctx.settings.volume.inspector)
        self.viewer.status_message.emit("Every viewer setting restored to its default")
        return True

    def _remember_view(self) -> None:
        """Written soon after any change (the Editor is not always closed
        cleanly): the arrangement of THIS kind of file, the mode and plane
        as the default for kinds never arranged, and the look every image
        shares (display conventions, 3-D effect and parameters, clip
        planes). See :mod:`bidsmgr.viz.memory`."""
        from ....viz import memory

        if self.source is None:
            return
        scene = self.ctx.scene
        state = scene.model_dump(mode="json", include=set(layouts.LAYOUT_KEYS))
        kind = self._preset_id
        look = memory.volume_look(scene)
        d = scene.display

        def keep(s):
            if kind:
                s.layout_state[kind] = layouts.arrangement(state)
            s.volume.mode = scene.mode
            s.volume.plane = scene.plane
            s.volume.labels = d.labels
            s.volume.colorbar = d.colorbar
            s.volume.ras = d.ras
            s.volume.radiological = d.radiological
            s.volume.space = d.space
            s.volume_look = look

        self.ctx.settings_hub.update(keep)

    # ------------------------------------------------------------------
    # Splitter sizes
    # ------------------------------------------------------------------

    def remember_sizes(self, key: str, splitter) -> None:
        """Keep a splitter's sizes (written after the drag settles)."""
        sizes = [int(v) for v in splitter.sizes()]
        if not sizes or sum(sizes) <= 0:
            return
        self._pending_sizes[key] = sizes
        self._sizes_timer.start()

    def restore_sizes(self, key: str, splitter, fallback: list[float]) -> None:
        """Apply remembered sizes, scaled to the splitter's size today;
        else the ``fallback`` fractions."""
        total = (splitter.height() if splitter.orientation() == Qt.Orientation.Vertical
                 else splitter.width()) or 600
        saved = self.ctx.settings.layout_sizes.get(key)
        fractions = ([v / sum(saved) for v in saved]
                     if saved and len(saved) == splitter.count() and sum(saved) > 0
                     else fallback)
        splitter.setSizes([max(1, int(total * f)) for f in fractions])

    def _flush_sizes(self) -> None:
        pending, self._pending_sizes = self._pending_sizes, {}
        if pending:
            self.ctx.settings_hub.update(lambda s: s.layout_sizes.update(pending))

    # ------------------------------------------------------------------
    # GUI-only actions
    # ------------------------------------------------------------------

    def gui_action(self, name: str, params) -> bool:
        if name == "slice":
            self.ctx.run("cursor.step", plane=self._active_plane(), n=int(params.get("n", 1)))
            return True
        if name == "play":
            self.toggle_play()
            return True
        if name == "save_view":
            self._ask_view_name()
            return True
        if name == "graph_beside":
            self.ctx.run("layout.set",
                         graph="bottom" if self.ctx.scene.layout.graph == "right" else "right")
            return True
        if name == "graph_maximize":
            self.set_graph_maximized(not self._graph_maximized)
            return True
        if name == "graph_detach":
            self.set_graph_detached(self._graph_window is None)
            return True
        if name == "inspector":
            self.set_inspector(not self.inspector_open())
            return True
        if name == "add_overlay":
            self.choose_overlay()
            return True
        if name == "qc":
            self.add_quality_map(str(params.get("map", "tsnr")))
            return True
        if name == "check_quality":
            # A toggle, as the time course is: again closes the panel.
            if self._quality_open:
                self.set_quality_open(False)
            else:
                self.check_quality()
            return True
        if name == "quality_below":
            self.ctx.run("layout.set", quality="right"
                         if self.ctx.scene.layout.quality == "bottom" else "bottom")
            return True
        if name == "quality_maximize":
            self.set_quality_maximized(not self._quality_max)
            return True
        if name == "quality_detach":
            self.set_quality_detached(self._quality_window is None)
            return True
        if name == "show_noise":
            self.set_noise(self._noise_look is None)
            return True
        if name == "header":
            self.show_header()
            return True
        if name == "deface_preview":
            self.preview_defacing()
            return True
        if name == "save_scene":
            self._ask_scene_name()
            return True
        if name == "command_line":
            self.show_command_line()
            return True
        if name == "restore_defaults":
            self.restore_all_defaults()
            return True
        if name == "mosaic_figure":
            self.save_mosaic_figure()
            return True
        if name == "navigate":
            self.navigate(str(params.get("entity", "run")), int(params.get("step", 1)))
            return True
        return False

    def show_command_line(self):
        """The view on screen as a command line (``viz.reproduce``)."""
        from ..panels.command_line import CommandLineDialog

        if self.source is None:
            return None
        dlg = CommandLineDialog(self.ctx.store, parent=self.viewer)
        self.command_line_dialog = dlg
        dlg.show()
        return dlg

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def add_overlay(self, path: Path, **look) -> bool:
        """Read ``path`` on a worker and draw it over the open image, in the
        look its contents call for (``viz.overlays``), or in ``look``
        (``display``, ``name``, ``visible``: what a saved scene recorded).
        False when there is no image to put it over."""
        if self.source is None:
            if self.viewer.current_file() is not None:
                # The base is still opening: the overlay follows it.
                self._pending_overlays.append((Path(path), look))
                return True
            self.viewer.status_message.emit("Open an image first: an overlay goes over one.")
            return False
        tag = self._overlay_job(look)
        self.viewer.loading_changed.emit(True, f"Reading {Path(path).name}")
        budget = memory_budget_bytes(self.ctx.settings.volume.memory_mb)
        self.ctx.jobs.start(tag, self._generation, open_overlay_with_context, Path(path),
                            self.viewer.current_root(), budget_bytes=budget)
        return True

    def _overlay_job(self, look: dict) -> str:
        self._overlay_seq += 1
        tag = f"overlay:{self._overlay_seq}"
        self._overlay_tags.append(tag)
        self._overlay_looks[tag] = dict(look)
        return tag

    def show_header(self, rule: str = ""):
        """The header beside the sidecar (one window per viewer, refilled for
        the image on screen); ``rule`` selects the row a validation finding
        is about. Returns the dialog, or None with no image open."""
        from ....viz.inspect import inspect
        from ..panels.header import HeaderDialog

        src = self.source
        if src is None:
            if self.viewer.current_file() is not None:
                # Still opening: shown once the header is read.
                self._pending_header = rule or ""
            return None
        rows = inspect(src, self.bids)
        old = getattr(self, "header_dialog", None)
        # Checked, not tracked through ``destroyed`` (CLAUDE.md guard 8d).
        if old is not None and not sip.isdeleted(old):
            old.close()
        dlg = HeaderDialog(src.path.name, rows, parent=self.viewer)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.header_dialog = dlg
        if rule:
            dlg.select_rule(rule)
        dlg.show()
        return dlg

    def add_quality_map(self, which: str, **look) -> bool:
        """Mean, standard deviation or temporal SNR of the open series,
        computed on a worker and drawn over it."""
        _layer, src = views.series_layer(self.ctx.store)
        if src is None or not src.fully_loaded:
            self.viewer.status_message.emit(
                "A quality map needs a 4-D series read whole: wait until it is read.")
            return False
        from ....viz.compute.qc import TITLES, quality_overlay

        tag = self._overlay_job({**look, "origin": f"qc:{which}"})
        self.viewer.loading_changed.emit(True, f"Computing the {TITLES[which].lower()}")
        self.ctx.jobs.start(tag, self._generation, quality_overlay, src, which,
                            config=self.qc_config())
        return True

    # ------------------------------------------------------------------
    # The quality check (bidsmgr.qc)
    # ------------------------------------------------------------------

    #: The base layer's look before "Show the noise" (None: not shown).
    _noise_look: Optional[dict] = None
    #: Check quality was asked for while the image was still opening.
    _pending_quality: bool = False

    _quality_panel = None
    _quality_open = False
    _quality_max = False
    _quality_window = None
    _quality_result = None
    #: Map key -> the layer that shows it (one each, shown and hidden).
    _evidence: dict = {}

    def _ensure_quality_panel(self):
        if self._quality_panel is None:
            from ..panels.quality_panel import QualityPanel

            panel = QualityPanel()
            am = self.viewer.action_manager
            # The panel's own corner, as the time course's: below or beside,
            # maximise, its own window; then More and close.
            panel.header.insert_corner(
                [am.button(a) for a in ("qc.below", "qc.maximize", "qc.detach")],
                before=panel.more_button)
            panel.check_requested.connect(lambda: self.check_quality(again=True))
            panel.show_map.connect(self.toggle_evidence)
            panel.go_to.connect(self.go_to_evidence)
            panel.noise.connect(self.set_noise)
            panel.plots.connect(self._quality_plots)
            panel.save.connect(lambda: self.save_quality(self._quality_result))
            panel.close_requested.connect(lambda: self.set_quality_open(False))
            self._quality_panel = panel
        return self._quality_panel

    def set_quality_open(self, on: bool) -> None:
        self._quality_open = bool(on)
        if not on:
            if self._quality_window is not None:
                self.set_quality_detached(False, apply=False)
            self._quality_max = False
        self.apply_quality()
        self.viewer.refresh_actions()

    def apply_quality(self) -> None:
        """The panel where the layout puts it, or nowhere."""
        panel = self._quality_panel
        right, below = self.quality_right, self.quality_below
        if not self._quality_open or panel is None:
            right.setVisible(False)
            below.setVisible(False)
            self.vsplit.setVisible(True)
            self.qsplit.setVisible(True)
            return
        if self._quality_window is not None:
            right.setVisible(False)
            below.setVisible(False)
            self.vsplit.setVisible(True)
            self.qsplit.setVisible(True)
            return
        on_below = self.ctx.scene.layout.quality == "bottom"
        host, other = (below, right) if on_below else (right, below)
        if panel.parentWidget() is not host:
            host.layout().addWidget(panel)
        panel.setVisible(True)
        first = host.isHidden()
        other.setVisible(False)
        host.setVisible(True)
        if first:
            if on_below:
                self.restore_sizes("volume.quality_below", self.qouter, [0.62, 0.38])
            else:
                self.restore_sizes("volume.quality", self.qsplit, [0.68, 0.32])
        # Maximised: the panel alone, the views and time course out of the way.
        if on_below:
            self.qsplit.setVisible(not self._quality_max)
            self.vsplit.setVisible(True)
        else:
            self.vsplit.setVisible(not self._quality_max)
            self.qsplit.setVisible(True)

    def set_quality_maximized(self, on: bool) -> None:
        self._quality_max = bool(on)
        self.apply_quality()
        self.viewer.refresh_actions()

    def set_quality_detached(self, on: bool, *, apply: bool = True) -> None:
        """The panel in its own window, or back; closing the window puts it
        back."""
        if on and self._quality_window is None:
            panel = self._ensure_quality_panel()
            win = _GraphWindow(self.viewer)
            win.setWindowTitle(f"Quality: {self.source.path.name}"
                               if self.source is not None else "Quality")
            win.take(panel)
            win.resize(max(panel.width(), 520), max(panel.height(), 640))
            win.closed.connect(lambda: self.set_quality_detached(False))
            self._quality_window = win
            self._quality_max = False
        elif not on and self._quality_window is not None:
            win, self._quality_window = self._quality_window, None
            win.closed.disconnect()
            if self._quality_panel is not None:
                self.quality_right.layout().addWidget(self._quality_panel)
            win.close()
            win.deleteLater()
        if apply:
            self.apply_quality()
        self.viewer.refresh_actions()

    def qc_config(self):
        """The methods and thresholds of Settings > Quality control."""
        from ....viz.settings import qc_config

        return qc_config(self.ctx.settings)

    def _quality_kind(self) -> Optional[str]:
        from ....qc import live

        return live.kind_of(self.source)

    def check_quality(self, *, again: bool = False) -> bool:
        """The quality check of the image on screen into the quality panel
        (opened), on a worker. False when there is none to run."""
        from ....qc import live

        panel = self._ensure_quality_panel()
        if not self._quality_open:
            self.set_quality_open(True)
        src = self.source
        if src is None or not src.fully_loaded:
            if self.viewer.current_file() is not None:
                # Still opening (the Editor's Quality check opened it): the
                # check follows once it is read.
                self._pending_quality = True
                panel.set_busy(f"Reading {self.viewer.current_file().name}")
                return True
            return False
        if live.kind_of(src) is None:
            panel.set_empty(src.path.name, reason=(
                "The quality check is for anatomical (T1w, T2w, FLAIR, PDw, T2starw) and "
                "diffusion images."))
            return False
        if again:
            live.forget(src)
        got = live.cached(src, self.qc_config())
        if got is not None:
            self._quality_result = got
            self.show_quality(got)
            return True
        panel.set_busy(f"Checking the quality of {src.path.name}")
        sidecar = dict(self.bids.sidecar) if self.bids is not None else {}
        self.ctx.jobs.start("quality", self._generation, live.result_for, src, sidecar,
                            config=self.qc_config())
        return True

    def _follow_quality(self) -> None:
        """The open panel, after another file was read: its result when one
        is kept, else the check, run at once when QC runs on opening."""
        from ....qc import live

        panel = self._ensure_quality_panel()
        src = self.source
        if src is None:
            return
        if live.kind_of(src) is None:
            panel.set_empty(src.path.name, reason=(
                "No quality check for this image: it is for anatomical (T1w, T2w, FLAIR, "
                "PDw, T2starw) and diffusion images."))
            return
        got = live.cached(src, self.qc_config())
        if got is not None or self.ctx.settings.qc.on_open:
            self.check_quality()
        else:
            panel.set_empty(src.path.name)

    def show_quality(self, result) -> None:
        """``result`` in the panel, with the dataset's context."""
        from ....qc import live

        src = self.source
        if src is None:
            return
        panel = self._ensure_quality_panel()
        if not self._quality_open:
            self.set_quality_open(True)
        root = self.scene_root()
        try:
            context = live.dataset_context(result, root)
        except Exception:  # noqa: BLE001 - the comparison is a bonus
            log.debug("could not read the dataset's quality results", exc_info=True)
            context = {}
        panel.set_result(result, src.path.name, context=context,
                         shown=self._shown_evidence(), noise_on=self._noise_look is not None,
                         plots_on=bool(self.ctx.scene.graph_visible and self.ctx.scene.graph.qc),
                         can_save=root is not None and not self.viewer.read_only)

    def _evidence_layer(self, key: str):
        lid = self._evidence.get(key)
        if not lid:
            return None
        return next((layer for layer in self.ctx.scene.layers if layer.id == lid), None)

    def _shown_evidence(self) -> set:
        return {k for k in self._evidence if (lay := self._evidence_layer(k)) is not None
                and lay.visible}

    def _sync_evidence_boxes(self) -> None:
        """The checkboxes follow the layers (hidden or removed elsewhere)."""
        panel = self._quality_panel
        if panel is None:
            return
        for key in list(self._evidence):
            layer = self._evidence_layer(key)
            if layer is None:
                self._evidence.pop(key, None)
            panel.set_map_shown(key, bool(layer is not None and layer.visible))

    def toggle_evidence(self, key: str, on: bool) -> None:
        """Show or hide one of the check's maps: ONE layer each, added the
        first time and shown and hidden after."""
        layer = self._evidence_layer(key)
        if on:
            if layer is None:
                lid = self.add_quality_evidence(self._quality_result, key)
                if lid:
                    self._evidence[key] = lid
            elif not layer.visible:
                self.ctx.run("layer.set", layer=layer.id, visible=True)
        elif layer is not None and layer.visible:
            self.ctx.run("layer.set", layer=layer.id, visible=False)
        self.ctx.qstore.flush()

    def add_quality_evidence(self, result, key: str) -> str:
        """Draw one of the check's maps over the image: ``result.maps[key]``,
        or ONE class of a labelled map (``"tissues:2"``, grey matter); the
        new layer's id, "" when there is none."""
        from ....viz.data.volume import array_volume
        from ....viz.overlays import Overlay
        from ....viz.scene import LabelTable, VolumeDisplay
        from ..panels.quality_panel import label_colours, split_key

        base, value = split_key(key)
        qmap = result.maps.get(base) if result is not None else None
        if qmap is None or self.source is None:
            return ""
        title = qmap.title
        data = qmap.data
        if qmap.kind == "labels":
            colours = label_colours(base, qmap)
            labels = {int(k): str(v) for k, v in qmap.labels.items()}
            if value is not None:
                # One class, one layer: shown and hidden on its own.
                title = labels.get(value, title)
                data = (np.asarray(qmap.data) == value).astype(np.uint8)
                labels = {1: title}
                colours = {1: colours.get(value, (128, 128, 128))}
            table = LabelTable(labels=labels, colors=colours)
            display = VolumeDisplay(label_table=table, opacity=0.45, interpolation="nearest")
        elif qmap.kind == "field":
            lo, hi = {"bias": (0.75, 1.25), "fa": (0.0, 1.0), "md": (0.0, 3.0)}.get(
                key, (0.0, 1.0))
            display = VolumeDisplay(colormap=qmap.colour, window=(lo, hi), opacity=0.75,
                                    threshold_mode="hide_below" if key != "bias"
                                    else "range", interpolation="nearest", gamma=1.0)
            if key == "bias":
                display.opacity = 0.55
        else:
            display = VolumeDisplay(colormap=qmap.colour, window=(0.5, 1.0), opacity=0.9,
                                    threshold_mode="hide_below", interpolation="nearest",
                                    outline_px=2.0, gamma=1.0)
            if key in ("artefacts", "cc", "face"):
                display.outline_px = 0.0
                display.opacity = 0.7
        vol = array_volume(data, qmap.affine, path=self.source.path, name=title)
        self._overlay_seq += 1
        n = str(self._overlay_seq)
        overlay = Overlay(source=vol, display=display, kind="mask", note=qmap.help, name=title)
        self._adopt_overlay(n, overlay, origin=f"qc-map:{key}")
        return f"overlay{n}"

    def go_to_evidence(self, evidence) -> None:
        """A finding's place: its volume, and its slice along the slice axis
        (the view turned to show that slice whole)."""
        from ..canvases.graph import go_to_slice

        if not isinstance(evidence, dict):
            return
        if evidence.get("volume") is not None:
            self.ctx.run("frame.set", frame=int(evidence["volume"]))
        if evidence.get("slice") is not None:
            go_to_slice(self.ctx, int(evidence["slice"]), self.bids)
            axis = int(evidence.get("axis", 2))
            plane = {0: "sagittal", 1: "coronal", 2: "axial"}.get(axis, "axial")
            if self.ctx.scene.mode == "single" and self.ctx.scene.plane != plane:
                self.ctx.run("view.plane", plane=plane)
        self.ctx.qstore.flush()

    def set_noise(self, on: bool) -> None:
        """Window the base image to its air (in a perceptual colour map), or
        put its look back: what MRIQC's background mosaic shows, live."""
        layer, src = views.base(self.ctx.store)
        if layer is None or src is None:
            return
        if on and self._noise_look is None:
            frame = np.asarray(src.scale(np.asarray(src.raw_frame(
                views.frame_of(self.ctx.store, layer, src)))), dtype=np.float32)
            from ....qc import masks as QM

            level = QM.air_level(frame)
            if level is None:
                self.viewer.status_message.emit(
                    "No noise to show: the air around the head is set to zero (by the "
                    "scanner, or by defacing), or the head fills the field of view.")
                if self._quality_panel is not None:
                    self._quality_panel.set_noise(False)
                return
            # Up to well above the noise: ghosts and ringing stand out of it,
            # the head is saturated.
            hi = level[0] + 10.0 * level[1]
            self._noise_look = {"window": layer.display.window,
                                "colormap": layer.display.colormap,
                                "gamma": layer.display.gamma}
            self.ctx.run("layer.set", layer=layer.id, window=(0.0, max(hi, 1e-6)),
                         colormap="viridis", gamma=1.0)
        elif not on and self._noise_look is not None:
            look, self._noise_look = self._noise_look, None
            params = {k: v for k, v in look.items() if v is not None}
            self.ctx.run("layer.set", layer=layer.id, **params)
        else:
            return
        self.ctx.qstore.flush()
        if self._quality_panel is not None:
            self._quality_panel.set_noise(self._noise_look is not None)
        self.viewer.action_manager.update_state(self.action_context())

    def _quality_plots(self, on: bool) -> None:
        if on and not self.ctx.scene.graph_visible:
            self.ctx.run("view.graph", value=True)
        self.ctx.run("graph.set", qc=bool(on))
        self.ctx.qstore.flush()

    def save_quality(self, result) -> Optional[Path]:
        """Write ``result`` into the dataset's QC derivative."""
        from ....qc import report

        root = self.scene_root()
        if self.viewer.read_only:
            self.viewer.status_message.emit("This folder is open for viewing only: nothing "
                                            "is written into it.")
            return None
        if root is None:
            self.viewer.status_message.emit("Open the image from a dataset to save its "
                                            "quality result.")
            return None
        try:
            path = report.save(result, root)
            report.write_group(root)
        except OSError as exc:
            self.viewer.status_message.emit(f"Could not save the quality result: {exc}")
            return None
        self.viewer.status_message.emit(f"Saved {path.relative_to(root).as_posix()}")
        return path

    _deface_ok: Optional[bool] = None

    def _deface_available(self) -> bool:
        """Whether a defacing engine is installed (looked up once)."""
        if self._deface_ok is None:
            try:
                from ....deface.run import available

                type(self)._deface_ok = bool(available())
            except Exception:  # noqa: BLE001 - no engine is an answer, not an error
                type(self)._deface_ok = False
        return bool(self._deface_ok)

    def preview_defacing(self, engine_id: str = "", **look) -> bool:
        """What defacing would remove from the image on screen, drawn over
        it; nothing is changed. The engine runs on a worker."""
        src = self.source
        if src is None or src.is_4d or src.in_memory:
            return False
        from ....viz.overlays import deface_preview

        tag = self._overlay_job({**look, "origin": f"deface:{engine_id}"})
        self.viewer.loading_changed.emit(True, "Running the defacing engine for a preview")
        self.ctx.jobs.start(tag, self._generation, deface_preview, src.path, engine_id)
        return True

    def _adopt_overlay(self, n: str, overlay, *, name: str = "", display=None,
                       visible: bool = True, in_3d: bool = True, origin: str = "") -> None:
        sid, lid = f"ovl{n}", f"overlay{n}"
        src = overlay.source
        self.ctx.store.sources[sid] = src
        label = name or overlay.name or src.path.name
        look = display if display is not None else overlay.display.model_dump()
        # What "restore defaults" puts back: the look its CONTENT called for
        # (an atlas in its labels, a map in heat), not the base's defaults.
        self._original_looks[lid] = overlay.display.model_dump()
        self.ctx.run("layer.add", id=lid, source=sid, name=label, display=look,
                     path="" if src.in_memory else str(src.path), origin=origin,
                     visible=bool(visible), in_3d=bool(in_3d))
        self.ctx.qstore.flush()
        base = self.source
        if (src.is_4d and not src.is_rgb and base is not None and not base.is_4d
                and not self.ctx.scene.graph_visible):
            # A series over an anatomy: its time course is why it is here.
            self.ctx.run("view.graph", value=True)
        self.viewer.loading_changed.emit(False, "")
        self.viewer.status_message.emit(f"Overlay {label}: {overlay.note}")
        self.viewer.overlay_added.emit(lid)

    def choose_overlay(self) -> None:
        """Pick an image of the dataset (this subject first), or Browse...
        for one from elsewhere, and put it over the open image."""
        from ..panels.overlay_picker import ask_for_overlay

        src = self.source
        if src is None:
            return
        path = ask_for_overlay(self.viewer, self.scene_root(), src.path,
                               base_affine=src.affine, base_shape=src.spatial)
        if path:
            self.add_overlay(Path(path))

    def toggle_play(self) -> None:
        if self._play_timer.isActive():
            self.stop_play()
            return
        if views.series_layer(self.ctx.store)[0] is None:
            return
        fps = self.ctx.settings.volume.fps
        self._play_timer.start(max(16, int(1000 / fps)))
        self.ctx.playing = True
        self.ctx.playing_changed.emit(True)
        self.viewer.refresh_actions()

    def stop_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
        if self.ctx.playing:
            self.ctx.playing = False
            self.ctx.playing_changed.emit(False)
            self.viewer.refresh_actions()

    def _play_tick(self) -> None:
        layer, src = views.series_layer(self.ctx.store)
        if layer is None or src is None:
            self.stop_play()
            return
        nxt = (layer.frame + 1) % max(src.n_frames, 1)
        if not src.frame_ready(nxt):
            return   # wait for the stream to catch up rather than skip
        self.ctx.run("frame.set", frame=nxt, layer=layer.id)

    # ------------------------------------------------------------------
    # What is on screen
    # ------------------------------------------------------------------

    @property
    def graph(self):
        """The time-course graph, once it has been shown."""
        return self._graph

    @property
    def render_canvas(self):
        """The 3-D canvas (None without a GPU)."""
        return self._render

    def figure_widget(self) -> QWidget:
        """What a screenshot shows: the views (and the graph, when it is
        open), never a panel or an editor. The mosaic's figure is its canvas."""
        page = self.pages.currentWidget()
        if self.ctx.scene.mode == "mosaic" and page is not None and hasattr(page, "canvas"):
            return page.canvas
        return self.vsplit

    # -- the side column ----------------------------------------------------

    def inspector_open(self) -> bool:
        if self._controls_peer is not None:
            return self._controls_peer.inspector_open()
        if self._host is not None:
            return not self.side.isHidden()
        return self.side.isVisible()

    def host_controls(self, host=None, *, tab: bool = True) -> None:
        """Show the controls column and its tab in ``host`` instead of beside
        these images, or bring them home (``host=None``).

        Two images driven as one keep ONE column, outside both, so the two
        are the same size. ``host`` provides ``column`` (a layout for the
        column), ``tab_row`` (a layout for the tab) and ``shown(bool)``
        (the column opened or closed). ``tab=False`` keeps the tab hidden
        at home (the column of the other image is the one on screen).
        """
        was = self.inspector_open() if self._controls_peer is None else False
        if self._host is not None:
            self._host.tab_row.removeWidget(self.side_tab)
        else:
            self._tab_row.removeWidget(self.side_tab)
        self._host = host
        if host is None:
            self._tab_row.addWidget(self.side_tab)
            self.hsplit.addWidget(self.side)
            self.hsplit.setStretchFactor(1, 0)
        else:
            host.tab_row.addWidget(self.side_tab)
            host.column.addWidget(self.side, 1)
        self.side_tab.setVisible(bool(tab) and bool(getattr(self.viewer, "panels", True)))
        self.side.setVisible(was and (host is not None or tab))
        if host is not None:
            host.shown(was)

    def share_controls(self, peer) -> None:
        """Open ``peer``'s controls column instead of this one's (Ctrl+I,
        the Controls entry): one column serves both halves of a synced
        pair. None gives this viewer its own again."""
        self._controls_peer = peer

    def _on_split_resized(self, width: int, old: int) -> None:
        """A narrowing viewer hides the controls column before squeezing the
        images away; widening brings it back. Only the user's own choice is
        remembered, and only a NARROWING hides it, so Ctrl+I always opens it."""
        if self._host is not None or self._controls_peer is not None:
            return
        need = self.side.minimumWidth() + SIDE_GIVES_WAY_PX
        if self.side.isVisible() and 0 < width < need and 0 < old and width < old:
            self.side.setVisible(False)
            self.side_tab.set_open(False)
            self._side_gave_way = True
            self.viewer.refresh_actions()
        elif (self._side_gave_way and width >= need + 40 and self.ctx.settings.volume.inspector
              and self._inspector is not None):
            self._side_gave_way = False
            self.side.setVisible(True)
            self.side_tab.set_open(True)
            self.viewer.refresh_actions()

    def set_inspector(self, on: bool, *, remember: bool = True) -> None:
        if self._controls_peer is not None:
            self._controls_peer.set_inspector(on, remember=remember)
            return
        if on and self._inspector is None:
            from ..panels.inspector import Inspector

            self._inspector = Inspector(self.ctx, self)
            self._inspector.add_requested.connect(self.choose_overlay)
            self.side.setWidget(self._inspector)
            # Never narrower than its controls: with no horizontal scroll bar
            # a narrower column cut them off at its edge.
            self.side.setMinimumWidth(
                self._inspector.minimumSizeHint().width()
                + self.side.verticalScrollBar().sizeHint().width() + 2)
        was = self.inspector_open()
        self._side_gave_way = False
        self.side.setVisible(bool(on))
        self.side_tab.set_open(bool(on))
        if self._host is not None:
            self._host.shown(bool(on))
        elif on and not was:
            self.restore_sizes("volume.side", self.hsplit, [0.76, 0.24])
        if remember and self.ctx.settings.volume.inspector != bool(on):
            self.ctx.settings_hub.update(lambda s: setattr(s.volume, "inspector", bool(on)))
        self.viewer.refresh_actions()

    @property
    def inspector(self):
        return self._inspector

    def restore_panels(self) -> None:
        """Open the controls column if the user left it open."""
        if self.ctx.settings.volume.inspector and not self.inspector_open():
            self.set_inspector(True, remember=False)

    def canvases(self, kind: str) -> list:
        if kind == "render":
            return [self._render] if self._render is not None and self._render.isVisible() else []
        if kind == "graph":
            return [self._graph] if self._graph is not None and self._graph.isVisible() else []
        page = self.pages.currentWidget()
        if page is None:
            return []
        if kind == "mosaic":
            from ..canvases.mosaic import MosaicCanvas

            return [w for w in page.findChildren(MosaicCanvas) if w.isVisible()]
        if page is self.tiles:
            return [w for t in self.tiles.tiles
                    if t != "render" and (w := self.tiles.widget_for(t)) is not None
                    and w.isVisible()]
        return []

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    def summary(self) -> str:
        src = self.source
        if src is None:
            return ""
        shape = "x".join(str(s) for s in src.facts.shape)
        text = f"{shape} · {src.facts.disk_dtype}"
        if src.is_rgb:
            text += " · RGB"
        if src.is_4d and not src.fully_loaded:
            text += f" · reading {src.loaded_frames}/{src.n_frames}"
        if src.truncated:
            text += (f" · first {src.n_frames} of {src.n_frames_total} volumes "
                     "(memory limit, Settings > Viewer)")
        return text

    def readout(self) -> str:
        return views.readout(self.ctx.store)

    def stop(self) -> None:
        self.stop_play()
        if self._graph_window is not None:
            self.set_graph_detached(False)
        if self._quality_window is not None:
            self.set_quality_detached(False)
        # A change made just before the window closed is kept.
        if self._view_timer.isActive():
            self._view_timer.stop()
            self._remember_view()
        if self._sizes_timer.isActive():
            self._sizes_timer.stop()
            self._flush_sizes()


__all__ = ["VolumePresenter", "open_with_context"]
