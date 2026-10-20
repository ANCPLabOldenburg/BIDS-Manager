"""Signals (MEG, EEG, iEEG, physio) in the viewer shell.

**MEG and EEG open on their metadata.** Channels, rate, duration, types,
filters, bad channels, read with ``preload=False`` on a worker; "Load signal"
then reads the recording and shows the traces, and Close drops it again.
A recording can be gigabytes, and most visits only need the card.

**Physio opens straight into the traces**, decided by its sidecar, with Fit
all (a channel or four fits whole, which three hundred MEG channels do not)
and, when the run has several physio files, "All of this run" on one clock.

Every control runs a command (``viz.commands.signal``), so a key, a button,
the palette and a linked viewer all do the same thing. Reading, resampling
and the spectrum run as jobs (QThread, never a pool: they end in scipy).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from ....viz.bids import full_ext, suffix_of
from ....viz.commands import signal as sigcmd
from ....viz.compute.filters import PRESETS, preset_of
from ....viz.data import formats
from ....viz.data.signal import SignalSource
from ....viz.scene import Scene, SignalLayer, SourceRef, TracesState
from ...widgets.primitives import ElidedLabel
from ..context import ViewerContext
from ..controls import NumberControl
from ..menus import popup_menu
from ...widgets.splitter import Splitter

log = logging.getLogger(__name__)

#: Decimals on every frequency field: three, because 0.01 Hz is where a
#: respiratory drift lives ("impossible to put a filter of 0.01").
_DECIMALS = 3

#: ONE row of what is reached for all the time, by purpose: how much of the
#: signal is on screen (amplitude, time span), what is done to it (the
#: filter band), checking and marking it (QC, annotation, saving), and the
#: spectrum; then leaving it. Everything else is in the controls column
#: (``panels.signal_controls``), grouped the same way.
TOOLBAR_ROWS = (
    ("widget:count", "widget:scale", "traces.reset", "|", "widget:preset", "|",
     "traces.quality", "annotate.toggle", "review.save", "|", "widget:psd", "stretch",
     "help.shortcuts", "signal.close"),
)

#: The width of a slider in the toolbar: enough to aim, not so much that
#: three of them push the toggles onto another line.
_TOOLBAR_SLIDER_PX = 84


def _read_meta(path) -> dict:
    """Worker side: the metadata card, never the samples."""
    from ....viz.data.signal import read_recording, summarize

    return summarize(read_recording(path, preload=False), path)


def _open(path, root, kind: str, together: bool) -> SignalSource:
    from ....viz.data.signal import open_meeg, open_physio

    if kind == "physio":
        return open_physio(path, root, together=together)
    return open_meeg(path, root)


class _SizedLabel(QLabel):
    """A label whose size is that of ``widest``, whatever it shows: a
    wrapping bar places its children once, at the size they ask for."""

    def __init__(self, widest: str) -> None:
        super().__init__("")
        self._widest = widest

    def sizeHint(self):  # noqa: N802 - Qt naming
        hint = super().sizeHint()
        hint.setWidth(max(hint.width(), self.fontMetrics().horizontalAdvance(self._widest) + 4))
        return hint


class SignalPresenter:
    """Signal content for a :class:`~bidsmgr.gui.viz.viewer.Viewer`."""

    kind = "signal"
    title = "Recording"
    empty_hint = "Select an EEG or MEG recording, or a physio file, in the BIDS tree."

    def __init__(self, viewer, ctx: ViewerContext) -> None:
        self.viewer = viewer
        self.ctx = ctx
        self.source: Optional[SignalSource] = None
        self.meta: Optional[dict] = None
        self._generation = 0
        self._path: Optional[Path] = None
        self._root: Optional[Path] = None
        self._file_kind = "meeg"
        #: Physio: every physio file of the run together (the default: the
        #: cardiac, breathing and trigger files of one run belong together).
        self.together = True
        #: An EEG/MEG recording opens on its metadata card and waits for
        #: "Load signal" (a recording can be gigabytes, and the Editor is
        #: browsed). A host whose user already asked to SEE it (the
        #: Converter's preview) sets this: the signal follows the card.
        self.load_on_open = False
        from PyQt6.QtCore import QTimer

        self._memory_timer = QTimer(viewer)
        self._memory_timer.setSingleShot(True)
        self._memory_timer.setInterval(400)
        self._memory_timer.timeout.connect(self._remember)
        #: Zen mode: the traces alone, no toolbar and no overview.
        self.zen = False
        self._relatives: list[Path] = []
        self._traces = None
        self.content = self._build_content()
        ctx.jobs.done.connect(self._on_job_done)
        ctx.jobs.failed.connect(self._on_job_failed)
        ctx.qstore.changed.connect(self._on_changed)
        ctx.status.connect(viewer.status_message)
        from ..bridge import connect_while_alive

        self._qc_params = ctx.settings.meeg_qc.model_dump_json()
        connect_while_alive(ctx.settings_hub.changed, viewer,
                            lambda v, _s: v.presenter.on_settings_changed())

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    def _build_content(self) -> QWidget:
        self.pages = QStackedWidget()
        self.pages.setObjectName("pane-dark")
        self.meta_page = self._build_meta_page()
        self.pages.addWidget(self.meta_page)
        traces = QWidget()
        traces.setObjectName("pane-dark")
        v = QVBoxLayout(traces)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.annotation_bar = self._build_annotation_bar()
        v.addWidget(self.annotation_bar)
        # The traces and, under them, the QC plots: a splitter, so the QC can
        # be given room (or all of it: Maximise).

        self.qc_split = Splitter(Qt.Orientation.Vertical)
        self.qc_split.setChildrenCollapsible(False)
        self.qc_split.setHandleWidth(6)
        holder = QWidget()
        holder.setObjectName("pane-dark")
        self._traces_slot = QVBoxLayout(holder)
        self._traces_slot.setContentsMargins(0, 0, 0, 0)
        self._traces_slot.setSpacing(0)
        self._traces_holder = holder
        self.qc_split.addWidget(holder)
        self.quality_row = self._build_quality_row()
        self.qc_split.addWidget(self.quality_row)
        self.qc_split.setStretchFactor(0, 3)
        self.qc_split.setStretchFactor(1, 2)
        v.addWidget(self.qc_split, 1)
        self.navigation = self._build_navigation()
        v.addWidget(self.navigation)
        self.traces_page = self._beside_controls(traces)
        self.pages.addWidget(self.traces_page)
        return self.pages

    def _beside_controls(self, traces: QWidget) -> QWidget:
        """The traces with the controls column on their right."""
        from ..panels.side_column import SideColumn
        from ..panels.signal_controls import SignalControls

        self.controls = SignalControls(self)

        def remember(on: bool) -> None:
            if self.ctx.settings.traces.controls_open != on:
                self.ctx.settings_hub.update(lambda s: setattr(s.traces, "controls_open", on))

        self.column = SideColumn(traces, self.controls, remember=remember,
                                 changed=self.viewer.refresh_actions)
        self.side_tab = self.column.tab
        self.side = self.column.scroll
        return self.column.widget

    # -- the controls column ---------------------------------------------------

    def inspector_open(self) -> bool:
        return self.column.is_open()

    def set_inspector(self, on: bool, *, remember: bool = True) -> None:
        self.column.set_open(on, remember=remember)

    def open_section(self, key: str) -> None:
        """Open the controls column at one section (QC's Settings button)."""
        section = self.controls.section(key)
        section.set_open(True)
        self.column.show_section(section)

    def _ensure_traces(self):
        """The pyqtgraph canvas is built when a signal first arrives: an
        Editor session that only browses tables never builds a plot."""
        if self._traces is None:
            from ..canvases.traces import TracesCanvas

            self._traces = TracesCanvas(self.ctx)
            # Under the annotation bar, above the QC and the navigation.
            self._traces_slot.addWidget(self._traces, 1)
            self.qc_tracks.follow(self._traces.plot)
        return self._traces

    @property
    def traces(self):
        return self._traces

    def _build_meta_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("pane-dark")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("viewer-meta-scroll")
        body = QWidget()
        body.setObjectName("pane-dark")
        self._meta_layout = QVBoxLayout(body)
        self._meta_layout.setContentsMargins(18, 16, 18, 16)
        self._meta_layout.setSpacing(6)
        self._meta_layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        bar = QFrame()
        bar.setObjectName("toolbar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 8, 14, 8)
        bl.addStretch(1)
        self.load_button = QPushButton("Load signal")
        self.load_button.setObjectName("load-signal-btn")
        self.load_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.load_button.clicked.connect(self.load_signal)
        bl.addWidget(self.load_button)
        bl.addStretch(1)
        outer.addWidget(bar)
        return page

    def _build_annotation_bar(self) -> QWidget:
        """What annotation mode adds: how to use it, the label of the next
        segment, delete, what is marked, save, and the way out."""
        from ...widgets.flow_layout import FlowBar
        from ....viz.commands.signal import BAD_LABELS

        bar = FlowBar(h_spacing=8, v_spacing=4)
        bar.setObjectName("toolbar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setContentsMargins(10, 4, 10, 4)
        title = QLabel("Annotation mode")
        title.setObjectName("viewer-meta-section")
        bar.addWidget(title)
        self.label_combo = QComboBox()
        self.label_combo.setEditable(True)
        self.label_combo.addItems(BAD_LABELS)
        self.label_combo.setToolTip(
            "The label of the next segment: BAD_ and a reason. mne-bids reads BAD_ "
            "rows of _events.tsv back as annotations, which MNE leaves out of epochs "
            "and spectra.")
        self.label_combo.activated.connect(
            lambda _i: self._run_ui("annotate.label", label=self.label_combo.currentText()))
        self.label_combo.lineEdit().editingFinished.connect(
            lambda: self._run_ui("annotate.label", label=self.label_combo.currentText()))
        bar.addWidget(self._group(self._label("Label"), self.label_combo))
        manager = self.viewer.action_manager
        bar.addWidget(manager.button("annotate.delete"))
        # Sized for its longest text: a wrapping bar places a child at the
        # size it had when placed, and a count that grew was cut to "1".
        self.review_summary = _SizedLabel("999 bad channels · 999 bad segments (9999.9 s)")
        self.review_summary.setObjectName("sidecar-footer-summary")
        bar.addWidget(self.review_summary)
        bar.addWidget(manager.button("review.save"))
        done = QPushButton("Done")
        done.setObjectName("tb-btn")
        done.setToolTip("Leave annotation mode (A); what is marked stays marked.")
        done.clicked.connect(lambda: self.viewer.trigger("annotate.toggle"))
        bar.addWidget(done)
        hint = QLabel("Drag across the traces to mark a bad segment · drag its edges to "
                      "adjust it · right-click it to relabel or delete it · click a channel "
                      "name to mark the channel bad")
        hint.setObjectName("sidecar-footer-summary")
        bar.addWidget(hint)
        bar.setVisible(False)
        return bar

    def _build_quality_row(self) -> QWidget:
        """QC under the traces: a bar (what to plot, for which channel types,
        how tall, and what to do with what QC found) over its plots, one per
        measure and channel type (``canvases.tracks``)."""
        from ... import icons
        from ..canvases.tracks import TracksPanel
        from ..menus import popup_menu

        from ..panels.panel_header import PanelHeader, corner_button

        row = QWidget()
        row.setObjectName("viz-tracks")
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        v = QVBoxLayout(row)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        # The same header as the image viewer's time course: what to plot and
        # what to do with what QC found on the left, the panel as a panel
        # (expand its plots, beside, maximise) in the upper-right corner.
        header = PanelHeader()
        header.setObjectName("viz-tracks-bar")
        bar = header.controls
        title = QLabel("QC")
        title.setObjectName("viewer-meta-section")
        bar.addWidget(title)

        def menu_button(text: str, icon: str, tip: str):
            btn = QPushButton(text)
            btn.setObjectName("tb-btn")
            btn.setIcon(icons.icon(icon))
            btn.setProperty("viz_icon", icon)   # re-coloured by the header
            btn.setToolTip(tip)
            menu = popup_menu(btn)
            menu.setToolTipsVisible(True)
            btn.setMenu(menu)
            bar.addWidget(btn)
            return btn, menu

        from ....viz.compute.meeg_qc import METRICS

        self.qc_plots_button, plots = menu_button(
            "Plots", "plots", "Which QC plots are drawn, one per measure and channel type; "
            "each can also be moved and hidden from its own header")
        self._qc_metric_actions = {}
        for mid, name, _unit, note in METRICS:
            act = plots.addAction(name[:1].upper() + name[1:])
            act.setCheckable(True)
            act.setToolTip(note)
            act.toggled.connect(lambda on, m=mid: self._toggle_qc_metric(m, on))
            self._qc_metric_actions[mid] = act
        plots.addSeparator()
        # The same choice, worded the same, as at the end of the image
        # viewer's Plots menu.
        self._qc_on_open = plots.addAction("Run QC when a file opens")
        self._qc_on_open.setCheckable(True)
        self._qc_on_open.setToolTip(
            "On: QC stays on from one file to the next and is computed as soon as a file "
            "opens. Off: every file opens with QC off.")
        self._qc_on_open.toggled.connect(self._set_qc_on_open)
        plots.aboutToShow.connect(
            lambda: self._qc_on_open.setChecked(self.ctx.settings.qc.on_open))
        self.qc_types_button, self._qc_types_menu = menu_button(
            "Types", "channel_types",
            "Which channel types are plotted (QC never mixes types: each has its own plots)")
        self._qc_types_menu.aboutToShow.connect(self._fill_qc_types_menu)
        self.qc_mark_button, self._qc_mark_menu = menu_button(
            "Mark as bad", "mark_bad",
            "Mark what QC found as bad: channels (all it suggests, or by reason) and "
            "segments (all it flagged, or by why). Undoable; saved only with Save to "
            "dataset.")
        self._qc_mark_menu.aboutToShow.connect(self._fill_qc_mark_menu)
        report = QPushButton("Report...")
        report.setObjectName("tb-btn")
        report.setToolTip("Every channel and every flagged segment, with the reasons, and "
                          "how to read them")
        report.clicked.connect(self.show_quality_report)
        bar.addWidget(report)
        settings = QPushButton("Settings...")
        settings.setObjectName("tb-btn")
        settings.setToolTip("The check's parameters (channel types, segment length, "
                            "thresholds, measures), in the advanced controls")
        settings.clicked.connect(lambda: self.open_section("signal.qc"))
        bar.addWidget(settings)

        self.qc_expand_button = header.add_corner(corner_button(
            "tracks_scroll", "Expand plots: each QC plot at a readable height, in a "
            "scrolling column (off: they share the room)", checkable=True))
        self.qc_expand_button.toggled.connect(
            lambda on: self._run_ui("traces.qc_view", scroll=bool(on)))
        self.qc_beside_button = header.add_corner(corner_button(
            "dock_right", "Beside: the QC plots to the right of the traces instead of "
            "under them", checkable=True))
        self.qc_beside_button.toggled.connect(
            lambda on: self._run_ui("traces.qc_view", beside=bool(on)))
        self.qc_max_button = header.add_corner(corner_button(
            "maximize", "Maximise: the QC plots take the traces' room too; again to bring "
            "the traces back", checkable=True))
        self.qc_max_button.toggled.connect(self._maximise_qc)
        self.qc_header = header
        v.addWidget(header)
        self.qc_tracks = TracksPanel(self.ctx, left_axis_px=56)
        self.qc_tracks.describe_x = self._describe_qc_x
        self.qc_tracks.clicked.connect(self._qc_go_to)
        self.qc_tracks.row_clicked.connect(self._qc_go_to_cell)
        self.qc_tracks.order_changed.connect(
            lambda order: self._run_ui("traces.qc_view", order=order))
        self.qc_tracks.hide_requested.connect(self._hide_qc_track)
        self.qc_tracks.align_source = self._qc_area
        v.addWidget(self.qc_tracks, 1)
        row.setVisible(False)
        self._qc_tracks_key = None
        self._quality: Optional[dict] = None
        #: (recording, parameters) the QC on screen was computed for.
        self._quality_of: Optional[tuple] = None
        return row

    # -- the QC plots ----------------------------------------------------------

    def _qc_track_list(self) -> list[dict]:
        """The plots to show, in the user's order, without the hidden ones."""
        from ....viz.compute.meeg_qc import tracks as qc_tracks

        res = self.quality_result()
        if res is None:
            return []
        tr = self.ctx.scene.traces
        made = qc_tracks(res, tr.qc_metrics, tr.qc_types or None)
        made = [t for t in made if t["id"] not in tr.qc_hidden]
        rank = {tid: k for k, tid in enumerate(tr.qc_order)}
        # Tracks the user never moved keep their default place, after those
        # the user did.
        return sorted(made, key=lambda t: (rank.get(t["id"], len(rank)),
                                           [m["id"] for m in made].index(t["id"])))

    def _sync_qc_tracks(self) -> None:
        tr = self.ctx.scene.traces
        # Rebuilt only when what they show changes; moving through the
        # recording only moves the window drawn on them.
        key = (id(self._quality), tuple(tr.qc_metrics), tuple(tr.qc_types),
               tuple(tr.qc_order), tuple(tr.qc_hidden), tr.qc_scroll)
        tracks = self._qc_track_list() if key != self._qc_tracks_key else None
        if tracks is not None:
            self._qc_tracks_key = key
            self.qc_tracks.set_mode("scroll" if tr.qc_scroll else "fit")
            self.qc_tracks.set_tracks(tracks, x_label="Time (s)")
        if self.qc_tracks.cards():
            src = self.source
            if src is not None:
                self.qc_tracks.set_x_range(0.0, float(src.duration))
                res = self.quality_result()
                step = float(res["segment_s"])
                start = float(res.get("start_time", 0.0))
                spans = [(float(t) - start, float(t) - start + step)
                         for t, bad in zip(res["times"], res["flagged"]) if bad]
                self.qc_tracks.set_regions(spans)
                self.qc_tracks.set_window((tr.t0, min(tr.t0 + tr.width, src.duration)))
                cursor = self.ctx.scene.cursor.time
                self.qc_tracks.set_marker(None if cursor is None
                                          else float(cursor) - float(src.start_time))
        for mid, act in self._qc_metric_actions.items():
            if act.isChecked() != (mid in tr.qc_metrics):
                act.blockSignals(True)
                act.setChecked(mid in tr.qc_metrics)
                act.blockSignals(False)
        if self.qc_expand_button.isChecked() != tr.qc_scroll:
            self.qc_expand_button.blockSignals(True)
            self.qc_expand_button.setChecked(tr.qc_scroll)
            self.qc_expand_button.blockSignals(False)
        if self.qc_beside_button.isChecked() != tr.qc_beside:
            self.qc_beside_button.blockSignals(True)
            self.qc_beside_button.setChecked(tr.qc_beside)
            self.qc_beside_button.blockSignals(False)
        want = Qt.Orientation.Horizontal if tr.qc_beside else Qt.Orientation.Vertical
        if self.qc_split.orientation() != want:
            self.qc_split.setOrientation(want)
            self._share_qc_room()

    def _share_qc_room(self) -> None:
        """The QC plots 40 % of the room, the traces the rest: when they
        appear and when they move beside or under. Left to the splitter, they
        opened at their minimum, one plot and a clipped second."""
        beside = self.ctx.scene.traces.qc_beside
        total = (self.qc_split.width() if beside else self.qc_split.height()) or 800
        self.qc_split.setSizes([int(total * 0.6), int(total * 0.4)])
        self.qc_tracks.request_align()

    def _set_qc_on_open(self, on: bool) -> None:
        if self.ctx.settings.qc.on_open != bool(on):
            self.ctx.settings_hub.update(lambda st: setattr(st.qc, "on_open", bool(on)))

    def _toggle_qc_metric(self, metric: str, on: bool) -> None:
        tr = self.ctx.scene.traces
        metrics = list(tr.qc_metrics)
        if on and metric not in metrics:
            metrics.append(metric)
        elif not on and metric in metrics:
            metrics.remove(metric)
        # Asking for a measure again shows every type of it again.
        hidden = [h for h in tr.qc_hidden if not (on and h.startswith(f"{metric}:"))]
        self._run_ui("traces.qc_view", metrics=metrics, hidden=hidden)

    def _fill_qc_types_menu(self) -> None:
        menu = self._qc_types_menu
        menu.clear()
        res = self.quality_result()
        tr = self.ctx.scene.traces
        present = list((res or {}).get("types", {}).items())
        every = menu.addAction("All types")
        every.setCheckable(True)
        every.setChecked(not tr.qc_types)
        every.triggered.connect(lambda: self._run_ui("traces.qc_view", types=[]))
        menu.addSeparator()
        for t, info in present:
            act = menu.addAction(info["label"])
            act.setCheckable(True)
            act.setChecked(not tr.qc_types or t in tr.qc_types)
            act.toggled.connect(lambda on, ty=t: self._toggle_qc_type(ty, on))
        if not present:
            menu.addAction("Switch QC on to list the types").setEnabled(False)

    def _toggle_qc_type(self, ch_type: str, on: bool) -> None:
        res = self.quality_result() or {}
        every = list(res.get("types", {}))
        tr = self.ctx.scene.traces
        shown = list(tr.qc_types or every)
        if on and ch_type not in shown:
            shown.append(ch_type)
        elif not on and ch_type in shown:
            shown.remove(ch_type)
        if not shown:
            return    # at least one type: an empty panel helps nobody
        self._run_ui("traces.qc_view", types=[] if set(shown) == set(every) else shown)

    def _hide_qc_track(self, track_id: str) -> None:
        tr = self.ctx.scene.traces
        self._run_ui("traces.qc_view", hidden=list(tr.qc_hidden) + [track_id])

    def _maximise_qc(self, on: bool) -> None:
        self._traces_holder.setVisible(not on)

    def _describe_qc_x(self, x: float) -> str:
        res = self.quality_result()
        if res is None:
            return f"{x:.1f} s"
        step = float(res["segment_s"])
        lo = (x // step) * step
        return f"{lo:.1f} to {lo + step:.1f} s"

    def _qc_go_to(self, x: float) -> None:
        tr = self.ctx.scene.traces
        self._run_ui("time.set", t0=max(0.0, float(x) - 0.4 * tr.width))

    def _qc_go_to_cell(self, x: float, channel: str) -> None:
        """A cell of a channel map: that channel, over that segment, on the
        traces, outlined, with the time cursor at its start."""
        res = self.quality_result()
        src = self.source
        if res is None or src is None:
            return
        step = float(res["segment_s"])
        start = math.floor(float(x) / step) * step
        self._run_ui("traces.go_to", channel=channel, onset=float(src.start_time) + start,
                     duration=step)
        self.viewer.status_message.emit(
            f"{channel}, {start:.1f} to {start + step:.1f} s (outlined on the traces)")

    def _qc_area(self) -> Optional[tuple[int, int]]:
        """The QC plots' left and right edges: the traces' plot area, its
        right edge pulled in to where the plots can reach."""
        traces = self._traces
        reach = self.qc_tracks.reach()
        if (traces is None or reach is None or not traces.plot.isVisible()
                or self.ctx.scene.traces.qc_beside):
            # Beside the traces there is no edge to share: the plots keep
            # their own axes.
            if traces is not None and getattr(self, "_traces_right_margin", 0):
                self._traces_right_margin = 0
                pi = traces.plot.getPlotItem()
                m = pi.layout.getContentsMargins()
                pi.layout.setContentsMargins(m[0], m[1], 0, m[3])
            if self.ctx.scene.traces.qc_beside:
                for card in self.qc_tracks.cards():
                    card.align(card.plot.mapToGlobal(card.plot.rect().topLeft()).x() + 56,
                               card.plot.mapToGlobal(card.plot.rect().topRight()).x() - 6)
            return None
        plot = traces.plot
        pi = plot.getPlotItem()
        r = pi.getViewBox().sceneBoundingRect()
        left = plot.mapToGlobal(plot.mapFromScene(r.topLeft())).x()
        right = plot.mapToGlobal(plot.mapFromScene(r.topRight())).x()
        # The traces' right edge pulled in to where the QC plots can reach.
        current = getattr(self, "_traces_right_margin", 0)
        full_right = right + current
        margin = int(max(0, full_right - reach[1]))
        if margin != current:
            self._traces_right_margin = margin
            m = pi.layout.getContentsMargins()
            pi.layout.setContentsMargins(m[0], m[1], margin, m[3])
        return int(max(left, reach[0])), int(min(full_right - margin, reach[1]))

    def _sync_quality(self) -> None:
        """Start QC the first time it is switched on for a recording (or
        when its parameters change); show or hide its plots and flags."""
        tr = self.ctx.scene.traces
        src = self.source
        on = bool(tr.quality and src is not None and self._file_kind == "meeg")
        settings = self.ctx.settings.meeg_qc
        # Checked again when the recording OR the user's parameters change.
        key = (id(src), settings.model_dump_json())
        if on and self._quality_of != key:
            from ....viz.commands.signal import bad_channels
            from ....viz.compute.meeg_qc import quality
            from ....viz.data.signal import line_frequency

            self._quality_of = key
            self._quality = None
            self.viewer.loading_changed.emit(True, "Running QC on the recording")
            self.ctx.jobs.start("quality", self._generation, quality, src,
                                settings=settings, line_freq=line_frequency(src),
                                exclude=bad_channels(self.ctx.store))
        shown = on and self._quality is not None and self._quality_of == key
        appeared = False
        if self.quality_row.isHidden() == shown:
            self.quality_row.setVisible(shown)
            appeared = shown
            if not shown and self.qc_max_button.isChecked():
                self.qc_max_button.setChecked(False)
        if shown:
            self._sync_qc_tracks()
            if appeared:
                self._share_qc_room()
        if self._traces is not None:
            flags = self._quality_flags() if shown else {}
            if flags != getattr(self._traces, "quality_flags", {}):
                self._traces.set_quality_flags(flags)

    def _quality_flags(self) -> dict:
        tokens = {"noisy": "warning", "flat": "accent", "uncorrelated": "warning",
                  "line noise": "purple"}
        out = {}
        for c in (self._quality or {}).get("channels", []):
            if c["reasons"]:
                follows = ("" if c.get("follows") is None
                           else f", follows its type at {c['follows']:.2f}")
                text = (", ".join(c["reasons"]) + f" (against the other {c['label']} "
                        f"channels: STD {c['std_z']:+.1f} z, peak-to-peak "
                        f"{c['ptp_z']:+.1f} z{follows})")
                out[c["name"]] = (tokens[c["reasons"][0]], text)
        return out

    def quality_result(self) -> Optional[dict]:
        """What the QC found for the recording on screen, or None."""
        src = self.source
        if self._quality is None or src is None or not self._quality_of:
            return None
        return self._quality if self._quality_of[0] == id(src) else None

    def on_settings_changed(self) -> None:
        """The QC parameters changed (here or on the Settings page): show
        them, and check again when QC is on. Other settings (a section
        folded) leave a half-edited form alone."""
        params = self.ctx.settings.meeg_qc.model_dump_json()
        if params == self._qc_params:
            return
        self._qc_params = params
        self.controls.put_qc_settings(self.ctx.settings.meeg_qc)
        self._sync_quality()
        self.controls.sync()

    def recheck_quality(self) -> None:
        """Run the QC again (new parameters, or bad channels marked since)."""
        self._quality_of = None
        if self.ctx.scene.traces.quality:
            self._sync_quality()
        else:
            self._run_ui("traces.quality", value=True)

    #: Why QC suggests a channel is bad. Line noise alone is not among them:
    #: it is usually the room, which a notch filter removes, not the channel.
    SUGGESTED_REASONS = ("flat", "noisy", "uncorrelated")
    #: What a flagged segment is labelled, by why it was flagged.
    SEGMENT_LABELS = (("BAD_muscle", "Muscle"), ("BAD_jump", "Jumps"),
                      ("BAD_noise", "Noise"))

    def _qc_channels(self, reasons=None) -> list[str]:
        """The channels QC found with any of ``reasons`` (default: the
        suggested ones), not marked bad yet."""
        from ....viz.commands.signal import bad_channels

        res = self.quality_result()
        if not res:
            return []
        wanted = set(reasons or self.SUGGESTED_REASONS)
        bads = bad_channels(self.ctx.store)
        return [c["name"] for c in res["channels"]
                if wanted & set(c["reasons"]) and c["name"] not in bads]

    def _qc_segments(self, label=None) -> dict[str, list]:
        """The flagged segments, ``(onset, duration)`` by the label each gets
        (only ``label``'s when given)."""
        from ....viz.compute.meeg_qc import bad_label

        res = self.quality_result()
        if res is None:
            return {}
        step = float(res["segment_s"])
        by_label: dict[str, list] = {}
        for t, bad, kinds in zip(res["times"], res["flagged"], res["segment_kinds"]):
            if not bad:
                continue
            name = bad_label(kinds)
            if label is None or name == label:
                by_label.setdefault(name, []).append((float(t), step))
        return by_label

    def mark_channels(self, reasons=None) -> int:
        """The channels QC found marked bad: the suggested ones (flat, noisy,
        uncorrelated), or those with any of ``reasons``. One undoable step."""
        names = self._qc_channels(reasons)
        if not names:
            return 0
        self._run_ui("channels.set_bad", names=names)
        self.viewer.status_message.emit(
            f"{len(names)} channel{'s' if len(names) != 1 else ''} marked bad; Save to "
            "dataset writes them into the run's _channels.tsv")
        return len(names)

    def mark_segments(self, label=None) -> int:
        """The flagged segments as bad segments, labelled by why
        (``BAD_muscle``, ``BAD_jump``, ``BAD_noise``), or only ``label``'s."""
        n = 0
        for name, segments in self._qc_segments(label).items():
            self._run_ui("annotate.add_many", segments=segments, label=name)
            n += len(segments)
        if n:
            self.viewer.status_message.emit(
                f"{n} flagged segment{'s' if n != 1 else ''} marked bad; Save to dataset "
                "writes them into the run's events.tsv")
        return n

    def _fill_qc_mark_menu(self) -> None:
        """The Mark as bad menu, counted for the QC on screen (a count of
        zero is shown, disabled, so the menu never changes shape)."""
        from ..menus import submenu

        menu = self._qc_mark_menu
        menu.clear()

        def item(target, text: str, n: int, tip: str, run, after: str = "") -> None:
            act = target.addAction(f"{text} ({n}){after}")
            act.setToolTip(tip)
            act.setEnabled(n > 0)
            act.triggered.connect(lambda _c=False: run())

        item(menu, "Suggested channels", len(self._qc_channels()),
             "Every channel QC found flat, noisy or uncorrelated with the others of "
             "its type, not marked bad yet", self.mark_channels)
        by_reason = submenu(menu, "Channels by reason")
        by_reason.setToolTipsVisible(True)
        for reason, text, tip in (
                ("flat", "Flat", "No signal: disconnected, or saturated"),
                ("noisy", "Noisy", "Far louder than the others of its type, and not "
                                   "following them"),
                ("uncorrelated", "Uncorrelated", "At a normal level but not following "
                                                 "the others of its type"),
                ("line noise", "Line noise", "Strong power at the mains frequency. Not "
                                             "suggested: it is usually the room, which a "
                                             "notch filter removes")):
            item(by_reason, text, len(self._qc_channels((reason,))), tip,
                 lambda r=reason: self.mark_channels((r,)))
        menu.addSeparator()
        segments = self._qc_segments()
        item(menu, "Flagged segments", sum(len(v) for v in segments.values()),
             "Every segment QC flagged, each labelled by why", self.mark_segments)
        by_why = submenu(menu, "Segments by why")
        by_why.setToolTipsVisible(True)
        for label, text in self.SEGMENT_LABELS:
            item(by_why, text, len(segments.get(label, [])),
                 f"The segments flagged for {text.lower()}, labelled {label}",
                 lambda lab=label: self.mark_segments(lab), after=f", as {label}")

    def show_quality_report(self):
        from ..panels.quality_report import QualityReport

        if self._quality is None or self.source is None:
            return None
        dlg = QualityReport(self._quality, self.source.path.name, parent=self.viewer)
        dlg.go_to.connect(lambda t: self._run_ui(
            "time.set", t0=max(0.0, t - self.source.start_time
                               - 0.4 * self.ctx.scene.traces.width)))
        dlg.mark_channels.connect(lambda names: self._run_ui("channels.set_bad", names=names))
        dlg.mark_segments.connect(self.mark_segments)
        self.quality_report = dlg
        dlg.show()
        return dlg

    def _build_navigation(self) -> QWidget:
        """|<  <  [the whole recording]  >  >|  and where the window is."""
        from ..canvases.overview import OverviewBar

        nav = QFrame()
        nav.setObjectName("toolbar")
        lay = QHBoxLayout(nav)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(4)
        manager = self.viewer.action_manager
        lay.addWidget(manager.button("time.start"))
        prev = QPushButton("<")
        prev.setObjectName("tb-btn")
        prev.setToolTip(manager.actions["time.prev"].toolTip())
        prev.clicked.connect(lambda: self.viewer.trigger("time.prev"))
        lay.addWidget(prev)
        self.overview = OverviewBar(self.ctx)
        self.overview.setToolTip("The whole recording: activity, events, bad spans and "
                                 "gaps. Click to go there, drag the window, wheel to page.")
        lay.addWidget(self.overview, 1)
        nxt = QPushButton(">")
        nxt.setObjectName("tb-btn")
        nxt.setToolTip(manager.actions["time.next"].toolTip())
        nxt.clicked.connect(lambda: self.viewer.trigger("time.next"))
        lay.addWidget(nxt)
        end = manager.button("time.end")
        lay.addWidget(end)
        # Sized to their text: a push button's default floor is 80 px, and
        # four of them made this row a 540 px floor under the whole pane.
        for btn in (nav.findChildren(QPushButton)):
            fm = btn.fontMetrics()
            btn.setFixedWidth(max(28, fm.horizontalAdvance(btn.text()) + 22))
        self.time_label = ElidedLabel("")
        self.time_label.setObjectName("sidecar-footer-summary")
        lay.addWidget(self.time_label)
        return nav

    # ------------------------------------------------------------------
    # Toolbar widgets
    # ------------------------------------------------------------------

    def toolbar_rows(self):
        return TOOLBAR_ROWS

    def make_widget(self, name: str):
        return {
            "preset": self._widget_preset, "psd": self._widget_psd,
            "scale": self._widget_scale, "count": self._widget_count,
        }.get(name, lambda: None)()

    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sidecar-footer-summary")
        return lbl

    def _group(self, *widgets) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        for w in widgets:
            h.addWidget(w)
        return box

    @staticmethod
    def _fit(spin: QDoubleSpinBox, widest: str) -> None:
        """As wide as its widest text in ITS font, whatever the platform,
        DPI or app font scale: a fixed pixel width cut "3600.0 s" off on a
        larger font."""
        spin.ensurePolished()
        spin.setFixedWidth(spin.fontMetrics().horizontalAdvance(widest) + 22)

    def _number(self, lo: float, hi: float, *, step: float, unit: str = "",
                default: Optional[float] = None, log: bool = False,
                decimals: Optional[int] = None, tip: str = "",
                widest: str = "0000.00") -> NumberControl:
        ctl = NumberControl(lo, hi, step=step, unit=unit, default=default, log=log,
                            decimals=decimals)
        ctl.slider.setMinimumWidth(_TOOLBAR_SLIDER_PX)
        ctl.slider.setFixedWidth(_TOOLBAR_SLIDER_PX)
        self._fit(ctl.spin, widest)
        if tip:
            ctl.setToolTip(tip + (f" Double-click the slider for {default:g}."
                                  if default is not None else ""))
        return ctl

    def _widget_channels(self) -> list[QWidget]:
        self.type_combo = QComboBox()
        self.type_combo.setToolTip("Which channel type to show")
        self.type_combo.currentIndexChanged.connect(
            lambda _i: self._run_ui("traces.type", ch_type=self.type_combo.currentData()))
        self.channels_button = QPushButton("Channels...")
        self.channels_button.setObjectName("tb-btn")
        self.channels_button.setToolTip("Pick the channels to show")
        self.channels_button.clicked.connect(self.open_channel_picker)
        self._multi_channel_widgets = [self.type_combo, self.channels_button]
        return list(self._multi_channel_widgets)

    def _widget_count(self) -> QWidget:
        """How many channels are on screen: in the toolbar, beside the
        scaling, since it is changed as often."""
        self.count_control = self._number(
            1, 500, step=1, decimals=0, default=sigcmd.DEFAULT_COUNT, widest="500",
            tip="How many channels are displayed at once (PgUp and PgDown page through "
                "the rest; the wheel scrolls one at a time).")
        self.count_control.value_changed.connect(
            lambda v: self._run_ui("traces.count", n=int(round(v))))
        group = self._group(self._label("Displayed channels"), self.count_control)
        # Shown for MEG and EEG with more than one channel (physio shows them all).
        self._count_group = group
        return group

    def _widget_scale(self) -> list[QWidget]:
        # Logarithmic, every step the same ratio, so the slider is as fine at
        # 0.1 as at 10.
        self.scale_control = self._number(
            0.01, 1000.0, step=0.1, decimals=2, default=1.0, log=True,
            tip="How large the traces are drawn, as a factor of their automatic scale "
                "([ and ] change it from the keyboard).")
        self.scale_control.value_changed.connect(
            lambda v: self._run_ui("traces.scale", value=float(v)))
        self.width_control = self._number(
            0.1, 3600.0, step=0.5, unit="s", decimals=1, default=sigcmd.DEFAULT_WIDTH_S,
            log=True, widest="3600.0 s", tip="Seconds on screen (= and - change it).")
        self.width_control.value_changed.connect(
            lambda v: self._run_ui("time.width", seconds=float(v)))
        return [self._group(self._label("Scaling"), self.scale_control),
                self._group(self._label("Time span"), self.width_control)]

    def _widget_events(self) -> QWidget:
        self.event_source = QComboBox()
        self.event_source.setToolTip("Where the events come from: the run's events.tsv, "
                                     "the trigger channel, or the recording's annotations")
        self.event_source.currentIndexChanged.connect(
            lambda _i: self._run_ui("events.source", which=self.event_source.currentData()))
        return self.event_source

    def _widget_preset(self) -> QWidget:
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip("A common band, applied at once (the notch is kept); "
                                     "the exact cut-offs and a notch are in the controls "
                                     "column, under Filters")
        self.preset_combo.activated.connect(self._on_preset)
        return self._group(self._label("Filter"), self.preset_combo)

    def _widget_filter_fields(self) -> None:
        """The three cut-offs and their Apply (the controls column)."""
        self.hp_spin, self.lp_spin, self.notch_spin = (QDoubleSpinBox() for _ in range(3))
        for spin, hi, tip in ((self.hp_spin, 500.0, "High-pass (Hz)"),
                              (self.lp_spin, 5000.0, "Low-pass (Hz)"),
                              (self.notch_spin, 1000.0, "Notch (Hz)")):
            spin.setRange(0.0, hi)
            spin.setDecimals(_DECIMALS)
            spin.setSingleStep(0.1)
            spin.setSpecialValueText("Off")
            self._compact(spin)
            spin.setToolTip(f"{tip}. Zero-phase. A cut-off at or above the Nyquist "
                            "frequency is refused, with the reason. Enter applies.")
            spin.lineEdit().returnPressed.connect(self.apply_filters)
        self.filter_apply_button = QPushButton("Apply")
        self.filter_apply_button.setObjectName("tb-btn")
        self.filter_apply_button.setToolTip("Apply the three cut-offs")
        self.filter_apply_button.clicked.connect(self.apply_filters)

    def _compact(self, spin: QDoubleSpinBox, widest: str = "5000.000") -> None:
        """A number field like the sliders' own: no arrows, a decimal point
        whatever the system locale, as wide as its numbers."""
        spin.setObjectName("viz-number")
        spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        spin.setLocale(self._c_locale())
        self._fit(spin, widest)

    @staticmethod
    def _c_locale():
        # A decimal point whatever the system says: "0,010" reads as ten to
        # an English reader (the same reason every number field here is C).
        from PyQt6.QtCore import QLocale

        loc = QLocale(QLocale.Language.C)
        loc.setNumberOptions(QLocale.NumberOption.OmitGroupSeparator)
        return loc

    def _presets(self) -> tuple:
        return PRESETS.get("physio" if self._file_kind == "physio" else "meeg", ())

    def _on_preset(self, index: int) -> None:
        data = self.preset_combo.itemData(index)
        if data is None or data == "custom":
            return
        hp, lp = data
        tr = self.ctx.scene.traces
        try:
            self.ctx.run("traces.filter", hp=hp, lp=lp, notch=tr.notch)
        except ValueError as exc:
            self.viewer.status_message.emit(str(exc))
            QMessageBox.information(self.viewer, "Filter not applied", str(exc))
            self.sync_widgets()
            return
        from ....viz.compute.filters import FilterSpec

        tr = self.ctx.scene.traces
        self.viewer.status_message.emit(
            f"Filter: {FilterSpec(tr.hp, tr.lp, tr.notch).describe()}")

    def _widget_resample(self) -> list[QWidget]:
        self.resample_spin = QDoubleSpinBox()
        self.resample_spin.setRange(0.0, 10000.0)
        self.resample_spin.setDecimals(0)
        self.resample_spin.setSpecialValueText("Off")
        self.resample_spin.setSuffix(" Hz")
        self.resample_spin.setToolTip("A new sampling rate: the recording is resampled "
                                      "on a worker, and the view kept where it is")
        self._compact(self.resample_spin, "10000 Hz")
        self.resample_button = QPushButton("Resample")
        self.resample_button.setObjectName("tb-btn")
        self.resample_button.clicked.connect(self.resample)
        return [self.resample_spin, self.resample_button]

    def _widget_psd(self) -> QWidget:
        btn = QPushButton("PSD")
        btn.setObjectName("tb-btn")
        btn.setToolTip("Welch power spectrum of the channels shown")
        menu = popup_menu(btn)
        self._psd_raw = menu.addAction("Spectrum of the raw signal")
        self._psd_raw.triggered.connect(lambda: self.show_psd(filtered=False))
        self._psd_filtered = menu.addAction("Spectrum of the filtered signal")
        self._psd_filtered.triggered.connect(lambda: self.show_psd(filtered=True))
        menu.aboutToShow.connect(
            lambda: self._psd_filtered.setEnabled(self._filter_active()))
        btn.setMenu(menu)
        self.psd_button = btn
        return btn

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: Path, root: Optional[Path]) -> None:
        if self._memory_timer.isActive():
            self._memory_timer.stop()
            self._remember()
        self._generation += 1
        self._release()
        self._path = Path(path)
        self._root = root
        self._file_kind = "meeg" if formats.is_recording_path(path) else "physio"
        self._relatives = []
        if self._file_kind == "physio":
            from ....viz.data.physio import related_recordings

            try:
                self._relatives = related_recordings(self._path)
            except OSError:
                self._relatives = []
            self.load_signal()
        else:
            self.ctx.jobs.start("meta", self._generation, _read_meta, self._path)

    def load_signal(self) -> None:
        if self._path is None:
            return
        self.viewer.show_loading(f"Loading signal: {self._path.name}")
        self.ctx.jobs.start("signal", self._generation, _open, self._path, self._root,
                            self._file_kind, self.together and len(self._relatives) > 1)

    def _release(self) -> None:
        for tag in ("meta", "signal", "resample", "psd", "overview", "quality"):
            self.ctx.jobs.cancel(tag)
        self._quality = None
        self._quality_of = None
        self.source = None
        self.meta = None
        store = self.ctx.store
        store.sources.clear()

    def clear(self) -> None:
        self._generation += 1
        self._release()
        self._path = None
        self.ctx.store.replace_scene(Scene())
        self.ctx.qstore.flush()

    def stop(self) -> None:
        if self._memory_timer.isActive():
            self._memory_timer.stop()
            self._remember()

    def _remember(self) -> None:
        """The trace options and filters, for the next recording of this
        kind, in any window (``viz.memory``)."""
        from ....viz import memory

        if self.source is None:
            return
        prefs = memory.trace_prefs(self.ctx.scene.traces)
        kind = self._file_kind
        self.ctx.settings_hub.update(lambda s: s.traces_state.__setitem__(kind, prefs))

    def _on_job_done(self, tag: str, generation: int, result) -> None:
        if generation != self._generation:
            return
        if tag == "meta":
            self.meta = result
            self._fill_meta(result)
            self.pages.setCurrentWidget(self.meta_page)
            self.viewer.on_first_frame()
            self.viewer.on_loaded(self._path)
            self.viewer.status_message.emit(
                f"{result.get('name', '')}: {result.get('n_channels', 0)} ch, "
                f"{result.get('sfreq', 0):.0f} Hz, {result.get('duration', 0):.1f} s")
            if self.load_on_open:
                self.load_signal()
        elif tag == "signal":
            self._adopt(result)
        elif tag == "resample":
            self._adopt(result, keep_view=True)
            self.viewer.loading_changed.emit(False, "")
            self.resample_button.setEnabled(True)
            self.viewer.status_message.emit(
                f"Resampled to {result.sfreq:.0f} Hz ({result.n_times:,} samples)")
        elif tag == "quality":
            self.viewer.loading_changed.emit(False, "")
            self._quality = result
            self._sync_quality()
            self.controls.sync()
            self.viewer.status_message.emit(f"QC: {result['summary']}")
        elif tag == "psd":
            self.viewer.loading_changed.emit(False, "")
            self.psd_button.setEnabled(True)
            from ..canvases.psd import PsdWindow

            win = PsdWindow(result, parent=self.viewer)
            win.show()
            self.psd_window = win
            for message in result.get("messages", []):
                self.viewer.status_message.emit(message)

    def _on_job_failed(self, tag: str, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        if tag in ("meta",):
            self.viewer.on_load_failed(self._path, message)
        elif tag == "signal":
            if self._file_kind == "meeg" and self.meta is not None:
                self.pages.setCurrentWidget(self.meta_page)
                self.viewer.show_content()
                QMessageBox.warning(self.viewer, "Load error",
                                    f"Could not load {self._path.name}:\n{message}")
            else:
                self.viewer.on_load_failed(self._path, message)
        elif tag == "quality":
            self.viewer.loading_changed.emit(False, "")
            self._quality_of = None
            self._run_ui("traces.quality", value=False)
            QMessageBox.information(self.viewer, "QC", message)
        elif tag in ("resample", "psd"):
            self.viewer.loading_changed.emit(False, "")
            for btn in (getattr(self, "resample_button", None), getattr(self, "psd_button", None)):
                if btn is not None:
                    btn.setEnabled(True)
            QMessageBox.warning(self.viewer, "Signal", message)

    def _adopt(self, src: SignalSource, *, keep_view: bool = False) -> None:
        """Put a read recording on screen. Everything measured in its own
        terms resets; standing preferences (dark plot, event colour) stay."""
        self.source = src
        store = self.ctx.store
        sid = "sig0"
        store.sources = {sid: src}
        if keep_view:
            scene = store.scene
            scene.traces.t0 = min(scene.traces.t0, max(0.0, src.duration - scene.traces.width))
            store.changed({"sources:sig0", "traces.time"})
        else:
            from ....viz import memory

            scene = Scene()
            scene.sources = {sid: SourceRef(id=sid, path=str(src.path), kind="signal")}
            scene.layers = [SignalLayer(id="sig", source=sid, name=src.path.name)]
            # Its own defaults, with the options left on the last recording
            # of this kind that suit it (``viz.memory``).
            state = memory.restore_traces(
                sigcmd.opening_state(src), self.ctx.settings.traces_state.get(self._file_kind),
                src, qc_on_open=self.ctx.settings.qc.on_open)
            if self._file_kind == "physio":
                # A run's physio is a handful of channels: all of them, always,
                # never a count to set.
                state["count"] = max(1, len(src.ch_names))
                state["offset"] = 0
            scene.traces = TracesState(**state)
            store.replace_scene(scene)
        self._ensure_traces()
        self.pages.setCurrentWidget(self.traces_page)
        self.viewer.on_first_frame()
        self.viewer.on_loaded(src.path)
        if src.note:
            self.viewer.status_message.emit(src.note)
        if not keep_view:
            types = ", ".join(src.type_label(t) for t in src.available_types)
            self.viewer.status_message.emit(
                f"Loaded {len(src.ch_names)} channels, {src.duration:.1f} s at "
                f"{src.sfreq:.0f} Hz | Types: {types}")
        self.ctx.qstore.flush()

    # ------------------------------------------------------------------
    # The metadata card
    # ------------------------------------------------------------------

    def _meta_clear(self) -> None:
        while self._meta_layout.count() > 1:
            item = self._meta_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _meta_add(self, widget: QWidget) -> None:
        self._meta_layout.insertWidget(self._meta_layout.count() - 1, widget)

    def _meta_section(self, title: str) -> None:
        lbl = QLabel(title)
        lbl.setObjectName("viewer-meta-section")
        self._meta_add(lbl)

    def _meta_row(self, key: str, value: str) -> None:
        row = QWidget()
        row.setObjectName("viewer-meta-row")
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        k = QLabel(key)
        k.setObjectName("viewer-meta-key")
        k.setMinimumWidth(96)
        k.setMaximumWidth(140)
        v = QLabel(value)
        v.setObjectName("viewer-meta-val")
        v.setWordWrap(True)
        v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        h.addWidget(k)
        h.addWidget(v, 1)
        self._meta_add(row)

    def _fill_meta(self, meta: dict) -> None:
        self._meta_clear()
        title = QLabel(meta.get("name", ""))
        title.setObjectName("viewer-meta-title")
        title.setWordWrap(True)
        self._meta_add(title)
        self._meta_section("Recording")
        self._meta_row("Channels", str(meta.get("n_channels", 0)))
        self._meta_row("Sampling rate", f"{meta.get('sfreq', 0):.1f} Hz")
        dur = meta.get("duration", 0.0)
        self._meta_row("Duration", f"{dur:.1f} s ({dur / 60:.1f} min)")
        self._meta_row("Time points", str(meta.get("n_times", 0)))
        if meta.get("meas_date"):
            self._meta_row("Measurement date", str(meta["meas_date"]))
        self._meta_section("Filters")
        for key, label in (("highpass", "High-pass"), ("lowpass", "Low-pass"),
                           ("line_freq", "Line frequency")):
            value = meta.get(key)
            self._meta_row(label, f"{value:g} Hz" if value not in (None, "") else "n/a")
        self._meta_section("Channel types")
        for ct, n in sorted(meta.get("ch_type_counts", {}).items()):
            label = "mag (axial grad)" if meta.get("is_ctf") and ct == "mag" else ct
            self._meta_row(label, str(n))
        if meta.get("is_ctf"):
            note = QLabel("CTF dataset: MEG sensors are axial gradiometers "
                          "(MNE labels them “mag”).")
            note.setObjectName("viewer-meta-note")
            note.setWordWrap(True)
            self._meta_add(note)
        bads = meta.get("bads") or []
        if bads:
            self._meta_section(f"Bad channels ({len(bads)})")
            self._meta_row("", ", ".join(bads))

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    _syncing = False

    def _run_ui(self, command_id: str, **params) -> None:
        if self._syncing:
            return
        try:
            self.ctx.run(command_id, **params)
        except ValueError as exc:
            self.viewer.status_message.emit(str(exc))

    def _filter_active(self) -> bool:
        tr = self.ctx.scene.traces
        return bool(tr.hp or tr.lp or tr.notch)

    def apply_filters(self) -> None:
        """Apply the three fields at once; a refused cut-off is said, and
        nothing changes."""
        try:
            changed = self.ctx.run("traces.filter", hp=self.hp_spin.value() or None,
                                   lp=self.lp_spin.value() or None,
                                   notch=self.notch_spin.value() or None)
        except ValueError as exc:
            self.viewer.status_message.emit(str(exc))
            QMessageBox.information(self.viewer, "Filter not applied", str(exc))
            return
        from ....viz.compute.filters import FilterSpec

        tr = self.ctx.scene.traces
        if changed:
            self.viewer.status_message.emit(
                f"Filter: {FilterSpec(tr.hp, tr.lp, tr.notch).describe()}")

    def resample(self) -> None:
        freq = self.resample_spin.value()
        if freq <= 0 or self.source is None:
            return
        from ....viz.data.signal import resampled

        self.viewer.loading_changed.emit(True, f"Resampling to {freq:.0f} Hz")
        self.resample_button.setEnabled(False)
        self.ctx.jobs.start("resample", self._generation, resampled, self.source, freq)

    def show_psd(self, *, filtered: bool = False) -> None:
        """The spectrum of the channels SHOWN, raw or filtered by choice."""
        src = self.source
        if src is None:
            return
        from ....viz.compute.filters import FilterSpec
        from ....viz.compute.spectral import psd

        tr = self.ctx.scene.traces
        indices = src.picks_for(tr.ch_type, tr.picks) or list(range(len(src.ch_names)))
        spec = FilterSpec(tr.hp, tr.lp, tr.notch) if filtered else None
        self.viewer.loading_changed.emit(True, "Computing the spectrum")
        self.psd_button.setEnabled(False)
        self.ctx.jobs.start("psd", self._generation, psd, src, indices, spec=spec)

    def open_channel_picker(self) -> None:
        src = self.source
        if src is None:
            return
        dlg = ChannelPicker(src, self.ctx.scene.traces.picks, parent=self.viewer)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._run_ui("traces.pick", names=dlg.selected())
            n = len(src.picks_for("all", self.ctx.scene.traces.picks))
            self.viewer.status_message.emit(f"Showing {n} channels")

    # ------------------------------------------------------------------
    # Scene -> widgets
    # ------------------------------------------------------------------

    def _on_changed(self, paths) -> None:
        if self.source is not None and any(
                p.startswith("traces") and p not in ("traces.annotations",) for p in paths):
            self._memory_timer.start()
        self._sync_quality()
        self.sync_widgets()
        self.viewer.update_footer()

    def sync_widgets(self) -> None:
        src = self.source
        tr = self.ctx.scene.traces
        self._syncing = True
        try:
            if not hasattr(self, "type_combo"):
                return
            if src is not None:
                wanted = [("all", "all")]
                if "mag" in src.available_types and "grad" in src.available_types:
                    wanted.append(("mag+grad", "mag+grad"))
                wanted += [(t, src.type_label(t)) for t in src.available_types]
                have = [(self.type_combo.itemData(i), self.type_combo.itemText(i))
                        for i in range(self.type_combo.count())]
                if have != wanted:
                    self.type_combo.clear()
                    for data, text in wanted:
                        self.type_combo.addItem(text, data)
                i = self.type_combo.findData(tr.ch_type)
                self.type_combo.setCurrentIndex(max(0, i))
                multi = len(src.ch_names) > 1
                # A type filter, a picker and a count can only say the same
                # thing about one channel: noise in a full toolbar.
                for w in self._multi_channel_widgets:
                    w.setVisible(multi)
                # Physio shows every channel: no count to set.
                count_group = getattr(self, "_count_group", None)
                if count_group is not None:
                    count_group.setVisible(multi and self._file_kind == "meeg")
                pool = len(src.picks_for(tr.ch_type, tr.picks)) or 1
                self.count_control.set_range(1, max(1, min(500, pool)))
                self.width_control.set_range(0.1, max(0.2, src.duration))
            self.count_control.set_value(tr.count)
            self.scale_control.set_value(tr.scale)
            self.width_control.set_value(tr.width)
            for spin, value in ((self.hp_spin, tr.hp), (self.lp_spin, tr.lp),
                                (self.notch_spin, tr.notch)):
                spin.setValue(value or 0.0)
            self._sync_presets(tr)
            if src is not None:
                sources = [("auto", "Events: auto")] + [(x, x) for x in src.event_sources()]
                have = [self.event_source.itemData(i) for i in range(self.event_source.count())]
                if have != [d for d, _t in sources]:
                    self.event_source.clear()
                    for data, text in sources:
                        self.event_source.addItem(text, data)
                self.event_source.setCurrentIndex(
                    max(0, self.event_source.findData(tr.event_source)))
                self.event_source.setVisible(bool(src.event_sources()))
                self._sync_annotation_bar(src, tr)
                t1 = min(tr.t0 + tr.width, src.duration)
                self.time_label.setText(f"{tr.t0 + src.start_time:.1f} - "
                                        f"{t1 + src.start_time:.1f} / "
                                        f"{src.duration:.1f} s")
            self.controls.sync()
        finally:
            self._syncing = False

    def _sync_annotation_bar(self, src, tr) -> None:
        from ....viz.commands.signal import bad_channels, bad_spans

        self.annotation_bar.setVisible(bool(tr.annotate) and self._file_kind == "meeg")
        if not tr.annotate:
            return
        if self.label_combo.currentText() != tr.annotate_label:
            self.label_combo.setCurrentText(tr.annotate_label)
        spans = bad_spans(self.ctx.store)
        seconds = sum(sp.duration for sp in spans)
        n_bad = len(bad_channels(self.ctx.store))
        self.review_summary.setText(
            f"{n_bad} bad channel{'s' if n_bad != 1 else ''} · {len(spans)} bad "
            f"segment{'s' if len(spans) != 1 else ''} ({seconds:.1f} s)")
        from ....viz.data.signal import channels_sibling

        self.review_summary.setToolTip(
            "" if channels_sibling(src.path) is not None else
            "This recording has no _channels.tsv beside it: its bad channels are kept "
            "for this session but cannot be saved to the dataset.")

    def _sync_presets(self, tr) -> None:
        presets = self._presets()
        wanted = [("No filter", (None, None))] + [(label, (hp, lp)) for label, hp, lp in presets]
        have = [self.preset_combo.itemText(i) for i in range(self.preset_combo.count())]
        if have != [t for t, _d in wanted] + ["Custom"]:
            self.preset_combo.clear()
            for text, data in wanted:
                self.preset_combo.addItem(text, data)
            self.preset_combo.addItem("Custom", "custom")
        kind = "physio" if self._file_kind == "physio" else "meeg"
        label = "No filter" if tr.hp is None and tr.lp is None else preset_of(kind, tr.hp, tr.lp)
        index = self.preset_combo.findText(label) if label else -1
        self.preset_combo.setCurrentIndex(index if index >= 0 else self.preset_combo.count() - 1)

    def action_context(self) -> dict[str, Any]:
        src = self.source
        showing = src is not None and self.pages.currentWidget() is self.traces_page
        tr = self.ctx.scene.traces
        return {
            "traces": showing,
            "meeg": self._file_kind == "meeg",
            "physio": self._file_kind == "physio",
            "fit": showing and self._file_kind == "physio",
            "filtered": bool(tr.hp or tr.lp or tr.notch),
            "events": bool(src and src.event_sources()),
            "events.on": tr.events,
            "normalize": tr.normalize,
            "relatives": len(self._relatives) > 1,
            "together": self.together,
            "traces.butterfly": tr.butterfly,
            "traces.clip": tr.clip,
            "traces.remove_dc": tr.remove_dc,
            "traces.page_scale": tr.page_scale,
            "zen": self.zen,
            # Something to write: the session's review differs from the files.
            "review.changed": self._review_changed(),
            "annotate": bool(tr.annotate),
            "span.selected": tr.selected_span is not None,
            "quality": bool(tr.quality),
            "panel.inspector": self.inspector_open(),
        }

    def toolbar_wanted(self) -> bool:
        """The controls act on traces; the metadata card has its own button.
        Zen mode hides them."""
        return (self.source is not None and self.pages.currentWidget() is self.traces_page
                and not self.zen)

    #: Which mouse tables the help shows.
    mouse_canvases = ("traces",)

    def apply_mode(self) -> None:
        pass

    def apply_graph(self) -> None:
        pass

    def restore_panels(self) -> None:
        if self.ctx.settings.traces.controls_open and not self.inspector_open():
            self.set_inspector(True, remember=False)

    def canvases(self, kind: str) -> list:
        if kind == "traces" and self._traces is not None and self._traces.isVisible():
            return [self._traces]
        return []

    def figure_widget(self) -> QWidget:
        if self._traces is not None and self.pages.currentWidget() is self.traces_page:
            return self._traces
        return self.pages

    # ------------------------------------------------------------------
    # GUI actions
    # ------------------------------------------------------------------

    def gui_action(self, name: str, params) -> bool:
        if name == "inspector":
            self.set_inspector(not self.inspector_open())
            return True
        if name == "psd":
            self.show_psd(filtered=False)
            return True
        if name == "line":
            self.open_line_style()
            return True
        if name == "together":
            self.together = not self.together
            self.load_signal()
            return True
        if name == "close_signal":
            self.close_signal()
            return True
        if name == "save_review":
            self.save_review()
            return True
        if name == "zen":
            self.set_zen(not self.zen)
            return True
        return False

    def set_zen(self, on: bool) -> None:
        """The traces alone: no toolbar, no overview. Z again, or closing
        the signal, brings them back."""
        self.zen = bool(on)
        self.navigation.setVisible(not self.zen)
        self.viewer._sync_toolbar()
        self.viewer.refresh_actions()
        if self.zen:
            key = ", ".join(self.viewer.action_manager.key_labels("view.zen")) or "Z"
            self.viewer.status_message.emit(f"Zen mode: {key} brings the controls back")

    def _bads_changed(self) -> bool:
        src = self.source
        tr = self.ctx.scene.traces
        return bool(src is not None and tr.bads is not None and set(tr.bads) != set(src.bads))

    def _review_changed(self) -> bool:
        """Whether there is something the dataset can take: bad channels
        when the recording has a _channels.tsv, bad segments always (a run
        without an events table gets one)."""
        from ....viz.commands.signal import spans_changed
        from ....viz.data.signal import channels_sibling

        src = self.source
        if src is None:
            return False
        bads = self._bads_changed() and channels_sibling(src.path) is not None
        return bool(bads or spans_changed(self.ctx.store))

    def save_review(self) -> bool:
        """The bad channels into the recording's ``_channels.tsv`` and the
        bad segments into the run's ``_events.tsv``, as ONE undoable
        operation of the dataset. A failure is shown, not only said in the
        status line (where it went unnoticed)."""
        from ....editor.annotations import save_review
        from ....viz.bids import dataset_root
        from ....viz.commands.signal import bad_channels, bad_spans, spans_changed
        from ....viz.data.events import Event, events_sibling, read_events_tsv, run_base
        from ....viz.data.signal import channels_sibling

        src = self.source
        if src is None or self.viewer.read_only:
            return False
        tr = self.ctx.scene.traces
        root = self.viewer.current_root() or dataset_root(src.path) or src.path.parent
        bads = bad_channels(self.ctx.store)
        spans = bad_spans(self.ctx.store)
        channels_tsv = channels_sibling(src.path)
        events_tsv = events_sibling(src.path) or src.path.parent / f"{run_base(src.path)}_events.tsv"
        write_bads = tr.bads is not None and set(tr.bads) != set(src.bads)
        write_spans = spans_changed(self.ctx.store)
        problems = []
        if write_bads and channels_tsv is None:
            problems.append(f"{src.path.name} has no _channels.tsv beside it, so the bad "
                            "channels cannot be written.")
            write_bads = False
        try:
            result = save_review(
                root, recording=src.path.name,
                channels_tsv=channels_tsv if write_bads else None,
                bads=bads if write_bads else None,
                events_tsv=events_tsv if write_spans else None,
                spans=spans if write_spans else None, sfreq=src.sfreq)
        except Exception as exc:  # noqa: BLE001 - said, never swallowed
            log.exception("could not save the review of %s", src.path)
            QMessageBox.warning(self.viewer, "Not saved",
                                f"The review of {src.path.name} was not written:\n{exc}")
            return False
        if write_bads:
            src.bads = set(bads)
            src.bads_from = channels_tsv.name
        if write_spans:
            src.events_tsv = read_events_tsv(events_tsv)
            src.bad_spans = [Event(sp.onset, sp.duration, sp.label, "bad") for sp in spans]
        self.ctx.store.changed({"traces.look", "traces.annotations"})
        self.viewer.refresh_actions()
        done = []
        if result["channels"]:
            done.append(f"{len(bads)} bad channel{'s' if len(bads) != 1 else ''} in "
                        f"{channels_tsv.name}")
        if result["segments"]:
            done.append(f"{len(spans)} bad segment{'s' if len(spans) != 1 else ''} in "
                        f"{events_tsv.name}")
        message = ("Saved " + " and ".join(done) + "; undo it from the Editor's history"
                   if done else "The dataset already says so")
        if problems:
            QMessageBox.information(self.viewer, "Saved in part", "\n".join(problems))
        self.viewer.status_message.emit(message)
        return bool(done)

    def close_signal(self) -> None:
        """Drop the samples, back to the metadata card (MEG/EEG) or out."""
        self._generation += 1
        for tag in ("signal", "resample", "psd", "quality"):
            self.ctx.jobs.cancel(tag)
        self._quality = None
        self._quality_of = None
        self.source = None
        self.ctx.store.sources.clear()
        if self.zen:
            self.set_zen(False)
        if self._file_kind == "meeg" and self.meta is not None:
            self.pages.setCurrentWidget(self.meta_page)
            self.viewer.refresh_actions()
            self.viewer._sync_toolbar()
            self.viewer.status_message.emit("Signal closed")
        self.viewer.close_requested.emit()

    def open_line_style(self) -> None:
        from ..panels.line_style import MAX_WIDTH, LineStyleDialog

        src = self.source
        traces = self._traces
        shown = len(traces.shown_channels()) if traces is not None else 1
        cap = MAX_WIDTH if (traces is None or traces.max_pen_width(shown) > 1) else 1
        ts = self.ctx.settings.traces
        types = list(dict.fromkeys(src.ch_types)) if src is not None else []
        dlg = LineStyleDialog(ts.line_width, ts.line_color or None, channel_types=types,
                              max_width=cap, traces_shown=shown, parent=self.viewer)
        self.line_dialog = dlg
        dlg.exec()

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    def summary(self) -> str:
        src = self.source
        if src is not None:
            # A table names its BIDS suffix (physio, stim, motion); a
            # recording its format.
            kind = (suffix_of(src.path.name) or "physio") if self._file_kind == "physio" \
                else full_ext(src.path).lstrip(".")
            return (f"{len(src.ch_names)} ch · {src.sfreq:g} Hz · "
                    f"{src.duration:.1f} s · {kind}")
        if self.meta:
            m = self.meta
            return f"{m.get('n_channels', 0)} ch · {m.get('sfreq', 0):g} Hz"
        return ""

    def readout(self) -> str:
        tr = self.ctx.scene.traces
        src = self.source
        if src is None:
            return ""
        from ....viz.compute.filters import FilterSpec

        spec = FilterSpec(tr.hp, tr.lp, tr.notch)
        return spec.describe() if spec.active else ""


class ChannelPicker(QDialog):
    """Pick channels: a searchable list, filterable by type."""

    def __init__(self, src: SignalSource, picks: Optional[list[str]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select channels")
        self.resize(380, 540)
        self._src = src
        lay = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search channels")
        lay.addWidget(self.search)
        row = QHBoxLayout()
        row.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.setObjectName("ent-input")
        self.type_combo.addItem("all", "all")
        for t in src.available_types:
            self.type_combo.addItem(src.type_label(t), t)
        row.addWidget(self.type_combo, 1)
        lay.addLayout(row)
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        chosen = set(picks) if picks is not None else None
        for name, kind in zip(src.ch_names, src.ch_types):
            item = QListWidgetItem(f"{name}  [{src.type_label(kind)}]")
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setData(Qt.ItemDataRole.UserRole + 1, kind)
            self.list.addItem(item)
            item.setSelected(chosen is None or name in chosen)
        lay.addWidget(self.list, 1)
        quick = QHBoxLayout()
        all_btn = QPushButton("Select all")
        none_btn = QPushButton("Select none")
        all_btn.clicked.connect(self.list.selectAll)
        none_btn.clicked.connect(self.list.clearSelection)
        quick.addWidget(all_btn)
        quick.addWidget(none_btn)
        lay.addLayout(quick)
        self.search.textChanged.connect(self._filter)
        self.type_combo.currentIndexChanged.connect(lambda _i: self._filter())
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _filter(self) -> None:
        text = self.search.text().lower()
        kind = self.type_combo.currentData()
        for i in range(self.list.count()):
            item = self.list.item(i)
            name = item.data(Qt.ItemDataRole.UserRole)
            ch_kind = item.data(Qt.ItemDataRole.UserRole + 1)
            item.setHidden(bool(text and text not in name.lower())
                           or (kind != "all" and ch_kind != kind))

    def selected(self) -> Optional[list[str]]:
        names = [self.list.item(i).data(Qt.ItemDataRole.UserRole)
                 for i in range(self.list.count()) if self.list.item(i).isSelected()]
        return None if len(names) == len(self._src.ch_names) else names


__all__ = ["ChannelPicker", "SignalPresenter"]
