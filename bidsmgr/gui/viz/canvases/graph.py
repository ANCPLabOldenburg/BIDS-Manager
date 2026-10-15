"""The 4-D time-course graph: the voxel under the crosshair, across frames.

A ``scope`` x ``scope`` neighbourhood (1x1 up to 7x7) taken in the plane of
the current view. The x axis is real seconds when the frames have a known
time (``RepetitionTime``, ``VolumeTiming``, PET frame times, which are
uneven), the volume number otherwise; clicking a curve jumps to the frame
under the click.

ONE plot, whatever the neighbourhood. The old graph built one pyqtgraph plot,
axes and all, per voxel: a 7x7 neighbourhood was 49 plots and froze the
window for 640 ms (measured in a full MainWindow on a 1224-volume BOLD), and
dragging the crosshair with it open stalled 115 ms per move. Here every
neighbour is drawn by a SINGLE curve item, its series placed into its cell
and separated from the next by a gap; all the frame markers are one scatter
item; the cell grid is one line item. Four items, so building and updating
cost the same at 1x1 and at 7x7.

With one voxel the plot is an ordinary plot, with real axes in real units.
With a neighbourhood it is a grid of small multiples on a shared y range, so
the neighbours can be compared at a glance.

pyqtgraph reads no QSS: colours come from the viewer theme and are re-applied
on every theme change. Antialiasing is set per item, never process-wide.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from ....viz import views
from ....viz.scene import PLANE_AXIS
from ....viz.theme import parse_colour
from ...widgets.primitives import CappedLabel
from .. import fonts
from ..bridge import connect_while_alive
from ..context import ViewerContext
from ...widgets.splitter import Splitter

log = logging.getLogger(__name__)

_SCALINGS = (("raw", "Raw values"), ("percent", "% signal change"), ("demean", "Mean removed"))
_X_AXES = (("auto", "Automatic"), ("seconds", "Seconds"), ("frames", "Volume index"))
#: The neighbourhood choices: (scope, text). Scope n is a (2n-1)^2 square.
_SCOPES = ((1, "Voxel"), (2, "3 × 3"), (3, "5 × 5"), (4, "7 × 7"))
_Y_LABEL = {"raw": "Value", "percent": "Change (%)", "demean": "Value minus mean"}

#: Inner margin of a small-multiple cell, as a fraction of the cell.
_MARGIN = 0.07
#: Above this many points the curve is drawn 1 px wide. Qt strokes a wider
#: pen through its general path stroker: a 7x7 neighbourhood of 1224 volumes
#: (60,000 points) took 284 ms to paint at 1.5 px and 1.8 ms at 1 px, which
#: Qt draws with its fast cosmetic line code.
_THICK_LINE_MAX_POINTS = 4000
#: Width of the left axis of the graph and of the physio strip under it.
_LEFT_AXIS_PX = 56
#: Room above the graph's plot area for its top tick's number.
_TOP_PX = 9


def physio_for_graph(path, t0: float, t1: float, points: int = 2000) -> dict:
    """Worker side: the run's physio (every file of the run on one clock),
    cut to ``[t0, t1]`` run seconds, one TRACK per channel (see
    ``canvases.tracks``), in run seconds.

    A waveform keeps its own values (short gaps bridged), min/max-decimated
    to about ``points`` points, its vertical range the 0.5 to 99.5
    percentiles so one spike cannot flatten it. An event channel (a trigger,
    detected heartbeats: values only where something happened) is its event
    times. What the GUI thread receives is a few thousand numbers per
    channel, whatever the recording's length.
    """
    from ....viz.compute.decimate import peak_decimate
    from ....viz.data import physio as P
    from ....viz.data.signal import open_physio

    src = open_physio(path, None, together=True)
    seconds = src.start_time + np.arange(src.n_times, dtype=float) / src.sfreq
    inside = np.flatnonzero((seconds >= t0) & (seconds <= t1))
    if inside.size == 0:
        raise ValueError("the physio does not overlap the run")
    s0, s1 = int(inside[0]), int(inside[-1]) + 1
    indices = list(range(len(src.ch_names)))
    data = src.read(indices, s0, s1)
    mask = src.gap_mask(indices, s0, s1)
    if mask is None or mask.shape != data.shape:
        mask = ~np.isfinite(data)
    xs = seconds[s0:s1]
    bridge = max(2, int(round(P.BRIDGE_SECONDS * src.sfreq)))
    tracks = []
    for i, row in enumerate(data):
        name, ch_type = src.ch_names[i], src.ch_types[i]
        role = P.channel_role(row, mask[i], ch_type)
        track = {"id": f"physio:{name}", "title": name, "unit": "", "kind": "line",
                 "frames": False, "x": np.empty(0), "ys": [], "ticks": np.empty(0),
                 "colour": ch_type, "movable": False, "closable": False, "fmt": "{:.4g}",
                 "summary": "", "note": ""}
        if role == "events":
            times, _values = P.event_onsets(row, mask[i], xs)
            track.update(kind="events", ticks=np.asarray(times, dtype=float),
                         summary=P.describe_events(times), height=0.7)
            track["note"] = f"{name}: {track['summary']}"
        elif role == "waveform":
            filled, still, bridged = P.bridge_gaps(row, mask[i], bridge)
            finite = filled[np.isfinite(filled)]
            if finite.size:
                lo, hi = (float(v) for v in np.percentile(finite, (0.5, 99.5)))
                if hi > lo:
                    track["y_range"] = (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            xd, yd = peak_decimate(xs, filled, points // 2)
            track["x"] = np.asarray(xd, dtype=float)
            track["ys"] = [np.asarray(yd, dtype=float)]
            gaps = int(np.count_nonzero(np.diff(np.r_[0, still.astype(np.int8)]) == 1))
            notes = []
            if bridged:
                notes.append(f"{bridged:,} dropped samples bridged")
            if gaps:
                notes.append(f"{gaps:,} gaps")
            track["summary"] = ", ".join(notes) or ch_type
            track["note"] = (f"{name} ({ch_type}), the run's physiology on the time "
                             "course's clock" + (f": {track['summary']}" if notes else ""))
        else:
            track.update(kind="events", summary="no samples in this run", height=0.7)
        tracks.append(track)
    return {"names": [t["title"] for t in tracks], "tracks": tracks}


def _qc_track(row_id: str, title: str, ys: list, *, unit: str = "", **kw) -> dict:
    """A QC track on the volume axis (``frames``: x is the volume index)."""
    n = len(ys[0]) if ys else 0
    track = {"id": row_id, "title": title, "unit": unit, "kind": "line", "frames": True,
             "x": np.arange(n, dtype=float), "ys": [np.asarray(y, dtype=float) for y in ys],
             "ticks": np.empty(0), "movable": True, "closable": True, "summary": "",
             "note": "", "fmt": "{:.3g}"}
    track.update(kw)
    return track


def is_diffusion(src) -> bool:
    """A diffusion series: its b-values describe its volumes."""
    bvals = getattr(src, "bvals", None)
    return bvals is not None and len(bvals) == int(getattr(src, "loaded_frames", 0))


def dwi_qc_for_graph(src, rows=None, path=None, root=None, *, config=None, cancel=None,
                     progress=None) -> dict:
    """Worker side, for a diffusion series: the rows asked for, from the
    diffusion check (``bidsmgr.qc.dwi``), computed once and shared with the
    viewer's Check quality (``bidsmgr.qc.live``)."""
    from ....qc import live
    from ....viz import bids as VB
    from ....viz.compute import qc

    sidecar = VB.inherited_sidecar(Path(path), root) if path is not None else {}
    res = live.result_for(src, sidecar, config=config, cancel=cancel, progress=progress)
    help_of = {r: h for r, _t, h in qc.DWI_QC_ROWS}
    out: dict[str, dict] = {}
    for row_id in rows if rows is not None else qc.DWI_ROW_IDS:
        t = res.tracks.get(row_id)
        if t is None:
            continue
        if t.get("kind") == "image":
            image = np.asarray(t["image"], dtype=np.float32)
            n = image.shape[1]
            out[row_id] = {
                "id": row_id, "title": t["title"], "unit": "", "kind": "image",
                "frames": True, "image": image, "image_x": (-0.5, n - 0.5),
                "levels": tuple(t.get("levels", (-8.0, 8.0))),
                "value_name": t.get("value_name", ""), "colormap": t.get("colormap", ""),
                "height": t.get("height", 2.5), "rows": list(t.get("rows", [])),
                "movable": True, "closable": True, "fmt": "{:.1f}",
                "summary": t.get("summary", ""), "note": help_of.get(row_id, t.get("help", ""))}
            continue
        ys = [np.asarray(y, dtype=float) for y in t["ys"]]
        kw = {k: t[k] for k in ("legend", "colour", "rule") if k in t}
        if "ticks" in t:
            kw["ticks"] = np.asarray(t["ticks"], dtype=float)
        x = None
        if t.get("points"):
            # Sparse (the b=0 volumes only): the finite points, joined, or a
            # line broken at every gap would draw nothing at all.
            keep = np.isfinite(ys[0])
            x = np.flatnonzero(keep).astype(float)
            ys = [y[keep] for y in ys]
        track = _qc_track(row_id, t["title"], ys, unit=t.get("unit", ""),
                          summary=t.get("summary", ""),
                          note=help_of.get(row_id, t.get("help", "")), **kw)
        if x is not None:
            track["x"] = x
        out[row_id] = track
    for row_id, track in out.items():
        track["explain"] = f"plot.dwi.{row_id}"
    return {"rows": out, "skip": 0}


def qc_for_graph(src, rows=None, path=None, root=None, slice_axis: int = 2, *,
                 config=None, cancel=None, progress=None) -> dict:
    """Worker side: the QC ``rows`` asked for (ids of ``qc.QC_ROWS``), one
    track each, on the volume axis: ``{"rows": {id: track}, "skip": n}``.
    Each row is computed only when asked for; motion, the expensive one, is
    read from fMRIPrep's confounds when the run has them (and ``config``
    allows it) and estimated once for the three rows that show it. Every
    threshold is ``config``'s (a ``QcConfig``; Settings > Quality control)."""
    from ....qc.config import DEFAULT
    from ....viz.compute import motion as M
    from ....viz.compute import qc

    if is_diffusion(src):
        return dwi_qc_for_graph(src, rows, path, root, config=config, cancel=cancel,
                                progress=progress)
    cfg = config or DEFAULT
    bold = cfg.bold
    rows = set(rows if rows is not None else qc.BOLD_ROW_IDS)
    skip = (qc.non_steady_state(src, z=bold.nonsteady_z, cancel=cancel)
            if bold.skip_nonsteady else 0)
    n = src.loaded_frames
    out: dict[str, dict] = {}
    help_of = {r: h for r, _t, h in qc.QC_ROWS}
    if rows & {"dvars", "global"}:
        pv = qc.per_volume(src, skip=skip, fence_iqr=bold.dvars_fence_iqr, cancel=cancel)
        if "global" in rows:
            g = pv["global"]
            out["global"] = _qc_track(
                "global", "Global signal", [g], unit="a.u.", colour="dim",
                summary=f"median {np.nanmedian(g[skip:]):.4g}", note=help_of["global"])
        if "dvars" in rows:
            d = pv["dvars"]
            fence = pv["fence"]
            out["dvars"] = _qc_track(
                "dvars", "DVARS", [d], unit="% of mean", colour="purple",
                ticks=pv["flagged"].astype(float),
                rule=float(fence) if np.isfinite(fence) else None,
                summary=(f"median {np.nanmedian(d):.2f} %, fence {fence:.2f} %, "
                         f"{pv['flagged'].size} above it"),
                note=help_of["dvars"] + " The dashed line is the upper box-plot fence "
                                        f"(75th percentile + {bold.dvars_fence_iqr:g} IQR).")
    if rows & {"fd", "translation", "rotation"}:
        m = M.motion_for(src, path, root, skip=skip, config=cfg, cancel=cancel,
                         progress=progress)
        where = m.describe()
        if "fd" in rows:
            fd = m.fd
            limit = bold.fd_threshold_mm
            over = np.flatnonzero(np.nan_to_num(fd) > limit).astype(float)
            peak = float(np.nanmax(fd)) if np.isfinite(fd).any() else 0.0
            out["fd"] = _qc_track(
                "fd", "Framewise displacement", [fd], unit="mm", colour="accent",
                ticks=over, rule=limit,
                y_range=(0.0, max(peak, limit * 1.2) * 1.05),
                summary=(f"mean {np.nanmean(fd):.3f}, max {peak:.2f} mm, {over.size} above "
                         f"{limit:g} mm"),
                note=f"{help_of['fd']} Here: {where}. Dashed: {limit:g} mm.")
        if "translation" in rows:
            t = m.params[:, :3]
            out["translation"] = _qc_track(
                "translation", "Translation", [t[:, 0], t[:, 1], t[:, 2]], unit="mm",
                legend=["x", "y", "z"],
                summary=f"{float(t.min()):.2f} to {float(t.max()):.2f} mm",
                note=f"{help_of['translation']} Here: {where}.")
        if "rotation" in rows:
            r = np.degrees(m.params[:, 3:])
            out["rotation"] = _qc_track(
                "rotation", "Rotation", [r[:, 0], r[:, 1], r[:, 2]], unit="degrees",
                legend=["pitch", "roll", "yaw"],
                summary=f"{float(r.min()):.2f} to {float(r.max()):.2f} degrees",
                note=f"{help_of['rotation']} Here: {where}.")
    if rows & {"outliers", "spikes", "carpet"}:
        sample = qc.sample_series(src, skip=skip, voxels=bold.sample_voxels,
                                  slice_axis=slice_axis, cancel=cancel)
        if "outliers" in rows:
            frac = qc.outlier_fraction(sample["series"], skip=skip) * 100.0
            limit = bold.outlier_limit_pct
            over = np.flatnonzero(np.nan_to_num(frac) > limit).astype(float)
            peak = float(np.nanmax(frac)) if np.isfinite(frac).any() else 0.0
            out["outliers"] = _qc_track(
                "outliers", "Outlier voxels", [frac], unit="%", colour="teal",
                ticks=over, rule=limit,
                y_range=(0.0, max(peak, limit * 1.2) * 1.05),
                summary=f"max {peak:.1f} %, {over.size} above {limit:g} %",
                note=(f"{help_of['outliers']} {sample['series'].shape[1]:,} sampled voxels. "
                      f"Dashed: {limit:g} % (afni_proc.py's censoring default is 5 %)."))
        if "spikes" in rows:
            sp = qc.slice_spikes(sample["slice_means"], skip=skip, z=bold.spike_z)
            where = ", ".join(f"volume {int(v)} (slice {int(z)})"
                              for v, z in zip(sp["flagged"][:4], sp["slice"][:4]))
            out["spikes"] = _qc_track(
                "spikes", "Slice spikes", [sp["score"]], unit="z", colour="warning",
                ticks=sp["flagged"].astype(float), rule=bold.spike_z,
                summary=(f"{sp['flagged'].size} volumes above {bold.spike_z:g}"
                         + (f": {where}" if where else "")),
                note=f"{help_of['spikes']} Dashed: {bold.spike_z:g} robust SD.")
        if "carpet" in rows:
            image = qc.carpet(sample["series"], sample["depth"], skip=skip)
            out["carpet"] = {
                "id": "carpet", "title": "Carpet plot", "unit": "", "kind": "image",
                "frames": True, "image": image, "image_x": (-0.5, n - 0.5),
                "levels": (-2.0, 2.0), "value_name": "z", "height": 2.5,
                "rows": [f"voxel group {k + 1} of {image.shape[0]} (edge to centre)"
                         for k in range(image.shape[0])],
                "movable": True, "closable": True, "fmt": "{:.2f}",
                "summary": "the head's edge at the top, its centre at the bottom",
                "note": help_of["carpet"]}
    for row_id, track in out.items():
        track["explain"] = f"plot.bold.{row_id}"
    return {"rows": out, "skip": skip}


def go_to_slice(ctx, k: int, bids_ctx=None) -> None:
    """Move the crosshair to slice ``k`` along the series' slice axis (the
    sidecar's SliceEncodingDirection, else k), keeping the rest of its
    position."""
    axis = 2
    if bids_ctx is not None:
        axis = {"i": 0, "j": 1, "k": 2}.get(
            str(bids_ctx.sidecar.get("SliceEncodingDirection", "k"))[:1], 2)
    voxel = views.cursor_voxel(ctx.store)
    if voxel is None:
        return
    v = list(voxel)
    v[axis] = int(k)
    ctx.run("cursor.set_voxel", i=v[0], j=v[1], k=v[2])


class TimecourseGraph(QWidget):
    """The graph panel (controls row + one plot)."""

    #: The panel needs this many pixels to show what it was asked to (the
    #: physio lanes): the host may grow it.
    wants_height = pyqtSignal(int)

    def __init__(self, ctx: ViewerContext, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setObjectName("pane-dark")
        self._context = None   # BidsContext of the file (frame times)
        # What is drawn, in DATA units, for queries and click mapping.
        self._dim = 0
        self._x: Optional[np.ndarray] = None
        self._x_is_time = False
        self._series: list[list[Optional[np.ndarray]]] = []
        self._voxels: list[list[Optional[tuple[int, int, int]]]] = []
        self._centres: dict[tuple[int, int], float] = {}
        self._ylim = (0.0, 1.0)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        import pyqtgraph as pg

        self._pg = pg
        lay.addWidget(self._build_controls())

        self.plot = pg.PlotWidget()
        pi = self.plot.getPlotItem()
        pi.setMenuEnabled(False)
        pi.hideButtons()
        self._vb = pi.getViewBox()
        self._vb.setMouseEnabled(x=False, y=False)
        self._vb.disableAutoRange()
        # Scrolling over the graph must not zoom it into nothing.
        # The plain wheel is the viewer's (it steps slices). Ctrl+wheel zooms
        # TIME, Shift+wheel pans it, double-click shows the whole run again.
        self.plot.wheelEvent = self._on_wheel
        self.plot.scene().sigMouseClicked.connect(self._on_click)
        #: The time window shown, in the graph's x units (None: the run).
        self._time_view: Optional[tuple[float, float]] = None
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(150)
        self._zoom_timer.timeout.connect(self._draw_tracks)
        # Seconds stay seconds: pyqtgraph would relabel a 1040 s run as
        # "1.04 ks".
        for name in ("left", "bottom"):
            pi.getAxis(name).enableAutoSIPrefix(False)
        self._grid_item = pg.PlotCurveItem(connect="pairs", antialias=False)
        self._center_item = pg.PlotCurveItem(connect="pairs", antialias=False)
        self._curve = pg.PlotCurveItem(antialias=True)
        self._markers = pg.ScatterPlotItem()
        # Every event of the run is ONE bar item (one band per event), under
        # the curve. Created once and hidden when empty, never removed.
        self._events_item = pg.BarGraphItem(x0=[0.0], x1=[0.0], y0=[0.0], y1=[0.0])
        self._events_item.setZValue(-5)
        self._events_item.setVisible(False)
        for item in (self._events_item, self._grid_item, self._center_item, self._curve,
                     self._markers):
            pi.addItem(item)
        # Both plots keep a left axis of ONE width, so their time axes line
        # up pixel for pixel (labels made the graph's wider than the strip's).
        pi.getAxis("left").setWidth(_LEFT_AXIS_PX)
        # Headroom, or the top tick's number is cut in half at the edge.
        pi.layout.setContentsMargins(1, _TOP_PX, 1, 1)
        self.plot.setMinimumHeight(70)
        # Graph above, physio below, the divider the user's to drag.
        self.split = Splitter(Qt.Orientation.Vertical)
        self.split.setChildrenCollapsible(False)
        self.split.setHandleWidth(6)
        self.split.addWidget(self.plot)
        lay.addWidget(self.split, 1)

        # Under the graph: one plot per QC row and per physio channel, each
        # its own (``canvases.tracks``), on the graph's time axis.
        from .tracks import TracksPanel

        self.tracks = TracksPanel(ctx, left_axis_px=_LEFT_AXIS_PX)
        self.tracks.setVisible(False)
        self.tracks.describe_x = self._describe_x
        self.tracks.clicked.connect(self._go_to_x)
        # A carpet's rows are voxels: a click on one goes to its moment. The
        # diffusion slice image's rows are slices: to that slice as well.
        self.tracks.row_clicked.connect(self._go_to_cell)
        self.tracks.reset_view.connect(lambda: self.set_time_view(None))
        self.tracks.wheel.connect(self._on_wheel)
        self.tracks.order_changed.connect(self._on_tracks_order)
        self.tracks.align_source = self._plot_area
        self.tracks.follow(self.plot)
        self.tracks.hide_requested.connect(lambda rid: self._toggle_qc_plot(rid, False))
        self._physio_src = None
        self._physio_key = None
        self._physio_generation = 0
        self._tracks_sized = False
        self._sized_height = 0
        # The series' per-volume quality, computed once per series.
        self._qc_src = None
        self._qc_key = None
        self._qc_generation = 0
        #: QC rows asked of a worker for the current series (computed or not).
        self._qc_asked: set[str] = set()
        #: QC rows being computed now: shown as cards with a spinner.
        self._qc_pending: set[str] = set()
        self._qc_progress = ""
        self.split.addWidget(self.tracks)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 2)
        self._sizes_timer = QTimer(self)
        self._sizes_timer.setSingleShot(True)
        self._sizes_timer.setInterval(400)
        self._sizes_timer.timeout.connect(self._remember_split)
        self.split.splitterMoved.connect(lambda *_a: self._sizes_timer.start())
        ctx.jobs.done.connect(self._on_job_done)
        ctx.jobs.failed.connect(self._on_job_failed)
        ctx.jobs.progressed.connect(self._on_job_progress)

        self._sync_controls()
        ctx.qstore.changed.connect(self._on_changed)
        connect_while_alive(ctx.theme_hub.changed, self, lambda w, _t: w.apply_theme())
        connect_while_alive(ctx.settings_hub.changed, self, lambda w, _s: w._on_settings())
        self.apply_theme()

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sidecar-footer-summary")
        return lbl

    def _group(self, label: str, widget: QWidget, tip: str) -> QWidget:
        """A label and its control, kept together when the bar wraps."""
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        lbl = self._label(label)
        lbl.setToolTip(tip)
        widget.setToolTip(tip)
        h.addWidget(lbl)
        h.addWidget(widget)
        return box

    def _build_controls(self) -> QWidget:
        """The graph's header (``panels.panel_header``): on the left, by
        purpose, WHAT is plotted (the series, the neighbourhood), HOW (values,
        time axis), and WHAT ELSE on the same axis (events, physiology, QC
        and its rows); in the upper-right corner, the panel as a panel
        (expand its plots, beside, maximise, own window) and More (the
        marker, export). The left side wraps: a plain row of these was once
        a 1139 px floor under the whole viewer."""
        from ..menus import popup_menu, submenu
        from ..panels.panel_header import PanelHeader, corner_button

        ctx = self.ctx
        header = PanelHeader()
        bar = header.controls
        # Which series is plotted: a name, or a choice when there are several
        # (the anatomy's own series and an overlaid BOLD, say).
        self.series_label = CappedLabel(260)
        self.series_label.setObjectName("sidecar-footer-summary")
        bar.addWidget(self.series_label)
        self.series_combo = QComboBox()
        self.series_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.series_combo.setMinimumContentsLength(10)
        self.series_combo.setToolTip("The 4-D image whose time course is plotted; the "
                                     "volume slider and Play follow it too.")
        self.series_combo.currentIndexChanged.connect(self._on_series_choice)
        self.series_combo.setVisible(False)
        bar.addWidget(self.series_combo)
        self.scope_combo = QComboBox()
        for value, text in _SCOPES:
            self.scope_combo.addItem(text, value)
        self.scope_combo.currentIndexChanged.connect(
            lambda i: ctx.run("graph.set", scope=int(self.scope_combo.itemData(i))))
        bar.addWidget(self._group(
            "Neighbourhood", self.scope_combo,
            "Which voxels are plotted: the crosshair voxel alone, or a square "
            "around it in the plane of the active view, one small plot per voxel."))
        self.scaling_combo = QComboBox()
        for value, text in _SCALINGS:
            self.scaling_combo.addItem(text, value)
        self.scaling_combo.currentIndexChanged.connect(
            lambda i: ctx.run("graph.set", scaling=self.scaling_combo.itemData(i)))
        bar.addWidget(self._group(
            "Values", self.scaling_combo,
            "Raw values as stored; percent signal change from each voxel's own "
            "mean (the usual way to read BOLD); or the mean removed."))
        self.x_combo = QComboBox()
        for value, text in _X_AXES:
            self.x_combo.addItem(text, value)
        self.x_combo.currentIndexChanged.connect(
            lambda i: ctx.run("graph.set", x_axis=self.x_combo.itemData(i)))
        bar.addWidget(self._group(
            "Time axis", self.x_combo,
            "Automatic plots seconds whenever the sidecar times the volumes "
            "(RepetitionTime, VolumeTiming, PET FrameTimesStart), else the "
            "volume index."))
        bar.addSpacing(6)
        self.events_box = QCheckBox("Events")
        self.events_box.setToolTip(
            "The run's _events.tsv on the same time axis: one band per event, "
            "coloured by trial type.")
        self.events_box.toggled.connect(lambda v: ctx.run("graph.set", events=bool(v)))
        bar.addWidget(self.events_box)
        self.physio_box = QCheckBox("Physiology")
        self.physio_box.setToolTip(
            "The run's physiological recordings (_physio.tsv.gz: cardiac, "
            "respiration, triggers) under the graph, on the same time axis, "
            "each placed by its own StartTime.")
        self.physio_box.toggled.connect(lambda v: ctx.run("graph.set", physio=bool(v)))
        bar.addWidget(self.physio_box)
        self.qc_box = QCheckBox("QC")
        self.qc_box.setToolTip(
            "Quality control, per volume, in plots under the graph: framewise "
            "displacement, translation and rotation, DVARS, outlier voxels, slice "
            "spikes, the global signal and a carpet plot. Computed when switched "
            "on (motion is read from fMRIPrep's confounds when the run has them); "
            "choose them with Plots.")
        self.qc_box.toggled.connect(lambda v: ctx.run("graph.set", qc=bool(v)))
        bar.addWidget(self.qc_box)
        from ... import icons

        self.qc_plots_button = QPushButton("Plots")
        self.qc_plots_button.setObjectName("tb-btn")
        self.qc_plots_button.setIcon(icons.icon("plots"))
        self.qc_plots_button.setProperty("viz_icon", "plots")   # re-coloured by the header
        self.qc_plots_button.setToolTip(
            "Which QC plots are drawn under the time course; each can also be moved and "
            "hidden from its own header")
        rows_menu = popup_menu(self.qc_plots_button)
        rows_menu.setToolTipsVisible(True)
        self._qc_plot_actions = {}
        from ....viz.compute.qc import DWI_QC_ROWS, QC_ROWS

        # Every row of both kinds, each shown for the kind of series on
        # screen (``_sync_controls``); translation and rotation are shared.
        seen = set()
        for row_id, title, help_text in tuple(QC_ROWS) + tuple(DWI_QC_ROWS):
            if row_id in seen:
                continue
            seen.add(row_id)
            act = rows_menu.addAction(title)
            act.setCheckable(True)
            act.setToolTip(help_text)
            act.toggled.connect(lambda on, r=row_id: self._toggle_qc_plot(r, on))
            self._qc_plot_actions[row_id] = act
        rows_menu.addSeparator()
        self._qc_on_open = rows_menu.addAction("Run QC when a file opens")
        self._qc_on_open.setCheckable(True)
        self._qc_on_open.setToolTip(
            "On: QC stays on from one file to the next and is computed as soon as a "
            "file opens. Off: every file opens with QC off.")
        self._qc_on_open.toggled.connect(self._set_qc_on_open)
        rows_menu.aboutToShow.connect(
            lambda: self._qc_on_open.setChecked(self.ctx.settings.qc.on_open))
        self.qc_plots_button.setMenu(rows_menu)
        bar.addWidget(self.qc_plots_button)

        # The corner: the panel as a panel.
        self.expand_button = header.add_corner(corner_button(
            "tracks_scroll", "Expand plots: each plot under the graph at a readable "
            "height, in a scrolling column (off: they share the room)", checkable=True))
        self.expand_button.toggled.connect(
            lambda on: ctx.run("graph.set", tracks_mode="scroll" if on else "fit"))
        self.more_button, more = header.more_menu("More: the marker, export")
        marker = submenu(more, "Marker size")
        from PyQt6.QtGui import QActionGroup

        group = QActionGroup(marker)
        group.setExclusive(True)
        self._marker_actions = {}
        for size in (4, 6, 8, 10, 12, 16):
            act = marker.addAction(f"{size} px")
            act.setCheckable(True)
            group.addAction(act)
            act.triggered.connect(lambda _c=False, v=size: ctx.run("graph.set", dot=v))
            self._marker_actions[size] = act
        self.marks_action = more.addAction("Marker on every voxel")
        self.marks_action.setCheckable(True)
        self.marks_action.setToolTip("The current-volume marker on every voxel of the "
                                     "neighbourhood; off, on the centre voxel only.")
        self.marks_action.toggled.connect(
            lambda v: ctx.run("graph.set", mark_neighbors=bool(v)))
        more.addSeparator()
        self.export_action = more.addAction("Export CSV...")
        self.export_action.setToolTip(
            "Save the plotted time courses as a CSV table: one row per volume, the "
            "time (or volume index) and one column per voxel, in the values shown.")
        self.export_action.triggered.connect(self._ask_export)
        self.header = header
        self.controls = bar
        return header

    def add_panel_buttons(self, buttons) -> None:
        """The buttons that act on the graph's PANEL (beside the views,
        maximised, in its own window): icons in the header's corner, before
        More."""
        self.header.insert_corner(buttons, before=self.more_button)

    def _set_qc_on_open(self, on: bool) -> None:
        if self.ctx.settings.qc.on_open != bool(on):
            self.ctx.settings_hub.update(lambda st: setattr(st.qc, "on_open", bool(on)))

    def _series_is_dwi(self) -> bool:
        _layer, src = views.series_layer(self.ctx.store)
        return bool(src is not None and is_diffusion(src))

    def _qc_rows_shown(self) -> list[str]:
        """The QC rows for the series on screen, in the user's order: a
        diffusion series shows its own rows (its defaults when none of them
        was ever chosen), a BOLD its own."""
        from ....viz.compute import qc

        rows = list(self.ctx.scene.graph.qc_rows)
        if self._series_is_dwi():
            mine = [r for r in rows if r in qc.DWI_ROW_IDS]
            own = [r for r in mine if r not in qc.BOLD_ROW_IDS]
            return mine if own else list(dict.fromkeys(list(qc.DWI_DEFAULT_ROWS) + mine))
        return [r for r in rows if r in qc.BOLD_ROW_IDS]

    def _toggle_qc_plot(self, row_id: str, on: bool) -> None:
        shown = self._qc_rows_shown()
        if on and row_id not in shown:
            shown.append(row_id)
        elif not on and row_id in shown:
            shown.remove(row_id)
        else:
            return
        # The other kind's rows keep their place.
        rows = [r for r in self.ctx.scene.graph.qc_rows if r not in shown]
        rows = [r for r in rows if r != row_id] + shown
        self.ctx.run("graph.set", qc_rows=rows)

    def _on_series_choice(self, _i: int) -> None:
        if self._series_syncing:
            return
        layer_id = self.series_combo.currentData()
        if layer_id:
            self.ctx.run("graph.set", layer=layer_id)

    _series_syncing = False

    def _sync_series(self, current) -> None:
        """Name the series plotted; offer a choice when there are several."""
        layers = views.series_layers(self.ctx.store)
        self._series_syncing = True
        try:
            if len(layers) > 1:
                ids = [lay.id for lay in layers]
                if [self.series_combo.itemData(i) for i in range(self.series_combo.count())] != ids:
                    self.series_combo.clear()
                    for lay in layers:
                        self.series_combo.addItem(lay.name or lay.id, lay.id)
                i = self.series_combo.findData(current.id)
                if i >= 0 and self.series_combo.currentIndex() != i:
                    self.series_combo.setCurrentIndex(i)
                self.series_combo.setVisible(True)
                self.series_label.setText("Series")
            else:
                self.series_combo.setVisible(False)
                name = current.name or current.id
                self.series_label.setText(name)
                self.series_label.setToolTip(name)
        finally:
            self._series_syncing = False

    def _sync_controls(self) -> None:
        g = self.ctx.scene.graph
        act = self._marker_actions.get(int(g.dot))
        for size, a in self._marker_actions.items():
            if a.isChecked() != (a is act):
                a.blockSignals(True)
                a.setChecked(a is act)
                a.blockSignals(False)
        i = self.scope_combo.findData(int(g.scope))
        if i >= 0 and self.scope_combo.currentIndex() != i:
            self.scope_combo.blockSignals(True)
            self.scope_combo.setCurrentIndex(i)
            self.scope_combo.blockSignals(False)
        layer, src = views.series_layer(self.ctx.store)
        qc_ok = bool(src is not None and src.fully_loaded and src.loaded_frames >= 3)
        if self.qc_box.isHidden() == qc_ok:
            self.qc_box.setVisible(qc_ok)
            self.qc_plots_button.setVisible(qc_ok)
        scroll = g.tracks_mode == "scroll"
        if self.expand_button.isChecked() != scroll:
            self.expand_button.blockSignals(True)
            self.expand_button.setChecked(scroll)
            self.expand_button.blockSignals(False)
        physio_ok = bool(self._context is not None and self._context.physio_paths)
        tracks_on = bool((g.qc and qc_ok) or (g.physio and physio_ok))
        if self.expand_button.isHidden() == tracks_on:
            self.expand_button.setVisible(tracks_on)
        if self.qc_box.isChecked() != g.qc:
            self.qc_box.blockSignals(True)
            self.qc_box.setChecked(g.qc)
            self.qc_box.blockSignals(False)
        self.qc_plots_button.setEnabled(g.qc)
        from ....viz.compute import qc as Q

        dwi = self._series_is_dwi()
        kind_rows = Q.DWI_ROW_IDS if dwi else Q.BOLD_ROW_IDS
        shown = self._qc_rows_shown()
        for row_id, act in self._qc_plot_actions.items():
            if act.isVisible() != (row_id in kind_rows):
                act.setVisible(row_id in kind_rows)
            if act.isChecked() != (row_id in shown):
                act.blockSignals(True)
                act.setChecked(row_id in shown)
                act.blockSignals(False)
        if self.marks_action.isChecked() != g.mark_neighbors:
            self.marks_action.blockSignals(True)
            self.marks_action.setChecked(g.mark_neighbors)
            self.marks_action.blockSignals(False)
        for combo, value in ((self.scaling_combo, g.scaling), (self.x_combo, g.x_axis)):
            i = combo.findData(value)
            if i >= 0 and combo.currentIndex() != i:
                combo.blockSignals(True)
                combo.setCurrentIndex(i)
                combo.blockSignals(False)
        ctx = self._context
        for box, on, available in (
            (self.events_box, g.events, bool(ctx is not None and ctx.events)),
            (self.physio_box, g.physio, bool(ctx is not None and ctx.physio_paths)),
        ):
            # Offered only where the run HAS events or physio.
            box.setVisible(available)
            if box.isChecked() != on:
                box.blockSignals(True)
                box.setChecked(on)
                box.blockSignals(False)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def set_context(self, bids_ctx) -> None:
        self._context = bids_ctx
        self._time_view = None
        self._physio_src = None
        self._physio_key = None
        self._physio_generation += 1
        self.ctx.jobs.cancel("graph-physio")
        self._sync_controls()
        self.refresh()

    def _on_changed(self, paths) -> None:
        if not self.isVisible():
            return
        if any(p in ("graph", "scene") for p in paths):
            self._sync_controls()
            self.refresh()
        elif any(p in ("cursor", "plane", "mode", "layers") or p.startswith("sources")
                 for p in paths):
            if any(p.startswith("sources") or p == "layers" for p in paths):
                # The series finished reading: the QC traces become possible.
                self._sync_controls()
            self.refresh()
        elif any(p.endswith(".frame") for p in paths):
            self.update_markers()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._sync_controls()
        self.refresh()

    def apply_theme(self) -> None:
        theme = self.ctx.theme
        pg = self._pg
        self.plot.setBackground(theme.plot_background)
        self._set_curve_pen()
        self._grid_item.setPen(pg.mkPen(theme.grid, width=1))
        cross = self.ctx.settings.crosshair.color
        self._center_item.setPen(pg.mkPen(cross, width=1.5))
        self._markers.setBrush(pg.mkBrush(cross))
        self._markers.setPen(pg.mkPen(cross))
        pi = self.plot.getPlotItem()
        for name in ("left", "bottom"):
            pi.getAxis(name).setPen(theme.plot_foreground)
        # The numbers and the title at the app's font size, and an axis wide
        # enough for them at that size.
        fonts.style_axes(pi, theme.plot_foreground)
        pi.getAxis("left").setWidth(fonts.axis_width(_LEFT_AXIS_PX))
        if self.tracks is not None:
            self.tracks.left_axis_px = fonts.axis_width(_LEFT_AXIS_PX)
            self.tracks.request_align()

    def _set_curve_pen(self, n_points: int = 0) -> None:
        width = 1.5 if n_points <= _THICK_LINE_MAX_POINTS else 1.0
        self._curve.setPen(self._pg.mkPen(self.ctx.theme.plot_foreground, width=width))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _x_axis(self, n: int) -> tuple[np.ndarray, bool]:
        mode = self.ctx.scene.graph.x_axis
        times = None
        if mode != "frames" and self._context is not None:
            times = self._context.frame_times(n)
        if times is not None:
            return np.asarray(times, dtype=float), True
        return np.arange(n, dtype=float), False

    def _neighbours(self, center, src) -> list[list[Optional[tuple[int, int, int]]]]:
        """The voxels of the grid, in the plane of the active view."""
        g = self.ctx.scene.graph
        dim = 2 * (g.scope - 1) + 1
        half = dim // 2
        plane = self.ctx.scene.plane if self.ctx.scene.mode == "single" else self.ctx.active_plane
        ornt = views.orientation(src)
        normal_axis = ornt.data_of[PLANE_AXIS[plane]]
        a, b = [ax for ax in range(3) if ax != normal_axis]
        out = []
        for di in range(-half, half + 1):
            row = []
            for dj in range(-half, half + 1):
                v = list(center)
                v[a] += di
                v[b] += dj
                if all(0 <= v[k] < src.spatial[k] for k in range(3)):
                    row.append((v[0], v[1], v[2]))
                else:
                    row.append(None)
            out.append(row)
        return out

    @staticmethod
    def _scaled(ts: np.ndarray, scaling: str) -> np.ndarray:
        ts = np.asarray(ts, dtype=float)
        if scaling == "percent":
            mean = float(np.nanmean(ts)) if ts.size else 0.0
            return (ts - mean) / mean * 100.0 if mean else ts * 0.0
        if scaling == "demean":
            return ts - float(np.nanmean(ts))
        return ts

    def refresh(self) -> None:
        if not self.isVisible():
            return
        store = self.ctx.store
        layer, src = views.series_layer(store)
        if layer is None or src is None:
            self._clear()
            return
        world = views.cursor_world(store)
        if world is None:
            return
        # The series' OWN voxel under the crosshair: an overlaid BOLD has
        # its own grid, 2-3 mm, wherever the anatomy beneath is.
        center = views.voxel_in(src, world)
        self._sync_series(layer)
        if center is None:
            self._clear()
            self.series_label.setText(f"{layer.name or layer.id}: the crosshair is outside it")
            return
        grid = self._neighbours(center, src)
        scaling = self.ctx.scene.graph.scaling
        # Times are matched against the FILE's frame count (the sidecar
        # describes every frame), then cut to what was read.
        x, is_time = self._x_axis(src.n_frames_total)
        n = src.loaded_frames
        x = x[:n]
        series = [[None if v is None else self._scaled(src.voxel_series(v)[:n], scaling)
                   for v in row] for row in grid]
        finite = [s for row in series for s in row if s is not None and s.size]
        if not finite or not x.size:
            self._clear()
            return
        centres: dict[tuple[int, int], float] = {}
        if len(grid) > 1:
            # Small multiples: each cell keeps its OWN baseline (its median)
            # and all share one vertical SCALE, as AFNI's graph window does.
            # One shared range drew every curve flat whenever neighbours'
            # means differed by more than their fluctuation (always, in
            # BOLD); one range per cell would make a 1 % wobble look as big
            # as a 10 % one. The scale is robust, or the bright
            # non-steady-state volumes at the start of a run would set it.
            half = 0.0
            for r, row in enumerate(series):
                for c, ts in enumerate(row):
                    if ts is None or not ts.size or not np.isfinite(ts).any():
                        continue
                    m = float(np.nanmedian(ts))
                    centres[(r, c)] = m
                    half = max(half, float(np.nanpercentile(np.abs(ts - m), 99.0)))
            half = half or 1.0
            lo, hi = -half, half
        else:
            lo = float(min(np.nanmin(s) for s in finite))
            hi = float(max(np.nanmax(s) for s in finite))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            self._clear()
            return
        if lo == hi:
            pad = 1.0 if lo == 0 else abs(lo) * 0.05
            lo, hi = lo - pad, hi + pad
        self._dim = len(grid)
        self._x = x
        self._x_is_time = is_time
        self._voxels = grid
        self._series = series
        self._centres = centres
        self._ylim = (lo, hi)
        self._draw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _cell_xy(self, r: int, c: int, x: np.ndarray, y: np.ndarray):
        """Data -> plot coordinates of cell (r, c) of the small multiples:
        the cell's own baseline subtracted, the shared scale applied, and
        anything beyond the scale pinned to the cell's edge (a spike stays
        in its own cell)."""
        dim = self._dim
        x0, x1 = float(self._x[0]), float(self._x[-1])
        lo, hi = self._ylim
        rel = np.clip(np.asarray(y, dtype=float) - self._centres.get((r, c), 0.0), lo, hi)
        sx = (1.0 - 2 * _MARGIN) / max(x1 - x0, 1e-12)
        sy = (1.0 - 2 * _MARGIN) / max(hi - lo, 1e-12)
        px = c + _MARGIN + (np.asarray(x, dtype=float) - x0) * sx
        py = (dim - 1 - r) + _MARGIN + (rel - lo) * sy
        return px, py

    def _draw(self) -> None:
        dim = self._dim
        pi = self.plot.getPlotItem()
        x = self._x
        if dim == 1:
            # One voxel: an ordinary plot with real axes in real units.
            ts = self._series[0][0]
            self._set_curve_pen(ts.size)
            self._curve.setData(x[: ts.size], ts, connect="finite")
            self._grid_item.setData([], [])
            self._center_item.setData([], [])
            pi.showAxis("left")
            pi.showAxis("bottom")
            fonts.axis_title(pi, "bottom", "Time" if self._x_is_time else "Volume",
                             self.ctx.theme.plot_foreground,
                             units="s" if self._x_is_time else None)
            # No rotated title on the value axis: in the axis width the tracks
            # share it overlapped the numbers. The Values menu says what the
            # axis holds, and the tooltip repeats it.
            pi.setLabel("left", None)
            pi.getAxis("left").setToolTip(_Y_LABEL.get(self.ctx.scene.graph.scaling, "Value"))
            lo, hi = self._ylim
            x_hi = float(x[-1]) if x[-1] > x[0] else float(x[0]) + 1.0
            xr = self._time_view or (float(x[0]), x_hi)
            self._vb.setRange(xRange=xr, yRange=(lo, hi),
                              padding=0.0 if self._time_view else 0.03)
        else:
            xs, ys = [], []
            gap = np.array([np.nan])
            for r, row in enumerate(self._series):
                for c, ts in enumerate(row):
                    if ts is None or not ts.size:
                        continue
                    px, py = self._cell_xy(r, c, x[: ts.size], ts)
                    xs += [px, gap]
                    ys += [py, gap]
            all_x = np.concatenate(xs)
            self._set_curve_pen(all_x.size)
            self._curve.setData(all_x, np.concatenate(ys), connect="finite")
            lines_x, lines_y = [], []
            for k in range(dim + 1):
                lines_x += [0, dim, k, k]
                lines_y += [k, k, 0, dim]
            self._grid_item.setData(np.asarray(lines_x, float), np.asarray(lines_y, float))
            h = dim // 2
            box_x = [h, h + 1, h + 1, h + 1, h + 1, h, h, h]
            box_y = [h, h, h, h + 1, h + 1, h + 1, h + 1, h]
            self._center_item.setData(np.asarray(box_x, float), np.asarray(box_y, float))
            pi.hideAxis("left")
            pi.hideAxis("bottom")
            self._vb.setRange(xRange=(0, dim), yRange=(0, dim), padding=0.0)
        self.update_markers()
        self._draw_events()
        self._draw_tracks()

    # ------------------------------------------------------------------
    # The run around the series: events and physio
    # ------------------------------------------------------------------

    def _to_x(self, seconds: np.ndarray) -> Optional[np.ndarray]:
        """Run seconds into the graph's x units (seconds, or volumes when the
        axis counts volumes and the repetition time is known)."""
        if self._x_is_time:
            return np.asarray(seconds, dtype=float)
        tr = getattr(self._context, "tr", None)
        if tr:
            return np.asarray(seconds, dtype=float) / float(tr)
        return None

    def event_bands(self) -> list[tuple[float, float, str]]:
        """``(start, end, label)`` of every event drawn, in x units (tests)."""
        return list(getattr(self, "_bands", []))

    def _bands_in_x(self):
        """``(starts, ends, labels)`` in x units: the run's events, or for a
        diffusion series its b=0 volumes (where the signal jumps, which
        reads as an artefact until you know why). None: nothing to draw."""
        ctx = self._context
        x = self._x
        if ctx is None or x is None or not x.size:
            return None
        events = list(getattr(ctx, "events", []) or [])
        if events and self.ctx.scene.graph.events:
            onsets = self._to_x(np.asarray([e.onset for e in events], dtype=float))
            ends = self._to_x(np.asarray([e.onset + e.duration for e in events], dtype=float))
            if onsets is None:
                return None
            return onsets, ends, [e.label for e in events]
        bvals = getattr(ctx, "bvals", None)
        if bvals is not None and len(bvals) >= x.size and x.size > 1:
            from ....viz.bids import shells_of

            b0 = np.flatnonzero(shells_of(bvals[: x.size]) == 0)
            if b0.size == 0 or b0.size == x.size:
                return None
            half = float(x[1] - x[0]) / 2.0
            return x[b0] - half, x[b0] + half, ["b=0"] * b0.size
        return None

    def _draw_events(self) -> None:
        item = self._events_item
        self._bands = []
        found = self._bands_in_x() if self._dim else None
        if found is None:
            item.setVisible(False)
            return
        onsets, ends, names = found
        x0, x1 = float(self._x[0]), float(self._x[-1])
        # An instantaneous event is still a visible band: a fixed share of
        # the run, so it reads the same at any length.
        least = max((x1 - x0) * 0.002, 1e-9)
        ends = np.maximum(ends, onsets + least)
        labels = sorted(set(names))
        theme = self.ctx.theme
        colours = {lab: QColor(*parse_colour(theme.series(i))) for i, lab in enumerate(labels)}
        for c in colours.values():
            c.setAlpha(70)
        keep = (ends >= x0) & (onsets <= x1)
        onsets, ends = onsets[keep], ends[keep]
        kept = [n for n, k in zip(names, keep) if k]
        if not kept:
            item.setVisible(False)
            return
        self._bands = [(float(a), float(b), n) for a, b, n in zip(onsets, ends, kept)]
        brushes = [self._pg.mkBrush(colours[n]) for n in kept]
        if self._dim == 1:
            lo, hi = self._vb.viewRange()[1]
            item.setOpts(x0=onsets, x1=ends, y0=np.full(onsets.size, lo),
                         y1=np.full(onsets.size, hi), brushes=brushes,
                         pens=[self._pg.mkPen(None)] * onsets.size)
        else:
            # The same bands in every cell of the neighbourhood.
            dim = self._dim
            sx = (1.0 - 2 * _MARGIN) / max(x1 - x0, 1e-12)
            px0, px1, py0, py1, br = [], [], [], [], []
            for r in range(dim):
                for c in range(dim):
                    px0.append(c + _MARGIN + (np.clip(onsets, x0, x1) - x0) * sx)
                    px1.append(c + _MARGIN + (np.clip(ends, x0, x1) - x0) * sx)
                    base = (dim - 1 - r) + _MARGIN
                    py0.append(np.full(onsets.size, base))
                    py1.append(np.full(onsets.size, base + 1.0 - 2 * _MARGIN))
                    br += brushes
            item.setOpts(x0=np.concatenate(px0), x1=np.concatenate(px1),
                         y0=np.concatenate(py0), y1=np.concatenate(py1), brushes=br,
                         pens=[self._pg.mkPen(None)] * len(br))
        item.setVisible(True)

    def _run_seconds(self) -> Optional[tuple[float, float]]:
        """The span shown, in run seconds (for the physio read): the zoomed
        window when there is one, so a zoom reads the physio at full detail
        for just that stretch."""
        if self._x is None or not self._x.size:
            return None
        x0, x1 = self._time_view or (float(self._x[0]), float(self._x[-1]))
        if self._x_is_time:
            return x0, x1
        tr = getattr(self._context, "tr", None)
        return (x0 * tr, x1 * tr) if tr else None

    # -- zooming in time ---------------------------------------------------

    def _full_x(self) -> Optional[tuple[float, float]]:
        if self._x is None or not self._x.size:
            return None
        return float(self._x[0]), float(self._x[-1])

    def set_time_view(self, window: Optional[tuple[float, float]]) -> None:
        """Show ``window`` (graph x units) of the run, or all of it."""
        full = self._full_x()
        if window is not None and full is not None:
            lo, hi = sorted(window)
            span = full[1] - full[0]
            minimum = span / 2000.0 if span > 0 else 1e-6
            if hi - lo < minimum:
                mid = (lo + hi) / 2.0
                lo, hi = mid - minimum / 2, mid + minimum / 2
            lo, hi = max(lo, full[0]), min(hi, full[1])
            if hi - lo >= 0.98 * span:
                window = None
            else:
                window = (lo, hi)
        if window == self._time_view:
            return
        self._time_view = window
        if self._dim == 1:
            self._draw()
        else:
            self._update_track_view()
        # The physio for the new stretch, once the wheel settles.
        self._zoom_timer.start()

    def time_view(self) -> Optional[tuple[float, float]]:
        return self._time_view

    def _on_wheel(self, event, card=None, *_a, **_k) -> None:
        mods = event.modifiers()
        ctrl = bool(mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        delta = event.angleDelta().y() or event.angleDelta().x()
        full = self._full_x()
        if not (ctrl or shift):
            # The plain wheel steps through the volumes, as over a slice
            # (down or right = forward in time). Over the plots under the
            # graph it scrolls their column when there is one to scroll.
            if card is not None and self.tracks.can_scroll():
                event.ignore()
                return
            steps = self._wheel_steps(event)
            if steps:
                self.ctx.run("frame.step", n=steps)
            event.accept()
            return
        if not delta or full is None:
            event.ignore()
            return
        lo, hi = self._time_view or full
        if ctrl:
            # Around the pointer, read from the track or the graph under it.
            widget = card.plot if card is not None else self.plot
            vb = widget.getPlotItem().getViewBox()
            at = vb.mapSceneToView(widget.mapToScene(event.position().toPoint())).x()
            if self._dim > 1 and card is None:
                at = (lo + hi) / 2.0
            at = min(max(at, lo), hi)
            factor = 0.8 if delta > 0 else 1.25
            self.set_time_view((at - (at - lo) * factor, at + (hi - at) * factor))
        else:
            shift_by = (hi - lo) * (0.1 if delta < 0 else -0.1)
            if lo + shift_by < full[0]:
                shift_by = full[0] - lo
            if hi + shift_by > full[1]:
                shift_by = full[1] - hi
            self.set_time_view((lo + shift_by, hi + shift_by))
        event.accept()

    def _wheel_steps(self, event) -> int:
        """Whole volumes from a wheel event (a trackpad's pixels add up)."""
        from .slice import WHEEL_ANGLE_STEP, WHEEL_PIXEL_STEP

        pd, ad = event.pixelDelta(), event.angleDelta()
        use_pixel = not pd.isNull()
        dx, dy = (pd.x(), pd.y()) if use_pixel else (ad.x(), ad.y())
        horizontal = abs(dx) > abs(dy)
        delta = dx if horizontal else -dy
        thresh = WHEEL_PIXEL_STEP if use_pixel else WHEEL_ANGLE_STEP
        self._wheel_acc = getattr(self, "_wheel_acc", 0.0) + delta
        steps = int(self._wheel_acc / thresh)
        self._wheel_acc -= steps * thresh
        return steps

    _SPLIT_KEY = "graph-tracks"

    def _remember_split(self) -> None:
        sizes = [int(v) for v in self.split.sizes()]
        if len(sizes) == 2 and all(v > 0 for v in sizes):
            self.ctx.settings_hub.update(lambda s: s.layout_sizes.update({self._SPLIT_KEY: sizes}))

    #: Pixels the graph keeps above the tracks.
    GRAPH_PX = 150

    def _size_split(self) -> None:
        """The tracks their remembered share, else room for each at its
        fitted height. Asks the host for that height first, never insists:
        a floor here was once a floor under the whole viewer."""
        need = self.tracks.wanted_height()
        self.wants_height.emit(self.GRAPH_PX + min(need, 560) + 50)
        total = max(self.split.height(), self.GRAPH_PX + 120)
        saved = self.ctx.settings.layout_sizes.get(self._SPLIT_KEY)
        if saved and len(saved) == 2 and sum(saved) > 0:
            below = int(total * saved[1] / sum(saved))
        else:
            below = min(need, total - self.GRAPH_PX)
        self.split.setSizes([max(total - below, 80), max(below, 90)])

    # -- tracks: QC rows and physio, each its own plot --------------------------

    def _frames_to_x(self, frames) -> np.ndarray:
        """Volume positions (fractional allowed) into the graph's x units."""
        f = np.asarray(frames, dtype=float)
        x = self._x
        if x is None or x.size < 2:
            return f if x is None or not x.size else f + float(x[0])
        out = np.interp(f, np.arange(x.size, dtype=float), x)
        out = np.where(f < 0, x[0] + f * (x[1] - x[0]), out)
        return np.where(f > x.size - 1, x[-1] + (f - (x.size - 1)) * (x[-1] - x[-2]), out)

    def _in_x(self, track: dict, skip: int) -> Optional[dict]:
        """A track in the graph's x units: QC tracks are per volume, physio
        in run seconds (None when seconds cannot be placed on a volume axis).
        The non-steady-state volumes are shaded on every QC track."""
        t = dict(track)
        if t.get("frames"):
            conv = self._frames_to_x
            if skip:
                t["bands"] = [(-0.5, skip - 0.5)]
        else:
            if self._to_x(np.zeros(1)) is None:
                return None
            conv = self._to_x
        t["x"] = conv(t.get("x", np.empty(0)))
        t["ticks"] = conv(t.get("ticks", np.empty(0)))
        if "image_x" in t:
            x0, x1 = conv(np.asarray(t["image_x"], dtype=float))
            t["image_x"] = (float(x0), float(x1))
        if t.get("bands"):
            t["bands"] = [tuple(float(v) for v in conv(np.asarray(b, dtype=float)))
                          for b in t["bands"]]
        x = self._x
        if x is not None and x.size > 1:
            t.setdefault("near", 0.5 * float(np.median(np.diff(x))))
        return t

    def _tracks_wanted(self) -> list[dict]:
        """Every track to show: the QC rows in the user's order, then the
        run's physio."""
        g = self.ctx.scene.graph
        out: list[dict] = []
        skip = 0
        if g.qc:
            have = self._qc_src["rows"] if self._qc_src is not None else {}
            skip = int(self._qc_src.get("skip", 0)) if self._qc_src is not None else 0
            for r in self._qc_rows_shown():
                if r in have:
                    out.append(have[r])
                elif r in self._qc_pending:
                    # Being computed: its card now, with a spinner, at the size
                    # the plot will have, so nothing jumps when it arrives.
                    out.append(self._pending_track(r))
        if g.physio and self._physio_src is not None:
            out += self._physio_src["tracks"]
        return [t for t in (self._in_x(t, skip) for t in out) if t is not None]

    def _draw_tracks(self) -> None:
        """The plots under the graph: per-volume QC and the run's physio,
        each computed once on a worker, never on a crosshair move."""
        ctx = self._context
        span = self._run_seconds()
        g = self.ctx.scene.graph
        if g.physio and ctx is not None and ctx.physio_paths and span is not None:
            key = (tuple(str(p) for p in ctx.physio_paths), round(span[0], 6),
                   round(span[1], 6))
            if self._physio_key != key:
                # Read and decimated ONCE per run and span: the run's physio
                # is millions of samples.
                self._physio_key = key
                self._physio_src = None
                self._physio_generation += 1
                self.ctx.jobs.start("graph-physio", self._physio_generation,
                                    physio_for_graph, ctx.physio_paths[0], span[0], span[1])
        _layer, src = views.series_layer(self.ctx.store)
        if g.qc and src is not None and src.fully_loaded:
            from ....qc.config import key as config_key

            cfg = self._qc_config()
            key = (id(src), src.loaded_frames, config_key(cfg))
            if self._qc_key != key:
                self._qc_key = key
                self._qc_src = None
                self._qc_asked = set()
                self._qc_pending = set()
            have = set(self._qc_src["rows"]) if self._qc_src is not None else set()
            # Only the rows not yet computed. A new job replaces a running
            # one, so it asks for everything still missing, not only the
            # row just switched on; a row that failed is not retried until
            # the rows asked for change.
            missing = set(self._qc_rows_shown()) - have
            if missing and not missing <= self._qc_asked:
                self._qc_asked = missing
                self._qc_pending = set(missing)
                self._qc_progress = ""
                self._qc_generation += 1
                bids = self._context
                axis = 2
                if bids is not None:
                    axis = {"i": 0, "j": 1, "k": 2}.get(
                        str(bids.sidecar.get("SliceEncodingDirection", "k"))[:1], 2)
                self.ctx.jobs.start(
                    "graph-qc", self._qc_generation, qc_for_graph, src, sorted(missing),
                    bids.path if bids is not None else None,
                    bids.root if bids is not None else None, axis, config=cfg)
        tracks = self._tracks_wanted()
        if not tracks:
            if not self.tracks.isHidden():
                self.tracks.setVisible(False)
            return
        first = self.tracks.isHidden()
        self.tracks.setVisible(True)
        self.tracks.set_mode(g.tracks_mode)
        self.tracks.set_tracks(tracks, x_label="Time (s)" if self._x_is_time else "Volume")
        self._update_track_view()
        need = self.tracks.wanted_height()
        if (first and not self._tracks_sized) or need > self._sized_height:
            # Re-sized when tracks are added: eight plots in four plots'
            # room is cramped.
            self._tracks_sized = True
            self._sized_height = need
            self._size_split()

    def _update_track_view(self) -> None:
        """The tracks on the graph's stretch of time, with its volume marked."""
        full = self._full_x()
        if full is None or self.tracks.isHidden():
            return
        if self._time_view is not None:
            lo, hi = self._time_view
        else:
            lo, hi = full
            pad = 0.03 * (hi - lo) if hi > lo else 0.5
            lo, hi = lo - pad, hi + pad
        self.tracks.set_x_range(lo, hi)
        self.tracks.set_marker(self._current_x())
        self.tracks.request_align()

    def _plot_area(self) -> Optional[tuple[int, int]]:
        """The time span both the graph and its tracks use, as global (left,
        right): the graph's plot area, its right edge pulled in to where the
        tracks can reach (they sit in cards beside a scroll bar), so a moment
        is at the same pixel in all of them. None for the small multiples,
        whose cells share no time axis."""
        pi = self.plot.getPlotItem()
        reach = self.tracks.reach() if not self.tracks.isHidden() else None
        if self._dim != 1 or not self.plot.isVisible() or reach is None:
            if getattr(self, "_right_margin", 0):
                self._right_margin = 0
                pi.layout.setContentsMargins(1, _TOP_PX, 1, 1)
            return None
        origin = self.plot.mapToGlobal(self.plot.rect().topLeft()).x()
        full_right = origin + self.plot.width()
        right = min(full_right, reach[1])
        margin = int(full_right - right)
        if getattr(self, "_right_margin", 0) != margin:
            self._right_margin = margin
            pi.layout.setContentsMargins(1, _TOP_PX, max(1, margin), 1)
        r = self._vb.sceneBoundingRect()
        left = self.plot.mapToGlobal(self.plot.mapFromScene(r.topLeft())).x()
        return int(max(left, reach[0])), int(right)

    def _current_x(self) -> Optional[float]:
        layer, src = views.series_layer(self.ctx.store)
        if layer is None or src is None or self._x is None or not self._x.size:
            return None
        t = views.frame_of(self.ctx.store, layer, src)
        return float(self._x[max(0, min(t, self._x.size - 1))])

    def _describe_x(self, x: float) -> str:
        """The moment under the pointer, as the tracks' readouts name it."""
        if self._x is None or not self._x.size:
            return f"{x:g}"
        i = int(np.argmin(np.abs(self._x - x)))
        if self._x_is_time:
            return f"{float(self._x[i]):.1f} s (volume {i})"
        tr = getattr(self._context, "tr", None)
        return f"volume {i}" + (f" ({i * float(tr):.1f} s)" if tr else "")

    def _go_to_x(self, x: float) -> None:
        if self._x is None or not self._x.size:
            return
        self.ctx.run("frame.set", frame=int(np.argmin(np.abs(self._x - x))))

    def _go_to_cell(self, x: float, row: str) -> None:
        """A cell of an image track: its moment, and for the diffusion slice
        image its slice (the crosshair moves to it along the slice axis)."""
        self._go_to_x(x)
        if not str(row).startswith("slice "):
            return
        try:
            k = int(str(row).split()[1])
        except (IndexError, ValueError):
            return
        go_to_slice(self.ctx, k, self._context)

    def _on_tracks_order(self, order: list) -> None:
        """The user moved a plot: the QC rows take its order (rows not yet
        on screen keep their place after them)."""
        rows = list(self.ctx.scene.graph.qc_rows)
        moved = [r for r in order if r in rows]
        self.ctx.run("graph.set", qc_rows=moved + [r for r in rows if r not in moved])

    def _on_job_done(self, tag: str, generation: int, result) -> None:
        if tag == "graph-physio" and generation == self._physio_generation:
            self._physio_src = result
        elif tag == "graph-qc" and generation == self._qc_generation:
            if self._qc_src is None:
                self._qc_src = result
            else:
                self._qc_src["rows"].update(result["rows"])
            self._qc_pending = set()
            self._tracks_sized = False
        else:
            return
        self._draw_tracks()

    def _on_job_failed(self, tag: str, generation: int, message: str) -> None:
        if tag == "graph-physio" and generation == self._physio_generation:
            self._physio_src = None
            self.ctx.status.emit(f"The run's physio could not be read: {message}")
        elif tag == "graph-qc" and generation == self._qc_generation:
            self._qc_pending = set()
            self.ctx.status.emit(f"The QC plots could not be computed: {message}")
            self._draw_tracks()

    def _on_job_progress(self, tag: str, generation: int, done: int, total: int) -> None:
        """How far the QC plots being computed are, in their cards."""
        if tag != "graph-qc" or generation != self._qc_generation or not self._qc_pending:
            return
        self._qc_progress = f"{100 * done // max(total, 1)} %" if total > 0 else ""
        text = self._pending_text()
        for r in self._qc_pending:
            card = self.tracks.card(r)
            if card is not None and card.readout.text() != text:
                card.readout.setText(text)

    def _pending_text(self) -> str:
        return "Computing" + (f", {self._qc_progress}" if self._qc_progress else "...")

    def _pending_track(self, row_id: str) -> dict:
        """The card of a QC row still being computed."""
        from ....viz.compute import qc

        dwi = self._series_is_dwi()
        rows = qc.DWI_QC_ROWS if dwi else qc.QC_ROWS
        title = next((t for r, t, _h in rows if r == row_id), row_id)
        image = row_id in ("carpet", "slices")
        return {"id": row_id, "title": title, "unit": "", "kind": "pending", "frames": True,
                "x": np.empty(0), "ys": [], "ticks": np.empty(0), "movable": True,
                "closable": True, "summary": self._pending_text(),
                "note": "Being computed in the background; the plot appears here.",
                "explain": f"plot.{'dwi' if dwi else 'bold'}.{row_id}",
                "height": 2.5 if image else 1}

    def _qc_config(self):
        """The methods and thresholds of Settings > Quality control."""
        from ....viz.settings import qc_config

        return qc_config(self.ctx.settings)

    def _on_settings(self) -> None:
        """A setting changed: the look, and the QC plots when their
        configuration did (``_draw_tracks`` compares it)."""
        self.apply_theme()
        if self.ctx.scene.graph.qc and self._qc_key is not None:
            self._draw_tracks()

    def physio_channels(self) -> list[str]:
        """The physio channels drawn (tests)."""
        got = self._physio_src
        if got is None or self.tracks.isHidden() or not self.ctx.scene.graph.physio:
            return []
        return list(got["names"])

    def shown_tracks(self) -> list[dict]:
        """The plots under the graph, in order: id, title, kind, summary,
        note and how many marks (tests and the tooltip). A card still being
        computed is not a plot yet: :meth:`pending_tracks`."""
        if self.tracks.isHidden():
            return []
        return [{"id": c.track["id"], "title": c.track["title"], "kind": c.track.get("kind"),
                 "summary": c.track.get("summary", ""), "note": c.track.get("note", ""),
                 "ticks": int(np.asarray(c.track.get("ticks", ())).size)}
                for c in self.tracks.cards() if c.track.get("kind") != "pending"]

    def pending_tracks(self) -> list[str]:
        """The ids of the cards still being computed, in order."""
        if self.tracks.isHidden():
            return []
        return [c.track["id"] for c in self.tracks.cards()
                if c.track.get("kind") == "pending"]

    def _marked_cells(self) -> list[tuple[int, int]]:
        """(row, col) of the cells that carry a marker, centre first."""
        h = self._dim // 2
        if not self.ctx.scene.graph.mark_neighbors:
            return [(h, h)]
        cells = [(h, h)]
        for r, row in enumerate(self._series):
            for c, ts in enumerate(row):
                if ts is not None and ts.size and (r, c) != (h, h):
                    cells.append((r, c))
        return cells

    def marker_points(self) -> list[tuple[float, float]]:
        """Where the markers are, in DATA units (x, value), centre first."""
        layer, src = views.series_layer(self.ctx.store)
        if layer is None or src is None or self._x is None or not self._series:
            return []
        t = views.frame_of(self.ctx.store, layer, src)
        out = []
        for r, c in self._marked_cells():
            ts = self._series[r][c]
            if ts is None or not ts.size:
                continue
            i = max(0, min(t, ts.size - 1))
            out.append((float(self._x[i]), float(ts[i])))
        return out

    def update_markers(self) -> None:
        if not self.tracks.isHidden():
            self.tracks.set_marker(self._current_x())
        points = self.marker_points()
        size = self.ctx.scene.graph.dot
        if not points:
            self._markers.setData([], [])
            return
        if self._dim == 1:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
        else:
            xs, ys = [], []
            for (r, c), (px, py) in zip(self._marked_cells(), points):
                mx, my = self._cell_xy(r, c, np.array([px]), np.array([py]))
                xs.append(float(mx[0]))
                ys.append(float(my[0]))
        self._markers.setData(xs, ys, size=size)

    def _clear(self) -> None:
        self._curve.setData([], [])
        self._grid_item.setData([], [])
        self._center_item.setData([], [])
        self._markers.setData([], [])
        self._events_item.setVisible(False)
        if not self.tracks.isHidden():
            self.tracks.setVisible(False)
        self._bands = []
        self._dim = 0
        self._x = None
        self._series = []

    def _on_click(self, event) -> None:
        """Jump to the frame nearest the click; a double-click shows the
        whole run again after a zoom."""
        if self._x is None or not len(self._x):
            return
        if event.double() and self._time_view is not None:
            self.set_time_view(None)
            return
        pos = self._vb.mapSceneToView(event.scenePos())
        x = pos.x()
        if self._dim > 1:
            c = int(np.floor(x))
            if not 0 <= c < self._dim:
                return
            x0, x1 = float(self._x[0]), float(self._x[-1])
            frac = (x - c - _MARGIN) / (1.0 - 2 * _MARGIN)
            x = x0 + float(np.clip(frac, 0.0, 1.0)) * (x1 - x0)
        frame = int(np.argmin(np.abs(self._x - x)))
        self.ctx.run("frame.set", frame=frame)
        event.accept()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _ask_export(self) -> None:
        if self._x is None:
            return
        layer, src = views.series_layer(self.ctx.store)
        base = src.path.name.split(".")[0] if src is not None else "timecourse"
        path, _ = QFileDialog.getSaveFileName(self, "Export time courses",
                                              f"{base}_timecourse.csv", "CSV (*.csv)")
        if path:
            self.export_csv(path)

    def export_csv(self, path) -> None:
        """Write the plotted series: the x column, then one column per voxel
        (the FILE's own indices, centre first), in the scaling shown."""
        import csv
        from pathlib import Path

        if self._x is None or not self._series:
            raise ValueError("nothing is plotted")
        scaling = self.ctx.scene.graph.scaling
        tag = {"raw": "", "percent": " (percent change)", "demean": " (demeaned)"}[scaling]
        h = self._dim // 2
        cells = [(h, h)] + [(r, c) for r in range(self._dim) for c in range(self._dim)
                            if (r, c) != (h, h)]
        columns, names = [], []
        for r, c in cells:
            ts, vox = self._series[r][c], self._voxels[r][c]
            if ts is None or vox is None:
                continue
            columns.append(ts)
            names.append(f"voxel {vox[0]} {vox[1]} {vox[2]}{tag}")
        n = len(self._x)
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_s" if self._x_is_time else "volume", *names])
            for i in range(n):
                writer.writerow([f"{self._x[i]:.6g}",
                                 *(f"{col[i]:.6g}" if i < col.size else "" for col in columns)])

    # ------------------------------------------------------------------
    # Queries (tests, screenshots, linked views)
    # ------------------------------------------------------------------

    def cell_count(self) -> int:
        """How many voxels are plotted (neighbours outside the volume are not)."""
        return sum(1 for row in self._series for ts in row if ts is not None)

    def center_series(self) -> Optional[np.ndarray]:
        if not self._series:
            return None
        h = self._dim // 2
        return self._series[h][h]

    def x_values(self) -> Optional[np.ndarray]:
        return None if self._x is None else self._x.copy()

    def x_is_time(self) -> bool:
        return self._x_is_time

    def marker_size(self) -> float:
        pts = self._markers.points()
        return float(pts[0].size()) if len(pts) else 0.0

    def mouse_locked(self) -> bool:
        return self._vb.state["mouseEnabled"] == [False, False]

    def view_to_scene(self, x: float, y: float):
        """Scene position of DATA point (x, y) in the centre cell (clicks)."""
        from PyQt6.QtCore import QPointF

        if self._dim > 1:
            h = self._dim // 2
            px, py = self._cell_xy(h, h, np.array([x]), np.array([y]))
            x, y = float(px[0]), float(py[0])
        return self._vb.mapViewToScene(QPointF(x, y))


__all__ = ["TimecourseGraph"]
