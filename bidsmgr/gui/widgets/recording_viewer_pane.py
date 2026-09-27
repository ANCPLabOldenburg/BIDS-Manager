"""EEG / MEG / iEEG recording viewer (Editor center pane, recording kind).

Sister widget to :class:`NiftiViewerPane` / :class:`TsvViewerPane` /
:class:`SidecarFormPane`. When the user clicks a recording file in the
BIDS tree (``.fif``, ``.edf``, ``.set``, ``.vhdr``, ``.cnt``, CTF ``.ds``,
...), :class:`bidsmgr.gui.editor_panel.EditorPanel` swaps its center pane
to this viewer.

Flow
----
1. **Metadata first.** Selecting a recording shows a themed metadata card
   (channels, sampling rate, duration, per-type counts, filters, bad
   channels, measurement date), read on a background thread
   (:class:`bidsmgr.workers.RecordingMetaWorker`, ``preload=False``).
2. **Load signal.** A "Load signal" button reads the full recording on a
   background thread (:class:`bidsmgr.workers.RecordingSignalWorker`,
   ``preload=True``) and switches to the interactive time-series viewer.
   Its **Close** button returns to the metadata card.

The time-series viewer is a restyled, annotation-free port of the MEEGqc
``qc_viewer`` time-series widget (pyqtgraph): channel-type filtering with
CTF axial-gradiometer handling, an individual-channel picker, visible
count / amplitude scale / time-window controls, navigation + channel
scroll + hover tooltips, raw-vs-normalise rendering, HP/LP/notch
filtering, resample (threaded), an interactive in-app PSD view
(pyqtgraph, MNE-like: per-channel + per-type-average tabs), and a
BIDS-native event overlay (sibling ``*_events.tsv`` plus stim-channel
``find_events``).

Theme handling: toolbar / metadata controls are QSS-driven; the
pyqtgraph plot reads the palette explicitly (it does not honour QSS),
following the :class:`NiftiViewerPane` pattern. Everything (reads, PSD,
resample) runs on ``QThread`` workers so the GUI never blocks; stale
results are dropped via a path guard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .primitives import ElidedLabel, PaneHeader
from .recording_formats import full_ext, is_recording_path  # noqa: F401
from .time_series_view import TimeSeriesView
from .spinner import BusySpinner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Qt-free read helpers (imported by the loader workers).
# ---------------------------------------------------------------------------
# Extensions ``mne.io.read_raw`` refuses to auto-dispatch (ambiguous between
# vendors). Try the generic reader first, then these in order. ``read_raw_ant``
# needs the optional ``antio`` package; absent it the error is surfaced cleanly.
_SPECIFIC_READERS: dict[str, tuple[str, ...]] = {
    ".cnt": ("read_raw_cnt", "read_raw_ant"),
    ".egi": ("read_raw_egi",),
    ".mff": ("read_raw_egi",),
}


def _read_raw(path, *, preload: bool):
    """Read a recording with MNE. ``preload`` controls full vs lazy load."""
    import mne

    p = str(path)
    try:
        return mne.io.read_raw(p, preload=preload, verbose=False)
    except Exception as primary:
        readers = _SPECIFIC_READERS.get(full_ext(path))
        if not readers:
            raise
        last = primary
        for name in readers:
            fn = getattr(mne.io, name, None)
            if fn is None:
                continue
            try:
                return fn(p, preload=preload, verbose=False)
            except Exception as exc:  # noqa: BLE001 - try the next reader
                last = exc
        raise last


def _summarize_raw(raw, path) -> dict:
    """Build a plain display summary from an ``mne.io.Raw`` (no Qt)."""
    import mne

    info = raw.info
    ch_names = list(info["ch_names"])
    ch_types = [mne.channel_type(info, i) for i in range(len(ch_names))]
    counts: dict[str, int] = {}
    for t in ch_types:
        counts[t] = counts.get(t, 0) + 1
    available = sorted(set(ch_types))
    is_ctf = (
        "mag" in available
        and "grad" not in available
        and (
            "ref_meg" in available
            or getattr(raw, "compensation_grade", None) is not None
            or full_ext(path) == ".ds"
        )
    )
    try:
        duration = float(raw.times[-1]) if raw.n_times else 0.0
    except Exception:
        duration = 0.0
    meas = info.get("meas_date")
    return {
        "filename": str(path),
        "name": Path(path).name,
        "n_channels": len(ch_names),
        "sfreq": float(info["sfreq"]),
        "duration": duration,
        "n_times": int(raw.n_times),
        "highpass": info.get("highpass"),
        "lowpass": info.get("lowpass"),
        "line_freq": info.get("line_freq"),
        "meas_date": str(meas) if meas else None,
        "ch_type_counts": counts,
        "available_ch_types": available,
        "is_ctf": is_ctf,
        "bads": list(info.get("bads") or []),
    }


# ===========================================================================
# The pane (metadata-first + time-series, with threaded loading)
# ===========================================================================
class RecordingViewerPane(QWidget):
    """Center pane for EEG/MEG/iEEG recordings.

    Bound to a single recording via :meth:`set_file`; pass ``None`` to
    clear and unload. Metadata loads on a worker thread first; a "Load
    signal" button loads the full recording (also threaded) and reveals
    the interactive time-series view. The view's Close returns here.
    """

    status_message = pyqtSignal(str)
    loading_changed = pyqtSignal(bool, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane-dark")
        self.setMinimumWidth(0)

        self._current_file: Optional[Path] = None
        self._current_root: Optional[Path] = None
        self._meta: Optional[dict] = None
        self._meta_worker = None
        self._signal_worker = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader("Recording"))

        self._stack = QStackedWidget()
        v.addWidget(self._stack, 1)

        self._hint = QLabel(
            "Select an EEG / MEG recording in the BIDS tree to view it."
        )
        self._hint.setObjectName("pane-hint")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setWordWrap(True)
        self._stack.addWidget(self._hint)

        self._loading_page = self._build_loading_page()
        self._stack.addWidget(self._loading_page)

        self._meta_page = self._build_meta_page()
        self._stack.addWidget(self._meta_page)

        # The pyqtgraph time-series view is built lazily, only when a signal
        # is actually loaded. Editor sessions that just browse JSON/TSV never
        # construct a pyqtgraph plot (cheaper + avoids offscreen-teardown
        # flakiness in the test suite), mirroring the NIfTI viewer.
        self._view: Optional[TimeSeriesView] = None

        self._stack.setCurrentWidget(self._hint)

    def _ensure_view(self) -> "TimeSeriesView":
        if self._view is None:
            self._view = TimeSeriesView()
            self._view.status_message.connect(self.status_message)
            self._view.loading_changed.connect(self.loading_changed)
            self._view.close_requested.connect(self._on_view_close)
            self._stack.addWidget(self._view)
        return self._view

    # ------------------------------------------------------------------ UI
    def _build_loading_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("pane-dark")
        lay = QVBoxLayout(page)
        lay.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        self._spinner = BusySpinner()
        row.addWidget(self._spinner)
        row.addStretch(1)
        lay.addLayout(row)
        # ELIDED: it carries the file's path, and a QStackedWidget sizes
        # itself to its LARGEST page whichever one is showing, so a plain
        # QLabel here is a floor under the whole pane even while hidden.
        self._loading_label = ElidedLabel(
            "", mode=Qt.TextElideMode.ElideMiddle,
        )
        self._loading_label.setObjectName("pane-hint")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._loading_label)
        lay.addStretch(1)
        return page

    def _build_meta_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("pane-dark")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("viewer-meta-scroll")
        self._meta_body = QWidget()
        self._meta_body.setObjectName("pane-dark")
        self._meta_layout = QVBoxLayout(self._meta_body)
        self._meta_layout.setContentsMargins(18, 16, 18, 16)
        self._meta_layout.setSpacing(6)
        self._meta_layout.addStretch(1)
        scroll.setWidget(self._meta_body)
        outer.addWidget(scroll, 1)

        bar = QFrame()
        bar.setObjectName("toolbar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 8, 14, 8)
        bl.setSpacing(8)
        bl.addStretch(1)
        self._load_btn = QPushButton("Load signal")
        self._load_btn.setObjectName("load-signal-btn")
        self._load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._load_btn.clicked.connect(self._load_signal)
        bl.addWidget(self._load_btn)
        bl.addStretch(1)
        outer.addWidget(bar)
        return page

    def _meta_section(self, title: str) -> None:
        lbl = QLabel(title)
        lbl.setObjectName("viewer-meta-section")
        self._meta_layout.insertWidget(self._meta_layout.count() - 1, lbl)

    def _meta_row(self, label: str, value: str) -> None:
        row = QWidget()
        row.setObjectName("viewer-meta-row")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)
        k = QLabel(label)
        k.setObjectName("viewer-meta-key")
        k.setMinimumWidth(96)
        k.setMaximumWidth(140)
        val = QLabel(value)
        val.setObjectName("viewer-meta-val")
        val.setWordWrap(True)
        val.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        rl.addWidget(k)
        rl.addWidget(val, 1)
        self._meta_layout.insertWidget(self._meta_layout.count() - 1, row)

    def _clear_meta(self) -> None:
        while self._meta_layout.count() > 1:
            item = self._meta_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _populate_meta(self, meta: dict) -> None:
        self._clear_meta()

        title = QLabel(meta.get("name", ""))
        title.setObjectName("viewer-meta-title")
        title.setWordWrap(True)
        self._meta_layout.insertWidget(self._meta_layout.count() - 1, title)

        self._meta_section("Recording")
        self._meta_row("Channels", str(meta.get("n_channels", 0)))
        self._meta_row("Sampling rate", f"{meta.get('sfreq', 0):.1f} Hz")
        dur = meta.get("duration", 0.0)
        self._meta_row("Duration", f"{dur:.1f} s ({dur / 60:.1f} min)")
        self._meta_row("Time points", str(meta.get("n_times", 0)))
        if meta.get("meas_date"):
            self._meta_row("Measurement date", str(meta["meas_date"]))

        self._meta_section("Filters")
        hp = meta.get("highpass")
        lp = meta.get("lowpass")
        lf = meta.get("line_freq")
        self._meta_row("High-pass", f"{hp:g} Hz" if hp not in (None, "") else "n/a")
        self._meta_row("Low-pass", f"{lp:g} Hz" if lp not in (None, "") else "n/a")
        self._meta_row("Line frequency", f"{lf:g} Hz" if lf not in (None, "") else "n/a")

        self._meta_section("Channel types")
        counts = meta.get("ch_type_counts", {})
        for ct, n in sorted(counts.items()):
            label = "mag (axial grad)" if meta.get("is_ctf") and ct == "mag" else ct
            self._meta_row(label, str(n))
        if meta.get("is_ctf"):
            note = QLabel(
                "CTF dataset: MEG sensors are axial gradiometers "
                "(MNE labels them “mag”)."
            )
            note.setObjectName("viewer-meta-note")
            note.setWordWrap(True)
            self._meta_layout.insertWidget(self._meta_layout.count() - 1, note)

        bads = meta.get("bads") or []
        if bads:
            self._meta_section(f"Bad channels ({len(bads)})")
            self._meta_row("", ", ".join(bads))

    # -------------------------------------------------------------- public
    def current_file(self) -> Optional[Path]:
        return self._current_file

    def set_file(self, path: Optional[Path], root: Optional[Path]) -> None:
        """Bind to a recording (or ``None`` to clear + unload)."""
        self._cancel_workers()
        self._current_file = path
        self._current_root = root
        if path is None:
            if self._view is not None:
                self._view.unload()
            self._spinner.set_busy(False)
            self._stack.setCurrentWidget(self._hint)
            self._hint.setText(
                "Select an EEG / MEG recording in the BIDS tree to view it."
            )
            self.loading_changed.emit(False, "")
            return

        from ...workers import RecordingMetaWorker

        self._show_loading(f"Loading metadata: {path.name}…")
        worker = RecordingMetaWorker(path, parent=self)
        worker.finished_with_meta.connect(self._on_meta)
        worker.failed.connect(self._on_meta_failed)
        worker.finished.connect(worker.deleteLater)
        self._meta_worker = worker
        worker.start()

    def _show_loading(self, message: str) -> None:
        self._loading_label.setText(message)
        self._spinner.set_busy(True, message="")
        self._stack.setCurrentWidget(self._loading_page)
        self.loading_changed.emit(True, message)

    def _cancel_workers(self) -> None:
        for w in (self._meta_worker, self._signal_worker):
            if w is not None:
                w.cancel()
        self._meta_worker = None
        self._signal_worker = None

    # ----------------------------------------------------------- callbacks
    def _on_meta(self, meta: dict, path: Path) -> None:
        if path != self._current_file:
            return
        self._meta_worker = None
        self._meta = meta
        self._spinner.set_busy(False)
        self.loading_changed.emit(False, "")
        self._populate_meta(meta)
        self._stack.setCurrentWidget(self._meta_page)
        self.status_message.emit(
            f"{meta.get('name', '')}: {meta.get('n_channels', 0)} ch, "
            f"{meta.get('sfreq', 0):.0f} Hz, {meta.get('duration', 0):.1f}s"
        )

    def _on_meta_failed(self, path: Path, error: str) -> None:
        if path != self._current_file:
            return
        self._meta_worker = None
        self._spinner.set_busy(False)
        self.loading_changed.emit(False, "")
        self._stack.setCurrentWidget(self._hint)
        self._hint.setText(f"Could not read {path.name}:\n{error}")
        self.status_message.emit(f"Error: {error.splitlines()[0] if error else ''}")

    def _load_signal(self) -> None:
        if self._current_file is None:
            return
        from ...workers import RecordingSignalWorker

        path = self._current_file
        self._show_loading(f"Loading signal: {path.name}…")
        worker = RecordingSignalWorker(path, parent=self)
        worker.finished_with_raw.connect(self._on_raw)
        worker.failed.connect(self._on_raw_failed)
        worker.finished.connect(worker.deleteLater)
        self._signal_worker = worker
        worker.start()

    def _on_raw(self, raw, path: Path) -> None:
        if path != self._current_file:
            return
        self._signal_worker = None
        self._spinner.set_busy(False)
        self.loading_changed.emit(False, "")
        view = self._ensure_view()
        view.set_current_filepath(path, self._current_root)
        view.load_raw(raw)
        self._stack.setCurrentWidget(view)

    def _on_raw_failed(self, path: Path, error: str) -> None:
        if path != self._current_file:
            return
        self._signal_worker = None
        self._spinner.set_busy(False)
        self.loading_changed.emit(False, "")
        self._stack.setCurrentWidget(self._meta_page)
        QMessageBox.warning(
            self, "Load error", f"Could not load {path.name}:\n{error}"
        )

    def _on_view_close(self) -> None:
        """Close button in the viewer: drop the signal, show metadata."""
        if self._view is not None:
            self._view.unload()
        if self._meta is not None:
            self._stack.setCurrentWidget(self._meta_page)
        else:
            self._stack.setCurrentWidget(self._hint)
        self.status_message.emit("Signal closed")

    # -------------------------------------------------------------- theme
    def repaint_for_palette(self, pal: dict) -> None:
        style = self.style()
        for w in [self, *self._meta_page.findChildren(QWidget),
                  self._loading_page, self._hint]:
            style.unpolish(w)
            style.polish(w)
            w.update()
        if self._view is not None:
            self._view.repaint_for_palette(pal)


__all__ = ["RecordingViewerPane", "is_recording_path"]
