"""NIfTI 2-D slice viewer (Editor center pane, image kind).

Sister widget to :class:`SidecarFormPane` and :class:`TsvViewerPane`.
When the user clicks a ``.nii`` or ``.nii.gz`` file in the BIDS tree
:class:`EditorPanel` swaps its center pane to this viewer.

What it shows
-------------
* Orientation buttons (Sagittal · Coronal · Axial) — Axial default.
* **Tri-view** toggle — when on, the canvas splits into three
  side-by-side panels (sagittal · coronal · axial). The crosshair
  voxel is shared: clicking any panel moves it, and all three
  re-render to the new slice indices.
* Slice slider (drives ``_cross_voxel`` along the active orientation
  axis), Volume slider (4-D), Brightness / Contrast.
* The 2-D slice itself, rendered with a crosshair at the current
  voxel. Clicking on the image moves the crosshair.
* **Graph** toggle — only enabled for 4-D non-RGB data. Opens a
  pyqtgraph line plot of the time-course at the crosshair voxel,
  with a marker at the current volume index.
* Footer: relative path inside the BIDS root + dimensions / dtype
  summary + voxel-value readout.

What it deliberately doesn't ship (v1)
--------------------------------------
* No 3-D dialog — :class:`Volume3DDialog` in BIDS-Manager v0.2.5 is
  ~1500 LOC of pyqtgraph + scikit-image; deferred to v2.
* No GIFTI / FreeSurfer surface viewer — different file kinds.

Implementation notes
--------------------
* :mod:`nibabel` is the loader. Structured (colour-FA / RGB) dtypes
  use ``recfunctions.structured_to_unstructured`` to flatten the
  components — same trick BIDS-Manager v0.2.5 uses. ``self._is_rgb``
  is set when 3- or 4-component voxels look like colour channels.
* :class:`ImageLabel` is the canvas: it triggers ``_refresh()`` on
  resize (so the pixmap follows splitter changes) and routes mouse
  clicks to the per-axis handler.
* Theme handling is QSS-only on the toolbar/footer + an explicit
  palette pull for the pyqtgraph plot (it doesn't read QSS).
  ``repaint_for_palette`` runs the unpolish/polish dance the other
  panes use plus re-applies the plot colours.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .image_label import ImageLabel
from .primitives import PaneHeader
from .spinner import BusySpinner

log = logging.getLogger(__name__)


# Orientation axis indices — match BIDS-Manager v0.2.5 convention.
_AXIS_SAGITTAL = 0
_AXIS_CORONAL = 1
_AXIS_AXIAL = 2
_AXES = (_AXIS_SAGITTAL, _AXIS_CORONAL, _AXIS_AXIAL)
_AXIS_LABELS = {
    _AXIS_SAGITTAL: "Sagittal",
    _AXIS_CORONAL:  "Coronal",
    _AXIS_AXIAL:    "Axial",
}

# Default crosshair colour — overridden per-user via AppSettings
# (``nifti_crosshair_color``). Material light-blue 300 reads well on
# both dark and bright slices.
_DEFAULT_CROSSHAIR_COLOR = "#4FC3F7"
_DEFAULT_CROSSHAIR_THICKNESS = 1
# Semi-opaque black "halo" drawn underneath when thickness >= 2 so
# the cross stays visible on saturated slices.
_CROSSHAIR_HALO = QColor(0, 0, 0, 160)


def _load_nifti(path: Path) -> tuple[Any, np.ndarray, dict]:
    """Load a NIfTI from disk into ``(img, data, meta)``.

    ``meta`` describes structured-dtype handling (colour-FA et al.).
    ``meta["is_rgb"]`` is True when 3- or 4-component voxels should
    be rendered as colour channels rather than as 4-D volumes.

    Raises whatever nibabel / numpy raise on load failure — callers
    decide whether to surface to the user.
    """
    import nibabel as nib  # local import: nibabel is heavy

    img = nib.load(str(path))
    try:
        data = img.get_fdata()
        return img, data, {}
    except Exception as exc:
        is_dtype_error = (
            exc.__class__.__name__ == "DTypePromotionError"
            or "VoidDType" in str(exc)
        )
        if not is_dtype_error:
            raise

    # Structured / RGB voxels: flatten components into the last axis.
    from numpy.lib import recfunctions as rfn

    dataobj = np.asanyarray(img.dataobj)
    if not getattr(dataobj.dtype, "fields", None):
        raise RuntimeError(
            f"NIfTI {path} has an unsupported dtype: {dataobj.dtype}"
        )
    unstructured = rfn.structured_to_unstructured(dataobj)
    vector_length = (
        int(unstructured.shape[-1])
        if unstructured.ndim == dataobj.ndim + 1
        else 1
    )
    meta = {
        "vector_axis": len(img.shape),
        "vector_length": vector_length,
        "is_rgb": (
            vector_length in (3, 4)
            and unstructured.ndim == dataobj.ndim + 1
        ),
    }
    return img, unstructured.astype(np.float32, copy=False), meta


class NiftiViewerPane(QWidget):
    """2-D slice viewer for ``.nii`` / ``.nii.gz`` files.

    Bound to a single file via :meth:`set_file`; pass ``None`` to
    clear. Read-only — there's no Save / Revert flow because viewing
    a volume doesn't edit it.
    """

    # Emitted whenever the bound file changes (useful for tests).
    # Fires the moment :meth:`set_file` runs, *before* the worker
    # thread has read the data.
    file_changed = pyqtSignal(object)
    # Emitted on successful load — the canvas is now populated. Tests
    # and downstream code should subscribe to this rather than
    # :sig:`file_changed` when they need to access the loaded array.
    loaded = pyqtSignal(Path)
    # Emitted when on-disk load fails. Args: (path, error_msg).
    load_failed = pyqtSignal(Path, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane-dark")

        self._current_file: Optional[Path] = None
        self._current_root: Optional[Path] = None
        # Loaded array + nibabel image. ``None`` means no file bound.
        self._data: Optional[np.ndarray] = None
        self._img: Any = None
        self._meta: dict = {}
        self._is_rgb: bool = False
        # Crosshair voxel in image-space (i, j, k). ``None`` before a
        # file is loaded.
        self._cross_voxel: Optional[list[int]] = None
        # Per-axis scale factor from the source slice to the displayed
        # pixmap; cached so click → voxel can invert it.
        self._img_scale: dict[int, tuple[float, float]] = {
            axis: (1.0, 1.0) for axis in _AXES
        }
        # Current orientation (default Axial) — also the axis the
        # slice slider controls in single-pane mode.
        self._orientation = _AXIS_AXIAL
        # Layout mode flags.
        self._tri_view: bool = False
        self._graph_visible: bool = False
        # pyqtgraph handles (lazy-initialised in _build_graph_panel).
        # ``_plot_layout`` is a GraphicsLayoutWidget hosting a
        # ``dim × dim`` grid of PlotItems (one per neighbour voxel
        # when scope > 1). ``_grid_cells`` is the active grid as a
        # list of dicts ``{plot, curve, marker, ts, is_center}``.
        self._plot_layout = None
        self._grid_cells: list[dict] = []
        # Crosshair styling. Pulled from AppSettings so the user's
        # picks survive across sessions; the inline popup writes back
        # via :meth:`_apply_crosshair_settings`.
        self._crosshair_color = QColor(_DEFAULT_CROSSHAIR_COLOR)
        self._crosshair_thickness = _DEFAULT_CROSSHAIR_THICKNESS
        self._load_persisted_crosshair()
        # Threaded loader handle — replaced on every set_file call.
        # The worker is kept alive on ``self`` so Python doesn't garbage
        # collect it mid-flight, and discarded once its result has
        # been routed.
        self._loader = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader("NIfTI"))

        # --- Toolbar: orientation pills + tri-view + graph + sliders --
        self._toolbar = self._build_toolbar()
        v.addWidget(self._toolbar)

        # --- Stacked content: hint vs. canvas -------------------------
        self._stack = QStackedLayout()
        self._stack.setContentsMargins(0, 0, 0, 0)
        v.addLayout(self._stack, 1)

        self._empty_hint = QLabel(
            "Select a NIfTI (.nii / .nii.gz) file in the BIDS tree "
            "to view it."
        )
        self._empty_hint.setObjectName("pane-hint")
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_hint.setWordWrap(True)
        self._stack.addWidget(self._empty_hint)

        self._canvas = self._build_canvas()
        self._stack.addWidget(self._canvas)

        # Loading page (index 2). Big 4-D BOLD runs can take seconds
        # to read off disk + decompress; the loader runs on a worker
        # thread and this page communicates progress.
        self._loading_panel = self._build_loading_panel()
        self._stack.addWidget(self._loading_panel)

        self._stack.setCurrentIndex(0)

        # --- Footer (path + summary), QSS-themed ---------------------
        self._footer = QFrame()
        self._footer.setObjectName("sidecar-footer")
        fl = QHBoxLayout(self._footer)
        fl.setContentsMargins(14, 6, 14, 6)
        fl.setSpacing(10)
        self._footer_path = QLabel("")
        self._footer_path.setObjectName("sidecar-footer-path")
        self._footer_summary = QLabel("")
        self._footer_summary.setObjectName("sidecar-footer-summary")
        self._voxel_value = QLabel("")
        self._voxel_value.setObjectName("sidecar-footer-summary")
        fl.addWidget(self._footer_path, 1)
        fl.addWidget(self._voxel_value)
        fl.addWidget(self._footer_summary)
        v.addWidget(self._footer)

        self._toolbar.setVisible(False)
        self._footer.setVisible(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_file(self) -> Optional[Path]:
        return self._current_file

    def set_file(
        self,
        path: Optional[Path],
        root: Optional[Path],
    ) -> None:
        """Bind the pane to a NIfTI file (or ``None`` to clear).

        The load itself runs on a :class:`NiftiLoaderWorker` so the
        GUI thread stays responsive — large 4-D BOLDs can take
        seconds to decompress. While loading the pane shows a busy
        spinner; the toolbar / footer stay hidden until the data is
        ready.
        """
        self._current_file = path
        self._current_root = root
        if path is None:
            self._clear()
            self.file_changed.emit(None)
            return

        # Cancel any in-flight load — the user moved on. We don't
        # interrupt the worker mid-read (nibabel's get_fdata is C
        # code that can't be cancelled cleanly), we just suppress
        # its emission so the stale data never reaches the GUI.
        if self._loader is not None:
            self._loader.cancel()
            self._loader = None

        self._show_loading(path)

        # Start the worker. Keep a reference so the QObject doesn't
        # get garbage collected. ``deleteLater`` on ``finished`` is the
        # usual Qt teardown pattern for QThread workers.
        from ...workers import NiftiLoaderWorker
        worker = NiftiLoaderWorker(path, parent=self)
        worker.finished_with_data.connect(self._on_load_complete)
        worker.failed.connect(self._on_load_failed)
        worker.finished.connect(worker.deleteLater)
        self._loader = worker
        worker.start()
        self.file_changed.emit(path)

    def _show_loading(self, path: Path) -> None:
        """Switch to the loading page + start the spinner."""
        self._loading_label.setText(f"Loading {path.name}…")
        self._loading_spinner.set_busy(True, message="")
        self._toolbar.setVisible(False)
        self._footer.setVisible(False)
        self._stack.setCurrentWidget(self._loading_panel)

    def _on_load_complete(
        self,
        img: Any,
        data: np.ndarray,
        meta: dict,
        path: Path,
    ) -> None:
        """Worker finished — populate the canvas from its result.

        Guarded against stale results: if the user changed selection
        between the worker starting and finishing, ``path`` will no
        longer match :attr:`_current_file`. We drop the stale data
        instead of overwriting the active view.
        """
        if path != self._current_file:
            return
        self._loading_spinner.set_busy(False)
        self._img = img
        self._data = data
        self._meta = meta or {}
        self._is_rgb = bool(self._meta.get("is_rgb"))
        # Default crosshair: centre voxel.
        if data.ndim >= 3:
            self._cross_voxel = [
                data.shape[0] // 2,
                data.shape[1] // 2,
                data.shape[2] // 2,
            ]
        else:
            self._cross_voxel = None

        # Volume slider: only 4-D non-RGB data has temporal volumes.
        is_4d = data.ndim == 4 and not self._is_rgb
        n_vols = data.shape[3] if is_4d else 1
        self._vol_slider.setMaximum(max(n_vols - 1, 0))
        self._vol_slider.setEnabled(n_vols > 1)
        self._vol_slider.setValue(0)
        self._vol_val.setText("0")

        # Graph toggle is only meaningful for 4-D non-RGB.
        self._graph_btn.setEnabled(is_4d)
        if not is_4d and self._graph_visible:
            # New file lost its 4th dimension — close the graph.
            self._graph_btn.setChecked(False)

        # Reset brightness / contrast to defaults when a new file is
        # bound so the previous file's settings don't leak.
        self._bright_slider.setValue(0)
        self._contrast_slider.setValue(100)

        self._set_orientation(self._orientation, refresh=False)
        self._toolbar.setVisible(True)
        self._footer.setVisible(True)
        self._stack.setCurrentWidget(self._canvas)
        self._update_footer()
        self._refresh()
        if self._graph_visible:
            self._update_graph()
        self.loaded.emit(path)

    def _on_load_failed(self, path: Path, error: str) -> None:
        if path != self._current_file:
            return
        self._loading_spinner.set_busy(False)
        log.warning("Could not load NIfTI %s: %s", path, error)
        self.load_failed.emit(path, error)
        self._clear()
        self._empty_hint.setText(
            f"Could not load {path.name}:\n{error}"
        )
        self._stack.setCurrentWidget(self._empty_hint)

    def repaint_for_palette(self, pal: dict) -> None:
        """Force QSS recomputation on dark↔light swap."""
        del pal
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()
        # Plot doesn't read QSS — push palette explicitly.
        self._apply_plot_palette()
        # Crosshair colour reads from the palette at paint time; force
        # a re-render so it picks up the new highlight colour.
        if self._data is not None:
            self._refresh()

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("sidecar-toolbar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 6, 14, 6)
        h.setSpacing(8)

        # Orientation pills — Axial default.
        self._sa_btn = QPushButton("Sagittal")
        self._co_btn = QPushButton("Coronal")
        self._ax_btn = QPushButton("Axial")
        self._orient_group = QButtonGroup(bar)
        self._orient_group.setExclusive(True)
        for btn in (self._sa_btn, self._co_btn, self._ax_btn):
            btn.setObjectName("tb-btn-toggle")
            btn.setCheckable(True)
            self._orient_group.addButton(btn)
            h.addWidget(btn)
        self._ax_btn.setChecked(True)
        self._sa_btn.clicked.connect(
            lambda: self._set_orientation(_AXIS_SAGITTAL)
        )
        self._co_btn.clicked.connect(
            lambda: self._set_orientation(_AXIS_CORONAL)
        )
        self._ax_btn.clicked.connect(
            lambda: self._set_orientation(_AXIS_AXIAL)
        )

        h.addSpacing(8)

        # Tri-view toggle — sagittal+coronal+axial side by side.
        self._tri_btn = QPushButton("Tri-view")
        self._tri_btn.setObjectName("tb-btn-toggle")
        self._tri_btn.setCheckable(True)
        self._tri_btn.setToolTip(
            "Show sagittal, coronal and axial panels side by side.\n"
            "The crosshair voxel is shared — clicking any panel moves "
            "it across all three."
        )
        self._tri_btn.toggled.connect(self._on_tri_toggled)
        h.addWidget(self._tri_btn)

        # Graph toggle — 4-D time-series at the crosshair voxel.
        self._graph_btn = QPushButton("Graph")
        self._graph_btn.setObjectName("tb-btn-toggle")
        self._graph_btn.setCheckable(True)
        self._graph_btn.setEnabled(False)
        self._graph_btn.setToolTip(
            "Plot the intensity time-course at the crosshair voxel.\n"
            "Available only for 4-D NIfTI files."
        )
        self._graph_btn.toggled.connect(self._on_graph_toggled)
        h.addWidget(self._graph_btn)

        h.addSpacing(8)

        # Crosshair settings — colour swatch + thickness. Settings
        # persist via AppSettings.
        cross_lbl = QLabel("Cross:")
        cross_lbl.setObjectName("sidecar-footer-summary")
        h.addWidget(cross_lbl)
        self._crosshair_swatch = QPushButton()
        self._crosshair_swatch.setObjectName("crosshair-swatch")
        self._crosshair_swatch.setFixedSize(22, 22)
        self._crosshair_swatch.setToolTip("Crosshair colour — click to change")
        self._crosshair_swatch.clicked.connect(self._pick_crosshair_color)
        self._refresh_crosshair_swatch()
        h.addWidget(self._crosshair_swatch)

        self._crosshair_thickness_spin = QSpinBox()
        self._crosshair_thickness_spin.setRange(1, 5)
        self._crosshair_thickness_spin.setValue(self._crosshair_thickness)
        self._crosshair_thickness_spin.setSuffix(" px")
        self._crosshair_thickness_spin.setFixedWidth(64)
        self._crosshair_thickness_spin.setToolTip(
            "Crosshair line thickness (1–5 px)"
        )
        self._crosshair_thickness_spin.valueChanged.connect(
            self._on_crosshair_thickness_changed
        )
        h.addWidget(self._crosshair_thickness_spin)

        h.addSpacing(12)

        # Slice slider — current orientation depth.
        self._slice_slider, self._slice_val = self._make_slider(
            "Slice", 0, 0,
        )
        h.addLayout(self._wrap_slider("Slice", self._slice_slider, self._slice_val))
        self._slice_slider.valueChanged.connect(self._on_slice_slider_changed)

        # Volume slider — only 4-D data drives this.
        self._vol_slider, self._vol_val = self._make_slider(
            "Volume", 0, 0,
        )
        h.addLayout(self._wrap_slider("Volume", self._vol_slider, self._vol_val))
        self._vol_slider.valueChanged.connect(self._on_vol_slider_changed)

        # Brightness ±1.0 (slider stores ±100 → /100).
        self._bright_slider, _bright_val = self._make_slider(
            "Brightness", -100, 100, default=0, show_value=False,
        )
        h.addLayout(self._wrap_slider("Brightness", self._bright_slider, None))
        self._bright_slider.valueChanged.connect(self._refresh)

        # Contrast 0..2.0 (slider stores 0..200 → /100).
        self._contrast_slider, _contrast_val = self._make_slider(
            "Contrast", 0, 200, default=100, show_value=False,
        )
        h.addLayout(self._wrap_slider("Contrast", self._contrast_slider, None))
        self._contrast_slider.valueChanged.connect(self._refresh)

        h.addStretch(1)
        return bar

    def _make_slider(
        self,
        label: str,
        lo: int,
        hi: int,
        *,
        default: int = 0,
        show_value: bool = True,
    ) -> tuple[QSlider, Optional[QLabel]]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(default)
        slider.setMinimumWidth(80)
        val_label = QLabel(str(default)) if show_value else None
        if val_label is not None:
            val_label.setObjectName("sidecar-footer-summary")
            slider.valueChanged.connect(
                lambda v, lbl=val_label: lbl.setText(str(v))
            )
        return slider, val_label

    @staticmethod
    def _wrap_slider(
        title: str,
        slider: QSlider,
        val_label: Optional[QLabel],
    ) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(1)
        hdr = QLabel(title)
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setObjectName("sidecar-footer-summary")
        box.addWidget(hdr)
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(slider)
        if val_label is not None:
            row.addWidget(val_label)
        box.addLayout(row)
        return box

    # ------------------------------------------------------------------
    # Canvas (single / tri-view + graph panel)
    # ------------------------------------------------------------------

    def _build_canvas(self) -> QWidget:
        canvas = QWidget()
        outer = QVBoxLayout(canvas)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._vsplit = QSplitter(Qt.Orientation.Vertical)
        self._vsplit.setHandleWidth(2)
        self._vsplit.setChildrenCollapsible(False)

        # Top: image area (single-pane OR tri-pane).
        self._image_stack = QStackedWidget()
        self._image_stack.addWidget(self._build_single_image())  # idx 0
        self._image_stack.addWidget(self._build_tri_image())     # idx 1
        self._image_stack.setCurrentIndex(0)
        self._vsplit.addWidget(self._image_stack)

        # Bottom: graph panel for 4-D time-series.
        self._graph_panel = self._build_graph_panel()
        self._graph_panel.setVisible(False)
        self._vsplit.addWidget(self._graph_panel)

        self._vsplit.setStretchFactor(0, 3)
        self._vsplit.setStretchFactor(1, 1)

        outer.addWidget(self._vsplit)
        return canvas

    def _build_loading_panel(self) -> QWidget:
        """Page shown in :attr:`_stack` while the worker is reading."""
        panel = QWidget()
        panel.setObjectName("pane-dark")
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        v.addStretch(1)
        self._loading_spinner = BusySpinner()
        # Stretch the spinner to centre it horizontally — the spinner
        # internally lays its glyph + label in a QHBoxLayout.
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._loading_spinner)
        row.addStretch(1)
        v.addLayout(row)
        self._loading_label = QLabel("")
        self._loading_label.setObjectName("pane-hint")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._loading_label)
        v.addStretch(1)
        return panel

    def _build_single_image(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._image_label = ImageLabel(
            update_fn=lambda: self._render_single_axis(),
            click_fn=lambda ev: self._on_image_clicked(
                ev, self._orientation, self._image_label,
            ),
        )
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored,
        )
        self._image_label.setMinimumSize(1, 1)
        lay.addWidget(self._image_label, 1)
        return w

    def _build_tri_image(self) -> QWidget:
        """Build the three side-by-side panels (sag / cor / ax)."""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(2)
        self._tri_labels: dict[int, ImageLabel] = {}
        for axis in _AXES:
            cell = QWidget()
            cv = QVBoxLayout(cell)
            cv.setContentsMargins(0, 0, 0, 0)
            cv.setSpacing(0)

            caption = QLabel(_AXIS_LABELS[axis])
            caption.setObjectName("sidecar-footer-summary")
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cv.addWidget(caption)

            label = ImageLabel(
                update_fn=lambda a=axis: self._render_axis_into_tri(a),
                click_fn=lambda ev, a=axis: self._on_image_clicked(
                    ev, a, self._tri_labels[a],
                ),
            )
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored,
            )
            label.setMinimumSize(1, 1)
            cv.addWidget(label, 1)
            h.addWidget(cell, 1)
            self._tri_labels[axis] = label
        return w

    def _build_graph_panel(self) -> QWidget:
        """Build the 4-D time-series plot panel (pyqtgraph).

        Layout (port of BIDS-Manager v0.2.5):

        * Top: controls row — ``Scope`` spinbox (1–4 → 1×1, 3×3, 5×5,
          7×7 neighbour grid), ``Dot size`` spinbox (1–20), and a
          ``Mark neighbors`` checkbox that toggles whether the volume
          marker is drawn on every cell or only on the centre voxel.
        * Bottom: a :class:`pyqtgraph.GraphicsLayoutWidget` hosting
          one plot per neighbour voxel. Mouse zoom / pan / wheel are
          disabled on every cell so the plot stays stable — the user
          can't accidentally scroll it into nothing.
        """
        panel = QWidget()
        panel.setObjectName("pane-dark")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        try:
            import pyqtgraph as pg
        except ImportError:  # pragma: no cover - dep listed in pyproject
            self._graph_btn.setVisible(False)
            placeholder = QLabel("pyqtgraph not available")
            placeholder.setObjectName("pane-hint")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(placeholder)
            return panel
        pg.setConfigOptions(antialias=True)

        # --- Controls row (Scope, Dot size, Mark neighbors) ----------
        controls = QHBoxLayout()
        controls.setContentsMargins(8, 0, 8, 0)
        controls.setSpacing(6)

        scope_lbl = QLabel("Scope:")
        scope_lbl.setObjectName("sidecar-footer-summary")
        controls.addWidget(scope_lbl)
        self._scope_spin = QSpinBox()
        self._scope_spin.setRange(1, 4)
        self._scope_spin.setValue(1)
        self._scope_spin.setToolTip(
            "Neighbourhood size around the crosshair voxel.\n"
            "1 = just the voxel · 2 = 3×3 · 3 = 5×5 · 4 = 7×7.\n"
            "Neighbours are taken from the plane perpendicular to "
            "the current orientation."
        )
        self._scope_spin.valueChanged.connect(self._update_graph)
        controls.addWidget(self._scope_spin)

        controls.addSpacing(10)
        dot_lbl = QLabel("Dot size:")
        dot_lbl.setObjectName("sidecar-footer-summary")
        controls.addWidget(dot_lbl)
        self._dot_size_spin = QSpinBox()
        self._dot_size_spin.setRange(1, 20)
        self._dot_size_spin.setValue(8)
        self._dot_size_spin.setToolTip(
            "Diameter of the volume-index marker drawn on each plot."
        )
        self._dot_size_spin.valueChanged.connect(self._update_graph_marker)
        controls.addWidget(self._dot_size_spin)

        controls.addSpacing(12)
        self._mark_neighbors_box = QCheckBox("Mark neighbors")
        self._mark_neighbors_box.setChecked(True)
        self._mark_neighbors_box.setToolTip(
            "When off, only the centre voxel's plot carries the "
            "current-volume marker."
        )
        self._mark_neighbors_box.stateChanged.connect(self._update_graph)
        controls.addWidget(self._mark_neighbors_box)

        controls.addStretch(1)
        lay.addLayout(controls)

        # --- Plot grid ----------------------------------------------
        self._plot_layout = pg.GraphicsLayoutWidget()
        # Disable the GraphicsView's mouse + scroll handlers so the
        # plot doesn't shrink/zoom when the user scrolls over it.
        self._plot_layout.setMouseTracking(False)
        view = self._plot_layout.viewport()
        if view is not None:
            view.setMouseTracking(False)
        # The widget itself has its own wheelEvent — swallow it so
        # the scroll doesn't propagate to any embedded viewbox.
        self._plot_layout.wheelEvent = lambda *_args, **_kw: None
        lay.addWidget(self._plot_layout, 1)

        self._apply_plot_palette()
        return panel

    def _apply_plot_palette(self) -> None:
        """Push the current Qt palette into every pyqtgraph plot cell."""
        if self._plot_layout is None:
            return
        try:
            import pyqtgraph as pg
        except ImportError:  # pragma: no cover
            return
        bg = self.palette().color(QPalette.ColorRole.Base)
        fg = self.palette().color(QPalette.ColorRole.Text)
        self._plot_layout.setBackground(bg)
        marker_brush = pg.mkBrush(self._crosshair_color)
        marker_pen = pg.mkPen(self._crosshair_color)
        curve_pen = pg.mkPen(fg, width=1.5)
        for cell in self._grid_cells:
            curve = cell.get("curve")
            if curve is not None:
                curve.setPen(curve_pen)
            marker = cell.get("marker")
            if marker is not None:
                marker.setBrush(marker_brush)
                marker.setPen(marker_pen)
            plot = cell.get("plot")
            if plot is not None:
                for axis_name in ("left", "bottom"):
                    ax = plot.getAxis(axis_name)
                    ax.setPen(fg)
                    ax.setTextPen(fg)

    # ------------------------------------------------------------------
    # Toggle handlers
    # ------------------------------------------------------------------

    def _on_tri_toggled(self, checked: bool) -> None:
        self._tri_view = checked
        # The orientation pills + slice slider only steer the
        # single-pane view. Greying them out (rather than hiding) keeps
        # the toolbar layout stable across toggles.
        for btn in (self._sa_btn, self._co_btn, self._ax_btn):
            btn.setEnabled(not checked)
        self._slice_slider.setEnabled(not checked and self._data is not None)
        self._image_stack.setCurrentIndex(1 if checked else 0)
        if self._data is not None:
            self._refresh()

    def _on_graph_toggled(self, checked: bool) -> None:
        self._graph_visible = checked
        self._graph_panel.setVisible(checked)
        if checked and self._data is not None:
            self._update_graph()

    # ------------------------------------------------------------------
    # Crosshair config
    # ------------------------------------------------------------------

    def _load_persisted_crosshair(self) -> None:
        """Pull crosshair color + thickness from AppSettings."""
        try:
            from ..app_settings import AppSettings
        except Exception:  # pragma: no cover - settings module always present
            return
        s = AppSettings.load()
        candidate = QColor(s.nifti_crosshair_color)
        if candidate.isValid():
            self._crosshair_color = candidate
        self._crosshair_thickness = max(
            1, min(int(s.nifti_crosshair_thickness or 1), 5),
        )

    def _refresh_crosshair_swatch(self) -> None:
        """Repaint the swatch button to the current colour."""
        if not hasattr(self, "_crosshair_swatch"):
            return
        col = self._crosshair_color.name()
        self._crosshair_swatch.setStyleSheet(
            f"QPushButton#crosshair-swatch {{"
            f"  background: {col};"
            f"  border: 1px solid rgba(0, 0, 0, 0.4);"
            f"  border-radius: 3px;"
            f"}}"
            f"QPushButton#crosshair-swatch:hover {{"
            f"  border: 1px solid {col};"
            f"}}"
        )

    def _pick_crosshair_color(self) -> None:
        dlg = QColorDialog(self._crosshair_color, self)
        dlg.setWindowTitle("Crosshair colour")
        if dlg.exec() == QColorDialog.DialogCode.Accepted:
            picked = dlg.currentColor()
            if picked.isValid():
                self._crosshair_color = picked
                self._refresh_crosshair_swatch()
                self._persist_crosshair()
                self._refresh_after_crosshair_change()

    def _on_crosshair_thickness_changed(self, value: int) -> None:
        self._crosshair_thickness = max(1, min(int(value), 5))
        self._persist_crosshair()
        self._refresh_after_crosshair_change()

    def _persist_crosshair(self) -> None:
        try:
            from ..app_settings import AppSettings
        except Exception:  # pragma: no cover
            return
        AppSettings.remember_nifti_crosshair(
            self._crosshair_color.name(), self._crosshair_thickness,
        )

    def _refresh_after_crosshair_change(self) -> None:
        """Re-render the slice(s) + graph markers with the new style."""
        if self._data is None:
            return
        self._refresh()
        # Push the new colour into the pyqtgraph plot too — the
        # markers track the crosshair colour by design.
        self._apply_plot_palette()

    def _on_slice_slider_changed(self, value: int) -> None:
        """Slider sets the crosshair component along the active axis."""
        self._slice_val.setText(str(value))
        if self._data is None or self._cross_voxel is None or self._tri_view:
            return
        self._cross_voxel[self._orientation] = value
        self._refresh()

    def _on_vol_slider_changed(self, value: int) -> None:
        self._vol_val.setText(str(value))
        self._refresh()
        # In graph mode the y-data doesn't change with volume — only
        # the marker moves. Avoid the full curve redraw.
        if self._graph_visible:
            self._update_graph_marker()

    # ------------------------------------------------------------------
    # Slice rendering
    # ------------------------------------------------------------------

    def _set_orientation(self, axis: int, *, refresh: bool = True) -> None:
        self._orientation = axis
        self._sa_btn.setChecked(axis == _AXIS_SAGITTAL)
        self._co_btn.setChecked(axis == _AXIS_CORONAL)
        self._ax_btn.setChecked(axis == _AXIS_AXIAL)
        if self._data is None:
            return
        vol = self._current_volume()
        axis_len = vol.shape[axis] if axis < vol.ndim else 0
        self._slice_slider.blockSignals(True)
        try:
            self._slice_slider.setMaximum(max(axis_len - 1, 0))
            self._slice_slider.setEnabled(
                axis_len > 1 and not self._tri_view
            )
            # Reflect the crosshair's position on the slider.
            if self._cross_voxel is not None:
                self._slice_slider.setValue(self._cross_voxel[axis])
                self._slice_val.setText(str(self._cross_voxel[axis]))
        finally:
            self._slice_slider.blockSignals(False)
        if refresh:
            self._refresh()

    def _current_volume(self) -> np.ndarray:
        """Return the 3-D volume to render (slices into 4-D data)."""
        assert self._data is not None
        if self._data.ndim == 4 and not self._is_rgb:
            vol_idx = max(
                0, min(self._vol_slider.value(), self._data.shape[3] - 1)
            )
            return self._data[..., vol_idx]
        return self._data

    def _refresh(self) -> None:
        """Repaint whichever canvas is currently visible."""
        if self._data is None or self._cross_voxel is None:
            return
        if self._tri_view:
            for axis in _AXES:
                self._render_axis_into_tri(axis)
        else:
            self._render_single_axis()
        self._update_voxel_value()

    def _render_single_axis(self) -> None:
        if self._data is None or self._cross_voxel is None:
            return
        self._render_axis_to_label(self._orientation, self._image_label)

    def _render_axis_into_tri(self, axis: int) -> None:
        if self._data is None or self._cross_voxel is None:
            return
        label = self._tri_labels.get(axis)
        if label is None:
            return
        self._render_axis_to_label(axis, label)

    def _render_axis_to_label(self, axis: int, label: ImageLabel) -> None:
        """Render the slice along ``axis`` (using ``_cross_voxel`` as
        the slice index) into ``label``."""
        vol = self._current_volume()
        if axis >= vol.ndim:
            return
        slice_idx = self._cross_voxel[axis]
        slice_idx = max(0, min(slice_idx, vol.shape[axis] - 1))

        if axis == _AXIS_SAGITTAL:
            slice_img = vol[slice_idx, :, :]
        elif axis == _AXIS_CORONAL:
            slice_img = vol[:, slice_idx, :]
        else:
            slice_img = vol[:, :, slice_idx]

        arr = slice_img.astype(np.float32)
        arr = arr - arr.min()
        peak = arr.max()
        if peak > 0:
            arr = arr / peak

        b = self._bright_slider.value() / 100.0
        c = self._contrast_slider.value() / 100.0
        arr = (arr - 0.5) * c + 0.5 + b
        arr = np.clip(arr, 0, 1)

        arr = (arr * 255).astype(np.uint8)
        arr = np.rot90(arr)

        if arr.ndim == 2:
            h, w = arr.shape
            img = QImage(
                arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8,
            )
        else:
            h, w, c_chan = arr.shape
            fmt = (
                QImage.Format.Format_RGB888 if c_chan == 3
                else QImage.Format.Format_RGBA8888
            )
            img = QImage(arr.tobytes(), w, h, w * c_chan, fmt)

        pix = QPixmap.fromImage(img)
        target = label.size()
        if target.width() < 2 or target.height() < 2:
            scaled = pix
        else:
            scaled = pix.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        if w > 0 and h > 0:
            self._img_scale[axis] = (scaled.width() / w, scaled.height() / h)
        else:
            self._img_scale[axis] = (1.0, 1.0)

        # Crosshair overlay. The user-configured colour + thickness
        # drive both the line and the centre marker square. A faint
        # dark halo is added only when thickness >= 2 — at width 1 the
        # halo would dominate and make the cross look thicker than
        # the user asked for.
        if not scaled.isNull():
            x_rot, y_rot = self._voxel_to_arr(self._cross_voxel, axis)
            sx, sy = self._img_scale[axis]
            x_s = int(x_rot * sx)
            y_s = int(y_rot * sy)
            thickness = max(1, self._crosshair_thickness)
            sq = max(4, int(min(scaled.width(), scaled.height()) * 0.025))
            half = sq // 2

            painter = QPainter(scaled)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

            if thickness >= 2:
                halo = QPen(_CROSSHAIR_HALO)
                halo.setWidth(thickness + 2)
                painter.setPen(halo)
                painter.drawLine(x_s, 0, x_s, scaled.height())
                painter.drawLine(0, y_s, scaled.width(), y_s)
                painter.drawRect(x_s - half, y_s - half, sq, sq)

            pen = QPen(self._crosshair_color)
            pen.setWidth(thickness)
            painter.setPen(pen)
            painter.drawLine(x_s, 0, x_s, scaled.height())
            painter.drawLine(0, y_s, scaled.width(), y_s)
            painter.drawRect(x_s - half, y_s - half, sq, sq)
            painter.end()

        label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Click → voxel mapping
    # ------------------------------------------------------------------

    def _on_image_clicked(
        self, event: QMouseEvent, axis: int, label: ImageLabel,
    ) -> None:
        if self._data is None or self._cross_voxel is None:
            return
        coords = self._label_pos_to_img_coords(
            event.position().toPoint(), axis, label,
        )
        if coords is None:
            return
        voxel = self._arr_to_voxel(coords[0], coords[1], axis)
        if voxel is None:
            return
        # Clamp to data bounds (defensive — clicks near the edge can
        # land 1 px outside the rotated slice).
        clamped = list(voxel)
        for i, dim in enumerate(self._data.shape[:3]):
            clamped[i] = max(0, min(clamped[i], dim - 1))
        self._cross_voxel = clamped
        # Keep the slice slider in sync with the active axis.
        if not self._tri_view:
            self._slice_slider.blockSignals(True)
            try:
                self._slice_slider.setValue(
                    self._cross_voxel[self._orientation]
                )
                self._slice_val.setText(
                    str(self._cross_voxel[self._orientation])
                )
            finally:
                self._slice_slider.blockSignals(False)
        self._refresh()
        if self._graph_visible:
            self._update_graph()

    def _label_pos_to_img_coords(
        self, pos, axis: int, label: ImageLabel,
    ) -> Optional[tuple[int, int]]:
        pix = label.pixmap()
        if pix is None or pix.isNull():
            return None
        pw, ph = pix.width(), pix.height()
        lw, lh = label.width(), label.height()
        off_x = (lw - pw) / 2
        off_y = (lh - ph) / 2
        x = pos.x() - off_x
        y = pos.y() - off_y
        if 0 <= x < pw and 0 <= y < ph:
            sx, sy = self._img_scale[axis]
            if sx <= 0 or sy <= 0:
                return None
            return int(x / sx), int(y / sy)
        return None

    def _arr_to_voxel(
        self, x: int, y: int, axis: int,
    ) -> Optional[tuple[int, int, int]]:
        if self._data is None or self._cross_voxel is None:
            return None
        vol = self._current_volume()
        # Slice index along the clicked axis stays fixed (you can't
        # change depth by clicking inside a 2-D slice). The crosshair
        # moves in the plane.
        i, j, k = self._cross_voxel
        if axis == _AXIS_SAGITTAL:
            j = x
            k = vol.shape[2] - 1 - y
        elif axis == _AXIS_CORONAL:
            i = x
            k = vol.shape[2] - 1 - y
        else:
            i = x
            j = vol.shape[1] - 1 - y
        return i, j, k

    def _voxel_to_arr(self, voxel, axis: int) -> tuple[int, int]:
        i, j, k = voxel
        vol = self._current_volume()
        if axis == _AXIS_SAGITTAL:
            x = j
            y = vol.shape[2] - 1 - k
        elif axis == _AXIS_CORONAL:
            x = i
            y = vol.shape[2] - 1 - k
        else:
            x = i
            y = vol.shape[1] - 1 - j
        return x, y

    # ------------------------------------------------------------------
    # 4-D time-series plot
    # ------------------------------------------------------------------

    def _update_graph(self) -> None:
        """Rebuild the time-series grid from the current crosshair.

        The grid is ``dim × dim`` where ``dim = 2 * (scope - 1) + 1``
        — i.e. scope=1 → 1×1, scope=2 → 3×3, scope=3 → 5×5, scope=4
        → 7×7. Neighbours are offset in the plane perpendicular to
        the current orientation so the grid layout corresponds
        visually to "what you'd see if you zoomed into this slice".

        Out-of-bounds neighbours leave their cell empty (a placeholder
        widget) so the grid stays a clean square.
        """
        if self._plot_layout is None or self._data is None:
            return
        # Reset state.
        self._plot_layout.clear()
        self._grid_cells = []
        if (
            self._cross_voxel is None
            or self._data.ndim != 4
            or self._is_rgb
        ):
            return

        try:
            import pyqtgraph as pg
        except ImportError:  # pragma: no cover
            return

        level = self._scope_spin.value()
        dim = 2 * (level - 1) + 1
        half = dim // 2
        i0, j0, k0 = self._cross_voxel
        orient = self._orientation
        n_vols = self._data.shape[3]
        mark_all = self._mark_neighbors_box.isChecked()

        # First pass — collect every neighbour's time-series so we can
        # set a shared y-range across the grid.
        cells: list[list[Optional[tuple[int, int, int, np.ndarray]]]] = []
        global_min = float("inf")
        global_max = float("-inf")
        for r, di in enumerate(range(-half, half + 1)):
            row_cells: list[Optional[tuple[int, int, int, np.ndarray]]] = []
            for c, dj in enumerate(range(-half, half + 1)):
                i, j, k = i0, j0, k0
                if orient == _AXIS_SAGITTAL:
                    j = j0 + di
                    k = k0 + dj
                elif orient == _AXIS_CORONAL:
                    i = i0 + di
                    k = k0 + dj
                else:
                    i = i0 + di
                    j = j0 + dj
                if not (
                    0 <= i < self._data.shape[0]
                    and 0 <= j < self._data.shape[1]
                    and 0 <= k < self._data.shape[2]
                ):
                    row_cells.append(None)
                    continue
                ts = np.asarray(self._data[i, j, k, :], dtype=float)
                global_min = min(global_min, float(ts.min()))
                global_max = max(global_max, float(ts.max()))
                row_cells.append((i, j, k, ts))
            cells.append(row_cells)
        if not np.isfinite(global_min) or not np.isfinite(global_max):
            return
        if global_min == global_max:  # constant signal — pad y range
            pad = 1.0 if global_min == 0 else abs(global_min) * 0.05
            global_min -= pad
            global_max += pad

        fg = self.palette().color(QPalette.ColorRole.Text)
        curve_pen = pg.mkPen(fg, width=1.5)
        marker_brush = pg.mkBrush(self._crosshair_color)
        marker_pen = pg.mkPen(self._crosshair_color)
        dot_size = self._dot_size_spin.value()
        vol_idx = self._vol_slider.value()
        vol_idx = max(0, min(vol_idx, n_vols - 1))

        for r in range(dim):
            for c in range(dim):
                cell = cells[r][c]
                if cell is None:
                    # Empty placeholder keeps the grid square.
                    self._plot_layout.addLabel("", row=r, col=c)
                    continue
                i, j, k, ts = cell
                plot = self._plot_layout.addPlot(row=r, col=c)
                plot.setMenuEnabled(False)
                plot.hideButtons()
                vb = plot.getViewBox()
                vb.setMouseEnabled(x=False, y=False)
                vb.setBackgroundColor(None)
                # Disable autorange and pin both axes — the grid is a
                # static visualisation, not an interactive plot.
                vb.disableAutoRange()
                plot.setXRange(0, max(n_vols - 1, 1), padding=0)
                plot.setYRange(global_min, global_max, padding=0.02)
                plot.hideAxis("bottom")
                plot.hideAxis("left")
                is_center = (r == half and c == half)
                curve = plot.plot(
                    np.arange(n_vols), ts, pen=curve_pen,
                )
                marker = None
                if mark_all or is_center:
                    y_val = float(ts[vol_idx])
                    marker = pg.ScatterPlotItem(
                        [vol_idx], [y_val],
                        size=dot_size,
                        brush=marker_brush,
                        pen=marker_pen,
                    )
                    plot.addItem(marker)
                self._grid_cells.append({
                    "plot": plot,
                    "curve": curve,
                    "marker": marker,
                    "ts": ts,
                    "is_center": is_center,
                    "voxel": (i, j, k),
                })

    def _update_graph_marker(self) -> None:
        """Move every grid marker to the current volume index.

        Cheap update path — only the markers move, the curves stay
        put. Reads dot size from the spinbox so resizing the dot
        applies live.
        """
        if not self._grid_cells or self._data is None:
            return
        if self._data.ndim != 4 or self._is_rgb:
            return
        vol_idx = self._vol_slider.value()
        dot_size = self._dot_size_spin.value()
        for cell in self._grid_cells:
            marker = cell.get("marker")
            if marker is None:
                continue
            ts = cell["ts"]
            idx = max(0, min(vol_idx, len(ts) - 1))
            marker.setData(
                [idx], [float(ts[idx])], size=dot_size,
            )

    # ------------------------------------------------------------------
    # Footer / readouts
    # ------------------------------------------------------------------

    def _update_footer(self) -> None:
        path = self._current_file
        root = self._current_root
        if path is None:
            self._footer_path.setText("")
            self._footer_summary.setText("")
            return
        if root is not None:
            try:
                rel = path.resolve().relative_to(root.resolve())
                self._footer_path.setText(str(rel))
            except ValueError:
                self._footer_path.setText(str(path))
        else:
            self._footer_path.setText(str(path))
        if self._data is None:
            self._footer_summary.setText("")
            return
        shape = "×".join(str(s) for s in self._data.shape)
        dtype = str(self._data.dtype)
        flavour = " · RGB" if self._is_rgb else ""
        self._footer_summary.setText(f"{shape} · {dtype}{flavour}")

    def _update_voxel_value(self) -> None:
        if self._data is None or self._cross_voxel is None:
            self._voxel_value.setText("")
            return
        i, j, k = self._cross_voxel
        try:
            if self._is_rgb:
                vec = np.asarray(self._data[i, j, k, :], dtype=float)
                txt = "[" + ", ".join(f"{v:.3g}" for v in vec) + "]"
            elif self._data.ndim == 4:
                val = self._data[i, j, k, self._vol_slider.value()]
                txt = f"{float(val):.3g}"
            else:
                val = self._data[i, j, k]
                txt = f"{float(val):.3g}"
        except (IndexError, ValueError):
            self._voxel_value.setText("")
            return
        self._voxel_value.setText(f"voxel ({i}, {j}, {k}) = {txt}")

    # ------------------------------------------------------------------
    # Reset / teardown
    # ------------------------------------------------------------------

    def _clear(self) -> None:
        self._data = None
        self._img = None
        self._meta = {}
        self._is_rgb = False
        self._cross_voxel = None
        self._image_label.clear()
        for label in getattr(self, "_tri_labels", {}).values():
            label.clear()
        self._slice_slider.setMaximum(0)
        self._slice_slider.setValue(0)
        self._slice_slider.setEnabled(False)
        self._slice_val.setText("0")
        self._vol_slider.setMaximum(0)
        self._vol_slider.setValue(0)
        self._vol_slider.setEnabled(False)
        self._vol_val.setText("0")
        self._graph_btn.setEnabled(False)
        if self._graph_visible:
            self._graph_btn.setChecked(False)
        if self._plot_layout is not None:
            self._plot_layout.clear()
        self._grid_cells = []
        self._toolbar.setVisible(False)
        self._footer.setVisible(False)
        self._footer_path.setText("")
        self._footer_summary.setText("")
        self._voxel_value.setText("")
        self._empty_hint.setText(
            "Select a NIfTI (.nii / .nii.gz) file in the BIDS tree "
            "to view it."
        )
        self._stack.setCurrentIndex(0)


__all__ = ["NiftiViewerPane"]
