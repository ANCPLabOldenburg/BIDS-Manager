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
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QShortcut,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
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

# Anatomical edge labels drawn on each slice: (left, right, top, bottom).
# These are STATIC and correct only because NIfTIs are loaded in RAS
# canonical orientation (``nib.as_closest_canonical`` in ``_load_nifti``),
# so voxel axes always increase toward R, A and S. Derived from the display
# transform in ``_voxel_to_arr`` — if either that or the canonicalisation
# changes, revisit these. (Left/Right follow neurological convention:
# patient-left on the image left.)
_ORIENT_LABELS = {
    _AXIS_SAGITTAL: {"left": "P", "right": "A", "top": "S", "bottom": "I"},
    _AXIS_CORONAL:  {"left": "L", "right": "R", "top": "S", "bottom": "I"},
    _AXIS_AXIAL:    {"left": "L", "right": "R", "top": "A", "bottom": "P"},
}

# For each plane, which canonical axis (0=L/R, 1=A/P, 2=S/I) maps to the
# on-screen horizontal and vertical of the displayed slice. Used to turn a
# per-axis display flip into a slice mirror + label swap. Matches the display
# transform in ``_voxel_to_arr`` / ``_ORIENT_LABELS`` above.
_PLANE_HV = {
    _AXIS_SAGITTAL: (1, 2),   # horizontal = A/P, vertical = S/I
    _AXIS_CORONAL:  (0, 2),   # horizontal = L/R, vertical = S/I
    _AXIS_AXIAL:    (0, 1),   # horizontal = L/R, vertical = A/P
}


def _native_flip_from_affine(affine: "np.ndarray") -> tuple[float, float, float]:
    """Per canonical RAS axis, whether the file stores it reversed (-1) or not.

    Derived from the raw (pre-canonical) affine: if the on-disk data increases
    toward L/P/I instead of R/A/S on some axis, that axis is flagged -1. This
    is what the "native storage orientation" view flips back. Axis permutations
    (e.g. sagittally-stored volumes) are collapsed to their RAS axis, so the
    anatomical L/R·A/P·S/I labels stay correct even for those.
    """
    try:
        import nibabel as nib
        ornt = nib.orientations.io_orientation(affine)  # per raw axis: [ras, flip]
        flip = [1.0, 1.0, 1.0]
        for ras_axis, sign in ornt:
            if not np.isnan(ras_axis):
                flip[int(ras_axis)] = float(sign)
        return (flip[0], flip[1], flip[2])
    except Exception:  # pragma: no cover - defensive
        return (1.0, 1.0, 1.0)

# Default crosshair colour — overridden per-user via AppSettings
# (``nifti_crosshair_color``). Material light-blue 300 reads well on
# both dark and bright slices.
_DEFAULT_CROSSHAIR_COLOR = "#4FC3F7"
_DEFAULT_CROSSHAIR_THICKNESS = 1

# Wheel scrolling: how much raw delta must accumulate before advancing one
# step. ``angleDelta`` is in 1/8-degree units (120 == one mouse-wheel notch);
# ``pixelDelta`` is device pixels (high-resolution trackpads). Separate
# thresholds keep a classic mouse at one-step-per-notch while taming the
# high-frequency stream a trackpad emits. Lower either value to scroll faster.
_WHEEL_ANGLE_STEP = 120.0
_WHEEL_PIXEL_STEP = 40.0
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

    raw = nib.load(str(path))
    native_flip = _native_flip_from_affine(np.asarray(raw.affine))
    img = nib.as_closest_canonical(raw)

    # Structured voxels (colour-FA and friends: dtype [('R','u1'),('G','u1'),
    # ('B','u1')]) cannot go through get_fdata() — it tries to cast the record
    # dtype to float64. Detect them from the DTYPE up front rather than by
    # catching and sniffing the failure: the exception's class and message vary
    # across numpy/nibabel versions (numpy 2.x raises a plain TypeError,
    # "Cannot cast array data from dtype([('R','u1'),...])"), so the old
    # message-matching guard let colour-FA files fail to open.
    if getattr(img.dataobj.dtype, "fields", None) is None:
        return img, img.get_fdata(), {"native_flip": native_flip}

    # Structured / RGB voxels: flatten components into the last axis.
    from numpy.lib import recfunctions as rfn

    dataobj = np.asanyarray(img.dataobj)
    unstructured = rfn.structured_to_unstructured(dataobj)
    vector_length = (
        int(unstructured.shape[-1])
        if unstructured.ndim == dataobj.ndim + 1
        else 1
    )
    meta = {
        "native_flip": native_flip,
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
        # 3-D GPU raycast view. Built lazily on first toggle so users who
        # never open it pay no OpenGL-context cost (and headless test runs
        # stay clear of GL entirely). ``_gl_page_index`` is its slot in
        # ``_image_stack`` once created.
        self._three_d: bool = False
        self._combo_view: bool = False
        self._is_3d_capable: bool = False
        # Whether this host can drive the GPU raycaster (OpenGL 3.3 core on
        # real hardware). When False the 3D / Multi-Planar 3D toggles are hidden
        # entirely — the viewer stays a pure 2-D tool.
        try:
            from .nifti_gl_view import gpu_available
            self._gpu_ok: bool = gpu_available()
        except Exception:  # noqa: BLE001 - never block the 2-D viewer
            self._gpu_ok = False
        # The two 3-D modes ("3D" = full render, "Multi-Planar 3D" = the three
        # planes + render in a grid) share ONE render + ONE control panel, so
        # they are always the SAME view (effect / lighting / clip / camera) —
        # not independent. The single ``_gl`` canvas + ``_gl_controls`` are
        # reparented into whichever page is active (both pages live in the same
        # window, so the GL context is preserved across the move).
        self._gl = None
        self._gl_controls = None
        self._gl_controls_scroll = None
        self._page_3d = None
        self._page_combo = None
        self._gl_slot_3d = None
        self._ctrl_slot_3d = None
        self._gl_slot_combo = None
        self._ctrl_slot_combo = None
        self._gl_page_index: Optional[int] = None
        self._combo_page_index: Optional[int] = None
        # The FIRST volume of the session lands in a default view — Multi-Planar
        # 3-D when a GPU is available, plain Multi-Planar otherwise. Applied
        # once; after that whichever view the user is in persists as they move
        # between scans.
        self._initial_view_applied: bool = False
        self._combo_labels: dict[int, ImageLabel] = {}
        self._combo_edge: dict = {}
        # Global display window (intensity low/high) for 2-D slices, computed
        # once per load so every slice shares one window (crisper + consistent,
        # like MRIcroGL) instead of per-slice min/max stretching.
        self._disp_lo: float = 0.0
        self._disp_hi: float = 1.0
        # Wheel-scroll state. ``_h_key_down`` mirrors the H key: while held,
        # the wheel drives the 4-D volume (time) axis instead of the slice.
        # It's tracked via an app-level event filter (installed at the end of
        # __init__) so it works even when the canvas doesn't hold keyboard
        # focus. ``_wheel_accum`` accumulates sub-notch deltas per scroll
        # target so trackpads don't over-scroll.
        self._h_key_down: bool = False
        self._wheel_accum: dict = {}
        # Anatomical edge labels (L/R·A/P·S/I) drawn around each panel.
        # On by default; toggled by the toolbar button and the "O" shortcut.
        self._show_orient_labels: bool = True
        # Orientation display. ``_ras_on`` (default) shows canonical RAS; when
        # off, the file's raw on-disk orientation is shown (2-D + 3-D + labels).
        # ``_radio_on`` mirrors left/right (radiological convention). Both fold
        # into one per-axis flip (:meth:`_display_flip`). ``_native_flip`` is
        # the raw storage flip discovered at load (identity for RAS files).
        self._ras_on: bool = True
        self._radio_on: bool = False
        self._native_flip: tuple[float, float, float] = (1.0, 1.0, 1.0)
        # Edge-label widgets: ``_single_edge`` for the single pane and one
        # dict per axis in ``_tri_edge`` for Multi view.
        self._single_edge: dict = {}
        self._tri_edge: dict = {}
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

        # Track the H key application-wide so "H + scroll" works over the
        # canvas without it needing keyboard focus. The filter never
        # consumes events — it only mirrors the key state.
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            # Remove the filter before the interpreter tears PyQt down.
            # A QApplication holding an event-filter pointer to this pane
            # during finalisation can leave sip visiting a freed wrapper
            # (SIGBUS in sip_api_visit_wrappers at exit). Qt auto-drops
            # this connection if the pane is destroyed first.
            app.aboutToQuit.connect(self._teardown_event_filter)

        # Single-key shortcuts (A/S/C/M/O). Scoped to the pane's focus,
        # which the canvas grabs on click / scroll.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._install_shortcuts()

        # Eagerly realise the 3-D GL page at construction (when a GPU is
        # present). The pane is built before the main window is shown, so the
        # host window is created render-to-texture-capable from the start.
        # Adding the first QOpenGLWidget to an already-visible window otherwise
        # forces Qt to recreate the native window — the "GUI closes and
        # reopens" the first time 3-D is opened on Windows / Linux.
        if self._gpu_ok:
            try:
                self._ensure_gl()
            except Exception as exc:  # noqa: BLE001 - never block the 2-D viewer
                log.warning("Could not pre-create the 3-D view: %s", exc)

    def _teardown_event_filter(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)

    def eventFilter(self, obj, event):  # noqa: N802 - Qt signature
        et = event.type()
        if et == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_H and not event.isAutoRepeat():
                self._h_key_down = True
        elif et == QEvent.Type.KeyRelease:
            if event.key() == Qt.Key.Key_H and not event.isAutoRepeat():
                self._h_key_down = False
        elif et == QEvent.Type.ApplicationDeactivate:
            # Losing focus can swallow the key-release; reset so H doesn't
            # get stuck "down" after an app switch.
            self._h_key_down = False
        return super().eventFilter(obj, event)

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
        self._native_flip = self._meta.get("native_flip", (1.0, 1.0, 1.0))
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

        # 3D modes — the raycaster takes a scalar 3-D (or 4-D non-RGB) volume,
        # or a colour volume: for RGB the shader uses the vector length as the
        # density (the FA of a colour-FA map) and the hue to tint the surface,
        # like MRIcroGL. 3 or 4 components only; a 5-component "colour" is not
        # something the shader can map.
        self._is_3d_capable = bool(
            data.ndim >= 3
            and (not self._is_rgb or (data.ndim == 4 and data.shape[3] in (3, 4)))
        )

        # Reset brightness / contrast to defaults when a new file is
        # bound so the previous file's settings don't leak.
        self._bright_slider.setValue(0)
        self._contrast_slider.setValue(100)

        self._set_orientation(self._orientation, refresh=False)
        # Re-apply edge labels for this file's flip (native mode differs per
        # file's storage orientation; the fixed-plane panels bake theirs once).
        self._refresh_all_orient_labels()
        self._compute_display_window()
        self._toolbar.setVisible(True)
        self._footer.setVisible(True)
        self._stack.setCurrentWidget(self._canvas)
        self._update_footer()
        self._refresh()
        if self._graph_visible:
            self._update_graph()
        # First real volume of the session opens in the default view; every
        # later scan keeps whatever view the user is currently in.
        if not self._initial_view_applied and data.ndim >= 3:
            self._initial_view_applied = True
            self._set_view_mode(
                "combo" if (self._is_3d_capable and self._gpu_ok) else "multi"
            )

        # If a 3-D mode is open, feed it the new volume; if the new file
        # can't be rendered in 3-D (e.g. RGB), drop back to plain 2-D.
        if (self._three_d or self._combo_view) and not self._is_3d_capable:
            self._set_view_mode("single")
        else:
            self._apply_mode_controls()
            self._push_volume_to_3d()
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
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(14, 6, 14, 6)
        outer.setSpacing(6)
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        outer.addLayout(row1)
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        outer.addLayout(row2)

        # --- Row 1: orientation + view toggles + sliders ----------------

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
            row1.addWidget(btn)
        self._ax_btn.setChecked(True)
        self._sa_btn.setToolTip("Sagittal view.  Shortcut: S")
        self._co_btn.setToolTip("Coronal view.  Shortcut: C")
        self._ax_btn.setToolTip("Axial view.  Shortcut: A")
        self._sa_btn.clicked.connect(
            lambda: self._set_orientation(_AXIS_SAGITTAL)
        )
        self._co_btn.clicked.connect(
            lambda: self._set_orientation(_AXIS_CORONAL)
        )
        self._ax_btn.clicked.connect(
            lambda: self._set_orientation(_AXIS_AXIAL)
        )

        row1.addSpacing(8)

        # Multi-Planar toggle — sagittal+coronal+axial side by side.
        self._tri_btn = QPushButton("Multi-Planar")
        self._tri_btn.setObjectName("tb-btn-toggle")
        self._tri_btn.setCheckable(True)
        self._tri_btn.setToolTip(
            "Show sagittal, coronal and axial panels side by side.\n"
            "The crosshair voxel is shared — clicking any panel moves "
            "it across all three. Scroll over a panel to step its slice.\n"
            "Shortcut: M"
        )
        self._tri_btn.toggled.connect(self._on_tri_toggled)
        row1.addWidget(self._tri_btn)

        # Graph toggle — 4-D time-series at the crosshair voxel.
        self._graph_btn = QPushButton("Graph")
        self._graph_btn.setObjectName("tb-btn-toggle")
        self._graph_btn.setCheckable(True)
        self._graph_btn.setEnabled(False)
        self._graph_btn.setToolTip(
            "Plot the intensity time-course at the crosshair voxel.\n"
            "Available only for 4-D NIfTI files.  Shortcut: G"
        )
        self._graph_btn.toggled.connect(self._on_graph_toggled)
        row1.addWidget(self._graph_btn)

        # 3D toggle — GPU volume raycasting. Enabled only for 3-D (or 4-D
        # non-RGB) data; disabled state is set on load.
        self._td_btn = QPushButton("3D")
        self._td_btn.setObjectName("tb-btn-toggle")
        self._td_btn.setCheckable(True)
        self._td_btn.setEnabled(False)
        self._td_btn.setToolTip(
            "GPU volume rendering (ray-cast 3-D view).\n"
            "Drag to rotate, scroll to zoom. For 4-D data the Volume "
            "slider still selects which volume is rendered.\n"
            "Shortcut: D"
        )
        self._td_btn.toggled.connect(self._on_3d_toggled)
        row1.addWidget(self._td_btn)

        # Multi-Planar 3D — the three 2-D planes plus the GPU render in one grid.
        self._quad_btn = QPushButton("Multi-Planar 3D")
        self._quad_btn.setObjectName("tb-btn-toggle")
        self._quad_btn.setCheckable(True)
        self._quad_btn.setEnabled(False)
        self._quad_btn.setToolTip(
            "Show the three orthogonal slices and the GPU volume render "
            "together in a 2×2 grid.\n"
            "Click / scroll a slice to move the crosshair; drag the render "
            "to rotate it.  Shortcut: P"
        )
        self._quad_btn.toggled.connect(self._on_combo_toggled)
        row1.addWidget(self._quad_btn)

        # No GPU (or no OpenGL 3.3) -> the viewer is 2-D only: hide both
        # 3-D toggles so the options never appear.
        if not self._gpu_ok:
            self._td_btn.setVisible(False)
            self._quad_btn.setVisible(False)

        row1.addSpacing(12)

        # Slice slider — current orientation depth.
        self._slice_slider, self._slice_val = self._make_slider(
            "Slice", 0, 0,
        )
        row1.addLayout(
            self._wrap_slider("Slice", self._slice_slider, self._slice_val)
        )
        self._slice_slider.valueChanged.connect(self._on_slice_slider_changed)

        # Volume slider — only 4-D data drives this.
        self._vol_slider, self._vol_val = self._make_slider(
            "Volume", 0, 0,
        )
        row1.addLayout(
            self._wrap_slider("Volume", self._vol_slider, self._vol_val)
        )
        self._vol_slider.valueChanged.connect(self._on_vol_slider_changed)

        # Brightness ±1.0 (slider stores ±100 → /100).
        self._bright_slider, _bright_val = self._make_slider(
            "Brightness", -100, 100, default=0, show_value=False,
        )
        row1.addLayout(self._wrap_slider("Brightness", self._bright_slider, None))
        self._bright_slider.valueChanged.connect(self._refresh)

        # Contrast 0..2.0 (slider stores 0..200 → /100).
        self._contrast_slider, _contrast_val = self._make_slider(
            "Contrast", 0, 200, default=100, show_value=False,
        )
        row1.addLayout(self._wrap_slider("Contrast", self._contrast_slider, None))
        self._contrast_slider.valueChanged.connect(self._refresh)

        row1.addStretch(1)

        # --- Row 2: display options (labels, crosshair, help) -----------

        # Orientation-label toggle — anatomical L/R·A/P·S/I markers.
        self._labels_btn = QPushButton("Orientation labels")
        self._labels_btn.setObjectName("tb-btn-toggle")
        self._labels_btn.setCheckable(True)
        self._labels_btn.setChecked(self._show_orient_labels)
        self._labels_btn.setToolTip(
            "Show anatomical orientation labels (L/R, A/P, S/I) around "
            "each slice.  Shortcut: O"
        )
        self._labels_btn.toggled.connect(self._on_labels_toggled)
        row2.addWidget(self._labels_btn)

        # RAS toggle — show canonical RAS orientation (on) or the file's raw
        # on-disk orientation (off). Affects the 2-D slices, the 3-D render and
        # the anatomical labels together. No visible change for RAS-stored files.
        self._ras_btn = QPushButton("RAS")
        self._ras_btn.setObjectName("tb-btn-toggle")
        self._ras_btn.setCheckable(True)
        self._ras_btn.setChecked(self._ras_on)
        self._ras_btn.setToolTip(
            "On: show canonical RAS orientation.  Off: show the file's raw "
            "on-disk orientation (2-D slices, 3-D render and labels).  No "
            "visible change for files already stored in RAS."
        )
        self._ras_btn.toggled.connect(self._on_ras_toggled)
        row2.addWidget(self._ras_btn)

        # Radiological — mirror left/right (radiological vs. neurological).
        self._radio_btn = QPushButton("Radiological")
        self._radio_btn.setObjectName("tb-btn-toggle")
        self._radio_btn.setCheckable(True)
        self._radio_btn.setChecked(self._radio_on)
        self._radio_btn.setToolTip(
            "Mirror left/right (radiological convention: patient-left on the "
            "image right).  Off = neurological.  Applies to the 2-D slices, "
            "3-D render and L/R labels."
        )
        self._radio_btn.toggled.connect(self._on_radio_toggled)
        row2.addWidget(self._radio_btn)

        row2.addSpacing(8)

        # Crosshair settings — colour swatch + thickness. Settings
        # persist via AppSettings.
        cross_lbl = QLabel("Cross:")
        cross_lbl.setObjectName("sidecar-footer-summary")
        row2.addWidget(cross_lbl)
        self._crosshair_swatch = QPushButton()
        self._crosshair_swatch.setObjectName("crosshair-swatch")
        self._crosshair_swatch.setFixedSize(22, 22)
        self._crosshair_swatch.setToolTip("Crosshair colour — click to change")
        self._crosshair_swatch.clicked.connect(self._pick_crosshair_color)
        self._refresh_crosshair_swatch()
        row2.addWidget(self._crosshair_swatch)

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
        row2.addWidget(self._crosshair_thickness_spin)

        row2.addStretch(1)

        # Help — keyboard / scroll cheat-sheet popup.
        self._help_btn = QPushButton("Shortcuts")
        self._help_btn.setObjectName("tb-btn-toggle")
        self._help_btn.setToolTip("Keyboard & scroll shortcuts")
        self._help_btn.clicked.connect(self._show_shortcuts_help)
        row2.addWidget(self._help_btn)

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
        # Pure-black visualization area (see #nifti-canvas in theme.qss). The
        # pages inside are plain QWidgets that paint no background of their
        # own, so this shows through behind every view mode.
        self._image_stack.setObjectName("nifti-canvas")
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

    def _make_orientation_overlay(
        self, image_label: ImageLabel,
    ) -> tuple[QWidget, dict]:
        """Wrap ``image_label`` in a grid with four edge labels around it.

        Returns ``(container, edges)`` where ``edges`` maps
        ``top/bottom/left/right`` to the :class:`QLabel` sitting on that
        side *outside* the image. Callers set the glyph text and toggle
        visibility. The image occupies the stretchy centre cell so the
        markers hug its edges as it resizes.
        """
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)

        edges: dict = {}
        for side in ("top", "bottom", "left", "right"):
            lbl = QLabel("")
            lbl.setObjectName("nifti-orient-label")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            font = lbl.font()
            font.setBold(True)
            lbl.setFont(font)
            edges[side] = lbl

        # Fixed width for the L/R markers keeps the image from shifting
        # when the glyph toggles on and off.
        edges["left"].setFixedWidth(16)
        edges["right"].setFixedWidth(16)

        grid.addWidget(edges["top"], 0, 1, Qt.AlignmentFlag.AlignHCenter)
        grid.addWidget(edges["left"], 1, 0, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(image_label, 1, 1)
        grid.addWidget(edges["right"], 1, 2, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(edges["bottom"], 2, 1, Qt.AlignmentFlag.AlignHCenter)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(1, 1)
        return container, edges

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
            wheel_fn=lambda ev: self._handle_wheel(ev, self._orientation),
        )
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored,
        )
        self._image_label.setMinimumSize(1, 1)
        overlay, self._single_edge = self._make_orientation_overlay(
            self._image_label,
        )
        lay.addWidget(overlay, 1)
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
                wheel_fn=lambda ev, a=axis: self._handle_wheel(ev, a),
            )
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored,
            )
            label.setMinimumSize(1, 1)
            overlay, edges = self._make_orientation_overlay(label)
            # Each Multi-view panel shows a fixed plane, so its markers
            # never change — set them once here.
            for side, text in self._display_labels(axis).items():
                edges[side].setText(text)
            self._tri_edge[axis] = edges
            cv.addWidget(overlay, 1)
            h.addWidget(cell, 1)
            self._tri_labels[axis] = label
        self._apply_orient_label_visibility()
        return w

    def _build_combo_page(self):
        """The "Multi-Planar 3D" page: 2×2 grid (3 slices + a render slot) +
        a controls slot on the right. The render + controls slots are filled
        with the SHARED GL widget / controls by :meth:`_mount_gl`, so this view
        and the pure "3D" view are always identical. Returns
        ``(page, gl_slot, ctrl_slot)``.
        """
        page = QWidget()
        outer = QHBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        placements = {
            _AXIS_AXIAL: (0, 0),
            _AXIS_CORONAL: (0, 1),
            _AXIS_SAGITTAL: (1, 0),
        }
        for axis, (r, c) in placements.items():
            cell = QWidget()
            cv = QVBoxLayout(cell)
            cv.setContentsMargins(0, 0, 0, 0)
            cv.setSpacing(0)
            caption = QLabel(_AXIS_LABELS[axis])
            caption.setObjectName("sidecar-footer-summary")
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cv.addWidget(caption)

            label = ImageLabel(
                update_fn=lambda a=axis: self._render_axis_into_combo(a),
                click_fn=lambda ev, a=axis: self._on_image_clicked(
                    ev, a, self._combo_labels[a],
                ),
                wheel_fn=lambda ev, a=axis: self._handle_wheel(ev, a),
            )
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored,
            )
            label.setMinimumSize(1, 1)
            overlay, edges = self._make_orientation_overlay(label)
            for side, text in self._display_labels(axis).items():
                edges[side].setText(text)
            self._combo_edge[axis] = edges
            cv.addWidget(overlay, 1)
            grid.addWidget(cell, r, c)
            self._combo_labels[axis] = label

        gl_slot = self._slot()          # the shared render is mounted here
        grid.addWidget(gl_slot, 1, 1)
        for i in range(2):
            grid.setRowStretch(i, 1)
            grid.setColumnStretch(i, 1)
        outer.addWidget(grid_host, 1)

        # Opaque (see #nifti-ctrl-slot): the canvas behind is black.
        ctrl_slot = self._slot("nifti-ctrl-slot")   # shared controls mount here
        ctrl_slot.setFixedWidth(226)
        outer.addWidget(ctrl_slot)

        self._apply_orient_label_visibility()
        return page, gl_slot, ctrl_slot

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
        self._set_view_mode("multi" if checked else "single")

    def _on_graph_toggled(self, checked: bool) -> None:
        self._graph_visible = checked
        self._graph_panel.setVisible(checked)
        if checked:
            # Start compact: the image keeps the bulk of the height and the
            # time-series gets ~30%. The splitter stays draggable, so users
            # can still make the plot bigger or smaller afterwards.
            total = self._vsplit.height() or 600
            self._vsplit.setSizes([int(total * 0.7), int(total * 0.3)])
            if self._data is not None:
                self._update_graph()

    # ------------------------------------------------------------------
    # 3-D GPU raycast view + view-mode switching
    # ------------------------------------------------------------------
    #
    # There are four mutually-exclusive "big view" modes, all routed through
    # :meth:`_set_view_mode`:
    #   single  — one 2-D slice (the default)
    #   multi   — three 2-D slices side by side (the "Multi view" toggle)
    #   3d      — the GPU volume render only (the "3D" toggle)
    #   combo   — three slices + the render in a 2×2 grid ("Ortho 3D")
    # Centralising the switch keeps the three checkable buttons in sync
    # without the pairwise-exclusion tangle (each toggle slot just names the
    # target mode; button states are re-synced here with signals blocked).

    def _on_3d_toggled(self, checked: bool) -> None:
        self._set_view_mode("3d" if checked else "single")

    def _on_combo_toggled(self, checked: bool) -> None:
        self._set_view_mode("combo" if checked else "single")

    def _set_view_mode(self, mode: str) -> None:
        # Build the shared GL render/controls on demand; if that fails (no GL
        # driver / headless), fall back to plain single-pane 2-D. Then mount
        # the shared render into the active 3-D page so both 3-D modes show the
        # very same view.
        if mode in ("3d", "combo"):
            try:
                self._ensure_gl()
                self._mount_gl(mode)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not open the 3-D view: %s", exc)
                mode = "single"

        self._tri_view = mode == "multi"
        self._three_d = mode == "3d"
        self._combo_view = mode == "combo"

        # Re-sync the three toggle buttons without re-entering their slots.
        for btn, on in (
            (self._tri_btn, self._tri_view),
            (self._td_btn, self._three_d),
            (self._quad_btn, self._combo_view),
        ):
            if btn.isChecked() != on:
                was = btn.blockSignals(True)
                btn.setChecked(on)
                btn.blockSignals(was)

        self._apply_mode_controls()

        if self._three_d and self._gl_page_index is not None:
            self._image_stack.setCurrentIndex(self._gl_page_index)
        elif self._combo_view and self._combo_page_index is not None:
            self._image_stack.setCurrentIndex(self._combo_page_index)
        elif self._tri_view:
            self._image_stack.setCurrentIndex(1)
        else:
            self._image_stack.setCurrentIndex(0)

        if self._data is not None:
            self._refresh()
            self._push_volume_to_3d()

    def _apply_mode_controls(self) -> None:
        """Enable/disable toolbar controls for the current view mode.

        * orientation pills + slice slider — single-pane 2-D only.
        * brightness / contrast / crosshair / labels — any mode that shows
          slices (single / multi / combo), i.e. not the pure 3-D render.
        * graph — 4-D and a slice is shown.
        * the three view-mode toggles stay live whenever their data
          precondition holds, so the user can switch straight between them.
        """
        has_data = self._data is not None
        multi_like = self._tri_view or self._combo_view
        slices_shown = has_data and not self._three_d
        is_4d = has_data and self._data.ndim == 4 and not self._is_rgb

        single_2d = has_data and not self._three_d and not multi_like
        for btn in (self._sa_btn, self._co_btn, self._ax_btn):
            btn.setEnabled(single_2d)
        if single_2d:
            self._slice_slider.setEnabled(self._slice_slider.maximum() > 0)
        else:
            self._slice_slider.setEnabled(False)

        for w in (self._bright_slider, self._contrast_slider,
                  self._crosshair_swatch, self._crosshair_thickness_spin):
            w.setEnabled(slices_shown)
        # The orientation-labels toggle also drives the 3-D orientation cube,
        # so it stays live in every mode once a volume is loaded.
        self._labels_btn.setEnabled(has_data)
        self._graph_btn.setEnabled(slices_shown and is_4d)

        self._tri_btn.setEnabled(has_data)
        self._td_btn.setEnabled(has_data and self._is_3d_capable and self._gpu_ok)
        self._quad_btn.setEnabled(has_data and self._is_3d_capable and self._gpu_ok)
        # RAS / Radiological reorient the 2-D slices and labels too, so they are
        # live whenever a volume is loaded (independent of the GPU 3-D views).
        self._ras_btn.setEnabled(has_data)
        self._radio_btn.setEnabled(has_data)

    def _toggle_3d(self) -> None:
        """'D' shortcut — flip the 3D button when it applies."""
        if self._td_btn.isEnabled():
            self._td_btn.toggle()

    def _toggle_combo(self) -> None:
        """'P' shortcut — flip the Multi-Planar 3D button when it applies."""
        if self._quad_btn.isEnabled():
            self._quad_btn.toggle()

    # -- 3-D slicer keyboard shortcuts (Shift + …) -------------------------

    def _slicer_active(self) -> bool:
        """Slicer shortcuts apply only while a 3-D render is on screen."""
        return (self._three_d or self._combo_view) and self._gl_controls is not None

    def _kbd_toggle_slicer(self) -> None:
        """Shift+Z — activate / deactivate the clip-plane slicer.

        While the slicer is on, Shift+drag orients the cut (azimuth/elevation)
        and Shift+scroll moves it through the volume.
        """
        if self._slicer_active():
            self._gl_controls.kbd_toggle_clip()

    def _kbd_clip_axial(self) -> None:
        """Shift+A — axial cut (plane normal along S–I)."""
        if self._slicer_active():
            self._gl_controls.kbd_set_axis(0, 90)

    def _kbd_clip_sagittal(self) -> None:
        """Shift+S — sagittal cut (plane normal along L–R)."""
        if self._slicer_active():
            self._gl_controls.kbd_set_axis(90, 0)

    def _kbd_clip_coronal(self) -> None:
        """Shift+C — coronal cut (plane normal along A–P)."""
        if self._slicer_active():
            self._gl_controls.kbd_set_axis(0, 0)

    def _kbd_clip_invert(self) -> None:
        """Shift+X — invert the cut direction."""
        if self._slicer_active():
            self._gl_controls.kbd_invert()

    def _ensure_gl(self):
        """Build the shared GL render + controls and both 3-D pages once.

        A single :class:`RaycastGLWidget` + :class:`Nifti3DControls` back BOTH
        the "3D" page and the "Multi-Planar 3D" page; they are reparented into
        whichever page is active (:meth:`_mount_gl`). That keeps the two views
        identical — same effect, lighting, clip and camera — rather than two
        independent renders.
        """
        if self._gl is not None:
            return
        from .nifti_gl_view import Nifti3DControls, RaycastGLWidget
        self._gl = RaycastGLWidget()
        self._gl.set_show_cube(self._show_orient_labels)
        self._gl.set_flip(*self._gl_flip())
        self._gl_controls = Nifti3DControls(self._gl, vertical=True)
        self._gl_controls_scroll = QScrollArea()
        # Name the scroll area AND its viewport so both paint the panel colour.
        # Without this the black image canvas behind shows through the
        # scrollbar gutter as a stripe down the controls column (very visible
        # in the light theme).
        self._gl_controls_scroll.setObjectName("nifti-ctrl-scroll")
        self._gl_controls_scroll.viewport().setObjectName("nifti-ctrl-viewport")
        self._gl_controls_scroll.setWidgetResizable(True)
        self._gl_controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._gl_controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._gl_controls_scroll.setWidget(self._gl_controls)
        self._gl_controls_scroll.setFixedWidth(224)

        (self._page_3d, self._gl_slot_3d,
         self._ctrl_slot_3d) = self._build_3d_page()
        (self._page_combo, self._gl_slot_combo,
         self._ctrl_slot_combo) = self._build_combo_page()
        self._gl_page_index = self._image_stack.addWidget(self._page_3d)
        self._combo_page_index = self._image_stack.addWidget(self._page_combo)
        self._mount_gl("3d")   # park it in the pure-3-D page by default

    @staticmethod
    def _slot(name: str = "") -> QWidget:
        w = QWidget()
        if name:
            w.setObjectName(name)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        return w

    def _mount_gl(self, mode: str) -> None:
        """Reparent the shared render + controls into the active page's slots."""
        if self._gl is None:
            return
        if mode == "combo":
            gl_slot, ctrl_slot = self._gl_slot_combo, self._ctrl_slot_combo
        else:
            gl_slot, ctrl_slot = self._gl_slot_3d, self._ctrl_slot_3d
        # addWidget reparents (auto-removing from the previous slot). Same
        # top-level window on both sides, so the GL context is preserved.
        gl_slot.layout().addWidget(self._gl)
        ctrl_slot.layout().addWidget(self._gl_controls_scroll)

    def _build_3d_page(self):
        """The pure "3D" page: full-width render + right-hand controls slot."""
        page = QWidget()
        page.setObjectName("nifti-canvas")
        h = QHBoxLayout(page)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(2)
        gl_slot = self._slot()
        ctrl_slot = self._slot("nifti-ctrl-slot")
        ctrl_slot.setFixedWidth(226)
        h.addWidget(gl_slot, 1)
        h.addWidget(ctrl_slot)
        return page, gl_slot, ctrl_slot

    def _volume_spacing(self) -> tuple[float, float, float]:
        """Voxel spacing (mm) of the loaded image, defaulting to isotropic."""
        try:
            zooms = self._img.header.get_zooms()[:3]
            return tuple(
                float(z) if z and z > 0 else 1.0 for z in zooms
            )  # type: ignore[return-value]
        except Exception:  # pragma: no cover - header always present in practice
            return (1.0, 1.0, 1.0)

    def _push_volume_to_3d(self) -> None:
        """Send the current (4-D-aware) 3-D volume to the shared GL render."""
        if self._data is None or self._gl is None:
            return
        vol = self._current_volume()
        ndim = getattr(vol, "ndim", 0)
        # Scalar volume, or a colour one (RGB/RGBA) the shader renders tinted.
        ok = ndim == 3 or (
            ndim == 4 and self._is_rgb and vol.shape[3] in (3, 4)
        )
        if not ok:
            return
        self._gl.set_flip(*self._gl_flip())
        self._gl.set_volume_float(vol, self._volume_spacing())

    # ------------------------------------------------------------------
    # Orientation labels
    # ------------------------------------------------------------------

    def _display_flip(self) -> tuple[float, float, float]:
        """Per canonical axis (L/R, A/P, S/I) display flip, folding the RAS
        (native-orientation) and Radiological toggles together."""
        nf = (1.0, 1.0, 1.0) if self._ras_on else self._native_flip
        rx = -1.0 if self._radio_on else 1.0   # radiological mirrors L/R only
        return (nf[0] * rx, nf[1], nf[2])

    def _gl_flip(self) -> tuple[float, float, float]:
        """Flip vector for the 3-D render.

        It carries a constant extra L/R mirror versus the 2-D flip: a 3-D
        look-at view of the volume is the mirror image of a 2-D slice shown in
        the neurological convention, so this keeps the 3-D render's left/right
        aligned with the 2-D slices under both RAS/native and radiological.
        """
        fx, fy, fz = self._display_flip()
        return (-fx, fy, fz)

    def _display_labels(self, axis: int) -> dict:
        """Anatomical edge labels for ``axis`` with the display flip applied."""
        lab = dict(_ORIENT_LABELS[axis])
        h, v = _PLANE_HV[axis]
        f = self._display_flip()
        if f[h] < 0:
            lab["left"], lab["right"] = lab["right"], lab["left"]
        if f[v] < 0:
            lab["top"], lab["bottom"] = lab["bottom"], lab["top"]
        return lab

    def _refresh_all_orient_labels(self) -> None:
        """Re-apply edge labels everywhere (the fixed-plane Multi-Planar and
        combo panels bake theirs once, so a flip toggle must reset them)."""
        self._update_single_orient_labels()
        for axis, edges in self._tri_edge.items():
            for side, text in self._display_labels(axis).items():
                edges[side].setText(text)
        for axis, edges in self._combo_edge.items():
            for side, text in self._display_labels(axis).items():
                edges[side].setText(text)

    def _update_single_orient_labels(self) -> None:
        """Set the single pane's edge glyphs for the current orientation."""
        if not self._single_edge:
            return
        for side, text in self._display_labels(self._orientation).items():
            self._single_edge[side].setText(text)

    def _apply_orient_label_visibility(self) -> None:
        """Show/hide every edge label per :attr:`_show_orient_labels`."""
        groups = (
            [self._single_edge]
            + list(self._tri_edge.values())
            + list(self._combo_edge.values())
        )
        for group in groups:
            for lbl in group.values():
                lbl.setVisible(self._show_orient_labels)
        # The same toggle drives the 3-D orientation cube on the shared render.
        if self._gl is not None:
            self._gl.set_show_cube(self._show_orient_labels)

    def _on_labels_toggled(self, checked: bool) -> None:
        self._show_orient_labels = checked
        self._apply_orient_label_visibility()

    def _on_ras_toggled(self, checked: bool) -> None:
        """RAS on = canonical orientation; off = the file's raw storage order."""
        self._ras_on = checked
        self._apply_display_flip()

    def _on_radio_toggled(self, checked: bool) -> None:
        """Radiological (mirror L/R) vs. neurological convention."""
        self._radio_on = checked
        self._apply_display_flip()

    def _apply_display_flip(self) -> None:
        """Re-render 2-D + labels and re-orient the 3-D render after a toggle."""
        self._refresh_all_orient_labels()
        if self._gl is not None:
            self._gl.set_flip(*self._gl_flip())
        if self._data is not None:
            self._refresh()

    def _toggle_orient_labels(self) -> None:
        """'O' shortcut — flip the Labels button (its slot does the work).

        No-op in 3-D, where anatomical edge labels don't apply.
        """
        if self._labels_btn.isEnabled():
            self._labels_btn.toggle()

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _toggle_multi_view(self) -> None:
        """'M' shortcut — flip the Multi view button when it applies.

        Disabled while the 3-D view owns the canvas; toggling it there
        would yank the view out from under the user.
        """
        if self._tri_btn.isEnabled():
            self._tri_btn.toggle()

    def _toggle_graph(self) -> None:
        """'G' shortcut — flip the Graph button (only when it applies).

        The button is disabled for non-4-D data; toggling programmatically
        would bypass that, so guard on ``isEnabled``.
        """
        if self._graph_btn.isEnabled():
            self._graph_btn.toggle()

    def _shortcut_orientation(self, axis: int) -> None:
        """'A' / 'S' / 'C' shortcuts — jump to a single-pane orientation.

        Leaves any multi-plane / 3-D mode so the requested plane fills the
        pane.
        """
        if self._tri_view or self._three_d or self._combo_view:
            self._set_view_mode("single")
        self._set_orientation(axis)

    def _install_shortcuts(self) -> None:
        """Wire the single-key shortcuts, scoped to the viewer's focus.

        ``WidgetWithChildrenShortcut`` means they only fire while the pane
        (or a child) has focus — grabbed on click/scroll — so typing an
        'a' in a text field elsewhere never switches the plane.
        """
        specs = (
            ("A", lambda: self._shortcut_orientation(_AXIS_AXIAL)),
            ("S", lambda: self._shortcut_orientation(_AXIS_SAGITTAL)),
            ("C", lambda: self._shortcut_orientation(_AXIS_CORONAL)),
            ("M", self._toggle_multi_view),
            ("G", self._toggle_graph),
            ("O", self._toggle_orient_labels),
            ("D", self._toggle_3d),
            ("P", self._toggle_combo),
            # 3-D slicer (clip plane). Shift + drag/scroll while active orients
            # and navigates the cut.
            ("Shift+Z", self._kbd_toggle_slicer),
            ("Shift+A", self._kbd_clip_axial),
            ("Shift+S", self._kbd_clip_sagittal),
            ("Shift+C", self._kbd_clip_coronal),
            ("Shift+X", self._kbd_clip_invert),
        )
        for key, handler in specs:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(handler)

    def _show_shortcuts_help(self) -> None:
        """Pop up a cheat-sheet of mouse/trackpad gestures and key shortcuts.

        The action and the key/gesture live in separate columns so each is
        easy to scan.
        """
        # (Action, Key / gesture) grouped under section headers.
        sections = [
            ("Mouse & trackpad", [
                ("Move to previous / next slice", "Vertical scroll"),
                ("Previous / next volume (4-D time)", "Horizontal scroll"),
                ("Previous / next volume (4-D time)", "Hold H + scroll"),
                ("Move the crosshair", "Click or drag"),
            ]),
            ("Keys", [
                ("Axial view", "A"),
                ("Sagittal view", "S"),
                ("Coronal view", "C"),
                ("Multi-Planar (toggle)", "M"),
                ("3D volume render (toggle)", "D"),
                ("Multi-Planar 3D — planes + render (toggle)", "P"),
                ("Graph / time-series (toggle)", "G"),
                ("Orientation labels (toggle)", "O"),
            ]),
            ("3D view", [
                ("Rotate the volume", "Drag"),
                ("Pan", "Right-drag"),
                ("Zoom in / out", "Scroll"),
            ]),
            ("3D slicer (clip plane)", [
                ("Activate / deactivate slicer", "Shift+Z"),
                ("Orient the cut freely", "Shift+drag"),
                ("Navigate the slice", "Shift+scroll"),
                ("Axial cut", "Shift+A"),
                ("Sagittal cut", "Shift+S"),
                ("Coronal cut", "Shift+C"),
                ("Invert the cut", "Shift+X"),
            ]),
        ]

        html = [
            "<div style='padding:2px 6px'>",
            "<div style='font-size:13px;font-weight:600'>Viewer shortcuts</div>",
            "<div style='color:gray;font-size:11px;margin-bottom:2px'>"
            "Click or scroll the image once to focus it, then the keys "
            "apply here.</div>",
        ]
        for title, rows in sections:
            html.append(
                "<div style='color:gray;font-size:11px;font-weight:600;"
                f"margin-top:8px'>{title}</div>"
            )
            html.append("<table cellspacing='0' cellpadding='3'>")
            for action, key in rows:
                html.append(
                    "<tr>"
                    f"<td style='padding-right:18px'>{action}</td>"
                    f"<td align='right'><b>{key}</b></td>"
                    "</tr>"
                )
            html.append("</table>")
        html.append("</div>")

        menu = QMenu(self)
        try:
            from ..combo_popup import round_menu
            round_menu(menu)
        except Exception:  # pragma: no cover - cosmetic only
            pass
        lbl = QLabel("".join(html))
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setContentsMargins(12, 8, 12, 8)
        action = QWidgetAction(menu)
        action.setDefaultWidget(lbl)
        menu.addAction(action)
        menu.exec(self._help_btn.mapToGlobal(
            self._help_btn.rect().bottomLeft(),
        ))

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
        # Different 4-D volumes can differ in intensity range — re-window.
        self._compute_display_window()
        self._refresh()
        # In graph mode the y-data doesn't change with volume — only
        # the marker moves. Avoid the full curve redraw.
        if self._graph_visible:
            self._update_graph_marker()
        # In a 3-D mode the selected volume is what gets raycast — re-upload.
        if self._three_d or self._combo_view:
            self._push_volume_to_3d()

    # ------------------------------------------------------------------
    # Slice rendering
    # ------------------------------------------------------------------

    def _set_orientation(self, axis: int, *, refresh: bool = True) -> None:
        self._orientation = axis
        self._sa_btn.setChecked(axis == _AXIS_SAGITTAL)
        self._co_btn.setChecked(axis == _AXIS_CORONAL)
        self._ax_btn.setChecked(axis == _AXIS_AXIAL)
        self._update_single_orient_labels()
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

    def _compute_display_window(self) -> None:
        """Set ``_disp_lo/_disp_hi`` from a robust percentile of the volume.

        One window shared by every 2-D slice. Computed from a subsample so a
        several-hundred-MB BOLD run stays instant; RGB data keeps the trivial
        0..1 window (its slices are handled component-wise elsewhere).
        """
        self._disp_lo, self._disp_hi = 0.0, 1.0
        if self._data is None or self._is_rgb:
            return
        vol = self._current_volume()
        flat = np.asarray(vol, dtype=np.float32).ravel()
        if flat.size > 500_000:
            flat = flat[:: flat.size // 500_000]
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return
        lo, hi = np.percentile(finite, (1.0, 99.0))
        if hi <= lo:
            hi = lo + 1.0
        self._disp_lo, self._disp_hi = float(lo), float(hi)

    def _refresh(self) -> None:
        """Repaint whichever canvas is currently visible."""
        if self._data is None or self._cross_voxel is None:
            return
        if self._combo_view:
            for axis in _AXES:
                self._render_axis_into_combo(axis)
        elif self._tri_view:
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

    def _render_axis_into_combo(self, axis: int) -> None:
        if self._data is None or self._cross_voxel is None:
            return
        label = self._combo_labels.get(axis)
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
        if arr.ndim == 2:
            # Global intensity window (computed once per load) so every slice
            # shares one mapping — crisp and consistent, like MRIcroGL —
            # rather than each slice stretching its own min/max (flat, noisy).
            lo, hi = self._disp_lo, self._disp_hi
            arr = (arr - lo) / (hi - lo) if hi > lo else arr - lo
            # Gentle gamma to lift the mid-tones for MRIcroGL-like vibrancy /
            # contrast (brighter grey/white matter without clipping highlights).
            arr = np.clip(arr, 0.0, 1.0) ** 0.8
        else:
            # RGB / colour voxels: components are already display values.
            arr = arr / 255.0 if arr.max() > 1.0 else arr
        arr = np.clip(arr, 0, 1)

        b = self._bright_slider.value() / 100.0
        c = self._contrast_slider.value() / 100.0
        arr = (arr - 0.5) * c + 0.5 + b
        arr = np.clip(arr, 0, 1)

        arr = (arr * 255).astype(np.uint8)
        arr = np.rot90(arr)

        # Mirror the displayed slice to match the orientation flip (RAS/native +
        # radiological). Rows are screen-vertical, columns screen-horizontal.
        f = self._display_flip()
        h_ax, v_ax = _PLANE_HV[axis]
        if f[h_ax] < 0:
            arr = arr[:, ::-1]
        if f[v_ax] < 0:
            arr = arr[::-1, :]
        arr = np.ascontiguousarray(arr)

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
        # Grab focus so the A/S/C/M/O shortcuts target the viewer.
        self.setFocus(Qt.FocusReason.MouseFocusReason)
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
        x, y = self._unflip_arr_coords(x, y, axis, vol)
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

    def _flip_arr_coords(self, x: int, y: int, axis: int, vol) -> tuple[int, int]:
        """Apply the display flip to in-plane (x, y) slice coordinates."""
        f = self._display_flip()
        h_ax, v_ax = _PLANE_HV[axis]
        if f[h_ax] < 0:
            x = (vol.shape[h_ax] - 1) - x
        if f[v_ax] < 0:
            y = (vol.shape[v_ax] - 1) - y
        return x, y

    # The flip is its own inverse, so mapping a click back is the same op.
    _unflip_arr_coords = _flip_arr_coords

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
        return self._flip_arr_coords(x, y, axis, vol)

    # ------------------------------------------------------------------
    # Scroll on image -> step through the slice / volume stack
    # ------------------------------------------------------------------

    def _handle_wheel(self, event: QWheelEvent, axis: int) -> bool:
        """Route a wheel event from the image panel for ``axis``.

        * Plain **vertical** scroll steps the slice along ``axis``.
        * **Horizontal** scroll — or vertical scroll with the **H** key
          held — steps the 4-D volume (time), when the data is 4-D.

        This covers all input styles: a mouse wheel (angleDelta only),
        a trackpad's natural horizontal swipe (pixelDelta.x), and the
        H-key modifier for devices that can't scroll horizontally.
        Returns True when the event was consumed.
        """
        if self._data is None or self._cross_voxel is None:
            return False

        # Focus the viewer so the single-key shortcuts apply here.
        self.setFocus(Qt.FocusReason.MouseFocusReason)

        pd = event.pixelDelta()
        ad = event.angleDelta()
        # Prefer pixelDelta (high-resolution trackpads report it and, on
        # some macOS devices, one scroll direction leaves angleDelta.y at
        # 0 — the root of the "only scrolls one way" bug). Classic mice
        # report only angleDelta.
        use_pixel = not pd.isNull()
        dx = pd.x() if use_pixel else ad.x()
        dy = pd.y() if use_pixel else ad.y()
        thresh = _WHEEL_PIXEL_STEP if use_pixel else _WHEEL_ANGLE_STEP

        is_4d = self._data.ndim == 4 and not self._is_rgb

        # --- Time / volume axis -----------------------------------------
        # H + scroll (either wheel axis), or a horizontal-dominant swipe.
        if is_4d and self._h_key_down:
            steps = self._accum_steps(("vol", axis), dy or dx, thresh)
            if steps:
                # Scroll down / right -> forward in time.
                self._step_volume(-steps)
            return True

        if abs(dx) > abs(dy):
            if is_4d:
                steps = self._accum_steps(("vol", axis), dx, thresh)
                if steps:
                    self._step_volume(steps)  # scroll right -> forward
            # Swallow horizontal even on 3-D so it doesn't pan a parent.
            return True

        # --- Slice axis (plain vertical scroll) -------------------------
        steps = self._accum_steps(("slice", axis), dy, thresh)
        if steps:
            self._step_slice(axis, -steps)  # scroll up -> previous slice
        return True

    def _accum_steps(self, key, delta: float, thresh: float) -> int:
        """Accumulate ``delta`` under ``key`` and return whole steps.

        Sub-threshold movement is banked so slow trackpad scrolls still
        register while fast ones don't skip; the remainder is carried.
        """
        if not delta:
            return 0
        acc = self._wheel_accum.get(key, 0.0) + delta
        steps = int(acc / thresh)  # truncates toward zero for either sign
        self._wheel_accum[key] = acc - steps * thresh
        return steps

    def _step_slice(self, axis: int, delta: int) -> None:
        if not delta or self._cross_voxel is None:
            return
        vol = self._current_volume()
        if axis >= vol.ndim:
            return
        cur = self._cross_voxel[axis]
        new = max(0, min(vol.shape[axis] - 1, cur + delta))
        if new == cur:
            return
        if not self._tri_view and axis == self._orientation:
            # Single-pane: drive the slider; its slot updates the
            # crosshair and repaints.
            self._slice_slider.setValue(new)
        else:
            # Multi view: no per-panel slider — move the shared crosshair
            # voxel directly and repaint all panels.
            self._cross_voxel[axis] = new
            self._refresh()

    def _step_volume(self, delta: int) -> None:
        if not delta or self._data is None:
            return
        if not (self._data.ndim == 4 and not self._is_rgb):
            return
        cur = self._vol_slider.value()
        new = max(0, min(self._vol_slider.maximum(), cur + delta))
        if new != cur:
            # The slider's slot repaints and moves the graph marker.
            self._vol_slider.setValue(new)

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
        # Reset the 3-D modes: drop back to single-pane, disable the toggles,
        # and free the GPU textures of both GL views.
        self._is_3d_capable = False
        self._tri_view = self._three_d = self._combo_view = False
        for btn in (self._tri_btn, self._td_btn, self._quad_btn):
            was = btn.blockSignals(True)
            btn.setChecked(False)
            btn.setEnabled(False)
            btn.blockSignals(was)
        self._image_stack.setCurrentIndex(0)
        if self._gl is not None:
            self._gl.clear()
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
