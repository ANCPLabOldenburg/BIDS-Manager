"""PyQt GUI for converting DICOM data into BIDS and editing metadata."""

import sys
import os
import json
import re
import shutil
import pandas as pd
import numpy as np
from decimal import Decimal, InvalidOperation
try:  # NumPy exposes structured-array helpers from ``numpy.lib``
    from numpy.lib import recfunctions as rfn
except Exception:  # pragma: no cover - extremely old NumPy releases
    rfn = None
try:  # NumPy >= 1.20 provides dedicated exception classes
    from numpy import exceptions as np_exceptions  # type: ignore
except Exception:  # pragma: no cover - fallback for older versions
    np_exceptions = None
import threading
import time
import pydicom  # used to inspect DICOM headers when checking for mixed sessions
try:  # prefer relative import but fall back to direct when running as a script
    from .run_heudiconv_from_heuristic import is_dicom_file  # reuse existing helper
except Exception:  # pragma: no cover - packaging edge cases
    from run_heudiconv_from_heuristic import is_dicom_file  # type: ignore
try:
    import nibabel as nib
except ModuleNotFoundError as exc:
    if exc.name == '_bz2':
        import sys
        import types
        import io
        import subprocess

        class _SubprocessBZ2File(io.BufferedReader):
            """Minimal BZ2File replacement using the external ``bzip2`` binary."""

            def __init__(self, filename, mode="r", buffering=None, compresslevel=9):
                if "r" not in mode:
                    raise NotImplementedError(
                        "Writing not supported without Python bz2 module"
                    )
                proc = subprocess.Popen(
                    ["bzip2", "-dc", filename], stdout=subprocess.PIPE
                )
                if proc.stdout is None:  # pragma: no cover - should not happen
                    raise RuntimeError("Failed to open bzip2 subprocess")
                self._proc = proc
                super().__init__(proc.stdout)

            def close(self):
                try:
                    super().close()
                finally:
                    self._proc.stdout.close()
                    self._proc.wait()

        stub = types.ModuleType("bz2")
        stub.BZ2File = _SubprocessBZ2File
        sys.modules.setdefault("bz2", stub)
        import nibabel as nib
    else:  # pragma: no cover - unrelated import failure
        raise
from pathlib import Path
from collections import defaultdict
from functools import cmp_to_key
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from joblib import Parallel, delayed
from pandas.core.tools.datetimes import guess_datetime_format
from dataclasses import dataclass
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QTableWidgetSelectionRange, QGroupBox, QGridLayout,
    QTextEdit, QTreeView, QFileSystemModel, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QMessageBox, QAction, QSplitter, QDialog, QAbstractItemView,
    QMenuBar, QMenu, QSizePolicy, QComboBox, QSlider, QSpinBox,
    QCheckBox, QStyledItemDelegate, QDialogButtonBox, QListWidget, QScrollArea,
    QToolButton
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import (
    Qt,
    QModelIndex,
    QTimer,
    QProcess,
    QUrl,
    QRect,
    QPoint,
    QObject,
    QThread,
    QSize,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import (
    QPalette,
    QColor,
    QFont,
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QIcon,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import logging  # debug logging
import signal
import random
import string
import math
from bids_manager import dicom_inventory
from bids_manager.schema_renamer import (
    DEFAULT_SCHEMA_DIR,
    DERIVATIVES_PIPELINE_NAME,
    ENABLE_DWI_DERIVATIVES_MOVE,
    ENABLE_FIELDMap_NORMALIZATION,
    ENABLE_SCHEMA_RENAMER,
    SeriesInfo,
    apply_post_conversion_rename,
    build_preview_names,
    load_bids_schema,
)
from bids_manager.schema_renamer import normalize_study_name
try:
    import psutil
    HAS_PSUTIL = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PSUTIL = False

try:  # Surface reconstruction for the 3-D viewer (optional dependency)
    from skimage import measure as sk_measure
    HAS_SKIMAGE = True
except Exception:  # pragma: no cover - optional dependency
    sk_measure = None
    HAS_SKIMAGE = False

try:  # Hardware-accelerated 3-D rendering for volumes and surfaces
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import shaders as gl_shaders
    HAS_PYQTGRAPH = True
except Exception:  # pragma: no cover - optional dependency
    pg = None
    gl = None
    gl_shaders = None
    HAS_PYQTGRAPH = False

# Paths to images bundled with the application
LOGO_FILE = Path(__file__).resolve().parent / "miscellaneous" / "images" / "Logo.png"
ICON_FILE = Path(__file__).resolve().parent / "miscellaneous" / "images" / "Icon.png"
ANCP_LAB_FILE = Path(__file__).resolve().parent / "miscellaneous" / "images" / "ANCP_lab.png"
KAREL_IMG_FILE = Path(__file__).resolve().parent / "miscellaneous" / "images" / "Karel.jpeg"
JOCHEM_IMG_FILE = Path(__file__).resolve().parent / "miscellaneous" / "images" / "Jochem.jpg"

# Directory used to store persistent user preferences
PREF_DIR = Path(__file__).resolve().parent / "user_preferences"


class _ConflictScannerWorker(QObject):
    """Run the conflict detection scan in a background thread."""

    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, root_dir: str, finder: Callable[[str, int], dict], n_jobs: int):
        super().__init__()
        self._root_dir = root_dir
        self._finder = finder
        # ``n_jobs`` mirrors the CPU limit used for the main scanning step so we
        # do not overwhelm the system when running both operations back-to-back.
        self._n_jobs = max(1, n_jobs)

    @pyqtSlot()
    def run(self) -> None:
        """Execute the slow directory walk outside the GUI thread."""

        try:
            conflicts = self._finder(self._root_dir, self._n_jobs)
        except Exception as exc:  # pragma: no cover - runtime safety
            # Forward the error message back to the GUI thread so the caller
            # can decide how to handle it without freezing the interface.
            self.failed.emit(str(exc))
        else:
            self.finished.emit(conflicts)


def _create_directional_light_shader():
    """Build a lightweight directional lighting shader for ``GLMeshItem``.

    PyQtGraph ships with a fixed light direction inside the built-in
    ``'shaded'`` program which makes the surface difficult to interpret when the
    user wants to change the perceived light source.  Creating our own shader
    allows us to expose the light direction and intensity as uniforms that can
    be updated on the fly without recompiling the OpenGL program each time the
    sliders move.
    """

    if not HAS_PYQTGRAPH or gl is None:
        return None

    shader_mod = getattr(gl, "shaders", None)
    if shader_mod is None:
        return None

    try:
        vertex = shader_mod.VertexShader(
            """
            varying vec3 normal;
            void main() {
                normal = normalize(gl_NormalMatrix * gl_Normal);
                gl_FrontColor = gl_Color;
                gl_BackColor = gl_Color;
                gl_Position = ftransform();
            }
            """
        )
        fragment = shader_mod.FragmentShader(
            """
            uniform float lightDir[3];
            uniform float lightParams[2];
            varying vec3 normal;
            void main() {
                vec3 norm = normalize(normal);
                vec3 lightVec = normalize(vec3(lightDir[0], lightDir[1], lightDir[2]));
                float diffuse = max(dot(norm, lightVec), 0.0) * lightParams[0];
                float ambient = lightParams[1];
                float lighting = clamp(ambient + diffuse, 0.0, 1.0);
                vec4 colour = gl_Color;
                colour.rgb *= lighting;
                gl_FragColor = colour;
            }
            """
        )
    except Exception:  # pragma: no cover - shader compilation errors are runtime only
        return None

    return shader_mod.ShaderProgram(
        None,
        [vertex, fragment],
        uniforms={
            "lightDir": [0.0, 0.0, 1.0],
            # ``lightParams`` stores [diffuse_scale, ambient_strength].  We keep
            # a modest ambient component so that the mesh never becomes entirely
            # black when the light points away from the surface.
            "lightParams": [1.0, 0.35],
        },
    )


def _create_flat_color_shader():
    """Return an unlit shader for meshes when lighting is disabled."""

    if not HAS_PYQTGRAPH or gl is None:
        return None

    shader_mod = getattr(gl, "shaders", None)
    if shader_mod is None:
        return None

    try:
        vertex = shader_mod.VertexShader(
            """
            void main() {
                gl_FrontColor = gl_Color;
                gl_BackColor = gl_Color;
                gl_Position = ftransform();
            }
            """
        )
        fragment = shader_mod.FragmentShader(
            """
            void main() {
                gl_FragColor = gl_Color;
            }
            """
        )
    except Exception:  # pragma: no cover - shader compilation errors occur at runtime
        return None

    return shader_mod.ShaderProgram(None, [vertex, fragment], uniforms={})


_SLICE_ORIENTATIONS = (
    ("sagittal", 0, "Left", "Right"),
    ("coronal", 1, "Posterior", "Anterior"),
    ("axial", 2, "Inferior", "Superior"),
)


if HAS_PYQTGRAPH:

    class _AdjustableAxisItem(gl.GLAxisItem):
        """Axis item with configurable line width for better visibility."""

        def __init__(self, *args, **kwargs):
            self._line_width = 2.0
            super().__init__(*args, **kwargs)

        def setLineWidth(self, width: float) -> None:
            self._line_width = max(1.0, float(width))
            self.update()

        def paint(self):  # pragma: no cover - requires OpenGL context
            from OpenGL.GL import (
                GL_LINES,
                GL_LINE_SMOOTH,
                GL_LINE_SMOOTH_HINT,
                GL_NICEST,
                glBegin,
                glColor4f,
                glEnable,
                glEnd,
                glHint,
                glLineWidth,
                glVertex3f,
            )

            self.setupGLState()
            if self.antialias:
                glEnable(GL_LINE_SMOOTH)
                glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            glLineWidth(self._line_width)
            glBegin(GL_LINES)
            x, y, z = self.size()
            glColor4f(0, 1, 0, 0.6)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, z)
            glColor4f(1, 1, 0, 0.6)
            glVertex3f(0, 0, 0)
            glVertex3f(0, y, 0)
            glColor4f(0, 0, 1, 0.6)
            glVertex3f(0, 0, 0)
            glVertex3f(x, 0, 0)
            glEnd()


@dataclass
class _SliceControl:
    """Stores widgets controlling a single anatomical slicer."""

    checkbox: QCheckBox
    min_slider: QSlider
    max_slider: QSlider
    min_value: QLabel
    max_value: QLabel
    axis: int
    negative_name: str
    positive_name: str


class _AutoUpdateLabel(QLabel):
    """QLabel that triggers a callback whenever it is resized."""

    def __init__(self, update_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_fn = update_fn

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if callable(self._update_fn):
            self._update_fn()


class _ImageLabel(_AutoUpdateLabel):
    """Label that notifies on resize and mouse clicks."""

    def __init__(self, update_fn, click_fn, *args, **kwargs):
        super().__init__(update_fn, *args, **kwargs)
        self._click_fn = click_fn

    def mousePressEvent(self, event):
        if callable(self._click_fn):
            self._click_fn(event)
        super().mousePressEvent(event)

def _extract_subject(row) -> str:
    """Return subject identifier prioritising ``BIDS_name`` and stripping ``sub-``."""
    subj = str(row.get("BIDS_name") or row.get("subject") or row.get("sub") or "UNK")
    if subj.lower().startswith("sub-"):
        subj = subj[4:]
    return subj

def _compute_bids_preview(df, schema):
    """Returns a dict {row_index: (datatype, basename)} for preview; safe if schema is None."""
    out = {}
    if not schema:
        return out
    rows = []
    idxs = []
    for i, row in df.iterrows():
        subject = _extract_subject(row)
        session = row.get("session") or row.get("ses") or None
        modality = str(row.get("modality") or row.get("fine_modality") or row.get("BIDS_modality") or "")
        sequence = str(row.get("sequence") or row.get("SeriesDescription") or "")
        rep = row.get("rep") or row.get("repeat") or 1

        extra = {}
        for key in ("task", "task_hits", "acq", "run", "dir", "echo"):
            if row.get(key):
                extra[key] = str(row.get(key))

        rows.append(SeriesInfo(subject, session, modality, sequence, int(rep or 1), extra))
        idxs.append(i)

    proposals = build_preview_names(rows, schema)
    for (series, dt, base), idx in zip(proposals, idxs):
        out[idx] = (dt, base)
    return out

# ---- basic logging config ----
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def _terminate_process_tree(pid: int):
    """Terminate a process and all of its children without killing the GUI."""
    # Protect against invalid PIDs which may occur if a process fails to start
    if pid <= 0:
        return
    # Try killing the process group only if it's not the same as the GUI's
    # to avoid terminating the entire application or IDE.
    try:
        pgid = os.getpgid(pid)
        if pgid != os.getpgid(0):
            os.killpg(pgid, signal.SIGTERM)
            return
    except Exception:
        pass
    if HAS_PSUTIL:
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(children, timeout=3)
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            pass
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

def _get_ext(path: Path) -> str:
    """Return file extension with special handling for .nii.gz."""
    name = path.name.lower()
    if name.endswith('.nii.gz'):
        return '.nii.gz'
    return path.suffix.lower()


def _style_axes_3d(axis, fg_color: str, axis_bg: str) -> None:
    """Apply consistent styling to Matplotlib 3-D axes."""

    axis.set_facecolor(axis_bg)
    base = axis.get_facecolor()
    axis.xaxis.set_pane_color((*base[:3], 0.15))
    axis.yaxis.set_pane_color((*base[:3], 0.15))
    axis.zaxis.set_pane_color((*base[:3], 0.15))
    for ax in (axis.xaxis, axis.yaxis, axis.zaxis):
        ax.label.set_color(fg_color)
        ax.set_tick_params(colors=fg_color)
    axis.grid(False)


GIFTI_SURFACE_SUFFIXES = (".surf.gii", ".surf.gii.gz", ".gii", ".gii.gz")
FREESURFER_SURFACE_SUFFIXES = (
    ".pial",
    ".pial.gz",
    ".white",
    ".white.gz",
    ".inflated",
    ".inflated.gz",
    ".sphere",
    ".sphere.gz",
    ".orig",
    ".orig.gz",
    ".smoothwm",
    ".smoothwm.gz",
    ".midthickness",
    ".midthickness.gz",
)


def _dedup_parts(*parts: str) -> str:
    """Return underscore-joined parts with consecutive repeats removed."""
    # ``parts`` may contain elements that themselves contain underscores.  The
    # goal is to produce a clean path component without duplicate separators.
    tokens: list[str] = []
    for part in parts:
        # Split each piece on underscores so ``seq__name`` becomes ["seq", "name"]
        for t in str(part).split('_'):
            # Only keep tokens that are not a repeat of the previous one
            if t and (not tokens or t != tokens[-1]):
                tokens.append(t)
    return "_".join(tokens)


def _safe_stem(text: str) -> str:
    """Return filename-friendly version of ``text`` used for study folders."""

    cleaned = normalize_study_name(text)
    return re.sub(r"[^0-9A-Za-z_-]+", "_", cleaned).strip("_")


def _format_subject_id(num: int) -> str:
    """Return ID as three letters followed by three digits."""
    letters_idx = (num - 1) // 1000
    digits = (num - 1) % 1000 + 1
    letters = []
    for _ in range(3):
        letters.append(chr(ord("A") + letters_idx % 26))
        letters_idx //= 26
    return "".join(reversed(letters)) + f"{digits:03d}"


def _random_subject_id(existing: set[str]) -> str:
    """Return a unique random 3-letter/3-digit identifier."""
    while True:
        letters = ''.join(random.choices(string.ascii_uppercase, k=3))
        digits = ''.join(random.choices(string.digits, k=3))
        sid = letters + digits
        if sid not in existing:
            return sid


def _next_numeric_id(used: set[str]) -> str:
    """Return the next "sub-XXX" style identifier."""
    nums = []
    for name in used:
        m = re.fullmatch(r"sub-(\d+)", name)
        if m:
            try:
                nums.append(int(m.group(1)))
            except Exception:
                pass
    nxt = max(nums, default=0) + 1
    while True:
        candidate = f"sub-{nxt:03d}"
        if candidate not in used:
            return candidate
        nxt += 1


class AutoFillTableWidget(QTableWidget):
    """``QTableWidget`` with an Excel-like autofill handle.

    The widget exposes a small square in the bottom-right corner of the current
    selection.  Dragging this handle extends the selection and automatically
    fills the new cells either by cloning the original content or by continuing
    simple sequences (numeric, datetime, or text with trailing digits).
    """

    HANDLE_SIZE = 8  # Square size in device pixels

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._handle_rect = QRect()
        self._autofill_active = False
        self._autofill_origin_range: Optional[QTableWidgetSelectionRange] = None
        self._autofill_current_range: Optional[QTableWidgetSelectionRange] = None
        # Whenever the selection changes we repaint so the handle follows it.
        self.itemSelectionChanged.connect(self._refresh_handle)

    # ------------------------------------------------------------------
    # Painting utilities
    def _index_is_editable(self, row: int, column: int) -> bool:
        """Return ``True`` when the ``(row, column)`` cell can be edited."""

        model = self.model()
        index = model.index(row, column)
        if not index.isValid():
            return False
        flags = model.flags(index)
        return bool(flags & Qt.ItemIsEditable)

    def _range_is_editable(
        self, rng: Optional[QTableWidgetSelectionRange]
    ) -> bool:
        """Return ``True`` when *all* cells in ``rng`` expose the edit flag."""

        if rng is None:
            return False
        for row in range(rng.topRow(), rng.bottomRow() + 1):
            for col in range(rng.leftColumn(), rng.rightColumn() + 1):
                if not self._index_is_editable(row, col):
                    return False
        return True

    def _clamp_to_editable(
        self,
        origin: QTableWidgetSelectionRange,
        candidate: QTableWidgetSelectionRange,
    ) -> QTableWidgetSelectionRange:
        """Restrict ``candidate`` so autofill never crosses non-editable cells."""

        # Limit horizontal growth by checking each additional column in order.
        max_right = origin.rightColumn()
        if candidate.rightColumn() > max_right:
            for col in range(origin.rightColumn() + 1, candidate.rightColumn() + 1):
                if all(
                    self._index_is_editable(row, col)
                    for row in range(origin.topRow(), origin.bottomRow() + 1)
                ):
                    max_right = col
                else:
                    break

        # Limit vertical growth by checking each extra row with the approved columns.
        max_bottom = origin.bottomRow()
        if candidate.bottomRow() > max_bottom:
            for row in range(origin.bottomRow() + 1, candidate.bottomRow() + 1):
                if all(
                    self._index_is_editable(row, col)
                    for col in range(origin.leftColumn(), max_right + 1)
                ):
                    max_bottom = row
                else:
                    break

        if (
            max_right == origin.rightColumn()
            and max_bottom == origin.bottomRow()
        ):
            return origin

        return QTableWidgetSelectionRange(
            origin.topRow(),
            origin.leftColumn(),
            max_bottom,
            max_right,
        )
    def _refresh_handle(self) -> None:
        """Trigger a repaint so the autofill handle reflects the selection."""

        self._handle_rect = QRect()
        self.viewport().update()

    def _current_selection_range(self) -> Optional[QTableWidgetSelectionRange]:
        """Return the single active selection range (if any)."""

        ranges = self.selectedRanges()
        if len(ranges) != 1:
            return None
        return ranges[0]

    def paintEvent(self, event):  # noqa: D401 - Qt override
        super().paintEvent(event)

        rng = self._current_selection_range()
        if not self._range_is_editable(rng):
            self._handle_rect = QRect()
            return

        model_index = self.model().index(rng.bottomRow(), rng.rightColumn())
        rect = self.visualRect(model_index)
        if not rect.isValid():
            self._handle_rect = QRect()
            return

        size = self.HANDLE_SIZE
        self._handle_rect = QRect(
            rect.right() - size + 1,
            rect.bottom() - size + 1,
            size,
            size,
        )

        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self._handle_rect, self.palette().highlight())
        border = QPen(self.palette().color(QPalette.Dark))
        painter.setPen(border)
        painter.drawRect(self._handle_rect)
        painter.end()

    # ------------------------------------------------------------------
    # Mouse interaction handling
    def mousePressEvent(self, event):  # noqa: D401 - Qt override
        if (
            event.button() == Qt.LeftButton
            and self._handle_rect.contains(event.pos())
            and self._current_selection_range() is not None
        ):
            self._autofill_active = True
            self._autofill_origin_range = self._current_selection_range()
            self._autofill_current_range = self._autofill_origin_range
            self.viewport().setCursor(Qt.SizeAllCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: D401 - Qt override
        if self._autofill_active:
            new_range = self._compute_drag_range(event.pos())
            if new_range is not None:
                self._autofill_current_range = new_range
                # Replace the selection with the preview range so the user sees
                # the future extent of the autofill before releasing the mouse.
                self.blockSignals(True)
                self.clearSelection()
                self.setRangeSelected(new_range, True)
                self.blockSignals(False)
                self.viewport().update()
            event.accept()
            return

        if self._handle_rect.contains(event.pos()):
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.viewport().unsetCursor()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: D401 - Qt override
        if self._autofill_active and event.button() == Qt.LeftButton:
            try:
                self._finish_autofill()
            finally:
                self._autofill_active = False
                self.viewport().unsetCursor()
                self._autofill_origin_range = None
                self._autofill_current_range = None
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def _compute_drag_range(self, pos: QPoint) -> Optional[QTableWidgetSelectionRange]:
        """Return the preview range while dragging the autofill handle."""

        if self._autofill_origin_range is None:
            return None

        origin = self._autofill_origin_range
        row = self.rowAt(pos.y())
        col = self.columnAt(pos.x())
        if row < 0 or col < 0:
            return origin

        # Only allow extending outward from the original bottom/right edge.
        if row < origin.bottomRow():
            row = origin.bottomRow()
        if col < origin.rightColumn():
            col = origin.rightColumn()

        if row == origin.bottomRow() and col == origin.rightColumn():
            return origin

        candidate = QTableWidgetSelectionRange(
            origin.topRow(),
            origin.leftColumn(),
            row,
            col,
        )

        return self._clamp_to_editable(origin, candidate)

    # ------------------------------------------------------------------
    # Autofill logic
    def _finish_autofill(self) -> None:
        """Apply the autofill operation once the user releases the mouse."""

        if self._autofill_origin_range is None:
            return

        final_range = self._autofill_current_range or self._autofill_origin_range
        origin = self._autofill_origin_range

        if (
            final_range.bottomRow() == origin.bottomRow()
            and final_range.rightColumn() == origin.rightColumn()
        ):
            # Nothing changed; restore the original selection highlight.
            self.blockSignals(True)
            self.clearSelection()
            self.setRangeSelected(origin, True)
            self.blockSignals(False)
            self.viewport().update()
            return

        self._apply_autofill(origin, final_range)

        # Keep the extended range selected after the fill to match spreadsheet UX.
        self.blockSignals(True)
        self.clearSelection()
        self.setRangeSelected(final_range, True)
        self.blockSignals(False)
        self.viewport().update()

    def _apply_autofill(
        self,
        origin: QTableWidgetSelectionRange,
        target: QTableWidgetSelectionRange,
    ) -> None:
        """Populate ``target`` based on ``origin`` and the autofill heuristics."""

        extend_down = target.bottomRow() > origin.bottomRow()
        extend_right = target.rightColumn() > origin.rightColumn()

        if extend_right:
            self._fill_right(origin, target.rightColumn())

        if extend_down:
            base_right = target.rightColumn() if extend_right else origin.rightColumn()
            base_range = QTableWidgetSelectionRange(
                origin.topRow(),
                origin.leftColumn(),
                origin.bottomRow(),
                base_right,
            )
            self._fill_down(base_range, target.bottomRow())

    def _fill_right(self, base: QTableWidgetSelectionRange, target_right: int) -> None:
        """Extend ``base`` horizontally until ``target_right`` inclusive."""

        extra = target_right - base.rightColumn()
        if extra <= 0:
            return

        for row in range(base.topRow(), base.bottomRow() + 1):
            values = [
                self._get_item_text(row, col)
                for col in range(base.leftColumn(), base.rightColumn() + 1)
            ]
            new_values = self._extend_series(values, extra)
            for offset, value in enumerate(new_values, start=1):
                self._set_item_text(row, base.rightColumn() + offset, value)

    def _fill_down(self, base: QTableWidgetSelectionRange, target_bottom: int) -> None:
        """Extend ``base`` vertically until ``target_bottom`` inclusive."""

        extra = target_bottom - base.bottomRow()
        if extra <= 0:
            return

        for col in range(base.leftColumn(), base.rightColumn() + 1):
            values = [
                self._get_item_text(row, col)
                for row in range(base.topRow(), base.bottomRow() + 1)
            ]
            new_values = self._extend_series(values, extra)
            for offset, value in enumerate(new_values, start=1):
                self._set_item_text(base.bottomRow() + offset, col, value)

    # ------------------------------------------------------------------
    # Sequence helpers
    def _extend_series(self, values: list[str], steps: int) -> list[str]:
        """Return ``steps`` new values continuing ``values`` when possible."""

        if steps <= 0 or not values:
            return []

        numeric = self._extend_numeric_series(values, steps)
        if numeric is not None:
            return numeric

        datelike = self._extend_datetime_series(values, steps)
        if datelike is not None:
            return datelike

        patterned = self._extend_text_pattern_series(values, steps)
        if patterned is not None:
            return patterned

        # Fallback: repeat the original pattern cyclically.
        repeated = []
        for i in range(steps):
            repeated.append(values[i % len(values)])
        return repeated

    def _extend_numeric_series(
        self,
        values: list[str],
        steps: int,
    ) -> Optional[list[str]]:
        """Continue integer/decimal sequences when the pattern is consistent."""

        stripped = [v.strip() for v in values]
        if any(not s for s in stripped):
            return None

        # Try integer sequences first so "01", "02" keep their padding.
        if all(re.fullmatch(r"[+-]?\d+", s) for s in stripped):
            numbers = [int(s) for s in stripped]
            diff = 0
            if len(numbers) >= 2:
                diffs = [numbers[i] - numbers[i - 1] for i in range(1, len(numbers))]
                if all(d == diffs[0] for d in diffs):
                    diff = diffs[0]
            pad_width = len(stripped[-1].lstrip("+-"))
            has_leading_zero = stripped[-1].lstrip("+-").startswith("0") and pad_width > 1
            force_plus = stripped[-1].startswith("+")
            current = numbers[-1]
            generated: list[str] = []
            for _ in range(steps):
                current += diff
                text = str(current)
                if has_leading_zero and current >= 0:
                    text = f"{current:0{pad_width}d}"
                if force_plus and not text.startswith("-") and not text.startswith("+"):
                    text = "+" + text
                generated.append(text)
            return generated

        # Fall back to decimals when integers are not appropriate.
        decimals: list[Decimal] = []
        decimal_places = 0
        for s in stripped:
            try:
                dec = Decimal(s)
            except InvalidOperation:
                return None
            decimals.append(dec)
            if "." in s:
                decimal_places = max(decimal_places, len(s.split(".")[-1]))

        diff = Decimal(0)
        if len(decimals) >= 2:
            diffs = [decimals[i] - decimals[i - 1] for i in range(1, len(decimals))]
            if all(d == diffs[0] for d in diffs):
                diff = diffs[0]
        current = decimals[-1]
        generated = []
        for _ in range(steps):
            current += diff
            if decimal_places:
                generated.append(f"{current:.{decimal_places}f}")
            else:
                generated.append(str(current))
        return generated

    def _extend_datetime_series(
        self,
        values: list[str],
        steps: int,
    ) -> Optional[list[str]]:
        """Continue datetime-like strings when intervals are consistent."""

        try:
            parsed = pd.to_datetime(values, errors="raise", infer_datetime_format=True)
        except Exception:
            return None

        if parsed.isna().any():
            return None

        delta = pd.Timedelta(0)
        if len(parsed) >= 2:
            diffs = parsed.diff().iloc[1:]
            if not diffs.empty and all(d == diffs.iloc[0] for d in diffs):
                delta = diffs.iloc[0]

        last = parsed.iloc[-1]
        template = values[-1]
        fmt = guess_datetime_format(template)
        generated = []
        current = last
        for _ in range(steps):
            current = current + delta
            if fmt:
                generated.append(current.strftime(fmt))
            else:
                if "T" in template:
                    generated.append(current.isoformat())
                elif ":" in template:
                    generated.append(current.strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    generated.append(current.date().isoformat())
        return generated

    def _extend_text_pattern_series(
        self,
        values: list[str],
        steps: int,
    ) -> Optional[list[str]]:
        """Continue strings ending with digits (e.g. "scan01")."""

        matches = [re.match(r"^(.*?)(\d+)$", v.strip()) for v in values]
        if any(m is None for m in matches):
            return None

        prefixes = [m.group(1) for m in matches if m is not None]
        numbers = [int(m.group(2)) for m in matches if m is not None]
        if not prefixes or not numbers:
            return None
        if any(p != prefixes[0] for p in prefixes):
            return None

        diff = 0
        if len(numbers) >= 2:
            diffs = [numbers[i] - numbers[i - 1] for i in range(1, len(numbers))]
            if all(d == diffs[0] for d in diffs):
                diff = diffs[0]
        pad_width = len(matches[-1].group(2)) if matches[-1] is not None else 0
        current = numbers[-1]
        prefix = prefixes[-1]
        generated = []
        for _ in range(steps):
            current += diff
            text = f"{current:0{pad_width}d}" if pad_width else str(current)
            generated.append(f"{prefix}{text}")
        return generated

    # ------------------------------------------------------------------
    # Cell helpers
    def _get_item_text(self, row: int, column: int) -> str:
        """Return the text stored at ``(row, column)`` (empty if missing)."""

        item = self.item(row, column)
        return item.text() if item is not None else ""

    def _set_item_text(self, row: int, column: int, value: str) -> None:
        """Assign ``value`` to ``(row, column)``, creating an item if required."""

        item = self.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.setItem(row, column, item)
        item.setText(value)


class MappingSortDialog(QDialog):
    """Dialog used to configure multi-level sorting for the metadata table."""

    def __init__(self, columns: list[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sort scanned metadata")
        self._columns = columns
        self._level_rows: list[tuple[QComboBox, QComboBox, QWidget]] = []

        layout = QVBoxLayout(self)
        info = QLabel(
            "Select the columns to sort by in priority order. "
            "Each level is applied sequentially, just like Excel's multi-column sort."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._levels_layout = QVBoxLayout()
        layout.addLayout(self._levels_layout)

        controls_layout = QHBoxLayout()
        self._add_level_btn = QPushButton("Add level")
        self._add_level_btn.clicked.connect(self._add_level)
        self._remove_level_btn = QPushButton("Remove level")
        self._remove_level_btn.clicked.connect(self._remove_level)
        controls_layout.addWidget(self._add_level_btn)
        controls_layout.addWidget(self._remove_level_btn)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Always start with a single sorting level configured.
        self._add_level()
        self._update_button_state()

    # ------------------------------------------------------------------
    def _add_level(self) -> None:
        """Append a new sorting level to the dialog."""

        if len(self._level_rows) >= len(self._columns):
            return

        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        column_combo = QComboBox(row_widget)
        column_combo.addItems(self._columns)
        # Try to pre-select the first column that is not already used.
        used = {combo.currentText() for combo, _order, _widget in self._level_rows}
        for idx, name in enumerate(self._columns):
            if name not in used:
                column_combo.setCurrentIndex(idx)
                break

        order_combo = QComboBox(row_widget)
        order_combo.addItems(["Ascending", "Descending"])

        row_layout.addWidget(QLabel(f"Level {len(self._level_rows) + 1}", row_widget))
        row_layout.addWidget(column_combo)
        row_layout.addWidget(order_combo)
        row_layout.addStretch()

        self._levels_layout.addWidget(row_widget)
        self._level_rows.append((column_combo, order_combo, row_widget))
        self._update_button_state()

    def _remove_level(self) -> None:
        """Remove the last configured sorting level."""

        if not self._level_rows:
            return
        combo, order_combo, widget = self._level_rows.pop()
        combo.deleteLater()
        order_combo.deleteLater()
        widget.deleteLater()
        self._update_button_state()

    def _update_button_state(self) -> None:
        """Enable/disable controls based on the current dialog state."""

        self._add_level_btn.setEnabled(len(self._level_rows) < len(self._columns))
        self._remove_level_btn.setEnabled(len(self._level_rows) > 1)

    def sort_instructions(self) -> list[tuple[str, bool]]:
        """Return the configured sorting hierarchy as ``(column, ascending)``."""

        instructions: list[tuple[str, bool]] = []
        seen: set[str] = set()
        for column_combo, order_combo, _widget in self._level_rows:
            column = column_combo.currentText()
            if not column or column in seen:
                continue
            ascending = order_combo.currentText() == "Ascending"
            instructions.append((column, ascending))
            seen.add(column)
        return instructions


class SubjectDelegate(QStyledItemDelegate):
    """Delegate to edit BIDS subject IDs without altering the 'sub-' prefix."""

    def createEditor(self, parent, option, index):  # noqa: D401 - Qt override
        return QLineEdit(parent)

    def setEditorData(self, editor, index):  # noqa: D401 - Qt override
        text = index.model().data(index, Qt.EditRole)
        suffix = text[4:] if text.startswith("sub-") else text
        editor.setText(suffix)
        editor.selectAll()

    def setModelData(self, editor, model, index):  # noqa: D401 - Qt override
        model.setData(index, "sub-" + editor.text(), Qt.EditRole)


class ShrinkableScrollArea(QScrollArea):
    """``QScrollArea`` variant that allows the parent splitter to shrink."""

    def minimumSizeHint(self) -> QSize:  # noqa: D401 - Qt override
        return QSize(0, 0)

    def sizeHint(self) -> QSize:  # noqa: D401 - Qt override
        hint = super().sizeHint()
        return QSize(max(0, hint.width()), max(0, hint.height()))


class BIDSManager(QMainWindow):
    """
    Main GUI for BIDS Manager.
    Provides two tabs: Convert (DICOM→BIDS pipeline) and Edit (BIDS dataset explorer/editor).
    Supports Windows and Linux.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BIDS Manager")
        if ICON_FILE.exists():
            self.setWindowIcon(QIcon(str(ICON_FILE)))
        self.resize(900, 900)
        self.setMinimumSize(640, 480)

        app = QApplication.instance()
        self._base_font = app.font()
        screen = app.primaryScreen()
        # Detect the OS DPI scaling in a resilient way.  ``logicalDotsPerInch``
        # is usually the most accurate value because it already factors in the
        # operating-system scale factor.  On some platforms (e.g. X11 with
        # unusual monitor reporting) this can be inaccurate, so we fall back to
        # the device pixel ratio when needed.  The result is stored as a
        # percentage relative to the 96 DPI base expected by Qt.
        self._os_dpi = self._detect_system_dpi(screen)

        # User-requested DPI scale.  By default we start from the detected
        # system value so that the UI matches the OS scaling out of the box.
        # Users can still fine-tune the value manually via the DPI dialog.
        self.dpi_scale = self._os_dpi

        # Paths
        self.dicom_dir = ""         # Raw DICOM directory
        self.bids_out_dir = ""      # Output BIDS directory
        self.tsv_path = ""          # Path to subject_summary.tsv
        self.heuristic_dir = ""     # Directory with heuristics
        # Lookup containers used to synchronise the mapping table with the
        # modality trees.  Keys are different path elements and values are lists
        # of row indices in ``self.mapping_table``.
        self.study_set = set()        # All study names encountered
        self.modb_rows = {}           # BIDS modality → [row, ...]
        self.mod_rows = {}            # (BIDS modality, modality) → [row, ...]
        self.seq_rows = {}            # (BIDS modality, modality, sequence) → rows
        self.study_rows = {}
        self.subject_rows = {}
        self.session_rows = {}
        self.spec_modb_rows = {}
        self.spec_mod_rows = {}
        self.spec_seq_rows = {}
        # Equivalent lookups when displaying the "given" subject names instead
        # of the BIDS names
        self.subject_rows_given = {}
        self.session_rows_given = {}
        self.spec_modb_rows_given = {}
        self.spec_mod_rows_given = {}
        self.spec_seq_rows_given = {}
        # Existing mappings found in output datasets
        self.existing_maps = {}
        self.existing_used = {}
        self.use_bids_names = True

        # Track repeat-detection state so we can notify the user when new
        # duplicates are found and keep the "Only last repeats" option in sync.
        self._last_repeat_count = 0

        # Flag used to skip expensive updates while the mapping table is being
        # populated programmatically.  ``QTableWidget`` emits ``itemChanged``
        # signals even when values are assigned in code, so we guard callbacks
        # that rebuild caches with this boolean.
        self._loading_mapping_table = False

        # Async process handles for inventory and conversion steps
        self.inventory_process = None  # QProcess for dicom_inventory
        self.conv_process = None       # QProcess for the conversion pipeline
        self.conv_stage = 0            # Tracks which step of the pipeline ran
        self.heurs_to_rename = []      # List of heuristics pending rename
        # Background worker used to detect mixed StudyInstanceUID folders
        self._conflict_thread: Optional[QThread] = None
        self._conflict_worker: Optional[_ConflictScannerWorker] = None

        # Root of the currently loaded BIDS dataset (None until loaded)
        self.bids_root = None

        # Schema information for proposed BIDS names
        self._schema = None
        if ENABLE_SCHEMA_RENAMER:
            try:
                self._schema = load_bids_schema(DEFAULT_SCHEMA_DIR)
            except Exception as e:
                print(f"[WARN] Could not load BIDS schema: {e}")
                self._schema = None
        self.inventory_df = None

        # Path to persistent user preferences
        self.pref_dir = PREF_DIR
        try:
            self.pref_dir.mkdir(exist_ok=True, parents=True)
        except Exception:
            pass
        self.exclude_patterns_file = self.pref_dir / "exclude_patterns.tsv"
        self.theme_file = self.pref_dir / "theme.txt"
        self.dpi_file = self.pref_dir / "dpi_scale.txt"
        # Load any previously stored DPI preference so the UI scale persists
        # across sessions.  Invalid or out-of-range values fall back to the
        # detected system scale.  This keeps the first launch aligned with the
        # host environment while preserving user tweaks afterwards.
        if self.dpi_file.exists():
            try:
                saved_dpi = int(self.dpi_file.read_text().strip())
            except ValueError:
                saved_dpi = None
            except Exception:
                saved_dpi = None
            if saved_dpi is not None:
                self.dpi_scale = max(50, min(200, saved_dpi))
        self.seq_dict_file = dicom_inventory.SEQ_DICT_FILE
        # Flag used to automatically reapply the suffix dictionary after a fresh
        # scan.  This keeps the scanned data table in sync with the latest
        # custom patterns without overriding manual edits made later via
        # ``applyMappingChanges``.
        self._apply_sequence_on_load = False

        # Spinner for long-running tasks
        self.spinner_label = None
        # Timer and unicode characters for the small animated spinner that
        # appears while long-running subprocesses are running
        self._spinner_timer = QTimer()
        self._spinner_timer.timeout.connect(self._spin)
        self._spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self._spinner_index = 0
        self._spinner_message = ""

        # Parallel settings
        # Use ~80% of available CPUs by default to avoid saturating the system
        # when running external tools in parallel.  ``os.cpu_count`` may return
        # ``None`` so fall back to 1 in that case.
        total_cpu = os.cpu_count() or 1
        # ``current`` is pre-set to about 80% of this value in the main window.
        self.num_cpus = max(1, round(total_cpu * 0.8))

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Tab widget
        self.tabs = QTabWidget()
        # Use larger font for tab labels
        font = QFont()
        font.setPointSize(10)
        self.tabs.setFont(font)
        main_layout.addWidget(self.tabs)

        # Initialize tabs
        self.initConvertTab()
        self.initEditTab()
        self._updateMappingControlsEnabled()

        # Theme support
        self.statusBar()
        self.themes = self._build_theme_dict()
        self.current_theme = None
        self.theme_btn = QPushButton("🌓")  # half-moon icon
        self.theme_btn.setFixedWidth(50)
        self.cpu_btn = QPushButton(f"CPU: {self.num_cpus}")
        self.cpu_btn.setFixedWidth(70)
        self.cpu_btn.clicked.connect(self.show_cpu_dialog)
        self.authorship_btn = QPushButton("Authorship")
        self.authorship_btn.setFixedWidth(90)
        self.authorship_btn.clicked.connect(self.show_authorship_dialog)
        self.dpi_btn = QPushButton(f"DPI: {self.dpi_scale}%")
        self.dpi_btn.setFixedWidth(80)
        self.dpi_btn.clicked.connect(self.show_dpi_dialog)
        # Create a container widget with layout to adjust position
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 2, 0, 6)  # left, top, right, bottom
        layout.setSpacing(8)
        layout.addWidget(self.theme_btn)
        layout.addWidget(self.cpu_btn)
        layout.addWidget(self.dpi_btn)
        layout.addWidget(self.authorship_btn)
        container.setLayout(layout)
        # Add the container to the status bar (left-aligned)
        self.statusBar().addWidget(container)
        # Create the theme menu
        theme_menu = QMenu(self)
        for name in self.themes.keys():
            act = theme_menu.addAction(name)
            act.triggered.connect(lambda _=False, n=name: self.apply_theme(n))
        self.theme_btn.setMenu(theme_menu)

        # Load previously saved theme preference
        default_theme = "Light"
        if self.theme_file.exists():
            try:
                default_theme = self.theme_file.read_text().strip() or default_theme
            except Exception:
                pass
        self.apply_theme(default_theme)

    def _build_theme_dict(self):
        """Return dictionary mapping theme names to QPalettes."""
        themes = {}

        dark_purple = QPalette()
        dark_purple.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_purple.setColor(QPalette.WindowText, Qt.white)
        dark_purple.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_purple.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_purple.setColor(QPalette.ToolTipBase, QColor(65, 65, 65))
        dark_purple.setColor(QPalette.ToolTipText, Qt.white)
        dark_purple.setColor(QPalette.Text, Qt.white)
        dark_purple.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_purple.setColor(QPalette.ButtonText, Qt.white)
        dark_purple.setColor(QPalette.Highlight, QColor(142, 45, 197))
        dark_purple.setColor(QPalette.HighlightedText, Qt.black)
        themes["Dark-purple"] = dark_purple

        dark_blue = QPalette()
        dark_blue.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_blue.setColor(QPalette.WindowText, Qt.white)
        dark_blue.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_blue.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_blue.setColor(QPalette.ToolTipBase, QColor(65, 65, 65))
        dark_blue.setColor(QPalette.ToolTipText, Qt.white)
        dark_blue.setColor(QPalette.Text, Qt.white)
        dark_blue.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_blue.setColor(QPalette.ButtonText, Qt.white)
        dark_blue.setColor(QPalette.Highlight, QColor(65, 105, 225))
        dark_blue.setColor(QPalette.HighlightedText, Qt.black)
        themes["Dark-blue"] = dark_blue

        dark_gold = QPalette()
        dark_gold.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_gold.setColor(QPalette.WindowText, Qt.white)
        dark_gold.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_gold.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_gold.setColor(QPalette.ToolTipBase, QColor(65, 65, 65))
        dark_gold.setColor(QPalette.ToolTipText, Qt.white)
        dark_gold.setColor(QPalette.Text, Qt.white)
        dark_gold.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_gold.setColor(QPalette.ButtonText, Qt.white)
        dark_gold.setColor(QPalette.Highlight, QColor(218, 165, 32))
        dark_gold.setColor(QPalette.HighlightedText, Qt.black)
        themes["Dark-gold"] = dark_gold

        light = QPalette()
        light.setColor(QPalette.Window, Qt.white)
        light.setColor(QPalette.WindowText, Qt.black)
        light.setColor(QPalette.Base, QColor(245, 245, 245))
        light.setColor(QPalette.AlternateBase, Qt.white)
        light.setColor(QPalette.ToolTipBase, Qt.white)
        light.setColor(QPalette.ToolTipText, Qt.black)
        light.setColor(QPalette.Text, Qt.black)
        light.setColor(QPalette.Button, QColor(240, 240, 240))
        light.setColor(QPalette.ButtonText, Qt.black)
        light.setColor(QPalette.Highlight, QColor(100, 149, 237))
        light.setColor(QPalette.HighlightedText, Qt.white)
        themes["Light"] = light

        beige = QPalette()
        beige.setColor(QPalette.Window, QColor(243, 232, 210))
        beige.setColor(QPalette.WindowText, Qt.black)
        beige.setColor(QPalette.Base, QColor(250, 240, 222))
        beige.setColor(QPalette.AlternateBase, QColor(246, 236, 218))
        beige.setColor(QPalette.ToolTipBase, QColor(236, 224, 200))
        beige.setColor(QPalette.ToolTipText, Qt.black)
        beige.setColor(QPalette.Text, Qt.black)
        beige.setColor(QPalette.Button, QColor(242, 231, 208))
        beige.setColor(QPalette.ButtonText, Qt.black)
        beige.setColor(QPalette.Highlight, QColor(196, 148, 70))
        beige.setColor(QPalette.HighlightedText, Qt.white)
        themes["Beige"] = beige

        ocean = QPalette()
        ocean.setColor(QPalette.Window, QColor(225, 238, 245))
        ocean.setColor(QPalette.WindowText, Qt.black)
        ocean.setColor(QPalette.Base, QColor(240, 248, 252))
        ocean.setColor(QPalette.AlternateBase, QColor(230, 240, 247))
        ocean.setColor(QPalette.ToolTipBase, QColor(215, 230, 240))
        ocean.setColor(QPalette.ToolTipText, Qt.black)
        ocean.setColor(QPalette.Text, Qt.black)
        ocean.setColor(QPalette.Button, QColor(213, 234, 242))
        ocean.setColor(QPalette.ButtonText, Qt.black)
        ocean.setColor(QPalette.Highlight, QColor(0, 123, 167))
        ocean.setColor(QPalette.HighlightedText, Qt.white)
        themes["Ocean"] = ocean

        hc = QPalette()
        hc.setColor(QPalette.Window, Qt.black)
        hc.setColor(QPalette.WindowText, Qt.white)
        hc.setColor(QPalette.Base, Qt.black)
        hc.setColor(QPalette.AlternateBase, Qt.black)
        hc.setColor(QPalette.ToolTipBase, Qt.black)
        hc.setColor(QPalette.ToolTipText, Qt.white)
        hc.setColor(QPalette.Text, Qt.white)
        hc.setColor(QPalette.BrightText, Qt.white)
        hc.setColor(QPalette.Button, Qt.black)
        hc.setColor(QPalette.ButtonText, Qt.white)
        hc.setColor(QPalette.Highlight, QColor(255, 215, 0))
        hc.setColor(QPalette.HighlightedText, Qt.black)
        themes["Contrast"] = hc

        hc_w = QPalette()
        hc_w.setColor(QPalette.Window, Qt.white)
        hc_w.setColor(QPalette.WindowText, Qt.black)
        hc_w.setColor(QPalette.Base, Qt.white)
        hc_w.setColor(QPalette.AlternateBase, Qt.white)
        hc_w.setColor(QPalette.ToolTipBase, Qt.white)
        hc_w.setColor(QPalette.ToolTipText, Qt.black)
        hc_w.setColor(QPalette.Text, Qt.black)
        hc_w.setColor(QPalette.BrightText, Qt.black)
        hc_w.setColor(QPalette.Button, Qt.white)
        hc_w.setColor(QPalette.ButtonText, Qt.black)
        hc_w.setColor(QPalette.Highlight, QColor(255, 215, 0))
        hc_w.setColor(QPalette.HighlightedText, Qt.black)
        themes["Contrast White"] = hc_w

        solar = QPalette()
        solar.setColor(QPalette.Window, QColor(253, 246, 227))
        solar.setColor(QPalette.WindowText, QColor(101, 123, 131))
        solar.setColor(QPalette.Base, QColor(255, 250, 240))
        solar.setColor(QPalette.AlternateBase, QColor(253, 246, 227))
        solar.setColor(QPalette.ToolTipBase, QColor(238, 232, 213))
        solar.setColor(QPalette.ToolTipText, QColor(88, 110, 117))
        solar.setColor(QPalette.Text, QColor(88, 110, 117))
        solar.setColor(QPalette.Button, QColor(238, 232, 213))
        solar.setColor(QPalette.ButtonText, QColor(88, 110, 117))
        solar.setColor(QPalette.Highlight, QColor(38, 139, 210))
        solar.setColor(QPalette.HighlightedText, Qt.white)
        themes["Solar"] = solar

        cyber = QPalette()
        cyber.setColor(QPalette.Window, QColor(20, 20, 30))
        cyber.setColor(QPalette.WindowText, QColor(0, 255, 255))
        cyber.setColor(QPalette.Base, QColor(30, 30, 45))
        cyber.setColor(QPalette.AlternateBase, QColor(25, 25, 35))
        cyber.setColor(QPalette.ToolTipBase, QColor(45, 45, 65))
        cyber.setColor(QPalette.ToolTipText, QColor(255, 0, 255))
        cyber.setColor(QPalette.Text, QColor(0, 255, 255))
        cyber.setColor(QPalette.Button, QColor(40, 40, 55))
        cyber.setColor(QPalette.ButtonText, QColor(255, 0, 255))
        cyber.setColor(QPalette.Highlight, QColor(255, 0, 128))
        cyber.setColor(QPalette.HighlightedText, Qt.white)
        themes["Cyber"] = cyber

        drac = QPalette()
        drac.setColor(QPalette.Window, QColor("#282a36"))
        drac.setColor(QPalette.WindowText, QColor("#f8f8f2"))
        drac.setColor(QPalette.Base, QColor("#1e1f29"))
        drac.setColor(QPalette.AlternateBase, QColor("#282a36"))
        drac.setColor(QPalette.ToolTipBase, QColor("#44475a"))
        drac.setColor(QPalette.ToolTipText, QColor("#f8f8f2"))
        drac.setColor(QPalette.Text, QColor("#f8f8f2"))
        drac.setColor(QPalette.Button, QColor("#44475a"))
        drac.setColor(QPalette.ButtonText, QColor("#f8f8f2"))
        drac.setColor(QPalette.Highlight, QColor("#bd93f9"))
        drac.setColor(QPalette.HighlightedText, Qt.black)
        themes["Dracula"] = drac

        nord = QPalette()
        nord.setColor(QPalette.Window, QColor("#2e3440"))
        nord.setColor(QPalette.WindowText, QColor("#d8dee9"))
        nord.setColor(QPalette.Base, QColor("#3b4252"))
        nord.setColor(QPalette.AlternateBase, QColor("#434c5e"))
        nord.setColor(QPalette.ToolTipBase, QColor("#4c566a"))
        nord.setColor(QPalette.ToolTipText, QColor("#eceff4"))
        nord.setColor(QPalette.Text, QColor("#e5e9f0"))
        nord.setColor(QPalette.Button, QColor("#4c566a"))
        nord.setColor(QPalette.ButtonText, QColor("#d8dee9"))
        nord.setColor(QPalette.Highlight, QColor("#88c0d0"))
        nord.setColor(QPalette.HighlightedText, Qt.black)
        themes["Nord"] = nord

        gruv = QPalette()
        gruv.setColor(QPalette.Window, QColor("#282828"))
        gruv.setColor(QPalette.WindowText, QColor("#ebdbb2"))
        gruv.setColor(QPalette.Base, QColor("#32302f"))
        gruv.setColor(QPalette.AlternateBase, QColor("#3c3836"))
        gruv.setColor(QPalette.ToolTipBase, QColor("#504945"))
        gruv.setColor(QPalette.ToolTipText, QColor("#fbf1c7"))
        gruv.setColor(QPalette.Text, QColor("#ebdbb2"))
        gruv.setColor(QPalette.Button, QColor("#504945"))
        gruv.setColor(QPalette.ButtonText, QColor("#ebdbb2"))
        gruv.setColor(QPalette.Highlight, QColor("#d79921"))
        gruv.setColor(QPalette.HighlightedText, Qt.black)
        themes["Gruvbox"] = gruv

        mono = QPalette()
        mono.setColor(QPalette.Window, QColor("#272822"))
        mono.setColor(QPalette.WindowText, QColor("#f8f8f2"))
        mono.setColor(QPalette.Base, QColor("#1e1f1c"))
        mono.setColor(QPalette.AlternateBase, QColor("#272822"))
        mono.setColor(QPalette.ToolTipBase, QColor("#3e3d32"))
        mono.setColor(QPalette.ToolTipText, QColor("#f8f8f2"))
        mono.setColor(QPalette.Text, QColor("#f8f8f2"))
        mono.setColor(QPalette.Button, QColor("#3e3d32"))
        mono.setColor(QPalette.ButtonText, QColor("#f8f8f2"))
        mono.setColor(QPalette.Highlight, QColor("#a6e22e"))
        mono.setColor(QPalette.HighlightedText, Qt.black)
        themes["Monokai"] = mono

        tokyo = QPalette()
        tokyo.setColor(QPalette.Window, QColor("#1a1b26"))
        tokyo.setColor(QPalette.WindowText, QColor("#c0caf5"))
        tokyo.setColor(QPalette.Base, QColor("#1f2335"))
        tokyo.setColor(QPalette.AlternateBase, QColor("#24283b"))
        tokyo.setColor(QPalette.ToolTipBase, QColor("#414868"))
        tokyo.setColor(QPalette.ToolTipText, QColor("#c0caf5"))
        tokyo.setColor(QPalette.Text, QColor("#c0caf5"))
        tokyo.setColor(QPalette.Button, QColor("#414868"))
        tokyo.setColor(QPalette.ButtonText, QColor("#c0caf5"))
        tokyo.setColor(QPalette.Highlight, QColor("#7aa2f7"))
        tokyo.setColor(QPalette.HighlightedText, Qt.white)
        themes["Tokyo"] = tokyo

        mocha = QPalette()
        mocha.setColor(QPalette.Window, QColor("#1e1e2e"))
        mocha.setColor(QPalette.WindowText, QColor("#cdd6f4"))
        mocha.setColor(QPalette.Base, QColor("#181825"))
        mocha.setColor(QPalette.AlternateBase, QColor("#1e1e2e"))
        mocha.setColor(QPalette.ToolTipBase, QColor("#313244"))
        mocha.setColor(QPalette.ToolTipText, QColor("#cdd6f4"))
        mocha.setColor(QPalette.Text, QColor("#cdd6f4"))
        mocha.setColor(QPalette.Button, QColor("#313244"))
        mocha.setColor(QPalette.ButtonText, QColor("#cdd6f4"))
        mocha.setColor(QPalette.Highlight, QColor("#f38ba8"))
        mocha.setColor(QPalette.HighlightedText, Qt.black)
        themes["Mocha"] = mocha

        pale = QPalette()
        pale.setColor(QPalette.Window, QColor("#292d3e"))
        pale.setColor(QPalette.WindowText, QColor("#a6accd"))
        pale.setColor(QPalette.Base, QColor("#1b1d2b"))
        pale.setColor(QPalette.AlternateBase, QColor("#222436"))
        pale.setColor(QPalette.ToolTipBase, QColor("#444267"))
        pale.setColor(QPalette.ToolTipText, QColor("#a6accd"))
        pale.setColor(QPalette.Text, QColor("#a6accd"))
        pale.setColor(QPalette.Button, QColor("#444267"))
        pale.setColor(QPalette.ButtonText, QColor("#a6accd"))
        pale.setColor(QPalette.Highlight, QColor("#82aaff"))
        pale.setColor(QPalette.HighlightedText, Qt.black)
        themes["Palenight"] = pale

        return themes

    def apply_theme(self, name: str):
        """Apply palette chosen from the Theme menu."""
        app = QApplication.instance()
        app.setPalette(self.themes[name])
        self.current_theme = name
        try:
            self.theme_file.write_text(name)
        except Exception:
            pass
        self._update_logo()
        self._apply_font_scale()

    def show_cpu_dialog(self) -> None:
        """Display dialog to choose number of CPUs."""
        dlg = CpuSettingsDialog(self, self.num_cpus)
        if dlg.exec_() == QDialog.Accepted:
            self.num_cpus = dlg.spin.value()
            self.cpu_btn.setText(f"CPU: {self.num_cpus}")

    def show_authorship_dialog(self) -> None:
        """Display authorship information dialog."""
        dlg = AuthorshipDialog(self)
        dlg.exec_()

    def show_dpi_dialog(self) -> None:
        """Display dialog to adjust UI scale."""
        dlg = DpiSettingsDialog(self, self.dpi_scale)
        if dlg.exec_() == QDialog.Accepted:
            self.dpi_scale = dlg.spin.value()
            self.dpi_btn.setText(f"DPI: {self.dpi_scale}%")
            self._apply_font_scale()
            # Persist the user-selected DPI so the preference is restored next
            # time the application starts.
            try:
                self.dpi_file.write_text(str(self.dpi_scale))
            except Exception:
                pass

    def _start_spinner(self, message: str) -> None:
        """Show animated spinner with *message* in the log group."""
        self._spinner_message = message
        self._spinner_index = 0
        if self.spinner_label is not None:
            self.spinner_label.setText(f"{message} {self._spinner_frames[0]}")
            self.spinner_label.show()
        self._spinner_timer.start(100)

    def _spin(self) -> None:
        if not self.spinner_label or not self.spinner_label.isVisible():
            return
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
        self.spinner_label.setText(
            f"{self._spinner_message} {self._spinner_frames[self._spinner_index]}"
        )

    def _stop_spinner(self) -> None:
        self._spinner_timer.stop()
        if self.spinner_label is not None:
            self.spinner_label.hide()

    def _is_dark_theme(self) -> bool:
        color = self.palette().color(QPalette.Window)
        brightness = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
        return brightness < 128

    def _apply_font_scale(self) -> None:
        """Apply current DPI scaling to the application font."""
        app = QApplication.instance()
        font = QFont(self._base_font)
        base_size = self._base_font.pointSizeF() or float(self._base_font.pointSize())
        # Calculate the font size relative to the detected system DPI so that
        # ``dpi_scale`` always represents the intended scale percentage of the
        # 96-DPI baseline.  Using a ratio keeps the default matching the OS while
        # letting users make incremental adjustments without surprises.
        scale_ratio = max(0.5, min(2.0, self.dpi_scale / max(self._os_dpi, 1)))
        scaled = max(1.0, base_size * scale_ratio)
        if self.current_theme in ("Contrast", "Contrast White"):
            font.setWeight(QFont.Bold)
            font.setPointSizeF(scaled + 1)
        else:
            font.setWeight(QFont.Normal)
            font.setPointSizeF(scaled)
        app.setFont(font)

        # Ensure the tab labels also scale with the selected DPI
        if hasattr(self, "tabs"):
            tab_font = QFont(font)
            tab_font.setPointSizeF(font.pointSizeF() + 1)
            self.tabs.setFont(tab_font)

    def _detect_system_dpi(self, screen):
        """Return the system DPI percentage relative to 96 with fallbacks.

        The logic prefers ``logicalDotsPerInch`` (Qt's effective DPI that already
        accounts for the OS scale).  When that value is implausible we try the
        physical DPI and the device pixel ratio, selecting the first reasonable
        candidate.  This helps avoid wildly large or tiny UI defaults on
        platforms that misreport DPI values.
        """

        if screen is None:
            return 100

        candidates = []
        try:
            logical = screen.logicalDotsPerInch()
            if logical > 1:
                candidates.append(logical / 96 * 100)
        except Exception:
            pass

        try:
            physical = screen.physicalDotsPerInch()
            if physical > 1:
                candidates.append(physical / 96 * 100)
        except Exception:
            pass

        try:
            ratio = screen.devicePixelRatio()
            if ratio > 0:
                candidates.append(ratio * 100)
        except Exception:
            pass

        for candidate in candidates:
            if 25 <= candidate <= 400:
                return int(round(candidate))

        # Fallback to a safe default when nothing sensible is reported.
        return 100

    def _update_logo(self) -> None:
        """Update logo pixmap based on current theme."""
        if not hasattr(self, "logo_label"):
            return
        if not LOGO_FILE.exists():
            return
        pix = QPixmap(str(LOGO_FILE))
        if self._is_dark_theme():
            img = pix.toImage()
            img.invertPixels()
            pix = QPixmap.fromImage(img)
        self.logo_label.setPixmap(pix.scaledToHeight(120, Qt.SmoothTransformation))

    def initConvertTab(self):
        """Create the Convert tab with a cleaner layout."""
        # This tab guides the user through the DICOM → BIDS workflow.
        # It contains controls to select directories, review the inventory TSV
        # and run the conversion pipeline while showing live logs.
        self.convert_tab = QWidget()
        main_layout = QVBoxLayout(self.convert_tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        cfg_group = QGroupBox("Configuration")
        cfg_layout = QGridLayout(cfg_group)

        dicom_label = QLabel("<b>Raw data Dir:</b>")
        self.dicom_dir_edit = QLineEdit()
        self.dicom_dir_edit.setReadOnly(True)
        dicom_browse = QPushButton("Browse…")
        dicom_browse.clicked.connect(self.selectDicomDir)
        cfg_layout.addWidget(dicom_label, 0, 0)
        cfg_layout.addWidget(self.dicom_dir_edit, 0, 1)
        cfg_layout.addWidget(dicom_browse, 0, 2)

        bids_label = QLabel("<b>BIDS Out Dir:</b>")
        self.bids_out_edit = QLineEdit()
        self.bids_out_edit.setReadOnly(True)
        bids_browse = QPushButton("Browse…")
        bids_browse.clicked.connect(self.selectBIDSOutDir)
        cfg_layout.addWidget(bids_label, 1, 0)
        cfg_layout.addWidget(self.bids_out_edit, 1, 1)
        cfg_layout.addWidget(bids_browse, 1, 2)

        tsvname_label = QLabel("<b>TSV Name:</b>")
        self.tsv_name_edit = QLineEdit("subject_summary.tsv")
        cfg_layout.addWidget(tsvname_label, 2, 0)
        cfg_layout.addWidget(self.tsv_name_edit, 2, 1, 1, 2)

        self.tsv_button = QPushButton("Scan files")
        self.tsv_button.clicked.connect(self.runInventory)
        self.tsv_stop_button = QPushButton("Stop")
        self.tsv_stop_button.setEnabled(False)
        self.tsv_stop_button.clicked.connect(self.stopInventory)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.tsv_button)
        btn_row.addWidget(self.tsv_stop_button)
        btn_row.addStretch()
        cfg_layout.addLayout(btn_row, 3, 0, 1, 3)

        # Place the configuration group and the logo on the same row with
        # equal width. The logo stays centered even when the window resizes.
        top_row = QHBoxLayout()
        top_row.addWidget(cfg_group)
        self.logo_label = QLabel()
        logo_container = QWidget()
        lc_layout = QVBoxLayout(logo_container)
        lc_layout.setContentsMargins(0, 0, 0, 0)
        lc_layout.addWidget(self.logo_label, alignment=Qt.AlignCenter)
        top_row.addWidget(logo_container)
        top_row.setStretch(0, 1)
        top_row.setStretch(1, 1)
        main_layout.addLayout(top_row)
        self._update_logo()

        self.left_split = QSplitter(Qt.Vertical)
        self.right_split = QSplitter(Qt.Vertical)

        self.tsv_group = QGroupBox("Scanned data viewer")
        tsv_layout = QVBoxLayout(self.tsv_group)
        self.tsv_detach_button = QPushButton("»")
        self.tsv_detach_button.setFixedWidth(20)
        self.tsv_detach_button.setFixedHeight(20)
        self.tsv_detach_button.clicked.connect(self.detachTSVWindow)
        self.tsv_detach_button.setFocusPolicy(Qt.NoFocus)
        header_row_tsv = QHBoxLayout()
        header_row_tsv.addStretch()
        header_row_tsv.addWidget(self.tsv_detach_button)
        tsv_layout.addLayout(header_row_tsv)
        self.tsv_tabs = QTabWidget()
        tsv_layout.addWidget(self.tsv_tabs)

        # --- Scanned metadata tab ---
        metadata_tab = QWidget()
        metadata_layout = QVBoxLayout(metadata_tab)
        metadata_toolbar = QHBoxLayout()
        self.tsv_actions_button = QToolButton()
        self.tsv_actions_button.setText("Actions")
        self.tsv_actions_button.setPopupMode(QToolButton.InstantPopup)
        self.tsv_actions_menu = QMenu(self.tsv_actions_button)

        self.tsv_sort_action = QAction("Sort", self)
        self.tsv_sort_action.setEnabled(False)
        self.tsv_sort_action.setToolTip("Sort the scanned metadata table")
        self.tsv_sort_action.triggered.connect(self._open_sort_dialog)
        self.tsv_actions_menu.addAction(self.tsv_sort_action)

        self.tsv_load_action = QAction("Load TSV…", self)
        self.tsv_load_action.triggered.connect(self.selectAndLoadTSV)
        self.tsv_actions_menu.addAction(self.tsv_load_action)

        self.tsv_generate_ids_action = QAction("Generate unique IDs", self)
        self.tsv_generate_ids_action.setEnabled(False)
        self.tsv_generate_ids_action.triggered.connect(self.generateUniqueIDs)
        self.tsv_actions_menu.addAction(self.tsv_generate_ids_action)

        self.tsv_detect_rep_action = QAction("Detect repeats", self)
        self.tsv_detect_rep_action.triggered.connect(self.detectRepeatedSequences)
        self.tsv_actions_menu.addAction(self.tsv_detect_rep_action)

        self.tsv_save_action = QAction("Save changes", self)
        self.tsv_save_action.setEnabled(False)
        self.tsv_save_action.triggered.connect(self.applyMappingChanges)
        self.tsv_actions_menu.addAction(self.tsv_save_action)

        self.tsv_actions_button.setMenu(self.tsv_actions_menu)
        metadata_toolbar.addWidget(self.tsv_actions_button)
        metadata_toolbar.addStretch()
        metadata_layout.addLayout(metadata_toolbar)
        self.mapping_table = AutoFillTableWidget()
        # Expose immutable DICOM metadata (StudyDescription, FamilyName,
        # PatientID) alongside the editable identifiers so users can see the
        # original values while editing BIDS-specific fields.
        self.mapping_table.setColumnCount(16)
        self.mapping_table.setHorizontalHeaderLabels([
            "include",
            "source_folder",
            "StudyDescription",
            "FamilyName",
            "PatientID",
            "BIDS_name",
            "subject",
            "GivenName",
            "session",
            "sequence",
            "Proposed BIDS name",
            "series_uid",
            "acq_time",
            "rep",
            "modality",
            "modality_bids",
        ])
        hdr = self.mapping_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setStretchLastSection(True)
        self.mapping_table.verticalHeader().setVisible(False)
        # Keep BIDS name edits constrained by the delegate despite the shifted
        # column indices introduced above.
        self.mapping_table.setItemDelegateForColumn(5, SubjectDelegate(self.mapping_table))
        self.mapping_table.itemChanged.connect(self._updateDetectRepeatEnabled)
        self.mapping_table.itemChanged.connect(self._onMappingItemChanged)
        metadata_layout.addWidget(self.mapping_table)

        self.tsv_tabs.addTab(metadata_tab, "Scanned metadata")

        # --- Suffix dictionary tab ---
        dict_tab = QWidget()
        dict_layout = QVBoxLayout(dict_tab)
        toggle_row = QHBoxLayout()
        self.use_custom_patterns_box = QCheckBox("Use custom suffix patterns")
        self.use_custom_patterns_box.toggled.connect(self._on_custom_toggle)
        toggle_row.addWidget(self.use_custom_patterns_box)
        toggle_row.addStretch()
        dict_layout.addLayout(toggle_row)

        self.seq_tabs_widget = QTabWidget()
        dict_layout.addWidget(self.seq_tabs_widget)
        dict_btn_row = QHBoxLayout()
        dict_btn_row.addStretch()
        self.seq_save_button = QPushButton("Save")
        # Saving the suffix dictionary also applies the changes immediately so
        # users do not need a separate "Apply" step.
        self.seq_save_button.clicked.connect(self.saveSequenceDictionary)
        dict_btn_row.addWidget(self.seq_save_button)
        restore_btn = QPushButton("Restore defaults")
        restore_btn.clicked.connect(self.restoreSequenceDefaults)
        dict_btn_row.addWidget(restore_btn)
        dict_btn_row.addStretch()
        dict_layout.addLayout(dict_btn_row)

        self.tsv_tabs.addTab(dict_tab, "Suffix dictionary")
        self.loadSequenceDictionary()

        self.tsv_scroll = ShrinkableScrollArea()
        self.tsv_scroll.setWidgetResizable(True)
        self.tsv_scroll.setWidget(self.tsv_group)
        self.tsv_container = QWidget()
        tsv_container_layout = QVBoxLayout(self.tsv_container)
        tsv_container_layout.setContentsMargins(0, 0, 0, 0)
        tsv_container_layout.addWidget(self.tsv_scroll)

        self.left_split.addWidget(self.tsv_container)

        self.filter_group = QGroupBox("Filter")
        modal_layout = QVBoxLayout(self.filter_group)
        self.modal_tabs = QTabWidget()
        full_tab = QWidget()
        full_layout = QVBoxLayout(full_tab)
        self.full_tree = QTreeWidget()
        self.full_tree.setColumnCount(1)
        self.full_tree.setHeaderLabels(["BIDS Modality"])
        hdr = self.full_tree.header()
        for i in range(self.full_tree.columnCount()):
            hdr.setSectionResizeMode(i, QHeaderView.Interactive)
        for i in range(self.full_tree.columnCount()):
            self.full_tree.resizeColumnToContents(i)
        full_layout.addWidget(self.full_tree)
        self.modal_tabs.addTab(full_tab, "General view")

        specific_tab = QWidget()
        specific_layout = QVBoxLayout(specific_tab)
        self.specific_tree = QTreeWidget()
        self.specific_tree.setColumnCount(3)
        self.specific_tree.setHeaderLabels(["Study/Subject", "Files", "Time"])
        s_hdr = self.specific_tree.header()
        for i in range(self.specific_tree.columnCount()):
            s_hdr.setSectionResizeMode(i, QHeaderView.Interactive)
        for i in range(self.specific_tree.columnCount()):
            self.specific_tree.resizeColumnToContents(i)
        specific_layout.addWidget(self.specific_tree)
        self.last_rep_box = QCheckBox("Only last repeats")
        self.last_rep_box.setEnabled(False)
        self.last_rep_box.toggled.connect(self._onLastRepToggled)
        specific_layout.addWidget(self.last_rep_box)
        self.modal_tabs.addTab(specific_tab, "Specific view")

        naming_tab = QWidget()
        naming_layout = QVBoxLayout(naming_tab)
        self.naming_table = QTableWidget()
        self.naming_table.setColumnCount(3)
        self.naming_table.setHorizontalHeaderLabels(["Study", "Given name", "BIDS name"])
        n_hdr = self.naming_table.horizontalHeader()
        for i in range(self.naming_table.columnCount()):
            n_hdr.setSectionResizeMode(i, QHeaderView.Interactive)
        n_hdr.setStretchLastSection(True)
        for i in range(self.naming_table.columnCount()):
            self.naming_table.resizeColumnToContents(i)
        self.naming_table.setItemDelegateForColumn(2, SubjectDelegate(self.naming_table))
        naming_layout.addWidget(self.naming_table)
        self.naming_table.itemChanged.connect(self._onNamingEdited)
        self.name_choice = QComboBox()
        self.name_choice.addItems(["Use BIDS names", "Use given names"])
        self.name_choice.setEnabled(False)
        self.name_choice.currentIndexChanged.connect(self._onNameChoiceChanged)
        naming_layout.addWidget(self.name_choice)
        self.modal_tabs.addTab(naming_tab, "Edit naming")

        # Always Exclude tab
        exclude_tab = QWidget()
        exclude_layout = QVBoxLayout(exclude_tab)
        self.exclude_table = QTableWidget()
        self.exclude_table.setColumnCount(2)
        self.exclude_table.setHorizontalHeaderLabels(["Active", "Pattern"])
        ex_hdr = self.exclude_table.horizontalHeader()
        ex_hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        ex_hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        exclude_layout.addWidget(self.exclude_table)

        ex_add_row = QHBoxLayout()
        self.exclude_edit = QLineEdit()
        ex_add_row.addWidget(self.exclude_edit)
        ex_add_btn = QPushButton("Add")
        ex_add_btn.clicked.connect(self._exclude_add)
        ex_add_row.addWidget(ex_add_btn)
        exclude_layout.addLayout(ex_add_row)

        ex_save_btn = QPushButton("Save")
        ex_save_btn.clicked.connect(self.saveExcludePatterns)
        exclude_layout.addWidget(ex_save_btn, alignment=Qt.AlignRight)
        self.modal_tabs.addTab(exclude_tab, "Always exclude")

        # Load saved exclude patterns now that the table exists
        self.loadExcludePatterns()

        header_row_filter = QHBoxLayout()
        self.filter_detach_button = QPushButton("»")
        self.filter_detach_button.setFixedWidth(20)
        self.filter_detach_button.setFixedHeight(20)
        self.filter_detach_button.clicked.connect(self.detachFilterWindow)
        self.filter_detach_button.setFocusPolicy(Qt.NoFocus)
        header_row_filter.addStretch()
        header_row_filter.addWidget(self.filter_detach_button)
        modal_layout.addLayout(header_row_filter)
        modal_layout.addWidget(self.modal_tabs)

        self.filter_scroll = ShrinkableScrollArea()
        self.filter_scroll.setWidgetResizable(True)
        self.filter_scroll.setWidget(self.filter_group)
        self.filter_container = QWidget()
        filter_container_layout = QVBoxLayout(self.filter_container)
        filter_container_layout.setContentsMargins(0, 0, 0, 0)
        filter_container_layout.addWidget(self.filter_scroll)

        self.right_split.addWidget(self.filter_container)
        self.left_split.setStretchFactor(0, 1)
        self.left_split.setStretchFactor(1, 1)
        self.right_split.setStretchFactor(0, 1)
        self.right_split.setStretchFactor(1, 1)

        self.preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(self.preview_group)
        self.preview_tabs = QTabWidget()

        text_tab = QWidget()
        text_lay = QVBoxLayout(text_tab)
        self.preview_text = QTreeWidget()
        self.preview_text.setColumnCount(2)
        self.preview_text.setHeaderLabels(["BIDS Path", "Original Sequence"])
        text_lay.addWidget(self.preview_text)
        self.preview_tabs.addTab(text_tab, "Text")

        tree_tab = QWidget()
        tree_lay = QVBoxLayout(tree_tab)
        self.preview_tree = QTreeWidget()
        self.preview_tree.setColumnCount(2)
        self.preview_tree.setHeaderLabels(["BIDS Structure", "Original Sequence"])
        tree_lay.addWidget(self.preview_tree)
        self.preview_tabs.addTab(tree_tab, "Tree")

        header_row_preview = QHBoxLayout()
        self.preview_detach_button = QPushButton("»")
        self.preview_detach_button.setFixedWidth(20)
        self.preview_detach_button.setFixedHeight(20)
        self.preview_detach_button.clicked.connect(self.detachPreviewWindow)
        self.preview_detach_button.setFocusPolicy(Qt.NoFocus)
        header_row_preview.addStretch()
        header_row_preview.addWidget(self.preview_detach_button)
        preview_layout.addLayout(header_row_preview)
        preview_layout.addWidget(self.preview_tabs)
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.generatePreview)
        preview_layout.addWidget(self.preview_button)

        btn_row = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.runFullConversion)
        self.run_stop_button = QPushButton("Stop")
        self.run_stop_button.setEnabled(False)
        self.run_stop_button.clicked.connect(self.stopConversion)
        btn_row.addStretch()
        btn_row.addWidget(self.run_button)
        btn_row.addWidget(self.run_stop_button)
        btn_row.addStretch()

        # Combine preview panel and run button so the splitter keeps the
        # original layout but allows resizing versus the log output.
        self.preview_container = QWidget()
        pv_lay = QVBoxLayout(self.preview_container)
        pv_lay.setContentsMargins(0, 0, 0, 0)
        pv_lay.setSpacing(6)
        self.preview_scroll = ShrinkableScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setWidget(self.preview_group)
        pv_lay.addWidget(self.preview_scroll)
        pv_lay.addLayout(btn_row)

        log_group = QGroupBox("Log Output")
        log_layout = QVBoxLayout(log_group)
        self.terminal_cb = QCheckBox("Show output in terminal")
        log_layout.addWidget(self.terminal_cb)
        if sys.platform == "win32":
            # Always show terminal output on Windows and hide the option
            self.terminal_cb.setChecked(True)
            self.terminal_cb.setVisible(False)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.document().setMaximumBlockCount(1000)
        log_layout.addWidget(self.log_text)
        self.spinner_label = QLabel()
        self.spinner_label.setAlignment(Qt.AlignLeft)
        self.spinner_label.hide()
        log_layout.addWidget(self.spinner_label)

        self.left_split.addWidget(self.preview_container)
        self.right_split.addWidget(log_group)

        splitter = QSplitter()
        splitter.addWidget(self.left_split)
        splitter.addWidget(self.right_split)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, 1)

        self.tabs.addTab(self.convert_tab, "Converter")

    def _add_preview_path(self, parts, orig_seq):
        """Insert path components into the preview tree, storing ``orig_seq`` on the leaf."""
        # ``parts`` is a sequence of folder/file names comprising a BIDS path.
        parent = self.preview_tree.invisibleRootItem()
        for idx, part in enumerate(parts):
            match = None
            for i in range(parent.childCount()):
                child = parent.child(i)
                if child.text(0) == part:
                    match = child
                    break
            if match is None:
                match = QTreeWidgetItem([part, ""])  # second column filled on leaf
                parent.addChild(match)
            parent = match
            if idx == len(parts) - 1:
                parent.setText(1, orig_seq)

    def generatePreview(self):
        logging.info("generatePreview → Building preview tree …")
        """Populate preview tabs based on checked sequences."""
        self.preview_text.clear()
        self.preview_tree.clear()
        multi_study = len(self.study_set) > 1

        selected = []
        for i in range(self.mapping_table.rowCount()):
            if self.mapping_table.item(i, 0).checkState() == Qt.Checked:
                selected.append(self.row_info[i])

        rep_counts = defaultdict(int)
        for info in selected:
            subj_key = info['bids'] if self.use_bids_names else f"sub-{info['given']}"
            key = (subj_key, info['ses'], info['seq'])
            rep_counts[key] += 1

        for info in selected:
            subj = info['bids'] if self.use_bids_names else f"sub-{info['given']}"
            study = info['study']
            ses = info['ses']
            modb = info['modb']
            seq = info['seq']

            # Preview for DWI derivative maps moved to derivatives/
            tag = None
            if modb == 'dwi':
                seq_low = seq.lower()
                for t in ("adc", "fa", "tracew", "colfa"):
                    if t in seq_low:
                        tag = t.upper()
                        break
            if tag:
                path_parts = []
                if multi_study:
                    path_parts.append(study)
                path_parts.extend(["derivatives", DERIVATIVES_PIPELINE_NAME, subj, ses, "dwi"])
                fname_prefix = "_".join([p for p in [subj, ses] if p])
                fname = f"{fname_prefix}_desc-{tag}_dwi.nii.gz"
                full = [p for p in path_parts if p] + [fname]
                self.preview_text.addTopLevelItem(QTreeWidgetItem(["/".join(full), seq]))
                self._add_preview_path(full, seq)
                continue

            prop_dt = info.get('prop_dt')
            prop_base = info.get('prop_base')
            if prop_dt and prop_base:
                path_parts = []
                if multi_study:
                    path_parts.append(study)
                path_parts.extend([subj, ses, prop_dt])
                files = [f"{prop_base}.nii.gz"]
                if prop_base.endswith("_physio"):
                    files = [f"{prop_base}.tsv", f"{prop_base}.json"]
                for fname in files:
                    full = [p for p in path_parts if p] + [fname]
                    self.preview_text.addTopLevelItem(QTreeWidgetItem(["/".join(full), seq]))
                    self._add_preview_path(full, seq)
                continue

            path_parts = []
            if multi_study:
                path_parts.append(study)
            path_parts.extend([subj, ses, modb])

            base_parts = [subj, ses, seq]
            key = (subj, ses, seq)
            if rep_counts[key] > 1 and info['rep']:
                base_parts.append(f"rep-{info['rep']}")
            base = _dedup_parts(*base_parts)

            if modb == "fmap":
                for suffix in ["magnitude1", "magnitude2", "phasediff"]:
                    fname = f"{base}_{suffix}.nii.gz"
                    full = path_parts + [fname]
                    self.preview_text.addTopLevelItem(QTreeWidgetItem(["/".join(full), seq]))
                    self._add_preview_path(full, seq)
            else:
                fname = f"{base}.nii.gz"
                full = path_parts + [fname]
                self.preview_text.addTopLevelItem(QTreeWidgetItem(["/".join(full), seq]))
                self._add_preview_path(full, seq)

        self.preview_text.expandAll()
        self.preview_tree.expandAll()

    # (Rest of code remains unchanged)

    def initEditTab(self):
        """
        Set up Edit tab to embed the full functionality of bids_editor_ancpbids.
        """
        # This tab provides a file browser, statistics viewer and the metadata
        # editor used to inspect and modify BIDS sidecars.  It mirrors the
        # standalone "bids-editor" utility but is embedded in this application.
        self.edit_tab = QWidget()
        edit_layout = QVBoxLayout(self.edit_tab)
        edit_layout.setContentsMargins(10, 10, 10, 10)
        edit_layout.setSpacing(8)

        # Internal menu bar for Edit features
        menu = QMenuBar()
        menu.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        menu.setMaximumHeight(24)
        file_menu = menu.addMenu("File")
        open_act = QAction("Open BIDS…", self)
        open_act.triggered.connect(self.openBIDSForEdit)
        file_menu.addAction(open_act)
        tools_menu = menu.addMenu("Tools")
        rename_act = QAction("Batch Rename…", self)
        rename_act.triggered.connect(self.launchBatchRename)
        tools_menu.addAction(rename_act)

        intended_act = QAction("Set Intended For…", self)
        intended_act.triggered.connect(self.launchIntendedForEditor)
        tools_menu.addAction(intended_act)

        refresh_act = QAction("Refresh scans.tsv", self)
        refresh_act.triggered.connect(self.refreshScansTsv)
        tools_menu.addAction(refresh_act)

        ignore_act = QAction("Edit .bidsignore…", self)
        ignore_act.triggered.connect(self.launchBidsIgnore)
        tools_menu.addAction(ignore_act)
        edit_layout.addWidget(menu)

        # Splitter between left (tree & stats) and right (metadata)
        splitter = QSplitter()
        splitter.setHandleWidth(4)

        # Left panel: BIDSplorer and BIDStatistics
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        left_layout.addWidget(QLabel('<b>BIDSplorer</b>'))
        self.model = QFileSystemModel()
        self.model.setRootPath("")
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setEditTriggers(QAbstractItemView.EditKeyPressed | QAbstractItemView.SelectedClicked)
        self.tree.setColumnHidden(1, True)
        self.tree.setColumnHidden(3, True)
        hdr = self.tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        hdr.setSectionResizeMode(2, QHeaderView.Interactive)
        left_layout.addWidget(self.tree)
        self.tree.clicked.connect(self.onTreeClicked)

        left_layout.addWidget(QLabel('<b>BIDStatistics</b>'))
        self.stats = QTreeWidget()
        self.stats.setHeaderLabels(["Metric", "Value"])
        self.stats.setAlternatingRowColors(True)
        s_hdr = self.stats.header()
        s_hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        s_hdr.setSectionResizeMode(1, QHeaderView.Interactive)
        left_layout.addWidget(self.stats)

        splitter.addWidget(left_panel)

        # Right panel: MetadataViewer (reused from original)
        self.viewer = MetadataViewer()
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(1, 2)

        edit_layout.addWidget(splitter)
        self.tabs.addTab(self.edit_tab, "Editor")

    def selectDicomDir(self):
        """Select the raw DICOM input directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select DICOM Directory")
        if directory:
            self.dicom_dir = directory
            self.dicom_dir_edit.setText(directory)

    def selectBIDSOutDir(self):
        """Select (or create) the BIDS output directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select/Create BIDS Output Directory")
        if directory:
            self.bids_out_dir = directory
            self.bids_out_edit.setText(directory)
            self.loadExcludePatterns()

    def selectAndLoadTSV(self):
        """Choose an existing TSV and load it into the table."""
        path, _ = QFileDialog.getOpenFileName(self, "Select TSV", self.bids_out_dir or "", "TSV Files (*.tsv)")
        if path:
            self.tsv_path = path
            self.tsv_name_edit.setText(os.path.basename(path))
            self.loadMappingTable()

    def _open_sort_dialog(self) -> None:
        """Display a dialog allowing the user to configure a multi-level sort."""

        if self.mapping_table.rowCount() <= 1:
            QMessageBox.information(self, "Nothing to sort", "Load multiple rows before sorting.")
            return

        headers = []
        for col in range(self.mapping_table.columnCount()):
            header_item = self.mapping_table.horizontalHeaderItem(col)
            if header_item is not None:
                headers.append(header_item.text())

        dialog = MappingSortDialog(headers, self)
        if dialog.exec_() != QDialog.Accepted:
            return

        instructions = dialog.sort_instructions()
        if not instructions:
            QMessageBox.information(self, "No columns selected", "Select at least one column to sort by.")
            return

        self._apply_mapping_sort(instructions)

    def _apply_mapping_sort(self, instructions: list[tuple[str, bool]]) -> None:
        """Reorder the mapping table using ``instructions`` as sort keys."""

        column_lookup: dict[str, int] = {}
        for idx in range(self.mapping_table.columnCount()):
            header_item = self.mapping_table.horizontalHeaderItem(idx)
            if header_item is not None:
                column_lookup[header_item.text()] = idx

        # Translate column names back into indices, ignoring stale entries.
        order = [(column_lookup.get(name), asc) for name, asc in instructions if name in column_lookup]
        order = [(idx, asc) for idx, asc in order if idx is not None]
        if not order:
            return

        row_count = self.mapping_table.rowCount()
        col_count = self.mapping_table.columnCount()
        rows: list[dict[str, Any]] = []
        for row in range(row_count):
            row_items: list[QTableWidgetItem] = []
            sort_values: list[tuple[Any, ...]] = []
            for col in range(col_count):
                item = self.mapping_table.item(row, col)
                if item is not None:
                    cloned = item.clone()
                else:
                    cloned = QTableWidgetItem()
                if col == 0:
                    # Preserve checkbox behaviour in the include column.
                    cloned.setFlags((cloned.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
                    state = item.checkState() if item is not None else Qt.Unchecked
                    cloned.setCheckState(state)
                    sort_value = (0, 1 if state == Qt.Checked else 0)
                else:
                    text = item.text() if item is not None else ""
                    sort_value = self._coerce_sort_value(text)
                row_items.append(cloned)
                sort_values.append(sort_value)
            info = self.row_info[row] if row < len(self.row_info) else {}
            rows.append(
                {
                    "items": row_items,
                    "sort_values": sort_values,
                    "row_info": info,
                    "original_index": row,
                }
            )

        def _compare(left: dict[str, Any], right: dict[str, Any]) -> int:
            for col_idx, ascending in order:
                l_val = left["sort_values"][col_idx]
                r_val = right["sort_values"][col_idx]
                if l_val == r_val:
                    continue
                if l_val < r_val:
                    return -1 if ascending else 1
                return 1 if ascending else -1
            return 0

        sorted_rows = sorted(rows, key=cmp_to_key(_compare))
        new_order = [row["original_index"] for row in sorted_rows]
        if new_order == list(range(len(sorted_rows))):
            return  # Already sorted

        previous_loading = self._loading_mapping_table
        self._loading_mapping_table = True
        self.mapping_table.blockSignals(True)
        self.mapping_table.setRowCount(0)
        for row_data in sorted_rows:
            idx = self.mapping_table.rowCount()
            self.mapping_table.insertRow(idx)
            for col_idx, item in enumerate(row_data["items"]):
                self.mapping_table.setItem(idx, col_idx, item)
        self.mapping_table.blockSignals(False)
        self._loading_mapping_table = previous_loading

        # Keep auxiliary structures synchronised with the new row order.
        self.row_info = [row["row_info"] for row in sorted_rows]
        if self.inventory_df is not None:
            try:
                self.inventory_df = self.inventory_df.iloc[new_order].reset_index(drop=True)
            except Exception:
                pass

        self._rebuild_lookup_maps()
        self._schedule_mapping_refresh()

        if hasattr(self, "log_text"):
            summary = ", ".join(
                f"{name} ({'Ascending' if asc else 'Descending'})" for name, asc in instructions
            )
            self.log_text.append(f"Sorted metadata table by {summary}.")

    @staticmethod
    def _coerce_sort_value(value: str) -> tuple[Any, ...]:
        """Return a comparable tuple for ``value`` suitable for sorting."""

        text = value.strip()
        if not text:
            return (5, "")

        try:
            numeric = Decimal(text)
            return (1, numeric)
        except (InvalidOperation, ValueError):
            pass

        fmt = guess_datetime_format(text)
        if fmt:
            try:
                dt_value = pd.to_datetime(text, errors="raise")
                return (2, dt_value.to_pydatetime())
            except Exception:
                pass

        lowered = text.lower()
        return (3, lowered, text)

    def detachTSVWindow(self):
        """Detach the scanned data viewer into a separate window."""
        if getattr(self, "tsv_dialog", None):
            self.tsv_dialog.activateWindow()
            return
        self.tsv_dialog = QDialog(self, flags=Qt.Window)
        self.tsv_dialog.setWindowTitle("Scanned data viewer")
        lay = QVBoxLayout(self.tsv_dialog)
        self.tsv_container.setParent(None)
        lay.addWidget(self.tsv_container)
        self.tsv_dialog.finished.connect(self._reattachTSVWindow)
        self.tsv_dialog.showMaximized()

    def _reattachTSVWindow(self, *args):
        self.tsv_container.setParent(None)
        self.left_split.insertWidget(0, self.tsv_container)
        self.tsv_dialog = None

    def detachFilterWindow(self):
        """Detach the filter panel into a separate window."""
        if getattr(self, "filter_dialog", None):
            self.filter_dialog.activateWindow()
            return
        self.filter_dialog = QDialog(self, flags=Qt.Window)
        self.filter_dialog.setWindowTitle("Filter")
        lay = QVBoxLayout(self.filter_dialog)
        self.filter_container.setParent(None)
        lay.addWidget(self.filter_container)
        self.filter_dialog.finished.connect(self._reattachFilterWindow)
        self.filter_dialog.showMaximized()

    def _reattachFilterWindow(self, *args):
        self.filter_container.setParent(None)
        # Insert after the TSV panel but before preview_container
        self.right_split.insertWidget(0, self.filter_container)
        self.filter_dialog = None

    def detachPreviewWindow(self):
        """Detach the preview panel into a separate window."""
        if getattr(self, "preview_dialog", None):
            self.preview_dialog.activateWindow()
            return
        self.preview_dialog = QDialog(self, flags=Qt.Window)
        self.preview_dialog.setWindowTitle("Preview")
        lay = QVBoxLayout(self.preview_dialog)
        self.preview_container.setParent(None)
        lay.addWidget(self.preview_container)
        self.preview_dialog.finished.connect(self._reattachPreviewWindow)
        self.preview_dialog.showMaximized()

    def _reattachPreviewWindow(self, *args):
        self.preview_container.setParent(None)
        if self.left_split.indexOf(self.tsv_container) == -1:
            self.left_split.addWidget(self.preview_container)
        else:
            self.left_split.insertWidget(1, self.preview_container)
        self.preview_dialog = None

    def runInventory(self):
        logging.info("runInventory → Generating TSV …")
        """
        Scan DICOMs and generate subject_summary.tsv in the selected output directory.
        """
        if not self.dicom_dir or not os.path.isdir(self.dicom_dir):
            QMessageBox.warning(self, "Invalid DICOM Directory", "Please select a valid DICOM input directory.")
            return
        if not self.bids_out_dir:
            QMessageBox.warning(self, "No BIDS Output Directory", "Please select a BIDS output directory.")
            return

        os.makedirs(self.bids_out_dir, exist_ok=True)

        name = self.tsv_name_edit.text().strip() or "subject_summary.tsv"
        self.tsv_path = os.path.join(self.bids_out_dir, name)

        # Run dicom_inventory asynchronously
        if self.inventory_process and self.inventory_process.state() != QProcess.NotRunning:
            return

        # Remember to reapply the suffix dictionary automatically once the new
        # scan results have been loaded.  This ensures custom patterns take
        # effect immediately after a rescan without requiring another manual
        # save on the suffix tab.
        self._apply_sequence_on_load = self.use_custom_patterns_box.isChecked()

        # Clear the log so each scan run starts with a fresh history for the
        # user.
        self.log_text.clear()
        self.log_text.append("Starting TSV generation…")
        self.tsv_button.setEnabled(False)
        self.tsv_stop_button.setEnabled(True)
        self._start_spinner("Scanning files")
        self.inventory_process = QProcess(self)
        if self.terminal_cb.isChecked():
            # Forward stdout and stderr when the user wants to see terminal output
            self.inventory_process.setProcessChannelMode(QProcess.ForwardedChannels)
        else:
            # Discard output to avoid hangs on Windows when not showing the terminal
            self.inventory_process.setStandardOutputFile(QProcess.nullDevice())
            self.inventory_process.setStandardErrorFile(QProcess.nullDevice())
        self.inventory_process.finished.connect(self._inventoryFinished)
        args = [
            "-m",
            "bids_manager.dicom_inventory",
            self.dicom_dir,
            self.tsv_path,
            "--jobs",
            str(self.num_cpus),
        ]
        self.inventory_process.start(sys.executable, args)

    # ------------------------------------------------------------------
    # Helpers for detecting and reorganising multiple sessions in a
    # single folder
    # ------------------------------------------------------------------
    def _find_conflicting_studies(self, root_dir: str, n_jobs: int = 1) -> dict:
        """Return folders containing more than one StudyInstanceUID.

        Parameters
        ----------
        root_dir : str
            Top level directory that may contain mixed-session folders.

        Returns
        -------
        dict
            Mapping of folder path → {study_uid: [file1, file2, ...]} for
            folders that contain DICOMs from multiple sessions.
        """

        conflicts: dict[str, dict[str, list[str]]] = {}

        # Gather the folders containing DICOM files so we can evaluate them in
        # parallel without paying the scheduling cost for empty directories.
        folders_to_scan: List[Tuple[str, List[str]]] = []
        for folder, _dirs, files in os.walk(root_dir):
            dicom_files: List[str] = []
            for fname in files:
                fpath = os.path.join(folder, fname)
                if is_dicom_file(fpath):
                    dicom_files.append(fpath)
            if dicom_files:
                folders_to_scan.append((folder, dicom_files))

        def _scan_folder(data: Tuple[str, List[str]]) -> Optional[Tuple[str, Dict[str, List[str]]]]:
            folder, dicom_files = data
            study_map: Dict[str, List[str]] = {}
            for fpath in dicom_files:
                try:
                    ds = pydicom.dcmread(
                        fpath,
                        stop_before_pixels=True,
                        specific_tags=["StudyInstanceUID"],
                    )
                    uid = str(getattr(ds, "StudyInstanceUID", "")).strip()
                except Exception:
                    # Maintain the previous best-effort behaviour: unreadable
                    # files are skipped silently so the scan keeps running.
                    continue
                study_map.setdefault(uid, []).append(fpath)
            if len(study_map) > 1:
                return folder, study_map
            return None

        workers = max(1, n_jobs)
        if workers == 1:
            results = (_scan_folder(entry) for entry in folders_to_scan)
        else:
            # ``Parallel`` confines the evaluation to the configured number of
            # workers, matching the CPU limit used for the initial DICOM scan.
            results = Parallel(n_jobs=workers)(
                delayed(_scan_folder)(entry) for entry in folders_to_scan
            )

        for result in results:
            if result is None:
                continue
            folder, study_map = result
            conflicts[folder] = study_map

        return conflicts

    def _reorganize_conflicting_sessions(self, conflicts: dict) -> None:
        """Move files for **all** sessions into separate subfolders.

        This method ensures that each unique ``StudyInstanceUID`` found within a
        folder is placed in its own subdirectory.  Previously only the
        additional sessions were moved, leaving the first session's files in the
        root folder, which could lead to HeuDiConv processing multiple sessions
        together and crashing.  By relocating every session we guarantee a clean
        one-session-per-folder layout.

        Parameters
        ----------
        conflicts : dict
            Output of :meth:`_find_conflicting_studies` mapping folder paths to
            ``StudyInstanceUID`` → list of files.
        """

        for folder, uid_map in conflicts.items():
            # Iterate over each StudyInstanceUID and move its files into a
            # unique ``sessionX`` directory.
            for idx, (uid, files) in enumerate(uid_map.items(), start=1):
                # Determine a unique destination directory.  We increment the
                # numeric suffix if a folder with the same name already exists.
                new_dir = os.path.join(folder, f"session{idx}")
                suffix = idx
                while os.path.exists(new_dir):
                    suffix += 1
                    new_dir = os.path.join(folder, f"session{suffix}")
                os.makedirs(new_dir, exist_ok=True)

                # Move each file belonging to the current UID into ``new_dir``.
                for fpath in files:
                    shutil.move(fpath, os.path.join(new_dir, os.path.basename(fpath)))

                self.log_text.append(
                    f"Moved {len(files)} files with StudyInstanceUID {uid} to {new_dir}."
                )

    def _cleanup_conflict_worker(self) -> None:
        """Release resources used by the background conflict scanner."""

        if self._conflict_worker is not None:
            self._conflict_worker.deleteLater()
            self._conflict_worker = None
        if self._conflict_thread is not None:
            if self._conflict_thread.isRunning():
                self._conflict_thread.quit()
                self._conflict_thread.wait()
            self._conflict_thread.deleteLater()
            self._conflict_thread = None

    def _start_conflict_scan(self) -> None:
        """Check for mixed sessions without blocking the GUI thread."""

        if not self.dicom_dir or not os.path.isdir(self.dicom_dir):
            # Nothing to scan – proceed directly to loading the generated TSV.
            self.log_text.append("TSV generation finished.")
            self.loadMappingTable()
            return

        # Avoid launching multiple background scanners simultaneously.
        if self._conflict_thread is not None:
            if not self._conflict_thread.isRunning():
                self._cleanup_conflict_worker()
            else:
                return

        self._start_spinner("Checking sessions")
        self.log_text.append("Checking for multiple sessions in scan folders…")

        self._conflict_worker = _ConflictScannerWorker(
            self.dicom_dir,
            self._find_conflicting_studies,
            self.num_cpus,
        )
        self._conflict_thread = QThread(self)
        self._conflict_worker.moveToThread(self._conflict_thread)
        self._conflict_thread.started.connect(self._conflict_worker.run)
        self._conflict_worker.finished.connect(self._on_conflict_scan_finished)
        self._conflict_worker.failed.connect(self._on_conflict_scan_failed)
        self._conflict_thread.start()

    def _on_conflict_scan_finished(self, conflicts: dict) -> None:
        """Handle successful completion of the background conflict scan."""

        self._stop_spinner()
        self._cleanup_conflict_worker()
        if conflicts:
            folders = "\n".join(conflicts.keys())
            msg = (
                "Multiple sessions were detected in the following folders:\n"
                f"{folders}\n\n"
                "Would you like to move each session into its own subfolder?"
            )
            resp = QMessageBox.question(
                self,
                "Multiple sessions detected",
                msg,
                QMessageBox.Yes | QMessageBox.No,
            )
            if resp == QMessageBox.Yes:
                self._reorganize_conflicting_sessions(conflicts)
                # Re-run the inventory now that the folders have been separated.
                self.runInventory()
                return

        self.log_text.append("TSV generation finished.")
        self.loadMappingTable()

    def _on_conflict_scan_failed(self, error_message: str) -> None:
        """Fallback when conflict detection fails for any reason."""

        self._stop_spinner()
        self._cleanup_conflict_worker()
        logging.error("Conflict detection failed: %s", error_message)
        self.log_text.append("Failed to check for mixed sessions; continuing anyway.")
        self.log_text.append("TSV generation finished.")
        self.loadMappingTable()

    def _inventoryFinished(self):
        ok = self.inventory_process.exitCode() == 0 if self.inventory_process else False
        self.inventory_process = None
        self.tsv_button.setEnabled(True)
        self.tsv_stop_button.setEnabled(False)
        self._stop_spinner()
        if ok:
            self._start_conflict_scan()
        else:
            self.log_text.append("TSV generation failed.")
            # Avoid auto-applying the suffix dictionary when the scan failed;
            # the mapping table will not be refreshed and we should not reuse
            # the pending flag for future loads.
            self._apply_sequence_on_load = False

    def stopInventory(self):
        if self.inventory_process and self.inventory_process.state() != QProcess.NotRunning:
            pid = int(self.inventory_process.processId())
            _terminate_process_tree(pid)
            self.inventory_process = None
            self.tsv_button.setEnabled(True)
            self.tsv_stop_button.setEnabled(False)
            self._stop_spinner()
            self.log_text.append("TSV generation cancelled.")
            # Reset pending suffix reapplication because no new TSV will be
            # loaded after a cancellation.
            self._apply_sequence_on_load = False

    def applyMappingChanges(self):
        """Save edits in the scanned data table back to the TSV and refresh."""
        if not self.tsv_path or not os.path.isfile(self.tsv_path):
            return
        try:
            df = pd.read_csv(self.tsv_path, sep="\t", keep_default_na=False)
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Failed to load TSV: {exc}")
            return
        if df.shape[0] != self.mapping_table.rowCount():
            QMessageBox.warning(self, "Error", "Row count mismatch")
            return
        for i in range(self.mapping_table.rowCount()):
            df.at[i, "include"] = 1 if self.mapping_table.item(i, 0).checkState() == Qt.Checked else 0
            df.at[i, "source_folder"] = self.mapping_table.item(i, 1).text()
            study_item = self.mapping_table.item(i, 2)
            study_text = study_item.text() if study_item is not None else ""
            df.at[i, "StudyDescription"] = normalize_study_name(study_text.strip())
            df.at[i, "FamilyName"] = self.mapping_table.item(i, 3).text()
            df.at[i, "PatientID"] = self.mapping_table.item(i, 4).text()
            df.at[i, "BIDS_name"] = self.mapping_table.item(i, 5).text()
            df.at[i, "subject"] = self.mapping_table.item(i, 6).text()
            df.at[i, "GivenName"] = self.mapping_table.item(i, 7).text()
            df.at[i, "session"] = self.mapping_table.item(i, 8).text()
            df.at[i, "sequence"] = self.mapping_table.item(i, 9).text()
            df.at[i, "Proposed BIDS name"] = self.mapping_table.item(i, 10).text()
            df.at[i, "series_uid"] = self.mapping_table.item(i, 11).text()
            df.at[i, "acq_time"] = self.mapping_table.item(i, 12).text()
            df.at[i, "rep"] = self.mapping_table.item(i, 13).text()
            df.at[i, "modality"] = self.mapping_table.item(i, 14).text()
            df.at[i, "modality_bids"] = self.mapping_table.item(i, 15).text()

        # When editing the scanned data table we assume the user knows what
        # they are doing, so we do not enforce BIDS naming rules or uniqueness
        # here. Validation is still performed when editing via the naming table
        # and filter fields.
        try:
            df.to_csv(self.tsv_path, sep="\t", index=False)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save TSV: {exc}")
            return
        self.loadMappingTable()

    def generateUniqueIDs(self):
        """Assign random 3-letter/3-digit IDs to subjects without an identifier."""
        # Load previously assigned IDs from all studies in the output directory
        existing: dict[str, dict[str, str]] = {}
        existing_ids: set[str] = set()
        out_dir = Path(self.bids_out_dir)
        if out_dir.is_dir():
            for study_dir in out_dir.iterdir():
                if not study_dir.is_dir():
                    continue
                s_path = study_dir / ".bids_manager" / "subject_summary.tsv"
                if s_path.exists():
                    try:
                        sdf = pd.read_csv(s_path, sep="\t", keep_default_na=False)
                        for _, row in sdf.iterrows():
                            study_desc = str(row.get("StudyDescription", "")).strip()
                            bids_name = str(row.get("BIDS_name", "")).strip()
                            sid = str(row.get("subject", "")).strip() or str(row.get("GivenName", "")).strip()
                            if study_desc and bids_name and sid:
                                existing.setdefault(study_desc, {})[bids_name] = sid
                                existing_ids.add(sid)
                    except Exception:
                        pass

        id_map: dict[tuple[str, str], str] = {}
        for i in range(self.mapping_table.rowCount()):
            bids = self.mapping_table.item(i, 5).text().strip()
            study = self.mapping_table.item(i, 2).text().strip()
            if not bids:
                continue

            subj_item = self.mapping_table.item(i, 6)
            given_item = self.mapping_table.item(i, 7)

            sid = None
            prior = existing.get(study, {}).get(bids)
            if prior and subj_item.text().strip() == "" and given_item.text().strip() == "":
                resp = QMessageBox.question(
                    self,
                    "Subject exists",
                    "This subject already exist in the study. Would you like to use the same unique ID?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if resp == QMessageBox.Yes:
                    sid = prior

            if sid is None:
                key = (study, bids)
                if key not in id_map:
                    id_map[key] = _random_subject_id(existing_ids | set(id_map.values()))
                sid = id_map[key]

            if subj_item.text().strip() == "":
                subj_item.setText(sid)
            if given_item.text().strip() == "":
                given_item.setText(sid)
            self.row_info[i]['given'] = given_item.text()
            existing_ids.add(sid)

        self._rebuild_lookup_maps()
        self._schedule_mapping_refresh()
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.generatePreview)
        QTimer.singleShot(0, self._updateDetectRepeatEnabled)
        QTimer.singleShot(0, self._updateMappingControlsEnabled)
        QTimer.singleShot(0, self._auto_apply_existing_study_mappings)

    def _updateDetectRepeatEnabled(self, _item=None):
        """Enable repeat detection when BIDS and Given names are filled."""
        if not hasattr(self, "tsv_detect_rep_action"):
            return
        enabled = self.mapping_table.rowCount() > 0
        if enabled:
            for r in range(self.mapping_table.rowCount()):
                bids = self.mapping_table.item(r, 5)
                given = self.mapping_table.item(r, 7)
                if bids is None or given is None or not bids.text().strip() or not given.text().strip():
                    enabled = False
                    break
        self.tsv_detect_rep_action.setEnabled(enabled)

    @staticmethod
    def _is_visual_only_sequence(sequence: str, modality: str = "") -> bool:
        """Return ``True`` when the sequence should remain view-only."""

        seq_lower = (sequence or "").lower()
        mod_lower = (modality or "").lower()
        # The check is intentionally substring-based so variations such as
        # "Phoenix Report" or "Scout Image" are also captured.
        visual_tokens = ("phoenix report", "scout", "report")
        return any(tok in seq_lower for tok in visual_tokens) or any(
            tok in mod_lower for tok in visual_tokens
        )

    def _apply_visual_only_rules(self, row: int) -> None:
        """Disable inclusion toggles for rows that must stay view-only."""

        if not (0 <= row < self.mapping_table.rowCount()):
            return
        include_item = self.mapping_table.item(row, 0)
        if include_item is None or row >= len(self.row_info):
            return

        is_visual_only = bool(self.row_info[row].get("visual_only"))
        # Temporarily silence table signals while we adjust checkbox flags so
        # we do not trigger downstream refreshes.
        prev_block = self.mapping_table.signalsBlocked()
        self.mapping_table.blockSignals(True)
        try:
            if is_visual_only:
                include_item.setCheckState(Qt.Unchecked)
                include_item.setFlags(
                    (include_item.flags() & ~Qt.ItemIsEditable)
                    & ~Qt.ItemIsUserCheckable
                    & ~Qt.ItemIsEnabled
                )
                include_item.setToolTip(
                    "Reports and scout images are shown for reference only."
                )
            else:
                include_item.setFlags(
                    (include_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                    & ~Qt.ItemIsEditable
                )
                include_item.setToolTip("")
        finally:
            self.mapping_table.blockSignals(prev_block)

    def _count_non_visual_repeats(self) -> int:
        """Return the number of repeated acquisitions excluding visual-only rows."""

        count = 0
        for info in self.row_info:
            if info.get("visual_only"):
                continue
            rep_val = str(info.get("rep") or "").strip()
            if rep_val.isdigit() and int(rep_val) > 1:
                count += 1
        return count

    def _maybe_notify_repeats(self) -> None:
        """Inform the user when new repeats are detected and sync selection."""

        if not hasattr(self, "last_rep_box"):
            return
        repeat_count = self._count_non_visual_repeats()
        if repeat_count > 0:
            if not self.last_rep_box.isChecked():
                self.last_rep_box.setChecked(True)
            if self._last_repeat_count == 0:
                QMessageBox.information(
                    self,
                    "Repeated sequences detected",
                    (
                        "Repeated acquisitions were found. The \"Only last repeats\" "
                        "option is now active to keep the latest instance selected."
                    ),
                )
        self._last_repeat_count = repeat_count

    def _auto_apply_existing_study_mappings(self) -> None:
        """Queue a silent sync of BIDS names with existing output datasets."""

        # ``QTimer.singleShot`` triggers this helper outside of the current
        # signal handler, so keep the method lightweight and delegate the heavy
        # lifting to the shared implementation below.
        self._apply_existing_study_mappings(silent=True)

    def _apply_existing_study_mappings(self, *, silent: bool = True) -> None:
        """Align BIDS subject names with prior conversions stored on disk.

        Parameters
        ----------
        silent
            When ``True`` the method skips modal warnings if the BIDS output
            directory has not been configured yet.  This is the default for the
            automatic calls that happen after scans or when generating IDs.
        """

        out_dir = Path(self.bids_out_dir or "")
        if not out_dir.is_dir():
            if silent:
                logging.info(
                    "Skipping existing-study sync because no BIDS output "
                    "directory is available."
                )
            else:
                QMessageBox.warning(
                    self,
                    "No BIDS Output Directory",
                    "Please select a BIDS output directory.",
                )
            return

        if self.naming_table.rowCount() == 0:
            return

        # Gather the studies present in the naming table so we only touch the
        # datasets that are relevant for the current scan.
        studies: set[str] = set()
        for row in range(self.naming_table.rowCount()):
            study_item = self.naming_table.item(row, 0)
            if study_item is None:
                continue
            study_text = study_item.text().strip()
            if study_text:
                studies.add(study_text)

        if not studies:
            return

        # ``existing_by_given`` and ``existing_by_uid`` map (study, key) pairs
        # to the BIDS names already present on disk.  We keep track of both the
        # ``GivenName`` (which becomes the generated unique ID) and the
        # ``subject`` column written by the converter so that we can preserve
        # names even if only one identifier is available.
        existing_by_given: dict[tuple[str, str], str] = {}
        existing_by_uid: dict[tuple[str, str], str] = {}
        used_by_study: dict[str, set[str]] = {}
        has_existing: dict[str, bool] = {}

        for study in studies:
            safe = _safe_stem(str(study))
            s_path = out_dir / safe / ".bids_manager" / "subject_summary.tsv"
            has_existing[study] = s_path.exists()
            if not s_path.exists():
                continue
            try:
                df = pd.read_csv(s_path, sep="\t", keep_default_na=False)
            except Exception as exc:  # pragma: no cover - runtime resilience
                logging.warning("Failed to read %s: %s", s_path, exc)
                continue

            for _, row in df.iterrows():
                given_val = str(row.get("GivenName", "")).strip()
                bids_val = str(row.get("BIDS_name", "")).strip()
                uid_val = str(row.get("subject", "")).strip()
                if bids_val:
                    used_by_study.setdefault(study, set()).add(bids_val)
                if given_val and bids_val:
                    existing_by_given[(study, given_val)] = bids_val
                if uid_val and bids_val:
                    existing_by_uid[(study, uid_val)] = bids_val

        if not any(has_existing.values()):
            # Nothing to reconcile – leave the manually assigned names untouched.
            return

        self.naming_table.blockSignals(True)
        self.mapping_table.blockSignals(True)

        for row in range(self.naming_table.rowCount()):
            study_item = self.naming_table.item(row, 0)
            given_item = self.naming_table.item(row, 1)
            bids_item = self.naming_table.item(row, 2)
            if None in (study_item, given_item, bids_item):
                continue

            study = study_item.text().strip()
            given = given_item.text().strip()
            current = bids_item.text().strip()
            if not study:
                continue

            # Derive the subject unique ID from the mapping table if present.
            subject_uid = ""
            for idx, info in enumerate(self.row_info):
                if info['study'] == study and info['given'] == given:
                    subj_col = self.mapping_table.item(idx, 6)
                    if subj_col is not None:
                        subject_uid = subj_col.text().strip()
                    break

            used = used_by_study.setdefault(study, set()).copy()

            mapped = None
            if subject_uid:
                mapped = existing_by_uid.get((study, subject_uid))
            if mapped is None and given:
                mapped = existing_by_given.get((study, given))

            if mapped:
                new_bids = mapped
            elif not has_existing.get(study, False):
                new_bids = current
            else:
                if current and current not in used:
                    new_bids = current
                else:
                    new_bids = _next_numeric_id(used)

            if new_bids != current:
                bids_item.setText(new_bids)

            # Update downstream structures so future duplicate detection and
            # preview generation operate on the adjusted BIDS identifiers.
            used_by_study.setdefault(study, set()).add(new_bids)
            for idx, info in enumerate(self.row_info):
                if info['study'] == study and info['given'] == given:
                    info['bids'] = new_bids
                    table_item = self.mapping_table.item(idx, 5)
                    if table_item is not None:
                        table_item.setText(new_bids)

            key_for_map = given or subject_uid
            if key_for_map:
                self.existing_maps.setdefault(study, {})[key_for_map] = new_bids
            if subject_uid and subject_uid != given:
                self.existing_maps.setdefault(study, {})[subject_uid] = new_bids
            self.existing_used.setdefault(study, set()).add(new_bids)

        self.naming_table.blockSignals(False)
        self.mapping_table.blockSignals(False)

        self._rebuild_lookup_maps()
        QTimer.singleShot(0, self.populateModalitiesTree)
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.generatePreview)
        QTimer.singleShot(0, self._updateDetectRepeatEnabled)
        QTimer.singleShot(0, self._updateMappingControlsEnabled)

    def _updateMappingControlsEnabled(self):
        """Enable controls that require scanned data."""
        if not hasattr(self, "tsv_generate_ids_action"):
            return
        has_data = self.mapping_table.rowCount() > 0
        self.tsv_generate_ids_action.setEnabled(has_data)
        if hasattr(self, "tsv_sort_action"):
            self.tsv_sort_action.setEnabled(has_data)
        if hasattr(self, "tsv_save_action"):
            self.tsv_save_action.setEnabled(has_data)
        self.last_rep_box.setEnabled(has_data)
        self.name_choice.setEnabled(has_data)
        if not has_data:
            self.last_rep_box.setChecked(False)

    def _sync_row_info_from_table(self, row: int) -> None:
        """Update cached ``row_info`` details after an in-place table edit."""
        if not (0 <= row < len(self.row_info)):
            return

        def _text(col: int) -> str:
            item = self.mapping_table.item(row, col)
            return item.text().strip() if item is not None else ""

        info = self.row_info[row]
        info['study'] = _text(2)
        info['bids'] = _text(5)
        info['given'] = _text(7)
        info['ses'] = _text(8)
        info['seq'] = _text(9)
        info['rep'] = _text(13)
        info['mod'] = _text(14)
        info['modb'] = _text(15)
        info['visual_only'] = self._is_visual_only_sequence(info['seq'], info['mod'])
        self._apply_visual_only_rules(row)

    def _schedule_mapping_refresh(self) -> None:
        """Queue UI updates that reflect the current mapping table."""
        QTimer.singleShot(0, self.populateModalitiesTree)
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.generatePreview)
        QTimer.singleShot(0, self._updateDetectRepeatEnabled)
        QTimer.singleShot(0, self._updateMappingControlsEnabled)

    def _update_study_set(self) -> None:
        """Recompute list of studies present in the mapping table."""
        self.study_set = set()
        for r in range(self.mapping_table.rowCount()):
            study_item = self.mapping_table.item(r, 2)
            if study_item is None:
                continue
            study_text = study_item.text().strip()
            if study_text:
                self.study_set.add(study_text)

    def _onMappingItemChanged(self, item):
        """Handle edits in the scanned data table."""
        if self._loading_mapping_table or item is None:
            return

        column = item.column()
        row = item.row()

        if column == 2:
            raw_text = item.text().strip()
            cleaned = normalize_study_name(raw_text)

            if cleaned != item.text():
                # Temporarily block signals so updating the text does not trigger
                # additional ``itemChanged`` notifications.
                self.mapping_table.blockSignals(True)
                item.setText(cleaned)
                self.mapping_table.blockSignals(False)

            self._sync_row_info_from_table(row)

            bids_item = self.mapping_table.item(row, 5)
            if bids_item is not None:
                bids_item.setData(Qt.UserRole, cleaned)

            self._update_study_set()
            self._rebuild_lookup_maps()
            self._schedule_mapping_refresh()
            return

        if column == 0:
            # Checkbox state changed; refresh dependent views.
            self._schedule_mapping_refresh()
            return

        if column in {5, 7, 8, 9, 13, 14, 15}:
            self._sync_row_info_from_table(row)
            self._rebuild_lookup_maps()
            self._schedule_mapping_refresh()

    def detectRepeatedSequences(self):
        """Detect repeated sequences within each subject and assign numbers."""
        if self.mapping_table.rowCount() == 0:
            return

        rows = []
        for i in range(self.mapping_table.rowCount()):
            rows.append({
                'StudyDescription': self.mapping_table.item(i, 2).text().strip(),
                'BIDS_name': self.mapping_table.item(i, 5).text().strip(),
                'session': self.mapping_table.item(i, 8).text().strip(),
                'modality_bids': self.mapping_table.item(i, 15).text().strip(),
                'modality': self.mapping_table.item(i, 14).text().strip(),
                'sequence': self.mapping_table.item(i, 9).text().strip(),
                'acq_time': self.mapping_table.item(i, 12).text().strip(),
            })

        df = pd.DataFrame(rows)
        df['acq_sort'] = pd.to_numeric(df['acq_time'].str.replace(':', ''), errors='coerce')
        key_cols = ['StudyDescription', 'BIDS_name', 'session', 'modality_bids', 'modality', 'sequence']
        df.sort_values(['acq_sort'], inplace=True)
        df['rep'] = df.groupby(key_cols).cumcount() + 1
        counts = df.groupby(key_cols)['rep'].transform('count')
        df.loc[counts == 1, 'rep'] = ''
        df.loc[(counts > 1) & (df['rep'] == 1), 'rep'] = ''

        for i in range(self.mapping_table.rowCount()):
            val = df.at[i, 'rep']
            self.mapping_table.item(i, 13).setText(str(val) if str(val) else '')
            self.row_info[i]['rep'] = str(val) if str(val) else ''

        self._rebuild_lookup_maps()
        self._maybe_notify_repeats()
        QTimer.singleShot(0, self.populateModalitiesTree)
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.generatePreview)

    def scanExistingStudies(self):
        """Manually trigger a sync with existing converted studies."""

        # Keep a public wrapper so automated tests and power users can still
        # force a rescan from the console if needed.  The heavy lifting is
        # performed by :meth:`_apply_existing_study_mappings` which is also
        # used by the automated workflows.
        self._apply_existing_study_mappings(silent=False)


    def loadMappingTable(self):
        logging.info("loadMappingTable → Loading TSV into table …")
        """
        Load the generated TSV into the mapping_table for user editing.
        Columns: include, source_folder, StudyDescription, FamilyName,
        PatientID, BIDS_name, subject, GivenName, session, sequence,
        Proposed BIDS name, series_uid, acq_time, rep, modality, modality_bids
        """
        if not self.tsv_path or not os.path.isfile(self.tsv_path):
            return

        self._loading_mapping_table = True
        try:
            df = pd.read_csv(self.tsv_path, sep="\t", keep_default_na=False)

            # Respect custom suffix patterns when building preview names by
            # re-deriving modalities with the active dictionary before
            # computing BIDS proposals.  This keeps freshly scanned tables in
            # sync with the "Use custom suffix patterns" toggle without
            # requiring a manual "Save" on the suffix tab first.
            if self.use_custom_patterns_box.isChecked():
                for idx, row in df.iterrows():
                    seq = str(row.get("sequence") or "")
                    mod = dicom_inventory.guess_modality(seq)
                    modb = dicom_inventory.modality_to_container(mod)
                    df.at[idx, "modality"] = mod
                    df.at[idx, "modality_bids"] = modb

            preview_map = _compute_bids_preview(df, self._schema)
            df["proposed_datatype"] = [preview_map.get(i, ("", ""))[0] for i in df.index]
            df["proposed_basename"] = [preview_map.get(i, ("", ""))[1] for i in df.index]

            def _prop_path(r):
                base = r.get("proposed_basename")
                dt = r.get("proposed_datatype")
                if not base:
                    return ""
                ext = ".tsv" if str(base).endswith("_physio") else ".nii.gz"
                return f"{dt}/{base}{ext}"

            df["Proposed BIDS name"] = df.apply(_prop_path, axis=1)
            self.inventory_df = df

            # ----- load existing mappings without altering the TSV -----
            self.existing_maps = {}
            self.existing_used = {}
            studies = df["StudyDescription"].fillna("").unique()
            for study in studies:
                safe = _safe_stem(str(study))
                mpath = Path(self.bids_out_dir) / safe / ".bids_manager" / "subject_mapping.tsv"
                mapping = {}
                used = set()
                if mpath.exists():
                    try:
                        mdf = pd.read_csv(mpath, sep="\t", keep_default_na=False)
                        mapping = dict(zip(mdf["GivenName"].astype(str), mdf["BIDS_name"].astype(str)))
                        used = set(mapping.values())
                    except Exception:
                        pass
                # Store mapping info so we can validate name edits later on
                self.existing_maps[study] = mapping
                self.existing_used[study] = used

            self.study_set.clear()
            self.modb_rows.clear()
            self.mod_rows.clear()
            self.seq_rows.clear()
            self.study_rows.clear()
            self.subject_rows.clear()
            self.session_rows.clear()
            self.spec_modb_rows.clear()
            self.spec_mod_rows.clear()
            self.spec_seq_rows.clear()
            self.row_info = []

            # Populate table rows
            self.mapping_table.setRowCount(0)

            def _clean(val):
                """Return string representation of val or empty string for NaN."""
                return "" if pd.isna(val) else str(val)

            for _, row in df.iterrows():
                r = self.mapping_table.rowCount()
                self.mapping_table.insertRow(r)
                include_item = QTableWidgetItem()
                include_item.setFlags(
                    (include_item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable
                )
                include_item.setCheckState(
                    Qt.Checked if row.get('include', 1) == 1 else Qt.Unchecked
                )
                self.mapping_table.setItem(r, 0, include_item)

                src_item = QTableWidgetItem(_clean(row.get('source_folder')))
                src_item.setFlags(src_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 1, src_item)

                study_raw = _clean(row.get('StudyDescription'))
                study = normalize_study_name(study_raw)

                study_item = QTableWidgetItem(study)
                study_item.setFlags(study_item.flags() | Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 2, study_item)

                family_item = QTableWidgetItem(_clean(row.get('FamilyName')))
                family_item.setFlags(family_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 3, family_item)

                patient_item = QTableWidgetItem(_clean(row.get('PatientID')))
                patient_item.setFlags(patient_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 4, patient_item)

                bids_name = _clean(row.get('BIDS_name'))
                bids_item = QTableWidgetItem(bids_name)
                bids_item.setFlags(bids_item.flags() | Qt.ItemIsEditable)
                bids_item.setData(Qt.UserRole, study)
                self.study_set.add(study)
                self.mapping_table.setItem(r, 5, bids_item)

                subj_item = QTableWidgetItem(_clean(row.get('subject')))
                # Subject identifiers come from generated mappings or utilities
                # such as ``generateUniqueIDs``; keep them read-only in the
                # scanned data viewer so users do not alter them manually.
                subj_item.setFlags(subj_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 6, subj_item)

                given_item = QTableWidgetItem(_clean(row.get('GivenName')))
                # "GivenName" (the generated unique ID) must not be edited
                # directly in the scanned data table to avoid inconsistencies
                # with the subject mapping utilities.
                given_item.setFlags(given_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 7, given_item)

                session = _clean(row.get('session'))
                ses_item = QTableWidgetItem(session)
                ses_item.setFlags(ses_item.flags() | Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 8, ses_item)

                seq_item = QTableWidgetItem(_clean(row.get('sequence')))
                # Preserve the original DICOM sequence information as read-only
                # so the viewer remains a faithful reflection of the scan data.
                seq_item.setFlags(seq_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 9, seq_item)

                preview_item = QTableWidgetItem(_clean(row.get('Proposed BIDS name')))
                preview_item.setFlags(preview_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 10, preview_item)

                uid_item = QTableWidgetItem(_clean(row.get('series_uid')))
                uid_item.setFlags(uid_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 11, uid_item)

                acq_item = QTableWidgetItem(_clean(row.get('acq_time')))
                acq_item.setFlags(acq_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 12, acq_item)

                rep_item = QTableWidgetItem(_clean(row.get('rep')))
                # Allow editing the repeat number directly in the table
                rep_item.setFlags(rep_item.flags() | Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 13, rep_item)

                mod_item = QTableWidgetItem(_clean(row.get('modality')))
                mod_item.setFlags(mod_item.flags() & ~Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 14, mod_item)

                modb = _clean(row.get('modality_bids'))
                modb_item = QTableWidgetItem(modb)
                modb_item.setFlags(modb_item.flags() | Qt.ItemIsEditable)
                self.mapping_table.setItem(r, 15, modb_item)

                mod = _clean(row.get('modality'))
                seq = _clean(row.get('sequence'))
                run = _clean(row.get('rep'))
                given = _clean(row.get('GivenName'))
                prop_dt = _clean(row.get('proposed_datatype'))
                prop_base = _clean(row.get('proposed_basename'))
                self.row_info.append({
                    'study': study,
                    'bids': bids_name,
                    'given': given,
                    'ses': session,
                    'modb': modb,
                    'mod': mod,
                    'seq': seq,
                    'rep': run,
                    'prop_dt': prop_dt,
                    'prop_base': prop_base,
                    'n_files': _clean(row.get('n_files')),
                    'acq_time': _clean(row.get('acq_time')),
                    'visual_only': self._is_visual_only_sequence(seq, mod),
                })
                self._apply_visual_only_rules(r)
            self.log_text.append("Loaded TSV into mapping table.")

            # Apply always-exclude patterns before building lookup tables
            self.applyExcludePatterns()

            # Build modality/sequence lookup for tree interactions
            self._rebuild_lookup_maps()
            self._maybe_notify_repeats()

            self.populateModalitiesTree()
            self.populateSpecificTree()
            if getattr(self, 'last_rep_box', None) is not None and self.last_rep_box.isChecked():
                self._onLastRepToggled(True)

            if self._apply_sequence_on_load:
                # Auto-apply the suffix dictionary after a fresh scan so the
                # latest custom patterns are reflected in the newly loaded
                # mapping table without requiring another manual save.
                self.applySequenceDictionary()
                self._apply_sequence_on_load = False

            # Populate naming table
            self.naming_table.blockSignals(True)
            self.naming_table.setRowCount(0)
            name_df = df[["StudyDescription", "GivenName", "BIDS_name"]].copy()
            name_df = name_df.drop_duplicates(subset=["StudyDescription", "BIDS_name"])
            for _, row in name_df.iterrows():
                nr = self.naming_table.rowCount()
                self.naming_table.insertRow(nr)
                sitem = QTableWidgetItem(_clean(row["StudyDescription"]))
                sitem.setFlags(sitem.flags() & ~Qt.ItemIsEditable)
                self.naming_table.setItem(nr, 0, sitem)
                gitem = QTableWidgetItem(_clean(row["GivenName"]))
                gitem.setFlags(gitem.flags() & ~Qt.ItemIsEditable)
                self.naming_table.setItem(nr, 1, gitem)
                bitem = QTableWidgetItem(_clean(row["BIDS_name"]))
                bitem.setFlags(bitem.flags() | Qt.ItemIsEditable)
                self.naming_table.setItem(nr, 2, bitem)
            self.naming_table.blockSignals(False)
            self._updateMappingControlsEnabled()
            QTimer.singleShot(0, self._auto_apply_existing_study_mappings)
        finally:
            self._loading_mapping_table = False


    def _build_series_list_from_df(self, df):
        rows = []

        # ``heudiconv`` initially names outputs using a simplified stem derived
        # from the DICOM SeriesDescription.  To later locate those files for
        # renaming we reconstruct that stem here.  We mirror the logic used by
        # :mod:`build_heuristic_from_tsv` which appends ``rep-<N>`` when a
        # sequence appears multiple times for a given subject/session.
        rep_counts = (
            df.groupby(["BIDS_name", "session", "sequence"], dropna=False)["sequence"].transform("count")
        )
        rep_index = (
            df.groupby(["BIDS_name", "session", "sequence"], dropna=False).cumcount() + 1
        )

        for idx, row in df.iterrows():
            subject = _extract_subject(row)
            session = row.get("session") or row.get("ses") or None
            modality = str(row.get("modality") or row.get("fine_modality") or row.get("BIDS_modality") or "")
            sequence = str(row.get("sequence") or row.get("SeriesDescription") or "")
            # ``rep`` encodes repeat acquisitions detected earlier.  Leave it as
            # ``None`` for non-repeated series and cast to ``int`` when present.
            rep_val = row.get("rep") or row.get("repeat")
            rep = int(rep_val) if rep_val else None

            extra: dict[str, str] = {}
            for key in ("task", "task_hits", "acq", "run", "dir", "echo"):
                if row.get(key):
                    extra[key] = str(row.get(key))

            # Reconstruct the basename produced by the converter so
            # :func:`apply_post_conversion_rename` can locate existing files even
            # when their names no longer contain the raw sequence.  This uses the
            # subject ID, optional session, a "safe" version of the sequence and
            # ``rep-<N>`` when duplicates exist.
            if row.get("BIDS_name") and sequence:
                base_parts = [str(row["BIDS_name"])]
                if session:
                    base_parts.append(session)
                base_parts.append(_safe_stem(sequence))
                if rep_counts.iloc[idx] > 1:
                    base_parts.append(f"rep-{rep_index.iloc[idx]}")
                current_base = _dedup_parts(*base_parts)
                extra["current_bids"] = current_base

            rows.append(SeriesInfo(subject, session, modality, sequence, rep, extra))

        return rows

    def _post_conversion_schema_rename(self, bids_root: str, df):
        if not (ENABLE_SCHEMA_RENAMER and self._schema):
            return {}
        series_list = self._build_series_list_from_df(df)
        proposals = build_preview_names(series_list, self._schema)
        rename_map = apply_post_conversion_rename(
            bids_root=bids_root,
            proposals=proposals,
            also_normalize_fieldmaps=ENABLE_FIELDMap_NORMALIZATION,
            handle_dwi_derivatives=ENABLE_DWI_DERIVATIVES_MOVE,
            derivatives_pipeline_name=DERIVATIVES_PIPELINE_NAME,
        )
        return rename_map


    def populateModalitiesTree(self):
        """Build modalities tree with checkboxes synced to the table."""
        self.full_tree.blockSignals(True)
        self.full_tree.clear()
        # build nested mapping: BIDS modality → non‑BIDS modality → seq → info
        modb_map = {}
        for info in self.row_info:
            modb_map.setdefault(info['modb'], {})\
                    .setdefault(info['mod'], {})[(info['seq'], info['rep'])] = info

        for modb, mod_map in sorted(modb_map.items()):
            modb_item = QTreeWidgetItem([modb])
            modb_item.setFlags(modb_item.flags() | Qt.ItemIsUserCheckable)
            rows = self.modb_rows.get(modb, [])
            states = [self.mapping_table.item(r, 0).checkState() == Qt.Checked for r in rows]
            if states and all(states):
                modb_item.setCheckState(0, Qt.Checked)
            elif states and any(states):
                modb_item.setCheckState(0, Qt.PartiallyChecked)
            else:
                modb_item.setCheckState(0, Qt.Unchecked)
            modb_item.setData(0, Qt.UserRole, ('modb', modb))

            for mod, seqs in sorted(mod_map.items()):
                mod_item = QTreeWidgetItem([mod])
                mod_item.setFlags(mod_item.flags() | Qt.ItemIsUserCheckable)
                rows = self.mod_rows.get((modb, mod), [])
                states = [self.mapping_table.item(r, 0).checkState() == Qt.Checked for r in rows]
                if states and all(states):
                    mod_item.setCheckState(0, Qt.Checked)
                elif states and any(states):
                    mod_item.setCheckState(0, Qt.PartiallyChecked)
                else:
                    mod_item.setCheckState(0, Qt.Unchecked)
                mod_item.setData(0, Qt.UserRole, ('mod', modb, mod))
                for (seq, rep), info in sorted(seqs.items()):
                    visual_only = bool(info.get("visual_only"))
                    label = seq
                    if rep:
                        label = f"{seq} (rep {rep})"
                    seq_item = QTreeWidgetItem([label])
                    if visual_only:
                        seq_item.setFlags(
                            (seq_item.flags() & ~Qt.ItemIsUserCheckable) & ~Qt.ItemIsEnabled
                        )
                    else:
                        seq_item.setFlags(seq_item.flags() | Qt.ItemIsUserCheckable)
                    rows = self.seq_rows.get((modb, mod, seq, rep), [])
                    states = [self.mapping_table.item(r, 0).checkState() == Qt.Checked for r in rows]
                    if states and all(states):
                        seq_item.setCheckState(0, Qt.Checked)
                    elif states and any(states):
                        seq_item.setCheckState(0, Qt.PartiallyChecked)
                    else:
                        seq_item.setCheckState(0, Qt.Unchecked)
                    seq_item.setData(0, Qt.UserRole, ('seq', modb, mod, seq, rep))
                    mod_item.addChild(seq_item)
                modb_item.addChild(mod_item)
            self.full_tree.addTopLevelItem(modb_item)

        self.full_tree.expandAll()
        self.full_tree.blockSignals(False)
        try:
            self.full_tree.itemChanged.disconnect(self.onModalityItemChanged)
        except TypeError:
            pass
        self.full_tree.itemChanged.connect(self.onModalityItemChanged)
        for i in range(self.full_tree.columnCount()):
            self.full_tree.resizeColumnToContents(i)

    def onSpecificItemChanged(self, item, column):
        role = item.data(0, Qt.UserRole)
        if not role:
            return
        state = item.checkState(0)
        tp = role[0]
        if tp == 'study':
            rows = self.study_rows.get(role[1], [])
        elif tp == 'subject':
            if self.use_bids_names:
                rows = self.subject_rows.get((role[1], role[2]), [])
            else:
                rows = self.subject_rows_given.get((role[1], role[2]), [])
        elif tp == 'session':
            if self.use_bids_names:
                rows = self.session_rows.get((role[1], role[2], role[3]), [])
            else:
                rows = self.session_rows_given.get((role[1], role[2], role[3]), [])
        elif tp == 'modb':
            if self.use_bids_names:
                rows = self.spec_modb_rows.get((role[1], role[2], role[3], role[4]), [])
            else:
                rows = self.spec_modb_rows_given.get((role[1], role[2], role[3], role[4]), [])
        elif tp == 'mod':
            if self.use_bids_names:
                rows = self.spec_mod_rows.get((role[1], role[2], role[3], role[4], role[5]), [])
            else:
                rows = self.spec_mod_rows_given.get((role[1], role[2], role[3], role[4], role[5]), [])
        elif tp == 'seq':
            if self.use_bids_names:
                rows = self.spec_seq_rows.get((role[1], role[2], role[3], role[4], role[5], role[6], role[7]), [])
            else:
                rows = self.spec_seq_rows_given.get((role[1], role[2], role[3], role[4], role[5], role[6], role[7]), [])
        else:
            rows = []
        for r in rows:
            if self.row_info[r].get("visual_only"):
                continue
            self.mapping_table.item(r, 0).setCheckState(state)
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.populateModalitiesTree)
        QTimer.singleShot(0, self.populateSpecificTree)

    def _onNamingEdited(self, item):
        if item.column() != 2:
            return
        study = self.naming_table.item(item.row(), 0).text()
        given = self.naming_table.item(item.row(), 1).text()
        # Remember the existing BIDS name so we can restore it if validation fails
        old_bids = None
        for info in self.row_info:
            if info['study'] == study and info['given'] == given:
                old_bids = info['bids']
                break
        new_bids = item.text()
        # Ensure the prefix is kept and that names remain unique
        if not new_bids.startswith('sub-'):
            QMessageBox.warning(self, "Invalid name", "BIDS names must start with 'sub-'.")
            if old_bids is not None:
                item.setText(old_bids)
            return
        other_names = [
            self.naming_table.item(r, 2).text()
            for r in range(self.naming_table.rowCount())
            if r != item.row() and self.naming_table.item(r, 0).text() == study
        ]
        # Also consider names already present in the converted dataset
        used_names = set(other_names)
        used_names.update(self.existing_used.get(study, set()))
        if new_bids in used_names and self.existing_maps.get(study, {}).get(given) != new_bids:
            QMessageBox.warning(
                self,
                "Duplicate name",
                "This name is already assigned in this study.",
            )
            if old_bids is not None:
                item.setText(old_bids)
            return
        for idx, info in enumerate(self.row_info):
            if info['study'] == study and info['given'] == given:
                info['bids'] = new_bids
                self.mapping_table.item(idx, 5).setText(new_bids)
        # Keep internal mapping updated
        self.existing_maps.setdefault(study, {})[given] = new_bids
        self.existing_used.setdefault(study, set()).add(new_bids)
        self._rebuild_lookup_maps()
        QTimer.singleShot(0, self.populateModalitiesTree)
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.generatePreview)

    def _onNameChoiceChanged(self, _index=None):
        self.use_bids_names = self.name_choice.currentIndex() == 0
        QTimer.singleShot(0, self.generatePreview)
        QTimer.singleShot(0, self.populateSpecificTree)

    def _onLastRepToggled(self, checked=False):
        groups = defaultdict(list)
        for idx, info in enumerate(self.row_info):
            key = (
                info['study'], info['bids'], info['ses'],
                info['modb'], info['mod'], info['seq']
            )
            rep_num = int(info['rep']) if str(info['rep']).isdigit() else 1
            groups.setdefault(key, []).append((rep_num, idx))
        for items in groups.values():
            filtered = [entry for entry in items if not self.row_info[entry[1]].get("visual_only")]
            if len(filtered) < 2:
                continue
            if checked:
                max_idx = max(filtered, key=lambda x: x[0])[1]
                for _, i in filtered:
                    st = Qt.Checked if i == max_idx else Qt.Unchecked
                    self.mapping_table.item(i, 0).setCheckState(st)
            else:
                for _, i in filtered:
                    self.mapping_table.item(i, 0).setCheckState(Qt.Checked)
        QTimer.singleShot(0, self.populateSpecificTree)
        QTimer.singleShot(0, self.populateModalitiesTree)

    # ----- always exclude helpers -----
    def _exclude_add(self) -> None:
        pattern = self.exclude_edit.text().strip()
        if not pattern:
            return
        r = self.exclude_table.rowCount()
        self.exclude_table.insertRow(r)
        chk = QTableWidgetItem()
        chk.setFlags(chk.flags() | Qt.ItemIsUserCheckable)
        chk.setCheckState(Qt.Checked)
        self.exclude_table.setItem(r, 0, chk)
        self.exclude_table.setItem(r, 1, QTableWidgetItem(pattern))
        self.exclude_edit.clear()

    def loadExcludePatterns(self) -> None:
        if not hasattr(self, "exclude_table"):
            return
        self.exclude_table.setRowCount(0)
        patterns = []
        if self.exclude_patterns_file.exists():
            try:
                df = pd.read_csv(self.exclude_patterns_file, sep="\t", keep_default_na=False)
                for _, row in df.iterrows():
                    pat = str(row.get("pattern", ""))
                    active = bool(int(row.get("active", 1)))
                    patterns.append((pat, active))
            except Exception:
                pass
        if not patterns:
            patterns = [
                ("localizer", True),
                ("scout", True),
                ("phoenixzipreport", True),
                ("phoenix document", True),
                (".pdf", True),
                ("report", True),
                ("physlog", True),
            ]
        for pat, active in patterns:
            r = self.exclude_table.rowCount()
            self.exclude_table.insertRow(r)
            chk = QTableWidgetItem()
            chk.setFlags(chk.flags() | Qt.ItemIsUserCheckable)
            chk.setCheckState(Qt.Checked if active else Qt.Unchecked)
            self.exclude_table.setItem(r, 0, chk)
            self.exclude_table.setItem(r, 1, QTableWidgetItem(pat))
        self.applyExcludePatterns()

    def saveExcludePatterns(self) -> None:
        self.exclude_patterns_file.parent.mkdir(exist_ok=True, parents=True)
        rows = []
        for r in range(self.exclude_table.rowCount()):
            pat = self.exclude_table.item(r, 1).text().strip()
            if not pat:
                continue
            active = self.exclude_table.item(r, 0).checkState() == Qt.Checked
            rows.append({"active": int(active), "pattern": pat})
        pd.DataFrame(rows).to_csv(self.exclude_patterns_file, sep="\t", index=False)
        QMessageBox.information(self, "Saved", f"Updated {self.exclude_patterns_file}")
        self.applyExcludePatterns()

    def applyExcludePatterns(self) -> None:
        if not hasattr(self, "exclude_table"):
            return
        patterns = [
            self.exclude_table.item(r, 1).text().strip().lower()
            for r in range(self.exclude_table.rowCount())
            if self.exclude_table.item(r, 0).checkState() == Qt.Checked
        ]
        for r in range(self.mapping_table.rowCount()):
            seq = self.mapping_table.item(r, 9).text().lower()
            if any(p in seq for p in patterns):
                self.mapping_table.item(r, 0).setCheckState(Qt.Unchecked)

    # ----- suffix dictionary helpers -----
    def _custom_add(self, suffix: str) -> None:
        table = self.custom_tables.get(suffix)
        edit = self.custom_inputs.get(suffix)
        if table is None or edit is None:
            return
        pat = edit.text().strip()
        if not pat:
            return
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(pat))
        edit.clear()

    def _custom_remove(self, suffix: str) -> None:
        table = self.custom_tables.get(suffix)
        if table is None:
            return
        rows = sorted({item.row() for item in table.selectedItems()}, reverse=True)
        for r in rows:
            table.removeRow(r)

    def _collect_custom_patterns(self) -> Dict[str, Tuple[str, ...]]:
        patterns: Dict[str, Tuple[str, ...]] = {}
        for suffix, table in self.custom_tables.items():
            entries: list[str] = []
            for r in range(table.rowCount()):
                item = table.item(r, 0)
                if item is None:
                    continue
                text = item.text().strip()
                if text:
                    entries.append(text)
            if entries:
                patterns[suffix] = tuple(entries)
        return patterns

    def _set_custom_pattern_enabled(self, enabled: bool) -> None:
        for table in self.custom_tables.values():
            table.setEnabled(enabled)
        for edit in self.custom_inputs.values():
            edit.setEnabled(enabled)
        for button in self.custom_add_buttons.values():
            button.setEnabled(enabled)
        for button in self.custom_remove_buttons.values():
            button.setEnabled(enabled)
        if hasattr(self, "seq_save_button"):
            self.seq_save_button.setEnabled(enabled)

    def _refresh_suffix_dictionary(self) -> None:
        active = dicom_inventory.get_sequence_hint_patterns()
        custom = dicom_inventory.get_custom_sequence_dictionary()
        use_custom = dicom_inventory.is_sequence_dictionary_enabled()
        for suffix, label in self.active_labels.items():
            active_list = active.get(suffix, ())
            source = "Custom" if use_custom and custom.get(suffix) else "Default"
            label.setText(f"Active source: {source} ({len(active_list)} patterns)")

    def _on_custom_toggle(self, enabled: bool) -> None:
        dicom_inventory.set_sequence_dictionary_enabled(enabled)
        self._set_custom_pattern_enabled(enabled)
        self._refresh_suffix_dictionary()
        self.applySequenceDictionary()

    def loadSequenceDictionary(self) -> None:
        if not hasattr(self, "seq_tabs_widget"):
            return

        default_patterns = dicom_inventory.get_sequence_hint_patterns(source="default")
        custom_patterns = dicom_inventory.get_custom_sequence_dictionary()
        suffixes = sorted(set(default_patterns) | set(custom_patterns))
        use_custom = dicom_inventory.is_sequence_dictionary_enabled()

        self.seq_tabs_widget.clear()
        self.default_pattern_lists = {}
        self.custom_tables = {}
        self.custom_inputs = {}
        self.custom_add_buttons = {}
        self.custom_remove_buttons = {}
        self.active_labels = {}

        for suffix in suffixes:
            tab = QWidget()
            layout = QVBoxLayout(tab)

            active_label = QLabel()
            self.active_labels[suffix] = active_label
            layout.addWidget(active_label)

            default_group = QGroupBox("Default patterns")
            default_layout = QVBoxLayout(default_group)
            default_list = QListWidget()
            default_list.addItems(default_patterns.get(suffix, ()))
            default_list.setEnabled(False)
            self.default_pattern_lists[suffix] = default_list
            default_layout.addWidget(default_list)
            layout.addWidget(default_group)

            custom_group = QGroupBox("Custom patterns")
            custom_layout = QVBoxLayout(custom_group)
            table = QTableWidget()
            table.setColumnCount(1)
            table.setHorizontalHeaderLabels(["Pattern"])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            for pat in custom_patterns.get(suffix, ()):  # populate stored entries
                row = table.rowCount()
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(pat))
            self.custom_tables[suffix] = table
            custom_layout.addWidget(table)

            controls = QHBoxLayout()
            edit = QLineEdit()
            self.custom_inputs[suffix] = edit
            controls.addWidget(edit)
            add_btn = QPushButton("Add")
            add_btn.clicked.connect(lambda _=False, s=suffix: self._custom_add(s))
            self.custom_add_buttons[suffix] = add_btn
            controls.addWidget(add_btn)
            rm_btn = QPushButton("Remove")
            rm_btn.clicked.connect(lambda _=False, s=suffix: self._custom_remove(s))
            self.custom_remove_buttons[suffix] = rm_btn
            controls.addWidget(rm_btn)
            custom_layout.addLayout(controls)
            layout.addWidget(custom_group)

            self.seq_tabs_widget.addTab(tab, suffix)

        self.use_custom_patterns_box.blockSignals(True)
        self.use_custom_patterns_box.setChecked(use_custom)
        self.use_custom_patterns_box.blockSignals(False)
        self._set_custom_pattern_enabled(use_custom)
        self._refresh_suffix_dictionary()
        self.applySequenceDictionary()

    def saveSequenceDictionary(self) -> None:
        patterns = self._collect_custom_patterns()
        data = [
            {"modality": suffix, "pattern": pat}
            for suffix, pats in patterns.items()
            for pat in pats
        ]
        if data:
            self.seq_dict_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(data).to_csv(self.seq_dict_file, sep="\t", index=False)
        else:
            try:
                self.seq_dict_file.unlink()
            except Exception:
                pass
        dicom_inventory.update_sequence_dictionary(patterns)
        dicom_inventory.set_sequence_dictionary_enabled(self.use_custom_patterns_box.isChecked())
        QMessageBox.information(
            self,
            "Saved",
            f"Updated suffix patterns in {self.seq_dict_file}",
        )
        self._refresh_suffix_dictionary()
        # Applying the sequence dictionary immediately updates the scanned data
        # table so users do not need to trigger "Save changes" separately.
        self.applySequenceDictionary()
        self.applyMappingChanges()

    def restoreSequenceDefaults(self) -> None:
        dicom_inventory.restore_sequence_dictionary()
        self.loadSequenceDictionary()
        QMessageBox.information(
            self,
            "Restored",
            "Default suffix dictionary restored",
        )

    def applySequenceDictionary(self) -> None:
        if not hasattr(self, "custom_tables"):
            return

        patterns = self._collect_custom_patterns()
        dicom_inventory.update_sequence_dictionary(patterns)
        dicom_inventory.set_sequence_dictionary_enabled(self.use_custom_patterns_box.isChecked())
        self._refresh_suffix_dictionary()

        if self.mapping_table.rowCount() > 0:
            for i in range(self.mapping_table.rowCount()):
                seq = self.mapping_table.item(i, 9).text()
                mod = dicom_inventory.guess_modality(seq)
                modb = dicom_inventory.modality_to_container(mod)
                self.mapping_table.item(i, 14).setText(mod)
                self.mapping_table.item(i, 15).setText(modb)
                if i < len(self.row_info):
                    self.row_info[i]['mod'] = mod
                    self.row_info[i]['modb'] = modb
            self._rebuild_lookup_maps()
            QTimer.singleShot(0, self.populateModalitiesTree)
            QTimer.singleShot(0, self.populateSpecificTree)

    def _rebuild_lookup_maps(self):
        """Recompute internal lookup tables for tree interactions."""
        # Maps from tree nodes to row indices in ``self.mapping_table``. These
        # allow checkbox changes in the tree to update table rows and vice versa.
        self.modb_rows.clear()
        self.mod_rows.clear()
        self.seq_rows.clear()
        self.study_rows.clear()
        self.subject_rows.clear()
        self.session_rows.clear()
        self.spec_modb_rows.clear()
        self.spec_mod_rows.clear()
        self.spec_seq_rows.clear()

        # Same lookups but using the "given" subject names when that mode is
        # selected instead of the BIDS names
        self.subject_rows_given = {}
        self.session_rows_given = {}
        self.spec_modb_rows_given = {}
        self.spec_mod_rows_given = {}
        self.spec_seq_rows_given = {}
        for idx, info in enumerate(self.row_info):
            if info.get("visual_only"):
                continue
            # Populate lookup tables using BIDS subject names
            self.modb_rows.setdefault(info['modb'], []).append(idx)
            self.mod_rows.setdefault((info['modb'], info['mod']), []).append(idx)
            self.seq_rows.setdefault((info['modb'], info['mod'], info['seq'], info['rep']), []).append(idx)
            self.study_rows.setdefault(info['study'], []).append(idx)
            self.subject_rows.setdefault((info['study'], info['bids']), []).append(idx)
            self.session_rows.setdefault((info['study'], info['bids'], info['ses']), []).append(idx)
            self.spec_modb_rows.setdefault((info['study'], info['bids'], info['ses'], info['modb']), []).append(idx)
            self.spec_mod_rows.setdefault((info['study'], info['bids'], info['ses'], info['modb'], info['mod']), []).append(idx)
            self.spec_seq_rows.setdefault((info['study'], info['bids'], info['ses'], info['modb'], info['mod'], info['seq'], info['rep']), []).append(idx)
            gsub = f"sub-{info['given']}"
            # Equivalent lookups built from the given (non-BIDS) subject names
            self.subject_rows_given.setdefault((info['study'], gsub), []).append(idx)
            self.session_rows_given.setdefault((info['study'], gsub, info['ses']), []).append(idx)
            self.spec_modb_rows_given.setdefault((info['study'], gsub, info['ses'], info['modb']), []).append(idx)
            self.spec_mod_rows_given.setdefault((info['study'], gsub, info['ses'], info['modb'], info['mod']), []).append(idx)
            self.spec_seq_rows_given.setdefault((info['study'], gsub, info['ses'], info['modb'], info['mod'], info['seq'], info['rep']), []).append(idx)

    def _save_tree_expansion(self, tree):
        states = {}

        def recurse(item):
            path = []
            it = item
            while it is not None:
                path.insert(0, it.text(0))
                it = it.parent()
            states[tuple(path)] = item.isExpanded()
            for i in range(item.childCount()):
                recurse(item.child(i))

        for i in range(tree.topLevelItemCount()):
            recurse(tree.topLevelItem(i))
        return states

    def _restore_tree_expansion(self, tree, states):
        def recurse(item):
            path = []
            it = item
            while it is not None:
                path.insert(0, it.text(0))
                it = it.parent()
            if states.get(tuple(path)):
                item.setExpanded(True)
            for i in range(item.childCount()):
                recurse(item.child(i))

        for i in range(tree.topLevelItemCount()):
            recurse(tree.topLevelItem(i))


    def populateSpecificTree(self):
        """Build detailed tree (study→subject→session→modality)."""
        expanded = self._save_tree_expansion(self.specific_tree)
        self.specific_tree.blockSignals(True)
        self.specific_tree.clear()

        tree_map = {}
        for info in self.row_info:
            if self.use_bids_names:
                subj_key = info['bids']
            else:
                subj_key = f"sub-{info['given']}"
            tree_map.setdefault(info['study'], {})\
                    .setdefault(subj_key, {})\
                    .setdefault(info['ses'], {})\
                    .setdefault(info['modb'], {})\
                    .setdefault(info['mod'], {})[(info['seq'], info['rep'])] = info

        def _state(rows):
            states = [self.mapping_table.item(r, 0).checkState() == Qt.Checked for r in rows]
            if states and all(states):
                return Qt.Checked
            if states and any(states):
                return Qt.PartiallyChecked
            return Qt.Unchecked

        multi_study = len(tree_map) > 1

        for study, sub_map in sorted(tree_map.items()):
            st_item = QTreeWidgetItem([study])
            if multi_study:
                # Only expose the study-level checkbox when the dataset includes
                # multiple studies so that users do not see a redundant single
                # checkbox for the only available study.
                st_item.setFlags(st_item.flags() | Qt.ItemIsUserCheckable)
                st_item.setCheckState(0, _state(self.study_rows.get(study, [])))
                st_item.setData(0, Qt.UserRole, ('study', study))
            else:
                # Remove the user-checkable flag to hide the checkbox while
                # keeping the study label visible for context.
                st_item.setFlags(st_item.flags() & ~Qt.ItemIsUserCheckable)
            for subj, ses_map in sorted(sub_map.items()):
                su_item = QTreeWidgetItem([subj])
                su_item.setFlags(su_item.flags() | Qt.ItemIsUserCheckable)
                if self.use_bids_names:
                    rows = self.subject_rows.get((study, subj), [])
                else:
                    rows = self.subject_rows_given.get((study, subj), [])
                su_item.setCheckState(0, _state(rows))
                su_item.setData(0, Qt.UserRole, ('subject', study, subj))
                for ses, modb_map in sorted(ses_map.items()):
                    se_item = QTreeWidgetItem([ses])
                    se_item.setFlags(se_item.flags() | Qt.ItemIsUserCheckable)
                    if self.use_bids_names:
                        rows = self.session_rows.get((study, subj, ses), [])
                    else:
                        rows = self.session_rows_given.get((study, subj, ses), [])
                    se_item.setCheckState(0, _state(rows))
                    se_item.setData(0, Qt.UserRole, ('session', study, subj, ses))
                    for modb, mod_map in sorted(modb_map.items()):
                        mb_item = QTreeWidgetItem([modb, "", ""])
                        mb_item.setFlags(mb_item.flags() | Qt.ItemIsUserCheckable)
                        if self.use_bids_names:
                            rows = self.spec_modb_rows.get((study, subj, ses, modb), [])
                        else:
                            rows = self.spec_modb_rows_given.get((study, subj, ses, modb), [])
                        mb_item.setCheckState(0, _state(rows))
                        mb_item.setData(0, Qt.UserRole, ('modb', study, subj, ses, modb))
                        for mod, seqs in sorted(mod_map.items()):
                            mo_item = QTreeWidgetItem([mod, "", ""])
                            mo_item.setFlags(mo_item.flags() | Qt.ItemIsUserCheckable)
                            if self.use_bids_names:
                                rows = self.spec_mod_rows.get((study, subj, ses, modb, mod), [])
                            else:
                                rows = self.spec_mod_rows_given.get((study, subj, ses, modb, mod), [])
                            mo_item.setCheckState(0, _state(rows))
                            mo_item.setData(0, Qt.UserRole, ('mod', study, subj, ses, modb, mod))
                            for (seq, rep), info in sorted(seqs.items()):
                                label = seq
                                if rep:
                                    label = f"{seq} (rep {rep})"
                                files = str(info['n_files'])
                                time = info['acq_time']
                                sq_item = QTreeWidgetItem([label, files, time])
                                if info.get("visual_only"):
                                    sq_item.setFlags(
                                        (sq_item.flags() & ~Qt.ItemIsUserCheckable) & ~Qt.ItemIsEnabled
                                    )
                                else:
                                    sq_item.setFlags(sq_item.flags() | Qt.ItemIsUserCheckable)
                                if self.use_bids_names:
                                    rows = self.spec_seq_rows.get((study, subj, ses, modb, mod, seq, rep), [])
                                else:
                                    rows = self.spec_seq_rows_given.get((study, subj, ses, modb, mod, seq, rep), [])
                                sq_item.setCheckState(0, _state(rows))
                                sq_item.setData(0, Qt.UserRole, ('seq', study, subj, ses, modb, mod, seq, rep))
                                mo_item.addChild(sq_item)
                            mb_item.addChild(mo_item)
                        se_item.addChild(mb_item)
                    su_item.addChild(se_item)
                st_item.addChild(su_item)
            self.specific_tree.addTopLevelItem(st_item)

        self._restore_tree_expansion(self.specific_tree, expanded)
        if not expanded:
            self.specific_tree.expandAll()
        self.specific_tree.blockSignals(False)
        try:
            self.specific_tree.itemChanged.disconnect(self.onSpecificItemChanged)
        except TypeError:
            pass
        self.specific_tree.itemChanged.connect(self.onSpecificItemChanged)
        for i in range(self.specific_tree.columnCount()):
            self.specific_tree.resizeColumnToContents(i)


    def onModalityItemChanged(self, item, column):
        role = item.data(0, Qt.UserRole)
        if not role:
            return
        state = item.checkState(0)
        if role[0] == 'modb':
            modb = role[1]
            for r in self.modb_rows.get(modb, []):
                if self.row_info[r].get("visual_only"):
                    continue
                self.mapping_table.item(r, 0).setCheckState(state)
        elif role[0] == 'mod':
            modb, mod = role[1], role[2]
            for r in self.mod_rows.get((modb, mod), []):
                if self.row_info[r].get("visual_only"):
                    continue
                self.mapping_table.item(r, 0).setCheckState(state)
        elif role[0] == 'seq':
            modb, mod, seq, rep = role[1], role[2], role[3], role[4]
            for r in self.seq_rows.get((modb, mod, seq, rep), []):
                if self.row_info[r].get("visual_only"):
                    continue
                self.mapping_table.item(r, 0).setCheckState(state)
        QTimer.singleShot(0, self.populateModalitiesTree)

    def runFullConversion(self):
        logging.info("runFullConversion → Starting full pipeline …")
        if self.conv_process and self.conv_process.state() != QProcess.NotRunning:
            return
        if not self.tsv_path or not os.path.isfile(self.tsv_path):
            QMessageBox.warning(self, "No TSV", "Please generate the TSV first.")
            return
        if not self.bids_out_dir:
            QMessageBox.warning(self, "No BIDS Output", "Please select a BIDS output directory.")
            return

        # 1) Save updated TSV from table
        try:
            df_orig = pd.read_csv(self.tsv_path, sep="\t", keep_default_na=False)
            df_conv = df_orig.copy()
            for i in range(self.mapping_table.rowCount()):
                include = 1 if self.mapping_table.item(i, 0).checkState() == Qt.Checked else 0
                info = self.row_info[i]
                seq = self.mapping_table.item(i, 9).text()
                modb = self.mapping_table.item(i, 15).text()

                # Update df_orig with canonical BIDS name
                df_orig.at[i, 'BIDS_name'] = info['bids']
                df_orig.at[i, 'include'] = include
                df_orig.at[i, 'sequence'] = seq
                df_orig.at[i, 'modality_bids'] = modb

                # For conversion we may use given names
                conv_name = info['bids'] if self.use_bids_names else f"sub-{info['given']}"
                df_conv.at[i, 'BIDS_name'] = conv_name

            df_orig.to_csv(self.tsv_path, sep="\t", index=False)
            self.log_text.append("Saved updated TSV.")

            # Write temporary TSV for heuristic generation if using given names
            if self.use_bids_names:
                self.tsv_for_conv = self.tsv_path
            else:
                tmp_tsv = os.path.join(self.bids_out_dir, "tmp_subjects.tsv")
                df_conv.to_csv(tmp_tsv, sep="\t", index=False)
                self.tsv_for_conv = tmp_tsv
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save TSV: {e}")
            return

        # Paths for scripts
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.build_script = os.path.join(script_dir, "build_heuristic_from_tsv.py")
        self.run_script = os.path.join(script_dir, "run_heudiconv_from_heuristic.py")
        self.rename_script = os.path.join(script_dir, "post_conv_renamer.py")

        self.heuristic_dir = os.path.join(self.bids_out_dir, "heuristics")
        self.heurs_to_rename = []
        self.conv_stage = 0

        self.log_text.append("Building heuristics…")
        self._start_spinner("Converting")
        self.run_button.setEnabled(False)
        self.run_stop_button.setEnabled(True)
        self.conv_process = QProcess(self)
        if self.terminal_cb.isChecked():
            # Forward output so the user can monitor conversion progress
            self.conv_process.setProcessChannelMode(QProcess.ForwardedChannels)
        else:
            # Prevent blocked pipes on Windows when the terminal isn't shown
            self.conv_process.setStandardOutputFile(QProcess.nullDevice())
            self.conv_process.setStandardErrorFile(QProcess.nullDevice())
        self.conv_process.finished.connect(self._convStepFinished)
        args = [self.build_script, self.tsv_for_conv, self.heuristic_dir]
        self.conv_process.start(sys.executable, args)

    def _convStepFinished(self, exitCode, _status):
        if self.conv_stage == 0:
            if exitCode != 0:
                QMessageBox.critical(self, "Error", "build_heuristic failed")
                self.stopConversion()
                return
            self.log_text.append(f"Heuristics written to {self.heuristic_dir}")
            self.conv_stage = 1
            self.log_text.append("Running HeuDiConv…")
            args = [self.run_script, self.dicom_dir, self.heuristic_dir, self.bids_out_dir, '--subject-tsv', self.tsv_path]
            self.conv_process.start(sys.executable, args)
        elif self.conv_stage == 1:
            if exitCode != 0:
                QMessageBox.critical(self, "Error", "run_heudiconv failed")
                self.stopConversion()
                return
            self.log_text.append("HeuDiConv conversion complete.")
            self.conv_stage = 2
            self.heurs_to_rename = list(Path(self.heuristic_dir).glob("heuristic_*.py"))
            self._runNextRename()
        elif self.conv_stage == 2:
            if exitCode != 0:
                QMessageBox.critical(self, "Error", "post_conv_renamer failed")
                self.stopConversion()
                return
            if self.heurs_to_rename:
                self._runNextRename()
            else:
                self.log_text.append("Conversion pipeline finished successfully.")
                self._store_heuristics()
                if self.inventory_df is not None:
                    rename_map = self._post_conversion_schema_rename(self.bids_out_dir, self.inventory_df)
                    self.log_text.append(f"Schema renamer moved/renamed {len(rename_map)} files.")
                if getattr(self, 'tsv_for_conv', self.tsv_path) != self.tsv_path:
                    try:
                        os.remove(self.tsv_for_conv)
                    except Exception:
                        pass
                self.stopConversion(success=True)

    def _runNextRename(self):
        if not self.heurs_to_rename:
            self._convStepFinished(0, 0)
            return
        heur = self.heurs_to_rename.pop(0)
        dataset = heur.stem.replace("heuristic_", "")
        bids_path = os.path.join(self.bids_out_dir, dataset)
        self.log_text.append(f"Renaming fieldmaps for {dataset}…")
        args = [self.rename_script, bids_path]
        self.conv_process.start(sys.executable, args)

    def _store_heuristics(self):
        """Move heuristics into each dataset's .bids_manager folder."""
        try:
            hdir = Path(self.heuristic_dir)
            for heur in hdir.glob("heuristic_*.py"):
                dataset = heur.stem.replace("heuristic_", "")
                dst = Path(self.bids_out_dir) / dataset / ".bids_manager"
                dst.mkdir(exist_ok=True)
                shutil.move(str(heur), dst / heur.name)
            shutil.rmtree(hdir, ignore_errors=True)
        except Exception as exc:
            logging.warning(f"Failed to move heuristics: {exc}")

    def stopConversion(self, success: bool = False):
        if self.conv_process and self.conv_process.state() != QProcess.NotRunning:
            pid = int(self.conv_process.processId())
            _terminate_process_tree(pid)
        self.conv_process = None
        self._stop_spinner()
        self.run_button.setEnabled(True)
        self.run_stop_button.setEnabled(False)
        if not success:
            self.log_text.append("Conversion cancelled.")

    # ----- Edit tab methods (full bids_editor_ancpbids features) -----
    def openBIDSForEdit(self):
        """Prompt user to select a BIDS dataset root for editing."""
        p = QFileDialog.getExistingDirectory(self, "Select BIDS dataset")
        if p:
            self.bids_root = Path(p)
            self.model.setRootPath(p)
            self.tree.setRootIndex(self.model.index(p))
            self.viewer.clear()
            self.updateStats()
            self.loadExcludePatterns()

    def onTreeClicked(self, idx: QModelIndex):
        """When a file is clicked, load metadata if supported."""

        p = Path(self.model.filePath(idx))
        self.selected = p
        ext = _get_ext(p)
        lower_name = p.name.lower()
        gifti_candidate = any(lower_name.endswith(suffix) for suffix in GIFTI_SURFACE_SUFFIXES)
        freesurfer_candidate = any(
            lower_name.endswith(suffix) for suffix in FREESURFER_SURFACE_SUFFIXES
        )
        # ``is_dicom_file`` also checks for files without an extension.
        dicom_like = is_dicom_file(str(p))
        if (
            ext in ['.json', '.tsv', '.nii', '.nii.gz', '.html', '.htm']
            or dicom_like
            or gifti_candidate
            or freesurfer_candidate
        ):
            self.viewer.load_file(p)

    def updateStats(self):
        """Compute and display BIDS stats: total subjects, files, modalities."""
        root = self.bids_root
        if not root:
            return
        self.stats.clear()
        subs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith('sub-')]
        self.stats.addTopLevelItem(QTreeWidgetItem(["Total subjects", str(len(subs))]))
        files = list(root.rglob('*.*'))
        self.stats.addTopLevelItem(QTreeWidgetItem(["Total files", str(len(files))]))
        for sub in subs:
            si = QTreeWidgetItem([sub.name, ""])
            sessions = [d for d in sub.iterdir() if d.is_dir() and d.name.startswith('ses-')]
            if len(sessions) > 1:
                for ses in sessions:
                    s2 = QTreeWidgetItem([ses.name, ""])
                    mods = set(p.parent.name for p in ses.rglob('*.nii*'))
                    s2.addChild(QTreeWidgetItem(["Modalities", str(len(mods))]))
                    for m in mods:
                        imgs = len(list(ses.rglob(f'{m}/*.nii*')))
                        meta = len(list(ses.rglob(f'{m}/*.json'))) + len(list(ses.rglob(f'{m}/*.tsv')))
                        s2.addChild(QTreeWidgetItem([m, f"imgs:{imgs}, meta:{meta}"]))
                    si.addChild(s2)
            else:
                mods = set(p.parent.name for p in sub.rglob('*.nii*'))
                si.addChild(QTreeWidgetItem(["Sessions", "1"]))
                si.addChild(QTreeWidgetItem(["Modalities", str(len(mods))]))
                for m in mods:
                    imgs = len(list(sub.rglob(f'{m}/*.nii*')))
                    meta = len(list(sub.rglob(f'{m}/*.json'))) + len(list(sub.rglob(f'{m}/*.tsv')))
                    si.addChild(QTreeWidgetItem([m, f"imgs:{imgs}, meta:{meta}"]))
            self.stats.addTopLevelItem(si)
        self.stats.expandAll()

    def launchBatchRename(self):
        """Open the Batch Rename dialog from bids_editor_ancpbids."""
        if not self.bids_root:
            QMessageBox.critical(
                self,
                "Error",
                "Dataset not detected. Please load a dataset in File → Open BIDS",
            )
            return
        dlg = RemapDialog(self, self.bids_root)
        dlg.exec_()

    def launchIntendedForEditor(self):
        """Open the manual IntendedFor editor dialog."""
        if not self.bids_root:
            QMessageBox.critical(
                self,
                "Error",
                "Dataset not detected. Please load a dataset in File → Open BIDS",
            )
            return
        dlg = IntendedForDialog(self, self.bids_root)
        dlg.exec_()

    def refreshScansTsv(self):
        """Update ``*_scans.tsv`` files to match current filenames."""
        if not self.bids_root:
            QMessageBox.critical(
                self,
                "Error",
                "Dataset not detected. Please load a dataset in File → Open BIDS",
            )
            return
        try:
            from .post_conv_renamer import update_scans_tsv

            update_scans_tsv(self.bids_root)
            QMessageBox.information(self, "Refresh", "Updated scans.tsv files")
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Failed to update: {exc}")

    def launchBidsIgnore(self):
        """Open dialog to edit ``.bidsignore``."""
        if not self.bids_root:
            QMessageBox.critical(
                self,
                "Error",
                "Dataset not detected. Please load a dataset in File → Open BIDS",
            )
            return
        dlg = BidsIgnoreDialog(self, self.bids_root)
        dlg.exec_()


class RemapDialog(QDialog):
    """
    Batch Rename Conditions dialog (from bids_editor_ancpbids).
    Allows regex-based renaming across the dataset.
    """
    def __init__(self, parent, default_scope: Path):
        super().__init__(parent)
        self.setWindowTitle("Batch Remap Conditions")
        self.resize(1000, 600)
        self.bids_root = default_scope
        layout = QVBoxLayout(self)

        # Tabs for multiple conditions
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.add_condition()  # initial condition

        # Scope selector
        scope_layout = QHBoxLayout()
        scope_layout.addWidget(QLabel("Scope:"))
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["Entire dataset", "Selected subjects"])
        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText("sub-001, sub-002")
        self.subject_edit.setEnabled(False)
        self.scope_combo.currentIndexChanged.connect(
            lambda i: self.subject_edit.setEnabled(i == 1)
        )
        scope_layout.addWidget(self.scope_combo)
        scope_layout.addWidget(self.subject_edit)
        layout.addLayout(scope_layout)

        # Preview and Apply buttons
        btn_layout = QHBoxLayout()
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.preview)
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply)
        btn_layout.addWidget(self.preview_button)
        btn_layout.addWidget(self.apply_button)
        layout.addLayout(btn_layout)

        # Preview tree
        self.preview_tree = QTreeWidget()
        self.preview_tree.setColumnCount(2)
        self.preview_tree.setHeaderLabels(["Original", "New Name"])
        hdr = self.preview_tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.preview_tree)

    def add_condition(self):
        """Add a new tab with a table for regex pattern/replacement pairs."""
        tab = QWidget()
        fl = QVBoxLayout(tab)
        rules_tbl = QTableWidget(0, 2)
        rules_tbl.setHorizontalHeaderLabels(["Pattern", "Replacement"])
        hdr = rules_tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        rule_btns = QHBoxLayout()
        btn_addr = QPushButton("Add Rule")
        btn_addr.clicked.connect(lambda: rules_tbl.insertRow(rules_tbl.rowCount()))
        btn_delr = QPushButton("Delete Rule")
        btn_delr.clicked.connect(lambda: rules_tbl.removeRow(rules_tbl.currentRow()))
        btn_addc = QPushButton("Add Condition")
        btn_addc.clicked.connect(self.add_condition)
        btn_delc = QPushButton("Delete Condition")
        btn_delc.clicked.connect(lambda: self.delete_condition(tab))
        rule_btns.addWidget(btn_addr)
        rule_btns.addWidget(btn_delr)
        rule_btns.addWidget(btn_addc)
        rule_btns.addWidget(btn_delc)
        rule_btns.addStretch()
        fl.addLayout(rule_btns)
        fl.addWidget(rules_tbl)
        index = self.tabs.count() + 1
        self.tabs.addTab(tab, f"Condition {index}")

    def delete_condition(self, tab):
        """Remove a condition tab."""
        idx = self.tabs.indexOf(tab)
        if idx != -1:
            self.tabs.removeTab(idx)
        if self.tabs.count() == 0:
            self.add_condition()

    def get_scope_paths(self):
        """Retrieve file paths under the selected scope."""
        all_files = [p for p in self.bids_root.rglob('*') if p.is_file()]
        if self.scope_combo.currentIndex() == 0:
            return all_files
        subs = [s.strip() for s in self.subject_edit.text().split(',') if s.strip()]
        if not subs:
            return all_files
        scoped = []
        for p in all_files:
            parts = p.relative_to(self.bids_root).parts
            if parts and parts[0] in subs:
                scoped.append(p)
        return scoped

    def preview(self):
        """Show potential renaming preview in a tree widget."""
        self.preview_tree.clear()
        paths = self.get_scope_paths()
        for path in sorted(paths):
            name = path.name
            new_name = name
            for i in range(self.tabs.count()):
                tbl = self.tabs.widget(i).findChild(QTableWidget)
                for r in range(tbl.rowCount()):
                    pat_item = tbl.item(r, 0)
                    rep_item = tbl.item(r, 1)
                    if pat_item and pat_item.text():
                        new_name = re.sub(pat_item.text(), rep_item.text() if rep_item else "", new_name)
            if new_name != name:
                item = QTreeWidgetItem([path.relative_to(self.bids_root).as_posix(), new_name])
                self.preview_tree.addTopLevelItem(item)
        self.preview_tree.expandAll()

    def apply(self):
        """Apply renaming to files as shown in preview."""
        rename_map = {}
        for i in range(self.preview_tree.topLevelItemCount()):
            it = self.preview_tree.topLevelItem(i)
            orig_rel = Path(it.text(0))
            orig = self.bids_root / orig_rel
            new = orig.with_name(it.text(1))
            orig.rename(new)
            rename_map[orig_rel.as_posix()] = (
                (orig_rel.parent / it.text(1)).as_posix()
            )
        if rename_map:
            try:
                from .scans_utils import update_scans_with_map

                update_scans_with_map(self.bids_root, rename_map)
            except Exception:
                pass
        QMessageBox.information(self, "Batch Remap", "Rename applied.")
        self.accept()


class CpuSettingsDialog(QDialog):
    """Dialog to select the number of CPUs for parallel tasks."""

    def __init__(self, parent, current: int = 1):
        super().__init__(parent)
        self.setWindowTitle("Parallel Settings")
        layout = QVBoxLayout(self)

        total_cpu = os.cpu_count() or 1
        # Show the total number of logical CPUs.  The main window initializes
        # ``current`` to roughly 80% of this value.
        ram_text = "n/a"
        if HAS_PSUTIL:
            try:
                mem_gb = psutil.virtual_memory().total / (1024 ** 3)
                ram_text = f"{mem_gb:.1f} GB"
            except Exception:  # pragma: no cover - info retrieval failure
                pass

        layout.addWidget(QLabel(f"Available CPUs: {total_cpu}\nRAM: {ram_text}"))

        row = QHBoxLayout()
        row.addWidget(QLabel("CPUs to use:"))
        self.spin = QSpinBox()
        self.spin.setRange(1, total_cpu)
        self.spin.setValue(min(current, total_cpu))
        row.addWidget(self.spin)
        layout.addLayout(row)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)


class DpiSettingsDialog(QDialog):
    """Dialog to adjust UI scale (DPI)."""

    def __init__(self, parent, current: int = 100):
        super().__init__(parent)
        self.setWindowTitle("UI Scale")
        layout = QVBoxLayout(self)

        # ``current`` is expressed as a percentage relative to a base scale of
        # 100 (96 DPI).  The main window passes the detected system value by
        # default so the initial UI matches the host environment.  Users can
        # still nudge the scale above or below that baseline as needed.

        row = QHBoxLayout()
        row.addWidget(QLabel("Scale (%):"))
        self.spin = QSpinBox()
        self.spin.setRange(50, 200)
        self.spin.setSingleStep(25)
        self.spin.setValue(current)
        row.addWidget(self.spin)
        layout.addLayout(row)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)


class AuthorshipDialog(QDialog):
    """Dialog displaying information about the authors and acknowledgements."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Authorship")
        self.resize(500, 700)
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Lab logo and description
        if ANCP_LAB_FILE.exists():
            logo = QLabel()
            pix = QPixmap(str(ANCP_LAB_FILE))
            logo.setPixmap(pix.scaledToWidth(320, Qt.SmoothTransformation))
            logo.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo)
            layout.addSpacing(10)
        desc = QLabel(
            "This software has been developed in the Applied Neurocognitive "
            "Psychology Lab with the objective of facilitating the conversion "
            "to BIDS format, easy metadata handling, and quality control."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)

        # Authors
        auth_label = QLabel("<b>Authors</b>")
        auth_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(auth_label)
        layout.addSpacing(10)

        k_row = QHBoxLayout()
        k_row.setSpacing(20)
        if KAREL_IMG_FILE.exists():
            k_pic = QLabel()
            k_pic.setPixmap(QPixmap(str(KAREL_IMG_FILE)).scaledToWidth(140, Qt.SmoothTransformation))
            k_pic.setAlignment(Qt.AlignCenter)
            k_row.addWidget(k_pic)
        k_desc = QLabel(
            "<b>Dr. Karel López Vilaret<br/>BIDS Manager App Lead</b><br/><br/>"
            "I hold a PhD in Neuroscience and currently work as a scientific "
            "software developer. I build BIDS Manager, a tool designed to "
            "streamline BIDS conversion, metadata handling, and quality "
            "control—enabling researchers to manage neuroimaging data more "
            "efficiently."
        )
        k_desc.setWordWrap(True)
        k_row.addWidget(k_desc)
        layout.addLayout(k_row)
        layout.addSpacing(15)

        j_row = QHBoxLayout()
        j_row.setSpacing(20)
        if JOCHEM_IMG_FILE.exists():
            j_pic = QLabel()
            j_pic.setPixmap(QPixmap(str(JOCHEM_IMG_FILE)).scaledToWidth(140, Qt.SmoothTransformation))
            j_pic.setAlignment(Qt.AlignCenter)
            j_row.addWidget(j_pic)
        j_desc = QLabel(
            "<b>Prof. Dr. rer. nat. Jochem Rieger<br/>Applied Neurocognitive Psychology</b><br/><br/>"
            "Full Professor of Psychology at the University of Oldenburg and "
            "head of the Applied Neurocognitive Psychology group. His research "
            "focuses on open science, machine learning, and understanding the "
            "neural basis of perception, cognition, and action in realistic "
            "environments."
        )
        j_desc.setWordWrap(True)
        j_row.addWidget(j_desc)
        layout.addLayout(j_row)
        layout.addSpacing(15)

        # Acknowledgements
        ack_label = QLabel("<b>Acknowledgements</b>")
        ack_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(ack_label)
        ack = QLabel(
            "Dr. Jorge Bosch-Bayard\n"
            "Msc. Erdal Karaca\n"
            "Bsc. Pablo Alexis Olguín Baxman\n"
            "Dr. Amirhussein Abdolalizadeh Saleh\n"
            "Dr. Tina Schmitt\n"
            "Dr.-Ing. Andreas Spiegler\n"
            "Msc. Shari Hiltner"
        )
        ack.setWordWrap(True)
        ack.setAlignment(Qt.AlignCenter)
        layout.addWidget(ack)
        layout.addSpacing(10)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.accepted.connect(self.accept)
        layout.addWidget(btn_box)


class IntendedForDialog(QDialog):
    """Manual editor for fieldmap IntendedFor lists."""

    def __init__(self, parent, bids_root: Path):
        super().__init__(parent)
        self.setWindowTitle("Set IntendedFor")
        self.resize(900, 500)
        self.bids_root = bids_root

        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        (
            self.bold_tab,
            self.bold_tree,
            self.bold_intended,
            self.bold_func_list,
            self.bold_remove,
            self.bold_add,
            self.bold_save,
        ) = self._build_tab("bold")
        self.tabs.addTab(self.bold_tab, "BOLD")

        (
            self.dwi_tab,
            self.dwi_tree,
            self.dwi_intended,
            self.dwi_func_list,
            self.dwi_remove,
            self.dwi_add,
            self.dwi_save,
        ) = self._build_tab("dwi")

        self.b0_box = QCheckBox("Treat DWI b0 maps as fieldmaps")
        self.b0_box.toggled.connect(self._on_b0_toggle)
        layout.addWidget(self.b0_box)

        self.data = {}
        self._init_b0_state()
        self._collect()

    # ---- helpers ----
    def _build_tab(self, mode: str):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Fieldmap images:"))
        tree = QTreeWidget()
        tree.setHeaderHidden(True)
        if mode == "bold":
            tree.itemSelectionChanged.connect(lambda: self._on_left_selected("bold"))
        else:
            tree.itemSelectionChanged.connect(lambda: self._on_left_selected("dwi"))
        left_layout.addWidget(tree)
        layout.addLayout(left_layout, 2)

        mid_layout = QVBoxLayout()
        mid_layout.addWidget(QLabel("IntendedFor:"))
        intended = QListWidget()
        intended.setSelectionMode(QAbstractItemView.ExtendedSelection)
        mid_layout.addWidget(intended)
        rm_save = QHBoxLayout()
        remove = QPushButton("Remove")
        remove.clicked.connect(lambda: self._remove_selected(mode))
        save = QPushButton("Save")
        save.clicked.connect(lambda: self._save_changes(mode))
        rm_save.addWidget(remove)
        rm_save.addWidget(save)
        rm_save.addStretch()
        mid_layout.addLayout(rm_save)
        layout.addLayout(mid_layout, 2)

        right_layout = QVBoxLayout()
        label = "Functional images:" if mode == "bold" else "Diffusion images:"
        right_layout.addWidget(QLabel(label))
        func_list = QListWidget()
        func_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        right_layout.addWidget(func_list)
        add_btn = QPushButton("← Add")
        add_btn.clicked.connect(lambda: self._add_selected(mode))
        right_layout.addWidget(add_btn)
        right_layout.addStretch()
        layout.addLayout(right_layout, 2)

        return widget, tree, intended, func_list, remove, add_btn, save

    def _init_b0_state(self) -> None:
        has_b0 = self._has_any_b0()
        fmap_b0 = self._fmap_has_b0()
        self.b0_box.setEnabled(has_b0)
        self.b0_box.setChecked(fmap_b0)
        if fmap_b0 and self.tabs.indexOf(self.dwi_tab) == -1:
            self.tabs.addTab(self.dwi_tab, "DWI")

    def _on_b0_toggle(self) -> None:
        if self.b0_box.isChecked():
            if self.tabs.indexOf(self.dwi_tab) == -1:
                self.tabs.addTab(self.dwi_tab, "DWI")
        else:
            idx = self.tabs.indexOf(self.dwi_tab)
            if idx != -1:
                self.tabs.removeTab(idx)
        self._collect()

    def _collect(self) -> None:
        self.bold_tree.clear()
        if self.tabs.indexOf(self.dwi_tab) != -1:
            self.dwi_tree.clear()
        self.data.clear()
        if self.b0_box.isChecked():
            self._move_b0_maps()
        else:
            self._restore_b0_maps()
        for sub in sorted(self.bids_root.glob('sub-*')):
            if not sub.is_dir():
                continue
            sub_item_bold = QTreeWidgetItem([sub.name])
            self.bold_tree.addTopLevelItem(sub_item_bold)
            sub_item_dwi = None
            if self.tabs.indexOf(self.dwi_tab) != -1:
                sub_item_dwi = QTreeWidgetItem([sub.name])
                self.dwi_tree.addTopLevelItem(sub_item_dwi)
            sessions = [s for s in sub.glob('ses-*') if s.is_dir()]
            if sessions:
                for ses in sessions:
                    ses_item_bold = QTreeWidgetItem([ses.name])
                    sub_item_bold.addChild(ses_item_bold)
                    ses_item_dwi = None
                    if sub_item_dwi is not None:
                        ses_item_dwi = QTreeWidgetItem([ses.name])
                        sub_item_dwi.addChild(ses_item_dwi)
                    self._add_fmaps(ses, ses_item_bold, ses_item_dwi, sub.name, ses.name)
            else:
                self._add_fmaps(sub, sub_item_bold, sub_item_dwi, sub.name, None)
            sub_item_bold.setExpanded(True)
            if sub_item_dwi is not None:
                sub_item_dwi.setExpanded(True)

    def _move_b0_maps(self) -> None:
        """Move DWI b0/epi images into the ``fmap`` folder."""
        rename_map: dict[str, str] = {}
        for sub in self.bids_root.glob('sub-*'):
            if not sub.is_dir():
                continue
            sessions = [s for s in sub.glob('ses-*') if s.is_dir()]
            roots = sessions or [sub]
            for root in roots:
                dwi_dir = root / 'dwi'
                if not dwi_dir.is_dir():
                    continue
                fmap_dir = root / 'fmap'
                fmap_dir.mkdir(exist_ok=True)
                for nii in dwi_dir.glob('*.nii*'):
                    name = nii.name.lower()
                    if 'b0' not in name and '_epi' not in name:
                        continue
                    dst = fmap_dir / nii.name
                    if not dst.exists():
                        nii.rename(dst)
                        rename_map[(nii.relative_to(self.bids_root)).as_posix()] = (
                            dst.relative_to(self.bids_root).as_posix()
                        )
                    base = re.sub(r'\.nii(\.gz)?$', '', nii.name, flags=re.I)
                    for ext in ['.json', '.bval', '.bvec']:
                        src = dwi_dir / (base + ext)
                        if src.exists():
                            dst_file = fmap_dir / src.name
                            if not dst_file.exists():
                                src.rename(dst_file)
                                rename_map[(src.relative_to(self.bids_root)).as_posix()] = (
                                    dst_file.relative_to(self.bids_root).as_posix()
                                )
        if rename_map:
            try:
                from .scans_utils import update_scans_with_map

                update_scans_with_map(self.bids_root, rename_map)
            except Exception:
                pass

    def _restore_b0_maps(self) -> None:
        """Move previously relocated b0/epi images back to ``dwi``."""
        rename_map: dict[str, str] = {}
        for sub in self.bids_root.glob('sub-*'):
            if not sub.is_dir():
                continue
            sessions = [s for s in sub.glob('ses-*') if s.is_dir()]
            roots = sessions or [sub]
            for root in roots:
                fmap_dir = root / 'fmap'
                dwi_dir = root / 'dwi'
                if not fmap_dir.is_dir() or not dwi_dir.is_dir():
                    continue
                for nii in fmap_dir.glob('*.nii*'):
                    name = nii.name.lower()
                    if 'b0' not in name and '_epi' not in name:
                        continue
                    dst = dwi_dir / nii.name
                    if not dst.exists():
                        nii.rename(dst)
                        rename_map[(nii.relative_to(self.bids_root)).as_posix()] = (
                            dst.relative_to(self.bids_root).as_posix()
                        )
                    base = re.sub(r'\.nii(\.gz)?$', '', nii.name, flags=re.I)
                    for ext in ['.json', '.bval', '.bvec']:
                        src = fmap_dir / (base + ext)
                        if src.exists():
                            dst_file = dwi_dir / src.name
                            if not dst_file.exists():
                                src.rename(dst_file)
                                rename_map[(src.relative_to(self.bids_root)).as_posix()] = (
                                    dst_file.relative_to(self.bids_root).as_posix()
                                )
        if rename_map:
            try:
                from .scans_utils import update_scans_with_map

                update_scans_with_map(self.bids_root, rename_map)
            except Exception:
                pass

    def _has_any_b0(self) -> bool:
        for sub in self.bids_root.glob('sub-*'):
            if not sub.is_dir():
                continue
            sessions = [s for s in sub.glob('ses-*') if s.is_dir()]
            roots = sessions or [sub]
            for root in roots:
                for folder in [root / 'dwi', root / 'fmap']:
                    if not folder.is_dir():
                        continue
                    for nii in folder.glob('*.nii*'):
                        if self._is_b0(nii.name):
                            return True
        return False

    def _fmap_has_b0(self) -> bool:
        for sub in self.bids_root.glob('sub-*'):
            if not sub.is_dir():
                continue
            sessions = [s for s in sub.glob('ses-*') if s.is_dir()]
            roots = sessions or [sub]
            for root in roots:
                fmap_dir = root / 'fmap'
                if not fmap_dir.is_dir():
                    continue
                for nii in fmap_dir.glob('*.nii*'):
                    if self._is_b0(nii.name):
                        return True
        return False

    @staticmethod
    def _is_b0(name: str) -> bool:
        lower = name.lower()
        return 'b0' in lower or '_epi' in lower

    def _add_fmaps(self, root: Path, bold_parent: QTreeWidgetItem,
                   dwi_parent: QTreeWidgetItem | None,
                   sub: str, ses: str | None) -> None:
        fmap_dir = root / 'fmap'
        func_dir = root / 'func'
        dwi_dir = root / 'dwi'
        func_files = [f.relative_to(root).as_posix()
                      for f in sorted(func_dir.glob('*.nii*')) if f.is_file()]
        dwi_files = [f.relative_to(root).as_posix()
                     for f in sorted(dwi_dir.glob('*.nii*')) if f.is_file()]
        groups: dict[str, list[Path]] = {}
        if fmap_dir.is_dir():
            for js in fmap_dir.glob('*.json'):
                base = re.sub(
                    r'_(magnitude1|magnitude2|phasediff|phase1|phase2)\.json$',
                    '', js.name, flags=re.I)
                groups.setdefault(base, []).append(js)
        for base, files in groups.items():
            key = (sub, ses, base)
            bold_int, dwi_int = self._load_intended(files[0], root)
            self.data[key] = {
                'jsons': files,
                'funcs_bold': func_files,
                'funcs_dwi': dwi_files,
                'intended_bold': bold_int,
                'intended_dwi': dwi_int,
                'root': root,
            }
            if not (self.b0_box.isChecked() and self._is_b0(base)):
                item_bold = QTreeWidgetItem([base])
                item_bold.setData(0, Qt.UserRole, key)
                bold_parent.addChild(item_bold)
            if dwi_parent is not None and self._is_b0(base):
                item_dwi = QTreeWidgetItem([base])
                item_dwi.setData(0, Qt.UserRole, key)
                dwi_parent.addChild(item_dwi)

    def _load_intended(self, path: Path, root: Path) -> tuple[list[str], list[str]]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            val = meta.get('IntendedFor', [])
            prefix = root.relative_to(self.bids_root)

            def _strip(p: str) -> str:
                parts = Path(p)
                try:
                    parts = parts.relative_to(prefix)
                except ValueError:
                    pass
                return parts.as_posix()

            if isinstance(val, str):
                vals = [_strip(val)]
            elif isinstance(val, list):
                vals = [_strip(v) for v in val]
            else:
                vals = []
            bold = [v for v in vals if '/func/' in v]
            dwi = [v for v in vals if '/dwi/' in v]
            return bold, dwi
        except Exception:
            pass
        return [], []

    def _on_left_selected(self, mode: str) -> None:
        tree = self.bold_tree if mode == "bold" else self.dwi_tree
        intended = self.bold_intended if mode == "bold" else self.dwi_intended
        funcs = self.bold_func_list if mode == "bold" else self.dwi_func_list
        it = tree.currentItem()
        if not it:
            return
        key = it.data(0, Qt.UserRole)
        if not key:
            return
        info = self.data.get(key, {})
        intended.clear()
        for f in info.get(f'intended_{mode}', []):
            intended.addItem(f)
        funcs.clear()
        for f in info.get(f'funcs_{mode}', []):
            funcs.addItem(f)

    def _add_selected(self, mode: str) -> None:
        tree = self.bold_tree if mode == "bold" else self.dwi_tree
        func_list = self.bold_func_list if mode == "bold" else self.dwi_func_list
        it = tree.currentItem()
        if not it:
            return
        key = it.data(0, Qt.UserRole)
        if not key:
            return
        info = self.data[key]
        for sel in func_list.selectedItems():
            path = sel.text()
            if path not in info[f'intended_{mode}']:
                info[f'intended_{mode}'].append(path)
        self._on_left_selected(mode)

    def _remove_selected(self, mode: str) -> None:
        tree = self.bold_tree if mode == "bold" else self.dwi_tree
        intended = self.bold_intended if mode == "bold" else self.dwi_intended
        it = tree.currentItem()
        if not it:
            return
        key = it.data(0, Qt.UserRole)
        if not key:
            return
        info = self.data[key]
        remove = [s.text() for s in intended.selectedItems()]
        info[f'intended_{mode}'] = [p for p in info[f'intended_{mode}'] if p not in remove]
        self._on_left_selected(mode)

    def _save_changes(self, mode: str) -> None:
        tree = self.bold_tree if mode == "bold" else self.dwi_tree
        it = tree.currentItem()
        if not it:
            return
        key = it.data(0, Qt.UserRole)
        if not key:
            return
        info = self.data[key]
        val = sorted(info['intended_bold'] + info['intended_dwi'])
        prefix = info['root'].relative_to(self.bids_root)
        cleaned = []
        for p in val:
            path = Path(p)
            try:
                path = path.relative_to(prefix)
            except ValueError:
                pass
            cleaned.append(path.as_posix())
        val = cleaned
        for js in info['jsons']:
            try:
                with open(js, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                meta['IntendedFor'] = val
                with open(js, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=4)
                    f.write('\n')
            except Exception as exc:
                QMessageBox.warning(self, 'Error', f'Failed to save {js}: {exc}')
                return
        QMessageBox.information(self, 'Saved', 'IntendedFor updated.')


class BidsIgnoreDialog(QDialog):
    """Dialog to edit ``.bidsignore`` entries using two selection panels."""

    def __init__(self, parent, bids_root: Path):
        super().__init__(parent)
        self.bids_root = bids_root
        self.setWindowTitle("Edit .bidsignore")
        self.resize(700, 400)

        main = QVBoxLayout(self)

        # --- lists ---
        lists = QHBoxLayout()
        main.addLayout(lists)

        # Left panel: existing entries
        left_box = QVBoxLayout()
        lists.addLayout(left_box)
        left_box.addWidget(QLabel("Ignored files:"))
        self.ignore_list = QListWidget()
        self.ignore_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        left_box.addWidget(self.ignore_list)
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(self._remove_selected)
        left_box.addWidget(rm_btn)

        # Right panel: available files
        right_box = QVBoxLayout()
        lists.addLayout(right_box)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter files…")
        self.search.textChanged.connect(self._populate_lists)
        right_box.addWidget(self.search)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        right_box.addWidget(self.file_list)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_selected)
        right_box.addWidget(add_btn)

        # buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.save)
        btn_box.rejected.connect(self.reject)
        main.addWidget(btn_box)

        self.ignore_file = self.bids_root / ".bidsignore"
        self.entries: set[str] = set()
        if self.ignore_file.exists():
            self.entries = {
                line.strip()
                for line in self.ignore_file.read_text().splitlines()
                if line.strip()
            }

        self.all_files = [
            p.relative_to(self.bids_root).as_posix()
            for p in self.bids_root.rglob('*')
            if p.is_file()
        ]
        self._populate_lists()

    # --- helpers ---
    def _populate_lists(self) -> None:
        """Refresh both panels based on current entries and filter."""
        pattern = self.search.text().strip()

        self.ignore_list.clear()
        for path in sorted(self.entries):
            self.ignore_list.addItem(path)

        self.file_list.clear()
        for path in sorted(self.all_files):
            if path in self.entries:
                continue
            if pattern and pattern not in path:
                continue
            self.file_list.addItem(path)

    def _add_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self.entries.add(item.text())
        self._populate_lists()

    def _remove_selected(self) -> None:
        for item in self.ignore_list.selectedItems():
            self.entries.discard(item.text())
        self._populate_lists()

    # --- save ---
    def save(self) -> None:
        self.ignore_file.write_text("\n".join(sorted(self.entries)) + "\n")
        QMessageBox.information(self, "Saved", f"Updated {self.ignore_file}")
        self.accept()


class Volume3DDialog(QDialog):
    """Interactive 3-D renderer for NIfTI volumes."""

    def __init__(
        self,
        parent,
        data: np.ndarray,
        meta: Optional[dict] = None,
        voxel_sizes: Optional[Sequence[float]] = None,
        default_mode: Optional[str] = None,
        title: Optional[str] = None,
        dark_theme: bool = False,
    ) -> None:
        super().__init__(parent)
        if not HAS_PYQTGRAPH:
            raise RuntimeError(
                "3-D volume rendering requires the optional 'pyqtgraph' dependency."
            )
        self.setWindowTitle(title or "3D Volume Viewer")
        self.resize(1800, 1000)
        self.setMinimumSize(780, 560)
        self.setSizeGripEnabled(True)

        self._raw = np.asarray(data)
        if self._raw.ndim < 3:
            raise ValueError("3-D rendering requires data with at least three axes")

        self._meta = meta or {}
        self._voxel_sizes = self._normalise_voxel_sizes(voxel_sizes)
        self._voxel_sizes_vec = np.asarray(self._voxel_sizes, dtype=np.float32)
        self._dark_theme = bool(dark_theme)
        self._max_points = 120_000
        self._surface_step = 1
        self._initialising = True
        self._scalar_min = 0.0
        self._scalar_max = 0.0
        self._scalar_volume: Optional[np.ndarray] = None
        self._normalised_volume: Optional[np.ndarray] = None
        self._downsampled: Optional[np.ndarray] = None
        self._downsample_step = 1
        self._point_sorted_values: Optional[np.ndarray] = None
        self._point_sorted_indices: Optional[np.ndarray] = None
        self._point_shape: Optional[tuple[int, int, int]] = None
        self._scalar_hist_upper_edges: Optional[np.ndarray] = None
        self._scalar_hist_cumulative: Optional[np.ndarray] = None
        self._scatter_item: Optional[gl.GLScatterPlotItem] = None
        self._mesh_item: Optional[gl.GLMeshItem] = None
        self._axis_item: Optional[gl.GLAxisItem] = None
        self._current_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._colormap_cache: dict[tuple[str, float], mcolors.Colormap] = {}
        self._light_shader = _create_directional_light_shader()
        self._flat_shader = _create_flat_color_shader()
        self._lighting_enabled = True

        self._fg_color = "#f0f0f0" if self._dark_theme else "#202020"
        self._canvas_bg = "#202020" if self._dark_theme else "#ffffff"
        self._axis_label_items: list[gl.GLTextItem] = []
        self._data_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._slice_controls: dict[str, _SliceControl] = {}

        layout = QVBoxLayout(self)

        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(12)
        layout.addWidget(self._splitter)

        self._view_container = QWidget()
        view_layout = QVBoxLayout(self._view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)

        # ``GLViewWidget`` renders using OpenGL so panning/zooming the scene does
        # not require recomputing the voxel subset on every interaction.
        self.view = gl.GLViewWidget()
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setBackgroundColor(self._canvas_bg)
        self.view.opts["distance"] = 200
        self.view.opts["elevation"] = 20
        self.view.opts["azimuth"] = -60
        view_layout.addWidget(self.view)

        self._splitter.addWidget(self._view_container)

        settings_container = QWidget()
        settings_layout = QVBoxLayout(settings_container)
        settings_layout.setContentsMargins(6, 6, 6, 6)
        settings_layout.setSpacing(10)

        # Allow users to dock the settings pane to whichever edge feels most
        # comfortable when working on small screens.
        placement_layout = QHBoxLayout()
        placement_layout.setSpacing(6)
        placement_label = QLabel("Panel placement:")
        placement_layout.addWidget(placement_label)
        self.panel_location_combo = QComboBox()
        self.panel_location_combo.addItems(["Bottom", "Left", "Right"])
        # Default to a side-by-side layout so the viewport opens on the left.
        self.panel_location_combo.setCurrentText("Right")
        placement_layout.addWidget(self.panel_location_combo)
        placement_layout.addStretch()
        settings_layout.addLayout(placement_layout)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        agg_label = QLabel("Aggregation:")
        header_row.addWidget(agg_label)
        self.agg_combo = QComboBox()
        self._aggregators = {}
        self._init_aggregator_options()
        header_row.addWidget(self.agg_combo)
        header_row.addSpacing(12)

        render_label = QLabel("Render mode:")
        header_row.addWidget(render_label)
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems(["Point cloud", "Surface mesh"])
        self.render_mode_combo.setToolTip(
            "Switch between scattered voxels and iso-surface rendering.",
        )
        header_row.addWidget(self.render_mode_combo)
        header_row.addStretch()
        controls_layout.addLayout(header_row)

        # Stack the render-mode sliders beneath the mode selector so they are
        # visually associated with the chosen representation.
        slider_grid = QGridLayout()
        slider_grid.setContentsMargins(0, 0, 0, 0)
        slider_grid.setHorizontalSpacing(8)
        slider_grid.setVerticalSpacing(6)
        slider_grid.setColumnStretch(1, 1)

        self.thresh_text = QLabel("Threshold:")
        slider_grid.addWidget(self.thresh_text, 0, 0)
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(60)
        self.thresh_slider.setPageStep(5)
        self.thresh_slider.setToolTip(
            "Minimum normalised intensity included in the rendering.",
        )
        self.thresh_slider.valueChanged.connect(self._update_plot)
        slider_grid.addWidget(self.thresh_slider, 0, 1)
        self.thresh_label = QLabel("0.60")
        slider_grid.addWidget(self.thresh_label, 0, 2)

        self.point_label = QLabel("Point size:")
        slider_grid.addWidget(self.point_label, 1, 0)
        self.point_slider = QSlider(Qt.Horizontal)
        self.point_slider.setRange(1, 12)
        self.point_slider.setValue(4)
        self.point_slider.setToolTip("Marker diameter for point-cloud rendering.")
        self.point_slider.valueChanged.connect(self._update_plot)
        slider_grid.addWidget(self.point_slider, 1, 1)

        controls_layout.addLayout(slider_grid)
        controls_layout.addStretch()
        settings_layout.addWidget(controls_widget)

        options_group = QGroupBox("Rendering options")
        options = QGridLayout(options_group)
        options.setColumnStretch(1, 1)

        row = 0
        options.addWidget(QLabel("Colormap:"), row, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "twilight",
            "cubehelix",
            "Spectral",
            "coolwarm",
            "YlGnBu",
            "Greys",
            "bone",
        ])
        self.colormap_combo.setToolTip(
            "Select the matplotlib colormap for voxel intensities.",
        )
        self.colormap_combo.currentTextChanged.connect(lambda _: self._update_plot())
        options.addWidget(self.colormap_combo, row, 1)

        row += 1
        options.addWidget(QLabel("Opacity:"), row, 0)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.setToolTip(
            "Global alpha applied to rendered points or the surface mesh.",
        )
        options.addWidget(self.opacity_slider, row, 1)
        self.opacity_label = QLabel("0.80")
        options.addWidget(self.opacity_label, row, 2)

        row += 1
        options.addWidget(QLabel("Colour intensity:"), row, 0)
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(10, 200)
        self.intensity_slider.setValue(100)
        self.intensity_slider.setToolTip(
            "Scale factor applied to RGB values after colormap lookup (0.1–2.0×).",
        )
        options.addWidget(self.intensity_slider, row, 1)
        self.intensity_label = QLabel("1.00×")
        options.addWidget(self.intensity_label, row, 2)

        row += 1
        options.addWidget(QLabel("Maximum points:"), row, 0)
        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(1_000, 20_000_000)
        self.max_points_spin.setSingleStep(5_000)
        self.max_points_spin.setValue(self._max_points)
        self.max_points_spin.setToolTip(
            "Upper bound on voxels displayed in point-cloud mode.",
        )
        options.addWidget(self.max_points_spin, row, 1)

        row += 1
        options.addWidget(QLabel("Downsample step:"), row, 0)
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(0, 8)
        self.downsample_spin.setValue(0)
        self.downsample_spin.setToolTip(
            "Manual voxel stride applied before rendering (0 = automatic).",
        )
        options.addWidget(self.downsample_spin, row, 1)

        row += 1
        self.surface_step_label = QLabel("Marching cubes step:")
        options.addWidget(self.surface_step_label, row, 0)
        self.surface_step_spin = QSpinBox()
        self.surface_step_spin.setRange(1, 6)
        self.surface_step_spin.setValue(self._surface_step)
        self.surface_step_spin.setToolTip(
            "Sampling stride for marching cubes when computing iso-surfaces.",
        )
        options.addWidget(self.surface_step_spin, row, 1)

        row += 1
        self.axes_checkbox = QCheckBox("Show axes")
        self.axes_checkbox.setChecked(False)
        options.addWidget(self.axes_checkbox, row, 0, 1, 2)

        row += 1
        options.addWidget(QLabel("Axis thickness:"), row, 0)
        self.axis_thickness_slider = QSlider(Qt.Horizontal)
        self.axis_thickness_slider.setRange(1, 12)
        self.axis_thickness_slider.setValue(3)
        self.axis_thickness_slider.setToolTip(
            "Line width of the anatomical axes (pixels).",
        )
        options.addWidget(self.axis_thickness_slider, row, 1)
        self.axis_thickness_value = QLabel("3 px")
        options.addWidget(self.axis_thickness_value, row, 2)

        row += 1
        self.axis_labels_checkbox = QCheckBox("Show axis labels")
        self.axis_labels_checkbox.setChecked(False)
        options.addWidget(self.axis_labels_checkbox, row, 0, 1, 2)
        self.axis_thickness_slider.setEnabled(False)
        self.axis_labels_checkbox.setEnabled(False)
        settings_layout.addWidget(options_group)

        # Embed the colour bar alongside the rest of the controls so it folds
        # away with the settings pane.
        colorbar_group = QGroupBox("Colour bar")
        colorbar_layout = QVBoxLayout(colorbar_group)
        colorbar_layout.setContentsMargins(8, 6, 8, 6)
        colorbar_layout.setSpacing(4)
        self._colorbar_canvas = FigureCanvas(plt.Figure(figsize=(2.6, 1.4)))
        self._colorbar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._colorbar_canvas.setFixedHeight(180)
        self._colorbar_canvas.figure.patch.set_facecolor(self._canvas_bg)
        colorbar_layout.addWidget(self._colorbar_canvas)

        # Host the colour bar in its own lightweight container so it can share
        # the retractable pane while remaining independent from the parameter list.
        self._colorbar_panel = QWidget()
        colorbar_panel_layout = QVBoxLayout(self._colorbar_panel)
        colorbar_panel_layout.setContentsMargins(6, 6, 6, 6)
        colorbar_panel_layout.setSpacing(6)
        colorbar_panel_layout.addWidget(colorbar_group)

        slice_group = QGroupBox("Slice planes")
        slice_layout = QVBoxLayout(slice_group)
        slice_layout.setContentsMargins(8, 4, 8, 4)
        slice_layout.setSpacing(6)
        self._slice_controls.clear()
        for key, axis_idx, neg_name, pos_name in _SLICE_ORIENTATIONS:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(6)
            checkbox = QCheckBox(key.capitalize())
            checkbox.setChecked(False)
            row_layout.addWidget(checkbox)
            row_layout.addSpacing(4)
            row_layout.addWidget(QLabel(f"{neg_name}:"))
            min_slider = QSlider(Qt.Horizontal)
            min_slider.setRange(0, 100)
            min_slider.setValue(0)
            min_slider.setEnabled(False)
            row_layout.addWidget(min_slider, 1)
            min_value = QLabel("0.0 mm")
            row_layout.addWidget(min_value)
            row_layout.addSpacing(6)
            row_layout.addWidget(QLabel(f"{pos_name}:"))
            max_slider = QSlider(Qt.Horizontal)
            max_slider.setRange(0, 100)
            max_slider.setValue(100)
            max_slider.setEnabled(False)
            row_layout.addWidget(max_slider, 1)
            max_value = QLabel("0.0 mm")
            row_layout.addWidget(max_value)
            row_layout.addStretch()
            slice_layout.addLayout(row_layout)
            control = _SliceControl(
                checkbox=checkbox,
                min_slider=min_slider,
                max_slider=max_slider,
                min_value=min_value,
                max_value=max_value,
                axis=axis_idx,
                negative_name=neg_name,
                positive_name=pos_name,
            )
            self._slice_controls[key] = control
            checkbox.toggled.connect(lambda checked, name=key: self._on_slice_toggle(name, checked))
            min_slider.valueChanged.connect(lambda value, name=key: self._on_slice_slider_change(name, "min", value))
            max_slider.valueChanged.connect(lambda value, name=key: self._on_slice_slider_change(name, "max", value))
        slice_layout.addStretch()
        settings_layout.addWidget(slice_group)

        self.lighting_group = QGroupBox("Lighting")
        light_layout = QGridLayout(self.lighting_group)
        light_row = 0
        self.light_enable_checkbox = QCheckBox("Enable lighting")
        self.light_enable_checkbox.setChecked(True)
        light_layout.addWidget(self.light_enable_checkbox, light_row, 0, 1, 3)
        light_row += 1
        light_layout.addWidget(QLabel("Azimuth:"), light_row, 0)
        self.light_azimuth_slider = QSlider(Qt.Horizontal)
        self.light_azimuth_slider.setRange(-180, 180)
        self.light_azimuth_slider.setValue(-45)
        self.light_azimuth_slider.setToolTip(
            "Horizontal direction of the light source relative to the mesh (°).",
        )
        light_layout.addWidget(self.light_azimuth_slider, light_row, 1)
        self.light_azimuth_label = QLabel("-45°")
        light_layout.addWidget(self.light_azimuth_label, light_row, 2)

        light_row += 1
        light_layout.addWidget(QLabel("Elevation:"), light_row, 0)
        self.light_elevation_slider = QSlider(Qt.Horizontal)
        self.light_elevation_slider.setRange(-90, 90)
        self.light_elevation_slider.setValue(30)
        self.light_elevation_slider.setToolTip(
            "Vertical angle of the light source relative to the mesh (°).",
        )
        light_layout.addWidget(self.light_elevation_slider, light_row, 1)
        self.light_elevation_label = QLabel("30°")
        light_layout.addWidget(self.light_elevation_label, light_row, 2)

        light_row += 1
        light_layout.addWidget(QLabel("Intensity:"), light_row, 0)
        self.light_intensity_slider = QSlider(Qt.Horizontal)
        self.light_intensity_slider.setRange(10, 300)
        self.light_intensity_slider.setValue(130)
        self.light_intensity_slider.setToolTip(
            "Diffuse lighting strength (0.1–3.0×). Ambient light stays constant.",
        )
        light_layout.addWidget(self.light_intensity_slider, light_row, 1)
        self.light_intensity_label = QLabel("1.30×")
        light_layout.addWidget(self.light_intensity_label, light_row, 2)
        if self._light_shader is None and self._flat_shader is None:
            self.lighting_group.setEnabled(False)
        settings_layout.addWidget(self.lighting_group)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        settings_layout.addWidget(self.status_label)
        settings_layout.addStretch()

        self._panel_scroll = ShrinkableScrollArea()
        self._panel_scroll.setWidget(settings_container)
        self._panel_scroll.setWidgetResizable(True)
        self._panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # A nested splitter keeps the scrollable controls and the colour bar in
        # separate panes so each can be resized without affecting the other.
        self._panel_splitter = QSplitter(Qt.Vertical)
        self._panel_splitter.setChildrenCollapsible(False)
        self._panel_splitter.addWidget(self._panel_scroll)
        self._panel_splitter.addWidget(self._colorbar_panel)
        self._panel_splitter.setStretchFactor(0, 3)
        self._panel_splitter.setStretchFactor(1, 1)

        self._panel_container = QWidget()
        panel_container_layout = QVBoxLayout(self._panel_container)
        panel_container_layout.setContentsMargins(0, 0, 0, 0)
        panel_container_layout.setSpacing(0)
        panel_container_layout.addWidget(self._panel_splitter)

        self._splitter.addWidget(self._panel_container)

        default_name = None
        if default_mode and default_mode in self._aggregators:
            default_name = default_mode
        elif self._raw.ndim > 3:
            if self._meta.get("is_rgb"):
                default_name = "Mean |value|"
            else:
                default_name = "RMS (DTI)"

        if default_name:
            self.agg_combo.setCurrentText(default_name)

        self.panel_location_combo.currentTextChanged.connect(self._on_panel_location_change)
        self.agg_combo.currentTextChanged.connect(self._on_agg_change)
        self.render_mode_combo.currentTextChanged.connect(self._on_render_mode_change)
        self.opacity_slider.valueChanged.connect(self._on_opacity_change)
        self.intensity_slider.valueChanged.connect(self._on_intensity_change)
        self.max_points_spin.valueChanged.connect(self._on_max_points_change)
        self.downsample_spin.valueChanged.connect(self._on_downsample_change)
        self.surface_step_spin.valueChanged.connect(self._on_surface_step_change)
        self.axes_checkbox.toggled.connect(self._on_axes_toggle)
        self.axis_thickness_slider.valueChanged.connect(self._on_axis_thickness_change)
        self.axis_labels_checkbox.toggled.connect(self._on_axis_labels_toggle)
        self.light_enable_checkbox.toggled.connect(self._on_light_enabled_toggle)
        self.light_azimuth_slider.valueChanged.connect(self._on_light_setting_change)
        self.light_elevation_slider.valueChanged.connect(self._on_light_setting_change)
        self.light_intensity_slider.valueChanged.connect(self._on_light_setting_change)

        self._apply_panel_layout(self.panel_location_combo.currentText())
        self._update_light_controls_enabled()

        # Prime the UI labels without triggering expensive redraws while we are
        # still constructing the dialog.
        self._on_intensity_change(self.intensity_slider.value())
        self._on_axis_thickness_change(self.axis_thickness_slider.value())
        self._update_slice_labels()
        self._update_light_labels()
        self._update_light_shader()

        self._initialising = False
        self._update_mode_dependent_controls()
        self._compute_scalar_volume()
        self._update_plot()

    def _apply_panel_layout(self, location: str) -> None:
        """Reposition the settings pane relative to the 3-D viewport."""

        if not hasattr(self, "_splitter"):
            return

        normalised = (location or "bottom").strip().lower()
        if normalised not in {"bottom", "left", "right"}:
            normalised = "bottom"

        orientation = Qt.Vertical if normalised == "bottom" else Qt.Horizontal
        self._splitter.setOrientation(orientation)

        view_first = True
        if orientation == Qt.Horizontal:
            view_first = normalised != "left"

        panel_widget = getattr(self, "_panel_container", None)
        widgets = [self._view_container, panel_widget]
        if panel_widget is None:
            widgets = [self._view_container]
        if not view_first and len(widgets) == 2:
            widgets.reverse()
        for index, widget in enumerate(widgets):
            if widget is None:
                continue
            if self._splitter.indexOf(widget) != index:
                self._splitter.insertWidget(index, widget)

        # Bias the main splitter so the viewport consumes roughly three quarters
        # of the available space in the default layout.
        stretch = (3, 1) if view_first else (1, 3)
        self._splitter.setStretchFactor(0, stretch[0])
        if len(widgets) > 1:
            self._splitter.setStretchFactor(1, stretch[1])
            scale = 120
            self._splitter.setSizes([stretch[0] * scale, stretch[1] * scale])

        panel_splitter = getattr(self, "_panel_splitter", None)
        if isinstance(panel_splitter, QSplitter):
            # Rotate the internal splitter so the colour bar always occupies its
            # own pane next to the parameter controls regardless of docking side.
            if orientation == Qt.Vertical:
                panel_splitter.setOrientation(Qt.Horizontal)
                panel_splitter.setSizes([220, 160])
            else:
                panel_splitter.setOrientation(Qt.Vertical)
                panel_splitter.setSizes([300, 160])

    def _on_panel_location_change(self, location: str) -> None:
        self._apply_panel_layout(location)

    def _update_light_controls_enabled(self) -> None:
        enabled = bool(self.lighting_group.isEnabled() and self._lighting_enabled)
        for widget in (
            self.light_azimuth_slider,
            self.light_elevation_slider,
            self.light_intensity_slider,
        ):
            widget.setEnabled(enabled)
        for label in (
            self.light_azimuth_label,
            self.light_elevation_label,
            self.light_intensity_label,
        ):
            label.setEnabled(enabled)

    def _apply_mesh_shader(self) -> None:
        if self._mesh_item is None:
            return
        if self._light_shader is not None and self._lighting_enabled:
            self._mesh_item.setShader(self._light_shader)
        elif self._flat_shader is not None:
            self._mesh_item.setShader(self._flat_shader)
        else:
            self._mesh_item.setShader("shaded")

    def _on_light_enabled_toggle(self, checked: bool) -> None:
        self._lighting_enabled = bool(checked)
        self._update_light_controls_enabled()
        if self._lighting_enabled:
            self._update_light_shader()
        else:
            self._apply_mesh_shader()
        if not self._initialising and self._mesh_item is not None:
            self._mesh_item.update()
            self.view.update()

    def _normalise_voxel_sizes(self, voxel_sizes):
        if not voxel_sizes:
            return (1.0, 1.0, 1.0)
        values = tuple(float(v) for v in voxel_sizes[:3])
        if len(values) < 3:
            values = values + (1.0,) * (3 - len(values))
        return tuple(max(1e-6, abs(v)) for v in values)

    def _init_aggregator_options(self) -> None:
        if self._raw.ndim <= 3:
            self._aggregators["Intensity"] = "Intensity"
            self.agg_combo.addItem("Intensity")
            self.agg_combo.setEnabled(False)
        else:
            for name in ("RMS (DTI)", "Mean |value|", "Max |value|", "First volume"):
                self._aggregators[name] = name
                self.agg_combo.addItem(name)

    def _on_agg_change(self):
        if self._initialising:
            return
        self._compute_scalar_volume()
        self._update_plot()

    def _compute_scalar_volume(self) -> None:
        arr = np.asarray(self._raw, dtype=np.float32)
        if arr.ndim <= 3:
            scalar = arr
        else:
            axis = arr.ndim - 1
            mode = self.agg_combo.currentText()
            if mode == "RMS (DTI)":
                scalar = np.sqrt(np.nanmean(np.square(arr), axis=axis))
            elif mode == "Max |value|":
                scalar = np.nanmax(np.abs(arr), axis=axis)
            elif mode == "First volume":
                scalar = np.abs(np.take(arr, 0, axis=axis))
            else:
                scalar = np.nanmean(np.abs(arr), axis=axis)
        scalar = np.nan_to_num(scalar, nan=0.0, posinf=0.0, neginf=0.0)
        self._scalar_volume = scalar
        finite = np.isfinite(scalar)
        if not finite.any():
            self._scalar_min = 0.0
            self._scalar_max = 0.0
            self._normalised_volume = np.zeros_like(scalar, dtype=np.float32)
        else:
            min_val = float(np.min(scalar[finite]))
            max_val = float(np.max(scalar[finite]))
            self._scalar_min = min_val
            self._scalar_max = max_val
            if abs(max_val - min_val) < 1e-6:
                self._normalised_volume = np.zeros_like(scalar, dtype=np.float32)
            else:
                norm = (scalar - min_val) / (max_val - min_val)
                self._normalised_volume = norm.astype(np.float32, copy=False)
        self._build_scalar_histogram()
        self._prepare_downsampled()

    def _build_scalar_histogram(self) -> None:
        self._scalar_hist_upper_edges = None
        self._scalar_hist_cumulative = None
        volume = self._normalised_volume
        if volume is None:
            return

        flattened = np.asarray(volume, dtype=np.float32).ravel()
        if flattened.size == 0:
            self._scalar_hist_upper_edges = np.empty(0, dtype=np.float32)
            self._scalar_hist_cumulative = np.empty(0, dtype=np.int64)
            return

        edges = np.linspace(0.0, 1.0, 513, dtype=np.float32)
        counts, _ = np.histogram(flattened, bins=edges)
        cumulative = np.cumsum(counts[::-1], dtype=np.int64)[::-1]
        self._scalar_hist_upper_edges = edges[1:]
        self._scalar_hist_cumulative = cumulative

    def _prepare_downsampled(self) -> None:
        vol = self._normalised_volume
        if vol is None:
            self._downsampled = None
            self._downsample_step = 1
            self._data_bounds = None
            self._update_slice_labels()
            return
        manual_step = 0
        if hasattr(self, "downsample_spin"):
            manual_step = int(self.downsample_spin.value())
        if manual_step > 0:
            step = manual_step
        else:
            step = 1
            total = float(vol.size)
            if total > self._max_points:
                step = int(np.ceil(np.cbrt(total / self._max_points)))
        step = max(1, step)
        slices = tuple(slice(None, None, step) for _ in range(3))
        self._downsampled = vol[slices]
        self._downsample_step = step
        if self._downsampled.size == 0:
            self._data_bounds = None
        else:
            ds_shape = np.ones(3, dtype=np.float32)
            dims = min(self._downsampled.ndim, 3)
            ds_shape[:dims] = np.asarray(self._downsampled.shape[:dims], dtype=np.float32)
            scale = self._voxel_sizes_vec * float(step)
            spans = np.maximum((ds_shape - 1.0) * scale, 0.0)
            mins = np.zeros(3, dtype=np.float32)
            self._data_bounds = (mins, mins + spans)
        self._update_slice_labels()
        self._rebuild_point_cache()

    def _rebuild_point_cache(self) -> None:
        self._point_sorted_values = None
        self._point_sorted_indices = None
        self._point_shape = None
        downsampled = self._downsampled
        if downsampled is None:
            return

        flat = np.asarray(downsampled, dtype=np.float32).ravel()
        self._point_shape = downsampled.shape
        if flat.size == 0:
            self._point_sorted_values = np.empty(0, dtype=np.float32)
            self._point_sorted_indices = np.empty(0, dtype=np.int64)
            return

        order = np.argsort(flat, kind="mergesort")
        self._point_sorted_indices = order.astype(np.int64, copy=False)
        self._point_sorted_values = flat.take(order)

    def _indices_to_world(
        self, indices: np.ndarray, shape: tuple[int, int, int], scale: np.ndarray
    ) -> np.ndarray:
        coords_mm = np.empty((indices.size, 3), dtype=np.float32)
        if indices.size == 0:
            return coords_mm

        plane = int(shape[1]) * int(shape[2])
        axis0 = indices // plane
        remainder = indices - axis0 * plane
        axis1 = remainder // int(shape[2])
        axis2 = remainder - axis1 * int(shape[2])

        coords_mm[:, 0] = axis0.astype(np.float32, copy=False)
        coords_mm[:, 1] = axis1.astype(np.float32, copy=False)
        coords_mm[:, 2] = axis2.astype(np.float32, copy=False)
        coords_mm *= scale
        return coords_mm

    def _estimate_voxels_above_threshold(self, thr: float) -> int:
        if self._scalar_hist_upper_edges is None or self._scalar_hist_cumulative is None:
            if self._normalised_volume is None:
                return 0
            return int(np.count_nonzero(self._normalised_volume >= thr))

        idx = int(np.searchsorted(self._scalar_hist_upper_edges, thr, side="left"))
        if idx >= self._scalar_hist_cumulative.size:
            return 0
        return int(self._scalar_hist_cumulative[idx])

    def _current_color_intensity(self) -> float:
        slider = getattr(self, "intensity_slider", None)
        if slider is None:
            return 1.0
        return max(0.1, slider.value() / 100.0)

    def _get_adjusted_colormap(self, cmap_name: str) -> mcolors.Colormap:
        intensity = round(self._current_color_intensity(), 3)
        key = (cmap_name, intensity)
        cached = self._colormap_cache.get(key)
        if cached is not None:
            return cached
        base = plt.get_cmap(cmap_name)
        samples = base(np.linspace(0.0, 1.0, 512))
        samples[:, :3] = np.clip(samples[:, :3] * intensity, 0.0, 1.0)
        cmap = mcolors.ListedColormap(samples, name=f"{cmap_name}_x{intensity:.3f}")
        self._colormap_cache[key] = cmap
        return cmap

    def _map_colors(
        self, values: np.ndarray, cmap: mcolors.Colormap, alpha: float
    ) -> np.ndarray:
        # Clamp to the normalised range and map to RGBA so OpenGL can upload a
        # single colour array for all voxels.
        colours = np.asarray(cmap(np.clip(values, 0.0, 1.0)), dtype=np.float32)
        colours[:, 3] = np.clip(colours[:, 3] * alpha, 0.0, 1.0)
        return colours

    def _update_slice_labels(self) -> None:
        if not self._slice_controls:
            return
        bounds = self._data_bounds
        has_bounds = bounds is not None
        mins: Optional[np.ndarray]
        maxs: Optional[np.ndarray]
        if has_bounds:
            mins, maxs = bounds  # type: ignore[assignment]
        else:
            mins = maxs = None
        for control in self._slice_controls.values():
            slider_enabled = bool(has_bounds and control.checkbox.isChecked())
            control.min_slider.setEnabled(slider_enabled)
            control.max_slider.setEnabled(slider_enabled)
            if mins is None or maxs is None:
                control.min_value.setText("--")
                control.max_value.setText("--")
                continue
            axis = control.axis
            axis_min = float(mins[axis])
            axis_max = float(maxs[axis])
            span = axis_max - axis_min
            min_frac = control.min_slider.value() / 100.0
            max_frac = control.max_slider.value() / 100.0
            min_mm = axis_min + span * min_frac
            max_mm = axis_min + span * max_frac
            control.min_value.setText(f"{min_mm:.1f} mm")
            control.max_value.setText(f"{max_mm:.1f} mm")

    def _active_slice_bounds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if not self._slice_controls or self._data_bounds is None:
            return None
        mins, maxs = self._data_bounds
        min_bounds = np.asarray(mins, dtype=np.float32).copy()
        max_bounds = np.asarray(maxs, dtype=np.float32).copy()
        active = False
        for control in self._slice_controls.values():
            if not control.checkbox.isChecked():
                continue
            axis = control.axis
            axis_min = float(mins[axis])
            axis_max = float(maxs[axis])
            span = axis_max - axis_min
            min_frac = control.min_slider.value() / 100.0
            max_frac = control.max_slider.value() / 100.0
            min_bounds[axis] = float(axis_min + span * min_frac)
            max_bounds[axis] = float(axis_min + span * max_frac)
            active = True
        if not active:
            return None
        return min_bounds, max_bounds

    def _slice_status_suffix(self) -> str:
        bounds = self._active_slice_bounds()
        if bounds is None:
            return ""
        mins, maxs = bounds
        parts: list[str] = []
        for control in self._slice_controls.values():
            if not control.checkbox.isChecked():
                continue
            axis = control.axis
            parts.append(
                f"{control.negative_name}–{control.positive_name} "
                f"{mins[axis]:.1f}–{maxs[axis]:.1f} mm"
            )
        if not parts:
            return ""
        return " Slice windows: " + ", ".join(parts) + "."

    def _mask_coords(
        self, coords: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        mins, maxs = bounds
        if coords.size == 0:
            return np.zeros(0, dtype=bool)
        mask = np.ones(coords.shape[0], dtype=bool)
        for axis in range(3):
            mask &= (coords[:, axis] >= mins[axis]) & (coords[:, axis] <= maxs[axis])
        return mask

    def _apply_slice_to_points(
        self, coords: np.ndarray, values: Optional[np.ndarray]
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[tuple[np.ndarray, np.ndarray]]]:
        bounds = self._active_slice_bounds()
        if bounds is None or coords.size == 0:
            return coords, values, None
        mask = self._mask_coords(coords, bounds)
        if not np.any(mask):
            empty_coords = coords[:0].reshape(0, 3)
            empty_vals = values[:0] if values is not None else None
            return empty_coords, empty_vals, bounds
        filtered_coords = coords[mask]
        filtered_values = values[mask] if values is not None else None
        return filtered_coords, filtered_values, bounds

    def _clip_mesh_to_slices(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        values: Optional[np.ndarray],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[tuple[np.ndarray, np.ndarray]],
    ]:
        bounds = self._active_slice_bounds()
        if bounds is None or verts.size == 0:
            return verts, faces, values, None
        mask = self._mask_coords(verts, bounds)
        if not np.any(mask):
            empty_verts = verts[:0].reshape(0, 3)
            empty_faces = faces[:0].reshape(0, 3)
            empty_vals = values[:0] if values is not None else None
            return empty_verts, empty_faces, empty_vals, bounds
        selected = np.where(mask)[0]
        index_map = -np.ones(mask.shape[0], dtype=np.int64)
        index_map[selected] = np.arange(selected.size, dtype=np.int64)
        mapped_faces = index_map[faces]
        valid = (mapped_faces >= 0).all(axis=1)
        mapped_faces = mapped_faces[valid]
        if mapped_faces.size == 0:
            empty_verts = verts[:0].reshape(0, 3)
            empty_faces = faces[:0].reshape(0, 3)
            empty_vals = values[:0] if values is not None else None
            return empty_verts, empty_faces, empty_vals, bounds
        new_verts = verts[selected]
        new_values = values[selected] if values is not None else None
        return new_verts, mapped_faces.astype(np.int32, copy=False), new_values, bounds

    def _on_slice_toggle(self, name: str, _checked: bool) -> None:
        if name not in self._slice_controls:
            return
        self._update_slice_labels()
        if not self._initialising:
            self._update_plot()

    def _on_slice_slider_change(self, name: str, which: str, value: int) -> None:
        control = self._slice_controls.get(name)
        if control is None:
            return
        if which == "min":
            other = control.max_slider
            if value > other.value():
                other.blockSignals(True)
                other.setValue(value)
                other.blockSignals(False)
        else:
            other = control.min_slider
            if value < other.value():
                other.blockSignals(True)
                other.setValue(value)
                other.blockSignals(False)
        self._update_slice_labels()
        if not self._initialising and control.checkbox.isChecked():
            self._update_plot()

    def _remove_axis_labels(self) -> None:
        if not self._axis_label_items:
            return
        for item in self._axis_label_items:
            try:
                self.view.removeItem(item)
            except Exception:  # pragma: no cover - defensive cleanup
                continue
        self._axis_label_items.clear()

    def _update_axis_labels(self) -> None:
        self._remove_axis_labels()
        if (
            getattr(self, "axis_labels_checkbox", None) is None
            or not self.axes_checkbox.isChecked()
            or not self.axis_labels_checkbox.isChecked()
            or self._current_bounds is None
        ):
            return
        mins, maxs = self._current_bounds
        spans = np.maximum(maxs - mins, 1e-6)
        safe_spans = np.where(spans > 0, spans, 1.0)
        centre = (mins + maxs) / 2.0
        offsets = safe_spans * 0.05
        colour = QColor(self._fg_color)
        font = QFont("Helvetica", 13)

        for _name, axis_idx, neg_name, pos_name in _SLICE_ORIENTATIONS:
            if axis_idx == 0:
                neg_pos = [
                    float(mins[0] - offsets[0]),
                    float(centre[1]),
                    float(centre[2]),
                ]
                pos_pos = [
                    float(maxs[0] + offsets[0]),
                    float(centre[1]),
                    float(centre[2]),
                ]
            elif axis_idx == 1:
                neg_pos = [
                    float(centre[0]),
                    float(mins[1] - offsets[1]),
                    float(centre[2]),
                ]
                pos_pos = [
                    float(centre[0]),
                    float(maxs[1] + offsets[1]),
                    float(centre[2]),
                ]
            else:
                neg_pos = [
                    float(centre[0]),
                    float(centre[1]),
                    float(mins[2] - offsets[2]),
                ]
                pos_pos = [
                    float(centre[0]),
                    float(centre[1]),
                    float(maxs[2] + offsets[2]),
                ]
            for text, position in ((neg_name, neg_pos), (pos_name, pos_pos)):
                label = gl.GLTextItem()
                label.setData(text=text, pos=position, color=colour, font=font)
                self.view.addItem(label)
                self._axis_label_items.append(label)

    def _update_axis_item(self) -> None:
        if self._axis_item is not None:
            self.view.removeItem(self._axis_item)
            self._axis_item = None
        if not self.axes_checkbox.isChecked() or self._current_bounds is None:
            self._remove_axis_labels()
            return
        mins, maxs = self._current_bounds
        spans = np.maximum(maxs - mins, 1e-3)
        axis = _AdjustableAxisItem() if "_AdjustableAxisItem" in globals() else gl.GLAxisItem()
        axis.setSize(spans[0], spans[1], spans[2])
        axis.translate(float(mins[0]), float(mins[1]), float(mins[2]))
        thickness = getattr(self, "axis_thickness_slider", None)
        if thickness is not None and hasattr(axis, "setLineWidth"):
            axis.setLineWidth(float(thickness.value()))
        self.view.addItem(axis)
        self._axis_item = axis
        self._update_axis_labels()

    def _on_axes_toggle(self, checked: bool) -> None:
        slider = getattr(self, "axis_thickness_slider", None)
        if slider is not None:
            slider.setEnabled(checked)
        label_cb = getattr(self, "axis_labels_checkbox", None)
        if label_cb is not None:
            label_cb.setEnabled(checked)
            if not checked:
                self._remove_axis_labels()
        self._update_axis_item()

    def _on_axis_thickness_change(self, value: int) -> None:
        if hasattr(self, "axis_thickness_value"):
            self.axis_thickness_value.setText(f"{value} px")
        if self._axis_item is not None and hasattr(self._axis_item, "setLineWidth"):
            self._axis_item.setLineWidth(float(value))
            self.view.update()

    def _on_axis_labels_toggle(self, checked: bool) -> None:
        if not checked:
            self._remove_axis_labels()
        else:
            self._update_axis_labels()

    def _on_intensity_change(self, value: int) -> None:
        self.intensity_label.setText(f"{value / 100.0:.2f}×")
        if not self._initialising:
            self._update_plot()

    def _update_light_labels(self) -> None:
        self.light_azimuth_label.setText(f"{self.light_azimuth_slider.value():+d}°")
        self.light_elevation_label.setText(
            f"{self.light_elevation_slider.value():+d}°"
        )
        self.light_intensity_label.setText(
            f"{self.light_intensity_slider.value() / 100.0:.2f}×"
        )

    def _update_light_shader(self) -> None:
        shader = self._light_shader
        if shader is None or not self._lighting_enabled:
            self._apply_mesh_shader()
            return
        azimuth = math.radians(self.light_azimuth_slider.value())
        elevation = math.radians(self.light_elevation_slider.value())
        cos_el = math.cos(elevation)
        direction = [
            float(cos_el * math.cos(azimuth)),
            float(cos_el * math.sin(azimuth)),
            float(math.sin(elevation)),
        ]
        shader["lightDir"] = direction
        shader["lightParams"] = [
            max(0.0, self.light_intensity_slider.value() / 100.0),
            0.35,
        ]
        self._apply_mesh_shader()

    def _on_light_setting_change(self, _value: int) -> None:
        self._update_light_labels()
        self._update_light_shader()
        if self._initialising:
            return
        if self.render_mode_combo.currentText() == "Surface mesh" and self._mesh_item is not None:
            self._mesh_item.update()
            self.view.update()

    def _clear_colorbar(self) -> None:
        fig = self._colorbar_canvas.figure
        fig.clf()
        fig.patch.set_facecolor(self._canvas_bg)
        self._colorbar_canvas.draw_idle()

    def _update_colorbar(
        self,
        cmap: mcolors.Colormap,
        vmin: float = 0.0,
        vmax: float = 1.0,
        label: str = "Normalised intensity",
    ) -> None:
        fig = self._colorbar_canvas.figure
        fig.clf()
        # ``ScalarMappable`` draws a classic matplotlib colour bar giving users a
        # persistent reference for the current normalisation range. We render it
        # horizontally so it reads naturally underneath the legend label while
        # leaving more vertical room for the surrounding controls.
        ax = fig.add_axes([0.12, 0.35, 0.76, 0.32])
        norm = plt.Normalize(vmin, vmax)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(mappable, cax=ax, orientation="horizontal")
        colorbar.set_label(label, color=self._fg_color, labelpad=6)
        colorbar.ax.xaxis.set_ticks_position("bottom")
        colorbar.ax.xaxis.set_label_position("bottom")
        colorbar.ax.tick_params(colors=self._fg_color, which="both")
        for spine in colorbar.ax.spines.values():
            spine.set_color(self._fg_color)
        colorbar.ax.set_facecolor(self._canvas_bg)
        fig.patch.set_facecolor(self._canvas_bg)
        self._colorbar_canvas.draw_idle()

    def _remove_axis_labels(self) -> None:
        if not self._axis_label_items:
            return
        for item in self._axis_label_items:
            try:
                self.view.removeItem(item)
            except Exception:  # pragma: no cover - defensive cleanup
                continue
        self._axis_label_items.clear()

    def _update_axis_labels(self) -> None:
        self._remove_axis_labels()
        if (
            getattr(self, "axis_labels_checkbox", None) is None
            or not self.axes_checkbox.isChecked()
            or not self.axis_labels_checkbox.isChecked()
            or self._current_bounds is None
        ):
            return
        mins, maxs = self._current_bounds
        spans = np.maximum(maxs - mins, 1e-6)
        safe_spans = np.where(spans > 0, spans, 1.0)
        centre = (mins + maxs) / 2.0
        offsets = safe_spans * 0.05
        colour = QColor(self._fg_color)
        font = QFont("Helvetica", 13)

        for _name, axis_idx, neg_name, pos_name in _SLICE_ORIENTATIONS:
            if axis_idx == 0:
                neg_pos = [
                    float(mins[0] - offsets[0]),
                    float(centre[1]),
                    float(centre[2]),
                ]
                pos_pos = [
                    float(maxs[0] + offsets[0]),
                    float(centre[1]),
                    float(centre[2]),
                ]
            elif axis_idx == 1:
                neg_pos = [
                    float(centre[0]),
                    float(mins[1] - offsets[1]),
                    float(centre[2]),
                ]
                pos_pos = [
                    float(centre[0]),
                    float(maxs[1] + offsets[1]),
                    float(centre[2]),
                ]
            else:
                neg_pos = [
                    float(centre[0]),
                    float(centre[1]),
                    float(mins[2] - offsets[2]),
                ]
                pos_pos = [
                    float(centre[0]),
                    float(centre[1]),
                    float(maxs[2] + offsets[2]),
                ]
            for text, position in ((neg_name, neg_pos), (pos_name, pos_pos)):
                label = gl.GLTextItem()
                label.setData(text=text, pos=position, color=colour, font=font)
                self.view.addItem(label)
                self._axis_label_items.append(label)

    def _update_scene_bounds(self, mins: np.ndarray, maxs: np.ndarray) -> None:
        spans = np.maximum(maxs - mins, 1e-3)
        # Update the camera centre/distance so the current geometry fits snugly
        # inside the viewport regardless of the downsampling level.
        center = (mins + maxs) / 2.0
        self._current_bounds = (mins, maxs)
        self.view.opts["center"] = pg.Vector(center[0], center[1], center[2])
        self.view.opts["distance"] = float(np.linalg.norm(spans) * 1.2)
        self._update_axis_item()

    def _update_axis_item(self) -> None:
        if self._axis_item is not None:
            self.view.removeItem(self._axis_item)
            self._axis_item = None
        if not self.axes_checkbox.isChecked() or self._current_bounds is None:
            self._remove_axis_labels()
            return
        mins, maxs = self._current_bounds
        spans = np.maximum(maxs - mins, 1e-3)
        axis = _AdjustableAxisItem() if "_AdjustableAxisItem" in globals() else gl.GLAxisItem()
        axis.setSize(spans[0], spans[1], spans[2])
        axis.translate(float(mins[0]), float(mins[1]), float(mins[2]))
        thickness = getattr(self, "axis_thickness_slider", None)
        if thickness is not None and hasattr(axis, "setLineWidth"):
            axis.setLineWidth(float(thickness.value()))
        self.view.addItem(axis)
        self._axis_item = axis
        self._update_axis_labels()

    def _on_axes_toggle(self, checked: bool) -> None:
        slider = getattr(self, "axis_thickness_slider", None)
        if slider is not None:
            slider.setEnabled(checked)
        label_cb = getattr(self, "axis_labels_checkbox", None)
        if label_cb is not None:
            label_cb.setEnabled(checked)
            if not checked:
                self._remove_axis_labels()
        self._update_axis_item()

    def _on_axis_thickness_change(self, value: int) -> None:
        if hasattr(self, "axis_thickness_value"):
            self.axis_thickness_value.setText(f"{value} px")
        if self._axis_item is not None and hasattr(self._axis_item, "setLineWidth"):
            self._axis_item.setLineWidth(float(value))
            self.view.update()

    def _on_axis_labels_toggle(self, checked: bool) -> None:
        if not checked:
            self._remove_axis_labels()
        else:
            self._update_axis_labels()

    def _update_plot(self):
        if self._downsampled is None or self._normalised_volume is None:
            self.status_label.setText("No scalar volume available for rendering.")
            self._clear_colorbar()
            return

        thr = self.thresh_slider.value() / 100.0
        self.thresh_label.setText(f"{thr:.2f}")
        cmap = self.colormap_combo.currentText() or "viridis"
        alpha = self.opacity_slider.value() / 100.0
        mode = self.render_mode_combo.currentText()

        if mode == "Surface mesh":
            self._draw_surface_mesh(thr, cmap, alpha)
        else:
            self._draw_point_cloud(thr, cmap, alpha)

    def _handle_empty_point_cloud(self, shape: Sequence[int], scale: np.ndarray, thr: float) -> None:
        if self._scatter_item is not None:
            self.view.removeItem(self._scatter_item)
            self._scatter_item = None
        mins = np.zeros(3, dtype=np.float32)
        spans = np.maximum((np.asarray(shape, dtype=np.float32) - 1.0) * scale, 1e-3)
        maxs = mins + spans
        self._update_scene_bounds(mins, maxs)
        self._clear_colorbar()
        self.status_label.setText(
            f"No voxels above threshold {thr:.2f}. Lower the threshold to reveal data."
        )

    def _draw_point_cloud(self, thr: float, cmap_name: str, alpha: float) -> None:
        downsampled = self._downsampled
        if downsampled is None:
            self.status_label.setText("No data available for point-cloud rendering.")
            self._clear_colorbar()
            return

        if self._mesh_item is not None:
            # Switching from the iso-surface to point-cloud view discards the
            # cached ``GLMeshItem`` so only the scatter is redrawn.
            self.view.removeItem(self._mesh_item)
            self._mesh_item = None

        step = self._downsample_step
        scale = self._voxel_sizes_vec * float(step)
        shape = self._point_shape or downsampled.shape
        sorted_values = self._point_sorted_values
        sorted_indices = self._point_sorted_indices

        coords_mm: Optional[np.ndarray] = None
        values: Optional[np.ndarray] = None

        if (
            sorted_values is None
            or sorted_indices is None
            or self._point_shape is None
            or sorted_values.size == 0
        ):
            mask = downsampled >= thr
            coords = np.argwhere(mask)
            if coords.size == 0:
                self._handle_empty_point_cloud(shape, scale, thr)
                return
            coords_mm = coords.astype(np.float32, copy=False)
            coords_mm *= scale
            values = downsampled[mask].astype(np.float32, copy=False)
        else:
            start = int(np.searchsorted(sorted_values, thr, side="left"))
            if start >= sorted_values.size:
                self._handle_empty_point_cloud(shape, scale, thr)
                return

            selected_indices = sorted_indices[start:]
            selected_values = sorted_values[start:]
            if selected_indices.size > self._max_points:
                sample_idx = np.linspace(
                    0, selected_indices.size - 1, self._max_points, dtype=np.int64
                )
                selected_indices = selected_indices[sample_idx]
                selected_values = selected_values[sample_idx]

            if selected_indices.size == 0:
                self._handle_empty_point_cloud(shape, scale, thr)
                return

            if selected_indices.size > 1:
                order = np.argsort(selected_indices, kind="mergesort")
                selected_indices = selected_indices[order]
                selected_values = selected_values[order]

            coords_mm = self._indices_to_world(selected_indices, shape, scale)
            values = selected_values.astype(np.float32, copy=False)

        if coords_mm is None or values is None:
            self._handle_empty_point_cloud(shape, scale, thr)
            return

        coords_mm, values, slice_bounds = self._apply_slice_to_points(coords_mm, values)
        if coords_mm.size == 0:
            if slice_bounds is not None:
                if self._scatter_item is not None:
                    self.view.removeItem(self._scatter_item)
                    self._scatter_item = None
                self._clear_colorbar()
                mins, maxs = slice_bounds
                self._update_scene_bounds(mins, maxs)
                self.status_label.setText(
                    "Slice planes removed all voxels at threshold "
                    f"{thr:.2f}. Adjust the slicer sliders or disable slicing."
                )
                return
            self._handle_empty_point_cloud(shape, scale, thr)
            return

        if values is None:
            values = np.empty(coords_mm.shape[0], dtype=np.float32)
        cmap = self._get_adjusted_colormap(cmap_name)
        colors = self._map_colors(values, cmap, alpha)

        if self._scatter_item is None:
            # ``pxMode`` keeps the slider-controlled marker size in screen
            # pixels, mirroring the behaviour of the original matplotlib view.
            self._scatter_item = gl.GLScatterPlotItem(pxMode=True)
            self.view.addItem(self._scatter_item)
        self._scatter_item.setData(pos=coords_mm, color=colors, size=float(self.point_slider.value()))
        self._scatter_item.setGLOptions("translucent" if alpha < 0.999 else "opaque")

        mins = coords_mm.min(axis=0)
        maxs = coords_mm.max(axis=0)
        self._update_scene_bounds(mins, maxs)
        self._update_colorbar(cmap)

        total_voxels = self._estimate_voxels_above_threshold(thr)
        spin = getattr(self, "downsample_spin", None)
        downsample_source = "manual" if spin and spin.value() > 0 else "auto"
        lighting_suffix = ""
        if self._light_shader is not None or self._flat_shader is not None:
            lighting_suffix = " Lighting " + ("enabled." if self._lighting_enabled else "disabled.")
        self.status_label.setText(
            "Point cloud: "
            f"{coords_mm.shape[0]:,} voxels (threshold {thr:.2f}, opacity {alpha:.2f}, "
            f"point size {self.point_slider.value()}). "
            f"Downsample step {step} ({downsample_source}); "
            f"≈ total voxels ≥ threshold {total_voxels:,}.{lighting_suffix}"
            f"{self._slice_status_suffix()}"
        )

    def _draw_surface_mesh(self, thr: float, cmap_name: str, alpha: float) -> None:
        if not HAS_SKIMAGE:
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self._clear_colorbar()
            self.status_label.setText(
                "Surface rendering requires the optional 'scikit-image' dependency."
            )
            return

        volume = self._normalised_volume
        if volume is None:
            self.status_label.setText("No scalar volume available for rendering.")
            self._clear_colorbar()
            return

        if self._scatter_item is not None:
            self.view.removeItem(self._scatter_item)
            self._scatter_item = None

        step = self._downsample_step
        slices = tuple(slice(None, None, step) for _ in range(3))
        reduced = volume[slices]

        if min(reduced.shape) < 2:
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self._clear_colorbar()
            self.status_label.setText(
                "Volume is too small after downsampling to compute a surface mesh."
            )
            return

        vol_min = float(np.min(reduced))
        vol_max = float(np.max(reduced))
        if not (vol_min < thr < vol_max):
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self._clear_colorbar()
            self.status_label.setText(
                f"Iso level {thr:.2f} is outside the volume range [{vol_min:.2f}, {vol_max:.2f}]."
            )
            return

        try:
            verts, faces, _normals, values = sk_measure.marching_cubes(
                reduced,
                level=thr,
                step_size=max(1, self._surface_step),
                spacing=(
                    step * self._voxel_sizes[0],
                    step * self._voxel_sizes[1],
                    step * self._voxel_sizes[2],
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self._clear_colorbar()
            self.status_label.setText(f"Marching cubes failed: {exc}")
            return

        vertex_values = np.clip(values, 0.0, 1.0)
        verts, faces, vertex_values, slice_bounds = self._clip_mesh_to_slices(
            verts, faces, vertex_values
        )
        if verts.size == 0 or faces.size == 0:
            if slice_bounds is not None:
                if self._mesh_item is not None:
                    self.view.removeItem(self._mesh_item)
                    self._mesh_item = None
                self._clear_colorbar()
                mins, maxs = slice_bounds
                self._update_scene_bounds(mins, maxs)
                self.status_label.setText(
                    f"Slice planes removed all surface triangles at iso level {thr:.2f}. "
                    "Adjust the slicer sliders or disable slicing."
                )
                return
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self._clear_colorbar()
            self.status_label.setText(
                f"No closed surface found at iso level {thr:.2f}; adjust the slider."
            )
            return

        if vertex_values is None:
            vertex_values = np.zeros(verts.shape[0], dtype=np.float32)

        meshdata = gl.MeshData(vertexes=verts, faces=faces)
        cmap = self._get_adjusted_colormap(cmap_name)
        vertex_colors = self._map_colors(vertex_values, cmap, alpha)
        meshdata.setVertexColors(vertex_colors)

        if self._mesh_item is None:
            # ``GLMeshItem`` retains the uploaded vertex buffers so subsequent
            # slider tweaks only update colours instead of reallocating the mesh.
            self._mesh_item = gl.GLMeshItem(
                meshdata=meshdata,
                smooth=False,
                shader="shaded",
                drawEdges=False,
            )
            self.view.addItem(self._mesh_item)
        else:
            self._mesh_item.setMeshData(meshdata=meshdata)

        self._update_light_shader()
        self._mesh_item.setGLOptions("translucent" if alpha < 0.999 else "opaque")

        mins = np.min(verts, axis=0)
        maxs = np.max(verts, axis=0)
        self._update_scene_bounds(mins, maxs)
        self._update_colorbar(cmap, vmin=thr, vmax=1.0, label="Iso level (normalised)")

        total_voxels = int(np.count_nonzero(self._normalised_volume >= thr))
        spin = getattr(self, "downsample_spin", None)
        downsample_source = "manual" if spin and spin.value() > 0 else "auto"
        lighting_suffix = ""
        if self._light_shader is not None or self._flat_shader is not None:
            lighting_suffix = " Lighting " + ("enabled." if self._lighting_enabled else "disabled.")
        self.status_label.setText(
            "Surface mesh: "
            f"{verts.shape[0]:,} vertices / {faces.shape[0]:,} faces (iso {thr:.2f}, "
            f"opacity {alpha:.2f}). Downsample step {step} ({downsample_source}); "
            f"marching step {self._surface_step}. Total voxels ≥ level {total_voxels:,}.{lighting_suffix}"
            f"{self._slice_status_suffix()}"
        )

    def _on_render_mode_change(self, _mode: str) -> None:
        if self._initialising:
            return
        self._update_mode_dependent_controls()
        self._update_plot()

    def _update_mode_dependent_controls(self) -> None:
        mode = self.render_mode_combo.currentText()
        is_surface = mode == "Surface mesh"
        self.point_label.setVisible(not is_surface)
        self.point_slider.setVisible(not is_surface)
        self.point_slider.setEnabled(not is_surface)
        self.surface_step_label.setEnabled(is_surface)
        self.surface_step_spin.setEnabled(is_surface)
        if hasattr(self, "lighting_group"):
            self.lighting_group.setEnabled(
                is_surface
                and (self._light_shader is not None or self._flat_shader is not None)
            )
            self._update_light_controls_enabled()
        if is_surface:
            self.thresh_text.setText("Iso level:")
            self.thresh_slider.setToolTip(
                "Iso-surface level within the normalised volume (0–1)."
            )
        else:
            self.thresh_text.setText("Threshold:")
            self.thresh_slider.setToolTip(
                "Minimum normalised intensity included in the rendering."
            )

    def _on_opacity_change(self, value: int) -> None:
        self.opacity_label.setText(f"{value / 100.0:.2f}")
        if not self._initialising:
            self._update_plot()

    def _on_max_points_change(self, value: int) -> None:
        self._max_points = int(value)
        self._prepare_downsampled()
        if not self._initialising:
            self._update_plot()

    def _on_downsample_change(self, _value: int) -> None:
        self._prepare_downsampled()
        if not self._initialising:
            self._update_plot()

    def _on_surface_step_change(self, value: int) -> None:
        self._surface_step = max(1, int(value))
        if not self._initialising and self.render_mode_combo.currentText() == "Surface mesh":
            self._update_plot()


class Surface3DDialog(QDialog):
    """Interactive renderer for surface meshes from GIFTI or FreeSurfer."""

    def __init__(
        self,
        parent,
        vertices: np.ndarray,
        faces: np.ndarray,
        scalars: Optional[dict[str, np.ndarray]] = None,
        meta: Optional[dict[str, Any]] = None,
        title: Optional[str] = None,
        dark_theme: bool = False,
    ) -> None:
        super().__init__(parent)
        if not HAS_PYQTGRAPH:
            raise RuntimeError(
                "Surface rendering requires the optional 'pyqtgraph' dependency."
            )
        self._initialising = True
        self.setWindowTitle(title or "Surface Viewer")
        self.resize(1200, 900)
        self.setMinimumSize(780, 560)
        self.setSizeGripEnabled(True)

        verts = np.asarray(vertices, dtype=np.float32)
        faces_arr = np.asarray(faces, dtype=np.int32)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("Surface vertices must be an (N, 3) array")
        if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
            raise ValueError("Surface faces must be an (M, 3) array")
        if faces_arr.size and int(faces_arr.max(initial=-1)) >= verts.shape[0]:
            raise ValueError("Face indices exceed available vertices")

        self._vertices = verts
        self._faces = faces_arr
        self._meta = meta or {}
        self._dark_theme = bool(dark_theme)
        self._fg_color = "#f0f0f0" if self._dark_theme else "#202020"
        self._canvas_bg = "#202020" if self._dark_theme else "#ffffff"
        self._axis_label_items: list[gl.GLTextItem] = []
        self._data_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._slice_controls: dict[str, _SliceControl] = {}

        self._scalar_fields: dict[str, np.ndarray] = {}
        if scalars:
            for name, values in scalars.items():
                arr = np.asarray(values, dtype=np.float32).reshape(-1)
                if arr.shape[0] != verts.shape[0]:
                    continue
                safe_name = str(name).strip() or f"Field {len(self._scalar_fields) + 1}"
                self._scalar_fields[safe_name] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        self._mesh_item: Optional[gl.GLMeshItem] = None
        self._axis_item: Optional[gl.GLAxisItem] = None
        self._current_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._colormap_cache: dict[tuple[str, float], mcolors.Colormap] = {}
        self._light_shader = _create_directional_light_shader()
        self._flat_shader = _create_flat_color_shader()
        self._lighting_enabled = True

        if self._vertices.size:
            mins = np.min(self._vertices, axis=0).astype(np.float32, copy=False)
            maxs = np.max(self._vertices, axis=0).astype(np.float32, copy=False)
            self._data_bounds = (mins.copy(), maxs.copy())

        layout = QVBoxLayout(self)

        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(12)
        layout.addWidget(self._splitter)

        self._view_container = QWidget()
        view_layout = QVBoxLayout(self._view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)

        # ``GLViewWidget`` renders using OpenGL so panning/zooming the scene does
        # not require recomputing the mesh when interacting with the viewport.
        self.view = gl.GLViewWidget()
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setBackgroundColor(self._canvas_bg)
        self.view.opts["distance"] = 200
        self.view.opts["elevation"] = 20
        self.view.opts["azimuth"] = -60
        view_layout.addWidget(self.view)

        self._splitter.addWidget(self._view_container)

        settings_container = QWidget()
        settings_layout = QVBoxLayout(settings_container)
        settings_layout.setContentsMargins(6, 6, 6, 6)
        settings_layout.setSpacing(10)

        # Allow the combined colour bar + control pane to be docked on the
        # side or bottom of the viewport.
        placement_layout = QHBoxLayout()
        placement_layout.setSpacing(6)
        placement_layout.addWidget(QLabel("Panel placement:"))
        self.panel_location_combo = QComboBox()
        self.panel_location_combo.addItems(["Bottom", "Left", "Right"])
        placement_layout.addWidget(self.panel_location_combo)
        placement_layout.addStretch()
        settings_layout.addLayout(placement_layout)

        scalar_row = QHBoxLayout()
        scalar_row.setContentsMargins(0, 0, 0, 0)
        scalar_row.setSpacing(8)
        scalar_row.addWidget(QLabel("Scalar:"))
        self.scalar_combo = QComboBox()
        self.scalar_combo.addItem("Constant colour", userData=None)
        for name in sorted(self._scalar_fields):
            self.scalar_combo.addItem(name, userData=name)
        self.scalar_combo.setEnabled(bool(self._scalar_fields))
        scalar_row.addWidget(self.scalar_combo, 1)
        scalar_row.addStretch()
        settings_layout.addLayout(scalar_row)

        appearance_group = QGroupBox("Appearance")
        appearance_layout = QGridLayout(appearance_group)
        appearance_layout.setColumnStretch(1, 1)

        row = 0
        appearance_layout.addWidget(QLabel("Colormap:"), row, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "twilight",
            "cubehelix",
            "Spectral",
            "coolwarm",
            "YlGnBu",
            "Greys",
            "bone",
        ])
        self.colormap_combo.setToolTip(
            "Select the matplotlib colormap for vertex intensities.",
        )
        appearance_layout.addWidget(self.colormap_combo, row, 1)

        row += 1
        appearance_layout.addWidget(QLabel("Opacity:"), row, 0)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.setToolTip(
            "Global alpha applied to rendered surfaces.",
        )
        appearance_layout.addWidget(self.opacity_slider, row, 1)
        self.opacity_label = QLabel("0.80")
        appearance_layout.addWidget(self.opacity_label, row, 2)

        row += 1
        appearance_layout.addWidget(QLabel("Colour intensity:"), row, 0)
        self.color_intensity_slider = QSlider(Qt.Horizontal)
        self.color_intensity_slider.setRange(10, 200)
        self.color_intensity_slider.setValue(100)
        self.color_intensity_slider.setToolTip(
            "Scale factor applied to RGB values after colormap lookup (0.1–2.0×).",
        )
        appearance_layout.addWidget(self.color_intensity_slider, row, 1)
        self.color_intensity_label = QLabel("1.00×")
        appearance_layout.addWidget(self.color_intensity_label, row, 2)
        settings_layout.addWidget(appearance_group)

        # Keep the colour bar within the scroll area so it collapses together
        # with the rest of the surface controls.
        colorbar_group = QGroupBox("Colour bar")
        colorbar_layout = QVBoxLayout(colorbar_group)
        colorbar_layout.setContentsMargins(8, 6, 8, 6)
        colorbar_layout.setSpacing(4)
        self._colorbar_canvas = FigureCanvas(plt.Figure(figsize=(2.6, 1.4)))
        self._colorbar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._colorbar_canvas.setFixedHeight(180)
        self._colorbar_canvas.figure.patch.set_facecolor(self._canvas_bg)
        colorbar_layout.addWidget(self._colorbar_canvas)
        settings_layout.addWidget(colorbar_group)

        axes_group = QGroupBox("Axes")
        axes_layout = QGridLayout(axes_group)
        axes_layout.setColumnStretch(1, 1)

        row = 0
        self.axes_checkbox = QCheckBox("Show axes")
        self.axes_checkbox.setChecked(False)
        axes_layout.addWidget(self.axes_checkbox, row, 0, 1, 3)

        row += 1
        axes_layout.addWidget(QLabel("Axis thickness:"), row, 0)
        self.axis_thickness_slider = QSlider(Qt.Horizontal)
        self.axis_thickness_slider.setRange(1, 12)
        self.axis_thickness_slider.setValue(3)
        self.axis_thickness_slider.setToolTip(
            "Line width of the anatomical axes (pixels).",
        )
        self.axis_thickness_slider.setEnabled(False)
        axes_layout.addWidget(self.axis_thickness_slider, row, 1)
        self.axis_thickness_value = QLabel("3 px")
        axes_layout.addWidget(self.axis_thickness_value, row, 2)

        row += 1
        self.axis_labels_checkbox = QCheckBox("Show axis labels")
        self.axis_labels_checkbox.setChecked(False)
        self.axis_labels_checkbox.setEnabled(False)
        axes_layout.addWidget(self.axis_labels_checkbox, row, 0, 1, 3)
        settings_layout.addWidget(axes_group)

        slice_group = QGroupBox("Slice planes")
        slice_layout = QVBoxLayout(slice_group)
        slice_layout.setContentsMargins(8, 4, 8, 4)
        slice_layout.setSpacing(6)
        self._slice_controls.clear()
        for key, axis_idx, neg_name, pos_name in _SLICE_ORIENTATIONS:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(6)
            checkbox = QCheckBox(key.capitalize())
            checkbox.setChecked(False)
            row_layout.addWidget(checkbox)
            row_layout.addSpacing(4)
            row_layout.addWidget(QLabel(f"{neg_name}:"))
            min_slider = QSlider(Qt.Horizontal)
            min_slider.setRange(0, 100)
            min_slider.setValue(0)
            min_slider.setEnabled(False)
            row_layout.addWidget(min_slider, 1)
            min_value = QLabel("0.0 mm")
            row_layout.addWidget(min_value)
            row_layout.addSpacing(6)
            row_layout.addWidget(QLabel(f"{pos_name}:"))
            max_slider = QSlider(Qt.Horizontal)
            max_slider.setRange(0, 100)
            max_slider.setValue(100)
            max_slider.setEnabled(False)
            row_layout.addWidget(max_slider, 1)
            max_value = QLabel("0.0 mm")
            row_layout.addWidget(max_value)
            row_layout.addStretch()
            slice_layout.addLayout(row_layout)
            control = _SliceControl(
                checkbox=checkbox,
                min_slider=min_slider,
                max_slider=max_slider,
                min_value=min_value,
                max_value=max_value,
                axis=axis_idx,
                negative_name=neg_name,
                positive_name=pos_name,
            )
            self._slice_controls[key] = control
            checkbox.toggled.connect(lambda checked, name=key: self._on_slice_toggle(name, checked))
            min_slider.valueChanged.connect(
                lambda value, name=key: self._on_slice_slider_change(name, "min", value)
            )
            max_slider.valueChanged.connect(
                lambda value, name=key: self._on_slice_slider_change(name, "max", value)
            )
        slice_layout.addStretch()
        settings_layout.addWidget(slice_group)

        self.lighting_group = QGroupBox("Lighting")
        light_layout = QGridLayout(self.lighting_group)
        light_row = 0
        self.light_enable_checkbox = QCheckBox("Enable lighting")
        self.light_enable_checkbox.setChecked(True)
        light_layout.addWidget(self.light_enable_checkbox, light_row, 0, 1, 3)
        light_row += 1
        light_layout.addWidget(QLabel("Azimuth:"), light_row, 0)
        self.light_azimuth_slider = QSlider(Qt.Horizontal)
        self.light_azimuth_slider.setRange(-180, 180)
        self.light_azimuth_slider.setValue(-45)
        self.light_azimuth_slider.setToolTip(
            "Horizontal direction of the light source relative to the mesh (°).",
        )
        light_layout.addWidget(self.light_azimuth_slider, light_row, 1)
        self.light_azimuth_label = QLabel("-45°")
        light_layout.addWidget(self.light_azimuth_label, light_row, 2)

        light_row += 1
        light_layout.addWidget(QLabel("Elevation:"), light_row, 0)
        self.light_elevation_slider = QSlider(Qt.Horizontal)
        self.light_elevation_slider.setRange(-90, 90)
        self.light_elevation_slider.setValue(30)
        self.light_elevation_slider.setToolTip(
            "Vertical angle of the light source relative to the mesh (°).",
        )
        light_layout.addWidget(self.light_elevation_slider, light_row, 1)
        self.light_elevation_label = QLabel("30°")
        light_layout.addWidget(self.light_elevation_label, light_row, 2)

        light_row += 1
        light_layout.addWidget(QLabel("Intensity:"), light_row, 0)
        self.light_intensity_slider = QSlider(Qt.Horizontal)
        self.light_intensity_slider.setRange(10, 300)
        self.light_intensity_slider.setValue(130)
        self.light_intensity_slider.setToolTip(
            "Diffuse lighting strength (0.1–3.0×). Ambient light stays constant.",
        )
        light_layout.addWidget(self.light_intensity_slider, light_row, 1)
        self.light_intensity_label = QLabel("1.30×")
        light_layout.addWidget(self.light_intensity_label, light_row, 2)
        if self._light_shader is None and self._flat_shader is None:
            self.lighting_group.setEnabled(False)
        settings_layout.addWidget(self.lighting_group)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        settings_layout.addWidget(self.status_label)
        settings_layout.addStretch()

        self._panel_scroll = ShrinkableScrollArea()
        self._panel_scroll.setWidget(settings_container)
        self._panel_scroll.setWidgetResizable(True)
        self._panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._splitter.addWidget(self._panel_scroll)
        self._splitter.setStretchFactor(0, 4)
        self._splitter.setStretchFactor(1, 1)

        self.panel_location_combo.currentTextChanged.connect(self._on_panel_location_change)
        self.scalar_combo.currentIndexChanged.connect(self._on_scalar_change)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_change)
        self.opacity_slider.valueChanged.connect(self._on_opacity_change)
        self.color_intensity_slider.valueChanged.connect(self._on_color_intensity_change)
        self.axes_checkbox.toggled.connect(self._on_axes_toggle)
        self.axis_thickness_slider.valueChanged.connect(self._on_axis_thickness_change)
        self.axis_labels_checkbox.toggled.connect(self._on_axis_labels_toggle)
        self.light_enable_checkbox.toggled.connect(self._on_light_enabled_toggle)
        self.light_azimuth_slider.valueChanged.connect(self._on_light_setting_change)
        self.light_elevation_slider.valueChanged.connect(self._on_light_setting_change)
        self.light_intensity_slider.valueChanged.connect(self._on_light_setting_change)

        self._apply_panel_layout(self.panel_location_combo.currentText())
        self._update_light_controls_enabled()

        # Prepare labels and shader uniforms before the first draw call.
        self._on_axis_thickness_change(self.axis_thickness_slider.value())
        self._update_slice_labels()
        self._on_color_intensity_change(self.color_intensity_slider.value())
        self._update_light_labels()
        self._update_light_shader()

        self._initialising = False
        self._update_plot()

    def _apply_panel_layout(self, location: str) -> None:
        """Reorder the settings splitter when users move the control pane."""

        if not hasattr(self, "_splitter"):
            return

        normalised = (location or "bottom").strip().lower()
        if normalised not in {"bottom", "left", "right"}:
            normalised = "bottom"

        view_first = True
        if normalised == "bottom":
            self._splitter.setOrientation(Qt.Vertical)
            view_first = True
        else:
            self._splitter.setOrientation(Qt.Horizontal)
            view_first = normalised != "left"

        widgets = [self._view_container, self._panel_scroll]
        if not view_first:
            widgets.reverse()
        for index, widget in enumerate(widgets):
            if self._splitter.indexOf(widget) != index:
                self._splitter.insertWidget(index, widget)

        if view_first:
            self._splitter.setStretchFactor(0, 4)
            self._splitter.setStretchFactor(1, 1)
        else:
            self._splitter.setStretchFactor(0, 1)
            self._splitter.setStretchFactor(1, 4)

    def _on_panel_location_change(self, location: str) -> None:
        self._apply_panel_layout(location)

    def _update_light_controls_enabled(self) -> None:
        enabled = bool(self.lighting_group.isEnabled() and self._lighting_enabled)
        for widget in (
            self.light_azimuth_slider,
            self.light_elevation_slider,
            self.light_intensity_slider,
        ):
            widget.setEnabled(enabled)
        for label in (
            self.light_azimuth_label,
            self.light_elevation_label,
            self.light_intensity_label,
        ):
            label.setEnabled(enabled)

    def _apply_mesh_shader(self) -> None:
        if self._mesh_item is None:
            return
        if self._light_shader is not None and self._lighting_enabled:
            self._mesh_item.setShader(self._light_shader)
        elif self._flat_shader is not None:
            self._mesh_item.setShader(self._flat_shader)
        else:
            self._mesh_item.setShader("shaded")

    def _on_light_enabled_toggle(self, checked: bool) -> None:
        self._lighting_enabled = bool(checked)
        self._update_light_controls_enabled()
        if self._lighting_enabled:
            self._update_light_shader()
        else:
            self._apply_mesh_shader()
        if not self._initialising and self._mesh_item is not None:
            self._mesh_item.update()
            self.view.update()

    def _current_color_intensity(self) -> float:
        return max(0.1, self.color_intensity_slider.value() / 100.0)

    def _get_adjusted_colormap(self, cmap_name: str) -> mcolors.Colormap:
        intensity = round(self._current_color_intensity(), 3)
        key = (cmap_name, intensity)
        cached = self._colormap_cache.get(key)
        if cached is not None:
            return cached
        base = plt.get_cmap(cmap_name)
        samples = base(np.linspace(0.0, 1.0, 512))
        samples[:, :3] = np.clip(samples[:, :3] * intensity, 0.0, 1.0)
        cmap = mcolors.ListedColormap(samples, name=f"{cmap_name}_x{intensity:.3f}")
        self._colormap_cache[key] = cmap
        return cmap

    def _map_colors(
        self, values: np.ndarray, cmap: mcolors.Colormap, alpha: float
    ) -> np.ndarray:
        # ``values`` are pre-normalised to [0, 1] so we can reuse matplotlib
        # colour maps and simply blend in the requested opacity.
        colours = np.asarray(cmap(np.clip(values, 0.0, 1.0)), dtype=np.float32)
        colours[:, 3] = np.clip(colours[:, 3] * alpha, 0.0, 1.0)
        return colours

    def _update_slice_labels(self) -> None:
        if not self._slice_controls:
            return
        bounds = self._data_bounds
        has_bounds = bounds is not None
        mins: Optional[np.ndarray]
        maxs: Optional[np.ndarray]
        if has_bounds:
            mins, maxs = bounds  # type: ignore[assignment]
        else:
            mins = maxs = None
        for control in self._slice_controls.values():
            slider_enabled = bool(has_bounds and control.checkbox.isChecked())
            control.min_slider.setEnabled(slider_enabled)
            control.max_slider.setEnabled(slider_enabled)
            if mins is None or maxs is None:
                control.min_value.setText("--")
                control.max_value.setText("--")
                continue
            axis = control.axis
            axis_min = float(mins[axis])
            axis_max = float(maxs[axis])
            span = axis_max - axis_min
            min_frac = control.min_slider.value() / 100.0
            max_frac = control.max_slider.value() / 100.0
            min_mm = axis_min + span * min_frac
            max_mm = axis_min + span * max_frac
            control.min_value.setText(f"{min_mm:.1f} mm")
            control.max_value.setText(f"{max_mm:.1f} mm")

    def _active_slice_bounds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if not self._slice_controls or self._data_bounds is None:
            return None
        mins, maxs = self._data_bounds
        min_bounds = np.asarray(mins, dtype=np.float32).copy()
        max_bounds = np.asarray(maxs, dtype=np.float32).copy()
        active = False
        for control in self._slice_controls.values():
            if not control.checkbox.isChecked():
                continue
            axis = control.axis
            axis_min = float(mins[axis])
            axis_max = float(maxs[axis])
            span = axis_max - axis_min
            min_frac = control.min_slider.value() / 100.0
            max_frac = control.max_slider.value() / 100.0
            min_bounds[axis] = float(axis_min + span * min_frac)
            max_bounds[axis] = float(axis_min + span * max_frac)
            active = True
        if not active:
            return None
        return min_bounds, max_bounds

    def _slice_status_suffix(self) -> str:
        bounds = self._active_slice_bounds()
        if bounds is None:
            return ""
        mins, maxs = bounds
        parts: list[str] = []
        for control in self._slice_controls.values():
            if not control.checkbox.isChecked():
                continue
            axis = control.axis
            parts.append(
                f"{control.negative_name}–{control.positive_name} "
                f"{mins[axis]:.1f}–{maxs[axis]:.1f} mm"
            )
        if not parts:
            return ""
        return " Slice windows: " + ", ".join(parts) + "."

    def _mask_coords(
        self, coords: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        mins, maxs = bounds
        if coords.size == 0:
            return np.zeros(0, dtype=bool)
        mask = np.ones(coords.shape[0], dtype=bool)
        for axis in range(3):
            mask &= (coords[:, axis] >= mins[axis]) & (coords[:, axis] <= maxs[axis])
        return mask

    def _apply_slice_to_points(
        self, coords: np.ndarray, values: Optional[np.ndarray]
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[tuple[np.ndarray, np.ndarray]]]:
        bounds = self._active_slice_bounds()
        if bounds is None or coords.size == 0:
            return coords, values, None
        mask = self._mask_coords(coords, bounds)
        if not np.any(mask):
            empty_coords = coords[:0].reshape(0, 3)
            empty_vals = values[:0] if values is not None else None
            return empty_coords, empty_vals, bounds
        filtered_coords = coords[mask]
        filtered_values = values[mask] if values is not None else None
        return filtered_coords, filtered_values, bounds

    def _clip_mesh_to_slices(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        values: Optional[np.ndarray],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[tuple[np.ndarray, np.ndarray]],
    ]:
        bounds = self._active_slice_bounds()
        if bounds is None or verts.size == 0:
            return verts, faces, values, None
        mask = self._mask_coords(verts, bounds)
        if not np.any(mask):
            empty_verts = verts[:0].reshape(0, 3)
            empty_faces = faces[:0].reshape(0, 3)
            empty_vals = values[:0] if values is not None else None
            return empty_verts, empty_faces, empty_vals, bounds
        selected = np.where(mask)[0]
        index_map = -np.ones(mask.shape[0], dtype=np.int64)
        index_map[selected] = np.arange(selected.size, dtype=np.int64)
        mapped_faces = index_map[faces]
        valid = (mapped_faces >= 0).all(axis=1)
        mapped_faces = mapped_faces[valid]
        if mapped_faces.size == 0:
            empty_verts = verts[:0].reshape(0, 3)
            empty_faces = faces[:0].reshape(0, 3)
            empty_vals = values[:0] if values is not None else None
            return empty_verts, empty_faces, empty_vals, bounds
        new_verts = verts[selected]
        new_values = values[selected] if values is not None else None
        return new_verts, mapped_faces.astype(np.int32, copy=False), new_values, bounds

    def _on_slice_toggle(self, name: str, _checked: bool) -> None:
        if name not in self._slice_controls:
            return
        self._update_slice_labels()
        if not self._initialising:
            self._update_plot()

    def _on_slice_slider_change(self, name: str, which: str, value: int) -> None:
        control = self._slice_controls.get(name)
        if control is None:
            return
        if which == "min":
            other = control.max_slider
            if value > other.value():
                other.blockSignals(True)
                other.setValue(value)
                other.blockSignals(False)
        else:
            other = control.min_slider
            if value < other.value():
                other.blockSignals(True)
                other.setValue(value)
                other.blockSignals(False)
        self._update_slice_labels()
        if not self._initialising and control.checkbox.isChecked():
            self._update_plot()
    def _clear_colorbar(self) -> None:
        fig = self._colorbar_canvas.figure
        fig.clf()
        fig.patch.set_facecolor(self._canvas_bg)
        self._colorbar_canvas.draw_idle()

    def _update_colorbar(
        self, cmap: mcolors.Colormap, vmin: float, vmax: float, label: str = "Scalar value"
    ) -> None:
        fig = self._colorbar_canvas.figure
        fig.clf()
        # Render the colour bar horizontally so it mirrors the slice legend above
        # while preserving vertical space for the rest of the controls.
        ax = fig.add_axes([0.12, 0.35, 0.76, 0.32])
        norm = plt.Normalize(vmin, vmax)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(mappable, cax=ax, orientation="horizontal")
        colorbar.set_label(label, color=self._fg_color, labelpad=6)
        colorbar.ax.xaxis.set_ticks_position("bottom")
        colorbar.ax.xaxis.set_label_position("bottom")
        colorbar.ax.tick_params(colors=self._fg_color, which="both")
        for spine in colorbar.ax.spines.values():
            spine.set_color(self._fg_color)
        colorbar.ax.set_facecolor(self._canvas_bg)
        fig.patch.set_facecolor(self._canvas_bg)
        self._colorbar_canvas.draw_idle()

    def _on_color_intensity_change(self, value: int) -> None:
        self.color_intensity_label.setText(f"{value / 100.0:.2f}×")
        if not self._initialising:
            self._update_plot()

    def _update_light_labels(self) -> None:
        self.light_azimuth_label.setText(f"{self.light_azimuth_slider.value():+d}°")
        self.light_elevation_label.setText(
            f"{self.light_elevation_slider.value():+d}°"
        )
        self.light_intensity_label.setText(
            f"{self.light_intensity_slider.value() / 100.0:.2f}×"
        )

    def _update_light_shader(self) -> None:
        shader = self._light_shader
        if shader is None or not self._lighting_enabled:
            self._apply_mesh_shader()
            return
        azimuth = math.radians(self.light_azimuth_slider.value())
        elevation = math.radians(self.light_elevation_slider.value())
        cos_el = math.cos(elevation)
        direction = [
            float(cos_el * math.cos(azimuth)),
            float(cos_el * math.sin(azimuth)),
            float(math.sin(elevation)),
        ]
        shader["lightDir"] = direction
        shader["lightParams"] = [
            max(0.0, self.light_intensity_slider.value() / 100.0),
            0.35,
        ]
        self._apply_mesh_shader()

    def _on_light_setting_change(self, _value: int) -> None:
        self._update_light_labels()
        self._update_light_shader()
        if self._initialising:
            return
        if self._mesh_item is not None:
            self._mesh_item.update()
            self.view.update()

    def _update_scene_bounds(self, mins: np.ndarray, maxs: np.ndarray) -> None:
        spans = np.maximum(maxs - mins, 1e-3)
        # Keep the camera centred on the mesh to avoid jarring jumps when users
        # switch between scalar fields or adjust opacity.
        center = (mins + maxs) / 2.0
        self._current_bounds = (mins, maxs)
        self.view.opts["center"] = pg.Vector(center[0], center[1], center[2])
        self.view.opts["distance"] = float(np.linalg.norm(spans) * 1.2)
        self._update_axis_item()

    def _on_scalar_change(self, _index: int) -> None:
        self._update_plot()

    def _on_colormap_change(self, _name: str) -> None:
        self._update_plot()

    def _on_opacity_change(self, value: int) -> None:
        self.opacity_label.setText(f"{value / 100.0:.2f}")
        self._update_plot()

    def _update_plot(self) -> None:
        self._clear_colorbar()
        if self._faces.size == 0 or self._vertices.size == 0:
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self.status_label.setText("Surface mesh has no vertices or faces to render.")
            return

        alpha = self.opacity_slider.value() / 100.0
        cmap_name = self.colormap_combo.currentText() or "viridis"
        scalar_key = self.scalar_combo.currentData()
        cmap = self._get_adjusted_colormap(cmap_name)

        summary = (
            f"Surface mesh: {self._vertices.shape[0]:,} vertices / {self._faces.shape[0]:,} faces. "
        )

        normalised_values: Optional[np.ndarray] = None
        vmin: Optional[float] = None
        vmax: Optional[float] = None
        if scalar_key and scalar_key in self._scalar_fields:
            raw_values = self._scalar_fields[scalar_key]
            vmin = float(np.min(raw_values))
            vmax = float(np.max(raw_values))
            if math.isclose(vmin, vmax):
                normalised_values = np.zeros_like(raw_values, dtype=np.float32)
                vmin, vmax = vmin - 0.5, vmax + 0.5
            else:
                norm = (raw_values - vmin) / (vmax - vmin)
                normalised_values = np.clip(norm.astype(np.float32, copy=False), 0.0, 1.0)

        clipped_verts, clipped_faces, clipped_values, slice_bounds = self._clip_mesh_to_slices(
            self._vertices, self._faces, normalised_values
        )

        if clipped_verts.size == 0 or clipped_faces.size == 0:
            if slice_bounds is not None:
                if self._mesh_item is not None:
                    self.view.removeItem(self._mesh_item)
                    self._mesh_item = None
                self._clear_colorbar()
                mins, maxs = slice_bounds
                self._update_scene_bounds(mins, maxs)
                self.status_label.setText(
                    "Slice planes removed all surface triangles. "
                    "Adjust the slicer sliders or disable slicing."
                )
                return
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self.status_label.setText("Surface mesh has no vertices or faces to render.")
            return

        meshdata = gl.MeshData(vertexes=clipped_verts, faces=clipped_faces)
        if clipped_values is not None and scalar_key and scalar_key in self._scalar_fields:
            colours = self._map_colors(clipped_values, cmap, alpha)
            meshdata.setVertexColors(colours)
            assert vmin is not None and vmax is not None
            self._update_colorbar(cmap, vmin, vmax)
            summary += f"Scalar '{scalar_key}' range {vmin:.4g} – {vmax:.4g}. "
        else:
            base_color = cmap(0.6)
            colour = np.array(
                [[base_color[0], base_color[1], base_color[2], base_color[3] * alpha]],
                dtype=np.float32,
            )
            meshdata.setVertexColors(np.repeat(colour, clipped_verts.shape[0], axis=0))
            self._update_colorbar(cmap, 0.0, 1.0, label="Colormap preview")
            summary += "Constant colouring applied. "

        if self._mesh_item is None:
            self._mesh_item = gl.GLMeshItem(
                meshdata=meshdata,
                smooth=False,
                shader="shaded",
                drawEdges=False,
            )
            self.view.addItem(self._mesh_item)
        else:
            self._mesh_item.setMeshData(meshdata=meshdata)
        self._update_light_shader()
        self._mesh_item.setGLOptions("translucent" if alpha < 0.999 else "opaque")

        mins = np.min(clipped_verts, axis=0)
        maxs = np.max(clipped_verts, axis=0)
        self._update_scene_bounds(mins, maxs)

        summary += f"Opacity {alpha:.2f}."
        if self._light_shader is not None or self._flat_shader is not None:
            summary += " Lighting " + ("enabled." if self._lighting_enabled else "disabled.")
        if self._scalar_fields and (not scalar_key or scalar_key not in self._scalar_fields):
            summary += " Select a scalar field to colour the surface."
        summary += self._slice_status_suffix()
        self.status_label.setText(summary)


class FreeSurferSurfaceDialog(QDialog):
    """Simplified viewer tailored for FreeSurfer ``*.pial``/``*.white`` meshes."""

    def __init__(
        self,
        parent,
        vertices: np.ndarray,
        faces: np.ndarray,
        title: Optional[str] = None,
        dark_theme: bool = False,
    ) -> None:
        super().__init__(parent)
        if not HAS_PYQTGRAPH:
            raise RuntimeError(
                "Surface rendering requires the optional 'pyqtgraph' dependency."
            )

        self.setWindowTitle(title or "FreeSurfer Surface")
        self.resize(900, 720)
        self.setMinimumSize(620, 480)
        self.setSizeGripEnabled(True)

        verts = np.asarray(vertices, dtype=np.float32)
        faces_arr = np.asarray(faces, dtype=np.int32)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("Surface vertices must be an (N, 3) array")
        if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
            raise ValueError("Surface faces must be an (M, 3) array")
        if faces_arr.size and int(faces_arr.max(initial=-1)) >= verts.shape[0]:
            raise ValueError("Face indices exceed available vertices")

        self._vertices = verts
        self._faces = faces_arr
        self._fg_color = "#f0f0f0" if dark_theme else "#202020"
        self._canvas_bg = "#202020" if dark_theme else "#ffffff"

        self._mesh_item: Optional[gl.GLMeshItem] = None
        self._axis_item: Optional[gl.GLAxisItem] = None
        self._current_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._light_shader = _create_directional_light_shader()
        self._flat_shader = _create_flat_color_shader()
        self._lighting_enabled = self._light_shader is not None

        layout = QVBoxLayout(self)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(self._canvas_bg)
        self.view.opts["distance"] = 200
        self.view.opts["elevation"] = 20
        self.view.opts["azimuth"] = -60
        layout.addWidget(self.view, 1)

        controls = QGroupBox("Display settings")
        controls_layout = QGridLayout(controls)
        controls_layout.setColumnStretch(1, 1)

        # Provide a handful of constant colours so users can quickly switch the
        # appearance without diving into advanced scalar options.
        self._colour_presets: list[tuple[str, str]] = [
            ("Clay", "#d87c5a"),
            ("Slate", "#4f6d7a"),
            ("Bone", "#e9d8a6"),
            ("Forest", "#2a9d8f"),
            ("Carbon", "#5c5f66"),
        ]

        row = 0
        controls_layout.addWidget(QLabel("Surface colour:"), row, 0)
        self.color_combo = QComboBox()
        for name, colour in self._colour_presets:
            self.color_combo.addItem(name, userData=colour)
        controls_layout.addWidget(self.color_combo, row, 1, 1, 2)

        row += 1
        controls_layout.addWidget(QLabel("Opacity:"), row, 0)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(85)
        controls_layout.addWidget(self.opacity_slider, row, 1)
        self.opacity_label = QLabel("0.85")
        controls_layout.addWidget(self.opacity_label, row, 2)

        row += 1
        self.axes_checkbox = QCheckBox("Show axes")
        controls_layout.addWidget(self.axes_checkbox, row, 0, 1, 3)

        row += 1
        self.light_checkbox = QCheckBox("Enable lighting")
        self.light_checkbox.setChecked(self._lighting_enabled)
        controls_layout.addWidget(self.light_checkbox, row, 0, 1, 3)
        if self._light_shader is None and self._flat_shader is None:
            self._lighting_enabled = False
            self.light_checkbox.setEnabled(False)
            self.light_checkbox.setChecked(False)

        layout.addWidget(controls)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._initialising = True

        self.color_combo.currentIndexChanged.connect(self._on_color_change)
        self.opacity_slider.valueChanged.connect(self._on_opacity_change)
        self.axes_checkbox.toggled.connect(self._on_axes_toggle)
        self.light_checkbox.toggled.connect(self._on_light_toggle)

        if self._vertices.size:
            mins = np.min(self._vertices, axis=0).astype(np.float32, copy=False)
            maxs = np.max(self._vertices, axis=0).astype(np.float32, copy=False)
        else:
            mins = np.zeros(3, dtype=np.float32)
            maxs = np.zeros(3, dtype=np.float32)
        self._update_scene_bounds(mins, maxs)
        self._update_axis_item()
        self._update_light_shader()
        self._update_mesh()

        self._initialising = False
        self._update_status()

    def _current_colour_rgba(self) -> np.ndarray:
        colour_hex = self.color_combo.currentData()
        if not colour_hex:
            colour_hex = "#d87c5a"
        rgb = np.array(mcolors.to_rgb(str(colour_hex)), dtype=np.float32)
        alpha = np.clip(self.opacity_slider.value() / 100.0, 0.1, 1.0)
        colours = np.empty((self._vertices.shape[0], 4), dtype=np.float32)
        colours[:, :3] = rgb
        colours[:, 3] = alpha
        return colours

    def _update_mesh(self) -> None:
        if self._vertices.size == 0 or self._faces.size == 0:
            if self._mesh_item is not None:
                self.view.removeItem(self._mesh_item)
                self._mesh_item = None
            self.status_label.setText("Surface mesh has no vertices or faces to render.")
            return

        meshdata = gl.MeshData(vertexes=self._vertices, faces=self._faces)
        colours = self._current_colour_rgba()
        meshdata.setVertexColors(colours)

        if self._mesh_item is None:
            self._mesh_item = gl.GLMeshItem(
                meshdata=meshdata,
                smooth=False,
                shader="shaded",
                drawEdges=False,
            )
            self.view.addItem(self._mesh_item)
        else:
            self._mesh_item.setMeshData(meshdata=meshdata)

        self._apply_mesh_shader()
        alpha = self.opacity_slider.value() / 100.0
        self._mesh_item.setGLOptions("translucent" if alpha < 0.999 else "opaque")
        if not self._initialising:
            self.view.update()

    def _update_status(self) -> None:
        if self._vertices.size == 0 or self._faces.size == 0:
            summary = "Surface mesh has no vertices or faces to render."
        else:
            mins, maxs = self._current_bounds if self._current_bounds else (
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            )
            summary = (
                f"Surface mesh: {self._vertices.shape[0]:,} vertices / "
                f"{self._faces.shape[0]:,} faces. Opacity {self.opacity_slider.value() / 100.0:.2f}. "
                f"Colour preset: {self.color_combo.currentText()}. "
                f"Bounds X {mins[0]:.1f}–{maxs[0]:.1f} mm, "
                f"Y {mins[1]:.1f}–{maxs[1]:.1f} mm, Z {mins[2]:.1f}–{maxs[2]:.1f} mm. "
            )
        if self._light_shader is None and self._flat_shader is None:
            summary += " Lighting controls unavailable."
        else:
            summary += " Lighting " + ("enabled." if self._lighting_enabled else "disabled.")
        self.status_label.setText(summary)

    def _apply_mesh_shader(self) -> None:
        if self._mesh_item is None:
            return
        if self._light_shader is not None and self._lighting_enabled:
            self._mesh_item.setShader(self._light_shader)
        elif self._flat_shader is not None:
            self._mesh_item.setShader(self._flat_shader)
        else:
            self._mesh_item.setShader("shaded")

    def _update_light_shader(self) -> None:
        if self._light_shader is None:
            return
        self._light_shader["lightDir"] = [0.5, 0.3, 0.8]
        self._light_shader["lightParams"] = [1.0, 0.35]
        self._apply_mesh_shader()

    def _update_scene_bounds(self, mins: np.ndarray, maxs: np.ndarray) -> None:
        spans = np.maximum(maxs - mins, 1e-3)
        center = (mins + maxs) / 2.0
        self._current_bounds = (mins, maxs)
        self.view.opts["center"] = pg.Vector(center[0], center[1], center[2])
        self.view.opts["distance"] = float(np.linalg.norm(spans) * 1.2)

    def _update_axis_item(self) -> None:
        if not self.axes_checkbox.isChecked():
            if self._axis_item is not None:
                self.view.removeItem(self._axis_item)
                self._axis_item = None
            return
        if self._axis_item is None:
            axis = _AdjustableAxisItem() if '_AdjustableAxisItem' in globals() else gl.GLAxisItem()
            axis.setSize(1, 1, 1)
            self.view.addItem(axis)
            self._axis_item = axis
        mins, maxs = self._current_bounds if self._current_bounds else (
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        )
        spans = np.maximum(maxs - mins, 1e-3)
        self._axis_item.setSize(spans[0], spans[1], spans[2])
        if hasattr(self._axis_item, "setLineWidth"):
            self._axis_item.setLineWidth(2.0)

    def _on_color_change(self, _index: int) -> None:
        self._update_mesh()
        if not self._initialising:
            self._update_status()

    def _on_opacity_change(self, value: int) -> None:
        self.opacity_label.setText(f"{value / 100.0:.2f}")
        self._update_mesh()
        if not self._initialising:
            self._update_status()

    def _on_axes_toggle(self, _checked: bool) -> None:
        self._update_axis_item()
        if not self._initialising:
            self.view.update()

    def _on_light_toggle(self, checked: bool) -> None:
        self._lighting_enabled = bool(checked)
        self._apply_mesh_shader()
        if not self._initialising and self._mesh_item is not None:
            self._mesh_item.update()
        self._update_status()


class MetadataViewer(QWidget):
    """
    Metadata viewer/editor for JSON and TSV sidecars (from bids_editor_ancpbids).
    """
    def __init__(self):
        super().__init__()
        # Layout consists of a small toolbar followed by the actual viewer
        # widget.  A welcome message is shown when no file is loaded.
        vlay = QVBoxLayout(self)
        self.welcome = QLabel(
            "<h3>Metadata BIDSualizer</h3><br>Load data via File → Open or select a file to begin editing."
        )
        self.welcome.setAlignment(Qt.AlignCenter)
        vlay.addWidget(self.welcome)
        self.toolbar = QHBoxLayout()
        vlay.addLayout(self.toolbar)
        self.value_row = QHBoxLayout()
        vlay.addLayout(self.value_row)
        # Label used to show a spinner while a file is being loaded. It is
        # created as an overlay so it always appears above whatever viewer is
        # currently shown.  The font is enlarged for better visibility.
        self.loading_label = QLabel(self)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet(
            "font-size: 24px; background-color: rgba(0, 0, 0, 128);"
            "color: white;"
        )
        self.loading_label.hide()

        # Timer and frame sequence driving the spinner animation.
        self._load_timer = QTimer()
        self._load_timer.timeout.connect(self._spin_loading)
        self._load_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self._load_index = 0  # Current index into ``_load_frames``
        self._load_message = ""  # Message displayed next to the spinner
        self.viewer = None
        self.current_path = None
        self.data = None  # holds loaded NIfTI data when viewing images
        self.surface_data: Optional[dict[str, Any]] = None

    def clear(self):
        """Clear the toolbar and viewer when switching files."""
        def _clear_layout(lay):
            while lay.count():
                item = lay.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    _clear_layout(item.layout())
            lay.deleteLater()

        while self.toolbar.count():
            item = self.toolbar.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                _clear_layout(item.layout())
        while self.value_row.count():
            item = self.value_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                _clear_layout(item.layout())
        if self.viewer:
            self.layout().removeWidget(self.viewer)
            self.viewer.deleteLater()
            self.viewer = None
        self.data = None
        self.surface_data = None
        self.nifti_img = None
        self._nifti_meta = {}
        self.loading_label.hide()
        self._load_timer.stop()
        self.welcome.show()

    def _is_dark_theme(self) -> bool:
        """Detect whether the current palette is dark or light."""
        color = self.palette().color(QPalette.Window)
        brightness = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
        return brightness < 128

    def _start_loading(self, message: str) -> None:
        """Begin showing the loading spinner with ``message``."""
        self._load_message = message
        self._load_index = 0
        self.loading_label.setText(f"{message} {self._load_frames[0]}")
        self.loading_label.setGeometry(0, 0, self.width(), self.height())
        self.loading_label.raise_()
        self.loading_label.show()
        self._load_timer.start(100)

    def _spin_loading(self) -> None:
        """Advance the spinner animation by one frame."""
        if not self.loading_label.isVisible():
            return
        self._load_index = (self._load_index + 1) % len(self._load_frames)
        self.loading_label.setText(
            f"{self._load_message} {self._load_frames[self._load_index]}"
        )

    def _stop_loading(self) -> None:
        """Hide the loading spinner and stop the timer."""
        self._load_timer.stop()
        self.loading_label.hide()

    def _get_nifti_data(self, img):
        """Load a NIfTI image into an ``ndarray`` while handling RGB/struct dtypes.

        Some derived NIfTI maps (for example colour fractional anisotropy
        volumes) store multi-component voxels in a structured ``void`` dtype.
        ``nibabel`` cannot promote those to floats, which previously caused the
        viewer to crash.  We detect this situation, convert the array into a
        plain ``float`` ndarray and record metadata describing the additional
        vector axis so the rest of the viewer can render it correctly.
        """

        dtype_exc = None
        try:
            data = img.get_fdata()
            return data, {}
        except Exception as exc:
            # Only handle the structured-dtype failure; re-raise other errors so
            # they surface during debugging instead of being silently masked.
            is_dtype_error = False
            if np_exceptions is not None and isinstance(exc, np_exceptions.DTypePromotionError):
                is_dtype_error = True
            elif exc.__class__.__name__ == "DTypePromotionError":
                is_dtype_error = True
            elif "VoidDType" in str(exc):
                is_dtype_error = True
            if not is_dtype_error:
                raise
            dtype_exc = exc

        if rfn is None:
            raise RuntimeError(
                "Structured NIfTI data requires NumPy's recfunctions module to "
                "convert it into a regular array."
            ) from dtype_exc

        dataobj = np.asanyarray(img.dataobj)
        if not getattr(dataobj.dtype, "fields", None):
            # Unexpected dtype: ``get_fdata`` failed but the array is not
            # structured.  Re-raise the original exception so the caller knows.
            raise dtype_exc

        # ``structured_to_unstructured`` flattens the fields into the last axis
        # and keeps the voxel geometry intact, giving us a standard ndarray.
        unstructured = rfn.structured_to_unstructured(dataobj)
        vector_length = (
            int(unstructured.shape[-1])
            if unstructured.ndim == dataobj.ndim + 1
            else 1
        )
        meta = {
            "vector_axis": len(img.shape),
            "vector_length": vector_length,
            # Treat 3/4-component vectors as colour channels so they can be
            # rendered directly instead of exposing them as separate volumes.
            "is_rgb": vector_length in (3, 4)
            and unstructured.ndim == dataobj.ndim + 1,
        }
        # ``structured_to_unstructured`` already yields a float array when the
        # original data was floating point.  Cast explicitly to float32 so we do
        # not unnecessarily inflate the array when the original values were
        # stored as 32-bit floats.
        return unstructured.astype(np.float32, copy=False), meta

    def load_file(self, path: Path):
        """Load JSON, TSV, NIfTI or DICOM file into an editable viewer.

        The file is read in a background thread to keep the UI responsive and
        animate the loading spinner while potentially large datasets are processed.
        """

        self.current_path = path
        self.clear()
        self.welcome.hide()
        ext = _get_ext(path)
        lower_name = path.name.lower()
        gifti_candidate = any(lower_name.endswith(suffix) for suffix in GIFTI_SURFACE_SUFFIXES)
        freesurfer_candidate = any(lower_name.endswith(suffix) for suffix in FREESURFER_SURFACE_SUFFIXES)
        dicom = is_dicom_file(str(path))
        self._start_loading("Loading")
        # ``worker`` reads the file in a separate thread so the UI can keep
        # updating the spinner while potentially large data is loaded.
        result = {}
        load_error: Optional[Exception] = None

        def worker():
            nonlocal load_error
            try:
                if ext == '.json':
                    result['data'] = json.loads(path.read_text(encoding='utf-8'))
                elif ext == '.tsv':
                    result['df'] = pd.read_csv(path, sep='\t', keep_default_na=False)
                elif ext in ['.nii', '.nii.gz']:
                    img = nib.load(str(path))
                    result['img'] = img
                    data, meta = self._get_nifti_data(img)
                    result['data'] = data
                    result['nifti_meta'] = meta
                elif gifti_candidate:
                    surface = self._load_gifti_surface(path)
                    result['surface'] = surface
                    result['surface_type'] = 'gifti'
                elif freesurfer_candidate:
                    surface = self._load_freesurfer_surface(path)
                    result['surface'] = surface
                    result['surface_type'] = 'freesurfer'
                elif dicom:
                    # ``stop_before_pixels`` avoids loading heavy pixel data
                    dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
                    result['ds'] = dataset
                    # Preparing the nested metadata representation inside the
                    # worker keeps the GUI responsive when large headers are
                    # parsed.
                    result['dicom_tree'] = self._dicom_dataset_to_tree(dataset)
            except Exception as exc:  # pragma: no cover - interactive load errors
                load_error = exc

        if ext in ['.json', '.tsv', '.nii', '.nii.gz'] or dicom or gifti_candidate or freesurfer_candidate:
            t = threading.Thread(target=worker)
            t.start()
            while t.is_alive():
                QApplication.processEvents()
                time.sleep(0.05)
            t.join()
        self._stop_loading()

        if load_error is not None:
            logging.exception("Failed to load %s", path)
            QMessageBox.critical(
                self,
                "Open",
                f"Unable to load {path.name}: {load_error}",
            )
            self.welcome.show()
            return

        if ext == '.json':
            self._setup_json_toolbar()
            self.viewer = self._json_view(path, result.get('data'))
        elif ext == '.tsv':
            self._setup_tsv_toolbar()
            self.viewer = self._tsv_view(path, result.get('df'))
        elif ext in ['.nii', '.nii.gz']:
            self._setup_nifti_toolbar()
            self.viewer = self._nifti_view(
                path,
                (
                    result.get('img'),
                    result.get('data'),
                    result.get('nifti_meta'),
                ),
            )
        elif result.get('surface'):
            self.surface_data = result['surface']
            if result.get('surface_type'):
                self.surface_data['type'] = result['surface_type']
            self._setup_surface_toolbar()
            self.viewer = self._surface_view(path, self.surface_data)
        elif dicom:
            self.viewer = self._dicom_view(path, result.get('dicom_tree'))
            self.toolbar.addStretch()
        elif ext in ['.html', '.htm']:
            self.viewer = self._html_view(path)
            self.toolbar.addStretch()

        self.layout().addWidget(self.viewer)

    def resizeEvent(self, event):
        """Ensure images rescale when the window size decreases."""
        super().resizeEvent(event)
        self.loading_label.setGeometry(0, 0, self.width(), self.height())
        # If a NIfTI image is currently loaded, update the displayed slice
        if (
            self.data is not None
            and self.current_path
            and _get_ext(self.current_path) in ['.nii', '.nii.gz']
            and hasattr(self, 'img_label')
        ):
            self._update_slice()

    def _setup_surface_toolbar(self) -> None:
        """Toolbar for surface files featuring a single 3-D view button."""

        self.view_surface_btn = QPushButton("Surface View")
        self.view_surface_btn.clicked.connect(self._show_surface_view)
        self.toolbar.addWidget(self.view_surface_btn)
        self.toolbar.addStretch()

    def _surface_view(self, path: Path, surface: dict[str, Any]) -> QWidget:
        """Display a textual summary for a loaded surface mesh."""

        widget = QWidget()
        layout = QVBoxLayout(widget)

        info = QTextEdit()
        info.setReadOnly(True)
        info.setMinimumHeight(220)

        raw_vertices = surface.get('vertices')
        raw_faces = surface.get('faces')
        vertices = np.asarray(
            raw_vertices if raw_vertices is not None else np.empty((0, 3), dtype=np.float32),
            dtype=np.float32,
        )
        faces = np.asarray(
            raw_faces if raw_faces is not None else np.empty((0, 3), dtype=np.int32),
            dtype=np.int32,
        )
        scalars = surface.get('scalars') or {}
        meta = surface.get('meta') or {}

        lines = [
            f"File: {path.name}",
            f"Vertices: {vertices.shape[0]:,}",
            f"Faces: {faces.shape[0]:,}",
        ]

        if vertices.size:
            mins = vertices.min(axis=0)
            maxs = vertices.max(axis=0)
            extent_text = (
                f"Bounds (mm): X {mins[0]:.2f} – {maxs[0]:.2f}, "
                f"Y {mins[1]:.2f} – {maxs[1]:.2f}, "
                f"Z {mins[2]:.2f} – {maxs[2]:.2f}"
            )
            lines.append(extent_text)

        if scalars:
            lines.append("Scalar fields:")
            for name, values in scalars.items():
                arr = np.asarray(values, dtype=np.float32)
                if arr.size == 0:
                    continue
                vmin = float(np.min(arr))
                vmax = float(np.max(arr))
                lines.append(f"  • {name}: range {vmin:.4g} – {vmax:.4g}")
        else:
            lines.append("No embedded scalar fields detected.")

        structure = meta.get('structure')
        if structure:
            lines.append(f"Structure: {structure}")

        coord = meta.get('coordinate_system') or {}
        if coord:
            dataspace = coord.get('dataspace') or 'unknown'
            xformspace = coord.get('xformspace') or 'unknown'
            lines.append(f"Coordinate system: {dataspace} → {xformspace}")

        gifti_meta = meta.get('gifti_meta') or {}
        if gifti_meta:
            lines.append("GIFTI metadata:")
            for key in sorted(gifti_meta):
                lines.append(f"  • {key}: {gifti_meta[key]}")

        fs_header = meta.get('freesurfer_header') or {}
        if fs_header:
            lines.append("FreeSurfer metadata:")
            for key in sorted(fs_header):
                lines.append(f"  • {key}: {fs_header[key]}")

        info.setPlainText("\n".join(lines))
        layout.addWidget(info)

        hint = QLabel(
            "Use the Surface View button above to explore the mesh interactively."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch()
        return widget

    def _load_gifti_surface(self, path: Path) -> dict[str, Any]:
        """Load a GIFTI surface, returning vertices, faces and scalar data."""

        img = nib.load(str(path))
        coords = None
        faces = None
        scalars: dict[str, np.ndarray] = {}
        structure = ""
        coord_info: dict[str, Any] = {}

        for arr in img.darrays:
            intent = getattr(arr, 'intent', '') or ''
            data = np.asarray(arr.data)
            if intent == 'NIFTI_INTENT_POINTSET':
                coords = data.astype(np.float32, copy=False)
                structure = arr.meta.get('AnatomicalStructurePrimary') or structure
                cs = getattr(arr, 'coordsys', None)
                if cs is not None:
                    coord_info = {
                        'dataspace': getattr(cs, 'dataspace', ''),
                        'xformspace': getattr(cs, 'xformspace', ''),
                        'xform': getattr(cs, 'xform', None),
                    }
            elif intent == 'NIFTI_INTENT_TRIANGLE':
                faces = data.astype(np.int32, copy=False)
            else:
                if coords is None or data.shape[0] != coords.shape[0]:
                    continue
                if data.ndim == 1:
                    candidate = np.nan_to_num(
                        data.astype(np.float32, copy=False),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    name = arr.meta.get('Name') or f"Field {len(scalars) + 1}"
                    scalars[str(name)] = candidate
                elif data.ndim == 2 and data.shape[1] == 1:
                    candidate = np.nan_to_num(
                        data[:, 0].astype(np.float32, copy=False),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    name = arr.meta.get('Name') or f"Field {len(scalars) + 1}"
                    scalars[str(name)] = candidate

        if coords is None or faces is None:
            raise ValueError("GIFTI file does not contain surface geometry")

        gifti_meta = {}
        try:
            pairs = getattr(img.meta, 'data', None) or []
            gifti_meta = {str(item.name): str(item.value) for item in pairs}
        except Exception:  # pragma: no cover - defensive fallback
            gifti_meta = {}

        return {
            'vertices': coords,
            'faces': faces,
            'scalars': scalars,
            'meta': {
                'structure': structure,
                'coordinate_system': coord_info,
                'gifti_meta': gifti_meta,
            },
        }

    def _load_freesurfer_surface(self, path: Path) -> dict[str, Any]:
        """Load a FreeSurfer ``*.pial``/``*.white`` style surface mesh."""

        try:
            from nibabel.freesurfer import io as fsio
        except Exception as exc:  # pragma: no cover - optional dependency issues
            raise RuntimeError("nibabel FreeSurfer support is unavailable") from exc

        try:
            verts, faces, header = fsio.read_geometry(str(path), read_metadata=True)
        except TypeError:  # pragma: no cover - older nibabel releases
            verts, faces = fsio.read_geometry(str(path))
            header = {}

        header_dict = {}
        if isinstance(header, dict):
            header_dict = {str(k): str(v) for k, v in header.items()}

        return {
            'vertices': verts.astype(np.float32, copy=False),
            'faces': faces.astype(np.int32, copy=False),
            'scalars': {},
            'meta': {
                'structure': Path(path).stem,
                'freesurfer_header': header_dict,
            },
        }

    def _setup_json_toolbar(self):
        """Add buttons for JSON editing: Add Field, Delete Field, Save."""
        for txt, fn in [("Add Field", self._add_field), ("Delete Field", self._del_field), ("Save", self._save)]:
            btn = QPushButton(txt)
            btn.clicked.connect(fn)
            self.toolbar.addWidget(btn)
        self.toolbar.addStretch()

    def _setup_tsv_toolbar(self):
        """Add buttons for TSV editing: Add Row, Del Row, Add Col, Del Col, Save."""
        for txt, fn in [
            ("Add Row", self._add_row), ("Del Row", self._del_row),
            ("Add Col", self._add_col), ("Del Col", self._del_col), ("Save", self._save)
        ]:
            btn = QPushButton(txt)
            btn.clicked.connect(fn)
            self.toolbar.addWidget(btn)
        self.toolbar.addStretch()

    def _setup_nifti_toolbar(self):
        """Toolbar for NIfTI viewer with orientation buttons and sliders."""
        # Orientation buttons
        self.orientation = 2  # 0=sagittal, 1=coronal, 2=axial (default)
        self.ax_btn = QPushButton("Axial")
        self.co_btn = QPushButton("Coronal")
        self.sa_btn = QPushButton("Sagittal")
        for b in (self.ax_btn, self.co_btn, self.sa_btn):
            b.setCheckable(True)
        self.ax_btn.setChecked(True)
        self.ax_btn.clicked.connect(lambda: self._set_orientation(2))
        self.co_btn.clicked.connect(lambda: self._set_orientation(1))
        self.sa_btn.clicked.connect(lambda: self._set_orientation(0))
        self.toolbar.addWidget(self.sa_btn)
        self.toolbar.addWidget(self.co_btn)
        self.toolbar.addWidget(self.ax_btn)

        self.view3d_btn = QPushButton("3D View")
        self.view3d_btn.clicked.connect(self._show_3d_view)
        self.toolbar.addWidget(self.view3d_btn)

        self.graph_btn = QPushButton("Graph")
        self.graph_btn.setCheckable(True)
        self.graph_btn.clicked.connect(self._toggle_graph)
        self.toolbar.addWidget(self.graph_btn)

        # Helper to add slider+label vertically
        def add_slider(title, slider, val_label=None):
            box = QVBoxLayout()
            lab = QLabel(title)
            lab.setAlignment(Qt.AlignCenter)
            box.addWidget(lab)
            row = QHBoxLayout()
            row.addWidget(slider)
            if val_label is not None:
                row.addWidget(val_label)
            box.addLayout(row)
            self.toolbar.addLayout(box)

        # Slice slider
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self._update_slice)
        self.slice_val = QLabel("0")
        add_slider("Slice", self.slice_slider, self.slice_val)

        # Volume slider
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.valueChanged.connect(self._update_slice)
        self.vol_val = QLabel("0")
        add_slider("Volume", self.vol_slider, self.vol_val)
        # Brightness slider
        self.bright_slider = QSlider(Qt.Horizontal)
        self.bright_slider.setRange(-100, 100)
        self.bright_slider.setValue(0)
        self.bright_slider.valueChanged.connect(self._update_slice)
        add_slider("Brightness", self.bright_slider)

        # Contrast slider
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self._update_slice)
        add_slider("Contrast", self.contrast_slider)
        self.voxel_val_label = QLabel("N/A")
        self.value_row.addWidget(QLabel("Voxel value:"))
        self.value_row.addWidget(self.voxel_val_label)
        self.value_row.addStretch()
        self.toolbar.addStretch()

    def _show_3d_view(self) -> None:
        """Launch the interactive 3-D renderer for the loaded volume."""

        if getattr(self, "data", None) is None:
            QMessageBox.warning(self, "3D Viewer", "No NIfTI volume is currently loaded.")
            return

        voxel_sizes = None
        if getattr(self, "nifti_img", None) is not None:
            try:
                zooms = self.nifti_img.header.get_zooms()
                voxel_sizes = tuple(float(v) for v in zooms[:3])
            except Exception:  # pragma: no cover - defensive: malformed header
                voxel_sizes = None

        default_mode = None
        if self.current_path:
            name = self.current_path.name.lower()
            if any(tag in name for tag in ("dwi", "dti", "diffusion")):
                default_mode = "RMS (DTI)"

        title = (
            f"3D View – {self.current_path.name}"
            if self.current_path is not None
            else "3D Volume Viewer"
        )

        try:
            dialog = Volume3DDialog(
                self,
                self.data,
                meta=getattr(self, "_nifti_meta", {}),
                voxel_sizes=voxel_sizes,
                default_mode=default_mode,
                title=title,
                dark_theme=self._is_dark_theme(),
            )
        except Exception as exc:  # pragma: no cover - interactive error reporting
            logging.exception("Failed to initialise 3-D viewer")
            QMessageBox.critical(
                self,
                "3D Viewer",
                f"Unable to open 3D view: {exc}",
            )
            return

        dialog.exec_()

    def _show_surface_view(self) -> None:
        """Launch the interactive surface renderer for the loaded mesh."""

        if not self.surface_data:
            QMessageBox.warning(self, "Surface Viewer", "No surface mesh is currently loaded.")
            return

        title = (
            f"Surface View – {self.current_path.name}"
            if self.current_path is not None
            else "Surface Viewer"
        )

        surface_type = str(self.surface_data.get('type') or '').lower()

        try:
            if surface_type == 'freesurfer':
                dialog = FreeSurferSurfaceDialog(
                    self,
                    self.surface_data.get('vertices'),
                    self.surface_data.get('faces'),
                    title=title,
                    dark_theme=self._is_dark_theme(),
                )
            else:
                dialog = Surface3DDialog(
                    self,
                    self.surface_data.get('vertices'),
                    self.surface_data.get('faces'),
                    scalars=self.surface_data.get('scalars'),
                    meta=self.surface_data.get('meta'),
                    title=title,
                    dark_theme=self._is_dark_theme(),
                )
        except Exception as exc:  # pragma: no cover - interactive error reporting
            logging.exception("Failed to initialise surface viewer")
            QMessageBox.critical(
                self,
                "Surface Viewer",
                f"Unable to open surface view: {exc}",
            )
            return

        dialog.exec_()

    def _add_field(self):
        """Insert a new key/value pair into JSON tree."""
        tree = self.viewer
        sel = tree.currentItem() or tree.invisibleRootItem()
        item = QTreeWidgetItem(["newKey", "newValue"])
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        sel.addChild(item)
        tree.editItem(item, 0)

    def _del_field(self):
        """Delete selected field from JSON tree."""
        tree = self.viewer
        it = tree.currentItem()
        if it:
            parent = it.parent() or tree.invisibleRootItem()
            parent.removeChild(it)

    def _add_row(self):
        """Insert a new row into TSV table."""
        tbl = self.viewer
        tbl.insertRow(tbl.rowCount())

    def _del_row(self):
        """Delete selected row from TSV table."""
        tbl = self.viewer
        r = tbl.currentRow()
        if r >= 0:
            tbl.removeRow(r)

    def _add_col(self):
        """Insert a new column into TSV table."""
        tbl = self.viewer
        c = tbl.columnCount()
        tbl.insertColumn(c)
        tbl.setHorizontalHeaderItem(c, QTableWidgetItem(f"col{c+1}"))

    def _del_col(self):
        """Delete selected column from TSV table."""
        tbl = self.viewer
        c = tbl.currentColumn()
        if c >= 0:
            tbl.removeColumn(c)

    def _set_orientation(self, axis: int) -> None:
        """Set viewing orientation and update slice slider."""
        self.orientation = axis
        slider = getattr(self, 'vol_slider', None)
        vol_idx = slider.value() if slider is not None else 0
        vol = self._get_volume_data(vol_idx)
        axis_len = vol.shape[axis]
        self.slice_slider.setMaximum(max(axis_len - 1, 0))
        self.slice_slider.setEnabled(axis_len > 1)
        self.slice_slider.setValue(axis_len // 2)
        self.slice_val.setText(str(axis_len // 2))
        self._update_slice()

    def _nifti_view(self, path: Path, img_data=None) -> QWidget:
        """Create a simple viewer for NIfTI images with slice/volume controls."""
        meta = {}
        if img_data is None:
            self.nifti_img = nib.load(str(path))
            data, meta = self._get_nifti_data(self.nifti_img)
        else:
            if isinstance(img_data, tuple) and len(img_data) >= 2:
                self.nifti_img = img_data[0]
                data = img_data[1]
                if len(img_data) >= 3 and isinstance(img_data[2], dict):
                    meta = img_data[2] or {}
            else:
                self.nifti_img = img_data
                data, meta = self._get_nifti_data(self.nifti_img)

        if data is None:
            data, meta = self._get_nifti_data(self.nifti_img)

        self.data = data
        self._nifti_meta = meta or {}
        self._nifti_is_color = bool(self._nifti_meta.get("is_rgb"))
        widget = QWidget()
        vlay = QVBoxLayout(widget)

        self.cross_voxel = [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2]
        self._img_scale = (1.0, 1.0)

        self.img_label = _ImageLabel(self._update_slice, self._on_image_clicked)
        self.img_label.setAlignment(Qt.AlignCenter)
        # Allow the image to shrink as well as expand when resizing
        self.img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.img_label.setMinimumSize(1, 1)

        # Center the image label within a container so resizing doesn't shift
        # it vertically when the splitter changes size.
        img_container = QWidget()
        ic_layout = QVBoxLayout(img_container)
        ic_layout.setContentsMargins(0, 0, 0, 0)
        ic_layout.addWidget(self.img_label)

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(img_container)

        # Graph panel with scope selector
        self.graph_panel = QWidget()
        g_lay = QVBoxLayout(self.graph_panel)
        g_lay.setContentsMargins(0, 0, 0, 0)
        g_lay.setSpacing(2)

        self.graph_canvas = FigureCanvas(plt.Figure(figsize=(4, 2)))
        self.graph_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.graph_canvas.figure.subplots()
        g_lay.addWidget(self.graph_canvas)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope:"))
        self.scope_spin = QSpinBox()
        self.scope_spin.setRange(1, 4)
        self.scope_spin.setValue(1)
        self.scope_spin.valueChanged.connect(self._update_graph)
        scope_row.addWidget(self.scope_spin)
        scope_row.addSpacing(10)
        scope_row.addWidget(QLabel("Dot size:"))
        self.dot_size_spin = QSpinBox()
        self.dot_size_spin.setRange(1, 20)
        self.dot_size_spin.setValue(6)
        self.dot_size_spin.valueChanged.connect(self._update_graph)
        scope_row.addWidget(self.dot_size_spin)
        scope_row.addSpacing(15)
        self.mark_neighbors_box = QCheckBox("Mark neighbors")
        self.mark_neighbors_box.setChecked(True)
        self.mark_neighbors_box.stateChanged.connect(self._update_graph)
        scope_row.addWidget(self.mark_neighbors_box)
        scope_row.addStretch()
        g_lay.addLayout(scope_row)

        self.graph_panel.setVisible(False)
        self.splitter.addWidget(self.graph_panel)
        # Allow the image and graph to share space evenly when the graph is shown
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        vlay.addWidget(self.splitter)

        # Configure volume slider range. Structured RGB maps expose the last
        # axis as colour channels, not temporal volumes, so we keep the slider
        # disabled for those datasets.
        if data.ndim == 4 and not self._nifti_is_color:
            n_vols = data.shape[3]
        else:
            n_vols = 1
        self.vol_slider.setMaximum(max(n_vols - 1, 0))
        self.vol_slider.setEnabled(n_vols > 1)
        self.vol_slider.setValue(0)
        self.vol_val.setText("0")
        self.graph_btn.setVisible(n_vols > 1)

        # Initialize orientation and slice slider
        self._set_orientation(self.orientation)
        self._update_slice()
        return widget

    def _get_volume_data(self, vol_idx: int | None = None):
        """Return the 3-D volume used for display.

        Regular 4-D datasets encode the temporal dimension on the last axis and
        should be indexed by ``vol_idx``.  RGB/structured data keeps colour
        channels on the last axis which must remain untouched so they can be
        rendered as true-colour images.
        """

        if vol_idx is None:
            slider = getattr(self, 'vol_slider', None)
            vol_idx = slider.value() if slider is not None else 0

        if self.data.ndim == 4 and not getattr(self, '_nifti_is_color', False):
            vol_idx = max(0, min(vol_idx, self.data.shape[3] - 1))
            return self.data[..., vol_idx]
        return self.data

    def _update_slice(self):
        """Update displayed slice when slider moves."""
        slider = getattr(self, 'vol_slider', None)
        vol_idx = slider.value() if slider is not None else 0
        vol = self._get_volume_data(vol_idx)
        axis = getattr(self, 'orientation', 2)
        slice_idx = getattr(self, 'slice_slider', None).value() if hasattr(self, 'slice_slider') else vol.shape[axis] // 2
        self.slice_val.setText(str(slice_idx))
        self.vol_val.setText(str(vol_idx))
        if axis == 0:
            slice_img = vol[slice_idx, :, :]
        elif axis == 1:
            slice_img = vol[:, slice_idx, :]
        else:
            slice_img = vol[:, :, slice_idx]
        # Normalise the slice to 0..1 before applying display adjustments
        arr = slice_img.astype(np.float32)
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()

        # Apply brightness/contrast adjustments
        bright = getattr(self, 'bright_slider', None)
        contrast = getattr(self, 'contrast_slider', None)
        b_val = bright.value() / 100.0 if bright else 0.0
        c_factor = (contrast.value() / 100.0) if contrast else 1.0
        arr = (arr - 0.5) * c_factor + 0.5 + b_val
        arr = np.clip(arr, 0, 1)

        arr = (arr * 255).astype(np.uint8)
        arr = np.rot90(arr)
        if arr.ndim == 2:
            h, w = arr.shape
            img = QImage(arr.tobytes(), w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, c = arr.shape
            fmt = QImage.Format_RGB888 if c == 3 else QImage.Format_RGBA8888
            bytes_per_line = w * c
            img = QImage(arr.tobytes(), w, h, bytes_per_line, fmt)
        pix = QPixmap.fromImage(img)

        scaled = pix.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._img_scale = (scaled.width() / w, scaled.height() / h)

        # Draw crosshair after scaling for consistent width
        if self.cross_voxel is not None:
            x_rot, y_rot = self._voxel_to_arr(self.cross_voxel)
            scale_x, scale_y = self._img_scale
            x_s = int(x_rot * scale_x)
            y_s = int(y_rot * scale_y)
            # Use the highlight color so the crosshair is visible on any theme
            painter = QPainter(scaled)
            theme_color = self.palette().color(QPalette.Highlight)
            pen = QPen(theme_color)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawLine(x_s, 0, x_s, scaled.height())
            painter.drawLine(0, y_s, scaled.width(), y_s)
            square = max(2, int(min(scaled.width(), scaled.height()) * 0.02))
            half = square // 2
            painter.drawRect(x_s - half, y_s - half, square, square)
            painter.end()

        self.img_label.setPixmap(scaled)

        self._update_value()
        if self.graph_panel.isVisible():
            self._update_graph_marker()

    def _label_pos_to_img_coords(self, pos):
        # Convert click position on the scaled QLabel back to image coordinates
        pix = self.img_label.pixmap()
        if pix is None:
            return None
        pw, ph = pix.width(), pix.height()
        lw, lh = self.img_label.width(), self.img_label.height()
        off_x, off_y = (lw - pw) / 2, (lh - ph) / 2
        x = pos.x() - off_x
        y = pos.y() - off_y
        if 0 <= x < pw and 0 <= y < ph:
            scale_x, scale_y = self._img_scale
            return int(x / scale_x), int(y / scale_y)
        return None

    def _arr_to_voxel(self, x, y):
        # Map 2D display coordinates back to a voxel index within the volume
        vol_idx = self.vol_slider.value()
        vol = self._get_volume_data(vol_idx)
        axis = self.orientation
        slice_idx = self.slice_slider.value()
        if axis == 0:
            j = x
            k = vol.shape[2] - 1 - y
            return slice_idx, j, k
        elif axis == 1:
            i = x
            k = vol.shape[2] - 1 - y
            return i, slice_idx, k
        else:
            i = x
            j = vol.shape[1] - 1 - y
            return i, j, slice_idx

    def _voxel_to_arr(self, voxel):
        """Convert a voxel index back to 2-D array coordinates for drawing."""
        i, j, k = voxel
        vol_idx = self.vol_slider.value()
        vol = self._get_volume_data(vol_idx)
        axis = self.orientation
        if axis == 0:
            x = j
            y = vol.shape[2] - 1 - k
        elif axis == 1:
            x = i
            y = vol.shape[2] - 1 - k
        else:
            x = i
            y = vol.shape[1] - 1 - j
        return x, y

    def _on_image_clicked(self, event):
        coords = self._label_pos_to_img_coords(event.pos())
        if coords:
            voxel = self._arr_to_voxel(*coords)
            self.cross_voxel = list(voxel)
            self._update_slice()
            if self.graph_panel.isVisible():
                self._update_graph()

    def _update_value(self):
        if self.cross_voxel is None:
            self.voxel_val_label.setText("N/A")
            return
        vol_idx = self.vol_slider.value()
        i, j, k = self.cross_voxel
        if getattr(self, '_nifti_is_color', False):
            vec = np.asarray(self.data[i, j, k, :], dtype=float)
            if vec.size == 0:
                self.voxel_val_label.setText("N/A")
            else:
                components = ", ".join(f"{v:.3g}" for v in vec)
                self.voxel_val_label.setText(f"[{components}]")
            return

        if self.data.ndim == 4:
            val = self.data[i, j, k, vol_idx]
        else:
            val = self.data[i, j, k]
        self.voxel_val_label.setText(f"{float(val):.3g}")

    def _toggle_graph(self):
        visible = self.graph_btn.isChecked()
        self.graph_panel.setVisible(visible)
        total = self.splitter.size().height()
        if visible:
            self.splitter.setSizes([total // 2, total // 2])
            self._update_graph()
        else:
            self.splitter.setSizes([total, 0])

    def _update_graph(self):
        # Redraw the time-series graph for all voxels in the selected neighborhood
        # around ``self.cross_voxel``. Only valid for 4-D data.
        if (
            self.data.ndim != 4
            or self.cross_voxel is None
            or getattr(self, '_nifti_is_color', False)
        ):
            return

        level = self.scope_spin.value()
        dim = 2 * (level - 1) + 1
        half = dim // 2
        i0, j0, k0 = self.cross_voxel
        orient = self.orientation

        self.graph_canvas.figure.clf()
        axes = self.graph_canvas.figure.subplots(
            dim, dim, squeeze=False, sharex=True, sharey=True
        )

        line_color = "#000000" if not self._is_dark_theme() else "#ffffff"
        marker_color = self.palette().color(QPalette.Highlight).name()
        bg_color = self.palette().color(QPalette.Base).name()
        dot_size = self.dot_size_spin.value()
        self.graph_canvas.figure.set_facecolor(bg_color)
        self.markers = []
        self.marker_ts = []
        global_min = float("inf")
        global_max = float("-inf")

        for r, di in enumerate(range(-half, half + 1)):
            for c, dj in enumerate(range(-half, half + 1)):
                ax = axes[r][c]
                i, j, k = i0, j0, k0
                if orient == 0:
                    j = j0 + di
                    k = k0 + dj
                elif orient == 1:
                    i = i0 + di
                    k = k0 + dj
                else:
                    i = i0 + di
                    j = j0 + dj

                if not (0 <= i < self.data.shape[0] and 0 <= j < self.data.shape[1] and 0 <= k < self.data.shape[2]):
                    ax.axis("off")
                    continue

                ts_orig = self.data[i, j, k, :]
                ts = ts_orig
                global_min = min(global_min, ts_orig.min())
                global_max = max(global_max, ts_orig.max())
                ax.set_facecolor(bg_color)
                ax.plot(ts, color=line_color, linewidth=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(left=False, bottom=False)
                if self.mark_neighbors_box.isChecked() or (r == half and c == half):
                    self.marker_ts.append(ts)
                    idx = self.vol_slider.value()
                    marker, = ax.plot([idx], [ts[idx]], "o", color=marker_color, markersize=dot_size)
                    self.markers.append(marker)

        if global_min < global_max:
            for ax_row in axes:
                for ax in ax_row:
                    ax.set_ylim(global_min, global_max)

        self.graph_canvas.figure.tight_layout(pad=0.1)
        self.graph_canvas.draw()

    def _update_graph_marker(self):
        # Update the marker showing the current volume index on all axes
        if (
            not getattr(self, "markers", None)
            or not getattr(self, "marker_ts", None)
            or getattr(self, '_nifti_is_color', False)
        ):
            return
        marker_color = self.palette().color(QPalette.Highlight).name()
        idx = self.vol_slider.value()
        for marker, ts in zip(self.markers, self.marker_ts):
            i = max(0, min(idx, len(ts) - 1))
            marker.set_data([i], [ts[i]])
            marker.set_color(marker_color)
            marker.set_markersize(self.dot_size_spin.value())
        self.graph_canvas.draw_idle()

    def _json_view(self, path: Path, data=None) -> QTreeWidget:
        """Create a tree widget to show and edit JSON data."""
        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Key", "Value"])
        tree.setAlternatingRowColors(True)
        hdr = tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        hdr.setSectionResizeMode(1, QHeaderView.Interactive)
        if data is None:
            data = json.loads(path.read_text(encoding='utf-8'))
        self._populate_json(tree.invisibleRootItem(), data)
        tree.expandAll()
        tree.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        return tree

    def _populate_json(self, parent, data, editable: bool = True):
        """Recursively populate JSON-like data into the tree widget."""

        if isinstance(data, dict):
            for k, v in data.items():
                it = QTreeWidgetItem([str(k), '' if isinstance(v, (dict, list)) else str(v)])
                if editable:
                    it.setFlags(it.flags() | Qt.ItemIsEditable)
                parent.addChild(it)
                if isinstance(v, (dict, list)):
                    self._populate_json(it, v, editable)
        elif isinstance(data, list):
            for i, v in enumerate(data):
                it = QTreeWidgetItem([str(i), '' if isinstance(v, (dict, list)) else str(v)])
                if editable:
                    it.setFlags(it.flags() | Qt.ItemIsEditable)
                parent.addChild(it)
                if isinstance(v, (dict, list)):
                    self._populate_json(it, v, editable)

    def _tsv_view(self, path: Path, df=None) -> QTableWidget:
        """Create a table widget to show and edit TSV data."""
        if df is None:
            df = pd.read_csv(path, sep="\t", keep_default_na=False)
        tbl = QTableWidget(df.shape[0], df.shape[1])
        tbl.setAlternatingRowColors(True)
        hdr = tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        for j, col in enumerate(df.columns):
            tbl.setHorizontalHeaderItem(j, QTableWidgetItem(col))
            for i, val in enumerate(df[col].astype(str)):
                item = QTableWidgetItem(val)
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                tbl.setItem(i, j, item)
        for j in range(1, tbl.columnCount()):
            hdr.setSectionResizeMode(j, QHeaderView.Interactive)
        tbl.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        return tbl

    def _dicom_dataset_to_tree(self, dataset) -> Dict[str, Any]:
        """Return a nested mapping representing the supplied DICOM dataset.

        Iterating over all elements, especially sequences, can be expensive for
        large headers.  Performing the conversion in a worker thread keeps the
        GUI responsive while still producing a structure that mirrors the
        hierarchical layout of the dataset for display.
        """

        def ds_to_dict(ds) -> Dict[str, Any]:
            mapping: Dict[str, Any] = {}
            if ds is None:
                return mapping
            for elem in ds:
                name = elem.keyword or elem.name
                if elem.VR == "SQ":  # Sequence elements contain nested datasets
                    mapping[name] = [ds_to_dict(item) for item in elem.value]
                else:
                    # ``str`` ensures even binary/text values are rendered in a
                    # readable form inside the tree widget.
                    try:
                        mapping[name] = str(elem.value)
                    except Exception:
                        mapping[name] = "<unavailable>"
            return mapping

        file_meta = getattr(dataset, "file_meta", None)
        return {
            "File Meta Information": ds_to_dict(file_meta),
            "Dataset": ds_to_dict(dataset),
        }

    def _dicom_view(self, path: Path, metadata: Optional[Dict[str, Any]]) -> QTreeWidget:
        """Display precomputed DICOM metadata in a read-only tree."""

        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Tag", "Value"])
        tree.setAlternatingRowColors(True)
        hdr = tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        hdr.setSectionResizeMode(1, QHeaderView.Interactive)

        data = metadata or {"File Meta Information": {}, "Dataset": {}}
        self._populate_json(tree.invisibleRootItem(), data, editable=False)
        tree.expandAll()
        tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        return tree

    def _html_view(self, path: Path) -> QWebEngineView:
        """Display HTML file using a QWebEngineView for full rendering."""
        view = QWebEngineView()
        view.setUrl(QUrl.fromLocalFile(str(Path(path).resolve())))
        return view

    def _save(self):
        """Save edits made to JSON or TSV sidecar back to disk."""
        path = self.current_path
        if path.suffix.lower() == '.json':
            data = self._tree_to_obj(self.viewer.invisibleRootItem())
            path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        else:
            tbl = self.viewer
            hdrs = [tbl.horizontalHeaderItem(c).text() for c in range(tbl.columnCount())]
            rows = [{hdrs[c]: tbl.item(r, c).text() for c in range(tbl.columnCount())} for r in range(tbl.rowCount())]
            pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
        QMessageBox.information(self, "Saved", f"Saved {path}")

    def _tree_to_obj(self, root):
        """Convert tree representation back to nested JSON object."""
        obj = {} if root.childCount() else None
        for i in range(root.childCount()):
            ch = root.child(i)
            k, val = ch.text(0), ch.text(1)
            child = self._tree_to_obj(ch)
            obj[k] = child if child is not None else val
        return obj

def main() -> None:
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                u"bids.manager.1.0"
            )
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    if ICON_FILE.exists():
        app.setWindowIcon(QIcon(str(ICON_FILE)))
    win = BIDSManager()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

