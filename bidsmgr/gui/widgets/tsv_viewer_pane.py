"""TSV / TSV.GZ viewer + editor (Editor center pane, table kind).

Sister widget to :class:`SidecarFormPane`. When the user clicks a
``.tsv`` (or ``.tsv.gz``) file in the BIDS tree, :class:`EditorPanel`
swaps its center pane to this viewer.

* Parsed with pandas' C engine on a :class:`bidsmgr.workers.TsvLoaderWorker`
  thread (releases the GIL, so the GUI stays responsive) and bounded to a
  preview cap so even a multi-million-row table reads quickly.
* Rendered through a **lazy** :class:`_TsvTableModel`
  (:class:`~PyQt6.QtCore.QAbstractTableModel`) backed by plain Python
  lists. Crucially this creates **no per-cell objects** and only the
  visible cells are queried, so binding the data is O(1) - no main-thread
  freeze regardless of how many rows / columns the file has.
* Cells are inline-editable (double-click / F2 / select-then-click).
* Toolbar: ``+ Add row`` / ``+ Add column`` / ``− Delete row`` /
  ``− Delete column`` plus ``Revert`` / ``Save`` (manual-save model, same
  as the JSON sidecar pane). ``Save`` flushes to disk; ``Revert`` reloads.
* Switching files silently discards unsaved edits.

Theme handling: every palette colour comes from the global QSS.
:meth:`repaint_for_palette` runs the same unpolish/polish dance the
sidecar pane uses.
"""

from __future__ import annotations

import csv
import gzip
import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QStackedLayout,
    QStyledItemDelegate,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .flow_layout import flow
from .primitives import ElidedLabel, PaneHeader

log = logging.getLogger(__name__)


# Don't slurp arbitrarily-large TSVs into memory. BIDS TSVs (events /
# channels / participants / scans) are small in practice; if anyone passes
# a multi-million-row table we cap the preview here (the read is bounded to
# this, and the footer warns about truncation). Saving a truncated TSV
# would be destructive, so Save stays disabled when truncated on load.
_MAX_PREVIEW_ROWS = 5000


def _open_tsv_write(path: Path):
    """Open a ``.tsv`` or ``.tsv.gz`` for text writing (UTF-8)."""
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def _count_data_rows(path: Path) -> int:
    """Count data rows (newlines minus the header) at the C level.

    Used only when the preview was truncated, to report an exact total in
    the footer. Reads in binary chunks so the I/O (and zlib for ``.gz``)
    releases the GIL - it never freezes the worker.
    """
    is_gz = str(path).lower().endswith(".gz")
    opener = gzip.open if is_gz else open
    newlines = 0
    last = b""
    try:
        with opener(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                newlines += chunk.count(b"\n")
                last = chunk[-1:] or last
    except OSError:
        return 0
    if last and last != b"\n":
        newlines += 1  # final line had no trailing newline
    return max(0, newlines - 1)  # minus the header line


def _read_tsv(
    path: Path,
    *,
    max_rows: int = _MAX_PREVIEW_ROWS,
) -> tuple[list[str], list[list[str]], int]:
    """Read a TSV file. Returns ``(header, rows, total_rows)``.

    ``total_rows`` is the count of data rows on disk (so the footer can
    report truncation). Empty / malformed / missing files return
    ``([], [], 0)``.

    Parsed with pandas' **C engine** (releases the GIL during tokenising;
    Python's ``csv`` module is pure-Python and would hold the GIL, starving
    the worker so the loading spinner freezes). The read is **bounded** to
    ``max_rows + 1`` rows so a huge / wide file never materialises in full
    on the worker; the exact total is then counted cheaply at the C level.
    """
    import pandas as pd

    try:
        compression = "gzip" if str(path).lower().endswith(".gz") else "infer"
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            header=0,
            keep_default_na=False,
            na_filter=False,
            compression=compression,
            engine="c",
            on_bad_lines="skip",
            nrows=max_rows + 1,
        )
    except Exception as exc:  # noqa: BLE001 - empty / missing / malformed
        log.debug("could not read TSV %s: %s", path, exc)
        return [], [], 0

    header = [str(c) for c in df.columns]
    n_read = int(len(df))
    if not header and n_read == 0:
        return [], [], 0
    truncated = n_read > max_rows
    # Ragged rows: pandas pads short rows with NaN; coerce to "" so the
    # table mirrors the on-disk blanks.
    preview = df.head(max_rows).fillna("")
    rows = preview.astype(str).values.tolist()
    total = _count_data_rows(path) if truncated else n_read
    return header, rows, total


# ===========================================================================
# Lazy table model (no per-cell objects -> O(1) bind, no freeze)
# ===========================================================================
class _TsvTableModel(QAbstractTableModel):
    """A ``QAbstractTableModel`` over ``header: list[str]`` + ``rows``.

    Stores the parsed table as plain lists and serves cells on demand, so
    loading is O(1) (no ``QStandardItem`` per cell) and only visible cells
    are rendered. Supports inline cell editing + row/column add/remove for
    the toolbar.
    """

    # Emitted on any user cell edit (the pane uses it for dirty tracking).
    cellEdited = pyqtSignal()

    # Cell/column-highlight tints (semi-transparent; read on dark + light).
    _HL_BRUSH: dict[str, QColor] = {
        "err":   QColor(207, 34, 46, 120),
        "warn":  QColor(191, 135, 0, 130),
        "focus": QColor(79, 195, 247, 110),
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._header: list[str] = []
        self._rows: list[list[str]] = []
        # Findings highlighting: specific bad cells {(row, col): severity} take
        # priority; whole columns {col: severity} cover column-level findings.
        self._hl_cells: dict[tuple, str] = {}
        self._hl_cols: dict[int, str] = {}

    def set_highlights(self, cells: dict, cols: dict) -> None:
        """Set ``{(row, col): sev}`` cells + ``{col: sev}`` columns and repaint."""
        self._hl_cells = dict(cells)
        self._hl_cols = dict(cols)
        if self.rowCount() and self.columnCount():
            top = self.index(0, 0)
            bottom = self.index(self.rowCount() - 1, self.columnCount() - 1)
            self.dataChanged.emit(top, bottom, [Qt.ItemDataRole.BackgroundRole])

    # --- data ----------------------------------------------------------
    def set_table(self, header, rows) -> None:
        """Replace the whole table (load / restore). Rectangular-padded."""
        self.beginResetModel()
        self._hl_cells = {}
        self._hl_cols = {}
        self._header = [str(h) for h in header]
        w = len(self._header)
        self._rows = []
        for r in rows:
            cells = [str(x) for x in list(r)[:w]]
            if len(cells) < w:
                cells += [""] * (w - len(cells))
            self._rows.append(cells)
        self.endResetModel()

    def header(self) -> list[str]:
        return list(self._header)

    def rows(self) -> list[list[str]]:
        return [list(r) for r in self._rows]

    # --- QAbstractTableModel API --------------------------------------
    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._header)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            r, c = index.row(), index.column()
            if 0 <= r < len(self._rows) and 0 <= c < len(self._rows[r]):
                return self._rows[r][c]
            return ""
        if role == Qt.ItemDataRole.BackgroundRole:
            sev = self._hl_cells.get((index.row(), index.column())) \
                or self._hl_cols.get(index.column())
            color = self._HL_BRUSH.get(sev) if sev else None
            return QBrush(color) if color is not None else None
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._header[section] if 0 <= section < len(self._header) else ""
        return section + 1

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEditable
        )

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        r, c = index.row(), index.column()
        if not (0 <= r < len(self._rows)):
            return False
        while len(self._rows[r]) <= c:
            self._rows[r].append("")
        self._rows[r][c] = str(value)
        self.dataChanged.emit(
            index, index,
            [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole],
        )
        self.cellEdited.emit()
        return True

    # --- structural edits (toolbar) -----------------------------------
    def insert_row(self) -> None:
        n = len(self._rows)
        self.beginInsertRows(QModelIndex(), n, n)
        self._rows.append([""] * max(1, len(self._header)))
        self.endInsertRows()

    def remove_row(self, r: int) -> None:
        if 0 <= r < len(self._rows):
            self.beginRemoveRows(QModelIndex(), r, r)
            del self._rows[r]
            self.endRemoveRows()

    def add_column(self, name: str) -> None:
        c = len(self._header)
        self.beginInsertColumns(QModelIndex(), c, c)
        self._header.append(str(name))
        for row in self._rows:
            row.append("")
        self.endInsertColumns()

    def remove_column(self, c: int) -> None:
        if 0 <= c < len(self._header):
            self.beginRemoveColumns(QModelIndex(), c, c)
            del self._header[c]
            for row in self._rows:
                if c < len(row):
                    del row[c]
            self.endRemoveColumns()


class _HighlightDelegate(QStyledItemDelegate):
    """Paints the model's ``BackgroundRole`` brush behind a cell.

    A QSS ``::item`` rule makes ``QTableView`` ignore the model's background
    brush (the style draws the item), so findings highlights set on the model
    never showed. This delegate fills the cell with the brush first, then lets
    the default painter draw the (transparent-background) text + border on top.
    Selected cells keep the selection colour (the brush is skipped then).
    """

    def paint(self, painter, option, index):  # noqa: N802
        brush = index.data(Qt.ItemDataRole.BackgroundRole)
        from PyQt6.QtWidgets import QStyle
        if brush is not None and not (option.state & QStyle.StateFlag.State_Selected):
            painter.save()
            painter.fillRect(option.rect, brush)
            painter.restore()
        super().paint(painter, option, index)


class TsvViewerPane(QWidget):
    """Editable table view for BIDS ``.tsv`` files."""

    # Emitted after a successful save. Per-file revalidation hooks here.
    file_saved = pyqtSignal(Path)
    # Emitted when the disk write fails. Args: (file_path, error_msg).
    save_failed = pyqtSignal(Path, str)
    # Emitted whenever the dirty state flips. ``True`` means unsaved edits.
    dirty_changed = pyqtSignal(bool)
    # Emitted whenever undo/redo availability changes (Editor toolbar sync).
    history_changed = pyqtSignal()
    # Emitted after a background load completes (success or empty). Tests
    # wait on this; the Editor mirrors it to its busy spinner.
    loaded = pyqtSignal(Path)
    # Emitted while a background load is in flight / finishes. Args:
    # ``(busy, message)``. The Editor mirrors it to its toolbar spinner.
    loading_changed = pyqtSignal(bool, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane-dark")

        self._current_file: Optional[Path] = None
        self._current_root: Optional[Path] = None
        # Background TSV parse worker. Kept on self so it isn't GC'd.
        self._loader = None
        # Disk snapshot — used to compute dirty + restore on revert.
        self._original_header: list[str] = []
        self._original_rows: list[list[str]] = []
        # True when the on-disk file is larger than the preview cap (Save
        # disabled so we never drop the tail rows).
        self._truncated_on_load: bool = False
        self._dirty = False
        # Findings to highlight once the table is loaded: a list of
        # ``(column_name, line, severity)`` set by the Editor's fix /
        # highlight-all buttons. Applied on ``loaded`` (parsing is threaded).
        self._pending_findings: list = []
        # Undo/redo of in-memory table edits (snapshot = (header, rows)).
        from .edit_history import SnapshotHistory
        self._history = SnapshotHistory()
        self._pre_edit: Optional[tuple] = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader("Table"))

        # --- Edit toolbar ----------------------------------------------
        # A WRAPPING bar. A QHBoxLayout's minimum width is the sum of its
        # children, so six buttons put a 627-pixel floor under a pane the
        # user is meant to be able to drag narrow. See flow_layout.py.
        self._edit_toolbar = QFrame()
        self._edit_toolbar.setObjectName("sidecar-toolbar")
        et = flow(self._edit_toolbar, h_spacing=8, v_spacing=6)
        et.setContentsMargins(14, 6, 14, 6)

        self._add_row_btn = QPushButton("+ Add row")
        self._add_row_btn.setObjectName("tb-btn")
        self._add_row_btn.clicked.connect(self._on_add_row)
        et.addWidget(self._add_row_btn)

        self._del_row_btn = QPushButton("− Delete row")
        self._del_row_btn.setObjectName("tb-btn")
        self._del_row_btn.setEnabled(False)
        self._del_row_btn.clicked.connect(self._on_delete_row)
        et.addWidget(self._del_row_btn)

        self._add_col_btn = QPushButton("+ Add column")
        self._add_col_btn.setObjectName("tb-btn")
        self._add_col_btn.clicked.connect(self._on_add_column)
        et.addWidget(self._add_col_btn)

        self._del_col_btn = QPushButton("− Delete column")
        self._del_col_btn.setObjectName("tb-btn")
        self._del_col_btn.setEnabled(False)
        self._del_col_btn.clicked.connect(self._on_delete_column)
        et.addWidget(self._del_col_btn)

        self._dirty_chip = QLabel("")
        self._dirty_chip.setObjectName("sidecar-dirty-chip")
        self._dirty_chip.setVisible(False)
        # A continuous recording is a vector, and a grid of six-decimal
        # numbers cannot answer any question about its shape. Offered only
        # when the sidecar says there IS a sampling frequency, which is how
        # BIDS distinguishes a recording from a table of onsets.
        self._visualize_btn = QPushButton("  Visualize")
        # NOT "tb-btn". Every other control in this bar edits the table:
        # add a row, delete a column, save. This one leaves the table
        # behind and opens a different view of the same file, so it is
        # styled as the accent action it is rather than as one more verb in
        # a row of eight.
        self._visualize_btn.setObjectName("primary-btn")
        self._visualize_btn.setCheckable(True)
        self._visualize_btn.setVisible(False)
        self._visualize_btn.setToolTip(
            "Open the columns in the time-series viewer, the same one the "
            "MEG and EEG recordings use: channel picker, navigation, "
            "amplitude and window controls, zero-phase filtering, resample, "
            "spectrum and an events overlay. The sampling frequency and "
            "start time come from the sidecar; the channel kinds are read "
            "from the column names, so a cardiac trace is coloured and "
            "grouped as one."
        )
        self._visualize_btn.toggled.connect(self._on_visualize_toggled)
        et.addWidget(self._visualize_btn)

        # BIDS splits one run's physio by the ``recording`` entity, so the
        # cardiac trace, the belt and the trigger are three files describing
        # one acquisition. The question people bring to physio (did the
        # trigger fire where the ECG says it should) cannot be answered one
        # file at a time.
        self._together_btn = QPushButton("  All of this run")
        self._together_btn.setObjectName("tb-btn")
        self._together_btn.setCheckable(True)
        self._together_btn.setVisible(False)
        self._together_btn.setToolTip(
            "Show every physio recording of this run at once, on one "
            "clock. They are resampled onto the fastest one's grid and "
            "each keeps its own start time, because physio starts before "
            "the scanner does and by a different amount per device."
        )
        self._together_btn.toggled.connect(self._on_together_toggled)
        et.addWidget(self._together_btn)

        et.addWidget(self._dirty_chip)
        # No stretch: a wrapping row has no fixed right-hand edge to push
        # against, because where the edge is depends on how many rows there
        # turn out to be.
        self._revert_btn = QPushButton("Revert")
        self._revert_btn.setObjectName("tb-btn")
        self._revert_btn.setEnabled(False)
        self._revert_btn.clicked.connect(self.revert)
        et.addWidget(self._revert_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("tb-btn")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self.save)
        et.addWidget(self._save_btn)

        self._edit_toolbar.setVisible(False)
        v.addWidget(self._edit_toolbar)

        # --- Stacked content -------------------------------------------
        self._stack = QStackedLayout()
        self._stack.setContentsMargins(0, 0, 0, 0)
        v.addLayout(self._stack, 1)

        self._table = QTableView()
        self._table.setObjectName("tsv-view")
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self._table.horizontalHeader().setStretchLastSection(False)
        # Resize-to-contents must only SAMPLE a few rows, or it measures
        # every cell and freezes on big tables. 50 rows is plenty for a
        # sensible default width.
        self._table.horizontalHeader().setResizeContentsPrecision(50)
        self._table.verticalHeader().setVisible(True)
        self._model = _TsvTableModel(self)
        self._model.cellEdited.connect(self._on_cell_edited)
        self._table.setModel(self._model)
        # Paint findings highlights ourselves (QSS ::item styling otherwise makes
        # the view ignore the model's background brush).
        self._table.setItemDelegate(_HighlightDelegate(self._table))
        sel_model = self._table.selectionModel()
        if sel_model is not None:
            sel_model.currentChanged.connect(self._sync_delete_button_state)
        # Apply any deferred findings highlight once the (threaded) load lands.
        self.loaded.connect(lambda _p: self._apply_pending_highlights())

        self._empty_hint = QLabel(
            "Select a TSV file in the BIDS tree to view it."
        )
        self._empty_hint.setObjectName("pane-hint")
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_hint.setWordWrap(True)

        # Loading page (index 2): an animated spinner, mirroring the NIfTI
        # viewer.
        from .spinner import BusySpinner

        self._loading_page = QWidget()
        self._loading_page.setObjectName("pane-dark")
        lp = QVBoxLayout(self._loading_page)
        lp.addStretch(1)
        srow = QHBoxLayout()
        srow.addStretch(1)
        self._loading_spinner = BusySpinner()
        srow.addWidget(self._loading_spinner)
        srow.addStretch(1)
        lp.addLayout(srow)
        # ELIDED: it carries the file's path, and a QStackedWidget sizes
        # itself to its LARGEST page whichever one is showing, so a plain
        # QLabel here is a floor under the whole pane even while hidden.
        self._loading_label = ElidedLabel(
            "", mode=Qt.TextElideMode.ElideMiddle,
        )
        self._loading_label.setObjectName("pane-hint")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lp.addWidget(self._loading_label)
        lp.addStretch(1)

        # Index 0 = empty hint, 1 = table, 2 = loading, 3 = the plot (built
        # lazily, so a session that only reads tables never imports
        # pyqtgraph).
        self._stack.addWidget(self._empty_hint)
        self._stack.addWidget(self._table)
        self._stack.addWidget(self._loading_page)
        self._viewer_page: Optional[QWidget] = None
        self._signal_worker = None
        self._siblings: list = []
        self._timing: Optional[dict] = None
        self._stack.setCurrentIndex(0)

        # Footer (path + summary), QSS-driven so theme follows.
        self._footer = QFrame()
        self._footer.setObjectName("sidecar-footer")
        fl = QHBoxLayout(self._footer)
        fl.setContentsMargins(14, 6, 14, 6)
        fl.setSpacing(10)
        # ELIDED: a plain QLabel reports its full text width as its
        # MINIMUM, so a dataset-relative path was a floor of its own.
        self._footer_path = ElidedLabel("", mode=Qt.TextElideMode.ElideLeft)
        self._footer_path.setObjectName("sidecar-footer-path")
        self._footer_summary = ElidedLabel("")
        self._footer_summary.setObjectName("sidecar-footer-summary")
        fl.addWidget(self._footer_path, 1)
        fl.addWidget(self._footer_summary)
        v.addWidget(self._footer)

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def current_file(self) -> Optional[Path]:
        return self._current_file

    def is_dirty(self) -> bool:
        return self._dirty

    def highlight_findings(self, items) -> None:
        """Highlight TSV findings and scroll to the first.

        ``items`` is a list of ``(column_name, line, severity)`` where ``line``
        is the finding's 1-based row (including the header) or ``None`` for a
        whole-column finding. Deferred until the (threaded) load lands, so the
        Editor can call it right after switching to the TSV.
        """
        self._pending_findings = list(items or [])
        self._apply_pending_highlights()

    def clear_highlights(self) -> None:
        self._pending_findings = []
        self._model.set_highlights({}, {})

    def _apply_pending_highlights(self) -> None:
        items = self._pending_findings
        if not items:
            return
        header = self._model.header()
        if not header:
            return  # not loaded yet - stay pending
        nrows = self._model.rowCount()
        cells: dict = {}
        cols: dict = {}
        first = None
        for name, line, sev in items:
            if name not in header:
                continue
            c = header.index(name)
            if line is not None:
                r = line - 2  # line is 1-based incl. header; data row 0 == line 2
                if 0 <= r < nrows:
                    cells[(r, c)] = sev
                    if first is None:
                        first = (r, c)
            else:
                cols[c] = sev
                if first is None:
                    first = (0, c)
        if not cells and not cols:
            return  # nothing resolved (e.g. not a column) - stay pending
        self._model.set_highlights(cells, cols)
        if first is not None:
            self._table.scrollTo(self._model.index(*first))
        self._pending_findings = []

    def set_file(
        self,
        path: Optional[Path],
        root: Optional[Path],
    ) -> None:
        """Bind the pane to a TSV (or ``None`` to clear).

        The parse runs on a :class:`bidsmgr.workers.TsvLoaderWorker`; the
        spinner page shows while it reads and the table (lazy model) is
        bound instantly when it lands.
        """
        if self._loader is not None:
            self._loader.cancel()
            self._loader = None

        self._current_file = path
        self._current_root = root
        self._history.clear()
        # A new selection invalidates any deferred highlight from a prior fix
        # click (the Editor re-sets it after this returns when relevant).
        self._pending_findings = []
        if path is None:
            self._reset_model()
            self._original_header = []
            self._original_rows = []
            self._truncated_on_load = False
            self._dirty = False
            self._stack.setCurrentIndex(0)
            self._empty_hint.setText(
                "Select a TSV file in the BIDS tree to view it."
            )
            self._footer_path.setText("")
            self._footer_summary.setText("")
            self._edit_toolbar.setVisible(False)
            self._refresh_dirty_ui()
            self._pre_edit = self._snapshot()
            self.history_changed.emit()
            self.loading_changed.emit(False, "")
            return

        self._reset_model()
        self._loading_label.setText(f"Loading {path.name}…")
        self._loading_spinner.set_busy(True, message="")
        self._stack.setCurrentIndex(2)
        self._edit_toolbar.setVisible(False)
        self._footer_path.setText("")
        self._footer_summary.setText("")
        self.loading_changed.emit(True, f"Loading {path.name}…")

        from ...workers import TsvLoaderWorker

        worker = TsvLoaderWorker(path, _MAX_PREVIEW_ROWS, parent=self)
        worker.finished_with_data.connect(self._on_loaded)
        worker.failed.connect(self._on_load_failed)
        worker.finished.connect(worker.deleteLater)
        self._loader = worker
        worker.start()

    def _on_loaded(
        self,
        header: list,
        rows: list,
        total: int,
        path: Path,
    ) -> None:
        """Background parse finished — bind the table (guarded for staleness)."""
        if path != self._current_file:
            return
        self._loader = None
        self._loading_spinner.set_busy(False)
        self._original_header = list(header)
        self._original_rows = [list(r) for r in rows]
        self._truncated_on_load = total > len(rows)
        self._dirty = False

        if not header and not rows:
            self._reset_model()
            self._stack.setCurrentIndex(0)
            self._empty_hint.setText(
                "This TSV is empty or could not be parsed."
            )
        else:
            self._populate_model(header, rows)
            self._stack.setCurrentIndex(1)
        self._update_footer(path, self._current_root, len(rows), len(header), total)
        self._offer_visualizer(path, header, rows)
        self._edit_toolbar.setVisible(True)
        self._refresh_dirty_ui()
        self._pre_edit = self._snapshot()
        self.history_changed.emit()
        self.loading_changed.emit(False, "")
        self.loaded.emit(path)

    # ------------------------------------------------------------------
    # The viewer
    # ------------------------------------------------------------------

    def _offer_visualizer(
        self, path: Path, header: list, rows: list,
    ) -> None:
        """Show the Visualize toggle when this file is a continuous recording.

        Decided by the SIDECAR, not by the filename: BIDS puts the sampling
        frequency there, and a table of onsets (``_events.tsv``) has none.
        That also means a ``_stim.tsv.gz`` or any future continuous suffix
        is offered a viewer without this knowing the suffix exists.
        """
        from .physio_viewer import read_timing, related_recordings

        timing = read_timing(path)
        # The header row counts as a sample. A physio TSV has no column
        # names in the file (they are in the sidecar), so pandas reads the
        # first SAMPLE as the header and it belongs back in the data.
        if timing is not None and len(timing["columns"]) == len(header):
            rows = [list(header)] + [list(r) for r in rows]
        self._timing = timing
        self._plot_rows = rows if timing is not None else []
        self._visualize_btn.setVisible(timing is not None)
        siblings = related_recordings(path) if timing is not None else []
        self._siblings = siblings
        self._together_btn.setVisible(len(siblings) > 1)
        if timing is None:
            self._visualize_btn.setChecked(False)
            self._together_btn.setChecked(False)
        elif self._visualize_btn.isChecked():
            self._show_visualizer()

    def _on_visualize_toggled(self, showing: bool) -> None:
        if showing:
            self._show_visualizer()
        else:
            self._stack.setCurrentIndex(1)

    def _show_visualizer(self) -> None:
        """Read the whole recording on a worker, then hand it to the view.

        The table's preview is NOT shown first. It is bounded at five
        thousand rows, so on a 1.4-million-sample trigger channel it is the
        first four seconds, and showing it while the real read completes
        means the viewer visibly changes its mind about what the recording
        is. A spinner that says "reading" is the honest version.
        """
        if self._timing is None:
            self._visualize_btn.setChecked(False)
            return
        if self._viewer_page is None:
            try:
                from .time_series_view import TimeSeriesView

                self._viewer_page = TimeSeriesView(self)
                # Physio is a channel or four and fits in a window whole, so
                # this is the one consumer that offers Fit all. MEG and EEG
                # do not, where the whole recording is hundreds of millions
                # of samples.
                self._viewer_page.enable_fit_all(True)
                self._viewer_page.close_requested.connect(
                    lambda: self._visualize_btn.setChecked(False)
                )
            except Exception as exc:  # noqa: BLE001 - reported, not hidden
                log.warning("could not build the time-series viewer: %s", exc)
                self._visualize_btn.setChecked(False)
                self._visualize_btn.setEnabled(False)
                self._visualize_btn.setToolTip(f"Unavailable: {exc}")
                return
            self._stack.addWidget(self._viewer_page)

        self._stack.setCurrentWidget(self._viewer_page)
        self._start_signal_read()

    def _start_signal_read(self) -> None:
        """Read + build the RawArray off the GUI thread.

        A ``QThread``, not the pool: this ends in scipy, and a pooled thread
        retired between calls takes scipy's per-thread state with it
        (CLAUDE.md guard 8b).
        """
        from ...workers.meeg_recording_loader import RecordingComputeWorker
        from .physio_viewer import build_combined_raw, build_raw, read_columns

        path = self._current_file
        timing = dict(self._timing or {})
        if path is None:
            return

        previous = self._signal_worker
        if previous is not None:
            previous.cancel()

        together = self._together_btn.isChecked() and len(self._siblings) > 1
        siblings = list(self._siblings)

        def work():
            if together:
                raw, gaps = build_combined_raw(siblings)
                return path, raw, gaps, raw.n_times, 1
            columns, total, step = read_columns(path)
            if not columns:
                raise ValueError("no numeric columns could be read")
            raw, gaps = build_raw(columns, timing, step)
            return path, raw, gaps, total, step

        worker = RecordingComputeWorker(work, parent=self)
        worker.finished_with_result.connect(self._on_signal_read)
        worker.failed.connect(self._on_signal_failed)
        worker.finished.connect(worker.deleteLater)
        self._signal_worker = worker
        self.loading_changed.emit(True, "Reading the recording...")
        worker.start()

    def _on_together_toggled(self, _checked: bool) -> None:
        """Re-read in the other mode, if a viewer is open."""
        if self._visualize_btn.isChecked():
            self._start_signal_read()

    def _on_signal_read(self, result) -> None:
        path, raw, gaps, total, step = result
        self._signal_worker = None
        self.loading_changed.emit(False, "")
        if path != self._current_file or self._viewer_page is None:
            return
        self._viewer_page.set_current_filepath(path, self._current_root)
        self._viewer_page.load_raw(raw, gaps=gaps)
        if step > 1:
            self._viewer_page.status_message.emit(
                f"{total:,} samples held as {raw.n_times:,}: the recording "
                f"was read one sample in {step} to stay within memory"
            )

    def _on_signal_failed(self, message: str) -> None:
        self._signal_worker = None
        self.loading_changed.emit(False, "")
        log.warning("could not open the recording for viewing: %s", message)
        self._visualize_btn.setChecked(False)

    def _on_load_failed(self, path: Path, error: str) -> None:
        if path != self._current_file:
            return
        self._loader = None
        self._loading_spinner.set_busy(False)
        self._reset_model()
        self._stack.setCurrentIndex(0)
        self._empty_hint.setText(f"Could not load {path.name}:\n{error}")
        self._edit_toolbar.setVisible(False)
        self._dirty = False
        self._refresh_dirty_ui()
        self._pre_edit = self._snapshot()
        self.history_changed.emit()
        self.loading_changed.emit(False, "")
        self.loaded.emit(path)

    def save(self) -> bool:
        """Flush the model to disk. ``True`` on success / no-op."""
        if self._current_file is None:
            return True
        if not self._dirty:
            return True
        if self._truncated_on_load:
            msg = (
                f"File was truncated on load ({_MAX_PREVIEW_ROWS} rows "
                "shown); refusing to overwrite — saving would discard "
                "the tail rows."
            )
            log.warning("save refused for %s: %s", self._current_file, msg)
            self.save_failed.emit(self._current_file, msg)
            return False
        try:
            self._write_model_to_disk(self._current_file)
        except OSError as exc:
            log.warning("save failed for %s: %s", self._current_file, exc)
            self.save_failed.emit(self._current_file, str(exc))
            return False
        self._snapshot_current_model()
        self._dirty = False
        self._refresh_dirty_ui()
        self.file_saved.emit(self._current_file)
        return True

    def revert(self) -> None:
        """Reload the bound file from disk, dropping unsaved edits."""
        if self._current_file is None:
            return
        path, root = self._current_file, self._current_root
        self.set_file(path, root)

    def repaint_for_palette(self, pal: dict) -> None:
        """Same QSS-only refresh pattern as :class:`SidecarFormPane`.

        With one exception, and it is the reason this takes ``pal`` at all:
        **the plot is drawn by pyqtgraph, which reads no QSS.** Unpolishing
        and re-polishing it does nothing, so a dark/light swap left the
        plot on the old theme's background until the app was restarted and
        it happened to be built under the new one. The palette is handed
        down instead, the way the MEG/EEG viewer already hands it to its
        own plot.
        """
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()
        if self._viewer_page is not None:
            self._viewer_page.repaint_for_palette(pal)

    # ----------------------------------------------------------------------
    # Toolbar handlers
    # ----------------------------------------------------------------------

    def _on_add_row(self) -> None:
        if self._current_file is None:
            return
        if self._model.columnCount() == 0:
            self._model.add_column("col1")
        self._model.insert_row()
        self._mark_dirty()
        self._stack.setCurrentIndex(1)

    def _on_delete_row(self) -> None:
        idx = self._table.currentIndex()
        if not idx.isValid():
            return
        self._model.remove_row(idx.row())
        self._mark_dirty()

    def _on_add_column(self) -> None:
        if self._current_file is None:
            return
        name, ok = QInputDialog.getText(
            self, "Add column", "Column name:", text="newCol",
        )
        if not ok or not name.strip():
            return
        self._model.add_column(name.strip())
        self._mark_dirty()
        self._stack.setCurrentIndex(1)

    def _on_delete_column(self) -> None:
        idx = self._table.currentIndex()
        if not idx.isValid():
            return
        self._model.remove_column(idx.column())
        self._mark_dirty()

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    def _reset_model(self) -> None:
        self._model.set_table([], [])

    def _populate_model(
        self,
        header: list[str],
        rows: list[list[str]],
    ) -> None:
        # O(1) bind: the lazy model just stores the lists. Resize only
        # samples a few rows (precision set in __init__), so this is fast
        # no matter how big / wide the table is.
        self._model.set_table(header, rows)
        self._table.resizeColumnsToContents()

    def _snapshot_current_model(self) -> None:
        self._original_header = self._model.header()
        self._original_rows = self._model.rows()

    def _current_header(self) -> list[str]:
        return self._model.header()

    def _current_rows(self) -> list[list[str]]:
        return self._model.rows()

    def _write_model_to_disk(self, path: Path) -> None:
        header = self._model.header()
        rows = self._model.rows()
        with _open_tsv_write(path) as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)
            writer.writerows(rows)

    def _on_cell_edited(self) -> None:
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        if self._pre_edit is not None:
            self._history.record(self._pre_edit)
        self._pre_edit = self._snapshot()
        if not self._dirty:
            self._dirty = True
        self._refresh_dirty_ui()
        self.history_changed.emit()

    # ----------------------------------------------------------------------
    # Undo / redo (snapshot-based, in-memory; disk write still via Save)
    # ----------------------------------------------------------------------

    def _snapshot(self) -> tuple[list[str], list[list[str]]]:
        return (self._model.header(), self._model.rows())

    def _restore(self, snap: tuple[list[str], list[list[str]]]) -> None:
        header, rows = snap
        self._populate_model(list(header), [list(r) for r in rows])
        self._stack.setCurrentIndex(1 if (header or rows) else 0)
        self._dirty = (
            list(header) != list(self._original_header)
            or [list(r) for r in rows] != [list(r) for r in self._original_rows]
        )
        self._pre_edit = self._snapshot()
        self._refresh_dirty_ui()
        self.history_changed.emit()

    def can_undo(self) -> bool:
        return self._current_file is not None and self._history.can_undo

    def can_redo(self) -> bool:
        return self._current_file is not None and self._history.can_redo

    def undo(self) -> None:
        if self._current_file is None:
            return
        snap = self._history.undo(self._snapshot())
        if snap is not None:
            self._restore(snap)

    def redo(self) -> None:
        if self._current_file is None:
            return
        snap = self._history.redo(self._snapshot())
        if snap is not None:
            self._restore(snap)

    def _refresh_dirty_ui(self) -> None:
        has_file = self._current_file is not None
        editable = has_file and not self._truncated_on_load
        if self._dirty:
            self._dirty_chip.setText("unsaved changes")
            self._dirty_chip.setVisible(True)
        else:
            self._dirty_chip.setVisible(False)
        self._save_btn.setEnabled(editable and self._dirty)
        self._revert_btn.setEnabled(has_file and self._dirty)
        self._add_row_btn.setEnabled(editable)
        self._add_col_btn.setEnabled(editable)
        self._sync_delete_button_state()
        self.dirty_changed.emit(self._dirty)

    def _sync_delete_button_state(self, *args) -> None:
        del args
        editable = self._current_file is not None and not self._truncated_on_load
        idx = self._table.currentIndex()
        has_sel = idx.isValid()
        self._del_row_btn.setEnabled(
            editable and has_sel and self._model.rowCount() > 0
        )
        self._del_col_btn.setEnabled(
            editable and has_sel and self._model.columnCount() > 0
        )

    def _update_footer(
        self,
        path: Path,
        root: Optional[Path],
        shown_rows: int,
        cols: int,
        total_rows: int,
    ) -> None:
        if root is not None:
            try:
                rel = path.resolve().relative_to(root.resolve())
                self._footer_path.setText(str(rel))
            except ValueError:
                self._footer_path.setText(str(path))
        else:
            self._footer_path.setText(str(path))
        if shown_rows < total_rows:
            self._footer_summary.setText(
                f"{shown_rows} of {total_rows} rows shown · {cols} columns "
                f"· read-only (truncated)"
            )
        else:
            self._footer_summary.setText(
                f"{total_rows} rows · {cols} columns"
            )


__all__ = ["TsvViewerPane"]
