"""The Converter view — the pre-conversion half of the Inspector layout.

Reference: ``inspector_proto/proto.py`` ``ConverterView``,
``gui_mockups.html`` proposal 1.

M3 scope: this lands the **toolbar + path bars + Inspection table**
backed by a real :class:`InventoryTableModel`. The other three panes
in the prototype (raw-FS tree, filter tree, properties panel) and the
bottom dock (BIDS preview / Log / Conflicts / Stats) are stubbed as
empty placeholders so the splitter shape matches; they fill in during
later milestones.

Public API:

* :class:`ConverterPanel(parent=None)` — instantiate, place in a parent
  layout, optionally pass a :class:`bidsmgr.project.Project` so edits
  in the table append events.
* :meth:`ConverterPanel.start_scan(dicom_root, output_tsv, ...)` —
  programmatic entry point (the Scan… button calls this internally,
  tests bypass the file dialog by calling it directly).
* :meth:`ConverterPanel.load_inventory(df, output_tsv)` — swap in a
  ready DataFrame (used by the worker's ``finished`` handler and by
  tests that load a TSV directly).

Signals:

* ``log_message(str)``     — forwarded from the active worker for the
  bottom dock's Log tab to consume.
* ``scan_finished(object, object)`` — re-emits the worker's payload so
  a controller can persist the result (e.g. write a ``StageCompleted``
  event to the project file).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableView,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..project import Project, ScanImported, StageCompleted
from ..workers import ConvertWorker, MetadataWorker, ScanWorker, ValidateWorker
from .app_settings import AppSettings
from .delegates import (
    CellTextDelegate,
    CheckboxDelegate,
    StatusDelegate,
)
from .filter_pane import FilterPane
from .models import COLUMNS, MANDATORY_COLUMN_KEYS, InventoryTableModel
from .output_fs_pane import OutputFsPane
from .properties_panel import PropertiesPanel
from .raw_fs_pane import RawFsPane
from .widgets import BusySpinner, Chip, PaneHeader, PathBar, VSep

log = logging.getLogger(__name__)


class ConverterPanel(QWidget):
    """The pre-conversion Inspector layout.

    Owns one :class:`InventoryTableModel` at a time. ``start_scan`` /
    ``load_inventory`` replace the model; the table view re-binds.
    """

    log_message = pyqtSignal(str)
    scan_finished = pyqtSignal(object, object)
    convert_finished = pyqtSignal(int, object)

    def __init__(self, project: Optional[Project] = None, parent=None) -> None:
        super().__init__(parent)
        self._project = project
        self._model: Optional[InventoryTableModel] = None
        self._scan_worker: Optional[ScanWorker] = None
        self._convert_worker: Optional[ConvertWorker] = None
        self._metadata_worker: Optional[MetadataWorker] = None
        self._validate_worker: Optional[ValidateWorker] = None
        self._raw_root: Optional[Path] = None
        self._output_tsv: Optional[Path] = None
        self._bids_parent: Optional[Path] = None
        # Persistent settings — loaded fresh each construction so a
        # Settings dialog save propagates on the next open of this panel.
        self._app_settings: AppSettings = AppSettings.load()
        # Column visibility: persisted via QSettings under
        # ``inspector/columns/<key>``. The setter ``set_column_visible``
        # writes through to QSettings so a restart re-applies the choice.
        self._column_visible: dict[str, bool] = self._load_column_visibility()
        # Restore previously-used paths so the user reopens the app in
        # the same context.
        if self._app_settings.raw_root:
            self._raw_root = Path(self._app_settings.raw_root)
        if self._app_settings.bids_parent:
            self._bids_parent = Path(self._app_settings.bids_parent)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # ---------------- toolbar ----------------
        v.addWidget(self._build_toolbar())

        # ---------------- path bars ----------------
        raw_init = str(self._raw_root) if self._raw_root else "(no folder selected)"
        self._raw_pathbar = PathBar(
            "Raw input", raw_init, ok=bool(self._raw_root),
        )
        self._raw_pathbar.change_button.clicked.connect(self._on_pick_raw_dir)
        v.addWidget(self._raw_pathbar)

        bids_init = str(self._bids_parent) if self._bids_parent else "(not set)"
        self._bids_pathbar = PathBar(
            "BIDS output", bids_init, ok=bool(self._bids_parent),
        )
        self._bids_pathbar.change_button.clicked.connect(self._on_pick_bids_parent)
        v.addWidget(self._bids_pathbar)

        # ---------------- main vertical splitter ----------------
        # Top: 4-col horizontal splitter (panes). Bottom: tabbed dock.
        v_split = QSplitter(Qt.Orientation.Vertical)
        v_split.setHandleWidth(1)
        v_split.setChildrenCollapsible(False)

        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setHandleWidth(1)
        h_split.setChildrenCollapsible(False)

        # Col 1 is itself a vertical splitter: raw input tree on top,
        # BIDS output tree below. Lets the user watch the converted
        # tree grow without leaving the Converter view.
        self._raw_pane = RawFsPane()
        self._output_pane = OutputFsPane()
        col1 = QSplitter(Qt.Orientation.Vertical)
        col1.setHandleWidth(1)
        col1.setChildrenCollapsible(False)
        col1.addWidget(self._raw_pane)
        col1.addWidget(self._output_pane)
        col1.setStretchFactor(0, 1)
        col1.setStretchFactor(1, 1)
        col1.setSizes([320, 320])
        h_split.addWidget(col1)

        self._filter_pane = FilterPane()
        h_split.addWidget(self._filter_pane)
        h_split.addWidget(self._build_inspection_pane())
        self._properties = PropertiesPanel()
        self._properties.set_project(self._project)
        h_split.addWidget(self._properties)
        h_split.setStretchFactor(0, 0)
        h_split.setStretchFactor(1, 0)
        h_split.setStretchFactor(2, 1)
        h_split.setStretchFactor(3, 0)
        h_split.setSizes([240, 220, 720, 320])
        v_split.addWidget(h_split)
        v_split.addWidget(self._build_bottom_dock())
        v_split.setStretchFactor(0, 1)
        v_split.setStretchFactor(1, 0)
        v_split.setSizes([560, 200])
        v.addWidget(v_split, 1)

        # Stream worker progress messages straight into the Log tab,
        # but throttled: high-volume workers (e.g. dcm2niix parallel)
        # can emit hundreds of lines per second, and forcing
        # ``appendPlainText`` on every one locks the GUI's event loop
        # behind constant repaints. Buffer lines and flush at 10 Hz.
        self._log_buffer: list[str] = []
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(100)  # 10 Hz
        self._log_flush_timer.timeout.connect(self._flush_log_buffer)
        self._log_flush_timer.start()
        self.log_message.connect(self._on_log_message)

        # Apply restored startup state to the side panes.
        if self._raw_root is not None:
            self._raw_pane.set_root(self._raw_root)
        if self._bids_parent is not None:
            self._output_pane.set_root(self._bids_parent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def model(self) -> Optional[InventoryTableModel]:
        """Return the active model (``None`` before the first scan)."""
        return self._model

    def load_inventory(self, df: pd.DataFrame, output_tsv: Optional[Path] = None) -> None:
        """Swap in a fresh DataFrame as the table's data source.

        Called by :meth:`_on_scan_finished` and by tests / controllers
        that want to skip the scan worker (e.g. by loading a
        pre-generated TSV from disk).
        """
        self._model = InventoryTableModel(df, project=self._project, parent=self)
        # Apply the user's "Highlight aborts" preference to the fresh model.
        self._model.set_highlight_aborts(self._aborts_btn.isChecked())
        self._table.setModel(self._model)
        self._apply_column_widths()
        # Selection wiring: every selection change pushes the row into
        # the Properties panel. Selection model exists only after
        # ``setModel``, so connect here, not in the constructor.
        sel_model = self._table.selectionModel()
        if sel_model is not None:
            sel_model.currentRowChanged.connect(self._on_current_row_changed)
            sel_model.selectionChanged.connect(self._on_selection_changed)
        self._properties.bind_model(self._model)
        self._filter_pane.bind_model(self._model)
        self._raw_pane.bind_model(self._model)
        if self._raw_root is not None:
            self._raw_pane.set_root(self._raw_root)
        if output_tsv is not None:
            self._output_tsv = Path(output_tsv)
        # The BIDS output pathbar is **independent** of the scan TSV
        # location — the user sets it explicitly with the change button.
        # Swap the inspection-pane stack from placeholder → table.
        self._inspection_stack.setCurrentWidget(self._table)
        self._update_status_chips()
        self._rebuild_bids_preview()
        self._update_stats_label()
        # Refresh chips + previews when the model edits a row.
        def _on_model_changed(*_a):
            self._update_status_chips()
            self._rebuild_bids_preview()
            self._update_stats_label()
        self._model.dataChanged.connect(_on_model_changed)
        # Default-select row 0 if any rows exist so the Properties panel
        # has something to show without the user having to click first.
        if self._model.rowCount() > 0:
            self._table.selectRow(0)
            self._run_btn.setEnabled(True)
            self._run_btn.setToolTip("Run conversion on the included rows")
        else:
            self._run_btn.setEnabled(False)

    @staticmethod
    def _default_dataset_slug(folder: Path) -> str:
        """Mirror ``bidsmgr.cli.scan._default_dataset_slug`` so the
        prefilled placeholder matches what the CLI would generate.
        """
        import re as _re
        raw = Path(folder).name.lower()
        slug = _re.sub(r"[^a-z0-9_-]+", "-", raw)
        slug = _re.sub(r"-{2,}", "-", slug).strip("-_")
        return slug or "dataset"

    def start_scan(
        self,
        dicom_root: Path,
        output_tsv: Path,
        *,
        dataset: Optional[str] = None,
        line_freq: Optional[float] = None,
        montage: Optional[str] = None,
        n_jobs: int = 1,
        probe_convert: bool = False,
        skip_bids_guess: bool = False,
    ) -> ScanWorker:
        """Kick off a background scan.

        Returns the worker so tests / controllers can ``waitSignal`` on
        it. The view auto-loads the resulting DataFrame on
        ``finished_with_result``.
        """
        # Refuse to overlap two scans; the user must wait for the
        # in-flight one to finish (rare; CLI scans are < 60s).
        if self._scan_worker is not None and self._scan_worker.isRunning():
            log.warning("scan already in progress; ignoring new request")
            return self._scan_worker

        self._raw_root = Path(dicom_root)
        self._output_tsv = Path(output_tsv)
        self._raw_pathbar.set_value(str(self._raw_root), ok=True)

        self._scan_btn.setEnabled(False)
        self._spinner.set_busy(True, message=f"Scanning {self._raw_root.name}…")
        worker = ScanWorker(
            self._raw_root, self._output_tsv,
            dataset=dataset, line_freq=line_freq, montage=montage,
            n_jobs=n_jobs,
            probe_convert=probe_convert,
            skip_bids_guess=skip_bids_guess,
            parent=self,
        )
        worker.progress.connect(self._on_progress)
        worker.finished_with_result.connect(self._on_scan_finished)
        worker.failed.connect(self._on_scan_failed)
        worker.finished.connect(self._on_worker_qt_finished)
        self._scan_worker = worker
        worker.start()
        return worker

    # ------------------------------------------------------------------
    # Internals — toolbar
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("toolbar")
        bar.setFixedHeight(44)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 6, 14, 6)
        lay.setSpacing(8)

        self._scan_btn = QPushButton("⌖  Scan…")
        self._scan_btn.setObjectName("tb-btn")
        self._scan_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._scan_btn.clicked.connect(self._on_scan_clicked)
        lay.addWidget(self._scan_btn)

        # TSV filename input — the scan writes to ``<bids_parent>/<name>``
        # so the inventory ends up next to the converted BIDS tree, not
        # cluttering the raw data folder. Filename only; ``<bids_parent>``
        # is set independently via the path bar below.
        tsv_label = QLabel("TSV:")
        tsv_label.setObjectName("tb-mini-label")
        lay.addWidget(tsv_label)
        self._tsv_filename_edit = QLineEdit()
        self._tsv_filename_edit.setObjectName("tb-input")
        self._tsv_filename_edit.setPlaceholderText("inventory.tsv")
        self._tsv_filename_edit.setText(self._app_settings.scan_tsv_filename)
        self._tsv_filename_edit.setMaximumWidth(150)
        self._tsv_filename_edit.editingFinished.connect(self._on_tsv_filename_edited)
        lay.addWidget(self._tsv_filename_edit)

        lay.addWidget(VSep())

        # Live status chips (update on every scan complete). The
        # ``warn``, ``err`` and ``skip`` chips are clickable — clicking
        # opens an IssuesDialog listing the affected rows. The
        # ``valid`` chip stays passive (nothing to drill into).
        self._chip_valid = Chip("0 valid", "success")
        self._chip_warn = Chip("0 warnings", "warn")
        self._chip_err = Chip("0 error", "err")
        self._chip_skip = Chip("0 skipped")
        for c in (self._chip_valid, self._chip_warn, self._chip_err, self._chip_skip):
            lay.addWidget(c)
        # Hand-cursor + click → IssuesDialog for the three actionable chips.
        self._chip_warn.set_clickable(True)
        self._chip_err.set_clickable(True)
        self._chip_skip.set_clickable(True)
        self._chip_warn.clicked.connect(lambda: self._open_issues_dialog("warn"))
        self._chip_err.clicked.connect(lambda: self._open_issues_dialog("err"))
        self._chip_skip.clicked.connect(lambda: self._open_issues_dialog("skip"))

        # Toggle: highlight suspected-abort rows with a purple tint.
        # Aborts are already auto-deselected by the scanner; highlighting
        # Highlight aborts + Bulk edit buttons are constructed here so
        # they're available before _build_inspection_pane runs, but
        # they're added to the inspection pane's footer (closer to the
        # table they act on), not to this toolbar.
        self._aborts_btn = QPushButton("⌬  Highlight aborts")
        self._aborts_btn.setObjectName("tb-btn-toggle")
        self._aborts_btn.setCheckable(True)
        self._aborts_btn.setChecked(self._app_settings.highlight_aborts)
        self._aborts_btn.setToolTip(
            "Highlight in purple any row the scanner flagged as "
            "``suspected_abort`` (operator restart after a noisy / blurry "
            "attempt). These rows are already auto-deselected; the "
            "overlay just makes them easy to spot."
        )
        self._aborts_btn.toggled.connect(self._on_aborts_toggled)

        self._bulk_btn = QPushButton("✎  Bulk edit…")
        self._bulk_btn.setObjectName("tb-btn")
        self._bulk_btn.setEnabled(False)
        self._bulk_btn.setToolTip(
            "Select 2+ rows (cmd-/shift-click), then click to apply one "
            "value to one column across the whole selection. Updates "
            "entities + basenames in step."
        )
        self._bulk_btn.clicked.connect(self._on_bulk_edit_clicked)

        # Busy spinner + status message — visible only while a worker
        # is running. Lives in the centre of the toolbar so it's hard
        # to miss when something's in flight.
        self._spinner = BusySpinner()
        lay.addWidget(self._spinner)

        lay.addStretch(1)

        self._settings_btn = QPushButton("⚙  Settings…")
        self._settings_btn.setObjectName("tb-btn")
        self._settings_btn.clicked.connect(self._on_settings_clicked)
        lay.addWidget(self._settings_btn)

        self._run_btn = QPushButton("▶  Run conversion")
        self._run_btn.setObjectName("tb-btn-primary")
        # Enabled once a scan has populated the model.
        self._run_btn.setEnabled(False)
        self._run_btn.setToolTip("Scan first to populate the inventory")
        self._run_btn.clicked.connect(self._on_run_clicked)
        lay.addWidget(self._run_btn)

        return bar

    # ------------------------------------------------------------------
    # Internals — panes
    # ------------------------------------------------------------------

    def _build_placeholder(self, title: str) -> QWidget:
        """Empty pane with just a header — fills the slot until a later
        milestone wires the real widget."""
        pane = QWidget()
        pane.setObjectName("pane")
        pane.setMinimumWidth(180)
        v = QVBoxLayout(pane)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader(title))
        body = QLabel("(coming in a later milestone)")
        body.setObjectName("pane-hint")
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(body, 1)
        return pane

    def _build_inspection_pane(self) -> QWidget:
        pane = QWidget()
        pane.setObjectName("pane-dark")
        v = QVBoxLayout(pane)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader("Inspection"))

        # Use QStackedWidget so we can swap between "empty placeholder"
        # and the actual table without recreating widgets — keeps view
        # focus / selection state predictable across loads.
        self._inspection_stack = QStackedWidget()

        empty = QLabel("Pick a raw-data folder and click Scan… to populate.")
        empty.setObjectName("pane-hint")
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._inspection_stack.addWidget(empty)

        self._table = self._build_table()
        self._inspection_stack.addWidget(self._table)

        v.addWidget(self._inspection_stack, 1)

        # Footer: per-pane controls that act on the table's selection.
        # "Highlight aborts" toggles row-tint on the model;
        # "Bulk edit…" enables once ≥ 2 rows are selected. Previously
        # both lived on the converter toolbar — moving them next to the
        # table puts them closer to the rows they affect.
        footer = QFrame()
        footer.setObjectName("inspection-footer")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(10, 6, 10, 6)
        fl.setSpacing(8)
        fl.addWidget(self._aborts_btn)
        fl.addStretch(1)
        fl.addWidget(self._bulk_btn)
        v.addWidget(footer)
        return pane

    def _build_table(self) -> QTableView:
        t = QTableView()
        t.setObjectName("inv-table")
        t.setShowGrid(False)
        t.setAlternatingRowColors(False)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # Allow multi-row selection (cmd/shift-click) so the user can
        # pick a batch and bulk-edit one column across all of them.
        t.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        t.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        t.verticalHeader().setVisible(False)
        t.verticalHeader().setDefaultSectionSize(26)
        header = t.horizontalHeader()
        header.setHighlightSections(False)
        header.setStretchLastSection(False)

        # Right-click anywhere on the header opens the column-visibility menu.
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._show_column_menu)

        # Per-column delegates — reuse the M1 extractions.
        for col, spec in enumerate(COLUMNS):
            if spec.role == "checkbox":
                t.setItemDelegateForColumn(col, CheckboxDelegate(t))
            elif spec.role == "status":
                t.setItemDelegateForColumn(col, StatusDelegate(t))
            else:
                t.setItemDelegateForColumn(col, CellTextDelegate(spec.role, t))
        return t

    def _apply_column_widths(self) -> None:
        header = self._table.horizontalHeader()
        for col, spec in enumerate(COLUMNS):
            self._table.setColumnWidth(col, spec.width)
            if spec.stretch:
                header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        self._apply_column_visibility()

    # ------------------------------------------------------------------
    # Column visibility (persisted via QSettings)
    # ------------------------------------------------------------------

    @staticmethod
    def _settings() -> QSettings:
        # Org/app name is set globally in ``bidsmgr.main``; here we
        # construct a default ``QSettings`` so test fixtures can
        # redirect via ``QCoreApplication.setOrganizationName``.
        return QSettings()

    def _load_column_visibility(self) -> dict[str, bool]:
        """Read per-column visibility from QSettings; fall back to defaults."""
        s = self._settings()
        out: dict[str, bool] = {}
        for spec in COLUMNS:
            key = f"inspector/columns/{spec.key}"
            stored = s.value(key, None)
            if stored is None:
                out[spec.key] = spec.default_visible
            elif isinstance(stored, str):
                out[spec.key] = stored.lower() in ("1", "true", "yes")
            else:
                out[spec.key] = bool(stored)
            # Mandatory columns are always visible regardless of stored value.
            if spec.key in MANDATORY_COLUMN_KEYS:
                out[spec.key] = True
        return out

    def _apply_column_visibility(self) -> None:
        for col, spec in enumerate(COLUMNS):
            visible = self._column_visible.get(spec.key, spec.default_visible)
            self._table.setColumnHidden(col, not visible)

    def set_column_visible(self, key: str, visible: bool) -> None:
        """Toggle a column's visibility + persist the choice."""
        if key in MANDATORY_COLUMN_KEYS:
            return
        self._column_visible[key] = visible
        self._settings().setValue(f"inspector/columns/{key}", "1" if visible else "0")
        self._apply_column_visibility()

    def _show_column_menu(self, pos) -> None:
        """Build the header right-click menu listing every column."""
        menu = QMenu(self)
        for spec in COLUMNS:
            if spec.key in MANDATORY_COLUMN_KEYS:
                # Show as a disabled item so the user understands it
                # exists but cannot be hidden.
                act = QAction(spec.header or spec.key, menu)
                act.setEnabled(False)
                act.setCheckable(True)
                act.setChecked(True)
                menu.addAction(act)
                continue
            label = spec.header or spec.key
            act = QAction(label, menu)
            act.setCheckable(True)
            act.setChecked(self._column_visible.get(spec.key, spec.default_visible))
            act.toggled.connect(lambda checked, k=spec.key: self.set_column_visible(k, checked))
            menu.addAction(act)
        # Show at the header's global coordinate.
        header = self._table.horizontalHeader()
        menu.exec(header.mapToGlobal(pos))

    # ------------------------------------------------------------------
    # Internals — bottom dock
    # ------------------------------------------------------------------

    def _build_bottom_dock(self) -> QSplitter:
        """Bottom dock split into two tabbed halves.

        Left half:  Log + Conflicts   — the "what happened" pane (live
                                         output from workers + collision
                                         warnings).
        Right half: BIDS preview + Statistics — the "what will be
                                         produced" pane (predicted tree
                                         + row counts).

        Each half is its own ``QTabWidget``; a horizontal ``QSplitter``
        between them lets the user re-balance widths.
        """
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(1)
        split.setChildrenCollapsible(False)

        # ---------- left: Log + Conflicts ----------
        left_tabs = QTabWidget()
        left_tabs.setDocumentMode(True)
        left_tabs.setMovable(False)

        # Log — append-only stream from worker progress signals.
        # Styling lives in theme.qss under ``QPlainTextEdit#dock-log``.
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setObjectName("dock-log")
        self._log_view.setMaximumBlockCount(2000)  # rolling buffer
        left_tabs.addTab(self._log_view, "📋  Log")

        conflicts = QLabel("(no conflicts detected yet)")
        conflicts.setObjectName("pane-hint")
        conflicts.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_tabs.addTab(conflicts, "⚠  Conflicts")

        split.addWidget(left_tabs)

        # ---------- right: BIDS preview + Statistics ----------
        right_tabs = QTabWidget()
        right_tabs.setDocumentMode(True)
        right_tabs.setMovable(False)

        self._bids_preview = QTreeWidget()
        self._bids_preview.setHeaderHidden(True)
        self._bids_preview.setRootIsDecorated(False)
        self._bids_preview.setIndentation(14)
        self._bids_preview.setObjectName("dock-bids-preview")
        right_tabs.addTab(self._bids_preview, "📤  BIDS preview")

        stats = QLabel("(statistics fill in after a scan)")
        stats.setObjectName("pane-hint")
        stats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stats_label = stats
        right_tabs.addTab(stats, "📊  Statistics")

        split.addWidget(right_tabs)

        # Equal split by default; user can drag the handle to rebalance.
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([700, 700])
        return split

    def _on_log_message(self, text: str) -> None:
        if not text:
            return
        # Buffer; ``_flush_log_buffer`` drains every 100ms. This keeps
        # the GUI responsive when the worker is firehose-logging.
        self._log_buffer.append(text)
        # Mirror the latest line into the spinner's status text so the
        # user has a live one-liner of what's happening.
        if self._spinner.is_busy():
            first_line = text.splitlines()[0] if text else ""
            if len(first_line) > 80:
                first_line = first_line[:77] + "…"
            self._spinner.set_message(first_line)

    def _flush_log_buffer(self) -> None:
        if not self._log_buffer:
            return
        # Single appendPlainText for the whole batch — one repaint.
        self._log_view.appendPlainText("\n".join(self._log_buffer))
        self._log_buffer.clear()

    def _rebuild_bids_preview(self) -> None:
        """Repopulate the BIDS-preview tree from the current model.

        Groups included rows by ``dataset / sub / ses / datatype`` and
        lists the basenames. Skipped rows are omitted so the preview
        reflects what conversion will actually emit.
        """
        self._bids_preview.clear()
        if self._model is None:
            return
        df = self._model.dataframe()
        if df.empty:
            return

        # tree[dataset][subject][session][datatype] = [basename, ...]
        tree: dict = {}
        for idx in df.index:
            include_val = df.at[idx, "include"] if "include" in df.columns else 1
            included = True
            if isinstance(include_val, str):
                included = include_val.strip() not in ("0", "false", "False", "")
            else:
                try:
                    included = bool(include_val)
                except Exception:
                    included = True
            if not included:
                continue
            basename = str(df.at[idx, "proposed_basename"] or "")
            if not basename:
                continue
            datatype = str(df.at[idx, "proposed_datatype"] or "")
            subject = str(df.at[idx, "BIDS_name"] or "")
            session = str(df.at[idx, "session"] or "")
            dataset = str(df.at[idx, "dataset"] or "") if "dataset" in df.columns else ""

            ds_key = dataset or "(no dataset)"
            sub_key = subject or "(no subject)"
            ses_key = session or ""
            tree.setdefault(ds_key, {}).setdefault(sub_key, {}).setdefault(
                ses_key, {}
            ).setdefault(datatype or "(no datatype)", []).append(basename)

        for ds_key in sorted(tree.keys()):
            ds_item = QTreeWidgetItem([f"{ds_key}/"])
            self._bids_preview.addTopLevelItem(ds_item)
            for sub_key in sorted(tree[ds_key].keys()):
                sub_item = QTreeWidgetItem([f"{sub_key}/"])
                ds_item.addChild(sub_item)
                for ses_key in sorted(tree[ds_key][sub_key].keys()):
                    ses_parent = sub_item
                    if ses_key:
                        ses_item = QTreeWidgetItem([f"{ses_key}/"])
                        sub_item.addChild(ses_item)
                        ses_parent = ses_item
                    for dt_key in sorted(tree[ds_key][sub_key][ses_key].keys()):
                        dt_item = QTreeWidgetItem([f"{dt_key}/"])
                        ses_parent.addChild(dt_item)
                        for bn in sorted(tree[ds_key][sub_key][ses_key][dt_key]):
                            dt_item.addChild(QTreeWidgetItem([bn]))
        self._bids_preview.expandToDepth(2)

    def _update_stats_label(self) -> None:
        if self._model is None:
            self._stats_label.setText("(statistics fill in after a scan)")
            return
        df = self._model.dataframe()
        n_rows = len(df)
        # Unique subjects + sessions across all rows.
        subs = sorted({str(s) for s in df.get("BIDS_name", []) if s})
        ses = sorted({
            f"{s}/{x}" for s, x in zip(
                df.get("BIDS_name", []), df.get("session", []),
            ) if s and x
        })
        self._stats_label.setText(
            f"{n_rows} rows · {len(subs)} subjects · {len(ses)} session(s)"
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_scan_clicked(self) -> None:
        # 1. Raw input is required.
        if self._raw_root is None:
            self._on_pick_raw_dir()
            if self._raw_root is None:
                return
        # 2. BIDS output is required (the scan TSV lives under it now).
        if self._bids_parent is None:
            self._on_pick_bids_parent()
            if self._bids_parent is None:
                return
        # 3. Compute the scan-TSV path from the filename input + the
        # BIDS output folder.
        filename = self._tsv_filename_edit.text().strip() or "inventory.tsv"
        if not filename.lower().endswith(".tsv"):
            filename += ".tsv"
        self._output_tsv = self._bids_parent / filename
        # 4. Pick up the latest persisted settings on every scan so a
        # user who changed defaults in the Settings dialog without
        # restarting still gets the new values. The scan stamps every
        # row's ``dataset`` column with a slug derived from the raw
        # folder name; the user partitions rows across datasets by
        # editing that column in the inspector after the scan.
        s = AppSettings.load()
        self.start_scan(
            self._raw_root,
            self._output_tsv,
            dataset=(s.dataset_slug or None),
            line_freq=s.scan_line_freq if s.scan_line_freq else None,
            montage=s.scan_montage or None,
            n_jobs=s.scan_n_jobs,
            probe_convert=s.scan_probe_convert,
            skip_bids_guess=s.scan_skip_bids_guess,
        )

    def _on_tsv_filename_edited(self) -> None:
        """Persist the filename so it survives across sessions."""
        text = self._tsv_filename_edit.text().strip()
        if not text:
            return
        if not text.lower().endswith(".tsv"):
            text += ".tsv"
            self._tsv_filename_edit.setText(text)
        AppSettings.remember_tsv_filename(text)

    def _on_pick_raw_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Pick raw-data folder",
            str(self._raw_root or Path.home()),
        )
        if not d:
            return
        self._raw_root = Path(d)
        self._raw_pathbar.set_value(str(self._raw_root), ok=True)
        AppSettings.remember_raw_root(self._raw_root)
        # Live preview of the picked folder before any scan runs.
        self._raw_pane.set_root(self._raw_root)

    def _on_pick_bids_parent(self) -> None:
        """Pick the BIDS output parent dir at any time (before or after scan)."""
        start = str(self._bids_parent or self._raw_root or Path.home())
        d = QFileDialog.getExistingDirectory(self, "Pick BIDS output folder", start)
        if not d:
            return
        self._bids_parent = Path(d)
        self._bids_pathbar.set_value(str(self._bids_parent), ok=True)
        AppSettings.remember_bids_parent(self._bids_parent)
        # Live preview of the picked folder before any convert runs.
        self._output_pane.set_root(self._bids_parent)

    def _on_run_clicked(self) -> None:
        if self._model is None or self._output_tsv is None:
            return
        if self._convert_worker is not None and self._convert_worker.isRunning():
            return
        # Use the pre-set BIDS output if the user picked one; otherwise
        # prompt now.
        if self._bids_parent is None:
            self._on_pick_bids_parent()
            if self._bids_parent is None:
                return
        self.start_convert(self._bids_parent)

    def start_convert(self, bids_parent: Path, *, n_jobs: Optional[int] = None) -> ConvertWorker:
        """Run conversion against the live model + scan TSV. Returns the worker."""
        assert self._model is not None and self._output_tsv is not None, (
            "start_convert requires a populated model + output_tsv"
        )
        # Refresh persistent settings so a Settings-dialog save just
        # before clicking Run takes effect this run.
        self._app_settings = AppSettings.load()
        if n_jobs is None:
            n_jobs = self._app_settings.convert_n_jobs
        self._run_btn.setEnabled(False)
        self._scan_btn.setEnabled(False)
        self._spinner.set_busy(True, message="Converting…")
        worker = ConvertWorker(
            self._model.dataframe(),
            self._output_tsv,
            bids_parent,
            n_jobs=n_jobs,
            overwrite=self._app_settings.convert_overwrite,
            raw_root=self._raw_root,
            line_freq=self._app_settings.scan_line_freq or None,
            montage=self._app_settings.scan_montage or None,
            parent=self,
        )
        worker.progress.connect(self._on_progress)
        worker.finished_with_result.connect(self._on_convert_finished)
        worker.failed.connect(self._on_convert_failed)
        worker.finished.connect(self._on_convert_worker_qt_finished)
        self._convert_worker = worker
        worker.start()
        return worker

    def _on_convert_finished(self, rc: int, bids_parent: Path) -> None:
        if self._project is not None:
            self._project.append(StageCompleted(
                stage="convert",
                success=(rc == 0),
                summary={"bids_parent": str(bids_parent), "returncode": int(rc)},
            ))
        # Re-walk the output tree so the user sees the just-converted
        # files appear in the lower half of col 1.
        self._output_pane.set_root(bids_parent)
        self.convert_finished.emit(rc, bids_parent)

        # Post-convert chain. Only kicks off when convert succeeded
        # cleanly — a partial failure (rc != 0) shows the dialog and
        # leaves metadata / validate to the user (they likely need to
        # fix the failed subjects first).
        ran_post = False
        if rc == 0:
            ran_post = self._maybe_run_post_convert(bids_parent)
        if ran_post:
            return  # the dialog fires after the post-convert chain ends

        self._show_convert_dialog(rc, bids_parent)

    def _show_convert_dialog(self, rc: int, bids_parent: Path) -> None:
        msg = QMessageBox(self)
        if rc == 0:
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle("Conversion complete")
            msg.setText(f"Output written to: {bids_parent}")
        else:
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Conversion finished with errors")
            msg.setText(
                f"{rc} subject(s) failed. See "
                f"<bids_root>/.bidsmgr/errors/ for per-subject logs."
            )
        msg.exec()

    # ------------------------------------------------------------------
    # Post-convert chain
    # ------------------------------------------------------------------

    def _maybe_run_post_convert(self, bids_parent: Path) -> bool:
        """Kick off the metadata + validate chain if settings say so.

        Returns ``True`` if anything was started. The chain is:
        metadata → validate → dialog. Each step waits for the previous
        worker's ``finished`` before starting; failures fall through to
        the dialog with a warning instead of stopping the chain.
        """
        s = self._app_settings
        if not (s.post_run_metadata or s.post_run_validate):
            return False

        self._post_chain_bids_parent = bids_parent
        # State machine: track which steps are still pending so the
        # chain handler knows what to dispatch next.
        self._post_chain_pending: list[str] = []
        if s.post_run_metadata:
            self._post_chain_pending.append("metadata")
        if s.post_run_validate:
            self._post_chain_pending.append("validate")
        self._post_chain_dispatch()
        return True

    def _post_chain_dispatch(self) -> None:
        if not self._post_chain_pending:
            self._spinner.set_busy(False)
            self._show_convert_dialog(0, self._post_chain_bids_parent)
            return
        step = self._post_chain_pending.pop(0)
        if step == "metadata":
            self._spinner.set_busy(True, message="Generating metadata…")
            self._start_metadata_worker(self._post_chain_bids_parent)
        elif step == "validate":
            self._spinner.set_busy(True, message="Validating BIDS tree…")
            self._start_validate_worker(self._post_chain_bids_parent)

    def _start_metadata_worker(self, bids_parent: Path) -> None:
        s = self._app_settings
        worker = MetadataWorker(
            bids_parent,
            inventory_tsv=self._output_tsv,
            # Metadata-engine name defaults to each BIDS root's folder
            # name (i.e. the dataset slug) when ``name`` is None.
            name=None,
            fill_todos=s.post_metadata_fill_todos,
            parent=self,
        )
        worker.progress.connect(self._on_progress)
        worker.finished_with_result.connect(self._on_post_finished)
        worker.failed.connect(self._on_post_failed)
        self._metadata_worker = worker
        worker.start()

    def _start_validate_worker(self, bids_parent: Path) -> None:
        s = self._app_settings
        worker = ValidateWorker(
            bids_parent,
            strict=s.post_validate_strict,
            html_report=s.post_validate_html,
            parent=self,
        )
        worker.progress.connect(self._on_progress)
        worker.finished_with_result.connect(self._on_post_finished)
        worker.failed.connect(self._on_post_failed)
        self._validate_worker = worker
        worker.start()

    def _on_post_finished(self, rc: int, target: Path) -> None:
        # Record the project event for this stage.
        if self._project is not None:
            # Figure out which stage this came from.
            sender = self.sender()
            stage = (
                "metadata" if isinstance(sender, MetadataWorker)
                else "validate"
            )
            self._project.append(StageCompleted(
                stage=stage,
                success=(rc == 0),
                summary={"target": str(target), "returncode": int(rc)},
            ))
        self._post_chain_dispatch()

    def _on_post_failed(self, tb: str) -> None:
        self.log_message.emit(tb)
        # Continue the chain even if one step crashes; the user can
        # inspect the Log tab for the traceback.
        self._post_chain_dispatch()

    def _on_convert_failed(self, tb: str) -> None:
        self.log_message.emit(tb)
        QMessageBox.critical(self, "Conversion crashed", tb)

    def _on_convert_worker_qt_finished(self) -> None:
        self._run_btn.setEnabled(self._model is not None and self._model.rowCount() > 0)
        self._scan_btn.setEnabled(True)
        self._convert_worker = None
        # Spinner stays on if a post-convert worker (metadata / validate)
        # is queued; ``_post_chain_dispatch`` handles its own messaging.
        if not (self._metadata_worker and self._metadata_worker.isRunning()) \
                and not (self._validate_worker and self._validate_worker.isRunning()):
            if not getattr(self, "_post_chain_pending", []):
                self._spinner.set_busy(False)

    def _on_progress(self, message: str) -> None:
        self.log_message.emit(message)

    def repaint_for_palette(self, pal: dict) -> None:
        """Re-render any widget whose colors are read at construction time.

        Called by ``MainWindow`` when ``ThemeManager.apply`` fires its
        listeners on every dark↔light swap. QSS swap handles the
        majority of widgets automatically; this method is for the
        per-pane fallbacks (PropertiesPanel rebuild, side panes' tree
        items).
        """
        if hasattr(self, "_properties"):
            self._properties.repaint_for_palette(pal)
        if hasattr(self, "_raw_pane"):
            self._raw_pane.repaint_for_palette(pal)
        if hasattr(self, "_output_pane"):
            self._output_pane.repaint_for_palette(pal)
        if hasattr(self, "_filter_pane"):
            self._filter_pane.repaint_for_palette(pal)

    def _on_aborts_toggled(self, checked: bool) -> None:
        """Apply the toggle to the active model + persist for next launch."""
        AppSettings.remember_highlight_aborts(checked)
        if self._model is not None:
            self._model.set_highlight_aborts(checked)

    def _open_issues_dialog(self, severity: str) -> None:
        """Open a modeless dialog listing every row with the given state.

        Severity is one of ``"warn"`` / ``"err"`` / ``"skip"``.
        Activating a row in the dialog (double-click or Enter) selects
        the matching row in the inspection table so the user can jump
        to the Properties panel and fix it.
        """
        if self._model is None or self._model.rowCount() == 0:
            return
        from .issues_dialog import IssuesDialog
        dlg = IssuesDialog(self._model, severity, parent=self)
        dlg.row_selected.connect(self._jump_to_row)
        dlg.show()

    def _jump_to_row(self, row: int) -> None:
        """Select ``row`` in the inspection table and scroll it into view."""
        if self._model is None:
            return
        if not (0 <= row < self._model.rowCount()):
            return
        self._table.selectRow(row)
        self._table.scrollTo(
            self._model.index(row, 0),
            self._table.ScrollHint.PositionAtCenter,
        )

    def _on_settings_clicked(self) -> None:
        from .settings_dialog import SettingsDialog
        # Pass a fresh load so the dialog sees the latest persisted values.
        s = AppSettings.load()
        dlg = SettingsDialog(s, self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            self._app_settings = s
            # Forward the theme change to the listener registered by
            # MainWindow (which owns the ThemeManager). Done via the
            # parent's ``apply_theme`` if present so the panel stays
            # decoupled from QApplication.
            apply_fn = getattr(self.window(), "apply_theme", None)
            if callable(apply_fn):
                apply_fn(s.theme)

    def _on_current_row_changed(self, current, _previous) -> None:
        """Forward the table's row selection to the Properties panel."""
        if not current.isValid():
            self._properties.set_selected_row(None)
            return
        self._properties.set_selected_row(current.row())

    def _on_selection_changed(self, *_args) -> None:
        """Enable the bulk-edit button only when ≥ 2 rows are selected."""
        rows = self._selected_rows()
        self._bulk_btn.setEnabled(len(rows) >= 2)
        if len(rows) >= 2:
            self._bulk_btn.setToolTip(
                f"Apply one value to one column across the {len(rows)} "
                "selected rows."
            )

    def _selected_rows(self) -> list[int]:
        """Return source-DataFrame row indices for the user's selection."""
        sel = self._table.selectionModel()
        if sel is None:
            return []
        # ``selectedRows`` is one ``QModelIndex`` per selected row (col 0).
        return sorted({idx.row() for idx in sel.selectedRows()})

    def _on_bulk_edit_clicked(self) -> None:
        if self._model is None:
            return
        rows = self._selected_rows()
        if len(rows) < 2:
            return
        from .bulk_edit_dialog import BulkEditDialog
        dlg = BulkEditDialog(self._model, rows, parent=self)
        dlg.exec()
        # Status chips + previews refresh via the model's dataChanged
        # signal — the dispatcher's per-row writes already trigger them.

    def _on_scan_finished(self, df: pd.DataFrame, output_tsv: Path) -> None:
        self.load_inventory(df, output_tsv)
        if self._project is not None:
            row_ids = self._collect_row_ids(df)
            self._project.append(ScanImported(
                inventory_tsv=str(output_tsv),
                row_ids=tuple(row_ids),
            ))
            self._project.append(StageCompleted(
                stage="scan",
                success=True,
                summary={
                    "raw_root": str(self._raw_root) if self._raw_root else "",
                    "inventory_tsv": str(output_tsv),
                    "rows": int(len(df)),
                },
            ))
        self.scan_finished.emit(df, output_tsv)

    @staticmethod
    def _collect_row_ids(df: pd.DataFrame) -> list[str]:
        """Best-effort stable row IDs for the project log.

        Mirrors :meth:`InventoryTableModel.row_id`: prefer
        ``series_uid``, fall back to ``source_file``, then ``row-N``.
        """
        out: list[str] = []
        for i, row in df.reset_index(drop=True).iterrows():
            uid = str(row.get("series_uid", "") or "").strip()
            if uid:
                out.append(uid)
                continue
            src = str(row.get("source_file", "") or "").strip()
            if src:
                out.append(src)
                continue
            out.append(f"row-{i}")
        return out

    def _on_scan_failed(self, tb: str) -> None:
        self.log_message.emit(tb)
        QMessageBox.critical(self, "Scan failed", tb)

    def _on_worker_qt_finished(self) -> None:
        # Re-enable the Scan button once the worker's event loop wraps
        # up (this fires after both success and failure paths).
        self._scan_btn.setEnabled(True)
        self._scan_worker = None
        self._spinner.set_busy(False)

    # ------------------------------------------------------------------
    # Status chips
    # ------------------------------------------------------------------

    def _update_status_chips(self) -> None:
        """Recount severities from the active model and refresh chips."""
        if self._model is None:
            return
        counts = {"valid": 0, "warn": 0, "err": 0, "skip": 0}
        # Read row state via the public model API rather than poking
        # the cached list — avoids the test relying on internals.
        from .delegates import ROW_STATE_ROLE
        for row in range(self._model.rowCount()):
            state = self._model.data(self._model.index(row, 0), ROW_STATE_ROLE) or ""
            if state == "err":
                counts["err"] += 1
            elif state == "warn":
                counts["warn"] += 1
            elif state == "skip":
                counts["skip"] += 1
            else:
                counts["valid"] += 1
        # Rebuild chips in place — Qt's QLabel doesn't have a chip API.
        self._chip_valid.setText(f"{counts['valid']} valid")
        self._chip_warn.setText(f"{counts['warn']} warnings")
        self._chip_err.setText(f"{counts['err']} error")
        self._chip_skip.setText(f"{counts['skip']} skipped")


__all__ = ["ConverterPanel"]
