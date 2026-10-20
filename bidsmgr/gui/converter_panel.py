"""The Converter view — the pre-conversion half of the Inspector layout.

Reference: ``inspector_proto/proto.py`` ``ConverterView``,
``gui_mockups.html`` proposal 1.

M3 scope: this lands the **toolbar + path bars + Inspection table**
backed by a real :class:`InventoryTableModel`. The other three panes
in the prototype (raw-FS tree, filter tree, properties panel) and the
bottom dock (BIDS preview / Log / Stats) are stubbed as
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
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QToolButton,
    QPushButton,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..inventory.rebuild import InventoryVersionError
from ..project import Project, ScanImported, StageCompleted
from ..workers import ConvertWorker, MetadataWorker, ScanWorker, ValidateWorker
from . import icons
from .theme_manager import CUR
from .app_settings import AppSettings
from .delegates import (
    CellTextDelegate,
    CheckboxDelegate,
    ChoiceDelegate,
    StatusDelegate,
    builtin_montages,
)
from .filter_pane import FilterPane
from .models import COLUMNS, MANDATORY_COLUMN_KEYS, InventoryTableModel
from .output_fs_pane import OutputFsPane
from .properties_panel import PropertiesPanel
from .raw_fs_pane import RawFsPane
from .overview_pane import OverviewPane
from .widgets import BusySpinner, Chip, PaneHeader, PanelFrame, PathBar, VSep
from .widgets.panel_frame import CARD_GAP_PX, HEADER_EXTRAS, HEADER_LEAD, card_canvas
from .widgets.primitives import Segmented
from .widgets.splitter import Splitter

log = logging.getLogger(__name__)


#: Room a header needs beyond its text: the cell margins either side plus
#: the sort indicator Qt draws inside the section. In pixels because Qt
#: section widths are, but the TEXT is measured with font metrics, so this
#: stays right at any font scale (see cross-platform rule 4.1).
_HEADER_PADDING_PX = 26


#: Which view the Inspection area last showed: "table" or "overview".
_INSPECTION_VIEW_KEY = "inspector/view"


def _mtime(path) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def _repolish_combo(combo: QComboBox) -> None:
    """Force a ``QComboBox`` (and its popup view) to re-read the stylesheet.

    A global ``QApplication.setStyleSheet`` swap (the theme toggle) does not
    re-polish already-shown widgets, and a combo's drop-down view is a separate
    top-level widget that Qt never re-polishes at all. Without the unpolish /
    polish dance below, the closed combo and (worse) its popup keep stale
    dark-theme colours after switching to light, and vice versa.

    GENERAL RULE: any ``QComboBox`` whose colours come from QSS must be run
    through this on every palette swap (call it from the owner's
    ``repaint_for_palette``). Re-tinting icons is not enough.
    """
    if combo is None:
        return
    style = combo.style()
    targets = [combo]
    view = combo.view()
    if view is not None:
        targets.append(view)
        viewport = view.viewport()
        if viewport is not None:
            targets.append(viewport)
    for w in targets:
        style.unpolish(w)
        style.polish(w)
        w.update()



def _native(key: QKeySequence.StandardKey) -> str:
    """A standard shortcut as this OS writes it (Cmd+Z on macOS)."""
    return QKeySequence(key).toString(QKeySequence.SequenceFormat.NativeText)

class ConverterPanel(QWidget):
    """The pre-conversion Inspector layout.

    Owns one :class:`InventoryTableModel` at a time. ``start_scan`` /
    ``load_inventory`` replace the model; the table view re-binds.
    """

    log_message = pyqtSignal(str)
    scan_finished = pyqtSignal(object, object)
    convert_finished = pyqtSignal(int, object)
    # (current root, the new folder name) when the user asks for the dataset
    # name change to rename the project itself. The window does the move: this
    # panel knows what was asked, not how to close and reopen a project.
    project_rename_requested = pyqtSignal(object, str)

    def __init__(self, project: Optional[Project] = None, parent=None) -> None:
        super().__init__(parent)
        self._project = project
        self._model: Optional[InventoryTableModel] = None
        self._scan_worker: Optional[ScanWorker] = None
        self._convert_worker: Optional[ConvertWorker] = None
        self._metadata_worker: Optional[MetadataWorker] = None
        self._validate_worker: Optional[ValidateWorker] = None
        # The folder the Raw input bar shows: what Scan reads next. With a
        # table open it is that table's source until another folder is picked.
        self._raw_root: Optional[Path] = None
        # Where the data of the TABLE SHOWN is now (``inventory.sources``),
        # and what the last look for it found. The preview, the PSD and the
        # conversion read from here, never from a folder picked to scan next.
        self._sources = None
        self._source_check = None
        self._source_action = ""
        self._source_jobs = None
        self._source_generation = 0
        self._output_tsv: Optional[Path] = None
        self._bids_parent: Optional[Path] = None
        # Set when an explicit project is bound (Welcome -> Open/Create). The
        # output is then locked to this dataset root and the inventory lives in
        # the project bundle. ``None`` keeps the free-path (pick-output) flow.
        self._bids_root: Optional[Path] = None
        # The scan version currently loaded/active (``.bidsmgr/project/scans/
        # <version>/``). Each scan of a source folder is its own version with an
        # isolated curation history; ``self._project`` is that version's bundle.
        self._active_version_dir: Optional[Path] = None
        # In-memory redo stack for the active version: events popped by Undo,
        # re-appended by Redo. Cleared when a new edit diverges the timeline or
        # a different version is loaded (not persisted across reopen).
        self._redo_stack: list = []
        # Persistent settings — loaded fresh each construction so a
        # Settings dialog save propagates on the next open of this panel.
        self._app_settings: AppSettings = AppSettings.load()
        from .delegates.inspection_cells import set_confidence_bands

        _general = self._app_settings.classifier().general
        set_confidence_bands(_general.apply_at, _general.check_at)
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
        self._raw_pathbar.action_button.clicked.connect(self._on_source_action)
        v.addWidget(self._raw_pathbar)
        # The data can be moved, or its drive unplugged, while the app is in
        # the background: look again when it comes back to the front.
        app = QApplication.instance()
        if app is not None:
            app.applicationStateChanged.connect(self._on_application_state)

        bids_init = str(self._bids_parent) if self._bids_parent else "(not set)"
        self._bids_pathbar = PathBar(
            "BIDS output", bids_init, ok=bool(self._bids_parent),
        )
        self._bids_pathbar.change_button.clicked.connect(self._on_pick_bids_parent)
        v.addWidget(self._bids_pathbar)

        # ---------------- main vertical splitter ----------------
        # Top: 4-col horizontal splitter (panes). Bottom: tabbed dock.
        # The panels are cards on the window's canvas: the two top-level
        # splitters leave a gap between them (``card-split``); the ones
        # inside a card keep a hairline.
        v_split = Splitter(Qt.Orientation.Vertical)
        v_split.setObjectName("card-split")
        v_split.setHandleWidth(CARD_GAP_PX)
        v_split.setChildrenCollapsible(False)

        h_split = Splitter(Qt.Orientation.Horizontal)
        h_split.setObjectName("card-split")
        h_split.setHandleWidth(CARD_GAP_PX)
        h_split.setChildrenCollapsible(False)

        # Every side region is wrapped in a ``PanelFrame`` so it can be
        # collapsed toward its edge and detached into a floating window.
        # Collapsing a region hands its space to the inspection unit (the
        # work surface), which always carries the stretch.
        #   * Raw + Output trees: fold up individually (edge="top"); the
        #     whole left column also folds left as one (col1_frame).
        #   * Filter: folds left.   Properties: folds right.
        #   * Inspection + Properties detach together as one unit.
        self._panel_frames: list[PanelFrame] = []

        # Left column: raw tree over output tree, each individually
        # collapsible (fold up) + detachable, wrapped again so the whole
        # column folds to the left as one.
        self._raw_pane = RawFsPane()
        self._raw_pane.file_activated.connect(self._on_raw_file_activated)
        self._output_pane = OutputFsPane()
        self._raw_frame = PanelFrame(self._raw_pane, "Raw data tree", edge="top")
        self._output_frame = PanelFrame(self._output_pane, "Output data tree", edge="top")
        col1_inner = Splitter(Qt.Orientation.Vertical)
        col1_inner.setHandleWidth(1)
        col1_inner.setChildrenCollapsible(False)
        col1_inner.addWidget(self._raw_frame)
        col1_inner.addWidget(self._output_frame)
        col1_inner.setStretchFactor(0, 1)
        col1_inner.setStretchFactor(1, 1)
        col1_inner.setSizes([320, 320])
        self._raw_frame.attach_splitter(col1_inner)
        self._output_frame.attach_splitter(col1_inner)
        self._col1_frame = PanelFrame(
            col1_inner, "Raw / Output trees", edge="left",
            detachable=False, hide_inner_header=False,
        )
        h_split.addWidget(self._col1_frame)

        self._filter_pane = FilterPane()
        self._filter_pane.row_chosen.connect(self._jump_to_row)
        self._filter_pane.rows_menu_requested.connect(self._on_filter_menu)
        self._filter_pane.preview.set_context(self._plan_context)
        # Its title says which of its three views it shows.
        self._filter_frame = PanelFrame(self._filter_pane, self._filter_pane.mode_title(),
                                        edge="left")
        self._filter_pane.mode_changed.connect(lambda _m, title: self._filter_frame.set_title(title))
        h_split.addWidget(self._filter_frame)

        # Inspection (work surface, never collapsible) + Properties (folds
        # right) live in one horizontal sub-splitter; the surrounding frame
        # detaches both together as a single unit.
        self._properties = PropertiesPanel()
        self._properties.set_project(self._project)
        self._properties.renumber_requested.connect(self.open_renumber)
        self._properties_frame = PanelFrame(
            self._properties, "Properties", edge="right", detachable=False,
        )
        right_inner = Splitter(Qt.Orientation.Horizontal)
        right_inner.setHandleWidth(1)
        right_inner.setChildrenCollapsible(False)
        self._inspection_pane = self._build_inspection_pane()
        right_inner.addWidget(self._inspection_pane)
        right_inner.addWidget(self._properties_frame)
        right_inner.setStretchFactor(0, 1)
        right_inner.setStretchFactor(1, 0)
        right_inner.setSizes([720, 320])
        # Collapsing Properties hands its width to the inspection table.
        self._properties_frame.attach_splitter(right_inner, grow_target=self._inspection_pane)
        self._inspection_unit = PanelFrame(
            right_inner, "Inspection", collapsible=False, detachable=True,
        )
        # Floating, the Inspection has no title bar and its view switch sits
        # back in the pane's own row, which then needs room around it.
        self._inspection_unit.state_changed.connect(self._fit_view_switch_home)
        h_split.addWidget(self._inspection_unit)

        # Collapsing the left column or the filter hands width to the
        # inspection unit (the work surface), which grows automatically.
        self._col1_frame.attach_splitter(h_split, grow_target=self._inspection_unit)
        self._filter_frame.attach_splitter(h_split, grow_target=self._inspection_unit)
        self._panel_frames = [
            self._raw_frame, self._output_frame, self._col1_frame,
            self._filter_frame, self._properties_frame, self._inspection_unit,
        ]
        h_split.setStretchFactor(0, 0)
        h_split.setStretchFactor(1, 0)
        h_split.setStretchFactor(2, 1)
        h_split.setSizes([240, 220, 1040])
        v_split.addWidget(h_split)

        # The Log, under the columns: one card, folded from anywhere on its
        # header (the title bar sits on top of it like every other card's;
        # folded, only that bar is left at the bottom of the window), and
        # detachable. What the conversion will write is the Inspection's
        # Overview, beside the table.
        self._log_frame = PanelFrame(
            self._build_log_dock(), "Log", edge="top", collapsible=True, detachable=True,
        )
        self._panel_frames.append(self._log_frame)
        v_split.addWidget(self._log_frame)
        # Folding the Log hands its height to the panes above.
        self._log_frame.attach_splitter(v_split, grow_target=h_split)
        v_split.setStretchFactor(0, 1)
        v_split.setStretchFactor(1, 0)
        v_split.setSizes([600, 160])
        for frame in (self._col1_frame, self._filter_frame, self._inspection_unit,
                      self._log_frame):
            frame.make_card()
        v.addWidget(card_canvas(v_split), 1)

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

        # Project-first gate: with no active project the Converter starts reset
        # and locked (raw input / BIDS output / Scan disabled with a reason) so
        # a stale path or table from a previous session is never acted on. A
        # project passed to the constructor (the --project launch) is bound by
        # MainWindow via set_project, which un-gates.
        if self._project is None:
            self._set_no_project_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def model(self) -> Optional[InventoryTableModel]:
        """Return the active model (``None`` before the first scan)."""
        return self._model

    _NO_PROJECT_HINT = (
        "Open or create a dataset on the Home tab to start.\n\n"
        "The Converter always works inside a project: scanning, the raw input, "
        "and the locked BIDS output are enabled once a dataset is open."
    )
    _NO_PROJECT_TOOLTIP = (
        "Open or create a dataset on the Home tab first. The Converter works "
        "inside a project so nothing is scanned or written outside it."
    )

    def _set_no_project_state(self) -> None:
        """Gate the Converter until a project is opened/created.

        The project-first entry point: with no active project the Converter is
        reset to empty and the raw-input / BIDS-output / Scan controls are
        disabled (with a reason), so a stale path or table from a previous
        session can never be acted on. ``set_project`` re-enables them. The
        Editor is intentionally NOT gated (it doubles as a free viewer).
        """
        self._project = None
        self._bids_root = None
        self._bids_parent = None
        self._raw_root = None
        self._reset_inventory_view()
        # Path bars: clear + lock, with a reason on the buttons.
        self._raw_pathbar.set_value("(open or create a dataset first)", ok=False)
        self._clear_source_state()
        self._raw_pathbar.change_button.setEnabled(False)
        self._raw_pathbar.change_button.setToolTip(self._NO_PROJECT_TOOLTIP)
        self._bids_pathbar.set_value("(locked to the dataset you open)", ok=False)
        self._bids_pathbar.change_button.setEnabled(False)
        self._bids_pathbar.change_button.setToolTip(self._NO_PROJECT_TOOLTIP)
        # Scan needs a destination project; disable it too.
        self._scan_btn.setEnabled(False)
        self._scan_btn.setToolTip(self._NO_PROJECT_TOOLTIP)
        # Output tree shows nothing until a project is bound.
        self._output_pane.set_root(None)
        # Inspection placeholder explains the gate.
        if hasattr(self, "_inspection_empty"):
            self._inspection_empty.setText(self._NO_PROJECT_HINT)
        self._refresh_scans_combo()

    def set_project(self, project: Project, bids_root: Path) -> None:
        """Bind an open project and lock the output to its dataset root.

        Soft lock: the BIDS-output bar shows the dataset root and its
        ``change…`` button is disabled with a tooltip pointing at the Home tab,
        rather than a free folder picker (so the user cannot scatter subjects
        outside the project). The inventory + recording-metadata scaffold then
        live in the project bundle and the dataset name is the folder name.
        """
        # Switching datasets must NOT leak the previous project's rows / raw
        # input / metadata into the new tab. Clear the converter to its empty
        # state first; the resume below repopulates only if THIS project has a
        # scan of its own.
        self._reset_inventory_view()

        self._project = project
        self._bids_root = Path(bids_root)
        self._bids_parent = self._bids_root.parent
        self._properties.set_project(project)

        self._bids_pathbar.set_value(str(self._bids_root), ok=True)
        self._bids_pathbar.change_button.setEnabled(False)
        self._bids_pathbar.change_button.setToolTip(
            "Working inside this project. Use the Home tab to open or create a "
            "different dataset."
        )
        self._output_pane.set_root(self._bids_root)

        # Un-gate the project-mode controls: raw input is modifiable (you pick
        # what to scan), Scan is enabled, and the placeholder reverts to the
        # normal prompt. The BIDS-output bar stays locked above.
        self._raw_pathbar.change_button.setEnabled(True)
        self._raw_pathbar.change_button.setToolTip("")
        self._scan_btn.setEnabled(True)
        self._scan_btn.setToolTip("")
        if hasattr(self, "_inspection_empty"):
            self._inspection_empty.setText(
                "Pick a raw-data folder and click Scan… to populate."
            )

        # Resume: if this project already holds a scan, reload its inventory and
        # replay the curation edits so the user lands back in their table.
        self._resume_from_project()

    def _reset_inventory_view(self) -> None:
        """Return the converter to its pre-scan empty state.

        Called when binding a new project so one dataset's inventory, raw
        input, recording-metadata scaffold and curation history never bleed
        into another's tab. The output path bar is set by :meth:`set_project`
        right after this.
        """
        self._model = None
        self._active_version_dir = None
        self._raw_root = None
        self._output_tsv = None
        self._redo_stack.clear()
        # Detach the model from every dependent view.
        self._table.setModel(None)
        self._properties.bind_model(None)
        self._sources = None
        self._source_check = None
        self._properties.set_sources(None)
        self._filter_pane.bind_model(None)
        self._overview.bind_model(None)
        self._raw_pane.bind_model(None)
        self._raw_pane.set_root(None)
        # Raw input bar back to "nothing picked".
        self._raw_pathbar.set_value("(no folder selected)", ok=False)
        self._clear_source_state()
        # Inspection back to the placeholder; clear the chips.
        self._inspection_stack.setCurrentIndex(0)
        self._chip_valid.setText("0 valid")
        self._chip_warn.setText("0 warnings")
        self._chip_err.setText("0 error")
        self._chip_skip.setText("0 skipped")
        self._run_btn.setEnabled(False)
        self._revalidate_btn.setEnabled(False)

    def _resume_from_project(self) -> None:
        """Resume the project's most recent scan version (curation resume).

        No-op when the project has no scan yet. Each scan of a source folder is
        its own version with an isolated curation bundle; this loads the latest
        and refreshes the version picker. Older versions stay reachable via the
        toolbar Scan picker.
        """
        if self._bids_root is None:
            return
        from ..project import workspace
        versions = workspace.list_versions(self._bids_root)
        if versions:
            self._load_version(versions[-1])
        self._refresh_scans_combo()

    def _load_version(self, version, *, clear_redo: bool = True) -> None:
        """Open one scan version: load its inventory + replay its edit history.

        ``clear_redo`` is False only for the same-version reloads that Undo / Redo
        trigger, so the redo stack survives across them; switching to a different
        version (or a fresh scan) clears it.
        """
        from ..project import Project
        if clear_redo:
            self._redo_stack.clear()
        try:
            proj = Project.open(version.dir)
            df = pd.read_csv(version.inventory, sep="\t", dtype=str).fillna("")
        except Exception as exc:  # never block; leave the current view in place
            self.log_message.emit(
                f"Could not load scan '{version.version_id}': {exc}"
            )
            return
        # Keep the dataset column in lock-step with the actual (possibly
        # renamed-on-disk) project folder. Conversion writes into
        # ``<bids_parent>/<dataset>``; if the user renamed the dataset folder
        # outside the app, the stored value would be stale and convert would
        # target the wrong folder. The column is read-only in the GUI, so this
        # is the single source of truth.
        if self._bids_root is not None and "dataset" in df.columns:
            df["dataset"] = self._bids_root.name
        self._project = proj
        self._properties.set_project(proj)
        self._active_version_dir = Path(version.dir)
        # This scan's own source: never the folder another scan read. Where
        # the data really is now, and whether it is still there, is decided
        # when the table loads (``_resolve_sources``).
        self._raw_root = Path(version.raw_root) if version.raw_root else None
        # The model is rebuilt with this version's project, so its event overlay
        # (cell / entity / include edits) replays automatically.
        try:
            self.load_inventory(df, output_tsv=version.inventory)
        except InventoryVersionError as exc:
            # A scan saved before the columns were renamed. Say what to do
            # rather than letting a KeyError out of a slot, which PyQt turns
            # into an abort.
            self.log_message.emit(str(exc))
            QMessageBox.information(
                self, "This scan needs re-running", str(exc),
            )
            return
        self.log_message.emit(f"Resumed scan '{version.version_id}'")

    def _refresh_scans_combo(self) -> None:
        """Repopulate the toolbar scan-version picker (project mode only)."""
        combo = getattr(self, "_scans_combo", None)
        if combo is None:
            return
        in_project = self._bids_root is not None
        # In project mode the free-path TSV-filename field is irrelevant.
        self._tsv_label.setVisible(not in_project)
        self._tsv_filename_edit.setVisible(not in_project)

        # Undo / Redo are project-mode curation actions (act on the active
        # version's event log); show them once a table is loaded.
        show_edit = in_project and self._model is not None
        self._undo_btn.setVisible(show_edit)
        self._redo_btn.setVisible(show_edit)
        self._redo_btn.setEnabled(bool(self._redo_stack))

        if not in_project:
            self._scans_label.setVisible(False)
            combo.setVisible(False)
            return

        from ..project import workspace
        versions = workspace.list_versions(self._bids_root)
        combo.blockSignals(True)
        combo.clear()
        active = 0
        for i, v in enumerate(versions):
            combo.addItem(v.version_id, userData=str(v.dir))
            if self._active_version_dir is not None and \
                    Path(v.dir) == Path(self._active_version_dir):
                active = i
        if versions:
            combo.setCurrentIndex(active)
        combo.blockSignals(False)
        has = bool(versions)
        self._scans_label.setVisible(has)
        combo.setVisible(has)

    def _on_scans_combo_activated(self, idx: int) -> None:
        data = self._scans_combo.itemData(idx)
        if not data:
            return
        version_dir = Path(data)
        if self._active_version_dir is not None and \
                version_dir == Path(self._active_version_dir):
            return  # already showing this version
        from ..project import workspace
        for v in workspace.list_versions(self._bids_root):
            if Path(v.dir) == version_dir:
                self._load_version(v)
                self._refresh_scans_combo()
                break

    def _on_undo(self) -> None:
        """Undo the last curation edit; stash it for Redo."""
        if self._project is None:
            self.log_message.emit("Nothing to undo")
            return
        events = self._project.undo_last_user_edit()
        if events:
            self._redo_stack.append(events)
            self._reload_active_version()
            self.log_message.emit("Undid the last change" if len(events) == 1
                                  else f"Undid the last change ({len(events)} rows)")
        else:
            self.log_message.emit("Nothing to undo")
        self._update_undo_redo()

    def _on_redo(self) -> None:
        """Re-apply the most recently undone edit."""
        if self._project is None or not self._redo_stack:
            return
        events = self._redo_stack.pop()
        for event in events:
            self._project.append(event)
        self._reload_active_version()
        self.log_message.emit("Redid the change" if len(events) == 1
                              else f"Redid the change ({len(events)} rows)")
        self._update_undo_redo()

    def _on_user_edited(self) -> None:
        """A fresh edit diverges the timeline: previously-undone edits are gone."""
        if self._redo_stack:
            self._redo_stack.clear()
        self._update_undo_redo()

    def _update_undo_redo(self) -> None:
        """Sync the Redo button's enabled state with the redo stack."""
        if hasattr(self, "_redo_btn"):
            self._redo_btn.setEnabled(bool(self._redo_stack))

    def _reload_active_version(self) -> None:
        """Reload the active version so the model reflects the truncated log."""
        if self._active_version_dir is None or self._bids_root is None:
            return
        from ..project import workspace
        for v in workspace.list_versions(self._bids_root):
            if Path(v.dir) == Path(self._active_version_dir):
                # Preserve the redo stack across the undo/redo reload.
                self._load_version(v, clear_redo=False)
                self._refresh_scans_combo()
                return

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
        if output_tsv is not None:
            self._output_tsv = Path(output_tsv)
        self._properties.set_shims(self._series_shims())
        self._resolve_sources(df)
        self._filter_pane.bind_model(self._model)
        self._overview.bind_model(self._model)
        self._raw_pane.bind_model(self._model)
        if self._raw_root is not None:
            self._raw_pane.set_root(self._raw_root)
        # The BIDS output pathbar is **independent** of the scan TSV
        # location — the user sets it explicitly with the change button.
        # Swap the inspection-pane stack from placeholder → table.
        self._inspection_stack.setCurrentWidget(self._table)
        self._update_status_chips()
        self._update_meta_button_state()
        # Seed the model's dataset-wide defaults so inheritance columns show
        # the global value (muted) in rows that have no per-row override.
        self._model.set_global_spec(self._load_global_spec())
        # Refresh the chips when the model edits a row (the Overview follows
        # the model itself).
        self._model.dataChanged.connect(lambda *_a: self._update_status_chips())
        # A fresh user edit invalidates the redo stack (timeline diverged).
        self._model.userEdited.connect(self._on_user_edited)
        # Per-row acquisition overrides (device / institution) live in the
        # recording-metadata scaffold, not a TSV column. Persist it whenever the
        # model reports one changed so the convert verb (which reloads the
        # scaffold) sees the edit.
        self._model.recordingSpecChanged.connect(self._persist_recording_spec)
        # Default-select row 0 if any rows exist so the Properties panel
        # has something to show without the user having to click first.
        has_rows = self._model.rowCount() > 0
        # Re-validate is available once a scan has populated the inventory.
        self._revalidate_btn.setEnabled(has_rows)
        if has_rows:
            self._table.selectRow(0)
            self._run_btn.setEnabled(True)
            self._run_btn.setToolTip("Run conversion on the included rows")
        else:
            self._run_btn.setEnabled(False)

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
        preview_converter_fields: bool = True,
        skip_bids_guess: bool = False,
        user_hints=None,
        exclusions=None,
        index_widths=None,
        drop_entities=None,
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
        self._clear_source_state()

        # A fresh scan starts from clean metadata: drop any stale
        # recording-metadata scaffold left at this output path by a PREVIOUS
        # scan of different data (otherwise its defaults / event labels would
        # leak onto the new dataset). The scan re-seeds the scaffold from the
        # newly-detected event codes.
        from ..recording_meta import scaffold_sidecar_path
        stale = scaffold_sidecar_path(self._output_tsv)
        try:
            if stale.exists():
                stale.unlink()
        except OSError:
            log.warning("could not remove stale recording-metadata scaffold %s", stale)

        self._spinner.set_busy(True, message=f"Scanning {self._raw_root.name}…")
        self._begin_run("Scan", self._raw_root.name)
        worker = ScanWorker(
            self._raw_root, self._output_tsv,
            dataset=dataset, line_freq=line_freq, montage=montage,
            n_jobs=n_jobs,
            probe_convert=probe_convert,
            preview_converter_fields=preview_converter_fields,
            skip_bids_guess=skip_bids_guess,
            user_hints=user_hints,
            exclusions=exclusions,
            index_widths=index_widths,
            drop_entities=drop_entities,
            classifier_config=AppSettings.load().classifier(),
            parent=self,
        )
        worker.progress.connect(self._on_progress)
        worker.finished_with_result.connect(self._on_scan_finished)
        worker.failed.connect(self._on_scan_failed)
        worker.stopped.connect(self._on_scan_stopped)
        worker.finished.connect(self._on_worker_qt_finished)
        self._scan_worker = worker
        # Flip the Scan button to "Stop scanning" and lock out Run.
        self._set_scan_running(True)
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

        self._scan_btn = QPushButton("  Scan…")
        self._scan_btn.setObjectName("tb-btn")
        self._scan_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        icons.apply_button(self._scan_btn, "scan")
        self._scan_btn.clicked.connect(self._on_scan_clicked)
        lay.addWidget(self._scan_btn)

        # TSV filename input — the scan writes to ``<bids_parent>/<name>``
        # so the inventory ends up next to the converted BIDS tree, not
        # cluttering the raw data folder. Filename only; ``<bids_parent>``
        # is set independently via the path bar below.
        self._tsv_label = QLabel("TSV:")
        self._tsv_label.setObjectName("tb-mini-label")
        lay.addWidget(self._tsv_label)
        self._tsv_filename_edit = QLineEdit()
        self._tsv_filename_edit.setObjectName("tb-input")
        self._tsv_filename_edit.setPlaceholderText("inventory.tsv")
        self._tsv_filename_edit.setText(self._app_settings.scan_tsv_filename)
        self._tsv_filename_edit.setMaximumWidth(150)
        self._tsv_filename_edit.editingFinished.connect(self._on_tsv_filename_edited)
        lay.addWidget(self._tsv_filename_edit)

        # Scan-version picker (project mode): switch between scans of different
        # source folders within this dataset. Hidden until a project has >=1
        # scan; replaces the free-path TSV filename field in project mode.
        self._scans_label = QLabel("Scan table:")
        self._scans_label.setObjectName("tb-mini-label")
        self._scans_label.setVisible(False)
        lay.addWidget(self._scans_label)
        self._scans_combo = QComboBox()
        self._scans_combo.setObjectName("tb-input")
        self._scans_combo.setMinimumWidth(210)
        self._scans_combo.setToolTip(
            "Load a previous scan table in this project. Each scan of a source "
            "folder is its own version; pick one to resume curating it."
        )
        self._scans_combo.setVisible(False)
        self._scans_combo.activated.connect(self._on_scans_combo_activated)
        lay.addWidget(self._scans_combo)

        # Undo the last curation edit (project mode). Rides the active version's
        # event log so it reverts one cell / entity / include edit.
        self._undo_btn = QPushButton("  Undo")
        self._undo_btn.setObjectName("tb-btn-ghost")
        self._undo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        icons.apply_button(self._undo_btn, "undo")
        self._undo_btn.setToolTip(
            f"Undo the last change ({_native(QKeySequence.StandardKey.Undo)}). A change "
            "made to many "
            "rows at once (a folder's checkbox, a bulk edit) goes back in one step.")
        self._undo_btn.setVisible(False)
        self._undo_btn.clicked.connect(self._on_undo)
        lay.addWidget(self._undo_btn)

        self._redo_btn = QPushButton("  Redo")
        self._redo_btn.setObjectName("tb-btn-ghost")
        self._redo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        icons.apply_button(self._redo_btn, "redo")
        self._redo_btn.setToolTip(
            f"Redo the last undone change ({_native(QKeySequence.StandardKey.Redo)})")
        self._redo_btn.setVisible(False)
        self._redo_btn.setEnabled(False)
        self._redo_btn.clicked.connect(self._on_redo)
        # Ctrl+Z / Cmd+Z anywhere in the Converter. A text field being typed
        # in keeps its own undo: Qt offers the key to the focused editor
        # first.
        # Every key the platform binds (Redo is Ctrl+Y and Ctrl+Shift+Z on
        # Windows, Shift+Cmd+Z on macOS), not just the first.
        for sequence, slot in ((QKeySequence.StandardKey.Undo, self._on_undo),
                               (QKeySequence.StandardKey.Redo, self._on_redo)):
            shortcut = QShortcut(self)
            shortcut.setKeys(sequence)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
        lay.addWidget(self._redo_btn)

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

        # Re-validate: force a full recompute of every row's valid / warning /
        # error state + the chip tallies. Edits update these live, but this is
        # the explicit, guaranteed-current control the user asked for. Enabled
        # only once a scan has populated the model.
        self._revalidate_btn = QPushButton("  Re-validate")
        self._revalidate_btn.setObjectName("tb-btn-ghost")
        self._revalidate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        icons.apply_button(self._revalidate_btn, "revalidate")
        self._revalidate_btn.setToolTip(
            "Recompute the valid / warning / error tallies for every row "
            "against the BIDS schema (after editing entities, tasks, includes, "
            "etc.)."
        )
        self._revalidate_btn.setEnabled(False)
        self._revalidate_btn.clicked.connect(self._on_revalidate_clicked)
        lay.addWidget(self._revalidate_btn)

        # Toggle: highlight suspected-abort rows with a purple tint.
        # Aborts are already auto-deselected by the scanner; highlighting
        # Highlight aborts + Bulk edit buttons are constructed here so
        # they're available before _build_inspection_pane runs, but
        # they're added to the inspection pane's footer (closer to the
        # table they act on), not to this toolbar.
        self._aborts_btn = QPushButton("  Highlight aborts")
        self._aborts_btn.setObjectName("tb-btn-toggle")
        icons.apply_button(self._aborts_btn, "highlight")
        self._aborts_btn.setCheckable(True)
        self._aborts_btn.setChecked(self._app_settings.highlight_aborts)
        self._aborts_btn.setToolTip(
            "Highlight in purple any row the scanner flagged as "
            "``suspected_abort`` (operator restart after a noisy / blurry "
            "attempt). These rows are already auto-deselected; the "
            "overlay just makes them easy to spot."
        )
        self._aborts_btn.toggled.connect(self._on_aborts_toggled)

        self._bulk_btn = QPushButton("  Bulk edit…")
        self._bulk_btn.setObjectName("tb-btn")
        icons.apply_button(self._bulk_btn, "bulk_edit")
        self._bulk_btn.setEnabled(False)
        self._bulk_btn.setToolTip(
            "Select 2+ rows (cmd-/shift-click), then click to apply one "
            "value to one column across the whole selection. Updates "
            "entities + basenames in step."
        )
        self._bulk_btn.clicked.connect(self._on_bulk_edit_clicked)

        # "Manage columns" opens a popup to toggle every column at once,
        # with a description of each. Replaces the one-at-a-time header
        # right-click menu (which still works as a shortcut).
        self._columns_btn = QPushButton("  Manage columns…")
        self._columns_btn.setObjectName("tb-btn")
        icons.apply_button(self._columns_btn, "columns")
        self._columns_btn.setToolTip(
            "Choose which columns the inspection table shows, with a note on "
            "what each one means."
        )
        self._columns_btn.clicked.connect(self._open_column_manager)

        # "Recording metadata" opens the dataset-level EEG/MEG enrichment
        # editor (device, institution, reference/ground, event labels) shared
        # across every recording. Per-row overrides live in the table itself.
        self._meta_btn = QPushButton("  Dataset metadata…")
        self._meta_btn.setObjectName("tb-btn")
        self._meta_btn.setToolTip(
            "Edit dataset-wide metadata grouped by where it is written: the "
            "parts that apply to every datatype (events -> events.tsv, "
            "phenotype -> phenotype/) plus the EEG/MEG acquisition defaults "
            "(-> the datatype sidecar). Saved beside the inventory."
        )
        self._meta_btn.clicked.connect(self._open_recording_meta)
        # Disabled until a scan loads EEG/MEG rows (it is EEG/MEG-only metadata).
        self._meta_btn.setEnabled(False)

        # Busy spinner + status message — visible only while a worker
        # is running. Lives in the centre of the toolbar so it's hard
        # to miss when something's in flight.
        self._spinner = BusySpinner()
        lay.addWidget(self._spinner)

        lay.addStretch(1)

        # The Settings gear lives in the global top header (left of the
        # theme toggle), not here -- see ``_TopHeader`` in main_window.

        self._run_btn = QPushButton("  Run conversion")
        self._run_btn.setObjectName("tb-btn-primary")
        icons.apply_button(self._run_btn, "run", color=CUR().get("primary_btn_text"))
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
        # No header of its own: the frame around it (with Properties) is
        # titled Inspection, and a second one read as a stutter. Its view
        # switch is named HEADER_LEAD, so the frame lifts it into that title
        # bar, after the title; floating, it comes back to this row.
        home = QWidget(pane)
        self._view_switch_home = QHBoxLayout(home)
        self._view_switch_home.setContentsMargins(0, 0, 0, 0)
        switch_holder = QWidget(home)
        switch_holder.setObjectName(HEADER_LEAD)
        sh = QHBoxLayout(switch_holder)
        sh.setContentsMargins(0, 0, 0, 0)
        self._view_switch = Segmented((
            ("table", "Table", "Every recording, one row each: what will be converted and how"),
            ("overview", "Overview",
             "What each subject will have, and where one differs from the others"),
        ), parent=switch_holder)
        self._view_switch.changed.connect(self.set_inspection_view)
        sh.addWidget(self._view_switch)
        self._view_switch_home.addWidget(switch_holder)
        self._view_switch_home.addStretch(1)
        v.addWidget(home)
        self._inspection_views = QStackedWidget()
        table_page = QWidget()
        tv = QVBoxLayout(table_page)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(0)

        # Use QStackedWidget so we can swap between "empty placeholder"
        # and the actual table without recreating widgets — keeps view
        # focus / selection state predictable across loads.
        self._inspection_stack = QStackedWidget()

        empty = QLabel("Pick a raw-data folder and click Scan… to populate.")
        empty.setObjectName("pane-hint")
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty.setWordWrap(True)
        self._inspection_empty = empty
        self._inspection_stack.addWidget(empty)

        self._table = self._build_table()
        self._inspection_stack.addWidget(self._table)

        tv.addWidget(self._inspection_stack, 1)

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
        fl.addWidget(self._meta_btn)
        fl.addWidget(self._columns_btn)
        fl.addWidget(self._bulk_btn)
        tv.addWidget(footer)
        self._inspection_views.addWidget(table_page)

        # The Overview: what each subject will have. A count picked there
        # selects its rows here (Properties follows); Show opens the table.
        self._overview = OverviewPane()
        self._overview.rows_chosen.connect(self.select_rows)
        self._overview.rows_shown.connect(
            lambda rows: (self.set_inspection_view("table"), self.select_rows(rows)))
        self._inspection_views.addWidget(self._overview)
        v.addWidget(self._inspection_views, 1)
        view = str(QSettings().value(_INSPECTION_VIEW_KEY, "table") or "table")
        self.set_inspection_view(view if view in ("table", "overview") else "table",
                                 remember=False)
        return pane

    # ------------------------------------------------------------------
    # Inspection: Table or Overview
    # ------------------------------------------------------------------

    def inspection_view(self) -> str:
        """``"table"`` or ``"overview"``."""
        return "overview" if self._inspection_views.currentWidget() is self._overview else "table"

    def set_inspection_view(self, view: str, *, remember: bool = True) -> None:
        """Show the table or the Overview in the Inspection area."""
        page = self._overview if view == "overview" else self._inspection_views.widget(0)
        self._inspection_views.setCurrentWidget(page)
        self._view_switch.set_current("overview" if view == "overview" else "table")
        if remember:
            QSettings().setValue(_INSPECTION_VIEW_KEY, self.inspection_view())

    def select_rows(self, rows: list) -> None:
        """Select inventory ``rows`` in the table, the first one current (so
        Properties and Filter follow it) and scrolled into view."""
        if self._model is None or not rows:
            return
        from PyQt6.QtCore import QItemSelection, QItemSelectionModel

        count = self._model.rowCount()
        wanted = sorted({int(r) for r in rows if 0 <= int(r) < count})
        if not wanted:
            return
        last_column = max(self._model.columnCount() - 1, 0)
        selection = QItemSelection()
        start = prev = wanted[0]
        for row in [*wanted[1:], None]:
            if row is not None and row == prev + 1:
                prev = row
                continue
            selection.select(self._model.index(start, 0), self._model.index(prev, last_column))
            if row is not None:
                start = prev = row
        sel = self._table.selectionModel()
        flags = QItemSelectionModel.SelectionFlag
        sel.select(selection, flags.ClearAndSelect | flags.Rows)
        first = self._model.index(wanted[0], 0)
        sel.setCurrentIndex(first, flags.NoUpdate)
        self._table.scrollTo(first, self._table.ScrollHint.PositionAtCenter)

    def _plan_context(self) -> tuple:
        """What the BIDS preview needs beside the table: the settings that
        change what is written, the scan's series features, the recording
        metadata, and each series' first DICOM (it names the series'
        ``sourcedata/`` folder, so it is looked up only when that is on)."""
        from .. import __version__
        from ..converter.output_plan import PlanOptions
        from ..schema import loader

        s = self._app_settings
        dataset = (self._bids_root.name if self._bids_root is not None else
                   (str(self._model.dataframe()["dataset"].iloc[0])
                    if self._model is not None and "dataset" in self._model.dataframe().columns
                    and len(self._model.dataframe()) else ""))
        options = PlanOptions(
            dataset=dataset,
            root=str(self._bids_root) if self._bids_root is not None else "",
            inventory_dir=str(self._output_tsv.parent) if self._output_tsv is not None else "",
            run_metadata=bool(s.post_run_metadata),
            fill_scope=s.placeholder_scope(),
            companions=bool(s.post_fixup_companions),
            citation=bool(s.post_fixup_citation),
            tables=bool(s.post_metadata_tables),
            intended_for=s.convert_intended_for,
            b0_field=bool(s.convert_b0_field),
            deface=bool(s.convert_deface),
            deface_engine=s.convert_deface_engine,
            sourcedata=s.convert_sourcedata,
            sourcedata_modalities=tuple(m.strip() for m in s.convert_sourcedata_modalities.split(",")
                                        if m.strip()),
            force_edf=bool(s.convert_force_edf),
            bids_version=loader.bids_version(),
            version=str(__version__),
        )
        features = {}
        if self._output_tsv is not None:
            from ..classifier.acquisition import cache

            key = (str(self._output_tsv), _mtime(cache.features_path(self._output_tsv)))
            if getattr(self, "_features_cache", (None,))[0] != key:
                try:
                    self._features_cache = (key, cache.read(self._output_tsv))
                except Exception:  # noqa: BLE001 - the preview shows less
                    self._features_cache = (key, {})
            features = self._features_cache[1]
        spec = self._model.global_spec() if self._model is not None else None
        first_files: dict[str, str] = {}
        if s.convert_sourcedata in ("copy", "archive") and self._sources is not None \
                and self._model is not None:
            df = self._model.dataframe()
            for uid in (df["series_uid"] if "series_uid" in df.columns else []):
                uid = str(uid or "").split("|", 1)[0].strip()
                if uid and uid not in first_files:
                    files = self._sources.series_files(uid)
                    if files:
                        first_files[uid] = str(files[0])
        return options, features, spec, first_files

    def _fit_view_switch_home(self) -> None:
        floating = self._inspection_unit.is_detached()
        self._view_switch_home.setContentsMargins(*((10, 8, 10, 6) if floating else (0, 0, 0, 0)))

    def _build_table(self) -> QTableView:
        t = QTableView()
        t.setObjectName("inv-table")
        t.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        t.customContextMenuRequested.connect(self._on_table_menu)
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
        # Spreadsheet-style sizing: every column is independently resizable
        # (no greedy stretch column to fight) and the table scrolls
        # horizontally when the columns overflow. The last visible section
        # stretches only to fill leftover space so there's no dead gap.
        t.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        header = t.horizontalHeader()
        header.setHighlightSections(False)
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        # Drag to reorder columns; the order is persisted (see
        # ``_persist_column_order`` / ``_restore_column_order``).
        header.setSectionsMovable(True)
        header.sectionMoved.connect(self._on_section_moved)
        # Double-clicking a column's right edge auto-fits it to its contents
        # so shrunk-too-small text becomes readable again. Works per column.
        header.setSectionsClickable(True)
        header.sectionHandleDoubleClicked.connect(self._table_resize_column_to_contents)
        # Column visibility is managed solely via the "Manage columns" button
        # (a header right-click menu interfered with drag-to-reorder).

        # Per-column delegates — reuse the M1 extractions.
        for col, spec in enumerate(COLUMNS):
            if spec.role == "checkbox":
                t.setItemDelegateForColumn(col, CheckboxDelegate(t))
            elif spec.role == "status":
                t.setItemDelegateForColumn(col, StatusDelegate(t))
            elif spec.key == "montage":
                # Dropdown-only: pick from the MNE built-in montages.
                t.setItemDelegateForColumn(
                    col, ChoiceDelegate(builtin_montages, role=spec.role,
                                        blank_label="(none)", parent=t),
                )
            elif spec.key == "line_freq":
                # Dropdown-only: the two electrical-grid conventions.
                t.setItemDelegateForColumn(
                    col, ChoiceDelegate(["50", "60"], role=spec.role,
                                        blank_label="(blank)", parent=t),
                )
            else:
                t.setItemDelegateForColumn(col, CellTextDelegate(spec.role, t))
        return t

    def _apply_column_widths(self) -> None:
        header = self._table.horizontalHeader()
        # Every column Interactive (user-resizable); seed a sensible width.
        # No per-column Stretch mode -- ``stretchLastSection`` fills any
        # leftover space, so each column (incl. the BIDS-name one) can be
        # freely shrunk or widened.
        for col, spec in enumerate(COLUMNS):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
            self._table.setColumnWidth(col, self._seed_width(spec))
        self._apply_column_visibility()
        self._restore_column_order()

    def _seed_width(self, spec) -> int:
        """A column's start width, never narrower than its own header.

        The widths were written as numbers, and a number cannot know how
        wide the text will be: at the app's 1.15x font scale, and on a
        Windows or Linux desktop with a different default font and DPI, half
        the headers came up clipped to two or three letters. A column whose
        title you cannot read is a column you cannot use.

        Measured with the header's own font metrics rather than assumed, so
        it is right at any font scale on any platform, and taken as a FLOOR
        so the hand-picked widths still win wherever they are wider.
        """
        header = self._table.horizontalHeader()
        text = spec.header
        if not text:
            return spec.width          # the checkbox and status columns
        metrics = header.fontMetrics()
        # Padding for the cell margins plus room for the sort indicator,
        # which Qt draws inside the section.
        needed = metrics.horizontalAdvance(text) + _HEADER_PADDING_PX
        return max(spec.width, needed)

    def _table_resize_column_to_contents(self, logical_index: int) -> None:
        """Auto-fit a column to its contents on header-handle double-click.

        Lets the user reopen a column they shrank too small to read.
        """
        if not (0 <= logical_index < len(COLUMNS)):
            return
        self._table.resizeColumnToContents(logical_index)

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
            # A column revealed for the first time can come back at width 0
            # (Qt collapses hidden interactive sections). Re-seed its width
            # so it shows up draggable rather than as a hairline.
            if visible and self._table.columnWidth(col) <= 0:
                self._table.setColumnWidth(col, self._seed_width(spec))

    def set_column_visible(self, key: str, visible: bool) -> None:
        """Toggle a column's visibility + persist the choice."""
        if key in MANDATORY_COLUMN_KEYS:
            return
        self._column_visible[key] = visible
        self._settings().setValue(f"inspector/columns/{key}", "1" if visible else "0")
        self._apply_column_visibility()

    def set_columns_visible(self, mapping: dict[str, bool]) -> None:
        """Apply visibility for many columns at once (from the manage dialog)
        and persist each. Mandatory columns are forced visible."""
        s = self._settings()
        for key, visible in mapping.items():
            if key in MANDATORY_COLUMN_KEYS:
                self._column_visible[key] = True
                continue
            self._column_visible[key] = visible
            s.setValue(f"inspector/columns/{key}", "1" if visible else "0")
        self._apply_column_visibility()

    def _open_column_manager(self) -> None:
        """Popup to toggle every column at once, with per-column notes."""
        from .column_manager_dialog import ColumnManagerDialog
        dlg = ColumnManagerDialog(dict(self._column_visible), self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            self.set_columns_visible(dlg.result_visibility())
            if dlg.order_was_reset():
                self.reset_column_order()

    def _open_recording_meta(self) -> None:
        """Open the dataset-level recording-metadata editor.

        Covers EEG/MEG and PET plus the sections that apply to every
        datatype; the dialog shows only the blocks whose datatype the scan
        actually found.

        Edits the scaffold beside the inventory TSV (the same file the scan
        seeds and the convert verb auto-discovers). Requires a loaded scan so
        there is an inventory to anchor the scaffold to.
        """
        if self._output_tsv is None:
            QMessageBox.information(
                self,
                "Recording metadata",
                "Run a scan first so the metadata can be saved beside the inventory.",
            )
            return
        from ..recording_meta import scaffold_sidecar_path
        from .recording_meta_dialog import RecordingMetaDialog

        from ..metadata.template_plan import (
            blood_conditions, example_paths_for, pair_counts, present_pairs,
        )

        # The (datatype, suffix) pairs the scan actually found, and one real
        # path per pair. Without them the dialog had to guess, and its guess was
        # that a datatype names its own suffix: true for eeg and meg by
        # coincidence, false for func, whose suffixes are bold and sbref. It
        # showed "sub-_func.json", which is not a BIDS name, and filed the
        # answers under "func/func", which nothing ever reads.
        df = self._model.dataframe() if self._model is not None else None
        scaffold = scaffold_sidecar_path(self._output_tsv)
        dlg = RecordingMetaDialog(
            scaffold, self._present_datatypes(), self,
            montage_suggestions=self._montage_suggestions(),
            scan_suggestions=self._scan_suggestions(),
            present_pairs=present_pairs(df),
            example_paths=example_paths_for(df),
            pair_counts=pair_counts(df),
            # A conditional requirement whose condition we can see holds: a
            # parent-fraction curve attached to a run makes the metabolite
            # fields real questions rather than ones nobody can answer.
            also_ask={("pet", "blood"): blood_conditions(df)},
            bids_root=self._bids_root,
            eyetrack=df is not None and "modality" in df.columns
            and bool((df["modality"].astype(str) == "eyetrack").any()),
        )
        if not dlg.exec():
            return
        if self._model is not None:
            # Re-flow the saved dataset defaults into every inherited row.
            self._model.set_global_spec(self._load_global_spec())
        if dlg.rename_project_to:
            self.project_rename_requested.emit(
                self._bids_root, dlg.rename_project_to,
            )

    def _load_global_spec(self):
        """Load the recording-metadata scaffold for the current inventory.

        Falls back to a default spec (PowerLineFrequency=50) when no scaffold
        exists yet, so inheritance always has a baseline.
        """
        from ..recording_meta import default_spec, load_spec, scaffold_sidecar_path
        if self._output_tsv is not None:
            path = scaffold_sidecar_path(self._output_tsv)
            if path.exists():
                try:
                    return load_spec(path)
                except Exception:
                    pass
        return default_spec()

    def _persist_recording_spec(self) -> None:
        """Write the model's live recording-metadata spec to the scaffold.

        Called when a per-row acquisition override changes. The scaffold sits
        beside the inventory TSV (the same file the scan seeds, the dialog
        edits, and the convert verb auto-discovers), so per-row device /
        institution edits survive to conversion time. No-op without an
        inventory anchor or a spec.
        """
        if self._model is None or self._output_tsv is None:
            return
        spec = self._model.global_spec()
        if spec is None:
            return
        from ..recording_meta import dump_spec, scaffold_sidecar_path
        try:
            path = scaffold_sidecar_path(self._output_tsv)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dump_spec(spec), encoding="utf-8")
        except OSError:
            log.warning("could not persist recording-metadata scaffold", exc_info=True)

    def _present_datatypes(self) -> set[str]:
        """The set of ``datatype`` values across the loaded model."""
        if self._model is None:
            return set()
        df = self._model.dataframe()
        if "datatype" not in df.columns:
            return set()
        return {str(v).strip().lower() for v in df["datatype"] if str(v).strip()}

    def _montage_suggestions(self) -> list[str]:
        """Distinct per-recording montage suggestions found at scan (for the
        dataset dialog's read-only summary). Order-preserving, deduplicated."""
        if self._model is None:
            return []
        df = self._model.dataframe()
        if "montage_suggestion" not in df.columns:
            return []
        return self._distinct_column("montage_suggestion")

    def _scan_suggestions(self) -> dict[str, list[str]]:
        """What the scan read out of the recordings, per BIDS field.

        Which column carries which field is stated once, in recording_meta, so
        the dataset dialog and the per-row panel offer the same hints. Two
        collectors used to do this with the field names written out here, and
        they had drifted: the dialog paired tracer with radionuclide into one
        "FDG / F18" string, which reads well as a summary and is not a value
        anyone should be offered for either field.
        """
        from ..recording_meta import SCAN_SUGGESTION_COLUMNS

        return {
            name: self._distinct_column(column)
            for name, column in SCAN_SUGGESTION_COLUMNS.items()
        }

    def _distinct_column(self, column: str) -> list[str]:
        """Order-preserving distinct non-blank values of a model column."""
        if self._model is None:
            return []
        df = self._model.dataframe()
        if column not in df.columns:
            return []
        seen: list[str] = []
        for v in df[column]:
            s = str(v).strip()
            if s and s.lower() not in ("nan", "none") and s not in seen:
                seen.append(s)
        return seen

    def _update_meta_button_state(self) -> None:
        """Enable the Dataset-metadata button whenever an inventory is loaded.

        The dialog has sections that apply to every datatype (events,
        phenotype) that
        apply to any dataset; the EEG/MEG-specific sections inside it are gated
        by the scanned datatypes.
        """
        self._meta_btn.setEnabled(self._model is not None and self._model.rowCount() > 0)

    # ------------------------------------------------------------------
    # Column order (drag-to-reorder, persisted via QSettings)
    # ------------------------------------------------------------------

    def _on_section_moved(self, _logical: int, _old_vis: int, _new_vis: int) -> None:
        """User dragged a header section -> persist the new visual order."""
        if getattr(self, "_restoring_order", False):
            return
        self._persist_column_order()

    def _persist_column_order(self) -> None:
        header = self._table.horizontalHeader()
        # Keys ordered by their current visual position.
        ordered = sorted(
            range(len(COLUMNS)), key=lambda logical: header.visualIndex(logical)
        )
        keys = [COLUMNS[logical].key for logical in ordered]
        self._settings().setValue("inspector/column_order", ",".join(keys))

    def default_column_order(self) -> list[str]:
        """The order the table opens in: the seven a person reads first.

        ``include`` and ``status`` stay where they are, at the very left,
        because they are the tick and the badge the eye goes to and a text
        column in front of them buries both.
        """
        leading = [
            k for k in InventoryTableModel.DEFAULT_LEADING_KEYS
            if any(c.key == k for c in COLUMNS)
        ]
        pinned = [c.key for c in COLUMNS if c.role in ("checkbox", "status")]
        rest = [
            c.key for c in COLUMNS
            if c.key not in leading and c.key not in pinned
        ]
        return pinned + leading + rest

    def reset_column_order(self) -> None:
        """Put the leading columns back at the front, keeping the rest.

        Offered as "Reset defaults" rather than applied silently, because a
        user who has arranged their own layout has said something about how
        they work and a release should not overrule it.
        """
        self._apply_column_order(self.default_column_order())
        self._persist_column_order()

    def _restore_column_order(self) -> None:
        """Re-apply the persisted visual order, or the default first time."""
        raw = self._settings().value("inspector/column_order", "")
        if not raw:
            # No stored layout: this is a fresh install or a fresh profile,
            # so the table opens in the order that reads best rather than in
            # the order the inventory happens to be built in.
            self._apply_column_order(self.default_column_order())
            return
        saved = [k for k in str(raw).split(",") if k]
        self._apply_column_order(saved)

    def _apply_column_order(self, keys: list[str]) -> None:
        """Move the header sections so they read in ``keys`` order."""
        key_to_logical = {spec.key: i for i, spec in enumerate(COLUMNS)}
        header = self._table.horizontalHeader()
        self._restoring_order = True
        try:
            target_visual = 0
            for key in keys:
                logical = key_to_logical.get(key)
                if logical is None:
                    continue
                current_visual = header.visualIndex(logical)
                if current_visual != -1 and current_visual != target_visual:
                    header.moveSection(current_visual, target_visual)
                target_visual += 1
        finally:
            self._restoring_order = False


    # ------------------------------------------------------------------
    # Internals — bottom dock
    # ------------------------------------------------------------------

    def _build_log_dock(self) -> QWidget:
        """The Log: the live output of scans and conversions.

        Append-only from the workers' progress signals, styled in theme.qss
        (``QPlainTextEdit#dock-log``): runs with headings, warnings and errors
        in colour, emptied for each run when the setting says so; every run
        also writes a file.

        One header, the panel's: its buttons (clear, the run's file, every
        log) sit in it, named HEADER_EXTRAS so the frame lifts them there
        and hands them back when the Log floats (GUI_LOOK_AND_FEEL 4.26).
        """
        from .widgets.run_log import RunLog

        self._log_view = RunLog()
        log_box = QWidget()
        log_box.setObjectName("dock-log-box")
        col = QVBoxLayout(log_box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        home = QWidget()
        home_row = QHBoxLayout(home)
        home_row.setContentsMargins(0, 0, 0, 0)
        home_row.setSpacing(0)
        home_row.addStretch(1)
        corner = self._build_log_corner()
        corner.setObjectName(HEADER_EXTRAS)
        home_row.addWidget(corner)
        col.addWidget(home)
        col.addWidget(self._log_view, 1)
        return log_box

    def _build_log_corner(self) -> QWidget:
        """Clear, the last run's file and every log, as the panel's corner
        icons."""
        corner = QWidget()
        corner.setObjectName("dock-corner")
        h = QHBoxLayout(corner)
        h.setContentsMargins(0, 0, 4, 0)
        h.setSpacing(2)
        self._log_buttons: dict[str, QToolButton] = {}
        for name, tip, slot in (
            ("log_clear", "Clear the Log", lambda: self._log_view.clear_log()),
            ("log_file", "Open the last run's log file", self._open_last_log),
            ("log_history", "Every log of this dataset...", self._show_log_history),
        ):
            btn = QToolButton()
            btn.setObjectName("dock-corner-btn")
            btn.setAutoRaise(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(tip)
            btn.setIcon(icons.icon(name))
            btn.clicked.connect(slot)
            h.addWidget(btn)
            self._log_buttons[name] = btn
        return corner

    def _log_root(self) -> Optional[Path]:
        """Whose logs: the open project's, else the user's."""
        from ..util.oplog import dataset_root_for

        return self._bids_root or dataset_root_for(self._output_tsv)

    def _open_last_log(self) -> None:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        from ..util import oplog

        path = oplog.latest(self._log_root())
        if path is None:
            self.log_message.emit("No log file yet: every operation writes one when it runs.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _show_log_history(self) -> None:
        from .log_history_dialog import LogHistoryDialog

        dlg = LogHistoryDialog(self._log_root(), self)
        self._log_history = dlg
        dlg.show()

    def _begin_run(self, title: str, detail: str = "", *, first: bool = True) -> None:
        """A heading in the Log for an operation. The first step of a
        user's action empties the panel when the setting says so; the steps
        that follow it (metadata, validation after a conversion) do not."""
        self._flush_log_buffer()
        self._log_view.begin_run(title, detail,
                                 clear=first and AppSettings.load().log_clear_on_run)

    def _on_log_message(self, text: str) -> None:
        if not text:
            return
        # Buffer; ``_flush_log_buffer`` drains every 100ms. This keeps
        # the GUI responsive when the worker is firehose-logging.
        self._log_buffer.append(text)
        # Mirror the latest line into the spinner's status text so the
        # user has a live one-liner of what's happening. No length guard
        # needed: the spinner's label elides to the toolbar's own width.
        if self._spinner.is_busy():
            self._spinner.set_message(text.splitlines()[0])

    def _flush_log_buffer(self) -> None:
        if not self._log_buffer:
            return
        # One append for the whole batch: one repaint.
        self._log_view.append_lines(self._log_buffer)
        self._log_buffer.clear()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_scan_clicked(self) -> None:
        # Toggle: while a scan is running this button reads "Stop scanning",
        # so a click requests a cooperative stop instead of starting a scan.
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.request_stop()
            self._scan_btn.setEnabled(False)
            self._scan_btn.setText("  Stopping…")
            self.log_message.emit(
                "Stop requested; finishing the read already in flight, then halting."
            )
            return
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
        # 3. Pick up the latest persisted settings on every scan so a user who
        # changed defaults in the Settings dialog without restarting still gets
        # the new values.
        s = AppSettings.load()
        # 4. Compute the scan-TSV path + dataset name. With a project open the
        # inventory lives in the project bundle and the dataset name is the
        # project's folder name (output is locked to the project root). The
        # free-path case keeps the toolbar filename + chosen BIDS-output folder
        # and lets the scanner derive the dataset slug from the raw folder name;
        # the user then partitions datasets by editing the ``dataset`` column.
        if self._bids_root is not None:
            # Scan into a fresh staging dir; on success it is promoted to a new
            # version under .bidsmgr/project/scans/ (so a second source scan
            # never overwrites the first). A failed scan just leaves staging to
            # be cleared by the next run.
            staging = self._bids_root / ".bidsmgr" / "project" / ".scan_staging"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True, exist_ok=True)
            self._output_tsv = staging / "inventory.tsv"
            dataset = self._bids_root.name
        else:
            filename = self._tsv_filename_edit.text().strip() or "inventory.tsv"
            if not filename.lower().endswith(".tsv"):
                filename += ".tsv"
            self._output_tsv = self._bids_parent / filename
            # No GUI dataset-slug setting any more: leave it unset so the
            # scanner derives the slug from the raw folder name (the CLI's
            # ``_default_dataset_slug`` default).
            dataset = None
        self.start_scan(
            self._raw_root,
            self._output_tsv,
            dataset=dataset,
            # line_freq / montage are recording metadata now (per-row dropdown
            # columns + the Recording-metadata editor), not scan settings.
            n_jobs=s.scan_n_jobs,
            probe_convert=s.scan_probe_convert,
            preview_converter_fields=s.scan_converter_preview,
            skip_bids_guess=s.scan_skip_bids_guess,
            user_hints=s.to_user_hints(),
            exclusions=s.to_exclusions(),
            index_widths=s.scan_index_widths,
            drop_entities=s.scan_drop_entities,
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
        """Pick the folder Scan reads next.

        It is NOT the open table's source: before 2026-10-09 it was written
        into the open scan's record, so an older scan named a newer scan's
        folder. Only when the open table's data is missing AND this folder
        holds it is it offered as that scan's new location.
        """
        d = QFileDialog.getExistingDirectory(
            self, "Pick raw-data folder",
            str(self._raw_root or Path.home()),
        )
        if not d:
            return
        picked = Path(d)
        if self._source_missing():
            from ..inventory import sources

            found, seen = sources.locate(self._model.dataframe(), self._sources.files_by_uid,
                                         picked, scanned=self._sources.scanned)
            if seen.found and QMessageBox.question(
                self, "Use this folder for the table shown?",
                f"This folder holds the source data of the scan table shown "
                f"({seen.found} of {seen.checked} recordings checked are in it).\n\n"
                "Yes: read that scan's data from here from now on.\n"
                "No: keep it as the folder to scan next.",
            ) == QMessageBox.StandardButton.Yes:
                self._relink(found, seen)
                return
        self._raw_root = picked
        AppSettings.remember_raw_root(self._raw_root)
        # Live preview of the picked folder before any scan runs.
        self._raw_pane.set_root(self._raw_root)
        self._show_source_state()

    # -- where the table's data is ---------------------------------------------------

    def _resolve_sources(self, df: pd.DataFrame) -> None:
        """Find the shown table's data (its files over its record), correct a
        wrong record, and tell the bar and Properties."""
        from ..inventory import sources

        label = None
        if self._active_version_dir is not None:
            from ..project import workspace

            label = workspace.read_version_meta(self._active_version_dir).get("source_label")
        files_by_uid = sources.load_files_by_uid(self._output_tsv) if self._output_tsv else {}
        found, seen = sources.resolve(df, files_by_uid, recorded=self._raw_root, label=label)
        if seen.corrected_from is not None:
            self._record_source(found.root)
            self.log_message.emit(
                f"This scan recorded its source as {seen.corrected_from}, but its files are "
                f"in {found.root}; the record is corrected.")
        self._sources, self._source_check = found, seen
        self._raw_root = found.root
        self._properties.set_sources(found)
        self._show_source_state()
        if seen.state in ("missing", "partial"):
            self.log_message.emit(sources.describe(seen, found.root)
                                  + " Use Locate... on the Raw input bar.")

    def _record_source(self, root: Optional[Path]) -> None:
        """Write where the active scan's data is into its version record."""
        if self._active_version_dir is None or root is None:
            return
        from ..project import workspace

        meta = workspace.read_version_meta(self._active_version_dir)
        try:
            workspace.write_version_meta(
                self._active_version_dir,
                source_label=meta.get("source_label") or self._active_version_dir.name,
                raw_root=str(root),
                status=meta.get("status") or "curating",
            )
        except OSError as exc:
            self.log_message.emit(f"Could not record the source folder: {exc}")

    def _source_missing(self) -> bool:
        return (self._model is not None and self._sources is not None
                and self._source_check is not None
                and self._source_check.state in ("missing", "partial"))

    def _clear_source_state(self) -> None:
        self._source_action = ""
        self._raw_pathbar.set_chip()
        self._raw_pathbar.set_action()

    def _show_source_state(self) -> None:
        """The Raw input bar: the shown table's source and whether its data is
        there, or the folder picked to scan next."""
        from ..inventory import sources

        bar = self._raw_pathbar
        table_root = self._sources.root if self._sources is not None else None
        if self._model is not None and table_root is not None and self._raw_root is not None \
                and self._raw_root != table_root:
            bar.set_value(str(self._raw_root), ok=True)
            bar.set_chip("", "Not scanned yet",
                         "Scan reads this folder into a new scan table. The table shown "
                         f"came from {table_root}.")
            bar.set_action("Show the table's source",
                           f"Back to {table_root}, the folder the table shown was scanned from")
            self._source_action = "back"
            return
        if self._source_missing():
            missing = self._source_check.state == "missing"
            bar.set_value(str(table_root), ok=False)
            bar.set_chip("err" if missing else "warn", "Moved" if missing else "Partly missing",
                         sources.describe(self._source_check, table_root))
            bar.set_action("Locate...",
                           "Pick the folder where this scan's source data is now. The preview "
                           "and the conversion read it from there.")
            self._source_action = "locate"
            return
        self._clear_source_state()
        if self._raw_root is not None:
            bar.set_value(str(self._raw_root), ok=self._raw_root.is_dir())

    def _on_source_action(self) -> None:
        if self._source_action == "back" and self._sources is not None:
            self._raw_root = self._sources.root
            self._raw_pane.set_root(self._raw_root)
            self._show_source_state()
        elif self._source_action == "locate":
            self._on_locate_source()

    def _on_locate_source(self) -> None:
        """Say where the shown table's data is now."""
        if self._model is None or self._sources is None:
            return
        from ..inventory import sources

        root = self._sources.root
        start = next((str(p) for p in ((root.parent if root else None), Path.home())
                      if p is not None and p.is_dir()), "")
        d = QFileDialog.getExistingDirectory(self, "Where is the source data now?", start)
        if not d:
            return
        found, seen = sources.locate(self._model.dataframe(), self._sources.files_by_uid,
                                     Path(d), scanned=self._sources.scanned)
        name = (self._sources.scanned or root or Path(d)).name
        if seen.checked and not seen.found:
            QMessageBox.warning(
                self, "Not this scan's data",
                f"None of the {seen.checked} recordings checked is in {d}.\n\n"
                f"Pick the folder that was scanned (it was called \"{name}\").")
            return
        if seen.state == "partial" and QMessageBox.question(
                self, "Part of the data is missing",
                f"Only {seen.found} of the {seen.checked} recordings checked are in {d}. "
                "Read this scan's data from there anyway?") != QMessageBox.StandardButton.Yes:
            return
        self._relink(found, seen)

    def _relink(self, found, seen) -> None:
        """The shown table's data is now at ``found.root``: remember it."""
        self._record_source(found.root)
        self._sources, self._source_check = found, seen
        self._raw_root = found.root
        self._properties.set_sources(found)
        self._raw_pane.set_root(found.root)
        self._show_source_state()
        self.log_message.emit(f"The source data of this scan is now read from {found.root}")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Closed: no more source checks, and none still running into the
        deletion that follows. An activation reaching a closed panel started
        one, and its signals racing the panel's deletion crashed the
        process (the GUI suite, about one run in six)."""
        app = QApplication.instance()
        if app is not None:
            try:
                app.applicationStateChanged.disconnect(self._on_application_state)
            except (TypeError, RuntimeError):
                pass
        if self._source_jobs is not None:
            self._source_jobs.stop_all()
        super().closeEvent(event)

    def _on_application_state(self, state) -> None:
        """Back in front: look for the table's data again (on a thread, in
        case it was on a drive that is now gone)."""
        if state != Qt.ApplicationState.ApplicationActive or self._sources is None \
                or self._model is None:
            return
        from ..inventory import sources
        from .viz.bridge import JobRunner

        if self._source_jobs is None:
            self._source_jobs = JobRunner(self)
            self._source_jobs.done.connect(self._on_source_checked)
        df = self._model.dataframe()
        cols = [c for c in ("series_uid", "source_folder", "source_file") if c in df.columns]
        self._source_generation += 1
        self._source_jobs.start("source-check", self._source_generation, sources.check,
                                self._sources, df[cols].copy())

    def _on_source_checked(self, _tag: str, generation: int, result) -> None:
        if generation != self._source_generation or self._sources is None:
            return
        before = self._source_check.state if self._source_check is not None else ""
        self._source_check = result
        if result.state != before:
            self._show_source_state()
            self._properties.set_sources(self._sources)
            if self._raw_root == self._sources.root:
                # The same path, there now or gone now: build the tree again.
                self._raw_pane.set_root(None)
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
        # Toggle: while a conversion is running this button reads "Stop
        # conversion", so a click requests a cooperative stop.
        if self._convert_worker is not None and self._convert_worker.isRunning():
            self._convert_worker.request_stop()
            self._run_btn.setEnabled(False)
            self._run_btn.setText("  Stopping…")
            self.log_message.emit(
                "Stop requested; finishing the subject already in flight, then halting."
            )
            return
        if self._model is None or self._output_tsv is None:
            return
        if self._source_missing() and self._source_check.state == "missing":
            from ..inventory import sources

            QMessageBox.warning(
                self, "The source data has moved",
                sources.describe(self._source_check, self._sources.root)
                + "\n\nUse Locate... on the Raw input bar to say where it is now.")
            return
        # Use the pre-set BIDS output if the user picked one; otherwise
        # prompt now.
        if self._bids_parent is None:
            self._on_pick_bids_parent()
            if self._bids_parent is None:
                return
        from ..converter.sourcedata import SourcedataOptions

        settings = AppSettings.load()
        options = SourcedataOptions.parse(settings.convert_sourcedata,
                                          settings.convert_sourcedata_modalities)
        if options.enabled and self._sources is not None:
            self._measure_sourcedata(options)
            return
        self.start_convert(self._bids_parent)

    def _measure_sourcedata(self, options) -> None:
        """How much keeping the source data adds, measured on a thread, then
        said before anything is converted (and refused when it will not fit)."""
        from ..cli.convert import sourcedata_estimate
        from .viz.bridge import JobRunner

        if getattr(self, "_sourcedata_jobs", None) is None:
            self._sourcedata_jobs = JobRunner(self)
            self._sourcedata_jobs.done.connect(self._on_sourcedata_measured)
            self._sourcedata_jobs.failed.connect(
                lambda _t, _g, msg: (self._spinner.set_busy(False),
                                     self.log_message.emit(f"Could not measure the source "
                                                           f"data: {msg}")))
        self._sourcedata_generation = getattr(self, "_sourcedata_generation", 0) + 1
        self._sourcedata_options = options
        self._spinner.set_busy(True, message="Measuring the source data...")
        self._sourcedata_jobs.start("sourcedata", self._sourcedata_generation,
                                    sourcedata_estimate, self._model.dataframe().copy(),
                                    self._sources, options)

    def _on_sourcedata_measured(self, _tag: str, generation: int, result) -> None:
        from ..converter import sourcedata as sd

        if generation != getattr(self, "_sourcedata_generation", 0) or self._bids_parent is None:
            return
        self._spinner.set_busy(False)
        files, size, kinds = result
        target = self._bids_root or self._bids_parent
        problem = sd.space_problem(target, size)
        if problem:
            QMessageBox.warning(self, "Not enough room for the source data",
                                problem + "\n\nNothing was converted. Free some space, or "
                                "turn the copy off or choose an archive in Settings > Convert.")
            return
        if files:
            what = ", ".join(sd.LABELS[k] for k in kinds)
            how = ("as they are" if self._sourcedata_options.form == "copy"
                   else "as one archive per series")
            if not self.confirm_sourcedata(
                    f"The conversion also keeps {files:,} source file(s), "
                    f"{sd.human_size(size)} of {what}, in sourcedata/ {how}.\n\n"
                    f"{sd.IDENTIFIERS_NOTE}"):
                return
        self.start_convert(self._bids_parent)

    def confirm_sourcedata(self, text: str) -> bool:
        """Ask before a conversion that keeps source data (tests replace it)."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Keep the source data")
        box.setText(text)
        convert = box.addButton("Convert", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(convert)
        box.exec()
        return box.clickedButton() is convert

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
        self._spinner.set_busy(True, message="Converting…")
        self._begin_run("Conversion", Path(bids_parent).name)
        worker = ConvertWorker(
            self._model.dataframe(),
            self._output_tsv,
            bids_parent,
            n_jobs=n_jobs,
            on_existing=self._app_settings.convert_on_existing,
            raw_root=self._sources.root if self._sources is not None else self._raw_root,
            skip_residuals=self._app_settings.convert_skip_residuals,
            preserve_curation=self._app_settings.convert_preserve_curation,
            force_edf=self._app_settings.convert_force_edf,
            deface=self._app_settings.convert_deface,
            deface_engine=self._app_settings.convert_deface_engine,
            intended_for=self._app_settings.convert_intended_for,
            b0_field=self._app_settings.convert_b0_field,
            sourcedata=self._app_settings.convert_sourcedata,
            sourcedata_modalities=self._app_settings.convert_sourcedata_modalities,
            parent=self,
        )
        worker.progress.connect(self._on_progress)
        worker.finished_with_result.connect(self._on_convert_finished)
        worker.failed.connect(self._on_convert_failed)
        worker.stopped.connect(self._on_convert_stopped)
        worker.finished.connect(self._on_convert_worker_qt_finished)
        self._convert_worker = worker
        # Flip the Run button to "Stop conversion" and lock out Scan.
        self._set_convert_running(True)
        worker.start()
        return worker

    def _on_convert_finished(self, rc: int, bids_parent: Path) -> None:
        if self._project is not None:
            self._project.append(StageCompleted(
                stage="convert",
                success=(rc == 0),
                summary={"bids_parent": str(bids_parent), "returncode": int(rc)},
            ))
        # Re-walk the output tree so the user sees the just-converted files.
        # In project mode keep the tree rooted at the DATASET (not its parent),
        # matching what set_project showed; only the free-path flow shows the
        # parent (which holds one folder per dataset slug).
        self._output_pane.set_root(
            self._bids_root if self._bids_root is not None else bids_parent
        )
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

    def _converted_datasets(self) -> Optional[list[str]]:
        """The datasets THIS conversion wrote, by folder name.

        The post-convert steps are given the parent directory, because that is
        where convert puts each dataset. Projects are normally kept side by
        side, so that parent is the home of all of them, and a metadata run
        there wrote this project's name into every one of its neighbours.

        In project mode there is exactly one; on a free path the inventory's
        ``dataset`` column says which, and only those are touched.
        """
        if self._bids_root is not None:
            return [self._bids_root.name]
        if self._model is None or "dataset" not in self._model.dataframe().columns:
            # Nothing to filter on. Say so with None rather than an empty list,
            # which would mean "touch nothing" and silently skip the step.
            return None
        names = sorted({
            name for name in (str(v).strip() for v in self._model.dataframe()["dataset"])
            if name and name.lower() not in ("nan", "none")
        })
        return names or None

    def _maybe_run_post_convert(self, bids_parent: Path) -> bool:
        """Kick off the metadata + validate chain if settings say so.

        Returns ``True`` if anything was started. The chain is:
        metadata → validate → dialog. Each step waits for the previous
        worker's ``finished`` before starting; failures fall through to
        the dialog with a warning instead of stopping the chain.
        """
        s = self._app_settings
        if not (s.post_run_metadata or s.post_run_validate or s.post_run_quality):
            return False

        self._post_chain_bids_parent = bids_parent
        # State machine: track which steps are still pending so the
        # chain handler knows what to dispatch next.
        self._post_chain_pending: list[str] = []
        if s.post_run_metadata:
            self._post_chain_pending.append("metadata")
        if s.post_run_validate:
            self._post_chain_pending.append("validate")
        if s.post_run_quality:
            self._post_chain_pending.append("quality")
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
        elif step == "quality":
            self._spinner.set_busy(True, message="Checking image quality…")
            self._start_quality_worker(self._post_chain_bids_parent)

    def _start_quality_worker(self, bids_parent: Path) -> None:
        self._begin_run("Quality check", first=False)
        """The quality check of the images this conversion wrote and nothing
        has checked yet, dataset by dataset."""
        from ..qc import report as RP
        from ..qc import run as RUN

        names = self._converted_datasets()
        roots = ([Path(bids_parent) / n for n in names] if names
                 else [Path(bids_parent)])
        roots = [r for r in roots if (r / "dataset_description.json").is_file()]
        jobs = []
        for root in roots:
            done = {str(r.get("path")) for r in RP.load_all(root)}
            paths = [p for p in RUN.find_images(root)
                     if p.relative_to(root).as_posix() not in done]
            if paths:
                jobs.append((root, paths))
        self._quality_queue = jobs
        self._next_quality_job()

    def _next_quality_job(self) -> None:
        import os

        from ..workers.quality import QualityWorker

        if not getattr(self, "_quality_queue", None):
            self._post_chain_dispatch()
            return
        root, paths = self._quality_queue.pop(0)
        self.log_message.emit(f"Checking the quality of {len(paths)} images in {root.name}")
        from ..viz.settings import qc_config
        from .viz.bridge import SettingsHub

        # The methods and thresholds of Settings > Quality control.
        worker = QualityWorker(root, paths, jobs=max(1, min(4, (os.cpu_count() or 2) // 2)),
                               config=qc_config(SettingsHub.instance().settings),
                               parent=self)
        worker.progress.connect(
            lambda d, t, n: self.log_message.emit(f"Quality checked {d}/{t}: {n}"))
        worker.finished_with_result.connect(lambda rows, r=root: self._on_quality_done(r, rows))
        worker.failed.connect(self._on_quality_failed)
        self._quality_worker = worker
        worker.start()

    def _on_quality_done(self, root: Path, rows) -> None:
        serious = sum(1 for r in rows for f in r.get("findings", [])
                      if f.get("level") in ("warning", "error"))
        self.log_message.emit(
            f"Quality check of {root.name}: {len(rows)} images, {serious} findings to look "
            "at (Editor, Tools, Quality check).")
        if self._project is not None:
            self._project.append(StageCompleted(
                stage="quality", success=True,
                summary={"target": str(root), "images": len(rows), "findings": serious}))
        self._next_quality_job()

    def _on_quality_failed(self, tb: str) -> None:
        self.log_message.emit(tb)
        self._next_quality_job()

    def _start_metadata_worker(self, bids_parent: Path) -> None:
        self._begin_run("Metadata", first=False)
        s = self._app_settings
        worker = MetadataWorker(
            bids_parent,
            datasets=self._converted_datasets(),
            inventory_tsv=self._output_tsv,
            # Metadata-engine name defaults to each BIDS root's folder
            # name (i.e. the dataset slug) when ``name`` is None.
            name=None,
            fill_scope=s.placeholder_scope(),
            generate_companions=s.post_fixup_companions,
            write_citation_file=s.post_fixup_citation,
            tables=s.post_metadata_tables,
            anonymise=s.post_anonymise,
            parent=self,
        )
        worker.progress.connect(self._on_progress)
        worker.finished_with_result.connect(self._on_post_finished)
        worker.failed.connect(self._on_post_failed)
        self._metadata_worker = worker
        worker.start()

    def _start_validate_worker(self, bids_parent: Path) -> None:
        self._begin_run("Validation", first=False)
        s = self._app_settings
        worker = ValidateWorker(
            bids_parent,
            datasets=self._converted_datasets(),
            strict=s.post_validate_strict,
            schema=s.validate_schema_version or None,
            max_rows=s.validate_max_rows,
            flag_todos=s.validate_flag_todos,
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

    def _on_convert_stopped(self) -> None:
        """The conversion halted because the user clicked Stop."""
        self.log_message.emit(
            "Conversion stopped. Subjects committed before the stop were kept."
        )

    def _on_convert_worker_qt_finished(self) -> None:
        # Restore the Run button from "Stop conversion" and re-enable Scan.
        self._set_convert_running(False)
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
        if hasattr(self, "_log_view"):
            self._log_view.repaint_for_palette(pal)
            for name, btn in getattr(self, "_log_buttons", {}).items():
                btn.setIcon(icons.icon(name))
        if hasattr(self, "_raw_pane"):
            self._raw_pane.repaint_for_palette(pal)
        if hasattr(self, "_output_pane"):
            self._output_pane.repaint_for_palette(pal)
        if hasattr(self, "_filter_pane"):
            self._filter_pane.repaint_for_palette(pal)
        # Re-tint the collapse/detach icons on every panel frame.
        for frame in getattr(self, "_panel_frames", []):
            frame._refresh_icons()
        # Re-tint toolbar icons + bottom-tab icons. The cache lives in
        # ``icons``; ``MainWindow`` clears it once per swap, so every
        # call below produces an icon in the newly applied palette.
        # While a scan / convert is running these buttons show the "stop"
        # glyph, so don't clobber it on a mid-run theme swap.
        scan_running = self._scan_worker is not None and self._scan_worker.isRunning()
        convert_running = self._convert_worker is not None and self._convert_worker.isRunning()
        icons.apply_button(self._scan_btn, "stop" if scan_running else "scan")
        icons.apply_button(self._aborts_btn, "highlight")
        icons.apply_button(self._bulk_btn, "bulk_edit")
        icons.apply_button(self._columns_btn, "columns")
        if hasattr(self, "_undo_btn"):
            icons.apply_button(self._undo_btn, "undo")
        if hasattr(self, "_redo_btn"):
            icons.apply_button(self._redo_btn, "redo")
        if hasattr(self, "_revalidate_btn"):
            icons.apply_button(self._revalidate_btn, "revalidate")
        icons.apply_button(
            self._run_btn,
            "stop" if convert_running else "run",
            color=pal.get("primary_btn_text"),
        )
        if hasattr(self, "_overview"):
            self._overview.repaint_for_palette(pal)
            _repolish_combo(self._overview.level_box)
        # The scan-table dropdown (and any QComboBox) does not re-read the
        # re-applied stylesheet on a global ``setStyleSheet`` swap on its own:
        # its popup view is a separate top-level widget Qt does not re-polish.
        # Run the unpolish/polish dance on the combo + its view so no stale
        # dark/light patch survives. See ``_repolish_combo``.
        if hasattr(self, "_scans_combo"):
            _repolish_combo(self._scans_combo)

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

    def reload_app_settings(self) -> None:
        """Re-read persisted settings into the live panel.

        Called by :class:`MainWindow` after the header Settings dialog is
        saved so changed scan / convert defaults take effect without a
        restart. (Scan / convert also reload settings at launch time, so
        this is belt-and-braces for any cached toolbar state.)
        """
        before = self._app_settings.scan_exclusions
        before_classifier = self._app_settings.classifier_config
        before_left_out = set(self._app_settings.scan_drop_entities)
        self._app_settings = AppSettings.load()
        self._sync_confidence_bands()
        # The BIDS preview follows the settings that change what is written.
        self._filter_pane.preview.refresh()
        if self._model is not None and self._app_settings.scan_exclusions != before:
            self._offer_exclusions()
        if self._model is not None and self._app_settings.classifier_config != before_classifier:
            self.offer_reclassify()
        added = [k for k in self._app_settings.scan_drop_entities if k not in before_left_out]
        if self._model is not None and added:
            self.offer_leave_out(added)

    def _sync_confidence_bands(self) -> None:
        from .delegates.inspection_cells import set_confidence_bands

        general = self._app_settings.classifier().general
        set_confidence_bands(general.apply_at, general.check_at)
        if getattr(self, "_table", None) is not None:
            self._table.viewport().update()

    def _series_shims(self) -> dict:
        """Series uid -> shim settings, from the scan's cached features."""
        from ..classifier.acquisition import cache

        if self._output_tsv is None:
            return {}
        return {uid: f.shim for uid, f in cache.read(self._output_tsv).items() if f.shim}

    def reclassify_changes(self):
        """What the current classifier settings would change in the open
        table (from the scan's cached features; no DICOM is read)."""
        from ..classifier.acquisition import cache
        from ..classifier.acquisition import scan as by_parameters

        if self._model is None or self._output_tsv is None:
            return []
        features = cache.read(self._output_tsv)
        if not features:
            return []
        return by_parameters.rescore(self._model.dataframe(), features,
                                     self._app_settings.classifier())

    def offer_reclassify(self, *, ask: bool = True) -> int:
        """After the classification settings change: name the open table
        again, asked first, as one Undo. Returns the rows renamed."""
        config = self._app_settings.classifier()
        changes = self.reclassify_changes()
        if not changes:
            return 0
        renamed = [c for c in changes if c.renamed]
        names = config.general.mode == "applied"
        if ask and renamed and names:
            answer = QMessageBox.question(
                self, "Name the table again",
                f"With the new classification settings, {len(renamed)} series would be "
                "named differently. Name them now? (One Undo takes it back.)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            names = answer == QMessageBox.StandardButton.Yes
        done = self._model.apply_classification(changes, names=names)
        if done:
            self.log_message.emit(f"Named again from the acquisition parameters: {done} "
                                  "series")
        return done

    def offer_leave_out(self, keys, *, ask: bool = True) -> int:
        """Entities newly chosen to leave out of names, offered to the table
        already open: what it would change, asked first, as one Undo.
        Returns the rows changed. (An entity no longer left out cannot be
        given back here: the table does not remember its value. A new scan
        does.)"""
        from ..inventory.leave_out import plan_leave_out

        plan = plan_leave_out(self._model.dataframe(), keys)
        if not plan:
            return 0
        if ask:
            listed = ", ".join(f"{k}-" for k in plan.entities)
            answer = QMessageBox.question(
                self, "Leave entities out of the open table?",
                f"You chose to leave {listed} out of proposed names. In the scan "
                f"table open now, {len(plan.changes)} recording(s) would be renamed.\n\n"
                + plan.summary()
                + "\n\nApply it now? One Undo takes it back. Otherwise it applies "
                  "from the next scan.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return 0
        changed = self._model.apply_leave_out(plan)
        if changed:
            self.log_message.emit(f"Left out of the open table's names: {plan.summary()}")
        return changed

    def _offer_exclusions(self) -> None:
        """Saved scan rules act on the table already open, not only on the
        next scan: say what they would change and apply it as one Undo."""
        from ..classifier.user_rules import plan_exclusions

        plan = plan_exclusions(self._model.dataframe(), self._app_settings.to_exclusions())
        if not plan:
            return
        parts = []
        if plan.exclude:
            parts.append(f"{len(plan.exclude)} series would be excluded")
        if plan.restore:
            parts.append(f"{len(plan.restore)} excluded by a rule you removed would be "
                         "included again")
        answer = QMessageBox.question(
            self, "Apply the scan rules to the open table?",
            "Your scan rules changed. In the scan table open now, " + " and ".join(parts)
            + ".\n\nApply them now? One Undo takes it back. Otherwise they apply "
              "from the next scan.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        changed = self._model.apply_exclusion_plan(plan)
        self.log_message.emit(f"Scan rules applied to the open table: {changed} row(s) changed")

    def _on_current_row_changed(self, current, _previous) -> None:
        """Forward the table's row selection to the Properties panel."""
        if not current.isValid():
            self._properties.request_row(None)
            self._filter_pane.reveal_row(None)
            self._overview.reveal_row(None)
            return
        self._properties.request_row(current.row())
        self._filter_pane.reveal_row(current.row())
        self._overview.reveal_row(current.row())

    def _on_selection_changed(self, *_args) -> None:
        """Enable the bulk-edit button only when ≥ 2 rows are selected."""
        rows = self._selected_rows()
        self._bulk_btn.setEnabled(len(rows) >= 2)
        if len(rows) >= 2:
            self._bulk_btn.setToolTip(
                f"Apply one value to one column across the {len(rows)} "
                "selected rows."
            )

    def _on_table_menu(self, pos) -> None:
        """Right-click on the inspection table: what applies to the rows
        chosen (the clicked row when it is not among them)."""
        if self._model is None:
            return
        index = self._table.indexAt(pos)
        rows = self._selected_rows()
        if index.isValid() and index.row() not in rows:
            self._table.selectRow(index.row())
            rows = [index.row()]
            # Properties follows now, before the menu opens: rebuilt under an
            # open menu, it could close it.
            self._properties.flush()
        if not rows:
            return
        menu = self._row_menu(self._table, rows)
        menu.popup(self._table.viewport().mapToGlobal(pos))
        self._table_menu = menu

    def _on_filter_menu(self, rows: list, where, rename) -> None:
        """Right-click in Filter / structure or recordings: the table's menu
        for the rows under what was clicked, led by renaming the subject or
        session when that is what was clicked."""
        if self._model is None or not rows:
            return
        # The press made a recording current, and so the table's row; let
        # Properties follow before the menu opens, as the table's menu does.
        self._properties.flush()
        menu = self._row_menu(self._filter_pane, rows)
        if rename is not None:
            from . import icons

            kind, label = rename
            act = QAction(icons.icon("bulk_edit"), f"Rename {kind} {label}...", menu)
            act.setToolTip(f"A new label for {label}, on every row under it, as one Undo")
            act.triggered.connect(lambda: self.rename_from_tree(kind, rows, label))
            first = menu.actions()[0] if menu.actions() else None
            menu.insertAction(first, act)
            menu.insertSeparator(first)
        menu.popup(where)
        self._filter_menu = menu

    def _row_menu(self, parent, rows: list):
        """The menu for inventory ``rows``, wherever they were right-clicked.

        One row: what Properties offers for it (preview, spectrum, other
        names, fieldmap runs, eye-tracker pairing), then its DICOM header.
        Any rows: new subject numbers for the people they belong to.
        """
        from ..inventory.renumber import identity
        from . import icons
        from .viz.menus import popup_menu

        menu = popup_menu(parent)
        menu.setToolTipsVisible(True)
        if len(rows) == 1:
            dicom = self._dicom_header_action(menu, rows[0])
            self._properties.add_row_actions(menu, rows[0],
                                             extra=[dicom] if dicom is not None else [])
        if not menu.isEmpty():
            menu.addSeparator()
        df = self._model.dataframe()
        people = len({identity(df, r) for r in rows})
        renumber = menu.addAction(
            icons.icon("bulk_edit"),
            "Give new subject numbers..." if people > 1 else "Give a new subject number...")
        renumber.setToolTip("Number the people of the chosen rows with the next free "
                            "subject numbers")
        renumber.triggered.connect(lambda: self.open_renumber(rows))
        return menu

    def _dicom_header_action(self, menu, row: int):
        """"Show the DICOM header" for a DICOM row (not yet in ``menu``), or
        ``None``."""
        first = self._first_dicom_of(row)
        if first is None:
            return None
        from PyQt6.QtGui import QAction

        from . import icons

        act = QAction(icons.icon("tree_dicom"), "Show the DICOM header", menu)
        act.setToolTip("The header of the first file of this series, in a window")
        act.triggered.connect(lambda: self.show_dicom_header(first))
        return act

    def _first_dicom_of(self, row: int) -> Optional[Path]:
        """The first DICOM of a row's series, where it is now, if any."""
        if self._model is None or self._sources is None:
            return None
        uid = str(self._model.dataframe().iloc[row].get("series_uid", "") or "").strip()
        if not uid:
            return None
        return next((f for f in self._sources.series_files(uid) if f.exists()), None)

    def _on_raw_file_activated(self, path) -> None:
        from ..inventory.mri_dicom import is_dicom_file

        if is_dicom_file(str(path)):
            self.show_dicom_header(Path(path))

    def show_dicom_header(self, path: Path):
        """``path``'s header in the DICOM window (one, reused)."""
        from .widgets.dicom_pane import DicomWindow

        win = getattr(self, "_dicom_window", None)
        if win is None:
            win = DicomWindow(self)
            win.pane.status_message.connect(self.log_message)
            self._dicom_window = win
        win.show_file(Path(path))
        return win

    def _taken_subject_labels(self, rows: list[int]) -> set[str]:
        from ..inventory.renumber import taken_labels

        others = []
        if self._bids_root is not None:
            from ..project import workspace

            for version in workspace.list_versions(self._bids_root):
                if self._active_version_dir is not None and \
                        Path(version.dir) == Path(self._active_version_dir):
                    continue
                try:
                    others.append(pd.read_csv(version.inventory, sep="\t", dtype=str,
                                              usecols=["participant_id"]))
                except (OSError, ValueError):
                    continue
        return taken_labels(self._model.dataframe(), rows, bids_root=self._bids_root,
                            others=others)

    def open_renumber(self, rows: list[int]) -> Optional[int]:
        """Plan new subject numbers for the people of ``rows``, show them,
        apply them as one Undo. Returns the rows changed (``None``: cancelled)."""
        if self._model is None or not rows:
            return None
        from ..inventory.renumber import plan_renumber
        from .renumber_dialog import RenumberDialog, apply_renumbering

        taken = self._taken_subject_labels(rows)
        width = AppSettings.load().scan_index_widths.get("sub")
        people = plan_renumber(self._model.dataframe(), rows, taken,
                               width=int(width) if width else None)
        dlg = RenumberDialog(people, taken, self)
        self._renumber_dialog = dlg
        if dlg.exec() != dlg.DialogCode.Accepted:
            return None
        changed = apply_renumbering(self._model, dlg.mapping())
        self.log_message.emit(f"New subject numbers: {changed} row(s) changed")
        self._flag_existing_subject_rows(self._model.dataframe())
        return changed

    def rename_from_tree(self, kind: str, rows: list[int], current: str,
                         new: Optional[str] = None) -> int:
        """Rename a subject or a session from Filter / structure: every row
        under it, as one Undo. ``new`` skips the question (scripts, tests)."""
        import re

        from PyQt6.QtWidgets import QInputDialog, QLineEdit

        if self._model is None or not rows:
            return 0
        prefix = "sub-" if kind == "subject" else "ses-"
        shown = current[len(prefix):] if current.startswith(prefix) else current
        if new is None:
            new, ok = QInputDialog.getText(
                self, f"Rename {kind}", f"New label for {prefix}{shown} "
                f"({len(rows)} series):", QLineEdit.EchoMode.Normal, shown)
            if not ok:
                return 0
        new = new.strip()
        new = new[len(prefix):] if new.startswith(prefix) else new
        if not re.fullmatch(r"[0-9A-Za-z]+", new or ""):
            QMessageBox.warning(self, f"Rename {kind}",
                                f"'{new}' cannot be a {kind} label: letters and digits only.")
            return 0
        if new == shown:
            return 0
        if kind == "subject" and new in self._taken_subject_labels(rows):
            answer = QMessageBox.question(
                self, "That subject exists",
                f"sub-{new} is already used. Converting will put this person's data into "
                "it (their sessions merge). Use it anyway?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return 0
        entity = "subject" if kind == "subject" else "session"
        changed = 0
        with self._model.edit_group():
            for row in rows:
                if self._model.set_entity(row, entity, new):
                    changed += 1
        self.log_message.emit(f"Renamed {prefix}{shown} to {prefix}{new} ({changed} row(s))")
        if kind == "subject":
            self._flag_existing_subject_rows(self._model.dataframe())
        return changed

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
        with self._model.edit_group():
            dlg.exec()
        # A removal applies row by row, so some rows can be left alone. Say
        # so: "24 rows changed" must never quietly mean "and 6 were not".
        skipped = dlg.skipped_count()
        if skipped:
            self._log_view.append_lines([
                f"Bulk edit: {dlg.changed_count()} row(s) changed, {skipped} "
                f"kept the entity because their datatype requires it."
            ])
        # Status chips + previews refresh via the model's dataChanged
        # signal — the dispatcher's per-row writes already trigger them.

    def _on_scan_finished(self, df: pd.DataFrame, output_tsv: Path) -> None:
        # Project mode: promote the staged scan to a new version with its own
        # isolated curation bundle (+ ScanImported / StageCompleted events), and
        # make it the active project. The shared, Qt-free orchestration is the
        # same engine ``bidsmgr-scan --project`` uses.
        if self._bids_root is not None:
            from ..project import Project
            from ..project.orchestration import import_scan_as_version
            version = import_scan_as_version(
                self._bids_root, Path(output_tsv),
                raw_root=self._raw_root,
                row_ids=tuple(self._collect_row_ids(df)),
                n_rows=len(df),
            )
            self._project = Project.open(version.dir)
            self._properties.set_project(self._project)
            self._active_version_dir = Path(version.dir)
            output_tsv = version.inventory
            # Heads-up: flag scanned rows whose subject id already exists in the
            # dataset (a fresh scan adding to an existing dataset).
            self._flag_existing_subject_rows(df)
        self.load_inventory(df, output_tsv)
        # Legacy standalone-bundle mode (a Project passed to the constructor
        # without a locked dataset root): still log the scan to that bundle.
        if self._bids_root is None and self._project is not None:
            self._project.append(ScanImported(
                inventory_tsv=str(output_tsv),
                row_ids=tuple(self._collect_row_ids(df)),
                raw_root=str(self._raw_root) if self._raw_root else None,
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
        self._refresh_scans_combo()
        self.scan_finished.emit(df, output_tsv)

    def _flag_existing_subject_rows(self, df: pd.DataFrame) -> None:
        """(Re)compute the incremental-collision warning for every scanned row.

        Idempotent: any prior collision note (tagged with
        ``dataset_identity.EXISTING_SUBJECT_TOKEN``) is stripped from every row
        first, then a fresh note is added only to rows whose CURRENT id is still
        on disk. This is what lets the warning clear after the user renames a
        clashing subject (e.g. sub-001 -> sub-002) and clicks Re-validate -- the
        stale "sub-001 ... DIFFERENT subject" note no longer lingers.

        Routes through the warnings system (a non-fatal issues note ->
        warn row -> warnings chip / Issues dialog). When participants.tsv carries
        identity (PatientID / names), the note is precise -- same subject (and
        whether this is a new session), a possible match (one field coincides),
        or a different subject reusing the id (with a suggested free id) --
        otherwise it degrades to a generic heads-up. The merge-aware commit keeps
        existing data safe regardless.
        """
        if "issues" not in df.columns:
            return
        from ..inventory import dataset_identity as di

        token = di.EXISTING_SUBJECT_TOKEN

        # 1) Strip any stale collision note from EVERY row so a rename / re-scan
        #    starts from a clean slate (the note bakes in the old id).
        for i in df.index:
            cur = str(df.at[i, "issues"] or "")
            if token not in cur:
                continue
            kept = [
                seg.strip() for seg in cur.split(" | ")
                if seg.strip() and not seg.strip().startswith(token)
            ]
            df.at[i, "issues"] = " | ".join(kept)

        if self._bids_root is None or "participant_id" not in df.columns:
            return
        existing = {p.name for p in self._bids_root.glob("sub-*") if p.is_dir()}
        if not existing:
            return
        identities = di.read_existing_identities(self._bids_root)
        next_free = di.next_free_subject_id(self._bids_root)
        sessions_cache: dict[str, set] = {}

        def _cell(idx, col: str) -> str:
            return str(df.at[idx, col]).strip() if col in df.columns else ""

        # 2) Re-add a fresh, tagged note to rows whose current id is on disk.
        for i in df.index:
            name = _cell(i, "participant_id")
            if not name or name not in existing:
                continue
            scanned = {
                "patient_id": _cell(i, "PatientID"),
                "given_name": _cell(i, "GivenName"),
                "family_name": _cell(i, "FamilyName"),
            }
            session = _cell(i, "session")
            if name not in sessions_cache:
                sessions_cache[name] = di.existing_sessions(self._bids_root, name)
            note = f"{token}: " + di.classify(
                name, scanned, session, identities.get(name),
                sessions_cache[name], next_free,
            )
            cur = str(df.at[i, "issues"] or "").strip()
            df.at[i, "issues"] = f"{cur} | {note}" if cur else note

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

    def _on_scan_stopped(self) -> None:
        """The scan halted because the user clicked Stop (no TSV written)."""
        self.log_message.emit("Scan stopped. No inventory was written.")

    def _on_worker_qt_finished(self) -> None:
        # Restore the Scan button from "Stop scanning"/"Stopping…" and
        # re-enable Run if a prior scan populated the model. Fires after
        # the success, failure, and stopped paths.
        self._scan_worker = None
        self._spinner.set_busy(False)
        self._set_scan_running(False)
        self._run_btn.setEnabled(self._model is not None and self._model.rowCount() > 0)

    # ------------------------------------------------------------------
    # Scan / Convert run-state button toggles (item: Stop + mutual lock)
    # ------------------------------------------------------------------

    def _set_scan_running(self, running: bool) -> None:
        """Flip the Scan button between "Scan…" and "Stop scanning", and
        lock the Run button out while a scan is in flight."""
        if running:
            self._scan_btn.setText("  Stop scanning")
            icons.apply_button(self._scan_btn, "stop")
            self._scan_btn.setToolTip("Stop the running scan")
            self._scan_btn.setEnabled(True)
            self._run_btn.setEnabled(False)  # mutual exclusion
        else:
            self._scan_btn.setText("  Scan…")
            icons.apply_button(self._scan_btn, "scan")
            self._scan_btn.setToolTip("")
            self._scan_btn.setEnabled(True)
            # Run re-enabled by the caller based on model state.

    def _set_convert_running(self, running: bool) -> None:
        """Flip the Run button between "Run conversion" and "Stop
        conversion", and lock the Scan button out while converting."""
        if running:
            self._run_btn.setText("  Stop conversion")
            icons.apply_button(self._run_btn, "stop")
            self._run_btn.setToolTip("Stop the running conversion")
            self._run_btn.setEnabled(True)
            self._scan_btn.setEnabled(False)  # mutual exclusion
        else:
            self._run_btn.setText("  Run conversion")
            icons.apply_button(self._run_btn, "run", color=CUR().get("primary_btn_text"))
            self._run_btn.setToolTip("")
            self._scan_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Status chips
    # ------------------------------------------------------------------

    def _on_revalidate_clicked(self) -> None:
        """Force a full re-validation of the inventory and refresh the tallies."""
        if self._model is None:
            return
        # Recompute the incremental-collision warnings against the CURRENT row
        # ids + on-disk subjects (so a rename like sub-001 -> sub-002 clears the
        # stale "already on disk / DIFFERENT subject" note). This mutates the
        # model's live DataFrame; revalidate_all then rebuilds row states from it.
        self._flag_existing_subject_rows(self._model.dataframe())
        self._model.revalidate_all()
        # ``revalidate_all`` emits dataChanged (which the listeners below also
        # react to), but call them directly so the refresh is immediate and
        # self-contained even if a listener was ever detached.
        self._update_status_chips()
        self.log_message.emit("Re-validated inventory against the BIDS schema")

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
