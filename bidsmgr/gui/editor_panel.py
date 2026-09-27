"""Post-conversion BIDS editor view (M6).

Visual reference: ``inspector_proto/proto.py`` ``EditorView`` (3-pane
horizontal splitter — BIDS tree | sidecar form | validation panel).

Through Step 3 the panel hosts:

* a toolbar with ``Open BIDS root…``, ``Validate dataset``, status
  chips, a busy spinner, and (disabled) file / folder validate stubs;
* a :class:`PathBar` showing the active dataset;
* the 3-pane horizontal splitter with the
  :class:`BidsTreePane` on the left (badges stamped from the in-memory
  :class:`bidsmgr.editor.types.ValidationReport`).

Center and right panes remain placeholders for Steps 4-6.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QMenu,
    QDialog,
    QFileDialog,
    QFrame,
    QMessageBox,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..deface import run as deface_run
from ..editor.types import FileVerdict, Severity, ValidationReport
from ..workers import FileReportWorker, FolderReportWorker, ReportWorker
from . import icons
from .combo_popup import menu_section, round_menu
from .widgets.bidsignore_pane import BidsIgnorePane
from .widgets.citation_pane import CitationPane
from .widgets import (
    BidsTreePane,
    BusySpinner,
    Chip,
    NiftiViewerPane,
    PanelFrame,
    PathBar,
    MrsViewerPane,
    RecordingViewerPane,
    SidecarFormPane,
    TsvViewerPane,
    ValidationPane,
    VSep,
    is_recording_path,
)
from .widgets.mrs_spectrum import is_mrs_path

log = logging.getLogger(__name__)


class EditorPanel(QWidget):
    """Post-convert BIDS editor (Editor pill in the top header).

    Emits :pyattr:`log_message` for every progress line so
    :class:`MainWindow` can mirror Editor activity into the status bar
    just like it already does for the Converter.
    """

    log_message = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("editor-panel")

        self._report: Optional[ValidationReport] = None
        self._report_worker: Optional[ReportWorker] = None
        # Partial-validate workers (toolbar's Validate file / folder
        # buttons). Kept on self so they're not garbage-collected
        # mid-flight.
        self._partial_worker: Optional[QObject] = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self._toolbar = self._build_toolbar()
        v.addWidget(self._toolbar)

        self._path_bar = PathBar("Dataset", "(none)", ok=False)
        self._path_bar.change_button.clicked.connect(self._on_change_root)
        v.addWidget(self._path_bar)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(1)
        self._splitter.setChildrenCollapsible(False)

        self._tree_pane = BidsTreePane()
        self._tree_pane.file_selected.connect(self._on_file_selected)
        self._tree_pane.rename_requested.connect(self._on_rename)
        self._tree_pane.entities_requested.connect(self._on_edit_entities)
        self._tree_pane.delete_requested.connect(self._on_delete)
        self._tree_pane.deface_requested.connect(self._on_deface)
        self._tree_pane.deface_compare_requested.connect(
            self._on_deface_compare
        )
        self._tree_pane.deface_revert_requested.connect(self._on_deface_revert)
        self._tree_pane.strip_requested.connect(self._on_strip)
        self._tree_pane.compare_requested.connect(self._on_compare_images)
        self._tree_pane.links_requested.connect(self._on_links)
        # Drive the Validate file/folder button enable-state from the
        # tree selection — file → file button, folder → folder button.
        self._tree_pane.file_selected.connect(
            self._sync_validate_buttons_from_selection
        )
        # Center pane is itself a stack — different viewer per file
        # kind. Index 0 = JSON sidecar (default for unsupported kinds
        # too — its empty-state hint guides the user to the JSON peer).
        # Index 1 = TSV table. Index 2 = NIfTI 2-D slice viewer.
        self._sidecar_form = SidecarFormPane()
        self._tsv_viewer = TsvViewerPane()
        self._nifti_viewer = NiftiViewerPane()
        self._recording_viewer = RecordingViewerPane()
        self._mrs_viewer = MrsViewerPane()
        self._center_stack = QStackedWidget()
        self._sidecar_form.apply_to_others_requested.connect(
            self._on_apply_field_to_others
        )
        self._sidecar_form.explain_requested.connect(self._on_explain_field)
        self._center_stack.addWidget(self._sidecar_form)
        self._center_stack.addWidget(self._tsv_viewer)
        self._center_stack.addWidget(self._nifti_viewer)
        self._center_stack.addWidget(self._recording_viewer)
        self._center_stack.addWidget(self._mrs_viewer)
        self._bidsignore_pane = BidsIgnorePane()
        self._center_stack.addWidget(self._bidsignore_pane)
        self._citation_pane = CitationPane()
        self._center_stack.addWidget(self._citation_pane)
        # Threaded panes drive the toolbar busy spinner + status bar.
        self._tsv_viewer.loading_changed.connect(self._on_pane_loading)
        self._recording_viewer.loading_changed.connect(self._on_pane_loading)
        self._recording_viewer.status_message.connect(self.log_message)
        # Undo/redo toolbar buttons follow the active editable pane.
        self._sidecar_form.history_changed.connect(self._sync_undo_redo)
        self._tsv_viewer.history_changed.connect(self._sync_undo_redo)
        self._center_stack.currentChanged.connect(self._sync_undo_redo)
        self._validation_pane = ValidationPane()
        self._validation_pane.fix_requested.connect(self._on_fix_requested)
        self._validation_pane.fix_group_requested.connect(
            self._on_fix_group_requested
        )
        self._validation_pane.accept_requested.connect(
            self._on_accept_finding
        )
        self._validation_pane.highlight_all_requested.connect(
            self._on_highlight_all_requested
        )
        # The BIDS tree folds to the left, the validation pane to the right,
        # and the center viewer grows into the freed space. The viewer is
        # not collapsible (it's the main surface) but IS detachable so the
        # JSON / TSV / NIfTI view can pop out into its own window.
        self._tree_frame = PanelFrame(self._tree_pane, "BIDS tree", edge="left")
        self._center_frame = PanelFrame(
            self._center_stack, "Viewer", collapsible=False, detachable=True,
            hide_inner_header=False,
        )
        self._validation_frame = PanelFrame(
            self._validation_pane, "Validation", edge="right",
        )
        self._panel_frames = [
            self._tree_frame, self._center_frame, self._validation_frame,
        ]
        self._splitter.addWidget(self._tree_frame)
        self._splitter.addWidget(self._center_frame)
        self._splitter.addWidget(self._validation_frame)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)
        self._splitter.setSizes([320, 700, 380])
        # Collapsing the tree / validation hands width to the center viewer.
        self._tree_frame.attach_splitter(self._splitter, grow_target=self._center_frame)
        self._validation_frame.attach_splitter(self._splitter, grow_target=self._center_frame)
        v.addWidget(self._splitter, 1)

        # Restore last-opened root if available.
        from .app_settings import AppSettings
        settings = AppSettings.load()
        if settings.editor_bids_root:
            root = Path(settings.editor_bids_root)
            if root.exists() and root.is_dir():
                self._set_root(root, persist=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tree_pane(self) -> BidsTreePane:
        return self._tree_pane

    def current_root(self) -> Optional[Path]:
        return self._tree_pane.root()

    def current_report(self) -> Optional[ValidationReport]:
        return self._report

    def start_dataset_validation(self) -> None:
        """Kick a :class:`ReportWorker` against the current root.

        Public so tests (and a future keyboard shortcut) can trigger
        the same flow without synthesising a button click.
        """
        root = self.current_root()
        if root is None:
            return
        if self._report_worker is not None and self._report_worker.isRunning():
            return  # debounce concurrent runs
        self._set_busy(True, "Validating dataset…")
        schema, max_rows, flag_todos = self._validation_engine_opts()
        worker = ReportWorker(
            root, strict=self._strict_btn.isChecked(),
            schema=schema, max_rows=max_rows, flag_todos=flag_todos, parent=self,
        )
        worker.progress.connect(self._on_progress)
        worker.finished_with_report.connect(self._on_report_ready)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(self._on_worker_finished)
        self._report_worker = worker
        worker.start()

    def start_file_validation(self) -> None:
        """Re-validate the currently-selected file (layer 1 only)."""
        root = self.current_root()
        if root is None:
            return
        target = self._tree_pane_current_path()
        if target is None or not target.is_file():
            return
        if self._partial_worker is not None and getattr(
            self._partial_worker, "isRunning", lambda: False,
        )():
            return
        self._set_busy(True, f"Validating {target.name}…")
        schema, max_rows, flag_todos = self._validation_engine_opts()
        worker = FileReportWorker(
            root, target, schema=schema, max_rows=max_rows,
            flag_todos=flag_todos, parent=self,
        )
        self._wire_partial_worker(worker)
        self._partial_worker = worker
        worker.start()

    def start_folder_validation(self) -> None:
        """Re-validate the currently-selected folder (layer 1 only)."""
        root = self.current_root()
        if root is None:
            return
        target = self._tree_pane_current_path()
        if target is None or not target.is_dir():
            return
        if self._partial_worker is not None and getattr(
            self._partial_worker, "isRunning", lambda: False,
        )():
            return
        self._set_busy(True, f"Validating {target.name}/…")
        schema, max_rows, flag_todos = self._validation_engine_opts()
        worker = FolderReportWorker(
            root, target, schema=schema, max_rows=max_rows,
            flag_todos=flag_todos, parent=self,
        )
        self._wire_partial_worker(worker)
        self._partial_worker = worker
        worker.start()

    def _wire_partial_worker(self, worker) -> None:
        worker.progress.connect(self._on_progress)
        worker.finished_with_verdicts.connect(self._on_partial_ready)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(self._on_partial_worker_finished)

    # ------------------------------------------------------------------
    # Toolbar / panes
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("toolbar")
        bar.setFixedHeight(44)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 6, 14, 6)
        lay.setSpacing(8)

        self._open_btn = QPushButton("  Open BIDS root…")
        self._open_btn.setObjectName("tb-btn")
        icons.apply_button(self._open_btn, "open_folder")
        self._open_btn.clicked.connect(self._on_change_root)
        lay.addWidget(self._open_btn)

        lay.addWidget(VSep())

        # Undo / Redo for in-memory edits to the active center pane (JSON
        # sidecar + TSV table). They delegate to whichever editable pane is
        # showing; disabled when nothing is undoable (or a NIfTI is shown).
        self._undo_btn = QPushButton("  Undo")
        self._undo_btn.setObjectName("tb-btn-ghost")
        icons.apply_button(self._undo_btn, "undo")
        self._undo_btn.setToolTip("Undo the last edit (JSON / TSV)")
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self._on_editor_undo)
        lay.addWidget(self._undo_btn)

        self._redo_btn = QPushButton("  Redo")
        self._redo_btn.setObjectName("tb-btn-ghost")
        icons.apply_button(self._redo_btn, "redo")
        self._redo_btn.setToolTip("Redo the last undone edit")
        self._redo_btn.setEnabled(False)
        self._redo_btn.clicked.connect(self._on_editor_redo)
        lay.addWidget(self._redo_btn)

        lay.addWidget(VSep())

        # Validate file / folder reuse the dataset report — but run
        # only layer 1 (per-file checks). Layer 2 is dataset-wide and
        # is reserved for the dataset-level button.
        self._validate_file_btn = QPushButton("  Validate file")
        self._validate_file_btn.setObjectName("tb-btn")
        icons.apply_button(self._validate_file_btn, "file_check")
        self._validate_file_btn.setEnabled(False)
        self._validate_file_btn.setToolTip(
            "Re-validate the file currently selected in the BIDS "
            "tree. Updates only that file's badge + Validation pane "
            "section; the dataset-wide report stays intact."
        )
        self._validate_file_btn.clicked.connect(self.start_file_validation)
        lay.addWidget(self._validate_file_btn)

        self._validate_folder_btn = QPushButton("  Validate folder")
        self._validate_folder_btn.setObjectName("tb-btn")
        icons.apply_button(self._validate_folder_btn, "folder_check")
        self._validate_folder_btn.setEnabled(False)
        self._validate_folder_btn.setToolTip(
            "Re-validate every file under the folder currently "
            "selected in the BIDS tree. Skips folder/dataset-level "
            "checks — use “Validate dataset” for the full pass."
        )
        self._validate_folder_btn.clicked.connect(self.start_folder_validation)
        lay.addWidget(self._validate_folder_btn)

        self._validate_dataset_btn = QPushButton("  Validate dataset")
        self._validate_dataset_btn.setObjectName("tb-btn")
        icons.apply_button(self._validate_dataset_btn, "dataset")
        self._validate_dataset_btn.setEnabled(False)
        self._validate_dataset_btn.clicked.connect(self.start_dataset_validation)
        lay.addWidget(self._validate_dataset_btn)

        # Deep-checks toggle — when on, "Validate dataset" reads NIfTI
        # headers and file contents (slower, more thorough); when off it
        # runs the fast structural pass used for live revalidation. Maps
        # to the validator's read-headers mode. State persists via
        # AppSettings (the ``editor_strict_validate`` key, kept for
        # back-compat).
        from .app_settings import AppSettings
        self._strict_btn = QPushButton("  Deep checks")
        self._strict_btn.setObjectName("tb-btn-toggle")
        icons.apply_button(self._strict_btn, "strict")
        self._strict_btn.setCheckable(True)
        self._strict_btn.setChecked(AppSettings.load().editor_strict_validate)
        self._strict_btn.setToolTip(
            "Deep validation checks\n\n"
            "When ON, “Validate dataset” also opens NIfTI image headers to "
            "catch truncated or corrupt .nii / .nii.gz files. This is the "
            "only extra file read, and the only thing this toggle changes.\n\n"
            "When OFF, the validator runs the fast structural pass — file "
            "naming, entities, sidecar fields, TSV columns, associations. "
            "This is the mode used for live revalidation as you edit.\n\n"
            "Note: EEG / MEG / iEEG recordings are validated from their "
            "sidecars + channels.tsv and are never opened, so this toggle "
            "has no effect on a recordings-only dataset.\n\n"
            "Tip: leave this OFF while editing and flip it ON once before "
            "a final review."
        )
        self._strict_btn.toggled.connect(self._on_strict_toggled)
        lay.addWidget(self._strict_btn)

        # Everything that acts on the dataset as a whole, behind one menu.
        # Four separate toolbar buttons for four rarely-used actions crowded
        # out the ones people press constantly, and the list is going to grow.
        # The validation buttons deliberately stay outside: they are the
        # Editor's main verb, not a tool.
        self._tools_btn = QPushButton("  Tools")
        self._tools_btn.setObjectName("tb-btn")
        icons.apply_button(self._tools_btn, "tools")
        self._tools_btn.setToolTip(
            "Things that act on the whole dataset: repairs, renaming, and "
            "the dashboard. Nothing here writes until you choose it, and "
            "everything can be undone."
        )
        self._tools_btn.setEnabled(False)
        self._tools_menu = QMenu(self._tools_btn)
        round_menu(self._tools_menu)
        self._tools_btn.setMenu(self._tools_menu)

        # Grouped by PURPOSE, with headings rather than sub-menus. Sixteen
        # flat verbs had become a list nobody could scan: "Index widths" and
        # "Put the face back" sat next to each other, and the only thing that
        # told you which one you wanted was reading all sixteen. Headings
        # answer "what am I trying to do" first and keep every tool one click
        # away, which a sub-menu would not.
        menu_section(self._tools_menu, "Look at the dataset")

        self._dashboard_action = self._tools_menu.addAction("Dashboard")
        self._dashboard_action.setToolTip(
            "What is actually in this dataset: subjects, sessions, "
            "modalities, how much of the metadata is answered, and where "
            "the findings are."
        )
        self._dashboard_action.triggered.connect(self._on_dashboard)

        self._compare_action = self._tools_menu.addAction("Compare images...")
        self._compare_action.setToolTip(
            "Put two NIfTI images side by side with one set of controls: "
            "crosshair, slice, plane, 3-D camera, effects and cut plane all "
            "stay together. Any two images, not just a defacing pair."
        )
        self._compare_action.triggered.connect(self._on_compare_images)

        menu_section(self._tools_menu, "Check and repair")

        self._coherence_action = self._tools_menu.addAction("Check coherence...")
        self._coherence_action.setToolTip(
            "Find where the dataset's files stop agreeing with each other: "
            "a scans row naming a file that is gone, a participants row for "
            "a deleted subject, a link pointing at nothing, a sidecar with "
            "no recording. Read-only until you choose a repair."
        )
        self._coherence_action.triggered.connect(self._on_coherence)

        self._links_action = self._tools_menu.addAction(
            "References (IntendedFor, Sources...)..."
        )
        self._links_action.setToolTip(
            "Which files each file points at, and whether that is right. "
            "Lists every IntendedFor, AssociatedEmptyRoom, Sources and "
            "AnatomicalImage in the dataset with its status: correct, "
            "pointing at a missing file, or disagreeing with the "
            "acquisition times. Link by ticking files, not by typing paths."
        )
        self._links_action.triggered.connect(self._on_links)

        self._fixups_action = self._tools_menu.addAction("Fix ups...")
        self._fixups_action.setToolTip(
            "Generate the companion files recordings are missing, write "
            "CITATION.cff, move repeated metadata up, and mark what is "
            "unanswered."
        )
        self._fixups_action.triggered.connect(self._on_fixups)

        menu_section(self._tools_menu, "Names and structure")

        self._rename_action = self._tools_menu.addAction("Rename entity...")
        self._rename_action.setToolTip(
            "Rename a subject, session, task or any other entity across "
            "the whole dataset, including the references to it inside "
            "IntendedFor, the scans tables and participants.tsv. You see "
            "the full plan before anything moves."
        )
        self._rename_action.triggered.connect(self._on_rename)

        self._replace_action = self._tools_menu.addAction(
            "Find and replace a value..."
        )
        self._replace_action.setToolTip(
            "Find every file where an entity has one value and give it "
            "another, inside the whole dataset, one subject or one session. "
            "The values are listed with how many files carry each, so there "
            "is nothing to guess."
        )
        self._replace_action.triggered.connect(self._on_replace_value)

        self._pad_action = self._tools_menu.addAction("Index widths...")
        self._pad_action.setToolTip(
            "Give every index entity a consistent width, so run-1 reads as "
            "run-01 throughout. Says where the widths already disagree, "
            "which is the case worth acting on. Both spellings are valid "
            "BIDS, so this is a house style."
        )
        self._pad_action.triggered.connect(self._on_pad_values)

        self._entities_action = self._tools_menu.addAction(
            "Add or remove an entity..."
        )
        self._entities_action.setToolTip(
            "Give a recording an entity the schema allows it, or take an "
            "optional one away. Only what the standard permits for those "
            "files is offered, and the entity lands in the position the "
            "standard puts it in. Starts on the tree selection, or on the "
            "whole dataset when there is none."
        )
        self._entities_action.triggered.connect(
            lambda: self._on_edit_entities(session_mode=False)
        )

        self._sessions_action = self._tools_menu.addAction("Sessions...")
        self._sessions_action.setToolTip(
            "Create a session for the selected recordings, or take them back "
            "out of the one they are in. The scans table travels to the level "
            "BIDS puts it at and every reference follows. Starts on the tree "
            "selection, or on the whole dataset when there is none."
        )
        self._sessions_action.triggered.connect(
            lambda: self._on_edit_entities(session_mode=True)
        )

        menu_section(self._tools_menu, "Identifiable data")

        self._deface_action = self._tools_menu.addAction("Deface...")
        self._deface_action.setToolTip(
            "Remove the face from anatomical images, so the dataset can be "
            "shared without the participants being identifiable. One "
            "undoable step; you see exactly which images, and which were "
            "skipped, before anything is written."
        )
        self._deface_action.triggered.connect(self._on_deface)

        self._strip_action = self._tools_menu.addAction("Remove the skull...")
        self._strip_action.setToolTip(
            "Keep only the brain. Removes the face and everything else "
            "outside it, so the result is a derivative rather than raw data "
            "and is written to derivatives/. One undoable step."
        )
        self._strip_action.triggered.connect(self._on_strip)

        self._deface_compare_action = self._tools_menu.addAction(
            "Compare with the original..."
        )
        self._deface_compare_action.setToolTip(
            "Show a selected image before defacing and after it, side by "
            "side, with one crosshair between them. Every defacing tool tells "
            "you to check the result; this is how you do it."
        )
        self._deface_compare_action.triggered.connect(self._on_deface_compare)

        self._deface_revert_action = self._tools_menu.addAction(
            "Put the face back..."
        )
        self._deface_revert_action.setToolTip(
            "Restore defaced images from the undefaced copies kept in "
            "sourcedata/ or in the edit history. Undo only reaches the last "
            "operation; this works however many edits came afterwards."
        )
        self._deface_revert_action.triggered.connect(self._on_deface_revert)

        menu_section(self._tools_menu, "Remove")

        self._delete_action = self._tools_menu.addAction("Delete...")
        self._delete_action.setToolTip(
            "Delete recordings, datatypes or sessions. The *_scans.tsv rows, "
            "the IntendedFor entries, a participants row left describing "
            "nothing and any emptied folder go with them, as one undoable "
            "step. Starts on the tree selection; the whole dataset is "
            "deliberately not one of the scopes."
        )
        self._delete_action.triggered.connect(self._on_delete)

        # Only meaningful for a dataset this tool did not convert, so it
        # hides itself once the dataset carries a project bundle.
        menu_section(self._tools_menu, "History")
        self._adopt_action = self._tools_menu.addAction("Track changes")
        self._adopt_action.setToolTip(
            "This dataset was not converted here, so there is no record "
            "of what it looked like before you started editing. Tracking "
            "writes that baseline into .bidsmgr/ and changes nothing "
            "else, so edits become reversible."
        )
        self._adopt_action.setVisible(False)
        self._adopt_action.triggered.connect(self._on_adopt)

        lay.addWidget(self._tools_btn)

        lay.addWidget(VSep())

        # Status chips — kept hidden until a report lands. Each chip
        # opens the file-issues dialog filtered by the matching
        # severity (same "jump to" pattern as the Converter's toolbar).
        self._chip_ok = Chip("0 valid", "success")
        self._chip_warn = Chip("0 warnings", "warn")
        self._chip_err = Chip("0 errors", "err")
        for chip, sev in (
            (self._chip_ok, "ok"),
            (self._chip_warn, "warn"),
            (self._chip_err, "err"),
        ):
            chip.setVisible(False)
            chip.set_clickable(True)
            chip.clicked.connect(
                lambda s=sev: self._open_issues_dialog(s)
            )
            lay.addWidget(chip)

        lay.addStretch(1)

        # Busy spinner + activity label, mirroring the Converter toolbar.
        self._spinner = BusySpinner()
        lay.addWidget(self._spinner)

        return bar

    # ------------------------------------------------------------------
    # Strict-validation toggle
    # ------------------------------------------------------------------

    def _on_strict_toggled(self, checked: bool) -> None:
        from .app_settings import AppSettings
        AppSettings.remember_editor_strict_validate(checked)

    @staticmethod
    def _allowed_severities() -> set:
        """Severities the Editor displays, from the ``validate_show`` setting.

        Applied uniformly to the tree dots, the status chips, and the
        Validation pane so "errors only" / "warnings only" hides the rest
        everywhere (not just in the pane list).
        """
        from .app_settings import AppSettings
        show = AppSettings.load().validate_show
        if show == "error":
            return {Severity.ERR}
        if show == "warning":
            return {Severity.WARN}
        return {Severity.ERR, Severity.WARN}

    @staticmethod
    def _validation_engine_opts() -> tuple[Optional[str], int, bool]:
        """Resolve the bidsval schema + max-rows + flag-todos knobs from settings.

        Threaded into every validation worker so the engine never reads
        settings itself (it stays Qt-free). ``schema`` is ``None`` for
        bidsval's bundled default.
        """
        from .app_settings import AppSettings
        s = AppSettings.load()
        return (
            s.validate_schema_version or None,
            int(s.validate_max_rows),
            bool(s.validate_flag_todos),
        )

    # ------------------------------------------------------------------
    # Open-root flow
    # ------------------------------------------------------------------

    def _on_change_root(self) -> None:
        from .app_settings import AppSettings
        current = self.current_root()
        start = str(current) if current else (
            AppSettings.load().editor_bids_root or str(Path.home())
        )
        chosen = QFileDialog.getExistingDirectory(
            self, "Open BIDS root", start,
        )
        if not chosen:
            return
        self._set_root(Path(chosen), persist=True)

    def _set_root(self, path: Path, *, persist: bool) -> None:
        self._tree_pane.set_root(path)
        self._tree_pane.clear_badges()
        self._path_bar.set_value(str(path), ok=True)
        # Switching roots invalidates any stored report + form state.
        self._report = None
        self._sidecar_form.set_file(None, None, None)
        self._tsv_viewer.set_file(None, None)
        self._nifti_viewer.set_file(None, None)
        self._recording_viewer.set_file(None, None)
        self._center_stack.setCurrentWidget(self._sidecar_form)
        self._validation_pane.set_report(None)
        self._validation_pane.set_current_file(None, None)
        self._hide_chips()
        # Enable dataset-level validation now that we have a root.
        self._validate_dataset_btn.setEnabled(True)
        self._tools_btn.setEnabled(True)
        self._refresh_deface_action()
        self._refresh_adopt_button()
        if persist:
            from .app_settings import AppSettings
            AppSettings.remember_editor_bids_root(path)

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_progress(self, message: str) -> None:
        self.log_message.emit(message)

    def _on_report_ready(
        self, report: ValidationReport, root: Path,
    ) -> None:
        self._report = report
        self._stamp_tree_badges(report)
        self._update_chips(report)
        # Refresh the form for whatever file the user already has
        # selected — the FileVerdict for it just became available.
        current = self._sidecar_form.current_file()
        if current is not None:
            self._sidecar_form.set_file(current, self.current_root(), report)
        # The validation panel binds against the in-memory report.
        self._validation_pane.set_report(report)
        self._validation_pane.set_current_file(current, self.current_root())
        self.log_message.emit(
            f"Validation done — {report.counts.get('ok', 0)} ok, "
            f"{report.counts.get('warn', 0)} warn, "
            f"{report.counts.get('err', 0)} err"
        )
        del root  # only kept on the signal for caller correlation

    def _on_worker_failed(self, traceback_text: str) -> None:
        self.log_message.emit(f"Validation failed:\n{traceback_text}")

    def _on_worker_finished(self) -> None:
        self._set_busy(False)
        # Re-enable the button only if a root is still loaded.
        self._validate_dataset_btn.setEnabled(self.current_root() is not None)
        self._report_worker = None

    def _on_partial_ready(
        self,
        verdicts: list[FileVerdict],
        bids_root: Path,
        target_path: Path,
    ) -> None:
        """Merge per-file or per-folder verdicts into the in-memory report.

        Existing FileVerdicts for the same paths are replaced; new ones
        append. Dataset / folder issues stay untouched (they belong to
        the dataset-wide pass).
        """
        del bids_root  # we already cache it as self._current_root
        if not verdicts:
            return
        # Ensure we have a report to merge into.
        if self._report is None:
            self._report = ValidationReport(bids_root=self.current_root())
        existing_by_path = {fv.path: i for i, fv in enumerate(self._report.files)}
        for fv in verdicts:
            if fv.path in existing_by_path:
                self._report.files[existing_by_path[fv.path]] = fv
            else:
                existing_by_path[fv.path] = len(self._report.files)
                self._report.files.append(fv)
        # Re-rollup severity + counts (cheap to recompute).
        self._recompute_report_summary(self._report)
        # Refresh the UI.
        self._stamp_tree_badges(self._report)
        self._update_chips(self._report)
        self._validation_pane.set_report(self._report)
        self._validation_pane.set_current_file(
            self._sidecar_form.current_file(), self.current_root(),
        )
        # Sidecar pane re-binds against the now-fresh verdict.
        current = self._sidecar_form.current_file()
        if current is not None:
            self._sidecar_form.set_file(
                current, self.current_root(), self._report,
            )
        self.log_message.emit(
            f"Validation done — {len(verdicts)} file"
            f"{'s' if len(verdicts) != 1 else ''} re-checked"
        )
        del target_path

    def _on_partial_worker_finished(self) -> None:
        self._set_busy(False)
        self._partial_worker = None
        self._sync_validate_buttons_from_selection(
            self._tree_pane_current_path(),
        )

    def _tree_pane_current_path(self) -> Optional[Path]:
        """Return the absolute path of the BIDS tree's current selection."""
        from .widgets.bids_tree_pane import PATH_ROLE
        item = self._tree_pane._tree.currentItem()
        if item is None:
            return None
        raw = item.data(0, PATH_ROLE)
        return Path(raw) if raw else None

    def _sync_validate_buttons_from_selection(
        self, path: Optional[Path],
    ) -> None:
        """Enable Validate file when a file is selected, Validate
        folder when a folder is selected.

        Both stay disabled until a BIDS root is open. A partial-validate
        already in flight also disables them (debounce).
        """
        has_root = self.current_root() is not None
        running = self._partial_worker is not None and getattr(
            self._partial_worker, "isRunning", lambda: False,
        )()
        from .theme_manager import CUR
        pal = CUR()
        if path is None or not has_root or running:
            self._validate_file_btn.setEnabled(False)
            self._validate_folder_btn.setEnabled(False)
            # No selection: both icons stay accent (default).
            icons.apply_button(self._validate_file_btn, "file_check",
                               color=pal.get("accent"))
            icons.apply_button(self._validate_folder_btn, "folder_check",
                               color=pal.get("accent"))
            return
        file_selected = path.is_file()
        folder_selected = path.is_dir()
        self._validate_file_btn.setEnabled(file_selected)
        self._validate_folder_btn.setEnabled(folder_selected)
        # Recolor icons: the button matching the selected kind goes
        # green (success); the other stays accent blue.
        icons.apply_button(
            self._validate_file_btn, "file_check",
            color=pal.get("success" if file_selected else "accent"),
        )
        icons.apply_button(
            self._validate_folder_btn, "folder_check",
            color=pal.get("success" if folder_selected else "accent"),
        )

    @staticmethod
    def _recompute_report_summary(report: ValidationReport) -> None:
        """Reset ``severity`` and ``counts`` after files were mutated.

        Counts are PER-FINDING (every error/warning issue), matching the
        adapter + bidsval; ``ok`` is the number of clean files.
        """
        all_issues = [i for i in report.dataset_issues if not i.mirrored]
        for issues in report.folder_issues.values():
            all_issues.extend(i for i in issues if not i.mirrored)
        for f in report.files:
            all_issues.extend(i for i in f.issues if not i.mirrored)
        report.counts = {
            "ok": sum(1 for f in report.files if f.severity is Severity.OK),
            "warn": sum(1 for i in all_issues if i.severity is Severity.WARN),
            "err": sum(1 for i in all_issues if i.severity is Severity.ERR),
        }
        if any(i.severity is Severity.ERR for i in all_issues):
            report.severity = Severity.ERR
        elif any(i.severity is Severity.WARN for i in all_issues):
            report.severity = Severity.WARN
        else:
            report.severity = Severity.OK

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stamp_tree_badges(self, report: ValidationReport) -> None:
        """Map :class:`FileVerdict` severities onto tree rows.

        ``FileVerdict.path`` is **relative to bids_root**; the tree
        items carry absolute paths, so we resolve here before handing
        off to the pane.
        """
        root = self.current_root()
        if root is None:
            return
        allowed = self._allowed_severities()
        severities: dict[Path, str] = {}
        counts: dict[Path, tuple[int, int]] = {}
        for fv in report.files:
            # Badge = worst ALLOWED issue on the file; clean / filtered-out
            # files get the green "ok" dot. So "errors only" leaves only error
            # files red and everything else green.
            issues = [i for i in fv.issues if i.severity in allowed]
            # Mirrored findings are the same finding shown on the editable
            # sidecar as well as on the data file. Counting both would double
            # every number in the tree.
            countable = [
                i for i in issues if not getattr(i, "mirrored", False)
            ]
            n_err = sum(1 for i in countable if i.severity is Severity.ERR)
            n_warn = sum(1 for i in countable if i.severity is Severity.WARN)
            if any(i.severity is Severity.ERR for i in issues):
                value = "err"
            elif any(i.severity is Severity.WARN for i in issues):
                value = "warn"
            else:
                value = "ok"
            absolute = (root / fv.path).resolve() if not fv.path.is_absolute() \
                else fv.path
            severities[absolute] = value
            if n_err or n_warn:
                counts[absolute] = (n_err, n_warn)
        self._tree_pane.set_badges(severities, counts)

    def _update_chips(self, report: ValidationReport) -> None:
        counts = report.counts
        allowed = self._allowed_severities()
        self._chip_ok.setText(f"{counts.get('ok', 0)} valid")
        self._chip_warn.setText(f"{counts.get('warn', 0)} warnings")
        self._chip_err.setText(f"{counts.get('err', 0)} errors")
        # The "Show findings" filter hides the chips for severities it excludes
        # (so "errors only" doesn't surface a warnings counter). "valid" stays.
        self._chip_ok.setVisible(True)
        self._chip_warn.setVisible(Severity.WARN in allowed)
        self._chip_err.setVisible(Severity.ERR in allowed)

    def _hide_chips(self) -> None:
        for chip in (self._chip_ok, self._chip_warn, self._chip_err):
            chip.setVisible(False)

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._spinner.set_busy(busy, message=message)
        # While a validation run is in flight we lock the trigger so
        # double-clicks don't pile up workers.
        self._validate_dataset_btn.setEnabled(
            (not busy) and (self.current_root() is not None)
        )

    # ------------------------------------------------------------------
    # Tree ↔ sidecar form wiring
    # ------------------------------------------------------------------

    def _on_file_selected(self, path: Path) -> None:
        """User picked a row in the BIDS tree.

        Routes to a viewer based on the file extension:

        * ``.tsv`` / ``.tsv.gz`` → :class:`TsvViewerPane` (table).
        * ``.nii`` / ``.nii.gz`` → :class:`NiftiViewerPane` (2-D slice
          viewer with orientation buttons + brightness/contrast).
        * EEG / MEG / iEEG recordings (``.fif``, ``.edf``, ``.set``,
          ``.vhdr``, ``.cnt``, CTF ``.ds``, …) →
          :class:`RecordingViewerPane` (metadata card + threaded
          time-series viewer).
        * everything else (JSON sidecars, …) → :class:`SidecarFormPane`.
        """
        if path.is_dir():
            # CTF .ds / EGI .mff are directory-shaped recordings — route
            # them to the recording viewer instead of the dir-clear path.
            if is_recording_path(path):
                self._show_recording(path)
                self._validation_pane.set_current_file(path, self.current_root())
                return
            # Plain directories don't carry sidecars. We still push the
            # folder down to the validation panel so its "Folder"
            # section can reflect the user's focus.
            self._sidecar_form.set_file(None, None, None)
            self._tsv_viewer.set_file(None, None)
            self._nifti_viewer.set_file(None, None)
            self._recording_viewer.set_file(None, None)
            self._center_stack.setCurrentWidget(self._sidecar_form)
            self._validation_pane.set_current_file(path, self.current_root())
            return
        name = path.name.lower()
        root = self.current_root()
        if name == "citation.cff":
            # YAML rather than JSON, so the sidecar form cannot render it.
            # It was the one dataset-level file that sent the user to a text
            # editor.
            self._citation_pane.set_file(path, root)
            self._center_stack.setCurrentWidget(self._citation_pane)
            self._validation_pane.set_current_file(path, root)
            return
        if name == ".bidsignore":
            # Not a text file as far as a user is concerned: what matters is
            # which files each pattern is hiding, which a text editor cannot
            # say.
            self._bidsignore_pane.set_root(root)
            self._bidsignore_pane.set_report(self._report)
            self._center_stack.setCurrentWidget(self._bidsignore_pane)
            self._validation_pane.set_current_file(path, root)
            return
        if name.endswith(".tsv") or name.endswith(".tsv.gz"):
            self._tsv_viewer.set_file(path, root)
            # Other panes get cleared so a future toggle back doesn't
            # show stale state for a different file.
            self._sidecar_form.set_file(None, None, None)
            self._nifti_viewer.set_file(None, None)
            self._recording_viewer.set_file(None, None)
            self._center_stack.setCurrentWidget(self._tsv_viewer)
        elif is_mrs_path(path):
            # An MRS file is a NIfTI by container only: the data block is a
            # complex free induction decay, and showing it as slices would
            # show nothing. Routed on the DATATYPE folder and the schema's
            # own suffix list, never by opening the file, because routing a
            # click must not read a volume off disk.
            self._mrs_viewer.set_file(path, root)
            self._sidecar_form.set_file(None, None, None)
            self._tsv_viewer.set_file(None, None)
            self._nifti_viewer.set_file(None, None)
            self._recording_viewer.set_file(None, None)
            self._center_stack.setCurrentWidget(self._mrs_viewer)
        elif name.endswith(".nii") or name.endswith(".nii.gz"):
            self._nifti_viewer.set_file(path, root)
            self._sidecar_form.set_file(None, None, None)
            self._tsv_viewer.set_file(None, None)
            self._recording_viewer.set_file(None, None)
            self._mrs_viewer.set_file(None, None)
            self._center_stack.setCurrentWidget(self._nifti_viewer)
        elif is_recording_path(path):
            self._show_recording(path)
        else:
            self._sidecar_form.set_file(path, root, self._report)
            self._tsv_viewer.set_file(None, None)
            self._nifti_viewer.set_file(None, None)
            self._recording_viewer.set_file(None, None)
            self._center_stack.setCurrentWidget(self._sidecar_form)
        self._validation_pane.set_current_file(path, root)

    def _show_recording(self, path: Path) -> None:
        """Route an EEG/MEG/iEEG recording to the recording viewer."""
        root = self.current_root()
        self._recording_viewer.set_file(path, root)
        self._sidecar_form.set_file(None, None, None)
        self._tsv_viewer.set_file(None, None)
        self._nifti_viewer.set_file(None, None)
        self._center_stack.setCurrentWidget(self._recording_viewer)

    def _on_pane_loading(self, busy: bool, message: str) -> None:
        """Mirror a center pane's threaded load onto the toolbar spinner.

        Only drives the spinner; it does not touch the validate buttons
        (those are owned by :meth:`_set_busy`). When a dataset validation
        is in flight we leave the spinner under its control.
        """
        if self._report_worker is not None and self._report_worker.isRunning():
            return
        self._spinner.set_busy(busy, message=message)

    # ------------------------------------------------------------------
    # Theme cascade (called by MainWindow._on_palette_changed)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Jump-to-file + jump-to-field interactivity
    # ------------------------------------------------------------------

    def _open_issues_dialog(self, severity: str) -> None:
        """Open a modal :class:`EditorIssuesDialog` filtered by severity.

        The dialog lists every file in the current report whose
        severity matches. Activating a card selects the file in the
        BIDS tree (which cascades through the panes).
        """
        if self._report is None or self.current_root() is None:
            return
        from .editor_issues_dialog import EditorIssuesDialog
        dlg = EditorIssuesDialog(
            self._report, severity, self.current_root(), parent=self,
        )
        dlg.file_selected.connect(self.select_file_in_tree)
        # The Fix buttons inside the listing go to the same place the
        # validation pane's do. They used to be drawn and connected to
        # nothing, so pressing one did nothing at all.
        dlg.fix_requested.connect(self._on_fix_from_dialog)
        dlg.show()

    def _on_fix_from_dialog(self, path: Path, field: str) -> None:
        """Select the file, then route to the field, in that order.

        The dialog lists files other than the one on screen, so the panes have
        to be showing the right file before the field can be focused in it.
        """
        self.select_file_in_tree(path)
        self._on_fix_requested(path, field)

    def select_file_in_tree(self, path: Path) -> None:
        """Select ``path`` in the BIDS tree (cascades to all panes).

        Public so other widgets (the issues dialog, the validation
        pane's fix-button handler) can use it.
        """
        # ``reveal`` opens the folders on the way down, which is also what
        # CREATES the rows: a folder is drawn when it is opened, so a row
        # deep in the dataset does not exist until something asks for it.
        item = self._tree_pane.reveal(Path(path))
        if item is not None:
            tree = self._tree_pane._tree
            tree.setCurrentItem(item)
            tree.scrollToItem(item)
            return
        # If the tree row can't be found (e.g. the file got renamed
        # between validation and the click), fall back to loading
        # the sidecar / TSV viewer directly so the click still works.
        self._on_file_selected(path)

    def _refresh_adopt_button(self) -> None:
        """Offer tracking only while the open dataset lacks a project bundle."""
        from ..project.adopt import is_managed

        root = self.current_root()
        self._adopt_action.setVisible(
            root is not None and not is_managed(root)
        )

    def _on_adopt(self) -> None:
        """Write the baseline that makes editing this dataset reversible."""
        from ..project.adopt import NotABidsDataset, adopt

        root = self.current_root()
        if root is None:
            return
        answer = QMessageBox.question(
            self, "Track changes to this dataset",
            f"This will create a .bidsmgr folder inside\n{root}\n\n"
            "and record what every file looks like right now, so edits made "
            "from here can be undone. Nothing else in the dataset is touched, "
            "and every BIDS tool ignores dot-folders, so validation is "
            "unaffected.\n\nOn a large dataset this reads every file once.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Ok,
        )
        if answer != QMessageBox.StandardButton.Ok:
            return
        try:
            result = adopt(root)
        except NotABidsDataset as exc:
            QMessageBox.warning(self, "Not a BIDS dataset", str(exc))
            return
        except OSError as exc:
            QMessageBox.warning(
                self, "Could not write",
                f"{root} could not be written to, so the baseline was not "
                f"recorded:\n\n{exc}",
            )
            return
        self.log_message.emit(
            f"tracking {result.files} file(s) in {result.root}"
        )
        self._refresh_adopt_button()
        self._tree_pane.set_root(root)

    def _on_rename(
        self, entity: str = "", value: str = "", focus: str = "",
    ) -> None:
        """Rename an entity across the dataset, after showing the plan.

        Called both from the toolbar with nothing preselected, and from the
        tree's right-click menu with the entity the user clicked, which is
        where the action is actually reached for: you notice a wrong subject
        label while looking at the subject.

        ``focus`` is that clicked path. It decides what starts TICKED, not
        what the plan contains: the plan is still the whole dataset, so
        widening the selection is a click rather than a restart.
        """
        from .rename_entity_dialog import RenameEntityDialog

        root = self.current_root()
        if root is None:
            return
        dlg = RenameEntityDialog(
            root, parent=self, entity=entity, value=value,
            focus=Path(focus) if focus else None,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        # Paths the panes are holding may no longer exist.
        self._sidecar_form.set_file(None, None, None)
        self._reload_open_image()
        self._tree_pane.set_root(root)
        if self._report is not None:
            self.start_dataset_validation()

    def _on_edit_entities(
        self,
        targets: Optional[list] = None,
        mode: str = "add",
        session_mode: bool = False,
    ) -> None:
        """Add or remove an entity, or create or remove a session.

        One dialog for both, because underneath they are one operation: a
        session is an entity that happens to name a folder. Reached from the
        Tools menu and from the tree's right-click menu, which passes what
        was clicked.

        It no longer refuses an empty selection. The dialog carries a scope
        chooser, so opening it with nothing picked starts on the whole
        dataset and the user narrows to a subject, a session or a datatype
        there. Refusing meant "put every anatomical into a session" was only
        reachable by selecting them all in the tree first.

        The refresh afterwards is the same one a rename needs, and for the
        same reason: files have moved, so every path a pane is holding may
        name something that is no longer there.
        """
        from .edit_entities_dialog import EditEntitiesDialog

        root = self.current_root()
        if root is None:
            return
        chosen = [Path(t) for t in (targets or self._tree_pane.selected_paths())]
        dlg = EditEntitiesDialog(
            root, chosen, parent=self, mode=mode, session_mode=session_mode,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._sidecar_form.set_file(None, None, None)
        self._reload_open_image()
        self._tree_pane.set_root(root)
        if self._report is not None:
            self.start_dataset_validation()

    def _on_delete(self, targets: Optional[list] = None) -> None:
        """Delete recordings, datatypes or sessions, after showing the plan.

        Reached from the Tools menu and from the tree's right-click, which
        passes what was clicked. Opened with nothing picked it starts on the
        first subject rather than refusing: the dialog's scope chooser is
        how you say what to delete, and the whole dataset is deliberately
        not one of its options. The refresh afterwards is the one a rename
        needs, for a stronger reason: every path a pane is holding may name
        a file that is now gone.
        """
        from .delete_dialog import DeleteDialog

        root = self.current_root()
        if root is None:
            return
        chosen = [Path(t) for t in (targets or self._tree_pane.selected_paths())]
        dlg = DeleteDialog(root, chosen, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        # Deliberately clears the centre pane before refreshing: the file it
        # was showing is one of the things that may have just been deleted.
        self._sidecar_form.set_file(None, None, None)
        self._reload_open_image()
        self._tree_pane.set_root(root)
        if self._report is not None:
            self.start_dataset_validation()

    def _on_links(self, targets: Optional[list] = None) -> None:
        """Edit the fields that point from one file to another.

        Opens on the whole dataset. A file may be passed in, which is what
        the tree's right-click does, and it is used to start on that file
        with a field it can actually carry. Nothing requires one: the dialog
        lists the files that carry the field on the left, so opening it cold
        is how you find the fieldmaps with nothing set.
        """
        from .linkage_dialog import LinkageDialog

        root = self.current_root()
        if root is None:
            return
        chosen = [
            Path(t) for t in (targets or self._tree_pane.selected_paths())
            if Path(t).is_file()
        ]
        dlg = LinkageDialog(root, chosen[0] if chosen else None, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._sidecar_form.set_file(None, None, None)
        self._tree_pane.set_root(root)
        if self._report is not None:
            self.start_dataset_validation()

    def _on_replace_value(self) -> None:
        """Find one entity value and replace it, inside a chosen scope."""
        from .replace_value_dialog import ReplaceValueDialog

        root = self.current_root()
        if root is None:
            return
        dlg = ReplaceValueDialog(root, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._sidecar_form.set_file(None, None, None)
        self._reload_open_image()
        self._tree_pane.set_root(root)
        if self._report is not None:
            self.start_dataset_validation()

    def _on_pad_values(self) -> None:
        """Give every value of an index entity the same width."""
        from .pad_values_dialog import PadValuesDialog

        root = self.current_root()
        if root is None:
            return
        dlg = PadValuesDialog(root, parent=self)
        dlg.exec()
        if dlg.applied_count():
            # Files moved, so whatever the panes are holding may be gone.
            self._sidecar_form.set_file(None, None, None)
            self._reload_open_image()
            self._tree_pane.set_root(root)
            if self._report is not None:
                self.start_dataset_validation()

    def _on_coherence(self) -> None:
        """Find where the dataset's files stop agreeing with each other."""
        from .coherence_dialog import CoherenceDialog

        root = self.current_root()
        if root is None:
            return
        dlg = CoherenceDialog(root, parent=self)
        dlg.exec()
        if dlg.applied_count():
            self._sidecar_form.set_file(None, None, None)
            self._tree_pane.set_root(root)
            if self._report is not None:
                self.start_dataset_validation()

    def _refresh_deface_action(self) -> None:
        """Grey the Deface entry out when it cannot run, and say why.

        Disabled rather than hidden on purpose. A missing menu item reads as
        "this tool cannot do that"; a greyed one whose tooltip names the
        install command reads as what it is, which is the difference between a
        user installing an extra and a user shipping a dataset with faces in
        it because they assumed the feature did not exist.
        """
        reason = deface_run.unavailable_reason()
        self._deface_action.setEnabled(reason is None)
        if reason:
            self._deface_action.setToolTip(reason)

    def _on_deface(self, targets: Optional[list] = None) -> None:
        """Remove faces from anatomical images, after showing which ones.

        Reached from the Tools menu, where it acts on the whole dataset unless
        the tree has a selection, and from the tree's right-click, which passes
        what was clicked.

        The images are rewritten in place, so the panes are cleared before the
        refresh: whatever the viewer is holding may be one of the files that
        just changed underneath it.
        """
        from .deface_dialog import DefaceDialog

        root = self.current_root()
        if root is None:
            return

        reason = deface_run.unavailable_reason()
        if reason:
            QMessageBox.information(self, "Defacing is not available", reason)
            return

        chosen = [Path(t) for t in (targets or self._tree_pane.selected_paths())]
        dlg = DefaceDialog(root, chosen or None, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._sidecar_form.set_file(None, None, None)
        self._reload_open_image()
        self._tree_pane.set_root(root)
        if self._report is not None:
            self.start_dataset_validation()

    def _reload_open_image(self) -> None:
        """Re-read whatever the viewer is showing, if it changed underneath.

        Defacing, stripping and restoring all rewrite files in place. The
        viewer holds a decoded array, not the file, so it went on showing the
        face until the user clicked away and back. Clearing first is what
        forces the re-read: `set_file` with the path it already has is a
        no-op as far as the pane is concerned.
        """
        current = self._nifti_viewer.current_file()
        if current is None:
            return
        root = self.current_root()
        if not Path(current).is_file():
            # It was deleted or moved out from under the viewer.
            self._nifti_viewer.set_file(None, None)
            return
        self._nifti_viewer.set_file(None, None)
        self._nifti_viewer.set_file(Path(current), root)

    def _on_compare_images(self, targets: Optional[list] = None) -> None:
        """Two images side by side, whatever they are.

        Takes the tree selection when there is one: two picked images open
        straight away, one opens on the left with the right still to choose.
        Needs no dataset, because comparing two files is not a dataset
        operation, but passes the root when there is one so the captions can
        show dataset-relative paths instead of bare names.
        """
        from .compare_dialog import open_compare

        picked = targets or self._tree_pane.selected_paths()
        dlg = open_compare(self, picked, root=self.current_root())
        dlg.exec()

    def _on_strip(self, targets: Optional[list] = None) -> None:
        """Keep only the brain, writing a derivative rather than editing raw.

        The same dialog as Deface, in its other mode: the two operations have
        the same shape (preview, pick an engine, one undoable run) and differ
        in what they keep and where the output goes.
        """
        from ..deface import engines as deface_engines
        from .deface_dialog import DefaceDialog

        root = self.current_root()
        if root is None:
            return

        chosen = [Path(t) for t in (targets or self._tree_pane.selected_paths())]
        dlg = DefaceDialog(
            root, chosen or None, parent=self,
            kind=deface_engines.KIND_STRIP,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        # A strip writes new files under derivatives/; an in-place one changes
        # what the viewer is holding. Refresh either way.
        self._sidecar_form.set_file(None, None, None)
        self._reload_open_image()
        self._tree_pane.set_root(root)
        if self._report is not None:
            self.start_dataset_validation()

    def _on_deface_compare(self, target=None) -> None:
        """Show one image before and after defacing.

        Reached from the Tools menu, where it takes whatever image the tree has
        selected, and from the tree's right-click on a ``.nii``, which passes
        the file. It deliberately opens even when there is no undefaced copy:
        the dialog explains WHY there is none, and "defaced during conversion,
        so one never existed" is the answer a user most needs to hear.
        """
        from .deface_compare import DefaceCompareDialog

        root = self.current_root()
        if root is None:
            return

        picked = [Path(target)] if target else self._tree_pane.selected_paths()
        images = [
            Path(p) for p in picked
            if Path(p).is_file() and Path(p).name.endswith((".nii", ".nii.gz"))
        ]
        if not images:
            QMessageBox.information(
                self, "Pick an image",
                "Select a .nii or .nii.gz file in the tree first. Comparing "
                "is per image, because defacing is.",
            )
            return

        try:
            rel = images[0].resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            QMessageBox.information(
                self, "Not in this dataset",
                f"{images[0]} is outside {root}.",
            )
            return
        DefaceCompareDialog(root, rel, parent=self).exec()

    def _on_deface_revert(self, targets: Optional[list] = None) -> None:
        """Restore defaced images from the copies that still have a face.

        Acts on the tree selection when there is one, the whole dataset
        otherwise, which is the same rule Deface uses. The dialog opens even
        when nothing can be restored, because "these were defaced during
        conversion, so no undefaced copy ever existed" is the answer the user
        needs and an absent menu item does not give it.
        """
        from .deface_revert_dialog import DefaceRevertDialog

        root = self.current_root()
        if root is None:
            return
        chosen = [Path(t) for t in (targets or self._tree_pane.selected_paths())]
        dlg = DefaceRevertDialog(root, chosen or None, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        # The images changed under whatever the viewer is holding.
        self._sidecar_form.set_file(None, None, None)
        self._reload_open_image()
        self._tree_pane.set_root(root)
        if self._report is not None:
            self.start_dataset_validation()

    def _on_dashboard(self) -> None:
        """What is in this dataset, counted.

        Uses the report already in hand rather than revalidating: drawing a
        summary is not a reason to re-run the validator over a whole dataset.
        """
        from .dashboard_dialog import DashboardDialog

        root = self.current_root()
        if root is None:
            return
        dialog = DashboardDialog(root, report=self._report, parent=self)
        dialog.file_selected.connect(self.select_file_in_tree)
        dialog.show()

    def _on_fixups(self) -> None:
        """Open the dataset-wide repairs, then revalidate what changed."""
        from .fixups_dialog import FixupsDialog

        root = self.current_root()
        if root is None:
            return
        dlg = FixupsDialog(root, parent=self, report=self._report)
        dlg.exec()
        # Generating a file or moving citation fields changes what the
        # validator would say, so the board must not keep showing the old
        # answer.
        if self._report is not None:
            self.start_dataset_validation()

    def _on_accept_finding(self, path, rule_id: str, field: str) -> None:
        """Record that a reviewer looked at a warning and kept it."""
        from PyQt6.QtWidgets import QInputDialog

        from ..editor.review import accept

        root = self.current_root()
        if root is None or path is None:
            return
        note, ok = QInputDialog.getText(
            self, "Accept this warning",
            f"Why is {rule_id} acceptable here?\n\nThe note is stored in the "
            "dataset, so whoever reviews it next sees your reasoning.",
        )
        if not ok:
            return
        try:
            # The acceptance is keyed by this string and looked up again on
            # the next validation run, so it has to be spelled the same way
            # everywhere. POSIX, as BIDS does. See editor/rename.py::_rel.
            rel = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
        except (ValueError, OSError):
            rel = Path(path).as_posix()
        accept(root, file=rel, rule_id=rule_id, field=field, note=note)
        self._validation_pane.reload_acceptances()
        self.log_message.emit(f"accepted {rule_id} in {rel}")

    def _on_explain_field(self, field: str) -> None:
        """Say which file the value in front of the user is actually in."""
        from ..editor import inheritance as inh

        root = self.current_root()
        current = self._sidecar_form.current_file()
        if root is None or current is None:
            return
        sources = inh.explain(root, current, field)
        if not sources:
            QMessageBox.information(
                self, field,
                f"No sidecar that applies to this file states {field}. What "
                "you see is the form offering the field, not a value.",
            )
            return
        lines = []
        for s in sources:
            where = "this file" if s.level == 0 else f"{s.level} level(s) up"
            mark = "  <- used" if s.winner else ""
            lines.append(f"{s.rel}\n    {where}: {s.value!r}{mark}")
        extra = ""
        if len(sources) > 1:
            extra = (
                "\n\nThe nearest one wins. The others are shadowed, which "
                "is legal but usually not what anyone intended."
            )
        elif sources[0].level > 0:
            extra = (
                "\n\nThis value is inherited. Editing it here writes a "
                "second copy into this file, and the two can then disagree."
            )
        QMessageBox.information(
            self, f"Where {field} comes from", "\n\n".join(lines) + extra,
        )

    def _on_apply_field_to_others(self, field: str) -> None:
        """State the field the user is editing across other files too."""
        from ..editor import bulk_edit as be
        from .bulk_field_dialog import BulkFieldDialog

        root = self.current_root()
        current = self._sidecar_form.current_file()
        if root is None or current is None:
            return
        # Offer the value the user has in front of them as the starting point.
        cache = getattr(self._sidecar_form, "_json_cache", None) or {}
        dlg = BulkFieldDialog(
            root, field,
            anchor=current,
            initial_value=cache.get(field),
            title=f"Apply {field} to other files",
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dlg.selected()
        if not chosen:
            return
        result = be.apply_value(
            root, chosen, field, dlg.value(),
            label=f"Set {field} in {len(chosen)} file(s)",
        )
        self._report_bulk_result(result)

    def _offer_type_fix(self, root, group, paths) -> bool:
        """A value of the wrong shape is repaired, not re-asked for."""
        from ..editor import bulk_edit as be

        plan = be.plan_coercions(root, group.field, paths=paths)
        if not plan:
            return False
        lines = "\n".join(
            f"  {c.rel}\n      {c.current_text()}" for c in plan[:12]
        )
        more = f"\n  and {len(plan) - 12} more" if len(plan) > 12 else ""
        answer = QMessageBox.question(
            self, f"Fix the type of {group.field}",
            f"{len(plan)} file(s) state {group.field} in a shape the standard "
            f"does not declare. Each value below would be converted, keeping "
            f"what it says:\n\n{lines}{more}\n\nA value that cannot be "
            "converted is left exactly as it is.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Ok,
        )
        if answer != QMessageBox.StandardButton.Ok:
            return True
        self._report_bulk_result(be.apply_coercions(root, plan, group.field))
        return True

    def _offer_tabular_fix(self, root, group, paths) -> bool:
        """Missing column, wrong column type, wrong column order."""
        from ..editor import tsv_edit as te

        tsvs = [p for p in paths if p.name.endswith(".tsv")]
        if not tsvs or not group.field:
            return False
        rule = (group.rule_id or "").upper()

        if "COLUMN_MISSING" in rule:
            plan = te.plan_add_column(root, tsvs, group.field)
            actionable = [c for c in plan if c.applicable]
            if not actionable:
                return False
            answer = QMessageBox.question(
                self, f"Add the {group.field} column",
                f"{len(actionable)} table(s) would gain a {group.field!r} "
                f"column filled with {te.NA!r}, which is what BIDS uses for a "
                "value that is not available.",
                QMessageBox.StandardButton.Cancel
                | QMessageBox.StandardButton.Ok,
                QMessageBox.StandardButton.Ok,
            )
            if answer == QMessageBox.StandardButton.Ok:
                written, failed = te.apply_add_column(
                    root, actionable, group.field,
                )
                self._report_tabular(written, failed)
            return True

        if "ORDER" in rule:
            plan = te.plan_reorder(root, tsvs)
            actionable = [c for c in plan if c.applicable]
            if not actionable:
                return False
            answer = QMessageBox.question(
                self, "Reorder columns",
                f"{len(actionable)} table(s) would have their known columns "
                "put into the standard's order. No cell changes value; "
                "columns the standard does not know keep their order at the "
                "end.",
                QMessageBox.StandardButton.Cancel
                | QMessageBox.StandardButton.Ok,
                QMessageBox.StandardButton.Ok,
            )
            if answer == QMessageBox.StandardButton.Ok:
                self._report_tabular(*te.apply_reorder(root, actionable))
            return True

        if "TYPE" in rule or "VALUE" in rule:
            plan = te.plan_coerce_column(root, tsvs, group.field)
            actionable = [c for c in plan if c.applicable]
            if not actionable:
                return False
            detail = "\n".join(f"  {c.rel}: {c.detail}" for c in actionable[:12])
            answer = QMessageBox.question(
                self, f"Fix the {group.field} column",
                f"Only the cells that do not match the declared type are "
                f"touched:\n\n{detail}",
                QMessageBox.StandardButton.Cancel
                | QMessageBox.StandardButton.Ok,
                QMessageBox.StandardButton.Ok,
            )
            if answer == QMessageBox.StandardButton.Ok:
                self._report_tabular(
                    *te.apply_coerce_column(root, actionable, group.field)
                )
            return True
        return False

    def _report_tabular(self, written, failed) -> None:
        if failed:
            QMessageBox.warning(
                self, "Some tables were not written",
                "\n".join(f"{p}: {why}" for p, why in failed[:6]),
            )
        else:
            self.log_message.emit(f"{len(written)} table(s) written")
        if written and self._report is not None:
            self.start_dataset_validation()

    def _on_fix_group_requested(self, group) -> None:
        """Fix one finding in every file it fired on.

        The grouped view knows exactly which files produced the finding, so
        they are passed straight through as the candidate list rather than
        re-derived from a scope: the question is not "where else could this
        apply" but "which of these do you want changed".
        """
        from ..editor import bulk_edit as be
        from .bulk_field_dialog import BulkFieldDialog

        root = self.current_root()
        if root is None or not group.field:
            return
        # A finding is recorded against the data file; the field is edited in
        # its sidecar. ``candidates`` follows that hop for us.
        paths = [root / p for p in group.files]

        # The repair depends on what is wrong, not just on which field. A
        # value of the wrong TYPE is repaired from the value already there; a
        # missing column is added; a value nobody has stated has to be asked
        # for. Offering "type a value" for all three would invite a user to
        # overwrite a number that was only badly formatted.
        if self._offer_tabular_fix(root, group, paths):
            return
        if self._offer_type_fix(root, group, paths):
            return

        cands = be.candidates(root, group.field, paths=paths)
        if not cands:
            QMessageBox.information(
                self, "Nothing to fix",
                "No editable sidecar was found for those files.",
            )
            return
        dlg = BulkFieldDialog(
            root, group.field,
            candidates=cands,
            title=f"Fix {group.field} in {len(cands)} file(s)",
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dlg.selected()
        if not chosen:
            return
        result = be.apply_value(
            root, chosen, group.field, dlg.value(),
            label=f"Fix {group.field} in {len(chosen)} file(s)",
        )
        self._report_bulk_result(result)

    def _report_bulk_result(self, result) -> None:
        """Tell the user what was written, and revalidate so it shows."""
        parts = [f"{len(result.written)} file(s) written"]
        if result.skipped:
            parts.append(f"{len(result.skipped)} already correct")
        if result.failed:
            parts.append(f"{len(result.failed)} failed")
        msg = ", ".join(parts) + "."
        if result.failed:
            msg += "\n\n" + "\n".join(
                f"{p}: {why}" for p, why in result.failed[:5]
            )
            QMessageBox.warning(self, "Some files were not written", msg)
        else:
            self.log_message.emit(msg)
        if result.written:
            # The sidecar on screen may be one of the files just changed.
            current = self._sidecar_form.current_file()
            if current is not None and current in result.written:
                self._sidecar_form.set_file(
                    current, self.current_root(), self._report,
                )
            self.start_dataset_validation()

    def _on_fix_requested(self, path: Path, field: str) -> None:
        """A user clicked a ValMessage's fix button.

        Navigate to where the problem is actually edited, then highlight it:

        * a TSV column finding -> open the ``.tsv`` and select that column;
        * a sidecar field finding -> open the ``.json`` and focus that field.
          When the finding sits on a data file (``*_bold.nii.gz``, where bidsval
          attaches sidecar metadata findings), redirect to its editable
          ``*_bold.json`` sibling so the field can be focused.
        """
        if path is None:
            return
        name = path.name.lower()

        if name.endswith(".tsv") or name.endswith(".tsv.gz"):
            self.select_file_in_tree(path)
            verdict = self._verdict_for(path)
            if verdict is not None and field:
                # Highlight every bad cell for this column. A column finding
                # carries all offending rows in ``lines`` (the viewer resolves
                # each 1-based row to a cell).
                items = self._tsv_highlight_items(verdict.issues, field=field)
                if items:
                    self._tsv_viewer.highlight_findings(items)
            return

        if name.endswith(".json"):
            target = path
        else:
            from ..editor.bidsmgr_checks import sibling_json
            sib = sibling_json(path)
            target = sib if (sib is not None and sib.exists()) else path

        if target != self._sidecar_form.current_file():
            self.select_file_in_tree(target)
        if field and target.name.lower().endswith(".json"):
            self._sidecar_form.focus_field(field)

    def _on_highlight_all_requested(self, path: Path) -> None:
        """Highlight every shown error/warning field (JSON) or cell (TSV) for
        ``path`` in the editor. For a data file, the editable target is its
        ``.json`` sibling (where the sidecar fields live)."""
        if path is None or self._report is None:
            return
        name = path.name.lower()
        allowed = self._allowed_severities()

        # TSV: highlight each finding's specific cell (or whole column when the
        # finding has no row), so the bad values are pinpointed, not the column.
        if name.endswith(".tsv") or name.endswith(".tsv.gz"):
            verdict = self._verdict_for(path)
            if verdict is None:
                return
            items = self._tsv_highlight_items(verdict.issues, allowed=allowed)
            if not items:
                return
            if path != self._tsv_viewer.current_file():
                self.select_file_in_tree(path)
            self._tsv_viewer.highlight_findings(items)
            return

        # JSON sidecar (or a data file -> its editable .json sibling).
        if name.endswith(".json"):
            target = path
        else:
            from ..editor.bidsmgr_checks import sibling_json
            sib = sibling_json(path)
            target = sib if (sib is not None and sib.exists()) else path
        verdict = self._verdict_for(target)
        if verdict is None:
            return
        rank = {"err": 2, "warn": 1}
        mapping: dict[str, str] = {}
        for issue in verdict.issues:
            if not issue.field or issue.severity not in allowed:
                continue
            sv = issue.severity.value
            if rank.get(sv, 0) > rank.get(mapping.get(issue.field, ""), 0):
                mapping[issue.field] = sv
        if not mapping:
            return
        if target != self._sidecar_form.current_file():
            self.select_file_in_tree(target)
        self._sidecar_form.highlight_fields(mapping)

    @staticmethod
    def _tsv_highlight_items(issues, *, field=None, allowed=None) -> list:
        """Flatten TSV findings to ``(column, line, severity)`` items, one per
        offending row. A column finding lists every bad row in ``lines``; we
        expand them all so each bad cell is highlighted (falls back to ``line``
        for column-level findings without rows)."""
        items: list = []
        for issue in issues:
            if not issue.field:
                continue
            if field is not None and issue.field != field:
                continue
            if allowed is not None and issue.severity not in allowed:
                continue
            rows = issue.lines or (
                [issue.line] if issue.line is not None else [None]
            )
            for ln in rows:
                items.append((issue.field, ln, issue.severity.value))
        return items

    def _verdict_for(self, path: Path) -> Optional[FileVerdict]:
        """The :class:`FileVerdict` in the current report for ``path`` (matched
        on the absolute path), or ``None``."""
        report = self._report
        root = self.current_root()
        if report is None or root is None:
            return None
        try:
            target_abs = str(path.resolve())
        except OSError:
            target_abs = str(path)
        for fv in report.files:
            cand = fv.path if fv.path.is_absolute() else (root / fv.path)
            try:
                cand_abs = str(cand.resolve())
            except OSError:
                cand_abs = str(cand)
            if cand_abs == target_abs:
                return fv
        return None

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Undo / redo (delegates to the active editable center pane)
    # ------------------------------------------------------------------

    def _active_history_pane(self):
        """The current center pane if it supports undo/redo, else None."""
        w = self._center_stack.currentWidget()
        if w in (self._sidecar_form, self._tsv_viewer):
            return w
        return None

    def _sync_undo_redo(self, *args) -> None:
        del args
        pane = self._active_history_pane()
        self._undo_btn.setEnabled(bool(pane) and pane.can_undo())
        self._redo_btn.setEnabled(bool(pane) and pane.can_redo())

    def _on_editor_undo(self) -> None:
        pane = self._active_history_pane()
        if pane is not None:
            pane.undo()

    def _on_editor_redo(self) -> None:
        pane = self._active_history_pane()
        if pane is not None:
            pane.redo()

    def showEvent(self, event) -> None:  # noqa: N802
        """Refresh the tree when the Editor becomes visible.

        A conversion (or any external change) may have written into the open
        dataset while the user was in the Converter tab. ``refresh`` re-walks
        the root while preserving the user's expand / selection state, so the
        editor reflects the current disk contents the moment it is shown (and
        the live ``QFileSystemWatcher`` keeps it current thereafter).
        """
        super().showEvent(event)
        self._tree_pane.refresh()

    def repaint_for_palette(self, pal: dict) -> None:
        self._tree_pane.repaint_for_palette(pal)
        self._sidecar_form.repaint_for_palette(pal)
        self._tsv_viewer.repaint_for_palette(pal)
        self._nifti_viewer.repaint_for_palette(pal)
        self._bidsignore_pane.repaint_for_palette(pal)
        self._citation_pane.repaint_for_palette(pal)
        self._recording_viewer.repaint_for_palette(pal)
        self._mrs_viewer.repaint_for_palette(pal)
        self._validation_pane.repaint_for_palette(pal)
        for frame in getattr(self, "_panel_frames", []):
            frame._refresh_icons()
        # Re-tint toolbar icons after the icon cache is cleared in
        # ``MainWindow._on_palette_changed``.
        icons.apply_button(self._open_btn, "open_folder")
        icons.apply_button(self._undo_btn, "undo")
        icons.apply_button(self._redo_btn, "redo")
        icons.apply_button(self._validate_dataset_btn, "dataset")
        icons.apply_button(self._strict_btn, "strict")
        icons.apply_button(self._tools_btn, "tools")
        # The two partial-validate buttons get their tint from the
        # current tree selection (green when their kind is selected,
        # accent blue otherwise) — defer to the sync helper so the
        # selection state survives a dark↔light swap.
        self._sync_validate_buttons_from_selection(self._tree_pane_current_path())


__all__ = ["EditorPanel"]
