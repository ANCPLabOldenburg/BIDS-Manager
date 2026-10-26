"""Post-conversion BIDS editor view (M6).

Visual reference: ``inspector_proto/proto.py`` ``EditorView`` (3-pane
horizontal splitter — BIDS tree | sidecar form | validation panel).

Through Step 3 the panel hosts:

* a toolbar with ``Open folder…`` (the one way to open something),
  ``Validate dataset``, status chips, a busy spinner, file / folder
  validate;
* a :class:`PathBar` showing the open folder and what it is: the PROJECT
  (every edit recorded, undoable) or a folder open for VIEWING only
  (nothing is written into it; editing and the dataset tools are off).
  A BIDS dataset is opened as a project after asking (the Converter
  switches too); any other folder opens for viewing, so BIDS Manager can
  be a plain viewer without leaving a trace;
* the 3-pane horizontal splitter with the
  :class:`BidsTreePane` on the left (badges stamped from the in-memory
  :class:`bidsmgr.editor.types.ValidationReport`).

Center and right panes remain placeholders for Steps 4-6.
"""

from __future__ import annotations

import os

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
from .widgets.dicom_pane import DicomPane
from .widgets.file_info_pane import FileInfoPane
from .widgets.gradient_pane import GradientPane
from .widgets.picture_pane import PicturePane
from .widgets.text_pane import TextPane
from .viz import Viewer
from .widgets import (
    BidsTreePane,
    BusySpinner,
    Chip,
    PanelFrame,
    PathBar,
    SidecarFormPane,
    TsvViewerPane,
    ValidationPane,
    VSep,
)
from ..viz.data.formats import is_mrs_path, is_recording_path
from .widgets.splitter import Splitter

log = logging.getLogger(__name__)


def ignore_message(choice, added: list, removed: list) -> str:
    """What an ignore choice did, for the status bar."""
    if choice.ignores:
        return f"Ignored in validation: {', '.join(added)}. Undo takes it back."
    n = choice.files
    return (f"Validated again: {n} file" + ("" if n == 1 else "s")
            + (f" ({', '.join(removed)} removed)" if removed else "")
            + ". Undo takes it back.")


class EditorPanel(QWidget):
    """Post-convert BIDS editor (Editor pill in the top header).

    Emits :pyattr:`log_message` for every progress line so
    :class:`MainWindow` can mirror Editor activity into the status bar
    just like it already does for the Converter.
    """

    log_message = pyqtSignal(str)
    #: The user chose to open a BIDS dataset as the project (Path): the
    #: window opens it, and both views follow.
    open_project_requested = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("editor-panel")
        #: The open project's dataset (the window sets it), if any.
        self._project_root: Optional[Path] = None
        #: The folder on screen is open for viewing only.
        self._view_only = False
        #: The file last opened from the tree: shown again after an undo or
        #: redo has changed the dataset under it.
        self._open_path: Optional[Path] = None
        self._guide_reports = 0         # validations finished (the guide counts them)
        self._guide_file_opens = 0      # files opened in the viewer

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

        # No change button: Open folder in the toolbar is the one way in.
        self._path_bar = PathBar("Dataset", "(none)", ok=False, button=None)
        self._path_bar.action_button.clicked.connect(self._on_path_action)
        v.addWidget(self._path_bar)

        # The three panels are cards on the canvas, a gap between them.
        from .widgets.panel_frame import CARD_GAP_PX

        self._splitter = Splitter(Qt.Orientation.Horizontal)
        self._splitter.setObjectName("card-split")
        self._splitter.setHandleWidth(CARD_GAP_PX)
        self._splitter.setChildrenCollapsible(False)

        self._tree_pane = BidsTreePane()
        self._tree_pane.file_selected.connect(self._on_file_selected)
        self._tree_pane.rename_requested.connect(self._on_rename)
        self._tree_pane.entities_requested.connect(self._on_edit_entities)
        self._tree_pane.delete_requested.connect(self._on_delete)
        self._tree_pane.deface_requested.connect(self._on_deface)
        self._tree_pane.sidecars_requested.connect(self._on_sidecars)
        self._tree_pane.deface_compare_requested.connect(
            self._on_deface_compare
        )
        self._tree_pane.deface_revert_requested.connect(self._on_deface_revert)
        self._tree_pane.strip_requested.connect(self._on_strip)
        self._tree_pane.compare_requested.connect(self._on_compare_images)
        self._tree_pane.links_requested.connect(self._on_links)
        self._tree_pane.anonymise_requested.connect(self._on_anonymise)
        self._tree_pane.ignore_requested.connect(self._on_ignore)
        self._tree_pane.unignore_requested.connect(self._on_unignore)
        self._tree_pane.ignore_list_requested.connect(self._on_ignore_list)
        self._tree_pane.overlay_requested.connect(self._on_overlay_requested)
        self._tree_pane.tree_changed.connect(self._on_tree_changed)
        self._tree_pane.overlay_base = self._overlay_base
        #: The last two images shown in the volume viewer, newest last: what a
        #: right-clicked image would be drawn over (the right-click itself
        #: selects, and so shows, the clicked one).
        self._images_shown: list[Path] = []
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
        # Volumes (MRI, PET, any NIfTI) go to the visualisation library's
        # viewer: commands, configurable keys, layouts, links.
        # Signals (MEG, EEG, iEEG) and spectra (MRS) through the same
        # library, each its own Viewer: a viewer is one kind at a time.
        self._nifti_viewer = Viewer(kind="volume")
        self._signal_viewer = Viewer(kind="signal")
        self._spectrum_viewer = Viewer(kind="spectrum")
        for viewer in (self._nifti_viewer, self._signal_viewer, self._spectrum_viewer):
            viewer.loading_changed.connect(self._on_pane_loading)
            viewer.status_message.connect(self.log_message)
            # Moving to the next run (or a fieldmap's target) goes through the
            # tree, so the tree, the sidecar and the validation pane follow.
            viewer.navigator = self.select_file_in_tree
        self._center_stack = QStackedWidget()
        self._sidecar_form.apply_to_others_requested.connect(
            self._on_apply_field_to_others
        )
        self._sidecar_form.explain_requested.connect(self._on_explain_field)
        self._center_stack.addWidget(self._sidecar_form)
        self._center_stack.addWidget(self._tsv_viewer)
        self._center_stack.addWidget(self._nifti_viewer)
        self._center_stack.addWidget(self._signal_viewer)
        self._center_stack.addWidget(self._spectrum_viewer)
        self._bidsignore_pane = BidsIgnorePane()
        self._center_stack.addWidget(self._bidsignore_pane)
        self._citation_pane = CitationPane()
        self._center_stack.addWidget(self._citation_pane)
        # Every other common file: documents (README, CHANGES, Markdown, HTML,
        # text), the gradient table (.bval / .bvec), pictures, and a card for
        # what BIDS Manager cannot show.
        self._text_pane = TextPane()
        self._gradient_pane = GradientPane()
        self._picture_pane = PicturePane()
        self._file_info_pane = FileInfoPane()
        # A DICOM (a raw folder opened to view, sourcedata/): its header.
        self._dicom_pane = DicomPane()
        self._dicom_pane.loading_changed.connect(self._on_pane_loading)
        self._dicom_pane.status_message.connect(self.log_message)
        for pane in (self._text_pane, self._gradient_pane, self._picture_pane,
                     self._file_info_pane, self._dicom_pane):
            self._center_stack.addWidget(pane)
        # A link in a README to a file of the folder opens that file.
        self._text_pane.file_requested.connect(self._on_document_link)
        # Threaded panes drive the toolbar busy spinner + status bar.
        self._tsv_viewer.loading_changed.connect(self._on_pane_loading)
        self._tsv_viewer.status_message.connect(self.log_message)
        # Undo/redo toolbar buttons follow the active editable pane and the
        # dataset's history (a save, an operation, a change seen on disk).
        self._sidecar_form.history_changed.connect(self._sync_undo_redo)
        self._tsv_viewer.history_changed.connect(self._sync_undo_redo)
        self._center_stack.currentChanged.connect(self._sync_undo_redo)
        self._tree_pane.tree_changed.connect(self._sync_undo_redo)
        for pane in (self._sidecar_form, self._tsv_viewer, self._text_pane,
                     self._citation_pane):
            pane.file_saved.connect(self._sync_undo_redo)
        self._bidsignore_pane.changed.connect(
            lambda message: self._after_ignore_change(message, from_pane=True))
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
        self._validation_pane.header_requested.connect(self._on_header_requested)
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
        for frame in self._panel_frames:
            frame.make_card()
        from .widgets.panel_frame import card_canvas

        v.addWidget(card_canvas(self._splitter), 1)

        # Restore the last folder, for viewing: editing happens in the open
        # project, which the window binds when one is opened.
        from .app_settings import AppSettings
        settings = AppSettings.load()
        if settings.editor_bids_root:
            root = Path(settings.editor_bids_root)
            if root.exists() and root.is_dir():
                self._set_root(root, persist=False, view_only=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tree_pane(self) -> BidsTreePane:
        return self._tree_pane

    def current_root(self) -> Optional[Path]:
        return self._tree_pane.root()

    def current_report(self) -> Optional[ValidationReport]:
        return self._report

    def guide_facts(self) -> dict:
        """What the guide's Editor path needs to know (``guide_steps``).
        ``_root`` is the dataset, for what the guide tracks itself."""
        root = self.current_root()
        report = self._report
        counts = report.counts if report is not None else {}
        worker = self._report_worker
        return {
            "_root": root,
            "root": root is not None,
            "view_only": self._view_only,
            "validating": worker is not None and worker.isRunning(),
            "validated": report is not None,
            "errors": int(counts.get("err", 0)),
            "warnings": int(counts.get("warn", 0)),
            "reports": self._guide_reports,
            "file_opens": self._guide_file_opens,
        }

    def guide_targets(self) -> dict:
        """The controls the guide can point at, by name."""
        return {"open": self._open_btn, "validate": self._validate_dataset_btn,
                "findings": self._validation_frame, "tree": self._tree_frame,
                "viewer": self._center_frame, "tools": self._tools_btn,
                "validate_file": self._validate_file_btn, "undo": self._undo_btn}

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
            flag_todos=flag_todos, strict=self._strict_btn.isChecked(), parent=self,
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
            flag_todos=flag_todos, strict=self._strict_btn.isChecked(), parent=self,
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

        self._open_btn = QPushButton("  Open folder…")
        self._open_btn.setObjectName("tb-btn")
        icons.apply_button(self._open_btn, "open_folder")
        self._open_btn.setToolTip(
            "Open a folder. A BIDS dataset can become the project (the Converter "
            "and the Editor both switch to it) or be opened for viewing only; any "
            "other folder opens for viewing, and nothing is written into it.")
        self._open_btn.clicked.connect(self._on_open_folder)
        lay.addWidget(self._open_btn)

        lay.addWidget(VSep())

        # Undo / Redo: the open file's unsaved edits first, one at a time,
        # then the dataset's history, where every change is ONE step however
        # many files it touched (a saved field, a rename of a subject, a fix
        # applied everywhere, a defacing run). Labelled with what they would
        # do; Cmd/Ctrl+Z anywhere in the Editor, a field being typed in
        # keeping its own undo.
        self._undo_btn = QPushButton("  Undo")
        self._undo_btn.setObjectName("tb-btn-ghost")
        icons.apply_button(self._undo_btn, "undo")
        self._undo_btn.setToolTip("Nothing to undo")
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self._on_editor_undo)
        lay.addWidget(self._undo_btn)

        self._redo_btn = QPushButton("  Redo")
        self._redo_btn.setObjectName("tb-btn-ghost")
        icons.apply_button(self._redo_btn, "redo")
        self._redo_btn.setToolTip("Nothing to redo")
        self._redo_btn.setEnabled(False)
        self._redo_btn.clicked.connect(self._on_editor_redo)
        lay.addWidget(self._redo_btn)
        # Right-click either one for the whole history, like a browser's
        # back button: every step newest first; picking one goes back (or
        # forward) to it.
        for which, btn in (("undo", self._undo_btn), ("redo", self._redo_btn)):
            btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            btn.customContextMenuRequested.connect(
                lambda pos, w=which, b=btn: self._show_history_menu(w, b.mapToGlobal(pos)))
        from PyQt6.QtGui import QKeySequence, QShortcut

        for sequence, slot in ((QKeySequence.StandardKey.Undo, self._on_editor_undo),
                               (QKeySequence.StandardKey.Redo, self._on_editor_redo)):
            shortcut = QShortcut(self)
            shortcut.setKeys(sequence)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

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

        self._quality_action = self._tools_menu.addAction("Quality check...")
        self._quality_action.setToolTip(
            "A fast quality check of every anatomical and diffusion image: "
            "noise, contrast, artefacts, coverage, the gradient table, motion "
            "and slice dropout, one row per image, coloured where an image "
            "sits far from the others of its kind. The air is measured only "
            "in images that are not defaced."
        )
        self._quality_action.triggered.connect(self._on_quality_check)

        menu_section(self._tools_menu, "Check and repair")

        self._coherence_action = self._tools_menu.addAction("Check coherence...")
        self._coherence_action.setToolTip(
            "Find where the dataset's files stop agreeing with each other: "
            "a scans row naming a file that is gone, a participants row for "
            "a deleted subject, a link pointing at nothing, a sidecar with "
            "no recording. Read-only until you choose a repair."
        )
        self._coherence_action.triggered.connect(self._on_coherence)

        self._tables_action = self._tools_menu.addAction("Scans and sessions tables...")
        self._tables_action.setToolTip(
            "Write the *_scans.tsv of each session (every recording, with when it "
            "was acquired) and the *_sessions.tsv of each subject with sessions. "
            "For a dataset that has none, or whose tables have fallen behind. "
            "Tables already there are kept and only added to. Shown before "
            "anything is written; one Undo."
        )
        self._tables_action.triggered.connect(self._on_tables)

        self._sidecars_action = self._tools_menu.addAction(
            "Write sidecars from the recordings...")
        self._sidecars_action.setToolTip(
            "Write the .json sidecar a recording is missing, or add what one lacks, "
            "from the data it was converted from, its own header, your answers and "
            "the sidecars that link to it. A value already there is never changed. "
            "Every field is shown with where it was read; one Undo."
        )
        self._sidecars_action.triggered.connect(lambda _c=False: self._on_sidecars())

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

        self._ignore_action = self._tools_menu.addAction("Ignored files...")
        self._ignore_action.setToolTip(
            "What validation leaves out (.bidsignore): every pattern with the "
            "files it hides, and the dataset's files to pick from. Right-click "
            "a file or folder in the tree to ignore it, or to stop."
        )
        self._ignore_action.triggered.connect(self._on_ignore_list)

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

        self._anonymise_action = self._tools_menu.addAction("Anonymise metadata...")
        self._anonymise_action.setToolTip(
            "Remove what identifies a participant from the metadata of every kind "
            "of file: dates shifted per participant, names and IDs removed, UIDs "
            "pseudonymised, recording headers and photos cleaned. Every change is "
            "shown first; one Undo."
        )
        self._anonymise_action.triggered.connect(self._on_anonymise)

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

    def _on_open_folder(self) -> None:
        from .app_settings import AppSettings
        current = self.current_root()
        start = str(current) if current else (
            AppSettings.load().editor_bids_root or str(Path.home())
        )
        chosen = QFileDialog.getExistingDirectory(self, "Open folder", start)
        if chosen:
            self.open_folder(Path(chosen))

    def open_folder(self, path: Path) -> str:
        """Open what the user picked. Inside the open project: the project,
        at that folder. A BIDS dataset: as the project, or for viewing only,
        after asking (opening it as the project changes the Converter too).
        Anything else: for viewing only, writing nothing into it. Returns
        "project", "view" or "" (cancelled)."""
        from ..project.adopt import looks_like_bids

        path = Path(path)
        project = self._project_root
        if project is not None and (path == project or project in path.parents):
            if self._view_only or self.current_root() != project:
                self.set_project_root(project)
            if path != project:
                self.select_file_in_tree(path)
            return "project"
        if looks_like_bids(path):
            choice = self.ask_bids_folder(path)
            if choice == "project":
                self.open_project_requested.emit(path)
            elif choice == "view":
                self.view_folder(path)
            return choice
        self.view_folder(path)
        self.log_message.emit(
            f"{path.name} is not a BIDS dataset: opened for viewing only, nothing is "
            "written into it")
        return "view"

    def ask_bids_folder(self, path: Path) -> str:
        """``"project"``, ``"view"`` or ``""`` for a BIDS dataset."""
        from PyQt6.QtWidgets import QMessageBox

        from ..project.adopt import is_managed

        managed = is_managed(path)
        current = self._project_root
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Open a BIDS dataset")
        box.setText(f"Open {path.name} as the project?")
        swap = (f" {current.name}, the project open now, stays in Recent projects."
                if current is not None and current != path else "")
        how = ("BIDS Manager keeps its history in the dataset's .bidsmgr/ folder, so "
               "every edit can be undone." if managed else
               "It is not a BIDS Manager project yet: opening it as one adds a "
               ".bidsmgr/ folder (and a README, .bidsignore or dataset description if it "
               "has none), so every edit can be undone. Nothing else changes.")
        box.setInformativeText(
            f"As the project, the Converter and the Editor both switch to it.{swap} "
            f"{how}\n\nOr view it only: nothing is written into it, and editing and "
            "the dataset tools are off.")
        as_project = box.addButton("Open as project", QMessageBox.ButtonRole.AcceptRole)
        view = box.addButton("View only", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(as_project)
        box.exec()
        clicked = box.clickedButton()
        return "project" if clicked is as_project else "view" if clicked is view else ""

    def set_project_root(self, path: Optional[Path]) -> None:
        """The open project's dataset (the window, on opening one): shown
        and editable, every edit recorded."""
        self._project_root = Path(path) if path is not None else None
        if path is not None:
            self._set_root(Path(path), persist=False)

    def view_folder(self, path: Path) -> None:
        """Open ``path`` for viewing only: nothing is written into it."""
        self._set_root(Path(path), persist=True, view_only=True)

    def is_view_only(self) -> bool:
        return self._view_only

    def _on_path_action(self) -> None:
        """The path bar's one action: back to the project, or open the
        folder on screen as the project."""
        root = self.current_root()
        if not self._view_only or root is None:
            return
        if self._project_root is not None:
            self.set_project_root(self._project_root)
        else:
            self.open_project_requested.emit(root)

    def _apply_mode(self, path: Path, view_only: bool) -> None:
        """Everything that differs between the project and a folder open for
        viewing: what the path bar says, and whether anything can write."""
        from ..project.adopt import looks_like_bids

        self._view_only = bool(view_only)
        bids = looks_like_bids(path)
        if view_only:
            self._path_bar.set_label("Folder")
            self._path_bar.set_chip(
                "warn", "view only",
                "Opened for viewing: nothing is written into this folder, and editing "
                "and the dataset tools are off." + (
                    " Open it as a project to edit it." if bids else
                    " It is not a BIDS dataset."))
            if self._project_root is not None:
                self._path_bar.set_action(f"Back to {self._project_root.name}",
                                          "Show the open project again")
            elif bids:
                self._path_bar.set_action(
                    "Open as project",
                    "Make this dataset the project: the Converter and the Editor both "
                    "switch to it, and every edit can be undone")
            else:
                self._path_bar.set_action("")
        else:
            self._path_bar.set_label("Dataset")
            in_project = self._project_root is not None and path == self._project_root
            self._path_bar.set_chip(
                "success" if in_project else "accent", "project" if in_project else "dataset",
                "The open project: every edit is recorded and can be undone."
                if in_project else "Edits are recorded in the dataset's .bidsmgr/ folder "
                                   "and can be undone.")
            self._path_bar.set_action("")
        for pane in (self._sidecar_form, self._tsv_viewer, self._citation_pane,
                     self._bidsignore_pane, self._text_pane):
            pane.set_read_only(view_only)
        for viewer in (self._nifti_viewer, self._signal_viewer, self._spectrum_viewer):
            viewer.set_read_only(view_only)
        self._tree_pane.read_only = view_only
        # Validation reads only; it means something for a BIDS dataset.
        self._can_validate = (not view_only) or bids

    #: Validation applies to the folder on screen (a BIDS dataset).
    _can_validate = True

    def _set_root(self, path: Path, *, persist: bool, view_only: bool = False) -> None:
        self._tree_pane.set_root(path)
        self._tree_pane.clear_badges()
        self._path_bar.set_value(str(path), ok=True)
        self._apply_mode(Path(path), view_only)
        # Switching roots invalidates any stored report + form state.
        self._report = None
        self._show_pane(self._sidecar_form)
        self._validation_pane.set_report(None)
        self._validation_pane.set_current_file(None, None)
        self._hide_chips()
        # Dataset-level validation and the tools, as the mode allows.
        self._validate_dataset_btn.setEnabled(self._can_validate)
        self._tools_btn.setEnabled(not view_only)
        self._tools_btn.setToolTip(
            "Open the folder as a project to use the dataset tools: each of them "
            "writes." if view_only else "")
        self._refresh_deface_action()
        if persist:
            from .app_settings import AppSettings
            AppSettings.remember_editor_bids_root(path)

    def _on_tree_changed(self, root) -> None:
        """The dataset changed on disk: re-read what the field levels depend on.

        ``Authors`` is recommended only while there is no CITATION.cff, so a
        level read before a citation file was written would stay wrong in
        every form. Read again on a pool thread: it is a walk of the tree
        (no computation, guard 8b) and the first form must not pay for it.
        """
        from PyQt6.QtCore import QThreadPool

        from .. import schema

        schema.forget_dataset()
        if root is not None:
            QThreadPool.globalInstance().start(lambda: schema.warm_dataset(Path(root)))

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_progress(self, message: str) -> None:
        self.log_message.emit(message)

    def _on_report_ready(
        self, report: ValidationReport, root: Path,
    ) -> None:
        self._report = report
        self._guide_reports += 1
        self._stamp_tree_badges(report)
        self._update_chips(report)
        # Refresh the form for whatever file the user already has
        # selected — the FileVerdict for it just became available.
        current = self._sidecar_form.current_file()
        if current is not None:
            self._sidecar_form.set_file(current, self.current_root(), report)
        # The validation panel binds against the in-memory report, and the
        # file in focus, in ONE render (two setters rendered it twice).
        self._validation_pane.set_report_and_file(report, current, self.current_root())
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
        self._validate_dataset_btn.setEnabled(
            self.current_root() is not None and self._can_validate)
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
        from ..editor import report_index

        report = self._report
        existing_by_path = {fv.path: i for i, fv in enumerate(report.files)}
        for fv in verdicts:
            if fv.path in existing_by_path:
                old = report.files[existing_by_path[fv.path]]
                self._adjust_counts(report, old, -1)
                report.files[existing_by_path[fv.path]] = fv
            else:
                existing_by_path[fv.path] = len(report.files)
                report.files.append(fv)
            self._adjust_counts(report, fv, +1)
        report_index.forget(report)
        # Refresh the UI: the merged files' badges only, and the folders above.
        root = self.current_root()
        if root is not None:
            self._tree_pane.update_badges(*self._badge_maps(verdicts, root))
        self._update_chips(report)
        self._validation_pane.set_report_and_file(
            report, self._sidecar_form.current_file(), self.current_root())
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
        has_root = self.current_root() is not None and self._can_validate
        running = self._partial_worker is not None and getattr(
            self._partial_worker, "isRunning", lambda: False,
        )()
        if path is None or not has_root or running:
            self._validate_file_btn.setEnabled(False)
            self._validate_folder_btn.setEnabled(False)
        else:
            self._validate_file_btn.setEnabled(path.is_file())
            self._validate_folder_btn.setEnabled(path.is_dir())
        # The icons in the app's icon colour, like every other toolbar icon
        # (they used to turn green for the selected kind, which read as a
        # verdict on the file). Enabled or not says which one applies.
        icons.apply_button(self._validate_file_btn, "file_check")
        icons.apply_button(self._validate_folder_btn, "folder_check")

    @staticmethod
    def _adjust_counts(report: ValidationReport, verdict: FileVerdict, sign: int) -> None:
        """Take one file's findings out of the counts (``sign`` -1) or put
        them in (+1), and the severity from the counts.

        By difference, because merging one re-validated file used to recount
        every finding in the report (8.3 s on 287,000 findings). The counts
        are per finding and include the dataset and folder findings, so the
        severity follows from them alone.
        """
        counts = dict(report.counts)
        issues = [i for i in verdict.issues if not i.mirrored]
        counts["err"] = counts.get("err", 0) + sign * sum(
            1 for i in issues if i.severity is Severity.ERR)
        counts["warn"] = counts.get("warn", 0) + sign * sum(
            1 for i in issues if i.severity is Severity.WARN)
        if verdict.severity is Severity.OK:
            counts["ok"] = counts.get("ok", 0) + sign
        report.counts = {k: max(0, v) for k, v in counts.items()}
        report.severity = (Severity.ERR if report.counts.get("err") else
                           Severity.WARN if report.counts.get("warn") else Severity.OK)

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
        severities, counts = self._badge_maps(report.files, root)
        self._tree_pane.set_badges(severities, counts)

    def _badge_maps(self, verdicts, root: Path) -> tuple[dict, dict]:
        """``({path: badge}, {path: (errors, warnings)})`` for ``verdicts``."""
        allowed = self._allowed_severities()
        allow_err, allow_warn = Severity.ERR in allowed, Severity.WARN in allowed
        severities: dict[str, str] = {}
        counts: dict[str, tuple[int, int]] = {}
        # Joined as strings, not resolved: the tree maps its root's spelling
        # once (``BidsTreePane._key``). A resolve per file was 4.4 s on a
        # 21,663-file report, and three passes over each file's findings
        # most of what was left.
        base = str(root) + os.sep
        for fv in verdicts:
            # Badge = worst ALLOWED issue on the file; clean / filtered-out
            # files get the green "ok" dot. Mirrored findings (the same
            # finding shown on the editable sidecar as well) colour the badge
            # but are not counted, or every number in the tree would double.
            worst = n_err = n_warn = 0
            for issue in fv.issues:
                if issue.severity is Severity.ERR and allow_err:
                    worst = 2
                    n_err += not issue.mirrored
                elif issue.severity is Severity.WARN and allow_warn:
                    worst = max(worst, 1)
                    n_warn += not issue.mirrored
            path = fv.path
            key = str(path) if path.is_absolute() else base + str(path)
            severities[key] = ("ok", "warn", "err")[worst]
            if n_err or n_warn:
                counts[key] = (n_err, n_warn)
        return severities, counts

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
            (not busy) and (self.current_root() is not None) and self._can_validate
        )

    # ------------------------------------------------------------------
    # Tree ↔ sidecar form wiring
    # ------------------------------------------------------------------

    def _on_file_selected(self, path: Path) -> None:
        """User picked a row in the BIDS tree; route it to its viewer.

        * ``.tsv`` / ``.tsv.gz`` -> :class:`TsvViewerPane` (table; a
          continuous recording also offers Visualize);
        * a NIfTI-MRS file in ``mrs/`` -> the spectrum viewer;
        * any other NIfTI -> the volume viewer;
        * EEG / MEG / iEEG recordings (``.fif``, ``.edf``, ``.set``,
          ``.vhdr``, ``.cnt``, CTF ``.ds``, ...) -> the signal viewer;
        * JSON -> :class:`SidecarFormPane`;
        * ``.bval`` / ``.bvec`` -> :class:`GradientPane`; README, CHANGES,
          LICENSE, Markdown, HTML and text -> :class:`TextPane`; pictures ->
          :class:`PicturePane`; a DICOM (by its content) ->
          :class:`DicomPane` (``formats.document_kind``);
        * anything else -> :class:`FileInfoPane` (what it is, and the
          system's app to open it).

        Routed from the NAME and the folder, never by reading the file
        (``viz.data.formats``), except a DICOM's 132-byte preamble. Every pane but the one shown is cleared, so
        none keeps a stale file (the MRS pane used to, after any selection
        but a NIfTI).
        """
        self._open_path = Path(path)
        self._guide_file_opens += 1
        root = self.current_root()
        if path.is_dir():
            if is_recording_path(path):
                # CTF .ds / EGI .mff are directory-shaped recordings.
                self._show_pane(self._signal_viewer, path, root)
            else:
                # A folder carries no sidecar; the validation pane still
                # follows the focus.
                self._show_pane(self._sidecar_form)
            self._validation_pane.set_current_file(path, root)
            return
        name = path.name.lower()
        if name == "citation.cff":
            # YAML rather than JSON, so the sidecar form cannot render it.
            self._citation_pane.set_file(path, root)
            self._center_stack.setCurrentWidget(self._citation_pane)
            self._validation_pane.set_current_file(path, root)
            return
        if name == ".bidsignore":
            # What matters is which files each pattern hides, which a text
            # editor cannot say.
            self._bidsignore_pane.set_root(root)
            self._bidsignore_pane.set_report(self._report)
            self._center_stack.setCurrentWidget(self._bidsignore_pane)
            self._validation_pane.set_current_file(path, root)
            return
        if name.endswith(".tsv") or name.endswith(".tsv.gz"):
            self._show_pane(self._tsv_viewer, path, root)
        elif is_mrs_path(path):
            # A NIfTI by container only: the data block is a complex FID.
            self._note_image(path)
            self._show_pane(self._spectrum_viewer, path, root)
        elif name.endswith((".nii", ".nii.gz", ".mgz", ".mgh")):
            self._note_image(path)
            self._show_pane(self._nifti_viewer, path, root)
        elif is_recording_path(path):
            self._show_pane(self._signal_viewer, path, root)
        elif name.endswith(".json"):
            self._show_pane(self._sidecar_form, path, root)
        else:
            from ..viz.data.formats import document_kind

            pane = {"gradients": self._gradient_pane, "picture": self._picture_pane,
                    "markdown": self._text_pane, "html": self._text_pane,
                    "text": self._text_pane, "dicom": self._dicom_pane,
                    }.get(document_kind(path), self._file_info_pane)
            self._show_pane(pane, path, root)
        self._validation_pane.set_current_file(path, root)

    def _note_image(self, path: Path) -> None:
        path = Path(path)
        self._images_shown = [p for p in self._images_shown if p != path][-1:] + [path]

    def _overlay_base(self, clicked: Path) -> Optional[Path]:
        """The image a right-clicked one would be drawn over: the last other
        image shown in the volume viewer (never a spectrum)."""
        clicked = Path(clicked)
        for p in reversed(self._images_shown):
            if p != clicked and p.is_file() and not is_mrs_path(p):
                return p
        return None

    def _on_header_requested(self, path, rule: str) -> None:
        """A header finding: show the image, with the header inspector open
        on the row that is the evidence for it."""
        if path is None:
            return
        path = Path(path)
        if self._nifti_viewer.current_file() != path:
            self.select_file_in_tree(path)
        self._nifti_viewer.show_header(rule)

    def _on_overlay_requested(self, overlay, base) -> None:
        """Show ``base`` again and draw ``overlay`` over it once it is open."""
        self.select_file_in_tree(Path(base))
        if self._nifti_viewer.current_file() != Path(base):
            self._show_pane(self._nifti_viewer, Path(base), self.current_root())
        self._nifti_viewer.add_overlay(Path(overlay))

    def _viewer_panes(self) -> tuple:
        return (self._sidecar_form, self._tsv_viewer, self._nifti_viewer,
                self._signal_viewer, self._spectrum_viewer, self._text_pane,
                self._gradient_pane, self._picture_pane, self._file_info_pane,
                self._dicom_pane)

    def _show_pane(self, pane, path: Optional[Path] = None,
                   root: Optional[Path] = None) -> None:
        """Bind ``pane`` to ``path`` and show it; clear every other one."""
        for other in self._viewer_panes():
            if other is not pane:
                self._clear_pane(other)
        if pane is self._sidecar_form:
            self._sidecar_form.set_file(path, root, self._report)
        elif path is not None:
            pane.set_file(path, root)
        self._name_pane(pane, path)
        self._center_stack.setCurrentWidget(pane)

    def _name_pane(self, pane, path: Optional[Path]) -> None:
        """Under the frame's "Viewer", the pane's own header names the file
        shown (a second "Sidecar" or "Table" said nothing new). The panes
        that name what they show themselves are left alone."""
        if pane not in (self._sidecar_form, self._tsv_viewer, self._nifti_viewer,
                        self._signal_viewer, self._spectrum_viewer):
            return
        from .widgets.primitives import PaneHeader

        header = pane.findChild(PaneHeader)
        if header is None:
            return
        if not hasattr(header, "_generic"):
            header._generic = header.text()
        header.setText(path.name if path is not None else header._generic)

    def _clear_pane(self, pane) -> None:
        if pane is self._sidecar_form:
            self._sidecar_form.set_file(None, None, None)
        else:
            pane.set_file(None, None)

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

    def _on_document_link(self, path: Path) -> None:
        root = self.current_root()
        if root is not None and (Path(path) == root or root in Path(path).parents):
            self.select_file_in_tree(Path(path))

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

    # -- what validation ignores (.bidsignore) ----------------------------

    def _on_ignore(self, choice) -> None:
        """Ignore in validation (the tree's right-click): one step of the
        history, written at once, the dataset validated again."""
        self._apply_ignore_choice(choice)

    def _on_unignore(self, choice) -> None:
        """Stop ignoring: the lines removed and the exceptions added in ONE
        step, however many files were picked."""
        self._apply_ignore_choice(choice)

    def _apply_ignore_choice(self, choice) -> None:
        from ..editor import bidsignore as bi

        root = self.current_root()
        if root is None or self._tree_pane.read_only:
            return
        added, removed = bi.apply_choice(root, choice)
        if not (added or removed):
            self._after_ignore_change("Nothing to change in .bidsignore.", changed=False)
            return
        self._after_ignore_change(ignore_message(choice, added, removed))

    def _after_ignore_change(self, message: str, *, changed: bool = True,
                             from_pane: bool = False) -> None:
        """The tree dims what is now ignored, the list recounts (the list
        recounts itself when the change was made there), and the findings
        follow by validating again (only when there were some)."""
        window = self.window()
        if hasattr(window, "statusBar"):
            window.statusBar().showMessage(message, 6000)
        if not changed:
            return
        self._tree_pane.refresh()
        if not from_pane and self._center_stack.currentWidget() is self._bidsignore_pane:
            self._bidsignore_pane.reload()
        self._sync_undo_redo()
        if self._report is not None:
            self.start_dataset_validation()

    def _on_ignore_list(self) -> None:
        """The ignore list in the centre, without "Show hidden files" (the
        file itself is a dotfile the tree hides)."""
        root = self.current_root()
        if root is None:
            return
        self._on_file_selected(root / ".bidsignore")

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

    def _on_quality_check(self) -> None:
        """The dataset's images, checked (non-modal: they are looked at
        while it stays open)."""
        from .quality_check_dialog import QualityCheckDialog

        root = self.current_root()
        if root is None:
            return
        from PyQt6 import sip

        old = getattr(self, "_quality_dialog", None)
        # Checked, not tracked through ``destroyed`` (CLAUDE.md guard 8d).
        if old is not None and not sip.isdeleted(old) and old.isVisible():
            old.raise_()
            old.activateWindow()
            return
        dlg = QualityCheckDialog(root, open_image=self.open_with_quality, parent=self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._quality_dialog = dlg
        dlg.show()

    def open_with_quality(self, path: Path) -> None:
        """Open ``path`` in the viewer and its quality report beside it."""
        self.select_file_in_tree(Path(path))
        presenter = getattr(self._nifti_viewer, "presenter", None)
        if presenter is not None and hasattr(presenter, "check_quality"):
            presenter.check_quality()

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

    def _on_anonymise(self, targets: Optional[list] = None) -> None:
        """Anonymise the dataset's metadata, after showing every change. From
        the tree's right-click, on the subjects and sessions clicked."""
        from ..anonymise.scope import scope_of
        from .anonymise_dialog import AnonymiseDialog

        root = self.current_root()
        if root is None:
            return
        scope = scope_of(root, [Path(t) for t in targets]) if targets else None
        dlg = AnonymiseDialog(root, parent=self, scope=scope)
        if dlg.exec() and dlg.written():
            self._sidecar_form.set_file(None, None, None)
            self._tree_pane.set_root(root)
            if self._open_path is not None:
                self._on_file_selected(self._open_path)
            self._sync_undo_redo()
            window = self.window()
            if hasattr(window, "statusBar"):
                window.statusBar().showMessage(
                    f"Anonymised {dlg.written()} file(s). Undo takes it back.", 6000)
            if self._report is not None:
                self.start_dataset_validation()

    def _on_tables(self) -> None:
        """Write the scans and sessions tables, after showing what changes."""
        from .tables_dialog import TablesDialog

        root = self.current_root()
        if root is None:
            return
        dlg = TablesDialog(root, parent=self)
        if dlg.exec() and dlg.written():
            self._tree_pane.set_root(root)
            if self._open_path is not None and self._open_path in dlg.written():
                self._on_file_selected(self._open_path)
            self._sync_undo_redo()
            window = self.window()
            if hasattr(window, "statusBar"):
                n = len(dlg.written())
                window.statusBar().showMessage(
                    f"Wrote {n} table{'s' if n != 1 else ''}. Undo takes "
                    f"{'it' if n == 1 else 'them'} back.", 6000)
            if self._report is not None:
                self.start_dataset_validation()

    def _on_sidecars(self, targets: Optional[list] = None,
                     complete: Optional[bool] = None) -> None:
        """Write or complete sidecars from the recordings, after showing every
        field. From the Tools menu it acts on the tree's selection, else the
        dataset; from the tree's right-click, on what was clicked."""
        from ..editor.sidecars import sidecar_path
        from .sidecars_dialog import SidecarsDialog

        root = self.current_root()
        if root is None:
            return
        chosen = [Path(t) for t in (targets or self._tree_pane.selected_paths())]
        dlg = SidecarsDialog(root, chosen or None, parent=self, complete=complete)
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.written():
            return
        written = dlg.written()
        self._tree_pane.set_root(root)
        if self._open_path is not None and (
                self._open_path in written
                or sidecar_path(self._open_path) in written):
            self._on_file_selected(self._open_path)
        self._sync_undo_redo()
        window = self.window()
        if hasattr(window, "statusBar"):
            n = len(written)
            window.statusBar().showMessage(
                f"Wrote {n} sidecar{'s' if n != 1 else ''}. Undo takes "
                f"{'it' if n == 1 else 'them'} back.", 6000)
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
        """The :class:`FileVerdict` in the current report for ``path``, or
        ``None``: one lookup (``editor.report_index``)."""
        from ..editor.report_index import find_verdict

        return find_verdict(self._report, self.current_root(), path)

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Undo / redo (delegates to the active editable center pane)
    # ------------------------------------------------------------------

    def _active_history_pane(self):
        """The current center pane if it keeps unsaved edits to undo, else None."""
        w = self._center_stack.currentWidget()
        if w in (self._sidecar_form, self._tsv_viewer):
            return w
        return None

    def _history_root(self) -> Optional[Path]:
        """The dataset whose history Undo and Redo walk: none when a folder
        is open for viewing only (nothing there was written by us)."""
        if self._view_only:
            return None
        return self.current_root()

    def _open_pane_unsaved(self) -> bool:
        """The shown file has edits not saved yet. A dataset step would reload
        it from disk and lose them, so it waits for a save or a revert."""
        is_dirty = getattr(self._center_stack.currentWidget(), "is_dirty", None)
        try:
            return bool(is_dirty()) if callable(is_dirty) else False
        except Exception:  # noqa: BLE001 - a pane mid-load
            return False

    def _dataset_step(self, which: str):
        """The dataset history's next step to undo or redo, or None."""
        root = self._history_root()
        if root is None:
            return None
        from ..project import operations as ops

        try:
            return (ops.peek_undo if which == "undo" else ops.peek_redo)(root)
        except OSError:
            return None

    @staticmethod
    def _step_text(step) -> str:
        """The step's label, with how many files it touched unless the label
        already says ("Fill RepetitionTime in 3 files")."""
        import re

        n = len(step.files)
        if n > 1 and not re.search(r"\b\d+ file", step.label):
            return f"{step.label} ({n} files)"
        return step.label

    def _sync_undo_redo(self, *args) -> None:
        del args
        from PyQt6.QtGui import QKeySequence

        def key(k) -> str:
            return QKeySequence(k).toString(QKeySequence.SequenceFormat.NativeText)

        pane = self._active_history_pane()
        name = getattr(getattr(pane, "_current_file", None), "name", "the file")
        blocked = self._open_pane_unsaved()
        for which, btn, verb, std in (
            ("undo", self._undo_btn, "Undo", QKeySequence.StandardKey.Undo),
            ("redo", self._redo_btn, "Redo", QKeySequence.StandardKey.Redo),
        ):
            mine = pane is not None and (
                pane.has_unsaved_history() if which == "undo" else pane.has_unsaved_redo())
            step = None if (mine or blocked) else self._dataset_step(which)
            if mine:
                tip = f"{verb} the last edit to {name} (not saved yet)"
            elif step is not None:
                tip = f"{verb}: {self._step_text(step)}"
            elif blocked:
                tip = f"Save or revert {name} first: its unsaved edits would be lost"
            else:
                tip = f"Nothing to {verb.lower()}"
            btn.setEnabled(mine or step is not None)
            more = "\nRight-click for the whole history." if step is not None else ""
            btn.setToolTip(f"{tip} ({key(std)}){more}")

    #: How many steps the history menu lists; older ones are summarised.
    HISTORY_MENU_LIMIT = 30

    @staticmethod
    def _when(at: str) -> str:
        """``2026-10-19T14:05:31`` as "14:05" today, else "19 Oct 14:05"."""
        import datetime

        try:
            moment = datetime.datetime.fromisoformat(at)
        except ValueError:
            return ""
        if moment.date() == datetime.date.today():
            return moment.strftime("%H:%M")
        return f"{moment.day} {moment.strftime('%b %H:%M')}"

    def history_menu_actions(self, which: str, menu) -> list:
        """Fill ``menu`` with the history Undo (or Redo) would walk; returns
        the actions that act, newest first."""
        verb = "undo" if which == "undo" else "redo"
        root = self._history_root()
        if root is None:
            menu.addAction(f"Nothing to {verb} here").setEnabled(False)
            return []
        if self._open_pane_unsaved():
            pane = self._center_stack.currentWidget()
            name = getattr(getattr(pane, "_current_file", None), "name", "the open file")
            menu.addAction(f"Save or revert {name} first").setEnabled(False)
            return []
        from ..project import operations as ops

        steps = ops.history(root, which)
        if not steps:
            menu.addAction(f"Nothing to {verb}").setEnabled(False)
            return []
        acted = []
        for i, step in enumerate(steps[:self.HISTORY_MENU_LIMIT]):
            # A tab puts the time in the menu's right-hand column.
            action = menu.addAction(f"{self._step_text(step)}\t{self._when(step.at)}")
            action.triggered.connect(
                lambda _c=False, n=i + 1: self._step_dataset_many(which, n))
            acted.append(action)
        if len(steps) > self.HISTORY_MENU_LIMIT:
            menu.addAction(f"and {len(steps) - self.HISTORY_MENU_LIMIT} earlier"
                           ).setEnabled(False)
        return acted

    def _show_history_menu(self, which: str, where) -> None:
        from .viz.menus import popup_menu

        menu = popup_menu(self)
        menu.setToolTipsVisible(True)
        self.history_menu_actions(which, menu)
        menu.popup(where)
        self._history_menu = menu

    def _step_dataset_many(self, which: str, count: int) -> None:
        """Undo (or redo) ``count`` steps at once, after one question."""
        if count <= 1:
            self._step_dataset(which)
            return
        root = self._history_root()
        if root is None or self._open_pane_unsaved():
            return
        from ..project import operations as ops

        steps = ops.history(root, which)[:count]
        if len(steps) < count:
            return
        verb = "Undo" if which == "undo" else "Redo"
        newest = ops.peek_undo(root, check=True) if which == "undo" else ops.peek_redo(root, check=True)
        if not self._confirm_steps(which, steps, newest.changed_since if newest else ()):
            return
        from .fs_watch import watchers_released

        step_fn = ops.undo_last if which == "undo" else ops.redo_last
        done = 0
        try:
            with watchers_released():
                for _ in range(count):
                    step_fn(root)
                    done += 1
        except Exception as exc:  # noqa: BLE001 - each step is all or nothing
            log.warning("%s stopped after %d step(s): %s", verb, done, exc)
            QMessageBox.warning(self, verb, f"{verb} stopped after {done} of {count} "
                                f"steps; the step that failed changed nothing.\n\n{exc}")
        self._after_history_step(root)
        window = self.window()
        if done and hasattr(window, "statusBar"):
            word = "Undone" if which == "undo" else "Redone"
            window.statusBar().showMessage(f"{word}: {done} steps", 6000)

    def _confirm_steps(self, which: str, steps, changed_since) -> bool:
        verb = "Undo" if which == "undo" else "Redo"
        files = len({f for step in steps for f in step.files})
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(verb)
        box.setText(f"{verb} the last {len(steps)} steps?")
        listed = "\n".join(f"  {self._step_text(s)}" for s in steps[:6])
        more = f"\n  and {len(steps) - 6} more" if len(steps) > 6 else ""
        text = f"{listed}{more}\n\n{files} file(s) in all, newest first."
        if changed_since:
            names = ", ".join(Path(p).name for p in changed_since[:3])
            text += (f"\n\nSome of them changed after the newest step, by hand or "
                     f"by another program ({names}). {verb} replaces those changes too.")
        box.setInformativeText(text)
        go = box.addButton(f"{verb} {len(steps)} steps", QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel if changed_since else go)
        box.exec()
        return box.clickedButton() is go

    def _on_editor_undo(self) -> None:
        pane = self._active_history_pane()
        if pane is not None and pane.has_unsaved_history():
            pane.undo()
            return
        self._step_dataset("undo")

    def _on_editor_redo(self) -> None:
        pane = self._active_history_pane()
        if pane is not None and pane.has_unsaved_redo():
            pane.redo()
            return
        self._step_dataset("redo")

    def _step_dataset(self, which: str) -> None:
        """Undo or redo ONE step of the dataset's history: every file of it
        together. A step of several files, or one whose files were changed
        since, asks first."""
        root = self._history_root()
        if root is None or self._open_pane_unsaved():
            return
        from ..project import operations as ops

        peek = ops.peek_undo if which == "undo" else ops.peek_redo
        step = peek(root, check=True)
        if step is None:
            return
        if (len(step.files) > 1 or step.changed_since) and not self._confirm_step(which, step):
            return
        from .fs_watch import watchers_released

        verb = "Undo" if which == "undo" else "Redo"
        try:
            # Folders may move back (a subject renamed), and Windows cannot
            # move a folder the tree is watching.
            with watchers_released():
                (ops.undo_last if which == "undo" else ops.redo_last)(root)
        except Exception as exc:  # noqa: BLE001 - all or nothing: say so, change nothing
            log.warning("%s failed: %s", verb, exc)
            QMessageBox.warning(self, verb, f"{verb} could not be completed, so nothing "
                                f"was changed.\n\n{exc}")
            return
        self._after_history_step(root)
        done = "Undone" if which == "undo" else "Redone"
        window = self.window()
        if hasattr(window, "statusBar"):
            window.statusBar().showMessage(f"{done}: {self._step_text(step)}", 6000)

    def _confirm_step(self, which: str, step) -> bool:
        verb = "Undo" if which == "undo" else "Redo"
        n = len(step.files)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(verb)
        box.setText(f'{verb} "{step.label}"?')
        text = (f"This puts {n} files back as they were before it." if which == "undo"
                else f"This makes the change to {n} files again.")
        if step.changed_since:
            k = len(step.changed_since)
            names = ", ".join(Path(p).name for p in step.changed_since[:3])
            more = f" and {k - 3} more" if k > 3 else ""
            text += (f"\n\n{k} of them changed after it, by hand or by another "
                     f"program ({names}{more}). {verb} replaces those changes too.")
        box.setInformativeText(text)
        go = box.addButton(verb, QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel if step.changed_since else go)
        box.exec()
        return box.clickedButton() is go

    def _after_history_step(self, root: Path) -> None:
        """The dataset changed under the panes: show it as it is now, the
        open file included (again, if it still exists)."""
        self._sidecar_form.set_file(None, None, None)
        self._reload_open_image()
        self._tree_pane.refresh()
        opened = self._open_path
        if opened is not None and opened.exists():
            self._on_file_selected(opened)
        if self._report is not None:
            self.start_dataset_validation()
        self._sync_undo_redo()

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
        self._text_pane.repaint_for_palette(pal)
        self._gradient_pane.repaint_for_palette(pal)
        self._picture_pane.repaint_for_palette(pal)
        self._file_info_pane.repaint_for_palette(pal)
        self._dicom_pane.repaint_for_palette(pal)
        self._signal_viewer.repaint_for_palette(pal)
        self._spectrum_viewer.repaint_for_palette(pal)
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
