"""Settings dialog — surface CLI knobs the GUI uses.

Reads / writes :class:`bidsmgr.gui.app_settings.AppSettings` via
``QSettings``. All changes are applied on **Save** (no live binding) so
the user can experiment with values and cancel without commit.

Tabs: Display / System / Scan / Convert + post-convert. The Convert tab
lays the post-convert chain out as an indented hierarchy (parent step +
its sub-options), and a "Restore defaults" button resets every widget to
the :class:`AppSettings` field defaults.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import schema
from ..classifier import sequence_dict
from ..classifier import user_rules
from ..util.system_info import SystemInfo, get_system_info
from .app_settings import AppSettings


def _indented(child: QWidget, *, indent: int = 22) -> QWidget:
    """Wrap ``child`` in a left-indented container so it reads as a
    sub-option nested under the checkbox above it (the post-convert
    hierarchy tree)."""
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(indent, 0, 0, 0)
    lay.setSpacing(6)
    lay.addWidget(child)
    lay.addStretch(1)
    return box


def _bind_children(parent_cb: QCheckBox, *children: QWidget) -> None:
    """Enable ``children`` only while ``parent_cb`` is checked, and sync
    immediately so the initial state is correct."""
    def _sync(checked: bool) -> None:
        for c in children:
            c.setEnabled(checked)
    parent_cb.toggled.connect(_sync)
    _sync(parent_cb.isChecked())


class SettingsDialog(QDialog):
    """Settings dialog: Display / System / Scan / Convert + post-convert.

    Theme + post-convert chain live under their natural homes. The
    inspector column visibility is NOT here — it's controlled via the
    table header's right-click menu, and that menu writes through to
    the same QSettings namespace.
    """

    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BIDS-Manager — Settings")
        self.resize(560, 600)
        self._settings = settings
        # Detected once: the worker-count spinboxes are capped at the host's
        # logical thread count so the user can never ask for more workers
        # than the machine has threads.
        self._sys: SystemInfo = get_system_info()

        v = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(self._build_bids_version_tab(), "BIDS version")
        tabs.addTab(self._build_display_tab(), "Display")
        tabs.addTab(self._build_system_tab(), "System")
        tabs.addTab(self._build_scan_tab(), "Scan")
        tabs.addTab(self._build_scan_rules_tab(), "Scan rules")
        tabs.addTab(self._build_convert_tab(), "Convert + post-convert")
        tabs.addTab(self._build_validation_tab(), "Validation")
        v.addWidget(tabs, 1)

        # Save / Cancel / Restore defaults.
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults,
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        buttons.button(
            QDialogButtonBox.StandardButton.RestoreDefaults
        ).clicked.connect(self._on_restore_defaults)
        v.addWidget(buttons)

        # Populate every widget from the current settings.
        self._load_into_widgets(self._settings)

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    # Font scale presets shown in the Display tab. The combo stores the
    # human-readable label; the float multiplier is the second element.
    _FONT_SCALE_PRESETS: list[tuple[str, float]] = [
        ("Compact (0.85x)",        0.85),
        ("Normal (1.00x)",         1.00),
        ("Comfortable (1.15x)",    1.15),
        ("Large (1.30x)",          1.30),
        ("Extra large (1.50x)",    1.50),
    ]

    # Combo presets for the header brand mark.
    _HEADER_LOGO_PRESETS: list[tuple[str, str]] = [
        ("Default (monochrome mark)", "default"),
        ("App icon (full color)",     "app_icon"),
    ]

    def _build_display_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["dark", "light"])
        form.addRow("Theme:", self._theme_combo)

        # Font scale: multiplies every font-size (QSS + delegate paints +
        # inline stylesheets + icon sizes) so the user can comfortably
        # nudge the whole UI up or down. Persisted under ``ui/font_scale``.
        self._font_scale_combo = QComboBox()
        for label, _value in self._FONT_SCALE_PRESETS:
            self._font_scale_combo.addItem(label)
        form.addRow("Font scale:", self._font_scale_combo)

        # Header brand artwork.
        self._header_logo_combo = QComboBox()
        for label, _value in self._HEADER_LOGO_PRESETS:
            self._header_logo_combo.addItem(label)
        form.addRow("Header logo:", self._header_logo_combo)

        hint = QLabel(
            "Theme can also be toggled live via the sun / moon button "
            "in the top header. Font scale and header logo apply on Save."
        )
        hint.setStyleSheet("color: #8b949e;")
        hint.setWordWrap(True)
        form.addRow("", hint)

        return w

    def _build_system_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        info = QGroupBox("System info (detected)")
        form = QFormLayout(info)

        threads = self._sys.logical_threads
        cores = self._sys.physical_cores
        cpu_txt = f"{threads} logical threads"
        if cores:
            cpu_txt += f"  /  {cores} physical cores"
        form.addRow("CPU:", QLabel(cpu_txt))

        ram_gib = self._sys.total_ram_gib
        form.addRow(
            "Memory:",
            QLabel(f"{ram_gib:.1f} GiB total" if ram_gib is not None else "unknown"),
        )

        v.addWidget(info)

        note = QLabel(
            "Parallel-worker counts (Scan and Convert) are capped at the "
            f"detected thread count ({threads}). Asking for more workers than "
            "the machine has threads only adds scheduling overhead, so the "
            "spinboxes will not go higher."
        )
        note.setStyleSheet("color: #8b949e;")
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)
        return w

    def _build_scan_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        defaults = QGroupBox("Scan defaults")
        form = QFormLayout(defaults)

        self._scan_jobs = QSpinBox()
        self._scan_jobs.setRange(1, self._sys.logical_threads)
        self._scan_jobs.setToolTip(
            f"Capped at the detected thread count ({self._sys.logical_threads})."
        )
        form.addRow("Parallel workers (-j):", self._scan_jobs)

        # EEG/MEG line frequency + montage are no longer scan settings: they
        # are recording metadata, edited as dropdowns per recording in the
        # inspection table (montage / line_freq columns) and dataset-wide in
        # the "Recording metadata" editor. Free-text fields here would have
        # been a second, inconsistent way to set the same values.

        self._scan_probe = QCheckBox(
            "Enable --probe-convert (run dcm2niix per series to enrich "
            "naming with the actual file count + extensions)"
        )
        form.addRow("Probe:", self._scan_probe)

        self._scan_preview = QCheckBox(
            "Record what the conversion fills in by itself"
        )
        self._scan_preview.setToolTip(
            "Note, per kind of file, the values dcm2niix and mne-bids produce, "
            "so the metadata form can show them instead of asking for a field "
            "nobody has to answer.\n\nCosts nothing extra: it reads what the "
            "scan, and any probe conversion, already produced. With Probe on "
            "it covers MRI and PET as well as EEG and MEG."
        )
        form.addRow("Converter fields:", self._scan_preview)

        self._scan_skip_bids_guess = QCheckBox(
            "Skip dcm2niix BidsGuess classifier (use only the legacy "
            "regex fallback layer)"
        )
        form.addRow("Classifier:", self._scan_skip_bids_guess)

        v.addWidget(defaults)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------
    # Scan rules tab (exclusions + user hints + read-only built-ins)
    # ------------------------------------------------------------------

    def _build_scan_rules_tab(self) -> QWidget:
        # Valid BIDS datatypes for the hint dropdowns (derivatives excluded -
        # user hints route to raw datatypes only).
        self._valid_datatypes = sorted(
            d for d in schema.list_datatypes() if d != "derivatives"
        )

        content = QWidget()
        v = QVBoxLayout(content)

        intro = QLabel(
            "These rules apply to MRI / DICOM series, which are classified "
            "from their SeriesDescription. EEG / MEG recordings are classified "
            "by a different built-in method (mne channel types) and are NOT "
            "affected by custom sequence hints. Path-based exclusions can still "
            "skip any modality."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #8b949e;")
        v.addWidget(intro)

        # Exclusions.
        excl_box = QGroupBox("Scan exclusions (skip matching series)")
        ebl = QVBoxLayout(excl_box)
        self._excl_table = QTableWidget(0, 3)
        self._excl_table.setHorizontalHeaderLabels(["Pattern", "Match against", "Mode"])
        self._excl_table.verticalHeader().setVisible(False)
        self._excl_table.setMinimumHeight(130)
        self._excl_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        ebl.addWidget(self._excl_table)
        ebl.addLayout(self._rule_row_buttons(self._add_exclusion_row, self._excl_table))
        v.addWidget(excl_box)

        # User hints (MRI / DICOM only).
        hint_box = QGroupBox("Custom sequence hints (MRI / DICOM only)")
        hbl = QVBoxLayout(hint_box)
        self._hint_table = QTableWidget(0, 6)
        self._hint_table.setHorizontalHeaderLabels(
            ["Patterns (comma-separated)", "Datatype", "Suffix", "Task", "Mode", "Force"]
        )
        self._hint_table.verticalHeader().setVisible(False)
        self._hint_table.setMinimumHeight(150)
        self._hint_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        hbl.addWidget(self._hint_table)
        force_hint = QLabel(
            "Datatype + suffix are chosen from the BIDS schema (no free text). "
            "'Force' overrides even the dcm2niix classifier; otherwise a hint "
            "only beats the built-in regex layer. 'Task' is the optional "
            "task-<label> for func rows."
        )
        force_hint.setWordWrap(True)
        force_hint.setStyleSheet("color: #8b949e;")
        hbl.addWidget(force_hint)
        hbl.addLayout(self._rule_row_buttons(self._add_hint_row, self._hint_table))
        v.addWidget(hint_box)

        # Read-only built-in criteria.
        builtin_box = QGroupBox("Built-in MRI classifier criteria (read-only)")
        bbl = QVBoxLayout(builtin_box)
        builtin_note = QLabel(
            "What the MRI classifier already matches. EEG / MEG do not use this "
            "table - their datatype comes from mne channel types."
        )
        builtin_note.setWordWrap(True)
        builtin_note.setStyleSheet("color: #8b949e;")
        bbl.addWidget(builtin_note)
        builtin = QTableWidget(0, 4)
        builtin.setHorizontalHeaderLabels(
            ["Label / group", "Datatype", "Suffix", "Match patterns"]
        )
        builtin.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        builtin.verticalHeader().setVisible(False)
        builtin.setMinimumHeight(240)
        builtin.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self._populate_builtin_criteria(builtin)
        bbl.addWidget(builtin)
        v.addWidget(builtin_box)
        v.addStretch(1)

        # Whole tab scrolls (not just the built-in table).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _rule_row_buttons(self, add_cb, table: QTableWidget) -> QHBoxLayout:
        bar = QHBoxLayout()
        add = QPushButton("Add row")
        add.clicked.connect(lambda: add_cb())
        rem = QPushButton("Delete selected")
        rem.clicked.connect(lambda: self._delete_selected_rows(table))
        bar.addStretch(1)
        bar.addWidget(add)
        bar.addWidget(rem)
        return bar

    @staticmethod
    def _delete_selected_rows(table: QTableWidget) -> None:
        for r in sorted({i.row() for i in table.selectedIndexes()}, reverse=True):
            table.removeRow(r)

    def _add_exclusion_row(self, pattern: str = "", target: str = "sequence",
                           mode: str = "substring") -> None:
        t = self._excl_table
        r = t.rowCount()
        t.insertRow(r)
        t.setItem(r, 0, QTableWidgetItem(pattern))
        target_cb = QComboBox()
        target_cb.addItems(list(user_rules.EXCLUSION_TARGETS))
        target_cb.setCurrentText(target if target in user_rules.EXCLUSION_TARGETS else "sequence")
        t.setCellWidget(r, 1, target_cb)
        mode_cb = QComboBox()
        mode_cb.addItems(list(user_rules.MATCH_MODES))
        mode_cb.setCurrentText(mode if mode in user_rules.MATCH_MODES else "substring")
        t.setCellWidget(r, 2, mode_cb)

    def _add_hint_row(self, patterns: str = "", datatype: str = "", suffix: str = "",
                      task: str = "", mode: str = "substring", force: bool = False) -> None:
        t = self._hint_table
        r = t.rowCount()
        t.insertRow(r)
        t.setItem(r, 0, QTableWidgetItem(patterns))

        # Datatype + suffix are constrained dropdowns (no hand-typed labels).
        # The suffix list depends on the chosen datatype, so it re-fills
        # whenever the datatype changes.
        dt_cb = QComboBox()
        dt_cb.addItems(self._valid_datatypes)
        suffix_cb = QComboBox()

        def _refill_suffixes(dt: str) -> None:
            suffix_cb.blockSignals(True)
            suffix_cb.clear()
            try:
                suffix_cb.addItems(sorted(schema.list_suffixes(dt)))
            except Exception:
                pass
            suffix_cb.blockSignals(False)

        dt_cb.currentTextChanged.connect(_refill_suffixes)
        if datatype in self._valid_datatypes:
            dt_cb.setCurrentText(datatype)
        _refill_suffixes(dt_cb.currentText())   # seed for the initial datatype
        if suffix:
            idx = suffix_cb.findText(suffix)
            if idx >= 0:
                suffix_cb.setCurrentIndex(idx)
        t.setCellWidget(r, 1, dt_cb)
        t.setCellWidget(r, 2, suffix_cb)

        t.setItem(r, 3, QTableWidgetItem(task))
        mode_cb = QComboBox()
        mode_cb.addItems(list(user_rules.MATCH_MODES))
        mode_cb.setCurrentText(mode if mode in user_rules.MATCH_MODES else "substring")
        t.setCellWidget(r, 4, mode_cb)
        force_item = QTableWidgetItem()
        force_item.setFlags(
            Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
        )
        force_item.setCheckState(Qt.CheckState.Checked if force else Qt.CheckState.Unchecked)
        t.setItem(r, 5, force_item)

    @staticmethod
    def _populate_builtin_criteria(table: QTableWidget) -> None:
        rows: list[tuple[str, str, str, str]] = []
        for label, hint in sequence_dict.SEQUENCE_HINTS.items():
            dt = hint.container_override or (hint.datatype or "")
            rows.append((label, dt, hint.suffix or "", ", ".join(hint.patterns)))
        for rgx, suffix, dt in sequence_dict._DWI_DERIVATIVE_PATTERNS:
            rows.append(("dwi-derivative", dt, suffix, rgx))
        for task_label, pats in sequence_dict.TASK_HINT_PATTERNS.items():
            rows.append((f"task:{task_label}", "func", "(task entity)", ", ".join(pats)))
        table.setRowCount(len(rows))
        for r, cells in enumerate(rows):
            for col, val in enumerate(cells):
                it = QTableWidgetItem(val)
                it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                table.setItem(r, col, it)

    def _read_scan_rules(self) -> tuple[list[dict], list[dict], Optional[str]]:
        """Read both editable tables into list[dict]. Returns
        ``(hints, exclusions, error)`` - ``error`` non-None means a hint /
        regex was invalid and the dialog must not save."""
        # Exclusions.
        exclusions: list[dict] = []
        for r in range(self._excl_table.rowCount()):
            item = self._excl_table.item(r, 0)
            pattern = item.text().strip() if item else ""
            if not pattern:
                continue
            mode = self._excl_table.cellWidget(r, 2).currentText()
            if mode == "regex":
                err = user_rules.validate_regex(pattern)
                if err:
                    return [], [], f"Exclusion regex {pattern!r} is invalid: {err}"
            exclusions.append({
                "pattern": pattern,
                "target": self._excl_table.cellWidget(r, 1).currentText(),
                "match_mode": mode,
            })

        # Hints.
        hints: list[dict] = []
        valid_datatypes = schema.list_datatypes()
        for r in range(self._hint_table.rowCount()):
            pat_item = self._hint_table.item(r, 0)
            patterns = [p.strip() for p in (pat_item.text() if pat_item else "").split(",") if p.strip()]
            if not patterns:
                continue
            # Datatype + suffix come from constrained dropdowns.
            datatype = self._hint_table.cellWidget(r, 1).currentText().strip()
            suffix = self._hint_table.cellWidget(r, 2).currentText().strip()
            task = (self._hint_table.item(r, 3).text().strip() if self._hint_table.item(r, 3) else "")
            mode = self._hint_table.cellWidget(r, 4).currentText()
            force_item = self._hint_table.item(r, 5)
            force = bool(force_item and force_item.checkState() == Qt.CheckState.Checked)

            if not datatype or not suffix:
                return [], [], f"Hint for {patterns!r} needs both a datatype and a suffix."
            if datatype == "derivatives" or datatype not in valid_datatypes:
                return [], [], (
                    f"Hint datatype {datatype!r} is not a valid BIDS datatype. "
                    f"Choose one of: {', '.join(sorted(valid_datatypes))}."
                )
            if suffix not in schema.list_suffixes(datatype):
                return [], [], (
                    f"Suffix {suffix!r} is not valid for datatype {datatype!r}."
                )
            if mode == "regex":
                for p in patterns:
                    err = user_rules.validate_regex(p)
                    if err:
                        return [], [], f"Hint regex {p!r} is invalid: {err}"
            hints.append({
                "patterns": patterns,
                "datatype": datatype,
                "suffix": suffix,
                "task": task,
                "entities": {},
                "match_mode": mode,
                "force": force,
            })
        return hints, exclusions, None

    def _build_convert_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        convert = QGroupBox("Convert defaults")
        form = QFormLayout(convert)

        self._convert_jobs = QSpinBox()
        self._convert_jobs.setRange(1, self._sys.logical_threads)
        self._convert_jobs.setToolTip(
            f"Capped at the detected thread count ({self._sys.logical_threads})."
        )
        form.addRow("Parallel workers (-j):", self._convert_jobs)

        # Policy for a subject that already exists in the dataset (incremental
        # conversion). New sessions/datatypes always merge in; this governs
        # files that collide with existing ones. Replaces the old overwrite
        # checkbox (which mapped only to "replace").
        self._convert_on_existing = QComboBox()
        for value, label in (
            ("skip",    "Skip: keep existing files, add only new (safe default)"),
            ("update",  "Update: replace only files whose content changed"),
            ("replace", "Replace: back up and replace colliding files"),
            ("error",   "Error: abort a subject if any file would be overwritten"),
        ):
            self._convert_on_existing.addItem(label, userData=value)
        self._convert_on_existing.setToolTip(
            "What to do when a subject already exists in the dataset. Adding a "
            "new session or datatype always merges in; this only governs files "
            "that collide with existing ones. Existing data is never lost on the "
            "default (Skip)."
        )
        form.addRow("Existing subjects:", self._convert_on_existing)

        self._convert_skip_residuals = QCheckBox(
            "Skip residual volumes (drop dcm2niix secondary duplicates such "
            "as ..._bolda / _Eq_ / _ROI that are not real images)"
        )
        self._convert_skip_residuals.setToolTip(
            "dcm2niix splits a single input series into the real image plus "
            "derived single-volume duplicates it names ..._bolda, ..._Eq_1, "
            "etc. These have no valid BIDS suffix. Recommended: on."
        )
        form.addRow("Residuals:", self._convert_skip_residuals)

        self._convert_force_edf = QCheckBox(
            "Force EDF for EEG / iEEG (re-encode recordings to EDF on convert)"
        )
        self._convert_force_edf.setToolTip(
            "Re-encode EEG / iEEG recordings to EDF instead of keeping the "
            "source format. Harmonises a study to one BIDS-native format, and "
            "makes a non-BIDS-native but mne-readable source (GDF, EGI, ...) "
            "convertible. MEG / NIRS are unaffected."
        )
        form.addRow("Force EDF:", self._convert_force_edf)

        v.addWidget(convert)

        # Post-convert chain laid out as an indented hierarchy: each step is
        # a parent checkbox; its sub-options sit indented beneath and are
        # enabled only while the parent is on.
        post = QGroupBox("Post-convert chain (run after every conversion)")
        pv = QVBoxLayout(post)
        pv.setSpacing(4)

        self._post_run_metadata = QCheckBox(
            "Generate metadata (dataset_description, participants.tsv, "
            "*_scans.tsv, sidecar audit)"
        )
        pv.addWidget(self._post_run_metadata)
        self._post_metadata_fill_todos = QCheckBox(
            "Insert 'TODO' placeholders for missing recommended fields"
        )
        pv.addWidget(_indented(self._post_metadata_fill_todos))

        self._post_run_validate = QCheckBox(
            "Validate dataset (bidsval schema-driven validation)"
        )
        pv.addWidget(self._post_run_validate)
        self._post_validate_strict = QCheckBox(
            "Deep checks: read NIfTI headers and file contents (slower)"
        )
        self._post_validate_strict.setToolTip(
            "When on, validation reads NIfTI headers and file contents in "
            "addition to the structural checks. More thorough, slower on "
            "large trees. Maps to the validator's read-headers mode."
        )
        self._post_validate_html = QCheckBox(
            "Write a self-contained validation_report.html (--html)"
        )
        pv.addWidget(_indented(self._post_validate_strict))
        pv.addWidget(_indented(self._post_validate_html))

        _bind_children(self._post_run_metadata, self._post_metadata_fill_todos)
        _bind_children(
            self._post_run_validate,
            self._post_validate_strict,
            self._post_validate_html,
        )

        v.addWidget(post)
        v.addStretch(1)
        return w

    def _build_bids_version_tab(self) -> QWidget:
        """Which version of the standard this session works to.

        Its own tab, and the first one, because it is not a validation
        preference. It decides which fields every metadata form asks for, which
        entities a filename may carry, what gets stamped into
        dataset_description.json and what validation reports. It used to sit
        under Validation, which is where it reached when validation was the only
        thing that read it.
        """
        w = QWidget()
        v = QVBoxLayout(w)

        box = QGroupBox("BIDS version")
        form = QFormLayout(box)

        self._validate_schema = QComboBox()
        self._validate_schema.addItem("Newest available (recommended)", userData="")
        for ver in schema.available_versions():
            self._validate_schema.addItem(f"BIDS {ver}", userData=ver)
        self._validate_schema.setToolTip(
            "The version of BIDS this session works to. Several ship with "
            "BIDS Manager.\n\nChoose an older one to work to a dataset that "
            "was built against it, so the forms ask for that version's fields "
            "and validation judges it by that version's rules."
        )
        self._validate_schema.currentIndexChanged.connect(self._describe_bids_version)
        form.addRow("Work to:", self._validate_schema)

        self._version_summary = QLabel()
        self._version_summary.setWordWrap(True)
        self._version_summary.setStyleSheet("color: #8b949e;")
        form.addRow("", self._version_summary)
        v.addWidget(box)

        note = QLabel(
            "Applies to the whole pipeline: what the metadata forms ask for, "
            "which entities a filename may carry, the BIDSVersion written into "
            "dataset_description.json, and what validation reports.\n\n"
            "The command line takes the same choice per run, as --schema."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #8b949e;")
        v.addWidget(note)
        v.addStretch(1)
        return w

    def _describe_bids_version(self) -> None:
        """Say what the highlighted version actually is, in its own terms.

        A version number alone does not tell anyone what changes. The counts do,
        and they come from the schema rather than from a note that would go
        stale.
        """
        chosen = self._validate_schema.currentData() or None
        try:
            namespace = schema.get_schema(chosen)
            self._version_summary.setText(
                f"BIDS {namespace.bids_version}: "
                f"{len(namespace.objects.datatypes)} datatypes, "
                f"{len(namespace.rules.entities)} entities, "
                f"{len(namespace.objects.metadata)} metadata fields."
            )
        except Exception:
            self._version_summary.setText("")

    def _build_validation_tab(self) -> QWidget:
        """Validation-engine (bidsval) knobs.

        The Editor's "Deep checks" toggle (read NIfTI headers / file contents)
        lives in the Editor toolbar, not here, because it is a per-run choice.
        These two are dataset-wide preferences worth persisting.
        """
        w = QWidget()
        v = QVBoxLayout(w)

        box = QGroupBox("Validation engine (bidsval)")
        form = QFormLayout(box)

        self._validate_max_rows = QSpinBox()
        self._validate_max_rows.setRange(1, 10_000_000)
        self._validate_max_rows.setSingleStep(1000)
        self._validate_max_rows.setToolTip(
            "Maximum number of rows scanned per TSV during column / value "
            "validation. Large tables are bounded to keep validation fast; "
            "raise this to scan more of very long tables."
        )
        form.addRow("Max TSV rows scanned:", self._validate_max_rows)

        # Which severities the Editor's Validation pane lists. The tree badges
        # and chips always reflect the full picture; this only filters the
        # findings list so you can focus on errors (or warnings).
        self._validate_show = QComboBox()
        for label, value in (
            ("Errors and warnings", "error_warning"),
            ("Errors only",         "error"),
            ("Warnings only",       "warning"),
        ):
            self._validate_show.addItem(label, userData=value)
        self._validate_show.setToolTip(
            "Filter which findings the Editor's Validation pane lists. The "
            "tree badges and the error / warning counters always show the "
            "full picture; this only narrows the list so you can focus."
        )
        form.addRow("Show findings:", self._validate_show)

        self._validate_flag_todos = QCheckBox(
            "Flag 'TODO' placeholder values as warnings"
        )
        self._validate_flag_todos.setToolTip(
            "A BIDS Manager convention: the metadata engine writes the literal "
            "string 'TODO' into missing recommended fields so you can find and "
            "fill them. On by default. Turn it off for exact parity with the "
            "standalone bidsval engine (which does not know this convention)."
        )
        form.addRow("TODO placeholders:", self._validate_flag_todos)

        v.addWidget(box)

        note = QLabel(
            "The Editor's \"Deep checks\" toggle (read NIfTI headers and file "
            "contents) lives in the Editor toolbar, since it is a per-run "
            "choice. Validation results are written into the project's "
            "<bids_root>/.bidsmgr/ folder, never into the BIDS tree."
        )
        note.setStyleSheet("color: #8b949e;")
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------
    # Widget <-> settings
    # ------------------------------------------------------------------

    @classmethod
    def _header_logo_index(cls, value: str) -> int:
        for i, (_label, key) in enumerate(cls._HEADER_LOGO_PRESETS):
            if key == value:
                return i
        return 0

    @classmethod
    def _closest_font_scale_index(cls, value: float) -> int:
        """Return the preset index whose multiplier is nearest *value*."""
        try:
            return min(
                range(len(cls._FONT_SCALE_PRESETS)),
                key=lambda i: abs(cls._FONT_SCALE_PRESETS[i][1] - value),
            )
        except Exception:
            return 1  # "Normal"

    def _load_into_widgets(self, s: AppSettings) -> None:
        """Push every value from ``s`` into the dialog's widgets.

        Used both on open (with the live settings) and by "Restore
        defaults" (with a fresh ``AppSettings()``), so the two paths can
        never drift. Worker counts are clamped to the detected thread cap.
        """
        cap = self._sys.logical_threads

        self._theme_combo.setCurrentText(s.theme)
        self._font_scale_combo.setCurrentIndex(
            self._closest_font_scale_index(s.font_scale)
        )
        self._header_logo_combo.setCurrentIndex(
            self._header_logo_index(s.header_logo)
        )

        self._scan_jobs.setValue(max(1, min(s.scan_n_jobs, cap)))
        self._scan_probe.setChecked(s.scan_probe_convert)
        self._scan_preview.setChecked(s.scan_converter_preview)
        self._scan_skip_bids_guess.setChecked(s.scan_skip_bids_guess)

        self._convert_jobs.setValue(max(1, min(s.convert_n_jobs, cap)))
        idx = self._convert_on_existing.findData(s.convert_on_existing)
        self._convert_on_existing.setCurrentIndex(idx if idx >= 0 else 0)
        self._convert_skip_residuals.setChecked(s.convert_skip_residuals)
        self._convert_force_edf.setChecked(s.convert_force_edf)

        self._post_run_metadata.setChecked(s.post_run_metadata)
        self._post_metadata_fill_todos.setChecked(s.post_metadata_fill_todos)
        self._post_run_validate.setChecked(s.post_run_validate)
        self._post_validate_strict.setChecked(s.post_validate_strict)
        self._post_validate_html.setChecked(s.post_validate_html)

        # BIDS version, then the validation engine's own knobs.
        sidx = self._validate_schema.findData(s.validate_schema_version)
        self._validate_schema.setCurrentIndex(sidx if sidx >= 0 else 0)
        self._describe_bids_version()
        self._validate_max_rows.setValue(max(1, int(s.validate_max_rows)))
        shidx = self._validate_show.findData(s.validate_show)
        self._validate_show.setCurrentIndex(shidx if shidx >= 0 else 0)
        self._validate_flag_todos.setChecked(s.validate_flag_todos)

        # Scan rules: rebuild both editable tables from the persisted lists
        # (clear first so Restore-defaults empties them).
        self._excl_table.setRowCount(0)
        for e in s.scan_exclusions:
            self._add_exclusion_row(
                pattern=str(e.get("pattern", "")),
                target=str(e.get("target", "sequence")),
                mode=str(e.get("match_mode", "substring")),
            )
        self._hint_table.setRowCount(0)
        for h in s.user_hints:
            pats = h.get("patterns", [])
            if isinstance(pats, str):
                pats = [pats]
            self._add_hint_row(
                patterns=", ".join(str(p) for p in pats),
                datatype=str(h.get("datatype", "")),
                suffix=str(h.get("suffix", "")),
                task=str(h.get("task", "") or ""),
                mode=str(h.get("match_mode", "substring")),
                force=bool(h.get("force", False)),
            )

    def _on_restore_defaults(self) -> None:
        """Reset all widgets to the AppSettings field defaults (not saved
        until the user clicks Save)."""
        self._load_into_widgets(AppSettings())

    def _on_save(self) -> None:
        # Validate the scan rules first so an invalid hint blocks the save
        # without losing the user's other edits.
        hints, exclusions, error = self._read_scan_rules()
        if error:
            QMessageBox.warning(self, "Invalid scan rule", error)
            return

        s = self._settings
        s.user_hints = hints
        s.scan_exclusions = exclusions
        s.theme = self._theme_combo.currentText()
        s.font_scale = self._FONT_SCALE_PRESETS[
            self._font_scale_combo.currentIndex()
        ][1]
        s.header_logo = self._HEADER_LOGO_PRESETS[
            self._header_logo_combo.currentIndex()
        ][1]

        s.scan_n_jobs = self._scan_jobs.value()
        s.scan_probe_convert = self._scan_probe.isChecked()
        s.scan_converter_preview = self._scan_preview.isChecked()
        s.scan_skip_bids_guess = self._scan_skip_bids_guess.isChecked()

        s.convert_n_jobs = self._convert_jobs.value()
        s.convert_on_existing = self._convert_on_existing.currentData() or "skip"
        # Keep the legacy flag in sync for any old reader.
        s.convert_overwrite = (s.convert_on_existing == "replace")
        s.convert_skip_residuals = self._convert_skip_residuals.isChecked()
        s.convert_force_edf = self._convert_force_edf.isChecked()

        s.post_run_metadata = self._post_run_metadata.isChecked()
        s.post_metadata_fill_todos = self._post_metadata_fill_todos.isChecked()
        s.post_run_validate = self._post_run_validate.isChecked()
        s.post_validate_strict = self._post_validate_strict.isChecked()
        s.post_validate_html = self._post_validate_html.isChecked()

        s.validate_schema_version = self._validate_schema.currentData() or ""
        s.validate_max_rows = self._validate_max_rows.value()
        s.validate_show = self._validate_show.currentData() or "error_warning"
        s.validate_flag_todos = self._validate_flag_todos.isChecked()

        s.save()
        # Adopt the chosen version now rather than at the next launch: every
        # schema answer in the process is memoised, so this also drops the
        # answers about the old one.
        schema.set_active_version(s.validate_schema_version)
        self.accept()


__all__ = ["SettingsDialog"]
