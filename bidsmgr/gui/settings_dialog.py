"""The Settings dialog: every knob the GUI uses, one page each.

Reads / writes :class:`bidsmgr.gui.app_settings.AppSettings` via
``QSettings`` and the viewer settings (``SettingsHub``). All changes are
applied on **Save** (no live binding) so the user can experiment with
values and cancel without commit.

The pages are listed down the left (``#settings-nav``), each with its
title and a line saying what it is for; a narrow dialog shows the list as
icons only. Every page scrolls on its own and its form rows wrap, so the
dialog shrinks from the sides without cutting anything off. The Convert
page lays the post-convert chain out as an indented hierarchy (parent step
+ its sub-options), and "Restore defaults" resets every widget to the
:class:`AppSettings` field defaults.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import schema
from ..classifier import sequence_dict
from ..deface import engines as deface_engines
from ..deface import run as deface_run
from ..classifier import user_rules
from ..util.system_info import SystemInfo, get_system_info
from .app_settings import AppSettings
from .typefaces import code


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
    """Settings, one page per subject, listed down the left.

    Theme + post-convert chain live under their natural homes. The
    inspector column visibility is NOT here: it's controlled via the
    table header's right-click menu, and that menu writes through to
    the same QSettings namespace.
    """

    #: Narrower than this (at font scale 1.0), the page list shows icons only.
    COMPACT_BELOW_PX = 640

    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BIDS Manager settings")
        self.setObjectName("settings-dialog")
        self._settings = settings
        # Detected once: the worker-count spinboxes are capped at the host's
        # logical thread count so the user can never ask for more workers
        # than the machine has threads.
        self._sys: SystemInfo = get_system_info()
        from .viz import fonts

        self.resize(fonts.px(860), fonts.px(660))

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, fonts.px(10))
        v.setSpacing(fonts.px(8))
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self._nav = QListWidget()
        self._nav.setObjectName("settings-nav")
        self._nav.setIconSize(QSize(fonts.px(18), fonts.px(18)))
        self._nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._nav.setUniformItemSizes(True)
        self._stack = QStackedWidget()
        self._stack.setObjectName("settings-pages")
        row.addWidget(self._nav)
        row.addWidget(self._stack, 1)
        v.addLayout(row, 1)
        self._page_titles: list[str] = []
        self._nav_compact: Optional[bool] = None

        self._add_page("BIDS version", "settings_bids",
                       "The version of the standard datasets are written and validated "
                       "against.", self._build_bids_version_tab())
        self._add_page("Display", "settings_display",
                       "Theme, text size and how the Editor shows the dataset.",
                       self._build_display_tab())
        self._add_page("System", "settings_system",
                       "What this computer offers the parallel steps.",
                       self._build_system_tab())
        self._add_page("Scan", "scan", "How a scan reads raw data and proposes names.",
                       self._build_scan_tab())
        self._add_page("Scan rules", "settings_rules",
                       "Your own hints for naming series, and series to leave out.",
                       self._build_scan_rules_tab())
        from ..classifier.acquisition.config import ClassifierConfig

        from .classifier_rules import ClassifierSettingsPage

        self._classifier_page = ClassifierSettingsPage(
            ("general", "weights", "anatomy", "gradient_echo", "echoes", "series", "contrast"),
            root=ClassifierConfig,
            hint="An MRI series is named from how it was acquired (echo, repetition and "
                 "inversion times, readout, volumes, diffusion weighting) and from the "
                 "series beside it, with dcm2niix's guess and the series name as second "
                 "opinions. Every name keeps its confidence and its evidence: the "
                 "Properties panel's Why this name. bidsmgr-scan and bidsmgr-classify "
                 "read the same settings from a JSON file (--print-classifier-config "
                 "writes one).")
        self._add_page("Classification", "settings_classify",
                       "How MRI series are named from their acquisition parameters.",
                       self._classifier_page, scroll=False)
        self._add_page("Convert", "settings_convert",
                       "How a conversion writes the dataset, and what runs after it.",
                       self._build_convert_tab())
        self._add_page("Validation", "file_check",
                       "What validation checks and what it shows.",
                       self._build_validation_tab())
        # The viewer's own pages, generated from its settings model. They edit
        # a copy; Save hands it to the hub, which every open viewer follows.
        from .viz.bridge import SettingsHub
        from .viz.settings_pages import QualitySettingsPage, ShortcutsPage, ViewerSettingsPage

        self._viz_settings = SettingsHub.instance().settings.model_copy(deep=True)
        self._qc_page = QualitySettingsPage()
        self._viewer_page = ViewerSettingsPage()
        self._shortcuts_page = ShortcutsPage()
        self._add_page("Quality control", "qc",
                       "When the quality checks run, with which methods, and every "
                       "threshold.", self._qc_page, scroll=False)
        self._add_page("Viewer", "settings_viewer",
                       "How images and signals open and look in every viewer.",
                       self._viewer_page, scroll=False)
        # Its tables' choice boxes are wide: scrolled sideways rather than
        # setting a floor under the whole dialog.
        self._add_page("Viewer shortcuts", "shortcuts",
                       "Keys and mouse gestures in the viewers.", self._shortcuts_page)
        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._nav.setCurrentRow(0)
        # A row too wide for a narrow page puts its field under its label.
        for form in self.findChildren(QFormLayout):
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

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
        self._save_btn = buttons.button(QDialogButtonBox.StandardButton.Save)
        # In the dialogs' footer strip, with room around it (the buttons sat
        # flush against the window's edge).
        from .viz import fonts

        footer = QFrame()
        footer.setObjectName("issue-dialog-footer")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(fonts.px(14), fonts.px(10), fonts.px(14), fonts.px(10))
        fl.addWidget(buttons)
        v.addWidget(footer)

        # Populate every widget from the current settings.
        self._load_into_widgets(self._settings)
        self._classifier_page.load(self._settings.classifier())
        self._viewer_page.load(self._viz_settings)
        self._qc_page.load(self._viz_settings)
        self._shortcuts_page.load(self._viz_settings)
        self._fit_nav()

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    def _add_page(self, title: str, icon: str, description: str, content: QWidget, *,
                  scroll: bool = True) -> None:
        """A page: its title, what it is for, and ``content`` (in a scroll
        area of its own unless it brings one)."""
        from . import icons
        from .viz import fonts

        page = QWidget()
        page.setObjectName("settings-page")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(fonts.px(18), fonts.px(14), fonts.px(14), 0)
        lay.setSpacing(fonts.px(4))
        head = QLabel(title)
        head.setObjectName("settings-page-title")
        lay.addWidget(head)
        if description:
            what = QLabel(description)
            what.setObjectName("dlg-hint")
            what.setWordWrap(True)
            lay.addWidget(what)
            lay.addSpacing(fonts.px(6))
        if scroll:
            area = QScrollArea()
            area.setObjectName("settings-scroll")
            area.setWidgetResizable(True)
            area.setFrameShape(QScrollArea.Shape.NoFrame)
            area.setWidget(content)
            lay.addWidget(area, 1)
        else:
            lay.addWidget(content, 1)
        self._stack.addWidget(page)
        # Every page's icon in the text colour: some of these glyphs are
        # accent-tinted elsewhere (Scan, Validation, QC), and a list where a
        # few icons are blue reads as if those pages were selected.
        item = QListWidgetItem(icons.icon(icon, color=icons.CUR().get("text")), title)
        item.setData(Qt.ItemDataRole.UserRole, title)
        item.setData(Qt.ItemDataRole.UserRole + 1, description)
        item.setToolTip(description)
        self._nav.addItem(item)
        self._page_titles.append(title)

    def page_titles(self) -> list[str]:
        """The pages, in order."""
        return list(self._page_titles)

    def show_page(self, title: str) -> None:
        """Open the page called ``title``."""
        self._nav.setCurrentRow(self._page_titles.index(title))

    def current_page(self) -> str:
        return self._page_titles[self._nav.currentRow()]

    def _fit_nav(self) -> None:
        """The page list as wide as its longest title (icons only when the
        dialog is narrow), measured in the list's own scaled font."""
        from .viz import fonts

        compact = self.width() < fonts.px(self.COMPACT_BELOW_PX)
        if compact == self._nav_compact:
            return
        self._nav_compact = compact
        nav = self._nav
        nav.ensurePolished()
        for i in range(nav.count()):
            item = nav.item(i)
            title = item.data(Qt.ItemDataRole.UserRole)
            item.setText("" if compact else title)
            item.setToolTip(title if compact
                            else str(item.data(Qt.ItemDataRole.UserRole + 1) or ""))
        icon = nav.iconSize().width()
        if compact:
            width = icon + fonts.px(34)
        else:
            metrics = nav.fontMetrics()
            longest = max(metrics.horizontalAdvance(t) for t in self._page_titles)
            width = longest + icon + fonts.px(56)
        nav.setFixedWidth(width)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt signature
        # Save is the default button (the accent, and what Enter presses).
        # Set on show: a dialog settles its default button when it appears.
        super().showEvent(event)
        self._save_btn.setDefault(True)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt signature
        super().resizeEvent(event)
        if self._nav_compact is not None:
            self._fit_nav()

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
        ("App icon (full colour)", "app_icon"),
        ("Monochrome mark",        "mark"),
    ]

    def _build_display_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        from .theme_manager import THEMES
        from .theme_menu import theme_swatch
        from .viz import fonts

        self._theme_combo = QComboBox()
        group = None
        for t in THEMES:
            if group is not None and t.group != group:
                self._theme_combo.insertSeparator(self._theme_combo.count())
            group = t.group
            self._theme_combo.addItem(theme_swatch(t.id, fonts.px(16)), t.label, t.id)
            self._theme_combo.setItemData(self._theme_combo.count() - 1, t.description,
                                          Qt.ItemDataRole.ToolTipRole)
        form.addRow("Theme:", self._theme_combo)

        # The user's touches on any theme: accent, tint, icon colours and
        # the file trees' colours (gui/appearance.py).
        from .appearance_editor import AppearanceEditor

        look = QGroupBox("Appearance")
        look_lay = QVBoxLayout(look)
        self._appearance_editor = AppearanceEditor()
        look_lay.addWidget(self._appearance_editor)
        self._theme_combo.currentIndexChanged.connect(
            lambda _i: self._appearance_editor.set_theme(self._theme_combo.currentData() or "dark"))
        form.addRow(look)

        # Font scale: multiplies every font-size (QSS + delegate paints +
        # inline stylesheets + icon sizes) so the user can comfortably
        # nudge the whole UI up or down. Persisted under ``ui/font_scale``.
        self._font_scale_combo = QComboBox()
        for label, _value in self._FONT_SCALE_PRESETS:
            self._font_scale_combo.addItem(label)
        form.addRow("Font scale:", self._font_scale_combo)

        self._log_clear_on_run = QCheckBox("Start each scan and conversion with an empty Log")
        self._log_clear_on_run.setToolTip(
            "The Converter's Log shows one run at a time. Every run is also kept as a "
            "file: the Log's history icon lists them.")
        form.addRow("Log:", self._log_clear_on_run)

        # Header brand artwork.
        self._header_logo_combo = QComboBox()
        for label, _value in self._HEADER_LOGO_PRESETS:
            self._header_logo_combo.addItem(label)
        form.addRow("Header logo:", self._header_logo_combo)

        # Editor tree: dotfiles and the machinery folders. Off by default,
        # because a dataset carries .bidsmgr/, .git/ and .bidsignore and none
        # of them are the data. On, they are shown dimmed.
        self._editor_show_hidden = QCheckBox("Show hidden files and folders")
        self._editor_show_hidden.setToolTip(
            "Dotfiles and dot-folders (.bidsignore, .bidsmgr, .git) are "
            "hidden by default. Shown, they are dimmed so they do not "
            "compete with the dataset. Needed to open .bidsignore."
        )
        form.addRow("Editor tree:", self._editor_show_hidden)

        # Save as you go. Safe because every editor write goes through the
        # operation log, so an edit made without being asked for can still be
        # undone after the pane has moved on.
        self._editor_autosave = QCheckBox("Save sidecar edits as you go")
        self._editor_autosave.setToolTip(
            "Off by default: edits wait for the Save button, and the toolbar "
            "says there are unsaved changes from the first keystroke either "
            "way.\n\nOn, a field commits when it loses focus or you press "
            "Enter and is written after a short pause, so a burst of typing "
            "is one write. Every write is reversible, so this cannot lose "
            "what was there before."
        )
        form.addRow("Editor saving:", self._editor_autosave)

        # A request for a star on GitHub (gui/star_request.py).
        self._ask_for_star = QCheckBox("Ask for a star on GitHub")
        self._ask_for_star.setToolTip(
            "BIDS Manager collects no usage data, so a star is how its authors know "
            "it is useful. It asks at most once a day, on the first launch of the day, "
            "and stops once you have opened the project's page from it. Home's Help us "
            "button opens the request at any time.")
        form.addRow("GitHub star:", self._ask_for_star)

        hint = QLabel(
            "The header's sun or moon button lists every theme and switches at once. "
            "Everything on this page applies on Save."
        )
        hint.setObjectName("dlg-hint")
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
        note.setObjectName("dlg-hint")
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

        self._scan_probe = QCheckBox("Probe each series with dcm2niix")
        self._scan_probe.setToolTip(
            "Runs dcm2niix on each series during the scan (--probe-convert), so the "
            "proposed names know the real number of files and their extensions."
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

        self._scan_skip_bids_guess = QCheckBox("Skip dcm2niix's BidsGuess classifier")
        self._scan_skip_bids_guess.setToolTip(
            "Propose datatypes and suffixes from the series names alone (the older "
            "pattern-matching layer), without dcm2niix's BidsGuess."
        )
        form.addRow("Classifier:", self._scan_skip_bids_guess)

        # Index widths.
        #
        # A group rather than a row of spin boxes: the first version was six
        # unlabelled numbers after the words "Index width", which says what
        # the control IS and nothing about what it does or why anyone would
        # touch it. The entities come from the schema, so a BIDS version that
        # adds one is offered it without an edit here.
        from ..editor.values import index_entities

        widths_box = QGroupBox("Index width in proposed names")
        widths_outer = QVBoxLayout(widths_box)
        widths_outer.setContentsMargins(12, 10, 12, 10)
        widths_outer.setSpacing(8)

        explain = QLabel(
            "An <b>index</b> entity is one whose value is a number: run, "
            "echo, and the others below. BIDS accepts " + code("run-1") + " and "
            + code("run-01") + " equally, so this is a house style rather "
            "than a correction.<br><br>"
            "Setting a width here makes the inspection table propose that "
            "width from the moment a scan finishes, so you never have to go "
            "and repad the dataset afterwards. <b>As found</b> keeps whatever "
            "the source gives, which is what every earlier version did and "
            "what stays out of your way."
        )
        explain.setWordWrap(True)
        explain.setObjectName("dlg-hint")
        widths_outer.addWidget(explain)

        self._index_widths: dict[str, QSpinBox] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        for i, entity in enumerate(index_entities()):
            try:
                info = schema.entity_key_info(entity)
                display, why = info.display_name, info.description.strip()
            except KeyError:
                display, why = entity, ""

            box = QSpinBox()
            box.setObjectName("ent-input")
            box.setRange(0, 6)
            box.setSpecialValueText("as found")
            box.setSuffix(" digits")
            box.setToolTip(why[:300] if why else f"The {entity} entity.")
            self._index_widths[entity] = box

            name = QLabel(code(f"{entity}-"))
            name.setToolTip(display)
            sample = QLabel("")
            sample.setObjectName("dlg-hint")
            # Live example, because "2 digits" is abstract and
            # "run-1 becomes run-01" is not.
            def _sync(value, entity=entity, sample=sample):
                sample.setText(
                    f"{entity}-7 stays {entity}-7" if not value
                    else f"{entity}-7 becomes {entity}-{str(7).zfill(value)}"
                )
            box.valueChanged.connect(_sync)
            _sync(box.value())

            row, col = divmod(i, 2)
            grid.addWidget(name, row, col * 3)
            grid.addWidget(box, row, col * 3 + 1)
            grid.addWidget(sample, row, col * 3 + 2)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(5, 1)
        widths_outer.addLayout(grid)

        # The two groups about names stand beside "Scan defaults", not
        # inside its form: as a form row they sat in the field column,
        # indented by the width of the longest label.
        v.addWidget(defaults)
        v.addWidget(widths_box)
        v.addWidget(self._build_leave_out_box())
        v.addStretch(1)
        return w

    def _build_leave_out_box(self) -> QGroupBox:
        """Entities left out of the names a scan proposes.

        The readable name is the checkbox's text, so it is what a click
        lands on; the key follows as code for those who think in filenames.
        Offered from the schema (``inventory.leave_out.offered``).
        """
        from ..inventory.leave_out import offered

        box = QGroupBox("Leave out of proposed names")
        outer = QVBoxLayout(box)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        explain = QLabel(
            "Entities you never want in a name, such as " + code("acq-") + ". "
            "Each one is still kept where the standard requires it, and where "
            "it is what tells two recordings apart, so no recording overwrites "
            "another. Ticking " + code("run-") + " keeps runs only where there "
            "really are two.<br><br>Applies from the next scan; a scan table "
            "already open is offered the change, as one Undo."
        )
        explain.setWordWrap(True)
        explain.setObjectName("dlg-hint")
        outer.addWidget(explain)

        self._drop_entities: dict[str, QCheckBox] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        # Two columns, read downwards in filename order: three did not fit
        # the longer names ("Phase-Encoding Direction") beside their keys.
        keys = offered()
        columns = 2
        per_column = -(-len(keys) // columns)
        for i, key in enumerate(keys):
            try:
                info = schema.entity_key_info(key)
                display, why = info.display_name, info.description.strip()
            except KeyError:
                display, why = key, ""
            check = QCheckBox(display)
            check.setToolTip(
                f"{display} ({key}-)\n\n{why[:300]}" if why else f"The {key} entity.")
            tag = QLabel(code(f"{key}-"))
            tag.setObjectName("dlg-hint")
            self._drop_entities[key] = check
            col, row = divmod(i, per_column)
            grid.addWidget(check, row, col * 3)
            grid.addWidget(tag, row, col * 3 + 1)
            if col < columns - 1:
                grid.setColumnMinimumWidth(col * 3 + 2, 14)
        for col in range(columns):
            grid.setColumnStretch(col * 3 + 1, 1)
        outer.addLayout(grid)
        return box

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
        intro.setObjectName("dlg-hint")
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
        force_hint.setObjectName("dlg-hint")
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
        builtin_note.setObjectName("dlg-hint")
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

    def _build_sourcedata_group(self) -> QWidget:
        """Keep the source data in sourcedata/: how, and for which
        modalities (``converter/sourcedata``)."""
        from ..converter import sourcedata as sd
        from .widgets.flow_layout import FlowBar

        group = QGroupBox("Source data")
        form = QFormLayout(group)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._convert_sourcedata = QComboBox()
        for value, label in (("none", "No (default)"),
                             ("copy", "Copy the files as they are"),
                             ("archive", "One archive per series (.tar.gz)")):
            self._convert_sourcedata.addItem(label, value)
        self._convert_sourcedata.setToolTip(
            "Keep each subject's source data in the dataset's sourcedata/ folder, the "
            "place BIDS reserves for data before conversion: "
            "sourcedata/sub-<label>/[ses-<label>/]<modality>/<series>/. A manifest.tsv "
            "per subject lists every file with its checksum.\n\nArchives are fewer "
            "files, compress DICOM well and keep paths short on Windows. The "
            "conversion is refused before it starts when the disk cannot hold them.")
        form.addRow("Keep in sourcedata/:", self._convert_sourcedata)
        self._sourcedata_boxes: dict[str, QCheckBox] = {}
        boxes = FlowBar(h_spacing=12, v_spacing=4)
        for m in sd.MODALITIES:
            box = QCheckBox(sd.LABELS[m])
            box.setToolTip(f"Keep the {sd.LABELS[m]} source data")
            boxes.addWidget(box)
            self._sourcedata_boxes[m] = box
        form.addRow("Modalities:", boxes)
        note = QLabel(sd.IDENTIFIERS_NOTE)
        note.setObjectName("sourcedata-note")
        note.setWordWrap(True)
        form.addRow("", note)

        def follow() -> None:
            on = self._convert_sourcedata.currentData() != "none"
            for box in self._sourcedata_boxes.values():
                box.setEnabled(on)
            note.setEnabled(on)

        self._convert_sourcedata.currentIndexChanged.connect(lambda _i: follow())
        self._sourcedata_follow = follow
        return group

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

        # Somebody curates a sidecar in the Editor, then re-converts that
        # subject. Deciding at file level throws the curation away; deciding at
        # field level keeps what a person stated and still takes what the fresh
        # pass newly knows.
        self._convert_preserve_curation = QCheckBox("Keep curated metadata")
        self._convert_preserve_curation.setToolTip(
            "Merge sidecars field by field instead of overwriting them.\n\n"
            "When a subject you already curated in the Editor is converted "
            "again, merge its JSON sidecars and _scans.tsv field by field: a "
            "value you stated is kept, a TODO placeholder is replaced, and "
            "anything the fresh conversion newly knows is added. Turn it off "
            "to let the fresh conversion win outright. Only has an effect "
            "with Update or Replace above. Recommended: on."
        )
        form.addRow("Curated metadata:", self._convert_preserve_curation)

        self._convert_skip_residuals = QCheckBox("Skip residual volumes")
        self._convert_skip_residuals.setToolTip(
            "Drop the secondary duplicates dcm2niix writes (..._bolda, _Eq_, _ROI), "
            "which are not real images.\n\n"
            "dcm2niix splits a single input series into the real image plus "
            "derived single-volume duplicates it names ..._bolda, ..._Eq_1, "
            "etc. These have no valid BIDS suffix. Recommended: on."
        )
        form.addRow("Residuals:", self._convert_skip_residuals)

        # How a fieldmap is linked to the runs it corrects. Measured with
        # sdcflows (what fMRIPrep uses): URIs are ignored by fMRIPrep 24.1 and
        # earlier, B0FieldIdentifier is read by every version.
        self._convert_intended_for = QComboBox()
        self._convert_intended_for.addItem("BIDS URIs (bids::sub-01/...), the current form",
                                           "uri")
        self._convert_intended_for.addItem("Paths from the subject (ses-1/func/...)",
                                           "relative")
        self._convert_intended_for.setToolTip(
            "How a fieldmap's IntendedFor names the runs it corrects. BIDS URIs are the "
            "current form; fMRIPrep 24.1 and earlier ignore them silently. Paths from the "
            "subject are the older form every version reads.")
        form.addRow("Fieldmap links:", self._convert_intended_for)
        self._convert_b0_field = QCheckBox("Also write B0FieldIdentifier and B0FieldSource")
        self._convert_b0_field.setToolTip(
            "The newer pairing of a fieldmap and its runs, read by every fMRIPrep "
            "version and preferred to IntendedFor. Recommended: on.")
        form.addRow("", self._convert_b0_field)

        self._convert_force_edf = QCheckBox("Re-encode EEG to EDF")
        self._convert_force_edf.setToolTip(
            "Re-encode EEG recordings to EDF instead of keeping the "
            "source format. Harmonises a study to one BIDS-native format, and "
            "makes a non-BIDS-native but mne-readable source (GDF, EGI, ...) "
            "convertible. MEG / NIRS are unaffected."
        )
        form.addRow("Force EDF:", self._convert_force_edf)

        # Defacing, with its engine beside it. Off by default: it is
        # destructive, so it has to be chosen rather than discovered.
        self._convert_deface = QCheckBox(
            "Remove faces from anatomical and PET images"
        )
        self._convert_deface.setToolTip(
            "Blank the face before the subject is committed, so the "
            "identifiable image never enters the dataset at all. Off by "
            "default because it cannot be undone from the conversion: the "
            "original stays in your raw data, not in the BIDS tree. Needs "
            "niimath, which ships with BIDS Manager."
        )
        form.addRow("Deface:", self._convert_deface)

        self._convert_deface_engine = QComboBox()
        self._convert_deface_engine.setObjectName("ent-input")
        for eng in deface_engines.ENGINES:
            self._convert_deface_engine.addItem(eng.label, eng.id)
        self._convert_deface_engine.setToolTip(
            "\n\n".join(f"{e.label}: {e.description}" for e in deface_engines.ENGINES)
        )
        form.addRow("Deface engine:", self._convert_deface_engine)

        reason = deface_run.unavailable_reason()
        if reason:
            self._convert_deface.setEnabled(False)
            self._convert_deface.setChecked(False)
            self._convert_deface_engine.setEnabled(False)
            # Shown, not hidden: a missing row reads as "this tool cannot do
            # that", which is how somebody ships a dataset with faces in it.
            self._convert_deface.setToolTip(reason)
        else:
            self._convert_deface.toggled.connect(
                self._convert_deface_engine.setEnabled
            )
            self._convert_deface_engine.setEnabled(
                self._convert_deface.isChecked()
            )

        v.addWidget(convert)
        v.addWidget(self._build_sourcedata_group())

        # Post-convert chain laid out as an indented hierarchy: each step is
        # a parent checkbox; its sub-options sit indented beneath and are
        # enabled only while the parent is on.
        post = QGroupBox("Post-convert chain (run after every conversion)")
        pv = QVBoxLayout(post)
        pv.setSpacing(4)

        self._post_run_metadata = QCheckBox("Generate metadata")
        self._post_run_metadata.setToolTip(
            "dataset_description.json, participants.tsv, the *_scans.tsv tables, "
            "what your metadata templates and per-file answers state, and an "
            "audit of every sidecar."
        )
        pv.addWidget(self._post_run_metadata)
        self._post_metadata_tables = QCheckBox("Write scans and sessions tables")
        self._post_metadata_tables.setToolTip(
            "A *_scans.tsv per session listing every recording with its "
            "acquisition time, and a *_sessions.tsv per subject with sessions. "
            "Every datatype the standard defines is listed, converted here or "
            "not. Tables already there are kept: their rows, columns and "
            "values; rows are only added, and blanks become n/a.\n\n"
            "The same as the Editor's Tools, Scans and sessions tables."
        )
        pv.addWidget(_indented(self._post_metadata_tables))
        self._post_metadata_fill_todos = QCheckBox("Write TODO into missing fields")
        self._post_metadata_fill_todos.setToolTip(
            "Only placeholders, nothing else. Every field the standard declares "
            "for a file that nothing filled (not the converter, not the scanner "
            "header, not your templates) gets the literal TODO, or n/a where the "
            "field accepts it, so the gap is visible in the file and reported "
            "by validation instead of being an absence nobody notices. No value "
            "is invented and an existing value is never overwritten.\n\n"
            "The TODOs follow this setting on every run: a narrower choice "
            "below takes back the ones a wider one wrote, and turning this off "
            "takes them all back."
        )
        pv.addWidget(_indented(self._post_metadata_fill_todos))

        # How much to mark. Separate from whether, because "mark the required
        # fields" and "mark everything the standard declares" are different
        # amounts of work and different amounts of noise.
        scope_row = QHBoxLayout()
        scope_row.setSpacing(8)
        scope_label = QLabel("In which fields:")
        self._metadata_fill_scope = QComboBox()
        for value, label in (
            ("required", "Required fields only"),
            ("recommended", "Required and recommended fields (default)"),
            ("optional", "Every declared field, including optional"),
        ):
            self._metadata_fill_scope.addItem(label, userData=value)
        self._metadata_fill_scope.setToolTip(
            "The levels are the ones the Editor's sidecar form shows, read for "
            "each file: a T1w without a task is not asked for a TaskName. The "
            "choices nest.\n\n"
            "Some fields never get a TODO, because a placeholder there would "
            "itself be an error: numbers, booleans and controlled vocabularies, "
            "fields that name another file (IntendedFor, AssociatedEmptyRoom) "
            "and fields that link files (B0FieldIdentifier). They stay absent "
            "and are listed in the metadata report.\n\n"
            "Used by the post-convert chain and by the Editor's Fix ups, so "
            "both do the same thing."
        )
        scope_row.addWidget(scope_label)
        scope_row.addWidget(self._metadata_fill_scope, 1)
        scope_holder = QWidget()
        scope_holder.setLayout(scope_row)
        pv.addWidget(_indented(scope_holder, indent=40))
        self._post_metadata_fill_todos.toggled.connect(
            self._metadata_fill_scope.setEnabled
        )

        # Dataset repairs. They run between metadata and validation, and they
        # are the same code the Editor's Fix ups button runs, so a dataset
        # gets the same result whichever moment the user chooses. Both are off
        # by default: one adds files and the other moves fields between them.
        self._post_fixup_companions = QCheckBox("Generate missing companion files")
        self._post_fixup_companions.setToolTip(
            "events.tsv, channels.tsv and JSON sidecars a recording should have.\n\n"
            "What can be read from a recording is read from it, so a "
            "channels table is real content. The rest is a stub carrying "
            "TODO rows.\n\nA generated events table is deliberately INVALID "
            "until you fill it in: TODO is not a valid onset, so validation "
            "reports an error for each one. That is the point. An empty but "
            "valid events table would be indistinguishable from a recording "
            "that genuinely had no events, and would pass quietly forever."
        )
        pv.addWidget(_indented(self._post_fixup_companions))
        self._post_fixup_citation = QCheckBox(
            "Write CITATION.cff from the dataset description"
        )
        self._post_fixup_citation.setToolTip(
            "Writes CITATION.cff from what dataset_description.json already "
            "says. This runs without asking, so here is exactly what it "
            "changes.\n\n"
            "MOVED: Authors. It is taken OUT of dataset_description.json, "
            "because stating authorship in both files is an error "
            "(AUTHORS_AND_CITATION_FILE_MUTUALLY_EXCLUSIVE).\n\n"
            "COPIED and KEPT: License, HowToAcknowledge and "
            "ReferencesAndLinks. They are written into the citation file and "
            "left where they are, so a value you typed does not disappear "
            "from the file you typed it into. The validator would rather "
            "each lived in one place only and says so as a warning "
            "(SINGLE_SOURCE_CITATION_FIELDS). That warning is the cost of "
            "not deleting your answer.\n\n"
            "An existing CITATION.cff is never overwritten."
        )
        # Said on the face of the setting too, not only on hover. This one
        # runs unattended at the end of a conversion, and a fix up that
        # removes a field a user typed cannot announce itself in a tooltip.
        pv.addWidget(_indented(self._post_fixup_citation))
        citation_note = QLabel(
            "Moves <b>Authors</b> out of dataset_description.json (stating it "
            "in both is an error). License, HowToAcknowledge and "
            "ReferencesAndLinks are copied and kept."
        )
        citation_note.setObjectName("dlg-hint")
        citation_note.setWordWrap(True)
        # Indented by margin rather than by ``_indented``, whose trailing
        # stretch would stop a wrapping label from using the width.
        citation_note.setContentsMargins(44, 0, 0, 4)
        pv.addWidget(citation_note)

        # Anonymisation, LAST in the metadata step: run before the
        # participants file and the tables are written, what it removed would
        # be written again.
        from ..anonymise import PRESET_LABELS, PRESETS

        anon_row = QHBoxLayout()
        anon_row.setContentsMargins(0, 0, 0, 0)     # its left edge on its siblings'
        anon_row.setSpacing(8)
        anon_label = QLabel("Anonymise the metadata, last:")
        self._post_anonymise = QComboBox()
        for value, label in PRESET_LABELS.items():
            self._post_anonymise.addItem(label, userData=value)
        tips = ["Off: nothing is anonymised (the default; patient names and IDs are "
                "still never written into sidecars)."]
        for value, policy in PRESETS.items():
            tips.append(f"{PRESET_LABELS[value]}:\n" + "\n".join(
                f"  - {line}" for line in policy.describe()))
        tips.append("Recordings' data are never changed. What a format does not let "
                    "BIDS Manager rewrite (EEGLAB, SNIRF, CTF...) is listed in the log. "
                    "The same as the Editor's Tools, Anonymise metadata.")
        self._post_anonymise.setToolTip("\n\n".join(tips))
        anon_row.addWidget(anon_label)
        anon_row.addWidget(self._post_anonymise, 1)
        anon_holder = QWidget()
        anon_holder.setLayout(anon_row)
        pv.addWidget(_indented(anon_holder))

        self._post_run_validate = QCheckBox("Validate the dataset")
        self._post_run_validate.setToolTip(
            "Schema-driven validation (bidsval), the same as the Editor's Validate "
            "dataset."
        )
        pv.addWidget(self._post_run_validate)
        self._post_validate_strict = QCheckBox("Deep checks (slower)")
        self._post_validate_strict.setToolTip(
            "When on, validation reads NIfTI headers and file contents in "
            "addition to the structural checks. More thorough, slower on "
            "large trees. Maps to the validator's read-headers mode."
        )
        self._post_validate_html = QCheckBox("Write an HTML validation report")
        self._post_validate_html.setToolTip(
            "A self-contained validation_report.html beside the dataset (--html)."
        )
        pv.addWidget(_indented(self._post_validate_strict))
        pv.addWidget(_indented(self._post_validate_html))

        self._post_run_quality = QCheckBox("Check image quality")
        self._post_run_quality.setToolTip(
            "Anatomical and diffusion images, a few seconds each, with the methods "
            "and thresholds of the Quality control page.\n\n"
            "Last: the fast quality check of every anatomical and diffusion image "
            "not checked yet, into derivatives/bidsmgr-qc/ (read it in the Editor, "
            "Tools, Quality check). The air is measured only in images that are not "
            "defaced."
        )
        pv.addWidget(self._post_run_quality)

        _bind_children(
            self._post_run_metadata,
            self._post_metadata_tables,
            self._post_metadata_fill_todos,
            self._post_fixup_companions,
            self._post_fixup_citation,
            anon_holder,
        )
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
        self._version_summary.setObjectName("dlg-hint")
        form.addRow("", self._version_summary)
        v.addWidget(box)

        note = QLabel(
            "Applies to the whole pipeline: what the metadata forms ask for, "
            "which entities a filename may carry, the BIDSVersion written into "
            "dataset_description.json, and what validation reports.\n\n"
            "The command line takes the same choice per run, as --schema."
        )
        note.setWordWrap(True)
        note.setObjectName("dlg-hint")
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
        note.setObjectName("dlg-hint")
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

        self._theme_combo.setCurrentIndex(max(0, self._theme_combo.findData(s.theme)))
        self._appearance_editor.set_theme(s.theme)
        self._appearance_editor.load(s.appearance())
        self._editor_show_hidden.setChecked(s.editor_show_hidden)
        self._editor_autosave.setChecked(s.editor_autosave)
        self._log_clear_on_run.setChecked(s.log_clear_on_run)
        self._ask_for_star.setChecked(s.ask_for_star)
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
        for entity, box in self._index_widths.items():
            box.setValue(int(s.scan_index_widths.get(entity, 0) or 0))
        for key, check in self._drop_entities.items():
            check.setChecked(key in s.scan_drop_entities)

        self._convert_jobs.setValue(max(1, min(s.convert_n_jobs, cap)))
        idx = self._convert_on_existing.findData(s.convert_on_existing)
        self._convert_on_existing.setCurrentIndex(idx if idx >= 0 else 0)
        self._convert_skip_residuals.setChecked(s.convert_skip_residuals)
        self._convert_intended_for.setCurrentIndex(
            max(0, self._convert_intended_for.findData(s.convert_intended_for)))
        self._convert_b0_field.setChecked(s.convert_b0_field)
        self._convert_sourcedata.setCurrentIndex(
            max(0, self._convert_sourcedata.findData(s.convert_sourcedata)))
        chosen = {m.strip() for m in s.convert_sourcedata_modalities.split(",")}
        for m, box in self._sourcedata_boxes.items():
            box.setChecked(m in chosen)
        self._sourcedata_follow()
        self._convert_preserve_curation.setChecked(
            s.convert_preserve_curation
        )
        self._convert_force_edf.setChecked(s.convert_force_edf)
        if self._convert_deface.isEnabled():
            self._convert_deface.setChecked(s.convert_deface)
        idx = self._convert_deface_engine.findData(s.convert_deface_engine)
        if idx >= 0:
            self._convert_deface_engine.setCurrentIndex(idx)
        self._convert_deface_engine.setEnabled(
            self._convert_deface.isChecked()
            and self._convert_deface.isEnabled()
        )

        self._post_run_metadata.setChecked(s.post_run_metadata)
        self._post_metadata_fill_todos.setChecked(s.post_metadata_fill_todos)
        idx = self._metadata_fill_scope.findData(s.metadata_fill_scope)
        self._metadata_fill_scope.setCurrentIndex(idx if idx >= 0 else 1)
        self._metadata_fill_scope.setEnabled(s.post_metadata_fill_todos)
        self._post_fixup_companions.setChecked(s.post_fixup_companions)
        self._post_fixup_citation.setChecked(s.post_fixup_citation)
        self._post_metadata_tables.setChecked(s.post_metadata_tables)
        self._post_anonymise.setCurrentIndex(
            max(0, self._post_anonymise.findData(s.post_anonymise)))
        self._post_run_validate.setChecked(s.post_run_validate)
        self._post_validate_strict.setChecked(s.post_validate_strict)
        self._post_validate_html.setChecked(s.post_validate_html)
        self._post_run_quality.setChecked(s.post_run_quality)

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
        from ..classifier.acquisition.config import ClassifierConfig
        from ..viz.settings import VizSettings

        self._classifier_page.load(ClassifierConfig())

        self._viewer_page.load(VizSettings())
        self._qc_page.load(VizSettings())
        self._shortcuts_page.load(VizSettings())

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
        s.theme = self._theme_combo.currentData() or "dark"
        look = self._appearance_editor.appearance()
        s.accent, s.custom_accent, s.tint, s.icon_style, s.tree_colours = (
            look.accent, dict(look.custom_accent), look.tint, look.icons,
            {k: dict(v) for k, v in look.tree.items()})
        s.editor_show_hidden = self._editor_show_hidden.isChecked()
        s.editor_autosave = self._editor_autosave.isChecked()
        s.log_clear_on_run = self._log_clear_on_run.isChecked()
        s.ask_for_star = self._ask_for_star.isChecked()
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
        # Zero means "as found", so it is absent rather than stored as 0:
        # the scan pass treats an empty map as nothing to do.
        s.scan_index_widths = {
            entity: box.value()
            for entity, box in self._index_widths.items() if box.value()
        }
        s.scan_drop_entities = [
            key for key, check in self._drop_entities.items() if check.isChecked()]

        s.convert_n_jobs = self._convert_jobs.value()
        s.convert_on_existing = self._convert_on_existing.currentData() or "skip"
        # Keep the legacy flag in sync for any old reader.
        s.convert_overwrite = (s.convert_on_existing == "replace")
        s.convert_skip_residuals = self._convert_skip_residuals.isChecked()
        s.convert_intended_for = self._convert_intended_for.currentData() or "uri"
        s.convert_b0_field = self._convert_b0_field.isChecked()
        s.convert_sourcedata = self._convert_sourcedata.currentData() or "none"
        s.convert_sourcedata_modalities = ",".join(
            m for m, box in self._sourcedata_boxes.items() if box.isChecked())
        s.convert_preserve_curation = (
            self._convert_preserve_curation.isChecked()
        )
        s.convert_force_edf = self._convert_force_edf.isChecked()
        s.convert_deface = self._convert_deface.isChecked()
        s.convert_deface_engine = (
            self._convert_deface_engine.currentData()
            or s.convert_deface_engine
        )

        s.post_run_metadata = self._post_run_metadata.isChecked()
        s.post_metadata_fill_todos = self._post_metadata_fill_todos.isChecked()
        s.metadata_fill_scope = (
            self._metadata_fill_scope.currentData() or "recommended"
        )
        s.post_fixup_companions = self._post_fixup_companions.isChecked()
        s.post_fixup_citation = self._post_fixup_citation.isChecked()
        s.post_metadata_tables = self._post_metadata_tables.isChecked()
        s.post_anonymise = self._post_anonymise.currentData() or "off"
        s.post_run_validate = self._post_run_validate.isChecked()
        s.post_validate_strict = self._post_validate_strict.isChecked()
        s.post_validate_html = self._post_validate_html.isChecked()
        s.post_run_quality = self._post_run_quality.isChecked()

        s.validate_schema_version = self._validate_schema.currentData() or ""
        s.validate_max_rows = self._validate_max_rows.value()
        s.validate_show = self._validate_show.currentData() or "error_warning"
        s.validate_flag_todos = self._validate_flag_todos.isChecked()
        classifier = s.classifier()
        self._classifier_page.apply_to(classifier)
        s.classifier_config = classifier.model_dump_json()

        s.save()
        self._save_viewer_settings()
        # Adopt the chosen version now rather than at the next launch: every
        # schema answer in the process is memoised, so this also drops the
        # answers about the old one.
        schema.set_active_version(s.validate_schema_version)
        self.accept()


    def _save_viewer_settings(self) -> None:
        """Hand the edited copy to the hub: stored once, and every open
        viewer (the Editor's, a comparison's, a defacing preview's) follows
        at once."""
        from .viz.bridge import SettingsHub

        new = self._viz_settings.model_copy(deep=True)
        self._viewer_page.apply_to(new)
        self._qc_page.apply_to(new)
        self._shortcuts_page.apply_to(new)
        # Layout memory, sizes and saved views are not edited here: they
        # survive a Save untouched, unless the page was asked to forget the
        # first two.
        from ..viz import memory

        current = SettingsHub.instance().settings
        new.view_presets = dict(current.view_presets)
        # Every memory (layouts, sizes, the 3-D look, trace and spectrum
        # options) as it is NOW: a view changed while this dialog was open
        # must not roll back. Forgotten as one when asked.
        memory.carry_memory(new, current)
        if self._viewer_page.forget_layouts:
            memory.forget(new)
        SettingsHub.instance().replace(new)


__all__ = ["SettingsDialog"]
