"""Properties panel — col 4 of the Converter view.

Schema-driven editor for the selected row's BIDS entities. Reference:
``inspector_proto/proto.py`` lines 669-722.

Layout (top → bottom):

1. **datatype / suffix combos** — required (* in red). Editing either
   re-derives the entity allow-list below.
2. **Entities form** — one row per entity allowed by the schema for
   the current ``(datatype, suffix)``. Required entities marked with
   ``*``; optional entities labelled ``opt``. Each field is a
   ``QLineEdit`` whose ``editingFinished`` writes back through
   :meth:`InventoryTableModel.set_entity`.
3. **Predicted path preview** — token-coloured monospace string built
   from ``schema.build_relative_path``.
4. **Validation messages** — one ``ValMessage`` per
   :class:`schema.ValidationVerdict` from
   :func:`schema.validate_entity_set`.
5. **Why this name?** — small provenance section reading from the
   project's ``ProvenanceMap`` when one exists.

The panel never owns state — it reads from the model on every
``set_selected_row`` call and writes through the model's API. This
keeps the data flow unidirectional (model → panel → model) and means
the panel auto-refreshes when the model emits ``dataChanged``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import pandas as pd

from .. import schema as schema_mod
from ..fixups.blood import is_blood_role
from ..recording_meta import merge_pet as _merge_pet
from ..metadata.template_plan import sidecar_section
from ..project import Project
from ..recording_meta import CURATED_SUGGESTIONS, SCAN_SUGGESTION_COLUMNS
from . import icons
from .delegates import builtin_montages
from .metadata_help import tooltip_for
from .models import InventoryTableModel
from .theme_manager import CUR, scaled_px
from .widgets import BusySpinner, PaneHeader, ValMessage
from .widgets.template_form import (
    CollapsibleSection,
    FieldLabel,
    build_field_widget,
    connect_field_widget,
    field_label_widget,
    level_legend,
    read_field_widget,
    write_field_widget,
)

# ONE label column for the whole panel. The hand-built rows used 76 and the
# schema-driven section 140, so the two halves of the panel did not line up with
# each other. A field name like EEGReference does not fit in 76, so the shared
# column is wider than the old one and long names elide into it.
_LABEL_COL = 120

# Datatypes that carry a recording sidecar: the per-row metadata section and the
# Compute-PSD action appear only for these.
_EEG_MEG_DATATYPES = frozenset({"eeg", "meg", "ieeg", "nirs"})

# Human display names for the datatypes that carry a recording sidecar.
_DATATYPE_NAMES = {
    "eeg": "EEG", "meg": "MEG", "ieeg": "iEEG", "nirs": "NIRS", "pet": "PET",
}


def _answered_field(name: str, value):
    """A stand-in for a field the schema does not declare for this file.

    A converter can write keys BIDS never mentions. They are still shown, so
    nothing appears from nowhere, but there is no level or vocabulary to show.
    """
    from ..metadata.template_plan import TemplateField

    kind = (
        "boolean" if isinstance(value, bool)
        else "number" if isinstance(value, (int, float))
        else "array" if isinstance(value, list)
        else "object" if isinstance(value, dict)
        else "string"
    )
    return TemplateField(name=name, level="optional", type=kind)


def _datatype_label(datatype: str) -> str:
    """Display name for a single datatype (``eeg`` -> ``EEG``)."""
    return _DATATYPE_NAMES.get(datatype, datatype.upper())


log = logging.getLogger(__name__)


# Stable display order for entities (matches BIDS spec). Entities not
# listed here are appended after these, schema-defined order.
_ENTITY_DISPLAY_ORDER: tuple[str, ...] = (
    "subject", "session", "task", "acquisition", "ceagent",
    "reconstruction", "direction", "run", "echo", "part", "chunk",
)


class _EntityRow(QWidget):
    """One ``[label] [QLineEdit]`` row for an entity.

    Self-contained so the form layout can drop in / pull out rows
    without leaking state. Emits no signal — the parent panel reads
    the value back when committing.
    """

    def __init__(
        self,
        entity_name: str,
        value: str,
        *,
        required: bool,
        deprecated: bool = False,
        removable: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.entity_name = entity_name
        # Scoped by object name. A bare "background: transparent" on a container
        # cascades into the tooltips and combo popups its children raise, and
        # they render see-through.
        self.setObjectName("entity-row")
        self.setStyleSheet("#entity-row { background: transparent; }")

        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        # The same eliding, column-width label the metadata form uses, so the
        # entities line up with everything below them and a long name like
        # "reconstruction" ends in an ellipsis rather than being cut mid-word.
        pal = CUR()
        tone = pal["muted"] if deprecated else (
            pal["text"] if required else pal["dim"]
        )
        lbl = FieldLabel(
            entity_name, tone, _LABEL_COL,
            mark=" *" if required else "", mark_colour=pal["error"], fixed=True,
        )
        if deprecated:
            lbl.setStyleSheet(
                "#field-label { color: %s; background: transparent; "
                "text-decoration: line-through; }" % tone
            )
        h.addWidget(lbl)

        self.edit = QLineEdit(value)
        self.edit.setObjectName("ent-input")
        self.edit.setPlaceholderText("—")
        h.addWidget(self.edit, 1)

        # Taking an entity OFF one recording, without going through Bulk
        # edit. Clearing the box does the same thing, but nothing on screen
        # said so: an empty field reads as "not filled in yet" rather than
        # as an instruction, which is why this is a button and not a hint.
        # Offered only where the schema allows the entity to go, and only
        # while there is something to take away.
        # Parented, and shown only once in the row: a widget with no parent
        # made visible is a top-level window for that moment, which took the
        # focus from a floating Inspection or Filter window on every click
        # (macOS then raised the main window over it).
        self.remove = QToolButton(self)
        self.remove.setObjectName("tb-btn")
        self.remove.setText("\u00d7")
        self.remove.setToolTip(
            f"Remove {entity_name} from this recording, so it stops "
            f"appearing in its BIDS name."
        )
        h.addWidget(self.remove, 0)
        self.remove.setVisible(removable and bool(value))

    def value(self) -> str:
        return self.edit.text().strip()



def _preview_job(record, work_dir, sources):
    """Worker side of a row preview. The conversion module is imported here,
    on the worker's thread: on the GUI thread its imports (mne, the
    backends) froze the click that asked for the preview."""
    from ..cli.convert import preview_row

    return preview_row(record, work_dir, sources)


def _row_spectrum(path) -> dict:
    """Worker side: read a source recording and take its spectrum."""
    from ..viz.compute.spectral import psd
    from ..viz.data.signal import SignalSource, read_recording

    src = SignalSource(path=Path(path), raw=read_recording(path, preload=True))
    return psd(src)


class PropertiesPanel(QWidget):
    """Right pane of the Converter view.

    Bind a model with :meth:`bind_model`, then call
    :meth:`set_selected_row` whenever the table's selection changes.
    Passing ``row=None`` blanks the form.
    """

    #: Give these rows' people new subject numbers (the Converter opens the
    #: dialog): asked for from the existing-subject warning.
    renumber_requested = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        # Low floor so the user can squeeze the Properties pane down and give
        # the inspection table more room (the splitter honours this).
        self.setMinimumWidth(56)
        self._model: Optional[InventoryTableModel] = None
        self._project: Optional[Project] = None
        self._row: Optional[int] = None
        self._suppress_writeback = False
        self._entity_rows: list[_EntityRow] = []
        # Where the scan's source data is now, and how to find each row's
        # files in it (``inventory.sources.ScanSources``): the PSD and the
        # preview read from it. Set by the ConverterPanel for the scan shown.
        self._sources = None
        self._shims: dict = {}
        # Background PSD computation. Only one runs at a time, on the viewer
        # library's job runner (a QThread: it ends in an FFT), and
        # ``_psd_row_id`` identifies which recording is computing so the
        # button renders its busy state even across a re-render.
        self._jobs = None
        self._psd_row_id: Optional[str] = None
        self._psd_generation = 0
        # The preview of what a row converts to: one at a time, on its own job.
        self._preview_row_id: Optional[str] = None
        self._preview_generation = 0
        self._preview_dir: Optional[Path] = None
        # Whether the schema-driven sidecar section is open. Folded by default:
        # it offers everything the file may carry, which is the right answer to
        # "what can I state about this recording" and the wrong thing to greet
        # someone with in a narrow pane. Remembered for the session so a user
        # working through a dataset opens it once, not once per row.
        self._sidecar_expanded = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(PaneHeader("Properties · no row selected"))
        self._header = outer.itemAt(0).widget()

        # Body lives inside a scroll area so the form can grow without
        # bumping the splitter handles. The scroll area's contents are
        # ``self._body`` which we rebuild on every set_selected_row.
        # A fixed holder in the scroll area, the body inside it: a new body
        # replaces the old within the holder, so neither is ever re-parented
        # (taking a widget out of a scroll area re-parents its whole tree).
        self._holder = QWidget()
        self._holder.setObjectName("props-panel")
        self._holder_layout = QVBoxLayout(self._holder)
        self._holder_layout.setContentsMargins(0, 0, 0, 0)
        self._holder_layout.setSpacing(0)
        self._body = QWidget()
        self._body.setObjectName("props-panel")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(14, 12, 14, 12)
        self._body_layout.setSpacing(7)
        self._holder_layout.addWidget(self._body)

        scroll = QScrollArea()
        self._scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._holder)
        # A scroll area otherwise reports its contents' minimum as its own, so
        # the pane could not be squeezed past whatever the widest card wanted.
        scroll.setMinimumWidth(0)
        scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        outer.addWidget(scroll, 1)

        # A selection renders once it settles; edits to the shown row render
        # once per burst.
        self._pending = False
        self._rendered_row: Optional[int] = None
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(40)
        self._render_timer.timeout.connect(self._on_render_timer)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(0)
        self._refresh_timer.timeout.connect(self._on_refresh_timer)
        self._links_cache: Optional[tuple] = None

        # Build the initial empty body (just a hint).
        self._render_empty()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bind_model(self, model: Optional[InventoryTableModel]) -> None:
        """Attach (or detach with ``None``) the inventory model.

        Connects ``dataChanged`` so external edits to the active row
        (e.g. the table's mirror cells) reflow into the form.
        """
        if self._model is not None:
            try:
                self._model.dataChanged.disconnect(self._on_model_data_changed)
            except (TypeError, RuntimeError):
                pass
        self._model = model
        if model is not None:
            model.dataChanged.connect(self._on_model_data_changed)
        self.set_selected_row(None)

    def set_project(self, project: Optional[Project]) -> None:
        """Attach a project for provenance lookups. Not used yet for writes."""
        self._project = project

    def set_shims(self, shims: Optional[dict]) -> None:
        """The shim settings of the shown scan's series (series uid ->
        values, from the classifier's features): the fieldmap links preview
        matches them the way the conversion matches the sidecars'."""
        self._shims = dict(shims or {})

    def set_sources(self, sources) -> None:
        """Where the shown scan's data is (``inventory.sources.ScanSources``,
        ``None`` for nothing): the PSD and preview buttons find each row's
        recording or series in it. Re-renders the current row, so a button
        follows whether its data can be found."""
        self._sources = sources
        if self._row is not None:
            self.set_selected_row(self._row)

    def repaint_for_palette(self, _pal: dict) -> None:
        """Rebuild the body so inline palette reads pick up new colors.

        The form is constructed from ``CUR()`` reads at render time
        (label colors, predicted-path tokens, validation borders), so a
        full re-render is the cleanest way to refresh after a theme
        swap. Re-renders the currently-selected row (or the empty hint
        if nothing is selected).
        """
        self.set_selected_row(self._row)

    def request_row(self, row: Optional[int]) -> None:
        """Show ``row`` once the selection settles: arrow keys held down
        through a table render the row they stop on, not every row passed.
        The header follows at once."""
        self._row = row
        self._pending = True
        if row is None:
            self._set_header("Properties · no row selected")
        else:
            self._set_header(f"Properties · row {row + 1} selected")
        self._render_timer.start()

    def flush(self) -> None:
        """Render a pending selection now (tests, and before a grab)."""
        if self._pending:
            self._render_timer.stop()
            self._pending = False
            self.set_selected_row(self._row)

    def _on_render_timer(self) -> None:
        if self._pending:
            self._pending = False
            self.set_selected_row(self._row)

    def set_selected_row(self, row: Optional[int]) -> None:
        """Render the form for ``row`` (or blank if ``None``).

        Called by the Converter view whenever the table selection
        changes. Cheap enough to call on every selection event — the
        body is rebuilt from scratch each time so we don't have to
        track per-row deltas.
        """
        same_row = row is not None and row == self._rendered_row
        keep = self._scroll.verticalScrollBar().value() if same_row else 0
        self._row = row
        self._pending = False
        self._render_timer.stop()
        self._rendered_row = row
        if self._model is None or row is None:
            self._render_empty()
            self._set_header("Properties · no row selected")
            return
        if not (0 <= row < self._model.rowCount()):
            self._render_empty()
            return
        self._set_header(f"Properties · row {row + 1} selected")
        self._render_for_row(row)
        if keep:
            # The same row again (an edit to it): stay where the user was.
            QTimer.singleShot(0, lambda v=keep: self._scroll.verticalScrollBar().setValue(v))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _clear_body(self) -> None:
        """A fresh, empty body in place of the old one. Swapping the whole
        container costs one deletion; taking its widgets out one by one
        re-parented each of them (a third of a render). The old body is
        deleted later: the edit that triggered this render may be a signal
        from one of its own controls."""
        body = QWidget()
        body.setObjectName("props-panel")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(7)
        old = self._body
        old.hide()
        self._holder_layout.removeWidget(old)
        old.deleteLater()
        self._body, self._body_layout = body, layout
        self._holder_layout.addWidget(body)
        self._entity_rows = []

    def _render_empty(self) -> None:
        self._clear_body()
        hint = QLabel(
            "Select a row in the inspection table to edit its entities."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {CUR()['dim']}; padding: 24px 0;")
        hint.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._body_layout.addWidget(hint)
        self._body_layout.addStretch(1)

    def _render_for_row(self, row: int) -> None:
        assert self._model is not None
        self._clear_body()

        # 1. datatype + suffix combos
        datatype, suffix = self._model.datatype_suffix(row)
        self._body_layout.addWidget(self._build_combo_row(
            "datatype", datatype, options=sorted(schema_mod.list_datatypes()),
            required=True, slot=self._on_datatype_changed,
        ))
        suffix_options = sorted(schema_mod.list_suffixes(datatype)) if datatype else []
        self._body_layout.addWidget(self._build_combo_row(
            "suffix", suffix, options=suffix_options,
            required=True, slot=self._on_suffix_changed,
        ))

        self._body_layout.addSpacing(4)
        self._body_layout.addWidget(self._divider())
        self._body_layout.addSpacing(2)

        # 2. Entities form
        ent_title = QLabel("Entities")
        ent_title.setStyleSheet(f"color: {CUR()['text']}; font-weight: 600;")
        sub = QLabel("  (schema-driven)")
        sub.setStyleSheet(f"color: {CUR()['dim']}; font-size: {scaled_px(10)}px;")
        head = QHBoxLayout()
        head.setSpacing(0)
        head.addWidget(ent_title)
        head.addWidget(sub)
        head.addStretch(1)
        head_wrap = QWidget()
        head_wrap.setLayout(head)
        head_wrap.setObjectName("entities-head")
        head_wrap.setStyleSheet("#entities-head { background: transparent; }")
        self._body_layout.addWidget(head_wrap)
        self._body_layout.addSpacing(2)

        entities = self._model.entities(row)
        ordered = self._ordered_entities(datatype, suffix)
        required_set = set(schema_mod.required_entities(datatype, suffix)) if datatype and suffix else set()
        deprecated_set = set(schema_mod.deprecated_entities(datatype, suffix)) if datatype and suffix else set()

        for entity in ordered:
            value = entities.get(entity, "")
            er = _EntityRow(
                entity,
                value,
                required=entity in required_set,
                deprecated=entity in deprecated_set,
                removable=self._model.entity_removable_on(row, entity),
            )
            er.edit.editingFinished.connect(
                lambda e=er: self._on_entity_committed(e.entity_name, e.value())
            )
            er.remove.clicked.connect(
                lambda _checked=False, e=er: self._on_entity_removed(e)
            )
            self._entity_rows.append(er)
            self._body_layout.addWidget(er)

        self._body_layout.addSpacing(6)
        self._body_layout.addWidget(self._divider())

        # 3. Predicted path preview
        sec = QLabel("PREDICTED PATH")
        sec.setStyleSheet(f"color: {CUR()['dim']}; font-size: {scaled_px(10)}px; font-weight: 600;")
        self._body_layout.addWidget(sec)
        self._body_layout.addWidget(self._build_path_preview(row, datatype, suffix, entities))
        if self._cell(row, "series_uid").strip() or self._cell(row, "source_file").strip():
            self._body_layout.addWidget(self._build_preview_row(row, datatype, suffix))
        if datatype == "fmap":
            section = self._build_fieldmap_links(row)
            if section is not None:
                self._body_layout.addSpacing(6)
                self._body_layout.addWidget(section)
        why = self._build_why_this_name(row)
        if why is not None:
            self._body_layout.addSpacing(6)
            self._body_layout.addWidget(why)

        # 4. Row-state notice (from the scanner's issues) +
        # schema validation. Two distinct sources of "what's wrong with
        # this row": scanner-detected operational issues vs. schema's
        # entity-set verdicts. Both render with the same ValMessage
        # widget so the user sees them as one ranked list.
        self._body_layout.addSpacing(8)
        for vmsg in self._build_row_issue_messages(row):
            self._body_layout.addWidget(vmsg)
        renumber = self._build_renumber_offer(row)
        if renumber is not None:
            self._body_layout.addWidget(renumber)
        for vmsg in self._build_validation_messages(datatype, suffix, entities):
            self._body_layout.addWidget(vmsg)

        # 5. Per-row metadata, split into two clearly-separated regions:
        # what applies to EVERY datatype (participant demographics and
        # companion files, written for MRI as much as for EEG) and what
        # belongs to THIS datatype (the recording sidecar, EEG / MEG / iEEG /
        # NIRS / PET). Sections within each region are labelled by their BIDS
        # destination file.
        self._append_metadata_title()
        self._append_region_label("Every datatype", agnostic=True)
        self._append_participant_section(row)
        self._append_companion_section(row)
        if datatype == "pet":
            self._append_region_label(
                f"{_datatype_label(datatype)} only", agnostic=False)
            self._append_pet_dose_section(row)
            self._append_blood_section(row)
        if self._cell(row, "modality").strip() == "eyetrack":
            self._append_region_label("Eye tracking only", agnostic=False)
            self._append_eyetrack_section(row)
        elif datatype in _EEG_MEG_DATATYPES:
            self._append_region_label(
                f"{_datatype_label(datatype)} only", agnostic=False)
            self._append_recording_section(row, datatype)

        # 6. Everything else the standard lets this file carry, asked exactly as
        # the dataset dialog asks it, but answered for this recording alone.
        self._append_sidecar_section(row, datatype, suffix)

        self._body_layout.addStretch(1)

    def _pet_eff(self, row: int, field: str) -> str:
        """Effective per-row PET value (scaffold override else dataset default)."""
        if self._model is None:
            return ""
        return self._model.pet_effective(row, field)

    def _on_pet_field_changed(self, key: str, value: str) -> None:
        """Commit a per-row PET override into the scaffold spec."""
        if self._suppress_writeback or self._model is None or self._row is None:
            return
        self._model.set_pet_override(self._row, key, value)

    def _build_combo_row(self, label_text: str, value: str, *, options: list[str],
                         required: bool, slot) -> QWidget:
        row = QWidget()
        # Scoped so the transparent bg does not cascade into the combo popup.
        row.setObjectName("meta-row")
        row.setStyleSheet("#meta-row { background: transparent; }")
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        # The same column every other row in the panel uses, so datatype and
        # suffix line up with the entities and the metadata below them instead
        # of starting their own column.
        pal = CUR()
        lbl = FieldLabel(
            label_text, pal["text"], _LABEL_COL,
            mark=" *" if required else "", mark_colour=pal["error"], fixed=True,
        )
        h.addWidget(lbl)

        combo = QComboBox()
        combo.setObjectName("ent-input")
        combo.setMinimumHeight(22)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        if options:
            combo.addItems(options)
        if value and value not in options:
            combo.addItem(value)
        if value:
            combo.setCurrentText(value)
        # Use ``activated`` (not ``currentTextChanged``) so we only fire on
        # user interaction, not on programmatic ``setCurrentText`` during
        # render.
        combo.activated.connect(lambda _i, c=combo: slot(c.currentText()))
        h.addWidget(combo, 1)
        return row

    # Source recording extensions, longest-first so ``.fif.gz`` wins over ``.fif``.
    _RECORDING_EXTS: tuple[str, ...] = (
        ".fif.gz", ".vhdr", ".edf", ".bdf", ".gdf", ".set", ".cnt", ".fif",
        ".con", ".sqd", ".ds", ".mff", ".snirf", ".nwb", ".mef",
    )
    # Fallback BIDS extension per non-MRI datatype when the source path is blank.
    _DATATYPE_DEFAULT_EXT: dict[str, str] = {
        "meg": ".fif", "eeg": ".edf", "ieeg": ".edf", "nirs": ".snirf",
    }

    def _preview_extension(self, row: int, datatype: str, suffix: str) -> str:
        """Pick the extension shown in the predicted-path preview.

        MRI datatypes render ``.nii.gz``; physio renders ``.tsv.gz``; EEG / MEG /
        iEEG / NIRS render the source recording's own extension (what mne-bids
        keeps by default) with a per-datatype fallback. Fixes the preview always
        showing ``.nii.gz`` for electrophysiology rows.
        """
        if suffix == "physio":
            return ".tsv.gz"
        if datatype in ("eeg", "ieeg", "meg", "nirs"):
            src = self._cell(row, "source_file").lower()
            for ext in self._RECORDING_EXTS:
                if src.endswith(ext):
                    return ext
            return self._DATATYPE_DEFAULT_EXT.get(datatype, "")
        return ".nii.gz"

    def _build_path_preview(
        self, row: int, datatype: str, suffix: str, entities: dict[str, str],
    ) -> QWidget:
        f = QFrame()
        f.setObjectName("path-preview")
        lay = QVBoxLayout(f)
        lay.setContentsMargins(11, 9, 11, 9)
        lay.setSpacing(0)
        pal = CUR()

        # Compose the same token list the prototype rendered, derived
        # from the schema instead of hard-coded data.
        pieces: list[str] = []
        if datatype and suffix and entities.get("subject"):
            try:
                rel = str(schema_mod.build_relative_path(
                    entities, datatype, suffix,
                    self._preview_extension(row, datatype, suffix),
                ))
                # Insert a newline between the directory part and the
                # basename so the preview wraps clearly.
                head, _slash, base = rel.rpartition("/")
                pieces.append(self._color_path_segment(head + "/", pal["accent"]))
                pieces.append("<br>")
                # Basename split: tokens are ``key-value`` pairs joined
                # by underscores, ending in the suffix.
                tokens = base.split("_")
                last_index = len(tokens) - 1
                for i, tok in enumerate(tokens):
                    if i == last_index:
                        # split off extension
                        if "." in tok:
                            suf, _, ext = tok.partition(".")
                            pieces.append(f'<span style="color:{pal["teal"]}">{suf}</span>')
                            pieces.append(f'<span style="color:{pal["dim"]}">.{ext}</span>')
                        else:
                            pieces.append(f'<span style="color:{pal["teal"]}">{tok}</span>')
                    else:
                        if "-" in tok:
                            pieces.append(f'<span style="color:{pal["purple"]}">{tok}</span>')
                        else:
                            pieces.append(f'<span style="color:{pal["accent"]}">{tok}</span>')
                    if i < last_index:
                        pieces.append("_")
            except (ValueError, KeyError, TypeError) as exc:
                pieces.append(f'<span style="color:{pal["error"]}">cannot build path: {exc}</span>')
        else:
            pieces.append(
                f'<span style="color:{pal["muted"]}">'
                'Pick datatype + suffix to preview the path.'
                '</span>'
            )

        lbl = QLabel("".join(pieces))
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setWordWrap(True)
        lbl.setObjectName("path-preview")
        lbl.setStyleSheet(
            f'#path-preview {{ font-size: {scaled_px(11)}px; color: {pal["text"]}; '
            'background: transparent; }'
        )
        lay.addWidget(lbl)
        return f

    @staticmethod
    def _color_path_segment(text: str, color: str) -> str:
        return f'<span style="color:{color}">{text}</span>'

    def _build_row_issue_messages(self, row: int) -> list[QWidget]:
        """One ``ValMessage`` per scanner-detected issue on the selected row.

        Severity is derived from the model's ``row_state``: ``err`` →
        red badge, ``warn`` → amber, ``skip`` → muted (we surface it
        as ``warn`` here so the user sees the explanation). Returns an
        empty list when the row has no issues.
        """
        if self._model is None:
            return []
        issues = self._model.row_issues(row)
        if not issues:
            return []
        state = self._model.row_state(row)
        sev = {"err": "err", "warn": "warn", "skip": "warn"}.get(state, "warn")

        msgs: list[QWidget] = []
        for i, text in enumerate(issues):
            # First entry carries the rule_id "SCANNER · <row-state>";
            # follow-up entries are continuations of the same row so
            # the label is just blank to keep the column quiet.
            rule = f"SCANNER · {state}" if i == 0 else ""
            msgs.append(ValMessage(sev, rule, text, None))
        return msgs

    def _build_renumber_offer(self, row: int) -> Optional[QWidget]:
        """Under the "this subject already exists" warning: the fix, one
        click away."""
        from ..inventory.dataset_identity import EXISTING_SUBJECT_TOKEN
        from ..inventory.renumber import rows_of_person

        if EXISTING_SUBJECT_TOKEN not in self._cell(row, "issues"):
            return None
        holder = QWidget()
        holder.setObjectName("meta-row")
        holder.setStyleSheet("#meta-row { background: transparent; }")
        h = QHBoxLayout(holder)
        h.setContentsMargins(0, 2, 0, 2)
        btn = QPushButton("Give this person a new subject number...")
        btn.setObjectName("tb-btn")
        btn.setToolTip("Number this person (all of their series) with the next free "
                       "subject number, so they are not merged into the existing subject.")
        df = self._model.dataframe()
        btn.clicked.connect(lambda _c=False: self.renumber_requested.emit(
            rows_of_person(df, row)))
        self.renumber_button = btn
        h.addWidget(btn)
        h.addStretch(1)
        return holder

    def _build_validation_messages(
        self, datatype: str, suffix: str, entities: dict[str, str],
    ) -> list[QWidget]:
        out: list[QWidget] = []
        if not datatype or not suffix:
            out.append(ValMessage(
                "warn", "SCHEMA",
                "Pick a datatype and suffix to validate the entity set.",
                None,
            ))
            return out

        verdicts = schema_mod.validate_entity_set(entities, datatype, suffix)
        if not verdicts:
            out.append(ValMessage(
                "ok", f"SCHEMA · {datatype}/{suffix}",
                "Entity set is valid.",
                None,
            ))
            return out
        for v in verdicts:
            sev = {
                schema_mod.Severity.ERROR: "err",
                schema_mod.Severity.WARNING: "warn",
                schema_mod.Severity.INFO: "ok",
            }.get(v.severity, "warn")
            out.append(ValMessage(sev, v.rule_id, v.message, None))
        return out

    @staticmethod
    def _ordered_entities(datatype: str, suffix: str) -> list[str]:
        """Display order: BIDS-spec order, fallback to schema-allow order."""
        if not datatype or not suffix:
            return list(_ENTITY_DISPLAY_ORDER)
        allowed = schema_mod.allowed_entities(datatype, suffix)
        # Put canonical order first (those that are in allowed), then any
        # remaining allowed entities in schema order.
        head = [e for e in _ENTITY_DISPLAY_ORDER if e in allowed]
        tail = [e for e in allowed if e not in head]
        return head + tail

    def _set_header(self, text: str) -> None:
        # The PaneHeader uppercases its constructor arg; we replace its
        # text directly to avoid restoring case rules.
        self._header.setText(text.upper())

    @staticmethod
    def _divider() -> QFrame:
        d = QFrame()
        d.setStyleSheet(
            f"background: {CUR()['subtle']}; max-height: 1px; "
            f"min-height: 1px; border: none;"
        )
        return d

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_entity_committed(self, entity: str, value: str) -> None:
        if self._suppress_writeback:
            return
        if self._model is None or self._row is None:
            return
        # Translate the schema entity name (``subject``) — already the
        # canonical form used by the schema engine and ProjectState.
        self._model.set_entity(self._row, entity, value or None)


    def _on_entity_removed(self, row_widget) -> None:
        """Take one entity off THIS recording.

        The schema decides whether it may go, and the button is only shown
        where it may, so this is the confirmation rather than the check. The
        row is rebuilt afterwards so the field empties and the button goes
        with it.
        """
        if self._model is None or self._row is None:
            return
        entity = row_widget.entity_name
        if not self._model.entity_removable_on(self._row, entity):
            return
        if self._model.set_entity(self._row, entity, ""):
            self.set_selected_row(self._row)

    def _on_datatype_changed(self, new_value: str) -> None:
        if self._model is None or self._row is None:
            return
        _dt, suffix = self._model.datatype_suffix(self._row)
        self._model.set_datatype_suffix(self._row, new_value, suffix)
        # Re-render: the suffix combo's options change with datatype.
        self.set_selected_row(self._row)

    def _on_suffix_changed(self, new_value: str) -> None:
        if self._model is None or self._row is None:
            return
        datatype, _sf = self._model.datatype_suffix(self._row)
        self._model.set_datatype_suffix(self._row, datatype, new_value)
        self.set_selected_row(self._row)

    # ------------------------------------------------------------------
    # Per-row metadata, grouped by destination file
    # ------------------------------------------------------------------

    def _region_label(self, text: str, *, agnostic: bool) -> QWidget:
        """A bold, colour-coded region divider separating the two metadata
        regions (every datatype vs this datatype only)."""
        pal = CUR()
        color = pal["teal"] if agnostic else pal["purple"]
        lbl = QLabel(
            f'<span style="color:{color};font-weight:800;'
            f'letter-spacing:0.6px;">{text.upper()}</span>'
        )
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setStyleSheet(
            f"font-size: {scaled_px(10)}px; background: transparent; "
            f"border-bottom: 1px solid {color}; padding-bottom: 2px;"
        )
        return lbl

    def _append_region_label(self, text: str, *, agnostic: bool) -> None:
        self._body_layout.addSpacing(10)
        self._body_layout.addWidget(self._region_label(text, agnostic=agnostic))

    def _append_metadata_title(self) -> None:
        """Heading that frames the per-row metadata block as the minimal,
        BIDS-recommended set (everything below is optional and inherits the
        dataset defaults)."""
        pal = CUR()
        self._body_layout.addSpacing(12)
        self._body_layout.addWidget(self._divider())
        title = QLabel("MINIMAL METADATA")
        title.setStyleSheet(
            f"color: {pal['text']}; font-weight: 800; letter-spacing: 0.6px; "
            f"font-size: {scaled_px(11)}px; background: transparent;"
        )
        self._body_layout.addWidget(title)
        sub = QLabel(
            "Essential BIDS fields for this recording. All optional; blank "
            "fields inherit the dataset defaults set in Dataset metadata."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(
            f"color: {pal['dim']}; font-size: {scaled_px(9)}px; background: transparent;"
        )
        self._body_layout.addWidget(sub)

    def _section_header(
        self, title: str, destination: str, *, agnostic: bool, tag: str,
    ) -> QWidget:
        """A colour-coded section title plus a dim ``<tag> -> <destination>`` note.

        ``agnostic`` colours the title; ``tag`` states which modalities the
        section applies to (``any datatype`` for the agnostic ones, or the
        recording's own datatype such as ``EEG``) so the destination is
        unambiguous.
        """
        pal = CUR()
        color = pal["teal"] if agnostic else pal["purple"]
        lbl = QLabel(
            f'<span style="color:{color};font-weight:700;">{title}</span>'
            f'<span style="color:{pal["dim"]};"> &middot; {tag} &rarr; {destination}</span>'
        )
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setObjectName("section-head")
        lbl.setStyleSheet(
            f"#section-head {{ font-size: {scaled_px(10)}px; background: transparent; }}"
        )
        # Wrap rather than set a floor: these carry a title plus a destination
        # path, and a plain QLabel reports all of it as its minimum width.
        lbl.setWordWrap(True)
        lbl.setMinimumWidth(0)
        lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        return lbl

    def _append_participant_section(self, row: int) -> None:
        """Demographics for ANY row (incl. MRI) -> participants.tsv.

        Modality-agnostic: every subject has a participants.tsv row. Handedness
        is user-entered (never auto-assumed).
        """
        self._body_layout.addSpacing(8)
        self._body_layout.addWidget(self._divider())
        self._body_layout.addWidget(self._section_header(
            "PARTICIPANT", "participants.tsv", agnostic=True, tag="any datatype"))
        self._body_layout.addWidget(self._meta_combo_row(
            "sex", "PatientSex", ["", "M", "F", "O"], self._cell(row, "PatientSex"), "",
        ))
        self._body_layout.addWidget(self._meta_edit_row(
            "age", "PatientAge", self._cell(row, "PatientAge"),
        ))
        self._body_layout.addWidget(self._meta_combo_row(
            "hand", "Handedness", ["", "R", "L", "A"], self._cell(row, "Handedness"), "",
        ))

    def _append_recording_section(self, row: int, datatype: str) -> None:
        """The one instruction the conversion needs that BIDS has no field for.

        A montage names an MNE electrode layout to APPLY during conversion, so
        that electrodes.tsv and coordsystem.json get written. It is not metadata
        about the recording, it is an instruction to the converter, which is why
        the standard has no field for it and why it is the single thing in this
        panel that does not come from the schema.

        Line frequency, reference and ground USED to be here as hand-built rows.
        They are BIDS fields, so they are now asked for by the schema-driven
        section below like everything else, and answering them there writes the
        same inventory column this used to write.
        """
        show_montage = datatype in ("eeg", "ieeg")

        self._body_layout.addSpacing(8)
        self._body_layout.addWidget(self._divider())
        self._body_layout.addWidget(self._section_header(
            "CONVERSION", "electrodes.tsv + coordsystem.json",
            agnostic=False, tag=_datatype_label(datatype)))

        if show_montage:
            self._body_layout.addWidget(self._meta_combo_row(
                "montage", "montage",
                ["(none)"], self._eff(row, "montage"), "(none)",
                fill_on_open=builtin_montages,
            ))
            suggestion = self._cell(row, "montage_suggestion")
            if suggestion:
                self._body_layout.addWidget(self._montage_hint(suggestion))

    # ------------------------------------------------------------------
    # Eye tracking: the run it belongs to, the screen, the timing
    # ------------------------------------------------------------------

    def _runs_of(self, row: int) -> list[int]:
        """The runs an eye tracker can sit beside (a recording with a task,
        not physio): this participant's first, then everyone's (the scan
        rarely knows whose eye-tracking file it is), each in acquisition
        order."""
        df = self._model.dataframe()
        me = self._cell(row, "participant_id")
        out = []
        for i in range(len(df)):
            if i == row or self._cell(i, "modality") == "eyetrack":
                continue
            if not self._model.entities(i).get("task") or \
                    self._model.datatype_suffix(i)[1] in ("physio", "sbref", "events"):
                continue
            if self._cell(i, "datatype") not in ("func", "eeg", "meg", "ieeg", "nirs", "beh",
                                                 "pet", "motion"):
                continue
            out.append(i)
        return sorted(out, key=lambda i: (self._cell(i, "participant_id") != me,
                                          self._cell(i, "participant_id"),
                                          self._cell(i, "study_date"), self._cell(i, "acq_time")))

    def _run_label(self, i: int) -> str:
        ents = self._model.entities(i)
        bits = [self._cell(i, "participant_id"), self._cell(i, "datatype")]
        if ents.get("session"):
            bits.append(f"ses-{ents['session']}")
        bits.append(f"task-{ents['task']}")
        if ents.get("run"):
            bits.append(f"run-{ents['run']}")
        when = self._cell(i, "acq_time")
        if len(when) >= 4 and when[:4].isdigit():
            bits.append(f"{when[:2]}:{when[2:4]}")
        return " · ".join(bits)

    def pair_with_run(self, row: int, other: int) -> None:
        """Name this eye-tracking recording after the run it was recorded
        during: its datatype folder, session, task, acquisition and run, as
        one undoable change."""
        if self._model is None:
            return
        m = self._model
        theirs = m.entities(other)
        datatype = self._cell(other, "datatype")
        with m.edit_group():
            m.set_datatype_suffix(row, datatype, "physio")
            for key in ("subject", "session", "task", "acquisition", "run"):
                m.set_entity(row, key, theirs.get(key) or None)
        self.set_selected_row(row)

    def _eye_row(self, label: str, tip: str, widget: QWidget) -> QWidget:
        holder = QWidget()
        holder.setObjectName("meta-row")
        holder.setStyleSheet("#meta-row { background: transparent; }")
        h = QHBoxLayout(holder)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        lbl = FieldLabel(label, CUR()["dim"], _LABEL_COL, fixed=True)
        lbl.setToolTip(tip)
        widget.setToolTip(tip)
        h.addWidget(lbl)
        h.addWidget(widget, 1)
        return holder

    def _eye_edit(self, row: int, field: str, *, scale: float = 1.0, pair: bool = False,
                  hint: str = "") -> QLineEdit:
        """A field of the eye-tracking block, shown in the unit people use
        (``scale`` from the stored one: cm for metres), the dataset default
        as its placeholder."""
        def show(value) -> str:
            if value in (None, "", []):
                return ""
            if pair:
                return " x ".join(f"{float(v) * scale:g}" for v in value)
            if isinstance(value, (int, float)):
                return f"{float(value) * scale:g}"
            return str(value)

        edit = QLineEdit(show(self._model.eyetrack_override(row, field)))
        edit.setObjectName("ent-input")
        default = show(self._model.eyetrack_default(field))
        edit.setPlaceholderText(f"{default} (dataset default)" if default else hint)

        def commit() -> None:
            text = edit.text().strip().replace(",", " ").replace("x", " ").replace("X", " ")
            value = None
            if text:
                if field in ("sync_message",):
                    value = edit.text().strip()
                else:
                    try:
                        numbers = [float(n) / scale for n in text.split()]
                    except ValueError:
                        edit.setStyleSheet(f"border-color: {CUR()['error']};")
                        return
                    if pair:
                        if len(numbers) != 2:
                            edit.setStyleSheet(f"border-color: {CUR()['error']};")
                            return
                        value = [round(n, 6) for n in numbers]
                    else:
                        value = round(numbers[0], 6)
            edit.setStyleSheet("")
            if self._model is not None:
                self._model.set_eyetrack_override(row, field, value)

        edit.editingFinished.connect(commit)
        return edit

    def _append_eyetrack_section(self, row: int) -> None:
        """The eye tracker's file knows the tracker, the eye, the calibration
        and the screen's resolution. Not the run it belongs to, the screen's
        size and distance (which the standard requires for gaze on a
        screen), nor where the run starts on its clock."""
        if self._model is None:
            return
        self._body_layout.addSpacing(8)
        self._body_layout.addWidget(self._divider())
        self._body_layout.addWidget(self._section_header(
            "EYE TRACKING", "physio sidecar + the run's events.json",
            agnostic=False, tag="this recording"))

        runs = self._runs_of(row)
        pick = QComboBox()
        pick.setObjectName("ent-input")
        # A combo asks for its longest entry as its minimum width; a run's
        # label would set a floor under the whole panel.
        pick.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        pick.setMinimumContentsLength(8)
        pick.addItem("(choose the run)" if runs else "(no run to choose)", -1)
        for i in runs:
            pick.addItem(self._run_label(i), i)
        pick.setEnabled(bool(runs))
        pick.activated.connect(
            lambda idx: (pick.itemData(idx) is not None and pick.itemData(idx) >= 0
                         and self.pair_with_run(row, int(pick.itemData(idx)))))
        self.eyetrack_run_pick = pick
        self._body_layout.addWidget(self._eye_row(
            "Run",
            "The run this eye tracker recorded during. Choosing it names the files after "
            "it (folder, session, task, run), so they sit beside the run in BIDS.", pick))

        self.eyetrack_fields = {
            "screen_size": self._eye_edit(row, "screen_size", scale=100, pair=True,
                                          hint="width x height, cm"),
            "screen_distance": self._eye_edit(row, "screen_distance", scale=100,
                                              hint="eye to screen, cm"),
            "eye_tracker_distance": self._eye_edit(row, "eye_tracker_distance", scale=100,
                                                   hint="eye to camera, cm"),
            "sync_message": self._eye_edit(row, "sync_message",
                                           hint="a message, e.g. the first trigger"),
            "start_time": self._eye_edit(row, "start_time", hint="seconds"),
        }
        rows = (
            ("Screen (cm)", "screen_size",
             "The visible screen, width x height in centimetres (ScreenSize, in metres "
             "in the file). Required with gaze measured on a screen."),
            ("Distance (cm)", "screen_distance",
             "From the eye to the screen, in centimetres (ScreenDistance)."),
            ("Tracker (cm)", "eye_tracker_distance",
             "From the eye to the tracker's camera, in centimetres (EyeTrackerDistance)."),
            ("Sync message", "sync_message",
             "The text of the message the experiment logged when the run started (the "
             "scanner's first trigger, say). StartTime becomes the first sample's time "
             "before it. Left empty, the recording is taken to start the run."),
            ("Start (s)", "start_time",
             "StartTime outright, in seconds from the run's start (negative when the eye "
             "tracker started first). Takes precedence over the message."),
        )
        for label, field, tip in rows:
            self._body_layout.addWidget(self._eye_row(label, tip, self.eyetrack_fields[field]))

    # ------------------------------------------------------------------
    # The schema-driven sidecar section, for this recording alone
    # ------------------------------------------------------------------

    def _append_sidecar_section(self, row: int, datatype: str, suffix: str) -> None:
        """Everything the standard lets this file carry, answered per recording.

        The same definition the dataset dialog renders, at a scope one step
        down. Before this the panel wrote its own sections, so which fields a
        user could state depended on which surface they happened to open, and
        the two lists disagreed. Now there is one list, and the dialog says what
        is true of every ``*_eeg.json`` while this says what is true of THIS one.

        Split the same way the dialog splits it. What THIS recording already
        answers, from its own header and its own entities, is settled and comes
        first, in green; what it does not is asked below. Everything stays
        editable, since correcting the one recording whose header lies is
        exactly what a per-row answer is for.

        Each control opens showing what the file will actually say, whichever
        layer said it, with the layer named in the tooltip. Answering here
        overrides that for this recording; clearing it hands the field back.
        """
        if self._model is None or not datatype:
            return
        answered = self._model.row_answered(row)
        section = sidecar_section(
            datatype, suffix or datatype,
            self._sidecar_example_path(row, datatype, suffix),
            answered=answered,
        )
        if not section.fields and not answered:
            return

        resolved = self._model.resolved_sidecar(row)
        stated = self._model.row_template(row)
        filename = section.target.rpartition("/")[2]

        # Built when opened, not on every click: it is the largest part of
        # the panel (a few hundred controls) and closed by default.
        box = CollapsibleSection(
            "Sidecar fields",
            subtitle=filename,
            badge=f"{len(section.fields)} to answer",
            expanded=self._sidecar_expanded,
            body_factory=lambda r=row, s=section, a=answered, res=resolved, st=stated:
                self._build_sidecar_body(r, datatype, s, a, res, st),
        )
        box.toggled_by_user.connect(self._remember_sidecar_expanded)
        self._body_layout.addSpacing(8)
        self._body_layout.addWidget(box)

    def _build_sidecar_body(self, row: int, datatype: str, section, answered: dict,
                            resolved: dict, stated: dict) -> QWidget:
        """The sidecar section's content (built on first opening)."""
        holder_all = QWidget()
        holder_all.setObjectName("meta-row")
        holder_all.setStyleSheet("#meta-row { background: transparent; }")
        col = QVBoxLayout(holder_all)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)

        class _Box:                          # the old ``box.add`` call sites
            @staticmethod
            def add(widget):
                col.addWidget(widget)

        box = _Box()
        # Settled first, in green: this recording's header and entities answer
        # these, and the user needs to see what they need NOT fill in.
        settled = {
            name: answered.get(name)
            for name in sorted(set(section.supplied) | set(answered))
        }
        if settled:
            box.add(self._build_answered_block(row, section, settled))
        box.add(level_legend(colour=True))

        for field in section.fields:
            widget = build_field_widget(field, self._field_suggestions(row, field.name))
            answer = resolved.get(field.name)
            if answer is not None:
                write_field_widget(widget, answer.value)

            tip = field.description or ""
            if answer is not None and field.name not in stated:
                # Inherited: say where from, so a value that appears out of
                # nowhere is accountable rather than mysterious.
                origin = self._model.sidecar_origin(row, field.name)
                tip = (tip + "\n\n" if tip else "") + f"Currently from {origin}."
            if tip:
                widget.setToolTip(tip)

            connect_field_widget(
                widget, lambda f=field, w=widget: self._on_sidecar_field_changed(f, w),
            )
            label = field_label_widget(field, width=_LABEL_COL, fixed=True)
            if tip:
                label.setToolTip(f"{field.name}\n\n{tip}")
            holder = QWidget()
            holder.setObjectName("meta-row")
            holder.setStyleSheet("#meta-row { background: transparent; }")
            line = QHBoxLayout(holder)
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(8)
            line.addWidget(label)
            line.addWidget(widget, 1)
            box.add(holder)
            # Reading the spectrum is how you find out which of 50 and 60 the
            # mains was, so the action sits directly under the field it answers
            # rather than in a block of its own.
            if field.name == "PowerLineFrequency" and datatype in _EEG_MEG_DATATYPES:
                box.add(self._build_psd_row(row))
        return holder_all

    def _build_answered_block(self, row: int, section, settled: dict) -> QWidget:
        """The fields this recording already answers, folded away and editable.

        Green because nothing in it is outstanding. Editable because the value
        was read out of the data, and when the data is wrong this is the only
        place to say so.
        """
        pal = CUR()
        # Every field the file may carry, not just the ones being asked, or
        # a required field the recording answered shows with no level.
        declared = {f.name: f for f in section.declared}
        box = CollapsibleSection(
            "Already answered by the conversion",
            subtitle="edit only to correct what the data says",
            badge=f"{len(settled)} fields",
            level=1,
            expanded=False,
            tone=pal["success"],
        )
        for name, value in settled.items():
            field = declared.get(name) or _answered_field(name, value)
            widget = build_field_widget(field, self._field_suggestions(row, name))
            if value is not None:
                write_field_widget(widget, value)
            widget.setToolTip(
                (field.description + "\n\n" if field.description else "")
                + ("The conversion reads this from the data and will write the "
                   "value shown. Type here only to correct it."
                   if value is not None else
                   "The conversion normally supplies this from the data.")
            )
            connect_field_widget(
                widget, lambda f=field, w=widget: self._on_sidecar_field_changed(f, w),
            )
            holder = QWidget()
            holder.setObjectName("meta-row")
            holder.setStyleSheet("#meta-row { background: transparent; }")
            line = QHBoxLayout(holder)
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(8)
            line.addWidget(field_label_widget(field, width=_LABEL_COL, fixed=True))
            line.addWidget(widget, 1)
            box.add(holder)
        return box

    def _sidecar_example_path(self, row: int, datatype: str, suffix: str) -> str:
        """The name of the file this row will produce, for the section heading."""
        if self._model is None:
            return ""
        basename = self._cell(row, "bids_name")
        return f"{datatype}/{basename}.json" if basename else ""

    def _field_suggestions(self, row: int, name: str) -> tuple:
        """Values BIDS Manager offers that the standard does not.

        Two sources: the vocabularies we curate, and whatever the scan read out
        of THIS recording's own header. Neither is applied on its own, because a
        vendor string can always parse wrongly and a wrong answer is worse than
        a blank one.
        """
        curated = CURATED_SUGGESTIONS.get(name, ())
        column = SCAN_SUGGESTION_COLUMNS.get(name, "")
        hint = self._cell(row, column) if column else ""
        return ((hint,) if hint and hint not in curated else ()) + curated

    def _remember_sidecar_expanded(self, expanded: bool) -> None:
        self._sidecar_expanded = expanded

    def _on_sidecar_field_changed(self, field, widget) -> None:
        """Commit one sidecar answer against this recording."""
        if self._suppress_writeback or self._model is None or self._row is None:
            return
        self._model.set_row_template_field(
            self._row, field.name, read_field_widget(widget, field),
        )

    # ------------------------------------------------------------------
    # PSD compute (per-row, threaded, mirrors the Editor recording viewer)
    # ------------------------------------------------------------------

    def _build_psd_row(self, row: int) -> QWidget:
        """A ``[Compute PSD] [spinner]`` action row, left-aligned under the
        field column. Disabled when no source recording can be located. While
        this row's recording is computing it shows the busy spinner; only one
        PSD runs at a time."""
        row_w = QWidget()
        row_w.setObjectName("meta-row")
        row_w.setStyleSheet("#meta-row { background: transparent; }")
        h = QHBoxLayout(row_w)
        # 76px label column + 8px spacing aligns the button under the field.
        h.setContentsMargins(84, 2, 0, 2)
        h.setSpacing(8)

        is_computing = (
            self._psd_row_id is not None
            and self._model is not None
            and self._psd_row_id is not None
            and self._psd_row_id == self._model.row_id(row)
        )
        path = self._resolve_source_path(row)

        btn = QPushButton("  Compute PSD")
        btn.setObjectName("tb-btn")
        icons.apply_button(btn, "psd")
        btn.setToolTip(
            "Compute the power spectral density of this recording. Reads the "
            "file on a background thread and opens the same interactive PSD "
            "viewer as the Editor."
        )
        if is_computing:
            btn.setText("  Computing PSD…")
            btn.setEnabled(False)
        elif path is None:
            btn.setEnabled(False)
            btn.setToolTip("No readable source recording found for this row.")
        else:
            btn.clicked.connect(lambda _=False, r=row: self.compute_psd(r))
        h.addWidget(btn)

        spinner = BusySpinner()
        if is_computing:
            spinner.set_busy(True, message="")
        h.addWidget(spinner)
        h.addStretch(1)
        return row_w

    def _resolve_source_path(self, row: int) -> Optional[Path]:
        """The row's recording (``source_file``) where it is now, or ``None``
        (DICOM rows have no source_file; a moved recording is not found)."""
        src = self._cell(row, "source_file").strip()
        if not src or self._sources is None:
            return None
        return self._sources.file(src)

    def _runner(self):
        """The panel's job runner (PSD and series previews), made on first use."""
        if self._jobs is None:
            from .viz.bridge import JobRunner

            self._jobs = JobRunner(self)
            self._jobs.done.connect(self._on_job_done)
            self._jobs.failed.connect(self._on_job_failed)
        return self._jobs

    def _build_fieldmap_links(self, row: int) -> Optional[QWidget]:
        """Which runs this fieldmap corrects, as the conversion will link
        them (``fixups.intended_for.plan_links``), with the reason; ticking
        and unticking makes it the user's own choice."""
        links = self._fieldmap_links(row)
        if links is None:
            return None
        candidates, chosen, own, reason = links
        box = CollapsibleSection(
            "Corrects these runs",
            subtitle=("your own choice" if own else "chosen by BIDS Manager"),
            badge=f"{len(chosen)} of {len(candidates)}",
            expanded=True,
        )
        pal = CUR()
        why = QLabel(reason if not own else
                     "Your own choice. The conversion writes exactly these.")
        why.setObjectName("dlg-hint")
        why.setWordWrap(True)
        why.setStyleSheet(f"color:{pal['dim']};")
        box.add(why)
        if not candidates:
            none = QLabel("No BOLD, diffusion or ASL run of this subject and session to "
                          "correct.")
            none.setObjectName("dlg-hint")
            none.setWordWrap(True)
            box.add(none)
            return box
        self.fieldmap_checks: dict[int, QCheckBox] = {}
        for r in candidates:
            name = self._cell(r, "bids_name") or self._cell(r, "sequence")
            check = QCheckBox(name)
            check.setChecked(r in chosen)
            check.setToolTip(f"{self._cell(r, 'sequence')}  ({self._cell(r, 'acq_time')})")
            check.toggled.connect(lambda _on, rr=row: self._on_fieldmap_choice(rr))
            box.add(check)
            self.fieldmap_checks[r] = check
        if own:
            back = QPushButton("Let BIDS Manager choose again")
            back.setObjectName("tb-btn-ghost")
            back.setToolTip("Forget your choice: the conversion links this fieldmap by "
                            "shim settings and acquisition times.")
            back.clicked.connect(lambda _c=False, rr=row: self._on_fieldmap_reset(rr))
            self.fieldmap_reset = back
            box.add(back)
        return box

    def _build_why_this_name(self, row: int) -> Optional[QWidget]:
        """What the acquisition parameters made of this series: the name and
        how sure, who decided, the evidence for and against, and the other
        names worth a click (each one undoable edit)."""
        info = self._model.classification(row) if self._model is not None else {}
        if not info:
            return None
        pal = CUR()
        chosen = info.get("chosen") or ""
        who = {"agreement": "the parameters and dcm2niix agree",
               "parameters": "named from the acquisition parameters",
               "dcm2niix": "dcm2niix's guess (the parameters were not sure)",
               "names": "the series name"}.get(info.get("decided_by", ""),
                                               "the parameters did not decide")
        box = CollapsibleSection(
            "Why this name", subtitle=who, badge=f"{info['confidence']:.2f}",
            expanded=bool(info.get("check")),
        )
        box.setToolTip("How the parameter classifier read this series: each line is one "
                       "piece of evidence and what it counted for or against. Settings > "
                       "Classification tunes it.")
        self.why_box = box
        head = QLabel(f"{chosen}" + (" · worth a look" if info.get("check") else ""))
        head.setObjectName("why-head")
        head.setStyleSheet(f"color:{pal['warning'] if info.get('check') else pal['text']};"
                           "font-weight:600; background:transparent;")
        box.add(head)
        if info.get("note"):
            note = QLabel(info["note"][0].upper() + info["note"][1:] + ".")
            note.setObjectName("dlg-hint")
            note.setWordWrap(True)
            box.add(note)
        from .viz import fonts

        self.why_lines = []
        for line in info.get("for", [])[:6]:
            holder = QWidget()
            holder.setObjectName("meta-row")
            holder.setStyleSheet("#meta-row { background: transparent; }")
            h = QHBoxLayout(holder)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)
            points = QLabel(f"{line['points']:+.1f}")
            points.setFont(fonts.font(11, mono=True))
            points.setFixedWidth(fonts.px(38))
            points.setStyleSheet(
                f"color:{pal['success'] if line['points'] > 0 else pal['error']};"
                "background:transparent;")
            text = QLabel(line["text"])
            text.setWordWrap(True)
            text.setToolTip({"parameters": "From the acquisition parameters",
                             "names": "From the series name",
                             "context": "From the series beside it"}.get(line["source"], ""))
            text.setStyleSheet(f"color:{pal['text']}; background:transparent;")
            h.addWidget(points)
            h.addWidget(text, 1)
            box.add(holder)
            self.why_lines.append(line["text"])
        # Entities the acquisition decided (ce- from the contrast agent),
        # whoever named the suffix.
        short = {"ceagent": "ce", "direction": "dir", "inversion": "inv", "part": "part"}
        entities = self._model.entities(row) if self._model is not None else {}
        for key, reason in (info.get("entities") or {}).items():
            value = entities.get(key)
            if not value:
                continue
            chip = QLabel(f"{short.get(key, key)}-{value}")
            chip.setFont(fonts.font(11, mono=True))
            chip.setStyleSheet(f"color:{pal['accent']}; background:transparent;")
            text = QLabel(reason[0].upper() + reason[1:] + ".")
            text.setWordWrap(True)
            text.setStyleSheet(f"color:{pal['text']}; background:transparent;")
            holder = QWidget()
            holder.setObjectName("meta-row")
            holder.setStyleSheet("#meta-row { background: transparent; }")
            h = QHBoxLayout(holder)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)
            h.addWidget(chip)
            h.addWidget(text, 1)
            box.add(holder)
            self.why_lines.append(f"{short.get(key, key)}-{value}: {reason}")
        # Alternatives worth a click: the next names with some weight, and
        # dcm2niix's own guess.
        options = self._alternative_names(row, info)
        self.why_alternatives = {}
        if options:
            row_w = QWidget()
            row_w.setObjectName("meta-row")
            row_w.setStyleSheet("#meta-row { background: transparent; }")
            from .widgets.flow_layout import flow

            bar = flow(row_w, h_spacing=6, v_spacing=4)
            for name, share in options:
                label = f"Use {name}" + (f" ({share:.2f})" if share is not None else
                                        " (dcm2niix)")
                button = QPushButton(label)
                button.setObjectName("tb-btn-ghost")
                button.setToolTip(f"Name this series {name} instead (one Undo takes it "
                                  "back)")
                button.clicked.connect(lambda _c=False, r=row, n=name: self.use_alternative(r, n))
                bar.addWidget(button)
                self.why_alternatives[name] = button
            box.add(row_w)
        return box

    def add_row_actions(self, menu, row: int, *, extra: Optional[list] = None) -> None:
        """What this panel offers for ``row``, as entries of a menu (the
        inspection table's right-click): the same wording, the same reasons
        an action is unavailable, the same handlers, so the two surfaces
        cannot drift apart."""
        if self._model is None or not (0 <= row < self._model.rowCount()):
            return
        from . import icons

        datatype, suffix = self._model.datatype_suffix(row)
        added = False
        # ``extra``: the caller's own actions on this row (the Converter's
        # DICOM header) join these, ahead of what names and links the row.
        if self._cell(row, "series_uid").strip() or self._cell(row, "source_file").strip():
            text, tip = self._preview_wording(datatype, suffix)
            act = menu.addAction(icons.icon("preview"), text + "...")
            blocker = self._preview_blocker(row)
            busy = self._preview_row_id is not None
            act.setEnabled(not blocker and not busy)
            act.setToolTip(blocker or ("A preview is already being made." if busy else
                                       tip + " Nothing is written to the dataset."))
            act.triggered.connect(lambda _c=False, r=row: self.preview(r))
            added = True
        if datatype in _EEG_MEG_DATATYPES:
            act = menu.addAction(icons.icon("psd"), "Compute the power spectrum")
            found = self._resolve_source_path(row) is not None
            act.setEnabled(found and self._psd_row_id is None)
            act.setToolTip("Welch power spectrum per channel and per channel type" if found
                           else "The recording of this row was not found.")
            act.triggered.connect(lambda _c=False, r=row: self.compute_psd(r))
            added = True
        for act in extra or []:
            menu.addAction(act)
            added = True
        sections = []
        options = self._alternative_names(row)
        if options:
            sub = menu.addMenu(icons.icon("settings_classify"), "Name it instead")
            for name, share in options:
                label = name + (f"   ({share:.2f})" if share is not None else "   (dcm2niix)")
                act = sub.addAction(label)
                act.setToolTip(f"Name this series {name} (one Undo takes it back)")
                act.triggered.connect(lambda _c=False, r=row, n=name: self.use_alternative(r, n))
            sections.append(sub)
        if datatype == "fmap":
            links = self._fieldmap_links(row)
            if links is not None:
                candidates, chosen, own, reason = links
                sub = menu.addMenu(icons.icon("channels"), "Corrects these runs")
                sub.setToolTipsVisible(True)
                for r in candidates:
                    act = sub.addAction(self._cell(r, "bids_name") or self._cell(r, "sequence"))
                    act.setCheckable(True)
                    act.setChecked(r in chosen)
                    act.setToolTip(f"{self._cell(r, 'sequence')}  ({self._cell(r, 'acq_time')})")
                    act.toggled.connect(lambda on, f=row, run=r: self._toggle_fieldmap_run(f, run, on))
                if not candidates:
                    none = sub.addAction("No BOLD, diffusion or ASL run to correct")
                    none.setEnabled(False)
                if own:
                    sub.addSeparator()
                    back = sub.addAction("Let BIDS Manager choose again")
                    back.triggered.connect(lambda _c=False, r=row: self._on_fieldmap_reset(r))
                sections.append(sub)
        if self._cell(row, "modality").strip() == "eyetrack":
            runs = self._runs_of(row)
            sub = menu.addMenu(icons.icon("events"), "Pair with a run")
            for i in runs[:40]:
                act = sub.addAction(self._run_label(i))
                act.setToolTip("Take this run's subject, session, task, acquisition and run "
                               "(one Undo)")
                act.triggered.connect(lambda _c=False, r=row, other=i: self.pair_with_run(r, other))
            if not runs:
                none = sub.addAction("No run with a task to pair with")
                none.setEnabled(False)
            sections.append(sub)
        if sections and added:
            # The actions first, then what names and links the row.
            menu.insertSeparator(sections[0].menuAction())

    def _alternative_names(self, row: int, info: Optional[dict] = None) -> list[tuple[str, Optional[float]]]:
        """The other names worth offering for ``row``: the classifier's next
        candidates with some weight (and their share), then dcm2niix's own
        guess (share ``None``); never the current name or a leave-out."""
        info = info if info is not None else (
            self._model.classification(row) if self._model is not None else {})
        if not info:
            return []
        shares = {a["name"]: a.get("p", 0) for a in info.get("alternatives", [])}
        options = [a["name"] for a in info.get("alternatives", [])[:2] if a.get("p", 0) >= 0.05]
        # When the name or dcm2niix decided, the parameters' own best name is
        # an alternative too (with their confidence).
        ours = info.get("ours") or ""
        if ours and ours != (info.get("chosen") or "") and ours not in options:
            shares[ours] = info.get("confidence", 0.0)
            options.insert(0, ours)
        other = info.get("other") or ""
        if other and other != (info.get("chosen") or "") and other not in options:
            options.append(other)
        current = f"{self._cell(row, 'datatype')}/{self._cell(row, 'bids_guess_suffix')}"
        return [(o, shares.get(o) if o != other or o in shares else None) for o in options
                if o and o != current and not o.startswith("exclude/")]

    def use_alternative(self, row: int, name: str) -> None:
        if self._model is not None and self._model.use_alternative(row, name):
            QTimer.singleShot(0, lambda: self.set_selected_row(row))

    def _session_runs(self, df, who: tuple[str, str]) -> list[int]:
        from ..fixups.intended_for import is_target_row

        out = []
        for r in range(len(df)):
            if (self._cell(r, "participant_id"), self._cell(r, "session")) != who:
                continue
            suffix = self._cell(r, "bids_guess_suffix") or \
                self._cell(r, "bids_name").rsplit("_", 1)[-1]
            if is_target_row(self._cell(r, "datatype"), suffix) and \
                    self._cell(r, "include") not in ("0", "false", "False"):
                out.append(r)
        return out

    def _fieldmap_links(self, row: int) -> Optional[tuple[list[int], set[int], bool, str]]:
        """``(candidate runs, chosen runs, the user's own choice?, reason)``
        for a fieldmap row, by the conversion's rule (``fixups.intended_for.
        plan_links``, computed once per table state); ``None`` otherwise."""
        from ..fixups.intended_for import plan_links

        key = (id(self._model), self._model.revision, id(self._shims))
        if self._links_cache is None or self._links_cache[0] != key:
            df = self._model.dataframe()
            self._links_cache = (key, df, *plan_links(df, shims=self._shims))
        _key, df, plan, reasons = self._links_cache
        if row not in plan:
            return None
        me = (self._cell(row, "participant_id"), self._cell(row, "session"))
        candidates = ({x for v in plan.values() for x in v} | set(self._session_runs(df, me)))
        candidates.discard(row)
        ordered = sorted(candidates, key=lambda r: self._cell(r, "bids_name"))
        own = reasons.get(row) == "your own choice"
        return ordered, set(plan.get(row, [])), own, reasons.get(row, "")

    def _toggle_fieldmap_run(self, row: int, run: int, on: bool) -> None:
        """The user ticks or unticks one run this fieldmap corrects (one Undo)."""
        links = self._fieldmap_links(row)
        if links is None or self._model is None:
            return
        _candidates, chosen, _own, _reason = links
        chosen = (chosen | {run}) if on else (chosen - {run})
        self._model.set_fieldmap_targets(row, [self._model.row_id(r) for r in sorted(chosen)])
        if self._row == row:
            QTimer.singleShot(0, lambda: self.set_selected_row(row))

    def _on_fieldmap_choice(self, row: int) -> None:
        if self._model is None:
            return
        ids = [self._model.row_id(r) for r, c in self.fieldmap_checks.items() if c.isChecked()]
        self._model.set_fieldmap_targets(row, ids)
        QTimer.singleShot(0, lambda: self.set_selected_row(row))

    def _on_fieldmap_reset(self, row: int) -> None:
        if self._model is not None:
            self._model.set_fieldmap_targets(row, None)
            QTimer.singleShot(0, lambda: self.set_selected_row(row))

    def _preview_blocker(self, row: int) -> str:
        """Why this row cannot be previewed now, or ``""``."""
        from ..inventory.types import NONIMAGE_ISSUE_TOKEN

        if NONIMAGE_ISSUE_TOKEN in self._cell(row, "issues"):
            return ("This series holds no image (a report or another non-image object), "
                    "so there is nothing to look at.")
        if self._sources is None or self._sources.root is None:
            return "The folder this scan read is not known."
        uid = self._cell(row, "series_uid").strip()
        if uid:
            files = self._sources.series_files(uid)
            found = bool(files) and files[0].exists()
            if not found and not files:
                found = self._sources.folder(self._cell(row, "source_folder")) is not None
        else:
            found = self._resolve_source_path(row) is not None
        if not found:
            return ("The source data of this row is not where the scan found it. Locate it "
                    "from the Raw input bar above.")
        return ""

    @staticmethod
    def _preview_wording(datatype: str, suffix: str) -> tuple[str, str]:
        """The button's words and what it shows, by what the row is."""
        if suffix == "physio" or datatype in _EEG_MEG_DATATYPES:
            return ("Preview the recording",
                    "Show the recording in the signal viewer before converting anything. "
                    "A physiology log is converted the way the conversion will; an EEG or "
                    "MEG recording is read as it is.")
        if datatype == "mrs":
            return ("Preview the spectrum",
                    "Convert this series now, the way the conversion will, and show its "
                    "spectrum.")
        return ("Preview the image",
                "Convert this row now, the way the conversion will, into a temporary "
                "folder, and look at what it makes (every image, when it makes several).")

    def _build_preview_row(self, row: int, datatype: str, suffix: str) -> QWidget:
        """``[Preview ...] [spinner]``: convert this one row now and look at
        what it makes, before converting anything for real."""
        row_w = QWidget()
        row_w.setObjectName("meta-row")
        row_w.setStyleSheet("#meta-row { background: transparent; }")
        h = QHBoxLayout(row_w)
        h.setContentsMargins(0, 4, 0, 2)
        h.setSpacing(8)
        busy = (self._preview_row_id is not None and self._model is not None
                and self._preview_row_id == self._model.row_id(row))
        text, tip = self._preview_wording(datatype, suffix)
        btn = QPushButton(text)
        btn.setObjectName("tb-btn")
        btn.setToolTip(tip + " Nothing is written to the dataset.")
        blocker = "" if busy else self._preview_blocker(row)
        if busy:
            btn.setText("Converting for a preview...")
            btn.setEnabled(False)
        elif blocker:
            btn.setEnabled(False)
            btn.setToolTip(blocker)
        elif self._preview_row_id is not None:
            btn.setEnabled(False)
        else:
            btn.clicked.connect(lambda _=False, r=row: self.preview(r))
        self.preview_button = btn
        h.addWidget(btn)
        spinner = BusySpinner()
        if busy:
            spinner.set_busy(True, message="")
        h.addWidget(spinner)
        h.addStretch(1)
        return row_w

    def preview(self, row: int) -> None:
        """Convert ``row`` now into a temporary folder and show what it makes."""
        import tempfile

        if self._model is None or self._preview_row_id or self._preview_blocker(row):
            return
        record = self._model.dataframe().iloc[row].to_dict()
        self._preview_dir = Path(tempfile.mkdtemp(prefix="bidsmgr-preview-"))
        self._preview_row_id = self._model.row_id(row)
        self._preview_generation += 1
        title = (self._cell(row, "SeriesDescription").strip()
                 or self._cell(row, "bids_name").strip()
                 or Path(self._cell(row, "source_file")).name)
        self._runner().start("preview", self._preview_generation, _preview_job,
                             record, self._preview_dir, self._sources)
        # The window opens NOW, on the click, with a spinner, and fills when
        # the row is converted: opened when it finished, it came up seconds
        # later, behind the main window.
        from .preview_dialog import PreviewDialog

        dlg = PreviewDialog(title, self._preview_dir, parent=self)
        self.preview_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        self.set_selected_row(row)

    def _preview_window(self):
        """The preview window still open for the current conversion."""
        dlg = getattr(self, "preview_dialog", None)
        return dlg if dlg is not None and dlg.isVisible() and dlg.is_pending() else None

    def _on_preview_ready(self, generation: int, files) -> None:
        if generation != self._preview_generation:
            return
        import shutil

        self._preview_row_id = None
        dlg = self._preview_window()
        if dlg is not None:
            dlg.set_files(files, self._preview_dir)
            # If the app is not in front (the user switched away while it
            # converted): the Dock icon bounces, the taskbar entry flashes.
            from PyQt6.QtWidgets import QApplication

            QApplication.alert(dlg)
        elif self._preview_dir is not None:
            # Closed while converting: nothing to show, nothing to keep.
            shutil.rmtree(self._preview_dir, ignore_errors=True)
        self._preview_dir = None
        if self._row is not None:
            self.set_selected_row(self._row)

    def _on_preview_failed(self, generation: int, msg: str) -> None:
        if generation != self._preview_generation:
            return
        import shutil

        self._preview_row_id = None
        if self._preview_dir is not None:
            shutil.rmtree(self._preview_dir, ignore_errors=True)
            self._preview_dir = None
        dlg = self._preview_window()
        if dlg is not None:
            dlg.set_failed(msg)
        if self._row is not None:
            self.set_selected_row(self._row)

    def compute_psd(self, row: int) -> None:
        """The spectrum of this row's recording: THE spectrum the Editor's
        viewer computes (``viz.compute.spectral.psd``, Welch over every
        channel's samples), so the same file gives the same answer in both
        places. It used to be ``Raw.compute_psd`` here, which drops stim and
        misc channels, and Welch in the viewer."""
        if self._model is None or self._psd_row_id is not None:
            return
        path = self._resolve_source_path(row)
        if path is None:
            QMessageBox.warning(
                self, "PSD", "No readable source recording found for this row."
            )
            return

        self._psd_row_id = self._model.row_id(row)
        self._psd_generation += 1
        self._runner().start("psd", self._psd_generation, _row_spectrum, path)
        # Re-render so the button shows its busy state (spinner + disabled).
        self.set_selected_row(row)

    def _on_job_done(self, tag: str, generation: int, result) -> None:
        if tag == "preview":
            self._on_preview_ready(generation, result)
            return
        if generation != self._psd_generation:
            return
        self._psd_row_id = None
        from .viz.canvases.psd import PsdWindow

        win = PsdWindow(result, parent=self)
        win.show()
        self._psd_window = win
        if self._row is not None:
            self.set_selected_row(self._row)

    def _on_job_failed(self, tag: str, generation: int, msg: str) -> None:
        if tag == "preview":
            self._on_preview_failed(generation, msg)
            return
        if generation != self._psd_generation:
            return
        self._psd_row_id = None
        QMessageBox.warning(self, "PSD error", str(msg))
        if self._row is not None:
            self.set_selected_row(self._row)

    def _montage_hint(self, suggestion: str) -> QWidget:
        return self._scan_hint("montage", "best montage match (from scan)", suggestion)

    def _scan_hint(self, field_key: str, prefix: str, value: str) -> QWidget:
        """A read-only scan-suggestion hint row (e.g. montage match, detected
        manufacturer). Aligned under the field column; not auto-applied. Carries
        an explicit QToolTip background so it never renders transparent."""
        pal = CUR()
        lbl = QLabel(
            f'<span style="color:{pal["dim"]};">{prefix}: </span>'
            f'<span style="color:{pal["teal"]};font-weight:700;">{value}</span>'
        )
        lbl.setObjectName("scan-hint")
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setWordWrap(True)
        # Scoped (objectName) so the transparent background does not leak into
        # this label's tooltip; the tooltip then uses the app QToolTip styling.
        lbl.setStyleSheet(
            f"#scan-hint {{ font-size: {scaled_px(10)}px; background: transparent; "
            "margin-left: 84px; }"
        )
        lbl.setToolTip(
            "Suggestion computed at scan; not applied automatically. Pick it in "
            "the field above if appropriate."
        )
        return lbl

    def _append_companion_section(self, row: int) -> None:
        """Link already-curated companion files (events/beh/stim/...) for a row.

        Modality-agnostic: any recording can carry curated sidecar companions
        the converter copies into the BIDS tree (place + name, no conversion).
        """
        self._body_layout.addSpacing(8)
        self._body_layout.addWidget(self._divider())
        self._body_layout.addWidget(self._section_header(
            "COMPANION FILES", "events / beh / stim (copied into BIDS)",
            agnostic=True, tag="any datatype"))

        self._companion_list = QListWidget()
        self._companion_list.setMaximumHeight(72)
        # Blood curves are companions too, but they have their own section and
        # are converted rather than copied, so they are not offered here.
        self._plain_companions = [
            (suffix, path)
            for suffix, path in self._companions(row)
            if not is_blood_role(suffix)
        ]
        for suffix, path in self._plain_companions:
            self._companion_list.addItem(f"{suffix}: {path}")
        self._body_layout.addWidget(self._companion_list)

        ctl = QWidget()
        ctl.setObjectName("companion-ctl")
        ctl.setStyleSheet("#companion-ctl { background: transparent; }")
        # Wraps rather than sets a floor: in a row, the three controls were
        # wider than the panel at a larger font and pushed everything in it
        # past the right edge.
        from .widgets.flow_layout import flow

        h = flow(ctl, h_spacing=6, v_spacing=4)
        self._companion_suffix = QComboBox()
        self._companion_suffix.setObjectName("ent-input")  # opaque popup styling
        self._companion_suffix.addItems(
            ["events", "beh", "stim", "physio", "channels", "electrodes"]
        )
        link = QPushButton("Link file…")
        link.clicked.connect(lambda _=False, r=row: self._link_companion(r))
        rem = QPushButton("Remove")
        rem.clicked.connect(lambda _=False, r=row: self._remove_companion(r))
        h.addWidget(self._companion_suffix)
        h.addWidget(link)
        h.addWidget(rem)
        self._body_layout.addWidget(ctl)

    def _append_pet_dose_section(self, row: int) -> None:
        """Import a dose file, for THIS recording, inside the PET region.

        The dataset dialog has the same importer, and that is the right
        place for a study-wide dose. This one exists because a dose is not
        always study-wide: a second tracer, a re-injection, a subject whose
        record came from a different sheet. Asking a user to leave the row
        they are looking at, open a dataset dialog and scope a block by
        subject label, in order to correct one scan, is asking them to do
        the tool's filing for it.

        It sits INSIDE the PET region rather than above it so that what the
        file filled in is the next thing the eye reaches: the sidecar
        fields below update in place, and the point of importing rather
        than passing a path through is being able to see that happen.
        """
        from ..metadata.pet_metadata_json import read_pet_metadata_json
        from ..metadata.pet_spreadsheet import read_pet_spreadsheet

        self._body_layout.addSpacing(8)
        self._body_layout.addWidget(self._divider())
        self._body_layout.addWidget(self._section_header(
            "DOSE FILE", "fills the PET fields below, for this recording",
            agnostic=False, tag="pet"))

        line = QWidget()
        line.setObjectName("dose-row")
        line.setStyleSheet("#dose-row { background: transparent; }")
        h = QHBoxLayout(line)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(FieldLabel("Import", CUR()["text"], _LABEL_COL, fixed=True))

        status = QLineEdit()
        status.setObjectName("ent-input")
        status.setReadOnly(True)
        status.setPlaceholderText("(nothing imported for this recording)")
        status.setToolTip(
            "A JSON keyed by BIDS field names, which is the shape pypet2bids "
            "accepts through --set-default-metadata-json, or a spreadsheet. "
            "Applied to THIS recording only. Keys BIDS does not define for a "
            "PET sidecar are reported and never written."
        )
        h.addWidget(status, 1)

        browse = QPushButton("Choose…")
        browse.setToolTip("Pick a dose file and apply it to this recording.")

        def choose() -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select a PET dose file for this recording", "",
                "PET metadata (*.json *.tsv *.csv *.xlsx *.ods);;All files (*)",
            )
            if not path:
                return
            chosen = Path(path)
            blocks = (
                read_pet_metadata_json(chosen)
                if chosen.suffix.lower() == ".json"
                else read_pet_spreadsheet(chosen)
            )
            # One recording, so every block in the file is for it: the
            # dataset-wide one and any subject block alike. A file scoped to
            # somebody else is the user's mistake to see, not ours to guess
            # at, so they are merged in the order the file states them.
            merged = None
            for block in blocks.values():
                merged = block if merged is None else _merge_pet(merged, block)
            if merged is None:
                status.setText(f"{chosen.name}: nothing readable")
                return

            applied = (
                self._model.apply_pet_block(row, merged)
                if self._model is not None else 0
            )
            status.setText(f"{chosen.name}: {applied} field(s)")
            # Re-render so the sidecar fields below show what arrived. The
            # whole reason for importing here rather than passing a path.
            self.set_selected_row(row)

        browse.clicked.connect(choose)
        h.addWidget(browse)
        self._body_layout.addWidget(line)

    def _append_blood_section(self, row: int) -> None:
        """Attach this PET run's blood curves.

        Quantitative PET rests on the arterial input function: what the tracer
        was doing in the blood while the scanner counted it in tissue. BIDS has
        a place for it and a lab usually has it in PMOD exports.

        Three series, each drawn one of two ways. How it was drawn is not a
        detail: a hand-drawn series and an autosampled one have different time
        resolution, so BIDS puts them in separate files under the ``recording``
        entity, and pet2bids asks on the console when it cannot tell. In a
        window that is a freeze with no visible cause, which is why the method
        is chosen here rather than guessed later.

        Storage is the ordinary companion list, tagged ``blood:<series>:<method>``,
        because a blood curve belongs to exactly one PET run in the same way an
        events file belongs to one functional run. The copier skips these tags;
        conversion happens in `fixups/blood.py`.
        """
        from ..fixups.blood import BLOOD_SERIES, parse_blood_role

        labels = {
            "wholeblood": "Whole blood",
            "plasma": "Plasma",
            "parentfraction": "Parent fraction",
        }

        self._body_layout.addSpacing(8)
        self._body_layout.addWidget(self._divider())
        self._body_layout.addWidget(self._section_header(
            "BLOOD SAMPLING", "PMOD .bld curves, converted to _blood.tsv",
            agnostic=False, tag="pet"))

        linked: dict[str, tuple[str, str]] = {}
        for tag, path in self._companions(row):
            parsed = parse_blood_role(tag)
            if parsed is not None:
                linked[parsed[0]] = (parsed[1], path)

        self._blood_methods: dict[str, QComboBox] = {}
        for series in BLOOD_SERIES:
            method, path = linked.get(series, ("manual", ""))

            line = QWidget()
            line.setObjectName("blood-row")
            line.setStyleSheet("#blood-row { background: transparent; }")
            h = QHBoxLayout(line)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)

            # The panel's shared eliding label, so these line up with the
            # entities above and every schema-driven field below, and so a long
            # name narrows with the pane instead of setting a floor for it.
            h.addWidget(FieldLabel(
                labels[series], CUR()["text"], _LABEL_COL, fixed=True,
            ))

            how = QComboBox()
            how.setObjectName("ent-input")  # opaque popup styling
            how.addItems(["manual", "automatic"])
            how.setCurrentText(method)
            how.setToolTip(
                "How the samples were drawn. Hand-drawn and autosampled series "
                "have different time resolution, so BIDS writes them to "
                "separate files under the recording entity."
            )
            how.currentTextChanged.connect(
                lambda text, r=row, s=series: self._set_blood_method(r, s, text)
            )
            h.addWidget(how)
            self._blood_methods[series] = how

            pick = QPushButton("Change" if path else "Link")
            pick.setToolTip("Choose the PMOD .bld export for this series")
            pick.clicked.connect(
                lambda _=False, r=row, s=series: self._link_blood(r, s)
            )
            h.addWidget(pick)

            if path:
                # A cross rather than the word: at the width this panel is
                # meant to reach, a second labelled button would not fit.
                clear = QPushButton("\u2715")
                clear.setObjectName("glyph-btn")
                clear.setToolTip("Unlink this curve")
                clear.setFixedWidth(scaled_px(24))
                clear.clicked.connect(
                    lambda _=False, r=row, s=series: self._clear_blood(r, s)
                )
                h.addWidget(clear)
            h.addStretch(1)
            self._body_layout.addWidget(line)

            if path:
                shown = QLabel(Path(path).name)
                shown.setToolTip(path)
                shown.setStyleSheet(
                    f"color: {CUR()['dim']}; font-size: {scaled_px(10)}px; "
                    f"background: transparent; margin-left: {scaled_px(_LABEL_COL + 8)}px;"
                )
                self._body_layout.addWidget(shown)

        note = QLabel(
            "Whether the plasma was dispersion-corrected, how metabolites were "
            "measured and the withdrawal rate are sidecar fields, asked below "
            "with everything else the standard declares."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            f"color: {CUR()['dim']}; font-size: {scaled_px(10)}px; "
            "background: transparent;"
        )
        self._body_layout.addWidget(note)

    def _set_blood(self, row: int, series: str, method: str, path: str) -> None:
        """Put one blood series into the companion list, replacing any there."""
        from ..fixups.blood import blood_role, parse_blood_role

        items = [
            (tag, existing)
            for tag, existing in self._companions(row)
            if (parse_blood_role(tag) or ("", ""))[0] != series
        ]
        if path:
            items.append((blood_role(series, method), path))
        self._write_companions(row, items)

    def _blood_method(self, series: str) -> str:
        combo = getattr(self, "_blood_methods", {}).get(series)
        return combo.currentText() if combo is not None else "manual"

    def _blood_path(self, row: int, series: str) -> str:
        from ..fixups.blood import parse_blood_role

        for tag, path in self._companions(row):
            parsed = parse_blood_role(tag)
            if parsed is not None and parsed[0] == series:
                return path
        return ""

    def _link_blood(self, row: int, series: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Link a blood curve", "",
            "PMOD blood files (*.bld);;All files (*)",
        )
        if not path:
            return
        self._set_blood(row, series, self._blood_method(series), path)

    def _set_blood_method(self, row: int, series: str, method: str) -> None:
        """Re-tag an already-linked curve. Nothing linked, nothing to record."""
        if self._suppress_writeback:
            return
        path = self._blood_path(row, series)
        if path:
            self._set_blood(row, series, method, path)

    def _clear_blood(self, row: int, series: str) -> None:
        self._set_blood(row, series, "manual", "")

    def _companions(self, row: int) -> list[tuple[str, str]]:
        raw = self._cell(row, "companion_files")
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return []
        out: list[tuple[str, str]] = []
        if isinstance(data, list):
            for it in data:
                if isinstance(it, dict) and it.get("suffix") and it.get("path"):
                    out.append((str(it["suffix"]), str(it["path"])))
        return out

    def _write_companions(self, row: int, items: list[tuple[str, str]]) -> None:
        if self._model is None:
            return
        payload = (
            json.dumps([{"suffix": s, "path": p} for s, p in items]) if items else ""
        )
        # dataChanged from bulk_set re-renders this section with the new list.
        self._model.bulk_set([row], "companion_files", payload)

    def _link_companion(self, row: int) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Link a curated companion file", "", "All files (*)",
        )
        if not path:
            return
        items = self._companions(row)
        items.append((self._companion_suffix.currentText(), path))
        self._write_companions(row, items)

    def _remove_companion(self, row: int) -> None:
        """Remove the selected companion.

        Matched by value, not by index: the list on screen hides blood curves,
        so a position in it is not a position in the stored list.
        """
        sel = self._companion_list.currentRow()
        shown = getattr(self, "_plain_companions", [])
        if not (0 <= sel < len(shown)):
            return
        target = shown[sel]
        items = self._companions(row)
        if target in items:
            items.remove(target)
            self._write_companions(row, items)

    def _meta_combo_row(self, label: str, key: str, options: list[str],
                        current: str, blank_label: str, *,
                        setter=None, editable: bool = False,
                        fill_on_open=None) -> QWidget:
        row_w = QWidget()
        # Scope the transparent background to THIS container only (objectName
        # selector) so it does not cascade into child combo popups / tooltips
        # and make them transparent. Those then use the app theme styling.
        row_w.setObjectName("meta-row")
        row_w.setStyleSheet("#meta-row { background: transparent; }")
        h = QHBoxLayout(row_w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        tip = tooltip_for(key)
        lbl = FieldLabel(label, CUR()["dim"], _LABEL_COL, fixed=True)
        if tip:
            lbl.setToolTip(tip)
        h.addWidget(lbl)

        on_change = setter or self._on_meta_field_changed
        combo = QComboBox()
        combo.setObjectName("ent-input")
        if tip:
            combo.setToolTip(tip)
        combo.setEditable(editable)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        combo.addItems(options)
        cur = current.strip()
        if not cur:
            combo.setCurrentText(blank_label)
        else:
            if combo.findText(cur) < 0:
                combo.addItem(cur)
            combo.setCurrentText(cur)
        combo.activated.connect(
            lambda _i, c=combo, k=key, bl=blank_label:
            on_change(k, "" if c.currentText() == bl else c.currentText())
        )
        if editable:
            # An editable combo also commits a free-typed value on focus loss.
            combo.lineEdit().editingFinished.connect(
                lambda c=combo, k=key, bl=blank_label:
                on_change(k, "" if c.currentText() == bl else c.currentText().strip())
            )
        if fill_on_open is not None:
            # An expensive list, filled the first time the user opens the box.
            # Building it on every row selection made selecting a row slow for
            # a list most people never look at.
            from .recording_meta_dialog import _fill_then_show

            combo.showPopup = _fill_then_show(combo, fill_on_open)
        h.addWidget(combo, 1)
        return row_w

    def _meta_edit_row(self, label: str, key: str, current: str, *,
                       setter=None) -> QWidget:
        row_w = QWidget()
        # Scope the transparent background to THIS container only (objectName
        # selector) so it does not cascade into child combo popups / tooltips
        # and make them transparent. Those then use the app theme styling.
        row_w.setObjectName("meta-row")
        row_w.setStyleSheet("#meta-row { background: transparent; }")
        h = QHBoxLayout(row_w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        tip = tooltip_for(key)
        lbl = FieldLabel(label, CUR()["dim"], _LABEL_COL, fixed=True)
        if tip:
            lbl.setToolTip(tip)
        h.addWidget(lbl)
        on_change = setter or self._on_meta_field_changed
        edit = QLineEdit(current)
        edit.setObjectName("ent-input")
        edit.setPlaceholderText("—")
        if tip:
            edit.setToolTip(tip)
        edit.editingFinished.connect(
            lambda e=edit, k=key: on_change(k, e.text().strip())
        )
        h.addWidget(edit, 1)
        return row_w

    def _eff(self, row: int, col: str) -> str:
        """Effective value (per-row override else inherited dataset default)."""
        if self._model is None:
            return ""
        return self._model.effective_value(row, col)

    def _acq_eff(self, row: int, field: str) -> str:
        """Effective per-row acquisition value (scaffold override else default)."""
        if self._model is None:
            return ""
        return self._model.acq_effective(row, field)

    def _cell(self, row: int, col: str) -> str:
        if self._model is None:
            return ""
        df = self._model.dataframe()
        if col not in df.columns or not (0 <= row < len(df)):
            return ""
        v = df.iloc[row][col]
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        s = str(v)
        return "" if s.lower() in ("nan", "none") else s

    def _on_meta_field_changed(self, key: str, value: str) -> None:
        if self._suppress_writeback or self._model is None or self._row is None:
            return
        self._model.bulk_set([self._row], key, value)

    def _on_acq_field_changed(self, key: str, value: str) -> None:
        """Commit a per-row device / institution override into the scaffold spec."""
        if self._suppress_writeback or self._model is None or self._row is None:
            return
        self._model.set_acq_override(self._row, key, value)

    def _on_model_data_changed(self, top_left, bottom_right, _roles=()) -> None:
        if self._model is None or self._row is None:
            return
        if top_left.row() <= self._row <= bottom_right.row():
            # One render for the whole burst: an edit fires dataChanged for
            # every cell it touches (and a bulk edit for every row).
            self._refresh_timer.start()

    def _on_refresh_timer(self) -> None:
        if self._model is None or self._row is None or self._pending:
            return
        # Re-render but suppress the writeback that the rebuilt
        # ``QLineEdit`` widgets would trigger via ``editingFinished``
        # on focus loss during the rebuild.
        self._suppress_writeback = True
        try:
            self.set_selected_row(self._row)
        finally:
            self._suppress_writeback = False


__all__ = ["PropertiesPanel"]
