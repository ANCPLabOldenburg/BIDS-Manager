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

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
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
    QVBoxLayout,
    QWidget,
)

import pandas as pd

from .. import schema as schema_mod
from ..fixups.blood import is_blood_role
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
_MODALITY_NAMES = {
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


def _modality_label(datatype: str) -> str:
    """Display name for a single datatype (``eeg`` -> ``EEG``)."""
    return _MODALITY_NAMES.get(datatype, datatype.upper())


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

    def value(self) -> str:
        return self.edit.text().strip()


class PropertiesPanel(QWidget):
    """Right pane of the Converter view.

    Bind a model with :meth:`bind_model`, then call
    :meth:`set_selected_row` whenever the table's selection changes.
    Passing ``row=None`` blanks the form.
    """

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
        # Raw input root, used to resolve a row's relative ``source_file`` to an
        # absolute recording path for the in-panel PSD compute (set by the
        # ConverterPanel whenever the scanned root changes).
        self._raw_root: Optional[Path] = None
        # Background PSD computation state. Only one runs at a time; the worker
        # is parented to this panel so it survives body rebuilds, and
        # ``_psd_row_id`` identifies which recording is computing so the button
        # renders its busy state even across a re-render.
        self._psd_worker = None
        self._psd_row_id: Optional[str] = None
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
        self._body = QWidget()
        self._body.setObjectName("props-panel")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(14, 12, 14, 12)
        self._body_layout.setSpacing(7)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._body)
        # A scroll area otherwise reports its contents' minimum as its own, so
        # the pane could not be squeezed past whatever the widest card wanted.
        scroll.setMinimumWidth(0)
        scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        outer.addWidget(scroll, 1)

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

    def set_raw_root(self, root: Optional[Path]) -> None:
        """Set the raw input root used to resolve relative ``source_file`` paths.

        The ConverterPanel calls this whenever the scanned root changes so the
        per-row PSD button can locate the recording on disk. Re-renders the
        current row so the button's enabled state reflects path availability.
        """
        self._raw_root = Path(root) if root is not None else None
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

    def set_selected_row(self, row: Optional[int]) -> None:
        """Render the form for ``row`` (or blank if ``None``).

        Called by the Converter view whenever the table selection
        changes. Cheap enough to call on every selection event — the
        body is rebuilt from scratch each time so we don't have to
        track per-row deltas.
        """
        self._row = row
        if self._model is None or row is None:
            self._render_empty()
            self._set_header("Properties · no row selected")
            return
        if not (0 <= row < self._model.rowCount()):
            self._render_empty()
            return
        self._set_header(f"Properties · row {row + 1} selected")
        self._render_for_row(row)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _clear_body(self) -> None:
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
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
            )
            er.edit.editingFinished.connect(
                lambda e=er: self._on_entity_committed(e.entity_name, e.value())
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

        # 4. Row-state notice (from the scanner's proposed_issues) +
        # schema validation. Two distinct sources of "what's wrong with
        # this row": scanner-detected operational issues vs. schema's
        # entity-set verdicts. Both render with the same ValMessage
        # widget so the user sees them as one ranked list.
        self._body_layout.addSpacing(8)
        for vmsg in self._build_row_issue_messages(row):
            self._body_layout.addWidget(vmsg)
        for vmsg in self._build_validation_messages(datatype, suffix, entities):
            self._body_layout.addWidget(vmsg)

        # 5. Per-row metadata, split into two clearly-separated regions:
        # modality-agnostic (participant demographics + companion files, written
        # for any modality incl. MRI) and modality-specific (the recording
        # sidecar, EEG/MEG/iEEG/NIRS only). Sections within each region are
        # labelled by their BIDS destination file.
        self._append_metadata_title()
        self._append_region_label("Modality-agnostic", agnostic=True)
        self._append_participant_section(row)
        self._append_companion_section(row)
        if datatype == "pet":
            self._append_region_label("Modality-specific", agnostic=False)
            self._append_blood_section(row)
        if datatype in _EEG_MEG_DATATYPES:
            self._append_region_label("Modality-specific", agnostic=False)
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
            '#path-preview { font-family: "SF Mono","Menlo","Monaco",monospace; '
            f'font-size: {scaled_px(11)}px; color: {pal["text"]}; '
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
        regions (modality-agnostic vs modality-specific)."""
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
        section applies to (``any modality`` for agnostic sections, or the
        recording's modality such as ``EEG``) so the destination is unambiguous.
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
            "PARTICIPANT", "participants.tsv", agnostic=True, tag="any modality"))
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
            agnostic=False, tag=_modality_label(datatype)))

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

        box = CollapsibleSection(
            "Sidecar fields",
            subtitle=filename,
            badge=f"{len(section.fields)} to answer",
            expanded=self._sidecar_expanded,
        )
        box.toggled_by_user.connect(self._remember_sidecar_expanded)
        self._body_layout.addSpacing(8)
        self._body_layout.addWidget(box)

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
        basename = self._cell(row, "proposed_basename")
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
            self._psd_worker is not None
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
            btn.clicked.connect(lambda _=False, r=row: self._on_compute_psd(r))
        h.addWidget(btn)

        spinner = BusySpinner()
        if is_computing:
            spinner.set_busy(True, message="")
        h.addWidget(spinner)
        h.addStretch(1)
        return row_w

    def _resolve_source_path(self, row: int) -> Optional[Path]:
        """Resolve a row's ``source_file`` to an existing absolute path.

        Mirrors the converter's resolution order: an absolute path as-is, else
        relative to the raw input root, the CWD, or ``.resolve()``. Returns
        ``None`` when nothing on disk matches (DICOM rows have no source_file)."""
        src = self._cell(row, "source_file").strip()
        if not src:
            return None
        p = Path(src)
        if p.is_absolute():
            return p if p.exists() else None
        candidates: list[Path] = []
        if self._raw_root is not None:
            candidates.append(self._raw_root / p)
        candidates.append(Path.cwd() / p)
        try:
            candidates.append(p.resolve())
        except Exception:
            pass
        for c in candidates:
            if c.exists():
                return c
        return None

    def _on_compute_psd(self, row: int) -> None:
        if self._model is None or self._psd_worker is not None:
            return
        path = self._resolve_source_path(row)
        if path is None:
            QMessageBox.warning(
                self, "PSD", "No readable source recording found for this row."
            )
            return
        from ..workers import RecordingComputeWorker

        self._psd_row_id = self._model.row_id(row)

        def _compute(p=path):
            import mne
            import numpy as np

            from .widgets.recording_viewer_pane import _read_raw

            raw = _read_raw(p, preload=True)
            sfreq = float(raw.info["sfreq"])
            fmax = min(sfreq / 2.0, 150.0)
            psd = raw.compute_psd(fmin=0.1, fmax=fmax, verbose=False)
            data = np.asarray(psd.get_data())
            freqs = np.asarray(psd.freqs)
            # compute_psd returns only data channels, in its own order - rows
            # align with psd.ch_names, not the raw channel list. Derive types
            # from the raw info by name so labels/types stay correct.
            names = list(psd.ch_names)
            raw_names = list(raw.ch_names)
            types = []
            for ch in names:
                try:
                    types.append(mne.channel_type(raw.info, raw_names.index(ch)))
                except ValueError:
                    types.append("misc")
            n = min(data.shape[0], len(names), len(types))
            return {
                "freqs": freqs,
                "data": data[:n],
                "ch_names": names[:n],
                "ch_types": types[:n],
            }

        worker = RecordingComputeWorker(_compute, parent=self)
        worker.finished_with_result.connect(self._on_psd_ready)
        worker.failed.connect(self._on_psd_failed)
        worker.finished.connect(worker.deleteLater)
        self._psd_worker = worker
        worker.start()
        # Re-render so the button shows its busy state (spinner + disabled).
        self.set_selected_row(row)

    def _on_psd_ready(self, result) -> None:
        self._psd_worker = None
        self._psd_row_id = None
        from .widgets.recording_viewer_pane import _PsdDialog

        dlg = _PsdDialog(result, parent=self)
        dlg.show()
        if self._row is not None:
            self.set_selected_row(self._row)

    def _on_psd_failed(self, msg) -> None:
        self._psd_worker = None
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
            agnostic=True, tag="any modality"))

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
        h = QHBoxLayout(ctl)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
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
        h.addStretch(1)
        self._body_layout.addWidget(ctl)

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
            # Re-render but suppress the writeback that the rebuilt
            # ``QLineEdit`` widgets would trigger via ``editingFinished``
            # on focus loss during the rebuild.
            self._suppress_writeback = True
            try:
                self.set_selected_row(self._row)
            finally:
                self._suppress_writeback = False


__all__ = ["PropertiesPanel"]
