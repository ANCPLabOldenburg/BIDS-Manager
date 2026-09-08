"""Dataset-level recording-metadata editor (the "global changes" surface).

Opened from the inspection footer's "Recording metadata" button, this dialog
edits the dataset-wide enrichment defaults plus the event-code labels, backed by
the same ``<inventory>.tsv.recording_meta.json`` scaffold the scan writes and the
convert verb auto-discovers. Per-row overrides (reference, ground, montage,
line_freq, demographics) live in the inspection table; this is for the values
shared across the dataset.

``montage`` and ``line_freq`` are dropdown-only here too (never hand-typed).
Auxiliary-channel / filter / extras tables are a later addition; this v1 covers
the highest-value fields: device, institution, reference/ground, and the event
map the scan seeded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QSizePolicy,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..metadata.template_plan import build_template_tree
from .widgets.template_form import TemplateTree, fit_popup_to_contents
from ..recording_meta import (
    CURATED_SUGGESTIONS,
    AcquisitionSpec,
    PetAcquisitionSpec,
    RecordingMetaSpec,
    dataset_description_as_bids,
    dataset_description_from_bids,
    resolve_sidecar_fields,
    dump_spec,
    load_spec,
)
from .delegates import builtin_montages
from .metadata_help import tooltip_for
from .app_settings import AppSettings
from .theme_manager import CUR

_NONE = "(none)"
_BLANK = "(blank)"

# Display names + an "EEG and MEG"-style joiner so a section whose field is
# shared by several present modalities is labelled with all of them.
_MODALITY_NAMES = {
    "eeg": "EEG", "meg": "MEG", "ieeg": "iEEG", "nirs": "NIRS", "pet": "PET",
}
_MODALITY_ORDER = ("eeg", "meg", "ieeg", "nirs", "pet")

# Datatypes whose presence makes the modality-specific region worth showing.
_SPECIFIC_DATATYPES = frozenset({"eeg", "meg", "ieeg", "nirs", "pet"})


def _join_modalities(mods: list[str]) -> str:
    names = [_MODALITY_NAMES.get(m, m.upper()) for m in mods]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _tighten_form(form: QFormLayout) -> None:
    """Compact a form layout so the stacked group boxes stay dense.

    Keeps the default window showing as many of the metadata fields as fit
    before the outer scroll area takes over for the rest.
    """
    form.setContentsMargins(8, 6, 8, 6)
    form.setVerticalSpacing(4)
    form.setHorizontalSpacing(8)



def _fill_then_show(combo: QComboBox, items):
    """Populate a combo the first time it is opened, then show it as usual.

    For a list that is expensive to compute and that most users never look at.
    The current text survives, so a stored value shows before the list exists.
    """
    original = combo.showPopup

    def show_popup() -> None:
        if combo.property("filled") is not True:
            combo.setProperty("filled", True)
            current = combo.currentText()
            combo.addItems([n for n in items() if combo.findText(n) < 0])
            if current:
                combo.setCurrentText(current)
        original()

    return show_popup


class RecordingMetaDialog(QDialog):
    """Edit the dataset-level recording-metadata scaffold."""

    def __init__(
        self,
        scaffold_path: Path,
        present_datatypes: Optional[set[str]] = None,
        parent: Optional[QWidget] = None,
        montage_suggestions: Optional[list[str]] = None,
        scan_suggestions: Optional[dict[str, list[str]]] = None,
        example_paths: Optional[dict] = None,
        pair_counts: Optional[dict] = None,
        present_pairs: Optional[list] = None,
        also_ask: Optional[dict] = None,
        bids_root: Optional[Path] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dataset metadata")
        # Distinct per-recording suggestions the scan found, surfaced as read-only
        # hints beside the matching dataset defaults (not auto-applied).
        # Montage is separate because it is the one hint with no BIDS field to
        # attach to: it names an electrode layout to apply, not a value.
        self._montage_suggestions = list(montage_suggestions or [])
        self._scan_suggestions = {
            name: list(values) for name, values in (scan_suggestions or {}).items()
        }
        # A fixed, modest default size. The whole body lives in a scroll area,
        # so when the visible sections need more room the OUTER scroll bar moves
        # the entire window content as one - the user asked for a scrollable
        # window, not a window that resizes itself to the content or whose inner
        # tables stretch. Still freely resizable (enlarge to see more at once).
        self.resize(560, 640)
        self._scaffold_path = Path(scaffold_path)
        # The dataset this template belongs to, so the form can start from the
        # name it was created with and can say what a rename would affect.
        self._bids_root = Path(bids_root) if bids_root is not None else None
        # Datatypes the scanned dataset actually contains. Drives which sections
        # / fields make sense: the recording-acquisition section is hidden for a
        # dataset with no EEG/MEG, and within it the scalp-EEG fields
        # (reference / ground / montage / cap) are hidden for MEG-only. The
        # agnostic sections (events, phenotype) always show.
        self._present = set(present_datatypes or {"eeg", "meg", "ieeg", "nirs"})
        # One real path per (datatype, suffix) from the scan, so a group can
        # name the file its answers reach instead of "sub-..._<datatype>.json".
        self._example_paths = dict(example_paths or {})
        # How many files each section speaks for, so it can say so.
        self._pair_counts = dict(pair_counts or {})
        # The (datatype, suffix) pairs the scan found. Without them the tree
        # falls back to one node per present datatype, which still works but
        # cannot name a real file.
        # No guessing. A datatype does not name its own suffix except by
        # coincidence, and inventing one produced sections for files that cannot
        # exist, filed under keys nothing reads. With no pairs the dialog shows
        # the agnostic section, which is always true of any dataset.
        self._also_ask = dict(also_ask or {})
        self._present_pairs = [
            (datatype, suffix)
            for datatype, suffix in (present_pairs or [])
            if datatype and suffix
        ]
        # Which sections the user folded last time, and whether levels are
        # coloured. Both are read from settings so the window opens the way it
        # was left.
        settings = AppSettings.load()
        self._colour_levels = settings.template_colour_levels
        self._collapsed_keys: set = set()
        self._spec = self._load()

        outer = QVBoxLayout(self)
        outer.setSpacing(6)
        intro = QLabel(
            "What will still be missing after conversion. Each section speaks "
            "for EVERY file of its kind: answer it once and it is written to "
            "all of them. What the conversion reads out of the data is folded "
            "away at the end of each section. For ONE recording, use the table."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)
        # Region 1 - MODALITY-SPECIFIC defaults (the recording sidecars): a
        # device/site block (all electrophysiology) + an EEG/iEEG reference &
        # montage block. The region header is hidden for a dataset with no
        # EEG/MEG (only the agnostic region then shows).
        # One acquisition group per modality, not one shared between them: an
        # EEG amplifier and a MEG dewar are different instruments, and a study
        # running both could previously state only one manufacturer, one model
        # and one mains frequency for the pair.
        # THE TEMPLATE, built from the schema and the scan rather than written
        # out here. One node per file, agnostic first, each collapsible, each
        # naming the file it writes in full. Replaces the hand-built groups for
        # the acquisition system, the EEG reference block, the MEG block, the
        # four PET blocks and the institution block, which between them named
        # fields in code and could not follow a change of BIDS version.
        self._tree_nodes = build_template_tree(
            self._present_pairs, self._example_paths,
            counts=self._pair_counts,
            also_ask=self._also_ask,
            # Do not ask for what the scan saw the conversion answer on THIS
            # dataset. The measured default list came from one tree with one
            # scanner; this came from theirs.
            answered=dict(self._spec.converter_preview or {}),
        )
        self._template = TemplateTree(
            self._tree_nodes,
            values=self._stored_template_values(),
            suggestions=self._field_suggestions(),
            # What the scan learned the conversion will fill in by itself.
            answered=dict(self._spec.converter_preview or {}),
            colour_levels=self._colour_levels,
            collapsed_keys=self._collapsed_keys,
        )
        bl.addWidget(self._template)

        # What the standard has no field for, and so cannot come from it: which
        # montage to apply on conversion, how trigger codes read, and the two
        # spreadsheets that feed participants and phenotype.
        bl.addWidget(self._region_label("BIDS Manager settings", agnostic=True))
        self._eeg_box = self._build_eeg_group()
        bl.addWidget(self._eeg_box)
        bl.addWidget(self._build_event_group())
        bl.addWidget(self._build_participants_group())
        bl.addWidget(self._build_phenotype_group())
        bl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)

        # Jump straight to a section. The layout is unchanged, so everything is
        # still there to scroll through if you prefer, but a dataset with four
        # modalities makes a long window and hunting for the PET reconstruction
        # box by dragging is not a good use of anyone's time.
        self._section_picker = QComboBox()
        self._section_picker.setToolTip("Jump to a section")
        # Its items are section titles, and a combo reports its widest item as
        # its minimum width, so the picker alone set the dialog's floor. The
        # popup is widened instead, where the width is what makes it readable.
        fit_popup_to_contents(self._section_picker)
        self._section_picker.setMinimumContentsLength(0)
        self._section_picker.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._section_picker.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        picker_row = QWidget()
        picker_layout = QHBoxLayout(picker_row)
        picker_layout.setContentsMargins(0, 0, 0, 0)
        picker_layout.setSpacing(6)
        picker_layout.addWidget(QLabel("Go to:"))
        picker_layout.addWidget(self._section_picker, 1)
        outer.addWidget(picker_row)

        # Every destination, in the order they appear: the template's own file
        # nodes first, then the settings BIDS has no field for. Choosing one
        # unfolds it and scrolls to it, so a long window never has to be dragged.
        self._jump_targets: list[tuple[str, object]] = []
        for node in self._all_nodes():
            if node.is_leaf:
                section = self._template.section_widget(node.key)
                if section is not None:
                    self._jump_targets.append((node.label, section))
        for box in body.findChildren(QGroupBox):
            if box.title():
                self._jump_targets.append((box.title(), box))
        for label, _widget in self._jump_targets:
            self._section_picker.addItem(label)
        fit_popup_to_contents(self._section_picker)

        def jump(index: int) -> None:
            if not (0 <= index < len(self._jump_targets)):
                return
            _label, widget = self._jump_targets[index]
            if hasattr(widget, "set_expanded"):
                widget.set_expanded(True)
            self._template.reveal(widget)
            scroll.ensureWidgetVisible(widget, 0, 0)

        self._section_picker.activated.connect(jump)
        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._populate()
        self._apply_modality_constraints()

    def _region_label(self, text: str, *, agnostic: bool) -> QLabel:
        """A bold, colour-coded region divider (agnostic = teal, modality-
        specific = purple) separating the two metadata regions."""
        pal = CUR()
        color = pal["teal"] if agnostic else pal["purple"]
        lbl = QLabel(
            f'<span style="color:{color};font-weight:800;'
            f'letter-spacing:0.6px;">{text.upper()}</span>'
        )
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setObjectName("region-rule")
        lbl.setStyleSheet(
            f"#region-rule {{ background: transparent; "
            f"border-bottom: 1px solid {color}; padding-bottom: 2px; }}"
        )
        return lbl

    def _apply_modality_constraints(self) -> None:
        """Show/hide modality-specific sections by what was scanned.

        A section is shown only when its modality was actually scanned: the
        EEG/MEG device block for electrophysiology, the montage and reference
        block only with scalp EEG or iEEG, the MEG block only with MEG, and the
        four PET blocks only with PET. The region header hides when none apply,
        which is what an MRI-only dataset sees.
        """
        has_eeg = bool(self._present & {"eeg", "ieeg"})
        # Flags consulted by build_spec so a section that is not shown keeps its
        # loaded values rather than being cleared by widgets nobody saw.
        self._eeg_applies = has_eeg
        # The template itself decides what to show: a section exists only for a
        # file the scan says will be written. Only the montage block, which BIDS
        # has no field for, still needs hiding by hand.
        self._eeg_box.setVisible(has_eeg)

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def _field_suggestions(self) -> dict:
        """Values we offer that the standard does not list.

        Two kinds, both suggestions and neither a restriction: vocabularies
        BIDS Manager curates because BIDS leaves the field free, and what the
        scan actually detected in THIS dataset, which is the better hint of the
        two and the reason it is offered rather than applied.
        """
        # What this scan detected comes first: it is about the dataset in front
        # of the user rather than about the world.
        return {
            name: list(self._scan_suggestions.get(name, ()))
            + [v for v in CURATED_SUGGESTIONS.get(name, ())
               if v not in self._scan_suggestions.get(name, ())]
            for name in set(self._scan_suggestions) | set(CURATED_SUGGESTIONS)
        }

    def _stored_template_values(self) -> dict:
        """What the scaffold already holds, keyed as the tree's nodes are.

        A file's section starts from what the DATASET already states, so a site
        stated once shows in every file that takes it rather than as an empty
        box inviting the user to state it again. The chain decides which fields
        a datatype accepts, so this cannot put an EEG reference on an MRI scan.
        """
        values: dict = {}
        for datatype, suffix in self._present_pairs:
            inherited = resolve_sidecar_fields(self._spec, datatype, suffix)
            if inherited:
                values[f"{datatype}/{suffix}"] = {
                    name: field.value for name, field in inherited.items()
                }
        for key, stated in (self._spec.sequence_templates or {}).items():
            values.setdefault(key, {}).update(stated)
        values["dataset_description"] = dataset_description_as_bids(
            self._spec.dataset_description
        )
        return values

    def _read_template(self, spec) -> None:
        """Put the tree's answers back where the converter reads them.

        Nothing new is invented to store them: the agnostic answers go to the
        dataset_description block, and a file's answers to the sequence template
        keyed exactly as its node is.
        """
        answers = self._template.values_by_key()

        dataset_description_from_bids(
            spec.dataset_description, answers.pop("dataset_description", {}),
        )

        # Templates the tree did not show (another modality's, from a shared
        # scaffold) are kept: the dialog only speaks for what it displayed.
        shown = {n.key for n in self._all_nodes() if n.is_leaf}
        kept = {
            key: value for key, value in (spec.sequence_templates or {}).items()
            if key not in shown
        }
        kept.update(answers)
        spec.sequence_templates = kept

    def _all_nodes(self):
        for root in self._tree_nodes:
            yield from root.walk()

    def _form_row(self, form: QFormLayout, label_text: str, widget, ui_key: str) -> None:
        """Add a labelled field row with the schema tooltip on BOTH the label
        and the field (so hovering either shows the explanation)."""
        tip = tooltip_for(ui_key)
        label = QLabel(label_text)
        if tip:
            label.setToolTip(tip)
            widget.setToolTip(tip)
        form.addRow(label, widget)

    def _suggestion_label(self, values: list[str]) -> Optional[QLabel]:
        """A read-only summary of what the scan detected, never applied.

        A vendor string can always parse wrongly, so a suggestion is shown
        beside the field and left for the user to accept.
        """
        values = [v for v in values if v]
        if not values:
            return None
        shown = ", ".join(sorted(set(values))[:4])
        label = QLabel(f"scan suggests: {shown}")
        label.setWordWrap(True)
        label.setStyleSheet(
            f"color:{CUR()['muted']}; background: transparent; font-style: italic;"
        )
        return label

    def _build_eeg_group(self) -> QGroupBox:
        """The montage, which BIDS has no field for.

        Everything else that used to live here (reference, ground, cap) IS a
        BIDS field and is asked once, in the section for the file it lands in.
        A montage is our own notion: which standard electrode layout mne-bids
        should apply during conversion, which the standard has no opinion about.
        """
        box = QGroupBox("Montage")
        box.setToolTip(
            "EEG and iEEG. Applied during conversion: it fills electrodes.tsv and coordsystem.json."
        )
        form = QFormLayout(box)
        _tighten_form(form)

        # The list of built-in montages comes from MNE, and asking for it
        # imports mne.channels, which takes about a second. Nobody should pay
        # that to open a metadata window, so the list is filled the first time
        # the box is actually opened.
        self._montage = QComboBox()
        self._montage.addItem(_NONE)
        self._montage.showPopup = _fill_then_show(self._montage, builtin_montages)
        self._form_row(form, "Montage:", self._montage, "montage")
        hint = self._suggestion_label(self._montage_suggestions)
        if hint is not None:
            form.addRow("", hint)
        return box

    def _build_event_group(self) -> QGroupBox:
        box = QGroupBox("Events")
        box.setToolTip(
            "Give each recorded trigger code a readable label. Any modality. Written to events.tsv."
        )
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)
        self._events = QTableWidget(0, 2)
        self._events.setHorizontalHeaderLabels(["Code", "Label"])
        self._events.horizontalHeader().setStretchLastSection(True)
        # Bound the table to a natural, modest height. It keeps its own size
        # (it does not stretch to fill); when the stacked sections together
        # exceed the window, the OUTER scroll area moves the whole body.
        self._events.setMinimumHeight(96)
        self._events.setMaximumHeight(160)
        v.addWidget(self._events)

        row = QHBoxLayout()
        add = QPushButton("Add row")
        add.clicked.connect(lambda: self._events.insertRow(self._events.rowCount()))
        rem = QPushButton("Remove selected")
        rem.clicked.connect(self._remove_selected_event)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        v.addLayout(row)
        return box

    def _remove_selected_event(self) -> None:
        rows = sorted({i.row() for i in self._events.selectedIndexes()}, reverse=True)
        for r in rows:
            self._events.removeRow(r)

    def _build_participants_group(self) -> QGroupBox:
        box = QGroupBox("Participants spreadsheet")
        box.setToolTip(
            "Any modality. Its columns are merged into participants.tsv."
        )
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)
        hint = QLabel(
            "Optional table keyed by participant_id. Its age / sex / handedness "
            "override the inventory; any extra columns (group, IQ, ...) are added "
            "to participants.tsv. A sibling <name>.json supplies column "
            "descriptions / Levels / Units."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)
        row = QHBoxLayout()
        self._participants_file = QLineEdit()
        self._participants_file.setObjectName("ent-input")
        self._participants_file.setReadOnly(True)
        self._participants_file.setPlaceholderText("(none)")
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._choose_participants_file)
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self._participants_file.clear())
        row.addWidget(self._participants_file, 1)
        row.addWidget(browse)
        row.addWidget(clear)
        v.addLayout(row)
        return box

    def _choose_participants_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select participants spreadsheet", "",
            "Tables (*.tsv *.csv *.xlsx *.ods);;All files (*)",
        )
        if path:
            self._participants_file.setText(path)

    def _build_phenotype_group(self) -> QGroupBox:
        box = QGroupBox("Phenotype tables")
        box.setToolTip(
            "Any modality. Written to phenotype/ with a codebook beside each table."
        )
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)
        hint = QLabel(
            "Measure tables keyed by participant_id -> phenotype/<measure>.tsv + .json."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)
        self._phenotype = QListWidget()
        self._phenotype.setMinimumHeight(70)
        self._phenotype.setMaximumHeight(110)
        v.addWidget(self._phenotype)
        row = QHBoxLayout()
        add = QPushButton("Add file…")
        add.clicked.connect(self._add_phenotype_file)
        rem = QPushButton("Remove selected")
        rem.clicked.connect(self._remove_phenotype)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        v.addLayout(row)
        return box

    def _add_phenotype_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select phenotype table(s)", "",
            "Tables (*.tsv *.csv *.xlsx *.ods);;All files (*)",
        )
        existing = {self._phenotype.item(i).text() for i in range(self._phenotype.count())}
        for p in paths:
            if p and p not in existing:
                self._phenotype.addItem(p)

    def _remove_phenotype(self) -> None:
        for item in self._phenotype.selectedItems():
            self._phenotype.takeItem(self._phenotype.row(item))

    # ------------------------------------------------------------------
    # load / populate / save
    # ------------------------------------------------------------------

    def _load(self) -> RecordingMetaSpec:
        spec = RecordingMetaSpec()
        if self._scaffold_path.exists():
            try:
                spec = load_spec(self._scaffold_path)
            except Exception:
                pass
        if not spec.dataset_description.name:
            # The name the dataset was created with. It is already in
            # dataset_description.json, and the form is where a user expects to
            # see it, so it starts there rather than as an empty box that then
            # writes a blank over a name they chose.
            spec.dataset_description.name = self._dataset_name_on_disk()
        return spec

    def _dataset_name_on_disk(self) -> str:
        """The Name the dataset currently carries, if there is one."""
        root = self._bids_root
        if root is None:
            return ""
        try:
            data = json.loads(
                (root / "dataset_description.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return ""
        return str(data.get("Name", "") or "").strip() if isinstance(data, dict) else ""

    def _populate(self) -> None:
        """Fill the parts BIDS has no field for. The rest is the tree's.

        Every sidecar field now lives in a template section, filled from the
        scaffold by :class:`TemplateTree` itself, so the only things left to
        restore here are the montage, the event map, and the two spreadsheets.
        """
        self._set_combo(self._montage, self._spec.defaults.montage, _NONE)

        event_map = self._spec.event_maps.get("*", {})
        self._events.setRowCount(0)
        for code, label in event_map.items():
            r = self._events.rowCount()
            self._events.insertRow(r)
            self._events.setItem(r, 0, QTableWidgetItem(str(code)))
            self._events.setItem(r, 1, QTableWidgetItem(str(label)))

        self._phenotype.clear()
        for p in self._spec.phenotype_files:
            self._phenotype.addItem(str(p))

        self._participants_file.setText(self._spec.participants_file or "")

    @staticmethod
    def _num_text(value) -> str:
        """Render a number for a line edit without a trailing ``.0``."""
        if value is None:
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def _build_pet_spec(self) -> PetAcquisitionSpec:
        """Keep whatever the scaffold carried for PET.

        The PET fields are asked once, in the section for the pet/pet file, and
        stored as a sequence template like every other file's. This block is no
        longer edited here, so it is preserved rather than rebuilt.
        """
        return self._spec.pet_defaults.model_copy(deep=True)

    @staticmethod
    def _set_combo(combo: QComboBox, value: Optional[str], blank: str) -> None:
        value = (value or "").strip()
        if not value:
            combo.setCurrentText(blank)
            return
        pos = combo.findText(value)
        if pos < 0:
            combo.addItem(value)
            pos = combo.findText(value)
        combo.setCurrentIndex(pos)

    @staticmethod
    def _combo_value(combo: QComboBox, blank: str) -> Optional[str]:
        text = combo.currentText()
        return None if text == blank else text

    def build_spec(self) -> RecordingMetaSpec:
        """Assemble a :class:`RecordingMetaSpec` from the current form state.

        Sections hidden for the scanned modality keep their loaded values
        (read from ``self._spec``) rather than reading the empty, never-shown
        widgets - so a MEG-only or MRI-only dataset never wipes EEG fields a
        shared scaffold may carry.
        """
        def _opt(edit: QLineEdit) -> Optional[str]:
            t = edit.text().strip()
            return t or None

        prev = self._spec.defaults

        # Every sidecar field is now asked once, in the section for the file it
        # lands in, and stored by _read_template. What survives here is the one
        # value the standard has no field for, the montage, plus whatever the
        # scaffold already carried, which this dialog does not speak for.
        montage = (
            self._combo_value(self._montage, _NONE)
            if self._eeg_applies else prev.montage
        )
        institution_name = prev.institution_name
        institution_dept = prev.institution_dept
        manufacturer = prev.manufacturer
        amplifier_model = prev.amplifier_model
        software_versions = prev.software_versions
        power_line_freq = prev.power_line_freq
        eeg_ref = prev.eeg_reference
        eeg_gnd = prev.eeg_ground
        cap = prev.cap_manufacturer
        dewar_position = prev.dewar_position
        associated_empty_room = prev.associated_empty_room
        subject_artefact_description = prev.subject_artefact_description

        acq = AcquisitionSpec(
            manufacturer=manufacturer,
            amplifier_model=amplifier_model,
            software_versions=software_versions,
            power_line_freq=power_line_freq,
            montage=montage,
            eeg_reference=eeg_ref,
            eeg_ground=eeg_gnd,
            cap_manufacturer=cap,
            institution_name=institution_name,
            institution_dept=institution_dept,
            dewar_position=dewar_position,
            associated_empty_room=associated_empty_room,
            subject_artefact_description=subject_artefact_description,
            software=self._spec.defaults.software,
            cap_model=self._spec.defaults.cap_model,
            aux_channels=self._spec.defaults.aux_channels,
            filters=self._spec.defaults.filters,
            extras=self._spec.defaults.extras,
        )

        event_map: dict[str, str] = {}
        for r in range(self._events.rowCount()):
            code_item = self._events.item(r, 0)
            label_item = self._events.item(r, 1)
            code = code_item.text().strip() if code_item else ""
            label = label_item.text().strip() if label_item else ""
            if code:
                event_map[code] = label

        event_maps = dict(self._spec.event_maps)
        if event_map:
            event_maps["*"] = event_map
        else:
            event_maps.pop("*", None)

        phenotype_files = [
            self._phenotype.item(i).text() for i in range(self._phenotype.count())
        ]

        return self._spec.model_copy(update={
            "defaults": acq,
            "pet_defaults": self._build_pet_spec(),
            "event_maps": event_maps,
            "phenotype_files": phenotype_files,
            "participants_file": self._participants_file.text().strip(),
        })

    #: Set when saving a name means renaming the project folder. The caller
    #: does the move: this dialog knows what was asked, not how to close and
    #: reopen a project.
    rename_project_to: Optional[str] = None

    def _on_save(self) -> None:
        spec = self.build_spec()
        self._read_template(spec)

        if not self._settle_the_name(spec):
            return

        self._scaffold_path.parent.mkdir(parents=True, exist_ok=True)
        self._scaffold_path.write_text(dump_spec(spec), encoding="utf-8")
        self.accept()

    def _settle_the_name(self, spec) -> bool:
        """Ask what a changed dataset name means. False if the user backed out.

        Two intentions wear the same edit. A folder named in a hurry gets its
        real name later, which is a rename; or the folder name is the one
        everybody uses and what is being typed is the dataset's title. Guessing
        was the old behaviour and it guessed wrong in a way nothing announced.
        """
        from .rename_dataset_dialog import (
            RENAME_ALL, ask_what_the_name_means, can_rename_to,
        )

        self.rename_project_to = None
        stated = (spec.dataset_description.name or "").strip()
        root = self._bids_root
        if root is None or not stated or stated == root.name:
            return True
        if stated == self._dataset_name_on_disk():
            # Unchanged since it was last settled; do not ask again.
            return True

        answer = ask_what_the_name_means(
            root.name, stated, self, can_rename=can_rename_to(root, stated),
        )
        if answer is None:
            return False
        if answer == RENAME_ALL:
            self.rename_project_to = stated
        return True


__all__ = ["RecordingMetaDialog"]
