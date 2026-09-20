"""The form has to be usable, not only correct.

Four things a user asked for after living with it: the curated dropdowns that
had quietly become plain boxes, being told which missing fields the data
normally supplies, the PSD action sitting with the field it answers, and panes
that can be squeezed narrow instead of setting their own floor.
"""

from __future__ import annotations

import pandas as pd
import pytest
from PyQt6.QtWidgets import QComboBox

from bidsmgr.gui.models import InventoryTableModel
from bidsmgr.gui.properties_panel import PropertiesPanel
from bidsmgr.gui.recording_meta_dialog import RecordingMetaDialog
from bidsmgr.gui.widgets.template_form import (
    build_field_widget,
    read_field_widget,
    write_field_widget,
)
from bidsmgr.gui.widgets.validation_pane import ValidationPane
from bidsmgr.metadata.template_plan import (
    ORIGIN_USUALLY_DERIVED,
    sidecar_section,
)
from bidsmgr.recording_meta import CURATED_SUGGESTIONS

pytestmark = pytest.mark.gui


def _field(datatype: str, suffix: str, name: str):
    return {f.name: f for f in sidecar_section(datatype, suffix).fields}[name]


def _any_field(datatype: str, suffix: str, name: str):
    """Including the fields the conversion supplies, which the form still shows."""
    section = sidecar_section(datatype, suffix, include_derived=True)
    return {f.name: f for f in section.fields}[name]


# ---------------------------------------------------------------------------
# Common answers, offered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "datatype,suffix,name",
    [
        ("eeg", "eeg", "EEGReference"),
        ("eeg", "eeg", "EEGGround"),
        ("eeg", "eeg", "Manufacturer"),
        ("anat", "T1w", "PulseSequenceType"),
        ("anat", "T1w", "ParallelAcquisitionTechnique"),
        ("pet", "pet", "ReconFilterType"),
    ],
)
def test_a_field_with_common_answers_offers_them(qtbot, datatype, suffix, name):
    """BIDS leaves these free text, which is not a reason to leave the user a
    blank box. The list is a suggestion; anything can still be typed."""
    field = _field(datatype, suffix, name)
    widget = build_field_widget(field, CURATED_SUGGESTIONS.get(name, ()))
    qtbot.addWidget(widget)
    assert isinstance(widget, QComboBox), f"{name} should offer its common answers"
    assert widget.isEditable(), f"{name} must still accept anything typed"
    assert widget.count() > 1


def test_a_numeric_field_gets_the_list_too(qtbot):
    """The mains is 50 or 60 everywhere on earth. Typing it as a number is what
    BIDS wants, and it came back as the string "60" before."""
    field = _field("eeg", "eeg", "PowerLineFrequency")
    widget = build_field_widget(field, CURATED_SUGGESTIONS["PowerLineFrequency"])
    qtbot.addWidget(widget)
    assert isinstance(widget, QComboBox)
    write_field_widget(widget, 60)
    assert read_field_widget(widget, field) == 60


def test_a_non_numeric_answer_to_an_anyof_field_stays_text(qtbot):
    field = _field("eeg", "eeg", "PowerLineFrequency")
    widget = build_field_widget(field, CURATED_SUGGESTIONS["PowerLineFrequency"])
    qtbot.addWidget(widget)
    write_field_widget(widget, "n/a")
    assert read_field_widget(widget, field) == "n/a"


# ---------------------------------------------------------------------------
# Saying why a field is being asked about
# ---------------------------------------------------------------------------


def test_a_field_the_data_usually_carries_is_marked_as_such():
    """Two different problems wear the same face. A field no converter supplies
    has to be typed. A field a converter USUALLY supplies and did NOT often
    means the value is still at the scanner, or an anonymiser removed it, and
    the user can go and get it rather than invent one.

    The case only arises once the scan has measured this dataset: EEGChannelCount
    is normally answered, so it is asked about only when these files lack it.
    """
    measured = {"SamplingFrequency": 500.0}   # no channel counts in this dataset
    by_name = {
        f.name: f for f in sidecar_section("eeg", "eeg", answered=measured).fields
    }
    assert by_name["EEGChannelCount"].origin == ORIGIN_USUALLY_DERIVED
    assert by_name["EEGReference"].origin != ORIGIN_USUALLY_DERIVED
    assert "SamplingFrequency" not in by_name


# ---------------------------------------------------------------------------
# The PSD action belongs with the field it answers
# ---------------------------------------------------------------------------


def test_compute_psd_sits_with_the_line_frequency(qtbot):
    from PyQt6.QtWidgets import QPushButton

    df = pd.DataFrame([{
        "include": "1", "proposed_datatype": "eeg", "bids_guess_suffix": "eeg",
        "proposed_basename": "sub-001_task-rest_eeg", "source_file": "/raw/a.edf",
        "BIDS_name": "sub-001", "line_freq": "", "montage": "",
        "eeg_reference": "", "eeg_ground": "",
    }])
    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    panel.bind_model(InventoryTableModel(df))
    panel.set_selected_row(0)

    psd = [b for b in panel.findChildren(QPushButton) if "PSD" in (b.text() or "")]
    assert psd, "the PSD action should be somewhere in the panel"
    # Reading the spectrum is how you decide 50 or 60, so it belongs in the same
    # section as the field, not in a block of its own.
    section = psd[0].parent()
    while section is not None and not hasattr(section, "_title_text"):
        section = section.parent()
    assert section is not None and section._title_text == "Sidecar fields"


# ---------------------------------------------------------------------------
# Panes that can be squeezed
# ---------------------------------------------------------------------------


def test_the_metadata_dialog_can_be_squeezed_narrow(qtbot, tmp_path):
    """Long group titles, a picker listing them, and section headers each
    reported their full text as a MINIMUM, so the dialog would not narrow past
    483 px however small its contents could be drawn."""
    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        present_datatypes={"eeg"}, present_pairs=[("eeg", "eeg")],
    )
    qtbot.addWidget(dlg)
    dlg.show()
    assert dlg.minimumSizeHint().width() < 260


@pytest.mark.parametrize("factory", [PropertiesPanel, ValidationPane])
def test_a_side_pane_sets_no_floor_of_its_own(qtbot, factory):
    pane = factory()
    qtbot.addWidget(pane)
    pane.show()
    assert pane.minimumSizeHint().width() <= pane.minimumWidth()


# ---------------------------------------------------------------------------
# The labels themselves
#
# Making the panes shrink first cost the field names entirely: an Ignored size
# policy lets Qt take a label to zero width, so the names vanished and the
# fields drew over where they had been. These pin both halves of the
# requirement, which pull against each other.
# ---------------------------------------------------------------------------


def _expand_all(widget):
    from bidsmgr.gui.widgets.template_form import CollapsibleSection

    for section in widget.findChildren(CollapsibleSection):
        section.set_expanded(True)


def test_every_field_name_is_visible(qtbot, tmp_path):
    from bidsmgr.gui.widgets.template_form import FieldLabel

    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        present_datatypes={"eeg"}, present_pairs=[("eeg", "eeg")],
    )
    qtbot.addWidget(dlg)
    dlg.resize(700, 800)
    dlg.show()
    _expand_all(dlg)
    qtbot.waitUntil(lambda: dlg.isVisible(), timeout=2000)

    labels = [w for w in dlg.findChildren(FieldLabel) if w.isVisible()]
    assert labels, "the form must show its field names"
    assert all(w.width() > 0 for w in labels)
    assert all(w.text() for w in labels)


def test_the_fields_share_one_column(qtbot, tmp_path):
    """Every control in a section starts at the same x, or the form reads as a
    jumble. What matters is where the FIELDS begin: the labels sit left-aligned
    within a shared column, so their own widths differ by design."""
    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        present_datatypes={"eeg"}, present_pairs=[("eeg", "eeg")],
    )
    qtbot.addWidget(dlg)
    dlg.resize(700, 800)
    dlg.show()
    _expand_all(dlg)

    for key in ("dataset_description", "eeg/eeg"):
        # The questions, which share one form. The fields folded away under
        # "already answered" sit in their own, one level deeper, so they start
        # further right by design.
        asked = dlg._template.asked(key)
        widgets = [
            w for name, w in dlg._template.widgets_for(key).items()
            if name in asked and w.isVisible()
        ]
        assert len(widgets) > 5
        lefts = {w.mapTo(dlg, w.rect().topLeft()).x() for w in widgets}
        assert len(lefts) == 1, f"{key}: controls start at {sorted(lefts)}"


def test_a_long_name_elides_and_keeps_the_whole_thing_in_the_tooltip(qtbot):
    """The label is rich text, because the name and its level mark are coloured
    differently, so the ellipsis is inside the markup."""
    from bidsmgr.gui.widgets.template_form import FieldLabel

    label = FieldLabel("ElectricalStimulationParameters:", "#fff", 120)
    qtbot.addWidget(label)
    label.resize(120, 20)
    assert "…" in label.text()
    assert "ElectricalStimulationParameters" not in label.text()
    assert label.toolTip() == "ElectricalStimulationParameters:"


def test_the_level_mark_survives_elision(qtbot):
    """The mark is why the label is coloured at all: it says what the standard
    asks. Eliding name and mark together would drop the mark first."""
    from bidsmgr.gui.widgets.template_form import FieldLabel

    label = FieldLabel("AVeryLongFieldNameIndeed:", "#fff", 90, mark=" *",
                       mark_colour="#f00")
    qtbot.addWidget(label)
    label.resize(90, 20)
    assert "*" in label.text()
    assert "#f00" in label.text()


def test_the_panel_lines_up_across_its_two_halves(qtbot):
    """The hand-built rows used a 76 px column and the schema-driven section
    140, so the two halves of the panel did not line up with each other."""
    from bidsmgr.gui.widgets.template_form import FieldLabel

    df = pd.DataFrame([{
        "include": "1", "proposed_datatype": "eeg", "bids_guess_suffix": "eeg",
        "proposed_basename": "sub-001_task-rest_eeg", "source_file": "/raw/a.edf",
        "BIDS_name": "sub-001", "line_freq": "", "montage": "",
        "eeg_reference": "", "eeg_ground": "",
    }])
    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    panel.bind_model(InventoryTableModel(df))
    panel.set_selected_row(0)
    panel.resize(480, 900)
    panel.show()
    _expand_all(panel)

    labels = [w for w in panel.findChildren(FieldLabel) if w.isVisible()]
    assert len(labels) > 5
    # Hand-built rows have no form column, so each label pins the same width and
    # every control after it starts at the same x.
    assert len({w.width() for w in labels}) == 1


def test_nothing_the_panel_offers_is_missing_from_the_dialog(qtbot, tmp_path):
    """The two surfaces render one definition, so neither may hide a field the
    other shows. The dialog used to DROP what the converter supplies, while the
    panel listed everything, so the mains frequency was in one and not the
    other."""
    from bidsmgr.metadata.template_plan import sidecar_section

    for datatype, suffix in (("eeg", "eeg"), ("meg", "meg"), ("pet", "pet")):
        panel_offers = {
            f.name
            for f in sidecar_section(datatype, suffix, include_derived=True).fields
        }
        dlg = RecordingMetaDialog(
            tmp_path / f"{datatype}.recording_meta.json",
            present_datatypes={datatype}, present_pairs=[(datatype, suffix)],
        )
        qtbot.addWidget(dlg)
        reachable = set(dlg._template.widgets_for(f"{datatype}/{suffix}"))
        assert not (panel_offers - reachable), (
            f"{datatype}/{suffix}: only in the panel: "
            f"{sorted(panel_offers - reachable)}"
        )


def test_the_mains_frequency_is_reachable_for_both_instruments(qtbot, tmp_path):
    for datatype in ("eeg", "meg"):
        dlg = RecordingMetaDialog(
            tmp_path / f"{datatype}.recording_meta.json",
            present_datatypes={datatype}, present_pairs=[(datatype, datatype)],
        )
        qtbot.addWidget(dlg)
        assert "PowerLineFrequency" in dlg._template.widgets_for(f"{datatype}/{datatype}")


def test_a_transparent_background_never_reaches_a_tooltip(qtbot, tmp_path):
    """An unscoped "background: transparent" on a widget cascades into the
    tooltip Qt raises for it, and the tooltip renders see-through. Every such
    rule has to name the widget it means."""
    from PyQt6.QtWidgets import QWidget

    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        present_datatypes={"eeg"}, present_pairs=[("eeg", "eeg")],
    )
    qtbot.addWidget(dlg)
    offenders = []
    for widget in dlg.findChildren(QWidget):
        sheet = widget.styleSheet()
        if "transparent" in sheet and "{" not in sheet:
            offenders.append((type(widget).__name__, sheet[:60]))
    assert not offenders, f"unscoped transparent rules: {offenders}"


# ---------------------------------------------------------------------------
# Building on demand
#
# A four-modality dataset is 559 controls, and Qt spends over a millisecond
# adding each one to a layout, so building them all to show a dozen cost two and
# a half seconds every time the window opened, and again on every theme swap.
# ---------------------------------------------------------------------------


def test_a_section_builds_nothing_until_it_is_opened(qtbot, tmp_path):
    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        present_datatypes={"eeg", "meg"},
        present_pairs=[("eeg", "eeg"), ("meg", "meg")],
    )
    qtbot.addWidget(dlg)
    assert dlg._template._widgets.get("eeg/eeg") in (None, {})

    opened = dlg._template.widgets_for("eeg/eeg")
    assert len(opened) > 10
    # And the other one is still untouched.
    assert dlg._template._widgets.get("meg/meg") in (None, {})


def test_a_section_nobody_opened_keeps_its_answers(qtbot, tmp_path):
    """Its controls were never built, so there is nothing to read back. Saving
    what the widgets hold would erase the answers of every section the user did
    not happen to look at."""
    from bidsmgr.recording_meta import RecordingMetaSpec, dump_spec, load_spec

    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    spec = RecordingMetaSpec()
    spec.sequence_templates["meg/meg"] = {"InstitutionName": "stated earlier"}
    scaffold.write_text(dump_spec(spec))

    dlg = RecordingMetaDialog(
        scaffold, present_datatypes={"eeg", "meg"},
        present_pairs=[("eeg", "eeg"), ("meg", "meg")],
    )
    qtbot.addWidget(dlg)
    dlg._on_save()

    saved = load_spec(scaffold).sequence_templates
    assert saved["meg/meg"]["InstitutionName"] == "stated earlier"


def test_opening_the_dialog_does_not_import_mne(qtbot, tmp_path):
    """The montage list comes from MNE, and asking for it imports mne.channels,
    which takes about a second. Nobody should pay that to open a metadata
    window; the list fills the first time the box is opened."""
    import sys

    for name in [m for m in sys.modules if m.startswith("mne")]:
        del sys.modules[name]

    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        present_datatypes={"eeg"}, present_pairs=[("eeg", "eeg")],
    )
    qtbot.addWidget(dlg)
    assert "mne.channels" not in sys.modules
    assert dlg._montage.count() == 1   # just the "(none)" entry, so far


# ---------------------------------------------------------------------------
# What is already settled, shown first
# ---------------------------------------------------------------------------


def test_the_settled_block_comes_first_and_reads_as_settled(qtbot, tmp_path):
    """Green, and above the questions. Ending a long section on it meant nobody
    saw what they already had before working through what they did not."""
    from bidsmgr.gui.theme_manager import CUR
    from bidsmgr.gui.widgets.template_form import CollapsibleSection

    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        present_datatypes={"eeg"}, present_pairs=[("eeg", "eeg")],
    )
    qtbot.addWidget(dlg)
    dlg.resize(700, 900)
    dlg.show()
    dlg._template.build("eeg/eeg")
    _expand_all(dlg)

    leaf = dlg._template.section_widget("eeg/eeg")
    inner = leaf.findChildren(CollapsibleSection)
    settled = [b for b in inner if b._title_text.startswith("Already answered")]
    assert settled, "the fields the conversion answers should have their own block"
    assert CUR()["success"] in settled[0]._header.styleSheet()

    # Above the questions: its header sits higher than the first field's label.
    asked = dlg._template.asked("eeg/eeg")
    first = next(w for n, w in dlg._template.widgets_for("eeg/eeg").items() if n in asked)
    assert settled[0].mapTo(dlg, settled[0].rect().topLeft()).y() < \
        first.mapTo(dlg, first.rect().topLeft()).y()


def test_the_panel_has_the_settled_block_too(qtbot):
    """A recording answers things by itself, and the per-file form has to say
    which, or the same field looks unanswered here and answered next door."""
    import json as _json

    from bidsmgr.gui.widgets.template_form import CollapsibleSection

    df = pd.DataFrame([{
        "include": "1", "proposed_datatype": "eeg", "bids_guess_suffix": "eeg",
        "proposed_basename": "sub-001_task-rest_eeg", "source_file": "/raw/a.edf",
        "BIDS_name": "sub-001", "task": "rest", "line_freq": "", "montage": "",
        "eeg_reference": "", "eeg_ground": "",
        "_derived_fields": _json.dumps({
            "SamplingFrequency": 500.0, "EEGChannelCount": 64,
        }),
    }])
    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    panel.bind_model(InventoryTableModel(df))
    panel.set_selected_row(0)

    settled = [
        b for b in panel.findChildren(CollapsibleSection)
        if b._title_text.startswith("Already answered")
    ]
    assert settled, "the per-file form needs the settled block as well"


def test_an_entity_the_row_already_carries_counts_as_answered(qtbot):
    """A task label on the row settles TaskName: the converter writes whatever
    the row says, so asking for it again is asking twice."""
    df = pd.DataFrame([{
        "include": "1", "proposed_datatype": "eeg", "bids_guess_suffix": "eeg",
        "proposed_basename": "sub-001_task-rest_eeg", "source_file": "/raw/a.edf",
        "BIDS_name": "sub-001", "task": "rest", "line_freq": "", "montage": "",
        "eeg_reference": "", "eeg_ground": "",
    }])
    model = InventoryTableModel(df)
    assert model.row_answered(0).get("TaskName") == "rest"

    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    panel.bind_model(model)
    panel.set_selected_row(0)
    from bidsmgr.metadata.template_plan import sidecar_section

    asked = {f.name for f in sidecar_section("eeg", "eeg", answered=model.row_answered(0)).fields}
    assert "TaskName" not in asked


# ---------------------------------------------------------------------------
# Numeric arrays survive the round trip
# ---------------------------------------------------------------------------


def test_a_numeric_array_comes_back_as_numbers(qtbot):
    """Typing 0 into ReconMethodParameterValues must reach the sidecar as 0.

    It came back as ``["0"]`` before, which the schema rejects with "items must
    be number": a field the user answered correctly turned into an error.
    """
    field = _field("pet", "pet", "ReconMethodParameterValues")
    widget = build_field_widget(field, ())
    qtbot.addWidget(widget)
    widget.setText("0")
    assert read_field_widget(widget, field) == [0]


def test_a_numeric_array_the_form_only_displayed_is_read_back_unchanged(qtbot):
    """The worse half of the same defect.

    ``FrameTimesStart`` and friends are written by the conversion and only
    SHOWN by the form. Reading them back as strings made them differ from what
    the conversion wrote, so the form restated them in the wrong shape and a
    template meant to add answers took valid ones away. Five fields failed this
    way on a PET scan whose sidecar had been correct.
    """
    for name, value in (
        ("FrameTimesStart", [0]),
        ("FrameDuration", [14400]),
        ("DecayCorrectionFactor", [1.99952]),
        ("ScatterFraction", [1.90358e-08]),
    ):
        field = _any_field("pet", "pet", name)
        widget = build_field_widget(field, ())
        qtbot.addWidget(widget)
        write_field_widget(widget, value)
        assert read_field_widget(widget, field) == value, name


def test_a_multi_frame_array_keeps_every_element(qtbot):
    """A dynamic scan has many frames. The form joins them with ", " to show
    them, so it has to split on the same comma to read them."""
    field = _any_field("pet", "pet", "FrameTimesStart")
    widget = build_field_widget(field, ())
    qtbot.addWidget(widget)
    write_field_widget(widget, [0, 10, 20, 300])
    assert read_field_widget(widget, field) == [0, 10, 20, 300]


def test_an_array_of_strings_is_still_left_alone(qtbot):
    """Only numeric arrays are split. A string list has its own row widget, and
    a value with a comma in it must not be torn in two."""
    field = _field("pet", "pet", "ReconMethodParameterLabels")
    widget = build_field_widget(field, ())
    qtbot.addWidget(widget)
    write_field_widget(widget, ["subsets, iterations"])
    assert read_field_widget(widget, field) == ["subsets, iterations"]


def test_a_vocabulary_combo_follows_its_row_like_every_other_control(qtbot):
    """A schema ``enum`` combo was the one control that kept Qt's own sizing.

    Every other combo the form builds — the suggestion box, the boolean — is put
    through ``_let_it_shrink``, which drops the minimum to zero and makes the
    control expand into its row. The ``field.enum`` branch was missed, so it
    kept the Preferred policy and sized itself to its widest item on first
    show. Inside a folded section that measurement happens while the widget is
    hidden, and ``MRAcquisitionType`` came out 49 px wide: too narrow to draw
    "3D" beside the arrow, so a field the conversion HAD answered read as a
    stunted box with a fragment of a glyph in it.
    """
    from PyQt6.QtWidgets import QSizePolicy

    field = _any_field("anat", "T1w", "MRAcquisitionType")
    assert field.enum, "this test needs a field the schema gives a vocabulary"

    widget = build_field_widget(field, ())
    qtbot.addWidget(widget)

    assert isinstance(widget, QComboBox)
    assert widget.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert widget.minimumWidth() == 0
    assert widget.minimumContentsLength() == 0


def test_every_control_in_the_answered_block_shares_the_field_column(qtbot, tmp_path):
    """The whole block, not one field.

    This is what the user actually reported: values that were in the scaffold
    but unreadable on screen. A control's own ``minimumSizeHint`` cannot catch
    it — the stunted combo agreed with itself that 49 px was enough. What gives
    it away is the row beside it: the block is one ``QFormLayout``, so every
    control in it is handed the same field column, and one that opts out of the
    layout is the bug.
    """
    import json

    from PyQt6.QtWidgets import QLineEdit

    from bidsmgr.gui.widgets.template_form import CollapsibleSection

    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    scaffold.write_text(json.dumps({
        "converter_preview": {
            "anat/T1w": {
                "MRAcquisitionType": "3D",
                "Manufacturer": "Siemens",
                "MagneticFieldStrength": 3,
                "MTState": False,
            },
        },
    }), encoding="utf-8")

    dlg = RecordingMetaDialog(
        scaffold, present_datatypes={"anat"}, present_pairs=[("anat", "T1w")],
    )
    qtbot.addWidget(dlg)
    dlg.resize(800, 900)
    dlg.show()
    for _ in range(6):
        before = len(dlg.findChildren(CollapsibleSection))
        _expand_all(dlg)
        if len(dlg.findChildren(CollapsibleSection)) == before:
            break

    blocks = [
        s for s in dlg.findChildren(CollapsibleSection)
        if "Already answered" in s._title_text
    ]
    assert blocks, "the conversion answered four fields; the block must exist"
    block = blocks[0]

    edits = [w for w in block.findChildren(QLineEdit) if w.isVisible()]
    combos = [c for c in block.findChildren(QComboBox) if c.isVisible()]
    assert edits, "MagneticFieldStrength is a plain box"
    assert combos, "MRAcquisitionType, Manufacturer and MTState are combos"

    # A combo's line edit is a child of the combo, so measure the combo itself.
    column = max(w.width() for w in edits if w.parent() not in combos)
    for combo in combos:
        assert combo.width() == column, (
            f"{combo.currentText()!r} is drawn {combo.width()}px wide in a "
            f"{column}px column, so its value is clipped"
        )
