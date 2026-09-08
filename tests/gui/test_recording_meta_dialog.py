"""GUI smoke tests for the Phase 4 recording-metadata surface:

* the constrained ``montage`` / ``line_freq`` dropdown delegate,
* the new editable per-row columns (reference / ground / demographics),
* the dataset-level :class:`RecordingMetaDialog` scaffold round-trip.

Marked ``gui`` so they run under ``QT_QPA_PLATFORM=offscreen``.
"""

from __future__ import annotations


import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import QComboBox

from bidsmgr.gui.delegates import ChoiceDelegate, builtin_montages
from bidsmgr.gui.models import COLUMNS
from bidsmgr.gui.recording_meta_dialog import RecordingMetaDialog
from bidsmgr.recording_meta import load_spec

pytestmark = pytest.mark.gui


def _spec(key: str):
    return next(c for c in COLUMNS if c.key == key)


def test_new_columns_registered_and_editable():
    for key in ("eeg_reference", "eeg_ground", "Handedness", "montage", "line_freq"):
        assert _spec(key).editable is True
    # Demographics are now editable so EEG/MEG subjects can be filled in-table.
    assert _spec("PatientSex").editable is True
    assert _spec("PatientAge").editable is True


def test_builtin_montages_nonempty():
    assert len(builtin_montages()) > 0


def test_choice_delegate_is_dropdown_only(qtbot):
    d = ChoiceDelegate(["50", "60"], blank_label="(blank)")
    editor = d.createEditor(None, None, None)
    assert isinstance(editor, QComboBox)
    assert editor.isEditable() is False  # never hand-typed
    assert [editor.itemText(i) for i in range(editor.count())] == ["(blank)", "50", "60"]


def test_choice_delegate_blank_maps_to_empty(qtbot):
    d = ChoiceDelegate(["50", "60"], blank_label="(blank)")
    model = QStandardItemModel(1, 1)
    idx = model.index(0, 0)
    model.setData(idx, "60", Qt.ItemDataRole.EditRole)

    editor = QComboBox()
    editor.addItems(["(blank)", "50", "60"])
    editor.setCurrentText("(blank)")
    d.setModelData(editor, model, idx)
    assert model.data(idx, Qt.ItemDataRole.EditRole) == ""  # blank -> empty string

    editor.setCurrentText("50")
    d.setModelData(editor, model, idx)
    assert model.data(idx, Qt.ItemDataRole.EditRole) == "50"


def test_dialog_phenotype_round_trip(qtbot, tmp_path):
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    dlg = RecordingMetaDialog(scaffold)
    qtbot.addWidget(dlg)
    dlg._phenotype.addItem("/data/edinburgh.tsv")
    dlg._phenotype.addItem("/data/bdi.csv")
    assert dlg.build_spec().phenotype_files == ["/data/edinburgh.tsv", "/data/bdi.csv"]
    dlg._on_save()
    assert load_spec(scaffold).phenotype_files == ["/data/edinburgh.tsv", "/data/bdi.csv"]


def test_dialog_participants_file_round_trip(qtbot, tmp_path):
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    dlg = RecordingMetaDialog(scaffold)
    qtbot.addWidget(dlg)
    dlg._participants_file.setText("/data/subjects.csv")
    assert dlg.build_spec().participants_file == "/data/subjects.csv"
    dlg._on_save()
    assert load_spec(scaffold).participants_file == "/data/subjects.csv"


def test_dialog_whole_is_scrollable(qtbot, tmp_path):
    """Structural invariant: exactly ONE scroll area wraps the whole body, and
    the intro label + button box live OUTSIDE it (so the scroll surface is the
    whole metadata body, not an inner widget)."""
    from PyQt6.QtWidgets import QScrollArea
    dlg = RecordingMetaDialog(tmp_path / "inv.tsv.recording_meta.json")
    qtbot.addWidget(dlg)
    areas = dlg.findChildren(QScrollArea)
    assert len(areas) == 1, "the whole body should be one scroll surface"
    # Inner widgets are bounded so they cannot stretch greedily to fill.
    assert 0 < dlg._events.maximumHeight() <= 160
    assert 0 < dlg._phenotype.maximumHeight() <= 110


def test_dialog_montage_suggestion_summary(qtbot, tmp_path):
    """The scan montage suggestions surface in the global dialog as a summary."""
    from PyQt6.QtWidgets import QLabel
    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json", present_datatypes={"eeg"},
        montage_suggestions=["standard_1005 (64/64)", "standard_1020 (19/19)"],
    )
    qtbot.addWidget(dlg)
    hints = [w.text() for w in dlg.findChildren(QLabel)
             if "scan suggests" in w.text() and "standard_1005 (64/64)" in w.text()]
    assert hints




# ---------------------------------------------------------------------------
# The template itself
#
# These replace fifteen tests that asserted widget attributes (dlg._manufacturer
# and friends). Those widgets were written out by hand, which is exactly what
# the template removed, so the tests had to change with them. What follows
# asserts what a user sees: which files are offered, which fields each one asks
# for, at what level, and that answers survive a save.
# ---------------------------------------------------------------------------


def _dialog(tmp_path, **kw):
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    return RecordingMetaDialog(scaffold, **kw), scaffold


def test_the_template_offers_one_section_per_kind_of_file(qtbot, tmp_path):
    """One section per KIND of file the scan says will be written, never for a
    modality the dataset does not contain, and titled as the class it speaks
    for rather than as one file that happens to be of that class."""
    dlg, _ = _dialog(
        tmp_path,
        present_datatypes={"eeg", "meg"},
        present_pairs=[("eeg", "eeg"), ("meg", "meg")],
        example_paths={
            ("eeg", "eeg"): "sub-001/eeg/sub-001_task-rest_eeg.json",
            ("meg", "meg"): "sub-002/meg/sub-002_task-rest_meg.json",
        },
    )
    qtbot.addWidget(dlg)
    keys = {n.key for n in dlg._all_nodes() if n.is_leaf}
    assert keys == {"dataset_description", "eeg/eeg", "meg/meg"}

    labels = {n.key: n.label for n in dlg._all_nodes() if n.is_leaf}
    assert labels["eeg/eeg"] == "every *_eeg.json"
    assert labels["meg/meg"] == "every *_meg.json"
    # The real name still shows, as an example of what the section covers.
    subtitles = {n.key: n.subtitle for n in dlg._all_nodes() if n.is_leaf}
    assert "sub-001_task-rest_eeg.json" in subtitles["eeg/eeg"]


def test_eeg_and_meg_do_not_share_a_section(qtbot, tmp_path):
    """REGRESSION: one box held a single manufacturer for both instruments."""
    dlg, _ = _dialog(
        tmp_path, present_datatypes={"eeg", "meg"},
        present_pairs=[("eeg", "eeg"), ("meg", "meg")],
    )
    qtbot.addWidget(dlg)
    eeg = dlg._template.widgets_for("eeg/eeg")
    meg = dlg._template.widgets_for("meg/meg")
    # A field both instruments have is asked once per instrument, in separate
    # widgets, so an answer for one is never the other's.
    assert "DeviceSerialNumber" in eeg and "DeviceSerialNumber" in meg
    assert eeg["DeviceSerialNumber"] is not meg["DeviceSerialNumber"]
    # And each asks only what its own datatype takes.
    assert "AssociatedEmptyRoom" in meg and "AssociatedEmptyRoom" not in eeg
    assert "CapManufacturer" in eeg and "CapManufacturer" not in meg
    # Both ARE asked about the amplifier. mne-bids writes Manufacturer only for
    # the formats whose header carries it, so a form that never asks leaves the
    # rest of a dataset with nothing. This used to be suppressed because the
    # built-in "the converter fills this" list counted a key mne-bids writes as
    # "n/a" as an answer.
    assert "Manufacturer" in eeg and "Manufacturer" in meg


def test_the_agnostic_section_asks_for_the_dataset_description(qtbot, tmp_path):
    """The fields whose absence produced NO_AUTHORS on every dataset."""
    dlg, _ = _dialog(tmp_path, present_datatypes={"eeg"})
    qtbot.addWidget(dlg)
    fields = dlg._template.widgets_for("dataset_description")
    assert {"Authors", "License", "Funding", "DatasetDOI"} <= set(fields)


def test_levels_come_from_the_schema(qtbot, tmp_path):
    """Which fields are required is the standard's answer at the version in
    use, not a list kept in the dialog."""
    dlg, _ = _dialog(tmp_path, present_datatypes={"eeg"})
    qtbot.addWidget(dlg)
    levels = {
        name: field.level
        for name, field in dlg._template._fields["dataset_description"].items()
    }
    assert levels["Name"] == "required"
    assert levels["License"] == "recommended"
    assert levels["Authors"] in ("optional", "recommended")


def test_a_field_the_converter_fills_is_not_asked(qtbot, tmp_path):
    """dcm2niix reads TracerName out of the DICOM, so asking is noise.

    It is still REACHABLE, folded away under "already answered by the
    conversion", because a field that simply vanished would leave a user who
    knows the scanner got it wrong with nowhere to correct it.
    """
    dlg, _ = _dialog(
        tmp_path, present_datatypes={"pet"}, present_pairs=[("pet", "pet")],
    )
    qtbot.addWidget(dlg)
    asked = dlg._template.asked("pet/pet")
    assert "TracerName" not in asked
    assert "ModeOfAdministration" in asked   # nothing in the data says this
    assert "TracerName" in dlg._template.supplied("pet/pet")
    assert "TracerName" in dlg._template.widgets_for("pet/pet")


def test_answers_round_trip_per_file(qtbot, tmp_path):
    """Each section's answers reach the storage the converter reads, keyed by
    the file they belong to, and come back when the dialog reopens."""
    from bidsmgr.gui.widgets.template_form import read_field_widget, write_field_widget

    kw = dict(
        present_datatypes={"eeg", "meg"},
        present_pairs=[("eeg", "eeg"), ("meg", "meg")],
    )
    dlg, scaffold = _dialog(tmp_path, **kw)
    qtbot.addWidget(dlg)
    write_field_widget(dlg._template.widgets_for("dataset_description")["Authors"],
                       ["Lopez, Karel", "Doe, Jane"])
    write_field_widget(dlg._template.widgets_for("eeg/eeg")["CapManufacturer"], "EasyCap")
    dlg._on_save()

    spec = load_spec(scaffold)
    assert spec.dataset_description.authors == ["Lopez, Karel", "Doe, Jane"]
    assert spec.sequence_templates["eeg/eeg"]["CapManufacturer"] == "EasyCap"

    again = RecordingMetaDialog(scaffold, **kw)
    qtbot.addWidget(again)
    field = again._template._fields["dataset_description"]["Authors"]
    assert read_field_widget(
        again._template.widgets_for("dataset_description")["Authors"], field,
    ) == ["Lopez, Karel", "Doe, Jane"]


def test_an_author_with_a_comma_stays_one_person(qtbot, tmp_path):
    """"Lopez, Karel" is one author. No text separator survives that, which is
    why the field gets a row per person rather than a box."""
    from bidsmgr.gui.widgets.template_form import read_field_widget, write_field_widget

    dlg, _ = _dialog(tmp_path, present_datatypes={"eeg"})
    qtbot.addWidget(dlg)
    widget = dlg._template.widgets_for("dataset_description")["Authors"]
    write_field_widget(widget, ["Lopez, Karel"])
    field = dlg._template._fields["dataset_description"]["Authors"]
    assert read_field_widget(widget, field) == ["Lopez, Karel"]


def test_a_dropdown_popup_is_wide_enough_to_read(qtbot, tmp_path):
    """A combo sized to the layout is narrower than its options, and the popup
    inherits that width, so long values arrived elided."""
    from PyQt6.QtWidgets import QComboBox

    dlg, _ = _dialog(
        tmp_path, present_datatypes={"pet"}, present_pairs=[("pet", "pet")],
    )
    qtbot.addWidget(dlg)
    combos = [
        w for w in dlg._template.widgets_for("pet/pet").values()
        if isinstance(w, QComboBox) and w.count() > 1
    ]
    assert combos, "expected at least one vocabulary dropdown"
    for combo in combos:
        widest = max(
            combo.fontMetrics().horizontalAdvance(combo.itemText(i))
            for i in range(combo.count())
        )
        assert combo.view().minimumWidth() >= widest


def test_the_fields_mne_bids_cannot_answer_are_asked(qtbot, tmp_path) -> None:
    """The regression that started this.

    BIDS requires EEGReference and EEGGround in every EEG sidecar, so mne-bids
    writes them as "n/a" when it does not know. The built-in list of what the
    converter fills counted that as answered, so the form stopped asking for the
    two fields a user most has to supply, and the dataset shipped with "n/a".
    """
    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        present_datatypes={"eeg"}, present_pairs=[("eeg", "eeg")],
    )
    qtbot.addWidget(dlg)
    asked = set(dlg._template.widgets_for("eeg/eeg"))
    for name in ("EEGReference", "EEGGround", "SoftwareFilters", "PowerLineFrequency"):
        assert name in asked, f"{name} must be asked: no converter can answer it"


def test_a_datatype_never_names_its_own_suffix(qtbot, tmp_path) -> None:
    """func has suffixes bold and sbref; there is no _func.json.

    The dialog used to invent ``(datatype, datatype)`` when nobody told it which
    files the scan found. True for eeg and meg by coincidence, so it looked
    fine, while func produced a section titled "sub-_func.json" whose answers
    were filed under a key the writer never reads.
    """
    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        present_datatypes={"func", "anat"},
        present_pairs=[("func", "func"), ("anat", "anat"), ("func", "bold")],
    )
    qtbot.addWidget(dlg)
    keys = {n.key for n in dlg._all_nodes() if n.is_leaf}
    assert "func/bold" in keys
    assert "func/func" not in keys and "anat/anat" not in keys


def test_a_section_speaks_for_every_file_of_its_kind(qtbot, tmp_path) -> None:
    """A section is a class of files, not one file.

    It used to be titled with a real filename borrowed from one subject, which
    reads as a list of individual files and hides the whole point: answer it
    once and every file of that kind gets it.
    """
    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        present_datatypes={"func"}, present_pairs=[("func", "bold")],
        pair_counts={("func", "bold"): 61},
    )
    qtbot.addWidget(dlg)
    node = next(n for n in dlg._all_nodes() if n.key == "func/bold")
    assert node.label == "every *_bold.json"
    assert "61 files" in node.subtitle
    assert "in func/" in node.subtitle


# ---------------------------------------------------------------------------
# What the conversion already answers
#
# The point of this form is what will still be MISSING after conversion. But a
# field the converter fills has to stay reachable: the converter reads it from
# the data, and when the data is wrong there is nowhere else to correct it.
# ---------------------------------------------------------------------------


def test_an_answered_field_is_shown_and_can_be_corrected(qtbot, tmp_path):
    from bidsmgr.gui.widgets.template_form import write_field_widget
    from bidsmgr.recording_meta import RecordingMetaSpec, dump_spec

    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    spec = RecordingMetaSpec()
    spec.converter_preview = {"meg/meg": {"PowerLineFrequency": 50.0}}
    scaffold.write_text(dump_spec(spec))

    dlg = RecordingMetaDialog(
        scaffold, present_datatypes={"meg"}, present_pairs=[("meg", "meg")],
    )
    qtbot.addWidget(dlg)
    widgets = dlg._template.widgets_for("meg/meg")
    assert "PowerLineFrequency" in widgets, (
        "a field the converter fills must still be reachable, or a wrong value "
        "in the data has nowhere to be corrected"
    )

    # Leaving it alone states nothing: storing it would freeze today's value.
    assert "PowerLineFrequency" not in dlg._template.values_by_key().get("meg/meg", {})

    write_field_widget(widgets["PowerLineFrequency"], 60)
    assert dlg._template.values_by_key()["meg/meg"]["PowerLineFrequency"] == 60.0


def test_the_fields_no_backend_can_answer_are_asked_not_hidden(qtbot, tmp_path):
    """mne-bids writes these as "n/a" because BIDS demands the key."""
    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        present_datatypes={"eeg"}, present_pairs=[("eeg", "eeg")],
    )
    qtbot.addWidget(dlg)
    asked = set(dlg._template.widgets_for("eeg/eeg"))
    answered = set(dlg._template._answered_values.get("eeg/eeg", {}))
    for name in ("EEGReference", "EEGGround", "PowerLineFrequency"):
        assert name in asked and name not in answered
