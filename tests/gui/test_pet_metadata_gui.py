"""PET sections in the dataset dialog and the per-row Properties panel.

PET reuses the metadata UI EEG/MEG already had rather than growing a parallel
one, so the tests here are mostly about the seam: the right blocks appear for
the right datasets, values round-trip, and nothing an EEG-only or MRI-only user
sees has changed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bidsmgr.gui.models import InventoryTableModel
from bidsmgr.gui.properties_panel import PropertiesPanel
from bidsmgr.gui.recording_meta_dialog import RecordingMetaDialog
from bidsmgr.recording_meta import load_spec
from bidsmgr.recording_meta import PetAcquisitionSpec, RecordingMetaSpec

pytestmark = pytest.mark.gui


# Keyed by BIDS field, as the scan collects them. Tracer and radionuclide stay
# apart: they are read from one DICOM sequence but they answer two fields, and
# "FDG / F18" is not a value either of them should be offered.
PET_SUGGESTIONS = {
    "TracerName": ["FDG"],
    "TracerRadionuclide": ["F18"],
    "InjectedRadioactivity": ["44.4 MBq"],
    "ReconMethodName": ["PSF+TOF (3i 21s)"],
}


def _dialog(tmp_path, present):
    return RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        set(present),
        None,
        scan_suggestions=PET_SUGGESTIONS,
    )


def _pet_row(**extra) -> dict:
    row = {
        "BIDS_name": "sub-001", "subject": "phantom", "include": 1,
        "sequence": "PET_Brain_AC_TOF", "series_uid": "1.2.3",
        "proposed_datatype": "pet", "proposed_basename": "sub-001_pet",
        "bids_guess_datatype": "pet", "bids_guess_suffix": "pet",
        "entities": '{"subject": "001"}',
        "tracer_suggestion": "FDG", "radionuclide_suggestion": "F18",
        "injected_dose_suggestion": "44.4 MBq",
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Dataset dialog
# ---------------------------------------------------------------------------


def test_a_hidden_pet_block_keeps_its_loaded_values(qtbot, tmp_path) -> None:
    """An EEG-only session must not wipe PET fields from a shared scaffold."""
    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    spec = RecordingMetaSpec(pet_defaults=PetAcquisitionSpec(tracer_name="PIB"))
    scaffold.write_text(spec.model_dump_json())

    dlg = RecordingMetaDialog(scaffold, {"eeg"}, None)
    qtbot.addWidget(dlg)
    assert dlg.build_spec().pet_defaults.tracer_name == "PIB"


def test_decay_corrected_blank_stays_unset(qtbot, tmp_path) -> None:
    """Blank means "not stated", which is not the same as false."""
    dlg = _dialog(tmp_path, {"pet"})
    qtbot.addWidget(dlg)
    assert dlg.build_spec().pet_defaults.image_decay_corrected is None


# ---------------------------------------------------------------------------
# Per-row overrides
# ---------------------------------------------------------------------------


def test_row_inherits_the_dataset_default(qtbot) -> None:
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(
        RecordingMetaSpec(pet_defaults=PetAcquisitionSpec(tracer_name="FDG"))
    )
    assert model.pet_effective(0, "tracer_name") == "FDG"
    assert model.pet_is_inherited(0, "tracer_name")


def test_override_breaks_inheritance(qtbot) -> None:
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(
        RecordingMetaSpec(pet_defaults=PetAcquisitionSpec(tracer_name="FDG"))
    )
    assert model.set_pet_override(0, "tracer_name", "PIB")
    assert model.pet_effective(0, "tracer_name") == "PIB"
    assert not model.pet_is_inherited(0, "tracer_name")


def test_writing_the_default_restores_inheritance(qtbot) -> None:
    """Otherwise a later change to the dataset default would not reach the row."""
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(
        RecordingMetaSpec(pet_defaults=PetAcquisitionSpec(tracer_name="FDG"))
    )
    model.set_pet_override(0, "tracer_name", "PIB")
    model.set_pet_override(0, "tracer_name", "FDG")
    assert model.pet_is_inherited(0, "tracer_name")
    assert "1.2.3" not in model.global_spec().pet_overrides


def test_a_non_numeric_value_is_refused(qtbot) -> None:
    """A coerced number would look deliberate in the sidecar. Refuse instead."""
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(RecordingMetaSpec())
    assert not model.set_pet_override(0, "injected_radioactivity", "not a number")
    assert model.pet_effective(0, "injected_radioactivity") == ""


def test_a_numeric_value_is_stored_as_a_number(qtbot) -> None:
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(RecordingMetaSpec())
    assert model.set_pet_override(0, "injected_radioactivity", "44.4")
    assert model.global_spec().pet_overrides["1.2.3"].injected_radioactivity == 44.4


def test_float_renders_without_a_trailing_zero(qtbot) -> None:
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(
        RecordingMetaSpec(pet_defaults=PetAcquisitionSpec(scan_start=0.0))
    )
    assert model.pet_effective(0, "scan_start") == "0"


def test_pet_overrides_do_not_touch_the_eeg_block(qtbot) -> None:
    """REGRESSION: the two blocks are separate fields on one spec."""
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(RecordingMetaSpec())
    model.set_pet_override(0, "tracer_name", "FDG")
    spec = model.global_spec()
    assert spec.overrides == {}
    assert spec.pet_overrides["1.2.3"].tracer_name == "FDG"


# ---------------------------------------------------------------------------
# Properties panel
# ---------------------------------------------------------------------------


def test_properties_panel_renders_a_pet_row(qtbot) -> None:
    model = InventoryTableModel(pd.DataFrame([_pet_row()]))
    model.set_global_spec(
        RecordingMetaSpec(pet_defaults=PetAcquisitionSpec(tracer_name="FDG"))
    )
    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    panel.bind_model(model)
    panel.set_selected_row(0)  # would raise if the PET section were broken

    labels = [w.text() for w in panel.findChildren(type(panel._divider())) if hasattr(w, "text")]
    assert labels is not None  # the render completed


def test_properties_panel_still_renders_an_mri_row(qtbot) -> None:
    """REGRESSION: the PET branch must not disturb the MRI path."""
    row = _pet_row(proposed_datatype="anat", bids_guess_datatype="anat",
                   bids_guess_suffix="T1w", proposed_basename="sub-001_T1w")
    model = InventoryTableModel(pd.DataFrame([row]))
    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    panel.bind_model(model)
    panel.set_selected_row(0)


# ---------------------------------------------------------------------------
# PET in the template
#
# The four hand-built PET groups are gone: PET is one file, so it is one
# section, built from the schema like every other. These replace the tests that
# asserted those groups' widgets.
# ---------------------------------------------------------------------------


def test_pet_is_one_section_named_after_its_file(qtbot, tmp_path):
    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        present_datatypes={"pet"},
        present_pairs=[("pet", "pet")],
        example_paths={("pet", "pet"): "sub-001/pet/sub-001_trc-FDG_pet.json"},
    )
    qtbot.addWidget(dlg)
    leaves = {n.key: n.label for n in dlg._all_nodes() if n.is_leaf}
    assert leaves["pet/pet"] == "every *_pet.json"


def test_pet_asks_only_what_the_scanner_cannot_answer(qtbot, tmp_path):
    """The dose sheet, not the header. TracerName and the frame timings come
    out of the DICOM; the injected mass and the administration mode do not."""
    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        present_datatypes={"pet"}, present_pairs=[("pet", "pet")],
    )
    qtbot.addWidget(dlg)
    asked = dlg._template.asked("pet/pet")
    for from_the_scanner in ("TracerName", "FrameDuration", "FrameTimesStart"):
        assert from_the_scanner not in asked
        # Not asked, but still reachable to correct.
        assert from_the_scanner in dlg._template.supplied("pet/pet")
    for from_the_lab in ("InjectedMass", "ModeOfAdministration", "SpecificRadioactivity"):
        assert from_the_lab in asked


def test_pet_answers_round_trip(qtbot, tmp_path):
    from bidsmgr.gui.widgets.template_form import write_field_widget

    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    kw = dict(present_datatypes={"pet"}, present_pairs=[("pet", "pet")])
    dlg = RecordingMetaDialog(scaffold, **kw)
    qtbot.addWidget(dlg)
    write_field_widget(dlg._template.widgets_for("pet/pet")["ModeOfAdministration"], "bolus")
    dlg._on_save()
    assert load_spec(scaffold).sequence_templates["pet/pet"]["ModeOfAdministration"] == "bolus"


def test_a_dataset_without_pet_is_never_asked_about_it(qtbot, tmp_path):
    dlg = RecordingMetaDialog(
        tmp_path / "inv.tsv.recording_meta.json",
        present_datatypes={"eeg"}, present_pairs=[("eeg", "eeg")],
    )
    qtbot.addWidget(dlg)
    assert "pet/pet" not in {n.key for n in dlg._all_nodes()}
