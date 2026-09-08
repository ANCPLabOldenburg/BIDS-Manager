"""Tests for ``bidsmgr.gui.properties_panel.PropertiesPanel``.

Covers:

* ``InventoryTableModel.set_entity`` / ``set_datatype_suffix`` /
  ``entities()`` / ``datatype_suffix()`` — the API the panel uses.
* The panel's render: form rows for every allowed entity, required
  markers, validation messages, predicted-path preview.
* Bidirectional sync: editing a field in the panel updates the
  model's basename; editing a mirror cell on the model updates the
  panel's form.
* Project integration: Properties-panel edits append
  :class:`UserSetEntity` events.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

from bidsmgr.gui.models import COLUMNS, InventoryTableModel
from bidsmgr.gui.properties_panel import PropertiesPanel
from bidsmgr.project import Project, ScanImported, UserSetCell, UserSetEntity


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Fixture rows
# ---------------------------------------------------------------------------


def _func_row(**overrides) -> dict:
    base = {
        "BIDS_name": "sub-001",
        "session": "ses-pre",
        "include": 1,
        "modality": "mri",
        "proposed_datatype": "func",
        "proposed_basename": "sub-001_ses-pre_task-rest_bold",
        "Proposed BIDS name": "sub-001_ses-pre_task-rest_bold",
        "bids_guess_suffix": "bold",
        "bids_guess_confidence": "0.97",
        "bids_guess_skip": False,
        "proposed_issues": "",
        "entities": json.dumps(
            {"subject": "001", "session": "pre", "task": "rest"},
            sort_keys=True,
        ),
        "task": "rest",
        "run": "",
        "source_file": "",
        "series_uid": "1.2.3.4",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Model API for the panel
# ---------------------------------------------------------------------------


def test_entities_reads_parsed_json() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row()]))
    assert m.entities(0) == {"subject": "001", "session": "pre", "task": "rest"}


def test_entities_returns_empty_for_malformed_json() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row(entities="{not valid")]))
    assert m.entities(0) == {}


def test_datatype_suffix_reads_columns() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row()]))
    assert m.datatype_suffix(0) == ("func", "bold")


def test_set_entity_adds_to_basename() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row()]))
    assert m.set_entity(0, "acquisition", "mprage") is True
    bn = m.dataframe().at[0, "proposed_basename"]
    assert "acq-mprage" in bn


def test_set_entity_removes_when_value_blank() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row()]))
    m.set_entity(0, "acquisition", "mprage")
    assert "acq-mprage" in m.dataframe().at[0, "proposed_basename"]
    m.set_entity(0, "acquisition", "")
    assert "acq-" not in m.dataframe().at[0, "proposed_basename"]


def test_set_entity_subject_updates_bids_name_mirror() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row()]))
    m.set_entity(0, "subject", "042")
    assert m.dataframe().at[0, "BIDS_name"] == "sub-042"
    assert m.dataframe().at[0, "proposed_basename"].startswith("sub-042_")


def test_set_entity_no_op_when_value_unchanged() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row()]))
    assert m.set_entity(0, "task", "rest") is False


def test_set_datatype_suffix_rebuilds() -> None:
    m = InventoryTableModel(pd.DataFrame([_func_row()]))
    m.set_datatype_suffix(0, "anat", "T1w")
    assert m.datatype_suffix(0) == ("anat", "T1w")


def test_set_entity_appends_user_set_entity_event(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    project.append(ScanImported(inventory_tsv="/x", row_ids=("1.2.3.4",)))
    m = InventoryTableModel(pd.DataFrame([_func_row()]), project=project)

    m.set_entity(0, "acquisition", "mprage")

    events = [e for e in project.log if isinstance(e, UserSetEntity)]
    assert len(events) == 1
    assert events[0].entity == "acquisition"
    assert events[0].value == "mprage"
    assert events[0].previous is None  # was absent before


# ---------------------------------------------------------------------------
# PropertiesPanel render
# ---------------------------------------------------------------------------


def _build_panel_with_model(qtbot, row: dict = None) -> tuple[PropertiesPanel, InventoryTableModel]:
    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    m = InventoryTableModel(pd.DataFrame([row or _func_row()]))
    panel.bind_model(m)
    panel.set_selected_row(0)
    return panel, m


def test_panel_renders_entity_rows_for_selected_row(qtbot) -> None:
    panel, _m = _build_panel_with_model(qtbot)
    names = [r.entity_name for r in panel._entity_rows]
    # func/bold's allowed entities, in display order.
    assert names[0] == "subject"
    assert "task" in names
    assert "acquisition" in names
    assert "run" in names


def test_panel_shows_current_entity_values(qtbot) -> None:
    panel, _m = _build_panel_with_model(qtbot)
    by_name = {r.entity_name: r.edit.text() for r in panel._entity_rows}
    assert by_name["subject"] == "001"
    assert by_name["task"] == "rest"
    assert by_name["session"] == "pre"
    assert by_name["acquisition"] == ""


def test_panel_committing_field_updates_model_basename(qtbot) -> None:
    panel, m = _build_panel_with_model(qtbot)
    # Simulate user typing into the acquisition field and tabbing out.
    acq_row = next(r for r in panel._entity_rows if r.entity_name == "acquisition")
    acq_row.edit.setText("mprage")
    panel._on_entity_committed("acquisition", "mprage")
    assert "acq-mprage" in m.dataframe().at[0, "proposed_basename"]


def test_panel_blank_when_no_row_selected(qtbot) -> None:
    panel, _m = _build_panel_with_model(qtbot)
    panel.set_selected_row(None)
    assert panel._entity_rows == []


def test_panel_model_data_changed_rerenders(qtbot) -> None:
    panel, m = _build_panel_with_model(qtbot)
    # Edit the model directly (simulating a table-cell edit on the
    # task column). The panel must reflect the new value after
    # ``dataChanged`` fires.
    task_col = next(i for i, c in enumerate(COLUMNS) if c.key == "task")
    m.setData(m.index(0, task_col), "motor")
    by_name = {r.entity_name: r.edit.text() for r in panel._entity_rows}
    assert by_name["task"] == "motor"


def test_panel_project_integration(qtbot, tmp_path: Path) -> None:
    project = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    project.append(ScanImported(inventory_tsv="/x", row_ids=("1.2.3.4",)))
    m = InventoryTableModel(pd.DataFrame([_func_row()]), project=project)

    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    panel.bind_model(m)
    panel.set_project(project)
    panel.set_selected_row(0)

    panel._on_entity_committed("acquisition", "mprage")
    events = [e for e in project.log if isinstance(e, UserSetEntity)]
    assert any(e.entity == "acquisition" and e.value == "mprage" for e in events)


# ---------------------------------------------------------------------------
# Per-row recording-metadata section (EEG/MEG)
# ---------------------------------------------------------------------------


def _eeg_row(**overrides) -> dict:
    base = {
        "BIDS_name": "sub-001",
        "session": "",
        "include": 1,
        "modality": "eeg",
        "proposed_datatype": "eeg",
        "proposed_basename": "sub-001_task-rest_eeg",
        "Proposed BIDS name": "sub-001_task-rest_eeg",
        "bids_guess_suffix": "eeg",
        "bids_guess_confidence": "1.00",
        "bids_guess_skip": False,
        "proposed_issues": "",
        "entities": json.dumps({"subject": "001", "task": "rest"}, sort_keys=True),
        "task": "rest",
        "run": "",
        "source_file": "sub-001/rec.edf",
        "series_uid": "",
        "montage": "",
        "line_freq": "",
        "eeg_reference": "",
        "eeg_ground": "",
        "PatientSex": "",
        "PatientAge": "",
        "Handedness": "",
    }
    base.update(overrides)
    return base


def test_per_row_meta_writeback_updates_model(qtbot) -> None:
    """The Properties panel's per-recording fields write back to the model."""
    panel, m = _build_panel_with_model(qtbot, _eeg_row())
    panel._on_meta_field_changed("eeg_reference", "Cz")
    panel._on_meta_field_changed("montage", "standard_1005")
    panel._on_meta_field_changed("Handedness", "R")
    panel._on_meta_field_changed("PatientSex", "F")
    df = m.dataframe()
    assert df.at[0, "eeg_reference"] == "Cz"
    assert df.at[0, "montage"] == "standard_1005"
    assert df.at[0, "Handedness"] == "R"
    assert df.at[0, "PatientSex"] == "F"


def test_per_row_meta_section_renders_for_eeg(qtbot) -> None:
    """Selecting an EEG row renders without error (the meta section shows)."""
    panel, _m = _build_panel_with_model(qtbot, _eeg_row())
    # The entity rows still render; the meta section is appended below them.
    names = [r.entity_name for r in panel._entity_rows]
    assert "subject" in names and "task" in names


def test_participant_section_writeback_for_mri(qtbot) -> None:
    """MRI (func) rows now get a per-row Participant section -> participants.tsv."""
    row = _func_row(PatientSex="", PatientAge="", Handedness="")
    panel, m = _build_panel_with_model(qtbot, row)
    panel._on_meta_field_changed("PatientSex", "F")
    panel._on_meta_field_changed("Handedness", "L")
    panel._on_meta_field_changed("PatientAge", "41")
    df = m.dataframe()
    assert df.at[0, "PatientSex"] == "F"
    assert df.at[0, "Handedness"] == "L"
    assert df.at[0, "PatientAge"] == "41"


def test_companion_link_writeback(qtbot) -> None:
    """Linking a companion file writes a JSON list into the row's cell."""
    import json as _json
    row = _eeg_row(companion_files="")
    panel, m = _build_panel_with_model(qtbot, row)
    panel._write_companions(0, [("events", "/data/sub-001_events.tsv")])
    stored = _json.loads(m.dataframe().at[0, "companion_files"])
    assert stored == [{"suffix": "events", "path": "/data/sub-001_events.tsv"}]
    # Round-trips back through the reader.
    assert panel._companions(0) == [("events", "/data/sub-001_events.tsv")]


# ---------------------------------------------------------------------------
# Per-row device / institution overrides (scaffold-backed, not TSV columns)
# ---------------------------------------------------------------------------


def _device_spec(**defaults):
    from bidsmgr.recording_meta import AcquisitionSpec, RecordingMetaSpec
    return RecordingMetaSpec(defaults=AcquisitionSpec(**defaults))


def test_per_row_device_override_writeback(qtbot) -> None:
    """The device fields write into the scaffold's per-row override. Institution
    is agnostic (dataset-level only) and NOT a per-row override."""
    panel, m = _build_panel_with_model(qtbot, _eeg_row())
    m.set_global_spec(_device_spec(manufacturer="Brain Products"))
    panel.set_selected_row(0)
    panel._on_acq_field_changed("manufacturer", "BioSemi")
    panel._on_acq_field_changed("amplifier_model", "actiCHamp")
    over = m.global_spec().overrides["sub-001/rec.edf"]
    assert over.manufacturer == "BioSemi"
    assert over.amplifier_model == "actiCHamp"
    # Not a TSV column.
    assert "manufacturer" not in m.dataframe().columns


def test_per_row_device_field_shows_effective_value(qtbot) -> None:
    """A blank override surfaces the inherited dataset default in the panel."""
    panel, m = _build_panel_with_model(qtbot, _eeg_row())
    m.set_global_spec(_device_spec(manufacturer="Brain Products"))
    panel.set_selected_row(0)
    assert panel._acq_eff(0, "manufacturer") == "Brain Products"


def test_section_headers_color_code_agnostic_vs_modality(qtbot) -> None:
    """Agnostic sections render in a different colour than modality-specific,
    and the modality-specific tag names the recording's actual modality."""
    from bidsmgr.gui.theme_manager import CUR
    panel, m = _build_panel_with_model(qtbot, _eeg_row())
    pal = CUR()
    agnostic = panel._section_header(
        "PARTICIPANT", "participants.tsv", agnostic=True, tag="any modality")
    modality = panel._section_header(
        "ACQUISITION", "sub-..._eeg.json", agnostic=False, tag="EEG")
    assert pal["teal"] in agnostic.text() and "any modality" in agnostic.text()
    assert pal["purple"] in modality.text() and "EEG" in modality.text()


def test_region_labels_split_agnostic_and_specific(qtbot) -> None:
    """An EEG row shows both region dividers (agnostic + modality-specific);
    an MRI row shows only the agnostic region (no recording sidecar)."""
    from PyQt6.QtWidgets import QLabel

    def region_texts(panel):
        return [
            w.text() for w in panel._body.findChildren(QLabel)
            if "MODALITY-AGNOSTIC" in w.text() or "MODALITY-SPECIFIC" in w.text()
        ]

    panel, _m = _build_panel_with_model(qtbot, _eeg_row())
    texts = region_texts(panel)
    assert any("MODALITY-AGNOSTIC" in t for t in texts)
    assert any("MODALITY-SPECIFIC" in t for t in texts)

    panel2, _m2 = _build_panel_with_model(qtbot, _func_row())  # MRI
    texts2 = region_texts(panel2)
    assert any("MODALITY-AGNOSTIC" in t for t in texts2)
    assert not any("MODALITY-SPECIFIC" in t for t in texts2)


def test_montage_match_rate_shown_for_eeg(qtbot) -> None:
    """The scan's montage suggestion (match rate) is surfaced as a read-only
    hint next to the per-row montage field."""
    from PyQt6.QtWidgets import QLabel
    row = _eeg_row(montage_suggestion="standard_1005 (64/64)")
    panel, _m = _build_panel_with_model(qtbot, row)
    hints = [
        w.text() for w in panel._body.findChildren(QLabel)
        if "montage match" in w.text() and "standard_1005 (64/64)" in w.text()
    ]
    assert hints, "montage match-rate hint should be shown"


def test_meg_row_can_state_a_meg_only_field(qtbot) -> None:
    """A MEG row is asked about the dewar; the answer is kept for that row.

    The MEG ACQUISITION box that used to hold this is gone. The field is not:
    it comes from the schema now, with the rest of what a *_meg.json may carry,
    so nothing had to be listed in code for it to be here.
    """
    row = _eeg_row(proposed_datatype="meg", bids_guess_suffix="meg", modality="meg",
                   proposed_basename="sub-001_task-rest_meg", montage_suggestion="")
    panel, m = _build_panel_with_model(qtbot, row)
    from bidsmgr.recording_meta import default_spec
    m.set_global_spec(default_spec()); panel.set_selected_row(0)

    m.set_row_template_field(0, "DewarPosition", "supine")
    assert m.row_template(0) == {"DewarPosition": "supine"}
    assert m.resolved_sidecar(0)["DewarPosition"].origin == "row"


def test_a_row_says_where_an_inherited_value_came_from(qtbot) -> None:
    """A value the row did not state is attributed, not left mysterious."""
    from bidsmgr.recording_meta import AcquisitionSpec, RecordingMetaSpec
    row = _eeg_row(proposed_datatype="meg", bids_guess_suffix="meg", modality="meg",
                   proposed_basename="sub-001_task-rest_meg")
    panel, m = _build_panel_with_model(qtbot, row)
    spec = RecordingMetaSpec()
    spec.modality_defaults["meg"] = AcquisitionSpec(institution_name="Oldenburg")
    m.set_global_spec(spec)
    panel.set_selected_row(0)

    assert m.resolved_sidecar(0)["InstitutionName"].value == "Oldenburg"
    assert m.sidecar_origin(0, "InstitutionName") == "the meg defaults"


def test_echoing_an_inherited_value_stores_nothing(qtbot) -> None:
    """Otherwise today's inherited answer freezes into the row, and a later
    change to the dataset default would skip this one recording."""
    from bidsmgr.recording_meta import AcquisitionSpec, RecordingMetaSpec
    row = _eeg_row(proposed_datatype="meg", bids_guess_suffix="meg", modality="meg")
    panel, m = _build_panel_with_model(qtbot, row)
    spec = RecordingMetaSpec()
    spec.modality_defaults["meg"] = AcquisitionSpec(institution_name="Oldenburg")
    m.set_global_spec(spec)
    panel.set_selected_row(0)

    m.set_row_template_field(0, "InstitutionName", "Oldenburg")
    assert m.row_template(0) == {}


def test_a_field_the_table_carries_is_written_to_the_table(qtbot) -> None:
    """One home per field. PowerLineFrequency is the line_freq column, so
    answering it in the form must not make a second copy in the scaffold."""
    row = _eeg_row()
    panel, m = _build_panel_with_model(qtbot, row)
    m.set_row_template_field(0, "PowerLineFrequency", "50")
    assert str(m.dataframe().iloc[0]["line_freq"]) == "50"
    assert "PowerLineFrequency" not in m.row_template(0)


def test_minimal_metadata_title_present(qtbot) -> None:
    from PyQt6.QtWidgets import QLabel
    panel, _m = _build_panel_with_model(qtbot, _eeg_row())
    texts = [w.text() for w in panel._body.findChildren(QLabel)]
    assert any("MINIMAL METADATA" in t for t in texts)


def test_manufacturer_suggestion_is_offered_not_applied(qtbot) -> None:
    """What the scan read out of this recording's header is offered as a choice.

    It is not filled in: a vendor string can always parse wrongly, and a wrong
    manufacturer is worse than a blank one.
    """
    row = _eeg_row(manufacturer_suggestion="Brain Products")
    panel, m = _build_panel_with_model(qtbot, row)
    assert "Brain Products" in panel._field_suggestions(0, "Manufacturer")
    assert "Manufacturer" not in m.row_template(0)


def test_per_row_has_no_institution_field(qtbot) -> None:
    """Institution is agnostic (dataset-level); it must NOT appear per-row."""
    panel, m = _build_panel_with_model(qtbot, _eeg_row())
    from bidsmgr.recording_meta import default_spec
    m.set_global_spec(default_spec()); panel.set_selected_row(0)
    # institution is not a per-row override field.
    from bidsmgr.gui.models.inventory import _ACQ_OVERRIDE_FIELDS
    assert "institution_name" not in _ACQ_OVERRIDE_FIELDS
    assert "institution_dept" not in _ACQ_OVERRIDE_FIELDS


def test_per_row_fields_have_tooltips(qtbot) -> None:
    """Every metadata input row carries an on-hover explanation."""
    from PyQt6.QtWidgets import QComboBox, QLineEdit
    panel, m = _build_panel_with_model(qtbot, _eeg_row())
    from bidsmgr.recording_meta import default_spec
    m.set_global_spec(default_spec()); panel.set_selected_row(0)
    tipped = sum(1 for w in panel._body.findChildren((QComboBox, QLineEdit)) if w.toolTip())
    assert tipped >= 8  # sex/hand/line_freq/manufacturer/ref/ground/montage/...


def test_nirs_row_has_no_reference_montage_section(qtbot) -> None:
    """NIRS gets the modality-specific ACQUISITION block but NOT the EEG/iEEG
    REFERENCE & MONTAGE block (montage/reference/ground are scalp-EEG concepts,
    matching the global dialog + the enrichment fixup)."""
    from PyQt6.QtWidgets import QLabel
    row = _eeg_row(proposed_datatype="nirs", bids_guess_suffix="nirs", modality="nirs",
                   proposed_basename="sub-001_task-rest_nirs")
    panel, _m = _build_panel_with_model(qtbot, row)
    texts = [w.text() for w in panel._body.findChildren(QLabel)]
    assert any("MODALITY-SPECIFIC" in t for t in texts)   # still modality-specific
    assert any("CONVERSION" in t for t in texts)          # line frequency lives there
    assert not any("montage match" in t for t in texts)   # no montage hint
    # Reference and ground are scalp-EEG concepts; NIRS is never asked.
    assert not any(t.startswith("reference") or t.startswith("ground") for t in texts)


# ---------------------------------------------------------------------------
# Per-row Compute-PSD button (threaded, mirrors the Editor recording viewer)
# ---------------------------------------------------------------------------


def _psd_buttons(panel):
    from PyQt6.QtWidgets import QPushButton
    return [
        b for b in panel._body.findChildren(QPushButton)
        if "Compute PSD" in b.text()
    ]


def test_eeg_row_has_compute_psd_button(qtbot) -> None:
    """An EEG row exposes a Compute-PSD action under the line-frequency field;
    it is disabled until the recording can be located on disk."""
    panel, _m = _build_panel_with_model(qtbot, _eeg_row())
    btns = _psd_buttons(panel)
    assert len(btns) == 1
    # No raw root set and the relative source_file does not exist -> disabled.
    assert not btns[0].isEnabled()


def test_compute_psd_button_enabled_when_source_resolvable(qtbot, tmp_path) -> None:
    rec = tmp_path / "sub-001" / "rec.edf"
    rec.parent.mkdir(parents=True)
    rec.write_bytes(b"\x00")
    panel, _m = _build_panel_with_model(qtbot, _eeg_row())
    panel.set_raw_root(tmp_path)
    btns = _psd_buttons(panel)
    assert len(btns) == 1 and btns[0].isEnabled()
    assert panel._resolve_source_path(0) == rec


def test_meg_and_ieeg_rows_have_psd_button(qtbot) -> None:
    for dt in ("meg", "ieeg"):
        row = _eeg_row(
            proposed_datatype=dt, bids_guess_suffix=dt, modality=dt,
            proposed_basename=f"sub-001_task-rest_{dt}",
        )
        panel, _m = _build_panel_with_model(qtbot, row)
        assert len(_psd_buttons(panel)) == 1, dt


def test_mri_row_has_no_psd_button(qtbot) -> None:
    """The PSD action only appears for the electrophysiology recording section."""
    panel, _m = _build_panel_with_model(qtbot)  # default func/bold MRI row
    assert _psd_buttons(panel) == []


def test_resolve_source_path_none_for_dicom_row(qtbot) -> None:
    panel, _m = _build_panel_with_model(qtbot)  # MRI row has blank source_file
    assert panel._resolve_source_path(0) is None


# ---------------------------------------------------------------------------
# Blood sampling, on a PET row
# ---------------------------------------------------------------------------


def _pet_row(**overrides) -> dict:
    row = _func_row(
        modality="pet",
        proposed_datatype="pet",
        proposed_basename="sub-001_trc-FDG_pet",
        bids_guess_suffix="pet",
        entities=json.dumps({"subject": "001", "tracer": "FDG"}, sort_keys=True),
        task="",
        companion_files="",
    )
    row.update(overrides)
    return row


def test_blood_section_appears_only_on_pet(qtbot) -> None:
    """A blood curve is a PET concept; nothing else should be asked about it."""
    def has_blood(panel: PropertiesPanel) -> bool:
        return any(
            "BLOOD SAMPLING" in w.text()
            for w in panel.findChildren(QLabel)
            if w.text()
        )

    pet_panel, _ = _build_panel_with_model(qtbot, _pet_row())
    assert has_blood(pet_panel)

    func_panel, _ = _build_panel_with_model(qtbot, _func_row())
    assert not has_blood(func_panel)


def test_linking_a_curve_stores_a_tagged_companion(qtbot, tmp_path) -> None:
    """Blood rides the companion list, tagged so the copier leaves it alone."""
    from bidsmgr.fixups.blood import parse_blood_role

    panel, model = _build_panel_with_model(qtbot, _pet_row())
    curve = tmp_path / "plasma.bld"
    curve.write_text("time\tactivity\n", encoding="utf-8")

    panel._set_blood(0, "plasma", "manual", str(curve))

    stored = json.loads(model.dataframe().iloc[0]["companion_files"])
    assert len(stored) == 1
    assert parse_blood_role(stored[0]["suffix"]) == ("plasma", "manual")
    assert stored[0]["path"] == str(curve)


def test_relinking_replaces_rather_than_accumulates(qtbot, tmp_path) -> None:
    """One curve per series. Picking again corrects the choice."""
    panel, model = _build_panel_with_model(qtbot, _pet_row())
    first, second = tmp_path / "a.bld", tmp_path / "b.bld"
    for f in (first, second):
        f.write_text("time\tactivity\n", encoding="utf-8")

    panel._set_blood(0, "plasma", "manual", str(first))
    panel._set_blood(0, "plasma", "automatic", str(second))

    stored = json.loads(model.dataframe().iloc[0]["companion_files"])
    assert len(stored) == 1
    assert stored[0]["suffix"] == "blood:plasma:automatic"
    assert stored[0]["path"] == str(second)


def test_clearing_removes_only_that_series(qtbot, tmp_path) -> None:
    panel, model = _build_panel_with_model(qtbot, _pet_row())
    for series in ("plasma", "wholeblood"):
        curve = tmp_path / f"{series}.bld"
        curve.write_text("time\tactivity\n", encoding="utf-8")
        panel._set_blood(0, series, "manual", str(curve))

    panel._clear_blood(0, "plasma")

    stored = json.loads(model.dataframe().iloc[0]["companion_files"])
    assert [s["suffix"] for s in stored] == ["blood:wholeblood:manual"]


def test_blood_does_not_appear_in_the_plain_companion_list(qtbot, tmp_path) -> None:
    """It has its own section, and it is converted rather than copied."""
    curve = tmp_path / "plasma.bld"
    curve.write_text("time\tactivity\n", encoding="utf-8")
    row = _pet_row(companion_files=json.dumps([
        {"suffix": "blood:plasma:manual", "path": str(curve)},
        {"suffix": "events", "path": "/tmp/events.tsv"},
    ]))
    panel, _ = _build_panel_with_model(qtbot, row)

    assert panel._plain_companions == [("events", "/tmp/events.tsv")]


def test_removing_a_plain_companion_matches_by_value(qtbot, tmp_path) -> None:
    """The regression the hidden rows would otherwise cause.

    The list on screen hides blood, so its row 0 is not the stored row 0.
    Removing by index would delete the blood curve instead.
    """
    curve = tmp_path / "plasma.bld"
    curve.write_text("time\tactivity\n", encoding="utf-8")
    row = _pet_row(companion_files=json.dumps([
        {"suffix": "blood:plasma:manual", "path": str(curve)},
        {"suffix": "events", "path": "/tmp/events.tsv"},
    ]))
    panel, model = _build_panel_with_model(qtbot, row)

    panel._companion_list.setCurrentRow(0)   # "events", the only one shown
    panel._remove_companion(0)

    stored = json.loads(model.dataframe().iloc[0]["companion_files"])
    assert [s["suffix"] for s in stored] == ["blood:plasma:manual"]
