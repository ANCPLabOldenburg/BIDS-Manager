"""Acquisition values stated for a datatype land on its recording's file only.

Issue #187 (ANCPLabOldenburg/BIDS-Manager): with ``defaults.manufacturer`` and
``defaults.eeg_reference`` in the recording-metadata file, ``bidsmgr-metadata``
wrote Manufacturer, EEGReference, EEGGround, InstitutionName, CapManufacturer
and more into ``*_events.json`` beside the EEG recording, and into every other
JSON in the folder (channels, electrodes, coordsystem, photo, physio...). The
chain asked whether a field belongs to the DATATYPE's recording, never to the
file being written.

The fix asks both, and these tests pin that it only ever REMOVES: every file
gets what the old rule gave it minus what the standard does not define for
that file, and nothing anywhere is new (the eye tracker's ``beh/*_physio.json``
would otherwise have taken the EEG amplifier's Manufacturer over its own).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr import schema
from bidsmgr.fixups.sidecar_schema import apply_stated_metadata, state_into
from bidsmgr.recording_meta import (
    AcquisitionSpec,
    RecordingMetaSpec,
    resolve_sidecar_fields,
)
from bidsmgr.recording_meta.chain import _ACQ_TO_BIDS

_DEFAULTS = AcquisitionSpec(
    manufacturer="BioSemi", amplifier_model="ActiveTwo", eeg_reference="CMS",
    eeg_ground="DRL", institution_name="University of Oldenburg",
    institution_dept="Psychology", cap_manufacturer="EasyCap", cap_model="actiCAP",
    power_line_freq=50, dewar_position="upright", software_versions="9.1",
    subject_artefact_description="none",
)
_SPEC = RecordingMetaSpec(defaults=_DEFAULTS)
_NAMES = [n for _attr, names in _ACQ_TO_BIDS for n in names]


def _stated(datatype: str, suffix: str, spec: RecordingMetaSpec = _SPEC) -> dict:
    data: dict = {}
    state_into(data, f"sub-01_task-x_{suffix}.json", datatype, suffix, spec)
    return data


@pytest.mark.parametrize("datatype, suffix", [
    ("eeg", "events"), ("eeg", "channels"), ("eeg", "electrodes"), ("eeg", "coordsystem"),
    ("meg", "events"), ("meg", "channels"), ("meg", "coordsystem"), ("meg", "headshape"),
    ("ieeg", "events"), ("ieeg", "electrodes"), ("nirs", "optodes"),
])
def test_files_beside_the_recording_take_none_of_it(datatype, suffix) -> None:
    """The report: an events file is not the recording."""
    assert not set(_stated(datatype, suffix)) & set(_NAMES)


def _old_rule(datatype: str) -> set:
    """What the chain gave a file of ``datatype`` before the fix: for each
    attribute set, the first of its BIDS names the RECORDING takes."""
    out = set()
    for attr, names in _ACQ_TO_BIDS:
        if getattr(_DEFAULTS, attr, None) in (None, "", [], {}):
            continue
        out |= {next((n for n in names if schema.field_applies(n, datatype, datatype)), None)}
    return out - {None}


@pytest.mark.parametrize("datatype", ["eeg", "meg", "ieeg", "func", "anat", "dwi", "pet"])
def test_the_recording_takes_exactly_what_it_took_before(datatype) -> None:
    main = {"func": "bold", "anat": "T1w", "dwi": "dwi", "pet": "pet"}.get(datatype, datatype)
    assert set(resolve_sidecar_fields(_SPEC, datatype, main)) == _old_rule(datatype)
    assert resolve_sidecar_fields(_SPEC, datatype, main)["Manufacturer"].value == "BioSemi"


def test_nothing_is_added_anywhere_and_only_misplaced_fields_go() -> None:
    """For every kind of file the standard defines: what is written now is
    what the old rule wrote, minus the fields this file does not take."""
    for datatype in schema.list_datatypes():
        for suffix in schema.list_suffixes(datatype):
            now = set(resolve_sidecar_fields(_SPEC, datatype, suffix))
            old = {n for n in _NAMES if schema.field_applies(n, datatype, datatype)}
            allowed = old | {"SoftwareVersions", "HardwareFilters", "SoftwareFilters"}
            assert now <= allowed, (datatype, suffix, now - old)
            for name in now:
                assert schema.field_applies(name, datatype, suffix), (datatype, suffix, name)


def test_the_eye_trackers_own_manufacturer_survives() -> None:
    """beh/*_physio.json takes a Manufacturer; the EEG amplifier's is not it."""
    data = {"Manufacturer": "SR-Research", "SamplingFrequency": 500}
    state_into(data, "sub-01_task-rest_recording-eye1_physio.json", "beh", "physio", _SPEC)
    assert data["Manufacturer"] == "SR-Research"


def test_every_layer_follows_the_same_rule() -> None:
    """The datatype block and a recording's own values, not just the dataset's."""
    spec = RecordingMetaSpec(
        datatype_defaults={"meg": AcquisitionSpec(manufacturer="MEGIN", dewar_position="supine")},
        overrides={"r1": AcquisitionSpec(eeg_reference="Cz")})
    assert not resolve_sidecar_fields(spec, "meg", "coordsystem")
    assert resolve_sidecar_fields(spec, "meg", "meg")["DewarPosition"].value == "supine"
    assert not resolve_sidecar_fields(spec, "eeg", "events", row_id="r1")
    assert resolve_sidecar_fields(spec, "eeg", "eeg", row_id="r1")["EEGReference"].value == "Cz"
    cells = {"power_line_freq": "60", "eeg_reference": "FCz"}
    assert not resolve_sidecar_fields(RecordingMetaSpec(), "eeg", "channels", row_values=cells)
    on_eeg = resolve_sidecar_fields(RecordingMetaSpec(), "eeg", "eeg", row_values=cells)
    assert on_eeg["PowerLineFrequency"].value == 60


def test_a_per_file_template_still_reaches_its_file() -> None:
    """The GUI's way of stating values was never affected."""
    spec = RecordingMetaSpec(sequence_templates={
        "eeg/events": {"StimulusPresentation": {"OperatingSystem": "Linux"}},
        "eeg/eeg": {"Manufacturer": "Brain Products"},
    })
    events = _stated("eeg", "events", spec)
    assert events == {"StimulusPresentation": {"OperatingSystem": "Linux"}}
    assert _stated("eeg", "eeg", spec)["Manufacturer"] == "Brain Products"


def test_the_metadata_step_writes_the_recording_alone(tmp_path: Path) -> None:
    """The issue's steps, on disk: only ``_eeg.json`` changes."""
    folder = tmp_path / "sub-01" / "eeg"
    folder.mkdir(parents=True)
    base = "sub-01_task-rest"
    files = {
        f"{base}_eeg.json": {"SamplingFrequency": 512, "EEGReference": "n/a"},
        f"{base}_events.json": {"value": {"Description": "Trigger code"}},
        f"{base}_channels.json": {"name": {"Description": "Channel"}},
        "sub-01_space-CapTrak_coordsystem.json": {"EEGCoordinateSystem": "CapTrak",
                                                   "EEGCoordinateUnits": "mm"},
        "sub-01_space-CapTrak_electrodes.json": {"name": {"Description": "Electrode"}},
    }
    for name, data in files.items():
        (folder / name).write_text(json.dumps(data))
    (tmp_path / "dataset_description.json").write_text(json.dumps(
        {"Name": "t", "BIDSVersion": "1.10.0"}))
    apply_stated_metadata(tmp_path, _SPEC)
    eeg = json.loads((folder / f"{base}_eeg.json").read_text())
    assert eeg["Manufacturer"] == "BioSemi" and eeg["EEGReference"] == "CMS"
    for name, data in files.items():
        if not name.endswith("_eeg.json"):
            assert json.loads((folder / name).read_text()) == data, name
