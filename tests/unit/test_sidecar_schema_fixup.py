"""Tests for the schema-driven, modality-agnostic sidecar repairs.

These assert against what the BIDS schema actually declares rather than against
a hardcoded expectation, because the whole point of the fixup is that it stops
being a hand-written list.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from bidsmgr.fixups.sidecar_schema import (
    fill_agnostic_fields,
    repair_array_types,
    repair_key_names,
    apply_stated_metadata,
    repair_converter_output,
)
from bidsmgr.recording_meta import RecordingMetaSpec


# ---------------------------------------------------------------------------
# Key casing
# ---------------------------------------------------------------------------


def test_eeg_misc_channel_count_is_respelled() -> None:
    """THE case this fixup was written for.

    BIDS spells the field MISCChannelCount for EEG and MiscChannelCount for MEG
    and iEEG. mne-bids writes the MEG spelling into EEG sidecars, so every EEG
    dataset carried a key the standard does not declare while the declared one
    read as missing. The value was never wrong, only its name.
    """
    data = {"MiscChannelCount": 0, "EEGChannelCount": 65}
    assert repair_key_names(data, "eeg", "eeg") == 1
    assert data["MISCChannelCount"] == 0
    assert "MiscChannelCount" not in data
    assert data["EEGChannelCount"] == 65


def test_meg_keeps_its_own_spelling() -> None:
    """The other half, and the reason this cannot be a blanket rename: the MEG
    spelling is correct FOR MEG. A fixup that 'corrected' it would break the
    datatype it came from."""
    data = {"MiscChannelCount": 2}
    assert repair_key_names(data, "meg", "meg") == 0
    assert data == {"MiscChannelCount": 2}


def test_a_key_the_schema_does_not_declare_is_left_alone() -> None:
    """Renaming is driven by the schema, so a private key keeps its name and
    its value rather than being guessed at."""
    data = {"MyLabsOwnField": 1, "bidsguess": "x"}
    assert repair_key_names(data, "anat", "T1w") == 0
    assert data == {"MyLabsOwnField": 1, "bidsguess": "x"}


def test_the_correct_spelling_wins_when_both_are_present() -> None:
    data = {"MiscChannelCount": 9, "MISCChannelCount": 0}
    repair_key_names(data, "eeg", "eeg")
    assert data == {"MISCChannelCount": 0}


# ---------------------------------------------------------------------------
# Array types
# ---------------------------------------------------------------------------


def test_a_scalar_is_wrapped_where_the_schema_wants_an_array() -> None:
    """dcm2niix writes a bare number for a single-frame PET acquisition where
    the schema declares one entry per frame."""
    data = {"FrameDuration": 60.0}
    assert repair_array_types(data, "pet", "pet") == 1
    assert data["FrameDuration"] == [60.0]


def test_an_existing_list_is_untouched() -> None:
    data = {"FrameDuration": [10.0, 20.0]}
    assert repair_array_types(data, "pet", "pet") == 0
    assert data["FrameDuration"] == [10.0, 20.0]


def test_a_field_the_schema_leaves_untyped_is_not_reshaped() -> None:
    """EchoTime is declared number-or-array. The schema permits both, so
    choosing one for the converter would be meddling, not repair."""
    data = {"EchoTime": 0.03}
    repair_array_types(data, "anat", "T1w")
    assert data["EchoTime"] == 0.03


# ---------------------------------------------------------------------------
# Agnostic fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("datatype, suffix", [
    ("anat", "T1w"), ("func", "bold"), ("pet", "pet"), ("eeg", "eeg"),
])
def test_institution_reaches_every_datatype(datatype, suffix) -> None:
    """Where the study was done has nothing to do with what recorded it, and
    the schema declares these fields for every datatype. They used to reach
    electrophysiology alone, because that was the only enrichment pass that
    ran."""
    spec = RecordingMetaSpec()
    spec.defaults.institution_name = "Uni Oldenburg"
    spec.defaults.institution_dept = "Neuropsychology"

    data: dict = {}
    assert fill_agnostic_fields(data, datatype, suffix, spec) == 2
    assert data["InstitutionName"] == "Uni Oldenburg"
    assert data["InstitutionalDepartmentName"] == "Neuropsychology"


def test_a_value_already_present_is_never_overwritten() -> None:
    """A converter that read the institution out of a DICOM header knows the
    scanner's own answer; a dataset-wide default must not clobber it."""
    spec = RecordingMetaSpec()
    spec.defaults.institution_name = "Default Site"
    data = {"InstitutionName": "From The DICOM Header"}
    fill_agnostic_fields(data, "anat", "T1w", spec)
    assert data["InstitutionName"] == "From The DICOM Header"


def test_no_spec_writes_nothing() -> None:
    data: dict = {}
    assert fill_agnostic_fields(data, "anat", "T1w", None) == 0
    assert data == {}


# ---------------------------------------------------------------------------
# End to end over a tree
# ---------------------------------------------------------------------------


def test_repair_sidecars_walks_the_tree(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    eeg = root / "sub-001" / "eeg"
    anat = root / "sub-001" / "anat"
    eeg.mkdir(parents=True)
    anat.mkdir(parents=True)
    (eeg / "sub-001_task-rest_eeg.json").write_text(
        json.dumps({"MiscChannelCount": 0, "SamplingFrequency": 500})
    )
    (anat / "sub-001_T1w.json").write_text(json.dumps({"EchoTime": 0.03}))

    spec = RecordingMetaSpec()
    spec.defaults.institution_name = "Uni Oldenburg"

    # Two passes, two owners. Conversion fixes what the converter wrote;
    # the metadata step writes what the user stated.
    assert repair_converter_output(root) == 1
    assert _apply(root, spec) == 2

    eeg_data = json.loads((eeg / "sub-001_task-rest_eeg.json").read_text())
    assert eeg_data["MISCChannelCount"] == 0
    assert "MiscChannelCount" not in eeg_data
    assert eeg_data["InstitutionName"] == "Uni Oldenburg"

    # The MRI sidecar gets the agnostic value too: this is the datatype the
    # old enrichment pass skipped entirely.
    anat_data = json.loads((anat / "sub-001_T1w.json").read_text())
    assert anat_data["InstitutionName"] == "Uni Oldenburg"
    assert anat_data["EchoTime"] == 0.03


def test_rerunning_changes_nothing(tmp_path: Path) -> None:
    """Idempotent: a second pass over a repaired tree must be a no-op, or the
    convert step would report work it did not do."""
    root = tmp_path / "staging"
    eeg = root / "sub-001" / "eeg"
    eeg.mkdir(parents=True)
    (eeg / "sub-001_task-rest_eeg.json").write_text(
        json.dumps({"MiscChannelCount": 0})
    )
    assert repair_converter_output(root) == 1
    assert repair_converter_output(root) == 0


# ---------------------------------------------------------------------------
# The row layer
#
# A correction aimed at ONE recording has to reach that recording's sidecar and
# no other. The pass walks files, not rows, so it has to be told which row
# produced which file; these lock that down.
# ---------------------------------------------------------------------------


def _apply(root, spec, rows=()):
    """Apply what the user stated, the way the metadata step does.

    ``rows`` are ``(basename, row_id)`` pairs, the two inventory columns the
    pass reads to learn which row produced which file.
    """
    import pandas as pd

    inventory = pd.DataFrame(
        [{"proposed_basename": b, "source_file": r} for b, r in rows]
    ) if rows else None
    return apply_stated_metadata(root, spec, inventory)


def test_a_row_answer_reaches_only_that_rows_sidecar(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    anat = root / "sub-001" / "anat"
    anat.mkdir(parents=True)
    for name in ("sub-001_T1w", "sub-002_T1w"):
        (anat / f"{name}.json").write_text(json.dumps({"EchoTime": 0.03}))

    spec = RecordingMetaSpec()
    spec.row_templates["/raw/one"] = {"InstitutionName": "Only mine"}

    _apply(root, spec, [("sub-001_T1w", "/raw/one"), ("sub-002_T1w", "/raw/two")])
    mine = json.loads((anat / "sub-001_T1w.json").read_text())
    theirs = json.loads((anat / "sub-002_T1w.json").read_text())
    assert mine["InstitutionName"] == "Only mine"
    assert "InstitutionName" not in theirs


def test_a_row_answer_overrides_what_the_converter_wrote(tmp_path: Path) -> None:
    """The file's own header normally stands. This recording contradicting it
    is the exception, and the only reason a per-row answer exists."""
    root = tmp_path / "staging"
    anat = root / "sub-001" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-001_T1w.json").write_text(
        json.dumps({"InstitutionName": "read from the DICOM"})
    )

    spec = RecordingMetaSpec()
    spec.row_templates["/raw/one"] = {"InstitutionName": "the correct one"}

    _apply(root, spec, [("sub-001_T1w", "/raw/one")])
    data = json.loads((anat / "sub-001_T1w.json").read_text())
    assert data["InstitutionName"] == "the correct one"


def test_a_backend_suffix_still_matches_its_row(tmp_path: Path) -> None:
    """dcm2niix appends its own tail for echoes and phase images. Those are
    still that row's outputs."""
    root = tmp_path / "staging"
    func = root / "sub-001" / "func"
    func.mkdir(parents=True)
    (func / "sub-001_task-rest_bold_e2.json").write_text(json.dumps({}))

    spec = RecordingMetaSpec()
    spec.row_templates["/raw/one"] = {"InstitutionName": "matched anyway"}

    _apply(root, spec, [("sub-001_task-rest_bold", "/raw/one")])
    data = json.loads((func / "sub-001_task-rest_bold_e2.json").read_text())
    assert data["InstitutionName"] == "matched anyway"


def test_a_pet_block_reaches_the_sidecar_through_the_chain(tmp_path: Path) -> None:
    """PET used to be resolved only by its own fixup, so a dose imported from a
    lab spreadsheet never appeared in the form claiming to show what the file
    would say. Both now read one chain."""
    root = tmp_path / "staging"
    pet = root / "sub-001" / "pet"
    pet.mkdir(parents=True)
    (pet / "sub-001_pet.json").write_text(json.dumps({}))

    spec = RecordingMetaSpec()
    spec.pet_defaults.mode_of_administration = "bolus"

    _apply(root, spec, [("sub-001_pet", "/raw/one")])
    data = json.loads((pet / "sub-001_pet.json").read_text())
    assert data["ModeOfAdministration"] == "bolus"


# ---------------------------------------------------------------------------
# Acquisition times
# ---------------------------------------------------------------------------


def test_an_unpadded_seconds_field_is_padded() -> None:
    """dcm2niix drops the leading zero on the seconds when the DICOM had one.

    ``2025-05-26T11:22:1.995000`` is not a date-time to any validator, and the
    metadata step copies it on into ``scans.tsv``, so one slip by the converter
    is reported as two errors. The instant is right; only the spelling is not.
    """
    from bidsmgr.fixups.sidecar_schema import repair_datetimes

    data = {"AcquisitionDateTime": "2025-05-26T11:22:1.995000"}
    assert repair_datetimes(data, "func", "bold") == 1
    assert data["AcquisitionDateTime"] == "2025-05-26T11:22:01.995000"


def test_a_time_that_is_already_padded_is_left_alone() -> None:
    from bidsmgr.fixups.sidecar_schema import repair_datetimes

    data = {"AcquisitionDateTime": "2025-05-26T11:22:01.995000"}
    assert repair_datetimes(data, "func", "bold") == 0


def test_every_part_of_the_time_is_padded() -> None:
    from bidsmgr.fixups.sidecar_schema import repair_datetimes

    data = {"AcquisitionTime": "9:5:3"}
    assert repair_datetimes(data, "func", "bold") == 1
    assert data["AcquisitionTime"] == "09:05:03"


def test_something_that_is_not_a_time_is_not_touched() -> None:
    """Only the two date-time keys are looked at, and only when they parse."""
    from bidsmgr.fixups.sidecar_schema import repair_datetimes

    data = {"AcquisitionDateTime": "unknown", "TaskName": "1:2:3"}
    assert repair_datetimes(data, "func", "bold") == 0
    assert data == {"AcquisitionDateTime": "unknown", "TaskName": "1:2:3"}
