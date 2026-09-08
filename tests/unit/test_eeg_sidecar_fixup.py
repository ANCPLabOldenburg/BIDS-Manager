"""Unit tests for the post-write EEG/MEG enrichment fixup."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from bidsmgr.fixups.eeg_sidecar import enrich_recording_sidecars
from bidsmgr.recording_meta import (
    AcquisitionSpec,
    AuxChannelSpec,
    FilterSpec,
    RecordingMetaSpec,
    TaskProtocol,
)


def _task(datatype="eeg", basename="sub-001_task-rest_eeg", row_id="r1", task="rest"):
    return SimpleNamespace(
        datatype=datatype, basename=basename, row_id=row_id, entities={"task": task},
    )


def _apply_stated(staging, spec, tasks=(), cells=None):
    """Write what the user stated, the way ``bidsmgr-metadata`` does.

    Sidecar FIELDS used to be written during conversion by this fixup, which
    walked the same chain as the template pass, so a field could be written
    twice and the winner depended on the order the two ran in. There is one
    pass now and it belongs to the metadata step. These tests follow it, because
    what they assert is what ends up in the file.
    """
    import pandas as pd
    from bidsmgr.fixups.sidecar_schema import apply_stated_metadata

    rows = [
        {"proposed_basename": t.basename, "source_file": t.row_id, **(cells or {})}
        for t in tasks
    ]
    return apply_stated_metadata(
        staging, spec, pd.DataFrame(rows) if rows else None,
    )


def _stage_eeg(tmp_path: Path, datatype="eeg", basename="sub-001_task-rest_eeg") -> Path:
    """Create a synthetic staged datatype dir with the files mne-bids writes."""
    prefix = basename.rsplit("_", 1)[0]
    d = tmp_path / ".tmp_bidsmgr" / "sub-001" / datatype
    d.mkdir(parents=True)
    (d / f"{basename}.json").write_text(
        json.dumps({"SamplingFrequency": 1000.0, "PowerLineFrequency": 50}),
        encoding="utf-8",
    )
    # mne-bids writes TSVs with a UTF-8 BOM; replicate that (encoding
    # "utf-8-sig") so the fixup's BOM tolerance is exercised.
    (d / f"{prefix}_channels.tsv").write_text(
        "name\ttype\tunits\tdescription\n"
        "Fp1\tEEG\tµV\tn/a\n"
        "ECG\tMISC\tn/a\tn/a\n"
        "EOG\tMISC\tn/a\tn/a\n",
        encoding="utf-8-sig",
    )
    (d / f"{prefix}_events.tsv").write_text(
        "onset\tduration\ttrial_type\n"
        "12.0\t0\tS 20\n"
        "72.0\t0\tS 21\n",
        encoding="utf-8-sig",
    )
    return tmp_path / ".tmp_bidsmgr" / "sub-001"


def _full_spec() -> RecordingMetaSpec:
    return RecordingMetaSpec(
        defaults=AcquisitionSpec(
            eeg_reference="Cz",
            eeg_ground="AFz",
            manufacturer="Brain Products",
            amplifier_model="BrainAmp Standard",
            institution_name="University X",
            institution_dept="Psychology",
            filters=[FilterSpec(name="LP", kind="Hardware", info={"cutoff": 260})],
            aux_channels={
                "ECG": AuxChannelSpec(
                    mne_type="ecg", bids_type="ECG", units="mV", description="ecg"
                ),
                "EOG": AuxChannelSpec(bids_type="EOG"),
            },
        ),
        event_maps={"rest": {"S 20": "eyes_open", "S 21": "eyes_closed"}},
        task_protocols={"rest": TaskProtocol(task_description="resting", instructions="relax")},
    )


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def test_noop_when_spec_none(tmp_path):
    staging = _stage_eeg(tmp_path)
    assert enrich_recording_sidecars(staging, [_task()], None) == 0


def test_sidecar_fields_written(tmp_path):
    """What the user stated reaches the sidecar. Written by the metadata step
    now; the task protocol still comes from the conversion, which is the pass
    that knows which task this recording belongs to."""
    staging = _stage_eeg(tmp_path)
    spec = _full_spec()
    assert _apply_stated(staging, spec, [_task()]) > 0
    enrich_recording_sidecars(staging, [_task()], spec)
    side = _read_json(staging / "eeg" / "sub-001_task-rest_eeg.json")
    assert side["EEGReference"] == "Cz"
    assert side["EEGGround"] == "AFz"
    assert side["Manufacturer"] == "Brain Products"
    assert side["ManufacturersModelName"] == "BrainAmp Standard"
    assert side["InstitutionName"] == "University X"
    assert side["InstitutionalDepartmentName"] == "Psychology"
    assert side["HardwareFilters"] == {"LP": {"cutoff": 260}}
    assert side["TaskDescription"] == "resting"
    assert side["Instructions"] == "relax"
    # PowerLineFrequency is owned by the backend; the fixup must not touch it.
    assert side["PowerLineFrequency"] == 50


def test_channels_retyped(tmp_path):
    staging = _stage_eeg(tmp_path)
    enrich_recording_sidecars(staging, [_task()], _full_spec())
    text = (staging / "eeg" / "sub-001_task-rest_channels.tsv").read_text()
    lines = {ln.split("\t")[0]: ln.split("\t") for ln in text.strip().splitlines()}
    assert lines["ECG"][1] == "ECG"
    assert lines["ECG"][2] == "mV"
    assert lines["EOG"][1] == "EOG"
    # The real EEG channel is left alone (not generic).
    assert lines["Fp1"][1] == "EEG"


def test_events_mapped_and_json_written(tmp_path):
    staging = _stage_eeg(tmp_path)
    enrich_recording_sidecars(staging, [_task()], _full_spec())
    ev = (staging / "eeg" / "sub-001_task-rest_events.tsv").read_text()
    assert "eyes_open" in ev and "eyes_closed" in ev
    assert "S 20" not in ev
    ev_json = _read_json(staging / "eeg" / "sub-001_task-rest_events.json")
    assert set(ev_json["trial_type"]["Levels"]) == {"eyes_open", "eyes_closed"}


def test_meg_skips_eeg_only_keys(tmp_path):
    staging = _stage_eeg(tmp_path, datatype="meg", basename="sub-001_task-rest_meg")
    spec = RecordingMetaSpec(
        defaults=AcquisitionSpec(eeg_reference="Cz", manufacturer="Elekta"),
    )
    _apply_stated(
        staging, spec, [_task(datatype="meg", basename="sub-001_task-rest_meg")],
    )
    side = _read_json(staging / "meg" / "sub-001_task-rest_meg.json")
    assert "EEGReference" not in side  # EEG-only key skipped for MEG
    assert side["Manufacturer"] == "Elekta"  # common key still written


def test_per_row_override_applies(tmp_path):
    staging = _stage_eeg(tmp_path)
    spec = RecordingMetaSpec(
        defaults=AcquisitionSpec(eeg_reference="Cz"),
        overrides={"r1": AcquisitionSpec(eeg_reference="FCz")},
    )
    _apply_stated(staging, spec, [_task(row_id="r1")])
    side = _read_json(staging / "eeg" / "sub-001_task-rest_eeg.json")
    assert side["EEGReference"] == "FCz"


def test_meg_specific_fields_written(tmp_path):
    """Only MEG fields mne-bids cannot derive are written here; the channel-
    derived ones are left to mne-bids and never appear from the spec."""
    staging = _stage_eeg(tmp_path, datatype="meg", basename="sub-001_task-rest_meg")
    spec = RecordingMetaSpec(
        defaults=AcquisitionSpec(
            dewar_position="upright",
            associated_empty_room="bids::sub-emptyroom/ses-x/meg/...",
            subject_artefact_description="occasional jaw clench",
        ),
    )
    _apply_stated(
        staging, spec, [_task(datatype="meg", basename="sub-001_task-rest_meg")],
    )
    side = _read_json(staging / "meg" / "sub-001_task-rest_meg.json")
    assert side["DewarPosition"] == "upright"
    assert side["AssociatedEmptyRoom"].startswith("bids::")
    assert side["SubjectArtefactDescription"] == "occasional jaw clench"
    # Channel-derived MEG facts are NOT written from the spec (mne-bids owns them).
    assert "ContinuousHeadLocalization" not in side
    assert "DigitizedLandmarks" not in side


def test_meg_fields_not_written_for_eeg(tmp_path):
    """MEG-only keys never leak into an EEG sidecar."""
    staging = _stage_eeg(tmp_path)
    spec = RecordingMetaSpec(
        defaults=AcquisitionSpec(dewar_position="supine", eeg_reference="Cz"),
    )
    _apply_stated(staging, spec, [_task()])
    side = _read_json(staging / "eeg" / "sub-001_task-rest_eeg.json")
    assert "DewarPosition" not in side
    assert side["EEGReference"] == "Cz"


# ---------------------------------------------------------------------------
# Which datatype takes which field is the schema's call, not a branch here
# ---------------------------------------------------------------------------


def _spec_with_everything():
    from bidsmgr.recording_meta import RecordingMetaSpec
    spec = RecordingMetaSpec()
    d = spec.defaults
    d.manufacturer = "Brain Products"
    d.cap_manufacturer = "EasyCap"
    d.cap_model = "M1"
    d.eeg_reference = "Cz"
    d.eeg_ground = "AFz"
    d.institution_name = "Uni Oldenburg"
    return spec


def _apply(tmp_path, datatype, existing=None):
    import json
    from bidsmgr.fixups.sidecar_schema import apply_sequence_template, fill_agnostic_fields
    p = tmp_path / f"{datatype}.json"
    data = dict(existing or {})
    spec = _spec_with_everything()
    apply_sequence_template(data, datatype, datatype, None, spec)
    fill_agnostic_fields(data, datatype, datatype, spec)
    p.write_text(json.dumps(data))
    return json.loads(p.read_text())


def test_the_cap_reaches_every_datatype_that_has_one(tmp_path):
    """REGRESSION: the cap fields were written for EEG alone, though the schema
    declares them for MEG and NIRS too, so a cap entered for either was
    silently dropped. iEEG has electrodes rather than a cap, and the schema
    says so."""
    for datatype in ("eeg", "meg", "nirs"):
        data = _apply(tmp_path, datatype)
        assert data.get("CapManufacturer") == "EasyCap", datatype
        assert data.get("CapManufacturersModelName") == "M1", datatype
    assert "CapManufacturer" not in _apply(tmp_path, "ieeg")


def test_reference_uses_the_name_its_datatype_uses(tmp_path):
    """The same spec value is EEGReference on a scalp recording and
    iEEGReference on an intracranial one."""
    eeg = _apply(tmp_path, "eeg")
    assert eeg["EEGReference"] == "Cz" and eeg["EEGGround"] == "AFz"
    ieeg = _apply(tmp_path, "ieeg")
    assert ieeg["iEEGReference"] == "Cz" and ieeg["iEEGGround"] == "AFz"
    assert "EEGReference" not in ieeg


def test_meg_takes_an_eeg_reference_only_with_simultaneous_eeg(tmp_path):
    """BIDS declares EEGReference for MEG because simultaneous EEG is common,
    not because every MEG run has it. mne-bids has already counted the
    channels, so state it only when there are some to reference."""
    alone = _apply(tmp_path, "meg", existing={"EEGChannelCount": 0})
    assert "EEGReference" not in alone

    combined = _apply(tmp_path, "meg", existing={"EEGChannelCount": 64})
    assert combined["EEGReference"] == "Cz"
