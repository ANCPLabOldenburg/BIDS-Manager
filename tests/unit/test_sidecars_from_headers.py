"""Sidecars written again from what is left: the recording's header, the
project's source data, the user's answers and the links other sidecars hold.

Small synthetic datasets, so every rule is pinned on a case built for it: the
NIfTI fields a header gives and the ones it must not pretend to give, a value
never overwritten, a TODO replaced only by a real value, a field left to the
sidecar higher up that states it, fieldmap links coming back from the other
images of the fieldmap, one Undo for the lot.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from bidsmgr.editor import headers, sidecars
from bidsmgr.editor.sidecars import (
    ANSWERS,
    FILENAME,
    HEADER,
    LINKS,
    PLACEHOLDER,
    SOURCE,
    apply_sidecars,
    plan_sidecars,
    recording_for,
    recordings,
    sidecar_path,
)

DCM2NIIX = b"TE=30;Time=102012.133;phase=1;mb=4"


def _nifti(path: Path, shape=(4, 4, 3), *, tr=2.0, descrip=b"", slice_timing=False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.int16), np.eye(4))
    hdr = img.header
    zooms = (2.0, 2.0, 2.0) + ((tr,) if len(shape) > 3 else ())
    hdr.set_zooms(zooms)
    hdr["pixdim"][4] = tr
    hdr.set_xyzt_units("mm", "sec")
    hdr["descrip"] = descrip
    if slice_timing:
        hdr.set_dim_info(freq=0, phase=1, slice=2)
        hdr["slice_code"] = 1           # sequential increasing
        hdr["slice_start"] = 0
        hdr["slice_end"] = shape[2] - 1
        hdr["slice_duration"] = 0.5
    nib.save(img, str(path))
    return path


def _json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    _json(root / "dataset_description.json", {"Name": "t", "BIDSVersion": "1.10.0"})
    return root


def _fields(plan) -> dict:
    return {f.name: (f.value, f.origin) for f in plan.added}


# ---------------------------------------------------------------------------
# What a header gives
# ---------------------------------------------------------------------------


def test_a_bold_header_gives_its_timing(tmp_path):
    bold = _nifti(tmp_path / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 5),
                  tr=0.85, descrip=DCM2NIIX, slice_timing=True)
    read = headers.read_header(bold, "func", "bold")
    assert read.fields["RepetitionTime"] == 0.85
    assert read.fields["EchoTime"] == 0.03
    assert read.fields["MultibandAccelerationFactor"] == 4
    assert read.fields["SliceTiming"] == [0.0, 0.5, 1.0]
    assert read.fields["SliceEncodingDirection"] == "k"
    # The sign of the phase encoding is not in a header: never guessed.
    assert "PhaseEncodingDirection" not in read.fields
    assert headers.ROUNDED_TE in read.note


def test_a_3d_image_by_another_tool_gives_no_repetition_time(tmp_path):
    # pixdim[4] = 1.0 in a 3-D image is a default, not a measurement.
    t1 = _nifti(tmp_path / "sub-01/anat/sub-01_T1w.nii.gz", tr=1.0)
    read = headers.read_header(t1, "anat", "T1w")
    assert read.fields == {}
    assert "no timing" in read.note


def test_a_3d_image_by_dcm2niix_gives_its_repetition_time(tmp_path):
    t1 = _nifti(tmp_path / "sub-01/anat/sub-01_T1w.nii.gz", tr=2.3,
                descrip=b"TE=2.1;Time=102520.255;phase=1")
    read = headers.read_header(t1, "anat", "T1w")
    assert read.fields == {"RepetitionTime": 2.3, "EchoTime": 0.0021}


def test_a_pet_header_says_why_it_gives_nothing(tmp_path):
    pet = _nifti(tmp_path / "sub-01/pet/sub-01_pet.nii.gz", (4, 4, 3, 2), tr=0.0)
    read = headers.read_header(pet, "pet", "pet")
    assert read.fields == {} and "tracer" in read.note


def test_a_spectrum_gives_its_header_extension(tmp_path):
    path = tmp_path / "sub-01/mrs/sub-01_svs.nii.gz"
    path.parent.mkdir(parents=True)
    img = nib.Nifti2Image(np.zeros((1, 1, 1, 8), dtype=np.complex64), np.eye(4))
    header = {"SpectrometerFrequency": [123.26260375976562], "ResonantNucleus": ["1H"],
              "EchoTime": 0.08, "InversionTime": 0, "PatientName": "Someone"}
    img.header.extensions.append(
        nib.nifti1.Nifti1Extension(44, json.dumps(header).encode()))
    nib.save(img, str(path))
    read = headers.read_header(path, "mrs", "svs")
    assert read.fields == {"SpectrometerFrequency": [123.262604],
                           "ResonantNucleus": ["1H"], "EchoTime": 0.08}


def test_a_physio_table_says_it_has_no_header(tmp_path):
    path = tmp_path / "sub-01/func/sub-01_task-rest_physio.tsv.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    read = headers.read_header(path, "func", "physio")
    assert read.fields == {} and "no header" in read.note


def test_an_meg_recording_is_described_by_mne_bids(tmp_path):
    mne = pytest.importorskip("mne")
    pytest.importorskip("mne_bids")
    info = mne.create_info(["MEG 0111", "MEG 0112", "STI 014"], 250.0,
                           ["mag", "grad", "stim"])
    info["line_freq"] = 50.0
    raw = mne.io.RawArray(np.zeros((3, 500)), info, verbose=False)
    path = tmp_path / "sub-01/meg/sub-01_task-rest_meg.fif"
    path.parent.mkdir(parents=True)
    raw.save(str(path), verbose=False)
    read = headers.read_header(path, "meg", "meg")
    assert read.fields["SamplingFrequency"] == 250.0
    assert read.fields["PowerLineFrequency"] == 50.0
    assert read.fields["TaskName"] == "rest"
    assert read.fields["RecordingDuration"] == pytest.approx(2.0, abs=0.01)
    assert "MNE-BIDS" in read.reader


# ---------------------------------------------------------------------------
# Which files, and where their sidecars go
# ---------------------------------------------------------------------------


def test_recordings_and_their_sidecars(tmp_path):
    root = _dataset(tmp_path)
    bold = _nifti(root / "sub-01/ses-1/func/sub-01_ses-1_task-rest_bold.nii.gz", (4, 4, 3, 2))
    physio = root / "sub-01/ses-1/func/sub-01_ses-1_task-rest_physio.tsv.gz"
    physio.write_bytes(b"")
    (root / "sub-01/ses-1/func/sub-01_ses-1_task-rest_events.tsv").write_text("onset\n")
    (root / "sub-01/ses-1/sub-01_ses-1_scans.tsv").write_text("filename\n")
    ds = root / "sub-01/ses-1/meg/sub-01_ses-1_task-rest_meg.ds"
    (ds / "inner").mkdir(parents=True)
    split = root / "sub-01/ses-1/meg/sub-01_ses-1_task-x_split-01_meg.fif"
    split.write_bytes(b"")
    hidden = _nifti(root / "sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz")
    (root / ".bidsignore").write_text("**/anat/\n")
    _nifti(root / "derivatives/p/sub-01/anat/sub-01_T1w.nii.gz")

    found = recordings(root)
    assert found == sorted([bold, physio, ds, split])
    assert hidden not in found
    assert sidecar_path(split).name == "sub-01_ses-1_task-x_meg.json"
    assert sidecar_path(ds).name == "sub-01_ses-1_task-rest_meg.json"
    assert sidecar_path(physio).name == "sub-01_ses-1_task-rest_physio.json"
    # A sidecar picked stands for its recording.
    side = _json(sidecar_path(bold), {})
    assert recording_for(side) == bold
    assert recordings(root, [side]) == [bold]


# ---------------------------------------------------------------------------
# Planning: what is written, and what never is
# ---------------------------------------------------------------------------


def test_a_missing_sidecar_is_written_from_the_header(tmp_path):
    root = _dataset(tmp_path)
    _nifti(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 5),
           tr=0.85, descrip=DCM2NIIX)
    plan = plan_sidecars(root)
    (one,) = plan.files
    assert not one.exists and one.sidecar == "sub-01/func/sub-01_task-rest_bold.json"
    fields = _fields(one)
    assert fields["RepetitionTime"] == (0.85, HEADER)
    assert fields["TaskName"] == ("rest", FILENAME)
    assert one.read == ["the NIfTI header"]
    assert plan.used_source is False   # not a project: nothing converted here


def test_a_file_with_a_sidecar_is_left_alone_unless_completing(tmp_path):
    root = _dataset(tmp_path)
    bold = _nifti(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 5),
                  tr=0.85, descrip=DCM2NIIX)
    _json(sidecar_path(bold), {"TaskName": "rest", "EchoTime": 0.031,
                               "RepetitionTime": "TODO"})
    plan = plan_sidecars(root)
    assert plan.files == [] and plan.skipped == [
        ("sub-01/func/sub-01_task-rest_bold.nii.gz", "has a sidecar")]

    plan = plan_sidecars(root, complete=True)
    (one,) = plan.files
    fields = {f.name: f for f in one.added}
    # A value is never overwritten, even where the header says otherwise.
    assert "EchoTime" not in fields and one.document["EchoTime"] == 0.031
    # A TODO is not a value: the header's replaces it, in place.
    assert fields["RepetitionTime"].replaces_todo
    assert list(one.document)[:3] == ["TaskName", "EchoTime", "RepetitionTime"]
    assert one.document["RepetitionTime"] == 0.85
    assert one.kept == 3


def test_a_placeholder_never_replaces_a_placeholder_or_a_value(tmp_path):
    root = _dataset(tmp_path)
    bold = _nifti(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 5), tr=1.0)
    _json(sidecar_path(bold), {"TaskName": "rest", "TaskDescription": "TODO",
                               "Instructions": "Rest."})
    plan = plan_sidecars(root, complete=True, fill_scope="recommended")
    (one,) = plan.files
    added = _fields(one)
    assert "TaskDescription" not in added and "Instructions" not in added
    assert any(origin == PLACEHOLDER for _v, origin in added.values())
    assert one.document["Instructions"] == "Rest."


def test_what_a_sidecar_higher_up_states_is_left_to_it(tmp_path):
    root = _dataset(tmp_path)
    _nifti(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 5),
           tr=0.85, descrip=DCM2NIIX)
    _json(root / "task-rest_bold.json", {"TaskName": "rest", "RepetitionTime": 0.85,
                                         "EchoTime": 0.025})
    (one,) = plan_sidecars(root).files
    added = _fields(one)
    assert "RepetitionTime" not in added and "TaskName" not in added
    assert "EchoTime" not in added
    inherited = {i.name: i for i in one.inherited}
    assert inherited["RepetitionTime"].source == "task-rest_bold.json"
    assert not inherited["RepetitionTime"].differs
    # The header disagrees: shown, never silently overridden.
    assert inherited["EchoTime"].differs and inherited["EchoTime"].found == 0.03


def test_the_source_data_overrides_a_disagreeing_sidecar_higher_up(tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    _nifti(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 5), tr=0.85)
    _json(root / "task-rest_bold.json", {"TaskName": "rest", "EchoTime": 0.031,
                                         "RepetitionTime": 0.85})
    _project(root, monkeypatch, {"sub-01_task-rest_bold.json": {
        "EchoTime": 0.03, "RepetitionTime": 0.85}})
    (one,) = plan_sidecars(root).files
    added = {f.name: f for f in one.added}
    # The exact value for this run wins here; the inherited one stays for others.
    assert added["EchoTime"].value == 0.03 and added["EchoTime"].origin == SOURCE
    assert added["EchoTime"].overrides == "task-rest_bold.json"
    assert added["EchoTime"].overridden == 0.031
    assert any("EchoTime is 0.031 in task-rest_bold.json" in n for n in one.notes)
    # A value that agrees is left to the sidecar higher up.
    assert [i.name for i in one.inherited] == ["RepetitionTime"]


def test_a_sidecar_beside_it_counts_as_its_sidecar(tmp_path):
    root = _dataset(tmp_path)
    _nifti(root / "sub-01/func/sub-01_task-rest_run-1_bold.nii.gz", (4, 4, 3, 5), tr=1.0)
    _json(root / "sub-01/func/sub-01_task-rest_bold.json", {"TaskName": "rest"})
    plan = plan_sidecars(root)
    assert plan.files == []
    assert plan.skipped[0][1] == "described by sub-01_task-rest_bold.json"


def test_an_unreadable_sidecar_is_not_overwritten(tmp_path):
    root = _dataset(tmp_path)
    bold = _nifti(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 5), tr=1.0)
    sidecar_path(bold).write_text("{ not json", encoding="utf-8")
    plan = plan_sidecars(root, complete=True)
    assert plan.files == [] and "not valid JSON" in plan.skipped[0][1]


# ---------------------------------------------------------------------------
# Links from the other sidecars
# ---------------------------------------------------------------------------


def _fieldmap(root: Path) -> dict:
    fmap = root / "sub-01/fmap"
    _nifti(fmap / "sub-01_acq-fm_magnitude1.nii.gz", tr=0.5, descrip=b"TE=5.2;Time=1;phase=1")
    _nifti(fmap / "sub-01_acq-fm_magnitude2.nii.gz", tr=0.5, descrip=b"TE=7.7;Time=1;phase=1")
    _nifti(fmap / "sub-01_acq-fm_phasediff.nii.gz", tr=0.5, descrip=b"TE=7.7;Time=1;phase=1")
    links = {"B0FieldIdentifier": "fmap0",
             "IntendedFor": ["bids::sub-01/func/sub-01_task-rest_bold.nii.gz"]}
    _json(fmap / "sub-01_acq-fm_magnitude1.json", dict(links, EchoTime=0.0052))
    _nifti(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 5), tr=1.0)
    return links


def test_a_fieldmap_takes_its_links_from_the_rest_of_the_fieldmap(tmp_path):
    root = _dataset(tmp_path)
    links = _fieldmap(root)
    plan = plan_sidecars(root)
    by_name = {Path(p.sidecar).name: p for p in plan.files}
    phasediff = _fields(by_name["sub-01_acq-fm_phasediff.json"])
    assert phasediff["B0FieldIdentifier"] == (links["B0FieldIdentifier"], LINKS)
    assert phasediff["IntendedFor"] == (links["IntendedFor"], LINKS)
    # Its two echo times, from the magnitude images (dcm2niix's two figures).
    assert phasediff["EchoTime1"] == (0.0052, HEADER)
    assert phasediff["EchoTime2"] == (0.0077, HEADER)
    assert "EchoTime" not in phasediff
    # The run the fieldmap is intended for takes it as its source.
    bold = _fields(by_name["sub-01_task-rest_bold.json"])
    assert bold["B0FieldSource"] == ("fmap0", LINKS)


# ---------------------------------------------------------------------------
# The project: source data and answers
# ---------------------------------------------------------------------------


def _project(root: Path, monkeypatch, produced: dict, answers=None) -> None:
    prov = root / ".bidsmgr/provenance"
    prov.mkdir(parents=True)
    task = {"row_id": "1.2.3", "series_uid": "1.2.3", "datatype": "func", "suffix": "bold",
            "basename": "sub-01_task-rest_bold",
            "outputs": ["sub-01_task-rest_bold.json", "sub-01_task-rest_bold.nii.gz"],
            "success": True}
    _json(prov / "sub-01.json", {"tasks": [task], "fieldmap_renames": []})

    calls = []

    def row(self, row_id):
        calls.append(row_id)
        import pandas as pd

        class _Sources:
            root = Path("/raw")

        return pd.Series({"series_uid": row_id, "bids_name": "renamed_since"}), _Sources(), None

    def fake_convert(row, work_dir, sources, *, spec=None, options=None):
        calls.append(row["bids_name"])
        calls.append(options)
        return produced

    monkeypatch.setattr(sidecars._Project, "row", row)
    monkeypatch.setattr("bidsmgr.cli.convert.sidecars_for_row", fake_convert)
    monkeypatch.setattr(sidecars._Project, "answers",
                        lambda self: answers or (None, {}, {}))
    return calls


def test_the_source_data_comes_first(tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    _nifti(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 5),
           tr=0.85, descrip=DCM2NIIX)
    source = {"Manufacturer": "Siemens", "RepetitionTime": 0.85, "EchoTime": 0.0301,
              "PhaseEncodingDirection": "j-"}
    calls = _project(root, monkeypatch, {"sub-01_task-rest_bold.json": source})
    plan = plan_sidecars(root)
    assert plan.used_source
    (one,) = plan.files
    fields = _fields(one)
    # The source's exact echo time, not the header's two figures.
    assert fields["EchoTime"] == (0.0301, SOURCE)
    assert fields["PhaseEncodingDirection"] == ("j-", SOURCE)
    # The header adds only what the source lacks.
    assert fields["MultibandAccelerationFactor"] == (4, HEADER)
    assert one.read[0] == "the data it was converted from, converted again"
    assert headers.ROUNDED_TE in " ".join(one.notes)
    # Converted under the name it had at conversion, whatever the table says now,
    # and with the options it had then (none recorded: the defaults).
    from bidsmgr.cli.convert import TaskOptions

    assert calls == ["1.2.3", "sub-01_task-rest_bold", TaskOptions()]


def test_the_source_is_converted_with_the_options_it_was_converted_with(tmp_path, monkeypatch):
    """Dummy volumes and the physiology layout as recorded at conversion, not
    what the table or Settings say now."""
    from bidsmgr.cli.convert import TaskOptions

    root = _dataset(tmp_path)
    _nifti(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 5), tr=0.85)
    calls = _project(root, monkeypatch, {"sub-01_task-rest_bold.json": {"EchoTime": 0.03}})
    prov = root / ".bidsmgr/provenance/sub-01.json"
    record = json.loads(prov.read_text())
    record["tasks"][0].update(drop_volumes=3, physio_layout="one_as_recorded", physio_align=True)
    prov.write_text(json.dumps(record))
    plan_sidecars(root)
    assert calls[-1] == TaskOptions(drop_volumes=3, physio_layout="one_as_recorded", physio_align=True)


def test_without_source_data_the_reason_is_given(tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    _nifti(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 5), tr=0.85)
    _project(root, monkeypatch, {"": "no DICOM of this series was found under /raw"})
    (one,) = plan_sidecars(root).files
    assert any("no DICOM of this series" in n for n in one.notes)
    assert _fields(one)["RepetitionTime"] == (0.85, HEADER)


def test_the_users_answers_win_over_the_header(tmp_path, monkeypatch):
    from bidsmgr.recording_meta import RecordingMetaSpec

    root = _dataset(tmp_path)
    _nifti(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 5), tr=0.85)
    spec = RecordingMetaSpec()
    spec.sequence_templates["func/bold"] = {"TaskDescription": "Eyes open",
                                            "RepetitionTime": 0.9}
    _project(root, monkeypatch, {"": "not found"}, answers=(spec, {}, {}))
    (one,) = plan_sidecars(root).files
    fields = _fields(one)
    assert fields["TaskDescription"] == ("Eyes open", ANSWERS)
    assert fields["RepetitionTime"] == (0.9, ANSWERS)


# ---------------------------------------------------------------------------
# A dataset that was anonymised stays anonymised
# ---------------------------------------------------------------------------


def test_new_fields_are_anonymised_like_the_dataset(tmp_path, monkeypatch):
    from bidsmgr.anonymise.state import STATE_FILE, State

    root = _dataset(tmp_path)
    _nifti(root / "sub-01/anat/sub-01_T1w.nii.gz", tr=2.0)
    _json(root / "sub-01/anat/sub-01_FLAIR.json", {"DeidentificationMethod": [
        "BIDS Manager 1.5.1 anonymisation (recommended): dates shifted"]})
    _nifti(root / "sub-01/anat/sub-01_FLAIR.nii.gz", tr=2.0)
    state = State(secret="s" * 32, offsets={"sub-01": 7 * 1000})
    (root / STATE_FILE).parent.mkdir(parents=True)
    (root / STATE_FILE).write_text(state.to_json(), encoding="utf-8")
    source = {"AcquisitionDateTime": "2023-09-14T10:25:20.255000",
              "DeviceSerialNumber": "66080", "PatientName": "Someone",
              "InstitutionAddress": "Somewhere 1", "RepetitionTime": 2.0}
    produced = {"sub-01_T1w.json": source}
    _project(root, monkeypatch, produced)
    task = {"row_id": "9.9", "datatype": "anat", "suffix": "T1w", "basename": "sub-01_T1w",
            "outputs": ["sub-01_T1w.json"], "success": True}
    _json(root / ".bidsmgr/provenance/sub-01.json", {"tasks": [task]})

    plan = plan_sidecars(root)
    assert plan.anonymised == "Recommended"
    (one,) = plan.files
    doc = one.document
    assert "PatientName" not in doc
    # Shifted by the participant's remembered number of days.
    import datetime as dt

    shifted = dt.date(2023, 9, 14) - dt.timedelta(days=7 * 1000)
    assert doc["AcquisitionDateTime"].startswith(shifted.isoformat())
    assert doc["DeviceSerialNumber"].startswith("device-")
    assert doc["InstitutionAddress"] == "n/a"
    assert any("anonymisation" in m for m in doc["DeidentificationMethod"])


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_writing_is_one_undoable_step(tmp_path):
    from bidsmgr.project import operations

    root = _dataset(tmp_path)
    bold = _nifti(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 5), tr=0.85)
    t1 = _nifti(root / "sub-01/anat/sub-01_T1w.nii.gz", tr=2.0, descrip=b"TE=2.1;Time=1")
    _json(sidecar_path(t1), {"Manufacturer": "Siemens", "EchoTime": "TODO"})
    before = sidecar_path(t1).read_text()

    plan = plan_sidecars(root, complete=True)
    written = apply_sidecars(root, plan.writes())
    assert sorted(written) == sorted([sidecar_path(bold), sidecar_path(t1)])
    assert json.loads(sidecar_path(t1).read_text()) == {
        "Manufacturer": "Siemens", "EchoTime": 0.0021, "RepetitionTime": 2.0}
    step = operations.peek_undo(root)
    assert step is not None and "2 sidecars" in step.label

    operations.undo_last(root)
    assert not sidecar_path(bold).exists()
    assert sidecar_path(t1).read_text() == before


def test_nothing_to_write_writes_nothing(tmp_path):
    root = _dataset(tmp_path)
    bold = _nifti(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 5), tr=0.85)
    _json(sidecar_path(bold), {"TaskName": "rest", "RepetitionTime": 0.85})
    plan = plan_sidecars(root, complete=True)
    assert plan.writes() == []
    assert apply_sidecars(root, plan.writes()) == []


def test_a_plan_can_be_stopped(tmp_path):
    root = _dataset(tmp_path)
    for i in range(3):
        _nifti(root / f"sub-0{i}/anat/sub-0{i}_T1w.nii.gz", tr=2.0, descrip=b"TE=2;Time=1")
    seen = []
    plan = plan_sidecars(root, progress=lambda d, t: seen.append((d, t)),
                         cancel=lambda: len(seen) >= 2)
    assert len(plan.files) == 2 and seen[0] == (0, 3)


# ---------------------------------------------------------------------------
# Files outside BIDS naming (2026-10-21): every readable file gets a sidecar
# ---------------------------------------------------------------------------


def test_every_readable_file_is_a_candidate_bids_name_or_not(tmp_path):
    root = _dataset(tmp_path)
    bids = _nifti(root / "sub-01/anat/sub-01_T1w.nii.gz")
    plain = [
        _nifti(root / "sub-01/anat/T1.nii.gz"),
        _nifti(root / "sub-01/anat/sub-01_T1w_defaced.nii.gz"),
        _nifti(root / "sub-01/anat/old/sub-01_T1w.nii.gz"),
        _nifti(root / "sub-01/extra/scan.nii.gz"),
        _nifti(root / "mri/subject1/t1_mprage.nii"),
    ]
    deriv = _nifti(root / "derivatives/prep/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz")
    source = _nifti(root / "sourcedata/t1.nii.gz")
    (root / "notes").mkdir()
    (root / "notes/report.pdf").write_bytes(b"%PDF-1.4")
    hidden = _nifti(root / ".bidsmgr/editor/originals/x.nii.gz")

    whole = recordings(root)
    assert whole == sorted([bids, *plain])
    # Not the dataset's own recordings, unless picked.
    assert deriv not in whole and source not in whole
    assert recordings(root, [root / "derivatives"]) == [deriv]
    assert recordings(root, [source]) == [source]
    assert hidden not in recordings(root, [root / ".bidsmgr"])


def test_a_sidecar_stands_for_its_image_not_its_gradient_table(tmp_path):
    root = _dataset(tmp_path)
    dwi = _nifti(root / "sub-01/dwi/sub-01_dwi.nii.gz", (4, 4, 3, 3))
    (root / "sub-01/dwi/sub-01_dwi.bval").write_text("0 1000 1000\n")
    (root / "sub-01/dwi/sub-01_dwi.bvec").write_text("0 1 0\n0 0 1\n0 0 0\n")
    side = _json(sidecar_path(dwi), {})
    assert recording_for(side, root) == dwi
    assert recordings(root, [side]) == [dwi]


def test_the_kind_comes_from_the_name_the_folder_or_the_file(tmp_path):
    from bidsmgr.editor.sidecars import (
        KIND_BIDS, KIND_CHOSEN, KIND_FOLDER, KIND_GUESS, KIND_NAME, kind_of)

    root = _dataset(tmp_path)
    cases = {
        "sub-01/anat/sub-01_T1w.nii.gz": ((4, 4, 3), ("anat", "T1w", KIND_BIDS)),
        "sub-01/anat/old/sub-01_FLAIR.nii.gz": ((4, 4, 3), ("anat", "FLAIR", KIND_NAME)),
        "scans/rest_bold.nii.gz": ((4, 4, 3, 4), ("func", "bold", KIND_NAME)),
        "sub-01/func/run2.nii.gz": ((4, 4, 3, 4), ("func", "bold", KIND_FOLDER)),
        "sub-01/anat/T1.nii.gz": ((4, 4, 3), ("anat", "T1w", KIND_FOLDER)),
        "mri/t1_mprage.nii": ((4, 4, 3), ("anat", "T1w", KIND_GUESS)),
        "mri/rest.nii.gz": ((4, 4, 3, 4), ("func", "bold", KIND_GUESS)),
    }
    for rel, (shape, want) in cases.items():
        _nifti(root / rel, shape)
        kind = kind_of(root, root / rel)
        assert (kind.datatype, kind.suffix, kind.how) == want, rel
        assert kind.why
    pet = _nifti(root / "mri/fdg.nii.gz", (4, 4, 3), tr=0.0, descrip=b"Time=92823.000")
    assert (kind_of(root, pet).datatype, kind_of(root, pet).how) == ("pet", KIND_GUESS)
    chosen = kind_of(root, root / "mri/t1_mprage.nii", ("anat", "T2w"))
    assert (chosen.datatype, chosen.suffix, chosen.how) == ("anat", "T2w", KIND_CHOSEN)
    # A BIDS name decides: a choice does not override it.
    assert kind_of(root, root / "sub-01/anat/sub-01_T1w.nii.gz", ("func", "bold")).bids


def test_a_file_outside_bids_gets_its_sidecar_beside_it(tmp_path):
    root = _dataset(tmp_path)
    # A sidecar at the top would apply to a BIDS bold run; it says nothing
    # about a file that is not named the BIDS way.
    _json(root / "task-rest_bold.json", {"TaskName": "rest", "RepetitionTime": 9.0})
    image = _nifti(root / "scans/rest_bold.nii.gz", (4, 4, 3, 5), tr=0.85, descrip=DCM2NIIX)
    (one,) = plan_sidecars(root, [image]).files
    assert one.sidecar == "scans/rest_bold.json"
    assert one.kind.how == "name" and not one.inherited
    assert _fields(one)["RepetitionTime"] == (0.85, HEADER)
    assert not one.kind.bids and one.kind.why == "the _bold in its name"
    apply_sidecars(root, plan_sidecars(root, [image]).writes())
    assert json.loads((root / "scans/rest_bold.json").read_text())["RepetitionTime"] == 0.85


def test_what_a_file_is_read_as_can_be_chosen(tmp_path):
    root = _dataset(tmp_path)
    image = _nifti(root / "mri/scan.nii.gz", (4, 4, 3, 5), tr=0.85, descrip=DCM2NIIX)
    (guessed,) = plan_sidecars(root, [image]).files
    assert (guessed.datatype, guessed.suffix) == ("func", "bold")
    (chosen,) = plan_sidecars(root, [image], kinds={"mri/scan.nii.gz": ("dwi", "dwi")}).files
    assert (chosen.datatype, chosen.suffix, chosen.kind.how) == ("dwi", "dwi", "chosen")
    # A dwi run takes no TaskName from its name.
    assert "TaskName" not in chosen.document


def test_a_value_the_bids_version_rejects_is_shown_not_written(tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    _nifti(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 5), tr=0.85)
    _project(root, monkeypatch, {"sub-01_task-rest_bold.json": {
        "SliceEncodingDirection": "x", "EchoTime": 0.03, "FlipAngle": -5}})
    (one,) = plan_sidecars(root).files
    rejected = {r.name: r for r in one.rejected}
    assert set(rejected) == {"SliceEncodingDirection", "FlipAngle"}
    assert "must be one of" in rejected["SliceEncodingDirection"].problem
    assert rejected["FlipAngle"].origin == SOURCE
    assert "SliceEncodingDirection" not in one.document
    assert one.document["EchoTime"] == 0.03


def test_the_selected_bids_version_is_the_one_followed(tmp_path):
    from bidsmgr import schema as schema_mod
    from bidsmgr.editor.sidecars import kind_choices

    root = _dataset(tmp_path)
    image = _nifti(root / "mri/scan.nii.gz")
    try:
        schema_mod.set_active_version("1.9.0")
        # MR spectroscopy came in 1.10: not something to read a file as in 1.9.
        assert "mrs" not in dict(kind_choices(image))
        assert plan_sidecars(root, [image]).bids_version == "1.9.0"
    finally:
        schema_mod.set_active_version(None)
    assert "mrs" in dict(kind_choices(image))
